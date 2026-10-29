//! The tools an episode calls: their declarations, reading calls out of a turn as the deployed
//! tool-call parser reads them, and running them.
//!
//! The simulators are WaterTAP, which only Python runs, so every tool runs in notebook 07's own
//! worker processes (its `worker` subcommand under `.venv-watertap`): one JSON object per line
//! in, `{"result": ...}` per line out. Calls already answered are cached; the simulators are
//! deterministic.

use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::LazyLock;

use fancy_regex::Regex;

use super::pyjson::{self, Value};

pub const RO: &str = "simulate_ro";
pub const SWRO: &str = "simulate_swro_system";
pub const CONVERT: &str = "convert_units";
pub const COMPARE: &str = "check_limits";

/// Simulations per episode; calls beyond this are answered with an error instead of run.
pub const MAX_CALLS: usize = 8;
/// convert_units calls per episode.
pub const MAX_CONVERTS: usize = 16;
/// check_limits calls per episode.
pub const MAX_COMPARES: usize = 4;
/// Tool rounds per episode; a turn that still calls after the last round ends the episode unanswered.
pub const MAX_ROUNDS: usize = 6;
pub const CALL_BUDGET_ERROR: &str =
    "Error: call budget exhausted (8 simulations per question); answer with what you have.";
pub const CONVERT_BUDGET_ERROR: &str =
    "Error: conversion budget exhausted (16 per question); answer with what you have.";
pub const COMPARE_BUDGET_ERROR: &str =
    "Error: check budget exhausted (4 per question); answer with what you have.";
pub const MALFORMED_CALL_ERROR: &str = "Error: malformed tool call";

/// The harness's own tools, as opposed to the simulators.
pub fn is_local(name: Option<&str>) -> bool {
    matches!(name, Some(CONVERT) | Some(COMPARE))
}

/// The declarations the prompt lists, and the JSON types each parameter takes.
pub struct ToolDecls {
    json: serde_json::Value,
    types: HashMap<(String, String), BTreeSet<Option<String>>>,
}

impl ToolDecls {
    /// A frozen declarations file (`07_error_recovery_tools.json`).
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let mut types = HashMap::new();
        for decl in json.as_array().ok_or("tool declarations are not a list")? {
            let f = &decl["function"];
            let name = f["name"].as_str().ok_or("a declaration without a name")?;
            if let Some(props) = f["parameters"]["properties"].as_object() {
                for (param, prop) in props {
                    let alternatives = match prop.get("anyOf").and_then(|a| a.as_array()) {
                        Some(any) => any.iter().collect::<Vec<_>>(),
                        None => vec![prop],
                    };
                    let set = alternatives
                        .iter()
                        .map(|t| t.get("type").and_then(|t| t.as_str()).map(str::to_owned))
                        .collect();
                    types.insert((name.to_owned(), param.clone()), set);
                }
            }
        }
        Ok(Self { json, types })
    }

    /// The declarations as the chat template takes them.
    pub fn json(&self) -> &serde_json::Value {
        &self.json
    }

    fn schema_types(&self, name: &str, param: &str) -> BTreeSet<Option<String>> {
        self.types
            .get(&(name.to_owned(), param.to_owned()))
            .cloned()
            .unwrap_or_else(|| BTreeSet::from([None]))
    }

    /// A `<parameter>` value typed as the deployed tool-call parser types it, by the tool's
    /// schema: a number for a number, the text for a string, else the text read as JSON.
    fn convert_param(&self, name: &str, param: &str, raw: &str) -> Value {
        let value = raw.trim();
        let lower = value.to_lowercase();
        if lower == "null" || lower == "none" {
            return Value::Null;
        }
        let types = self.schema_types(name, param);
        let has = |t: &str| types.contains(&Some(t.to_owned()));
        if has("number") || has("integer") {
            // Left as text when it is not a number; the server's validation rejects it.
            return pyjson::py_float(value)
                .map(Value::Float)
                .unwrap_or_else(|| Value::Str(value.to_owned()));
        }
        if has("string") {
            return Value::Str(value.to_owned());
        }
        pyjson::loads(value).unwrap_or_else(|_| Value::Str(value.to_owned()))
    }
}

/// One tool call of an episode.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    /// None when the call could not be read.
    pub name: Option<String>,
    /// An object, or None when the call could not be read.
    pub arguments: Option<Value>,
    pub malformed: bool,
    /// What came back: the tool's text, or the harness's error.
    pub result: Option<String>,
    /// The turn it was written in.
    pub turn: Option<usize>,
}

static CALL_BLOCK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").unwrap());
static FUNCTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)\A(?:\s*<function=([^>\s]+)>(.*?)</function>\s*)\z").unwrap()
});
static PARAMETER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>").unwrap());

/// Every `<tool_call>` block in one assistant turn, in Qwen3.5's XML form.
pub fn parse_calls(turn: &str, decls: &ToolDecls) -> Vec<Call> {
    let mut calls = Vec::new();
    for block in CALL_BLOCK.captures_iter(turn) {
        let body = block.unwrap().get(1).unwrap().as_str();
        let Some(f) = FUNCTION.captures(body).unwrap() else {
            calls.push(Call {
                name: None,
                arguments: None,
                malformed: true,
                result: None,
                turn: None,
            });
            continue;
        };
        let name = f.get(1).unwrap().as_str();
        let mut arguments = Value::Dict(Vec::new());
        for p in PARAMETER.captures_iter(f.get(2).unwrap().as_str()) {
            let p = p.unwrap();
            let param = p.get(1).unwrap().as_str();
            arguments.insert(
                param.to_owned(),
                decls.convert_param(name, param, p.get(2).unwrap().as_str()),
            );
        }
        calls.push(Call {
            name: Some(name.to_owned()),
            arguments: Some(arguments),
            malformed: false,
            result: None,
            turn: None,
        });
    }
    calls
}

/// Whether a tool's reply is an error: the server's own error text, text that is not JSON, or
/// JSON holding `"error"` (a key of an object, an item of a list, a substring of a string).
pub fn result_is_error(text: Option<&str>) -> bool {
    let Some(text) = text.filter(|t| !t.is_empty()) else {
        return true;
    };
    if text.starts_with("Error executing tool") || text.starts_with("Unknown tool") {
        return true;
    }
    match pyjson::loads(text) {
        Err(_) => true,
        Ok(Value::Dict(items)) => items.iter().any(|(k, _)| k == "error"),
        Ok(Value::List(items)) => items.iter().any(|v| v.as_str() == Some("error")),
        Ok(Value::Str(s)) => s.contains("error"),
        // Python raises here; no tool answers with a bare number.
        Ok(_) => false,
    }
}

/// Runs tool calls and returns their replies, in order.
pub trait Tools {
    fn call_many(&mut self, calls: &[(String, Value)]) -> Vec<String>;
}

/// Worker processes serving notebook 07's tool host, and a cache of every call already answered.
pub struct ToolPool {
    workers: Vec<Worker>,
    cache: HashMap<String, String>,
    /// Calls the workers actually ran.
    pub solved: usize,
}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Worker {
    fn call(&mut self, name: &str, arguments: &Value) -> Result<String, String> {
        let request = Value::Dict(vec![
            ("name".into(), Value::Str(name.into())),
            ("arguments".into(), arguments.clone()),
        ]);
        writeln!(self.stdin, "{}", pyjson::dumps(&request, false)).map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        if self
            .stdout
            .read_line(&mut line)
            .map_err(|e| e.to_string())?
            == 0
        {
            return Err("tool worker died".into());
        }
        let reply = pyjson::loads(&line)?;
        match reply.get("result") {
            Some(Value::Str(s)) => Ok(s.clone()),
            other => Err(format!("tool worker replied without a result: {other:?}")),
        }
    }
}

impl ToolPool {
    /// `n` workers: `python script worker`, the notebook's own tool server.
    pub fn new(python: &Path, script: &Path, n: usize) -> Result<Self, String> {
        Self::niced(python, script, n, 0)
    }

    /// `new`, the workers at `niceness` (`nice -n`): the simulations yield the CPU to the
    /// process driving the GPU.
    pub fn niced(python: &Path, script: &Path, n: usize, niceness: i32) -> Result<Self, String> {
        let workers = (0..n)
            .map(|_| {
                let mut command = if niceness != 0 {
                    let mut c = Command::new("nice");
                    c.arg("-n").arg(niceness.to_string()).arg(python);
                    c
                } else {
                    Command::new(python)
                };
                let mut child = command
                    .arg(script)
                    .arg("worker")
                    .env("PYTHONDONTWRITEBYTECODE", "1")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                    .map_err(|e| format!("{}: {e}", python.display()))?;
                let stdin = child.stdin.take().unwrap();
                let stdout = BufReader::new(child.stdout.take().unwrap());
                Ok(Worker {
                    child,
                    stdin,
                    stdout,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            workers,
            cache: HashMap::new(),
            solved: 0,
        })
    }

    /// Notebook 07's workers in a MembraneClaw checkout: its script under `.venv-watertap`.
    pub fn notebook_07(repo: &Path, n: usize) -> Result<Self, String> {
        let python: PathBuf = repo.join(".venv-watertap/bin/python");
        let script = repo.join("experiments/notebooks/smoke_test35/07_error_recovery.py");
        Self::new(&python, &script, n)
    }

    fn key(name: &str, arguments: &Value) -> String {
        pyjson::dumps(
            &Value::List(vec![Value::Str(name.into()), arguments.clone()]),
            true,
        )
    }
}

impl Tools for ToolPool {
    fn call_many(&mut self, calls: &[(String, Value)]) -> Vec<String> {
        let mut todo: Vec<(String, &str, &Value)> = Vec::new();
        let mut seen = BTreeSet::new();
        for (name, arguments) in calls {
            let k = Self::key(name, arguments);
            if !self.cache.contains_key(&k) && seen.insert(k.clone()) {
                todo.push((k, name, arguments));
            }
        }
        if !todo.is_empty() {
            let n = self.workers.len();
            let chunks: Vec<Vec<&(String, &str, &Value)>> = (0..n)
                .map(|i| todo.iter().skip(i).step_by(n).collect())
                .collect();
            let answered: Vec<Vec<(String, String)>> = std::thread::scope(|s| {
                let handles: Vec<_> = self
                    .workers
                    .iter_mut()
                    .zip(chunks)
                    .map(|(w, chunk)| {
                        s.spawn(move || {
                            chunk
                                .into_iter()
                                .map(|(k, name, args)| {
                                    (k.clone(), w.call(name, args).expect("tool worker failed"))
                                })
                                .collect()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("tool worker thread panicked"))
                    .collect()
            });
            for part in answered {
                self.cache.extend(part);
            }
            self.solved += todo.len();
        }
        calls
            .iter()
            .map(|(n, a)| self.cache[&Self::key(n, a)].clone())
            .collect()
    }
}

impl Drop for ToolPool {
    fn drop(&mut self) {
        for w in &mut self.workers {
            // Closing stdin ends the worker's loop.
            let _ = w.stdin.flush();
        }
        for w in self.workers.drain(..) {
            let Worker {
                mut child,
                stdin,
                stdout,
            } = w;
            drop(stdin);
            drop(stdout);
            let _ = child.wait();
        }
    }
}
