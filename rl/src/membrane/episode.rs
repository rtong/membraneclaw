//! An episode: the question's prompt, the policy's turns, the tools' replies in between.
//!
//! This is notebook 07's harness. Each round every live episode writes one turn. A turn that
//! closes a JSON object before any `<tool_call>` is the answer (with `stop_at_json`, cut at that
//! brace); a turn without calls is the answer; a turn cut off before `<|im_end|>` ends the
//! episode truncated; a turn with calls has them run — within the per-episode budgets — and the
//! replies come back as one user turn, unless the rounds have run out, which ends the episode
//! with its calls unanswered. `max_new_tokens` caps the policy's own tokens summed over its turns.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use serde_json::{Map, Value as Json};

use super::pyjson::Value;
use super::reward::json_spans;
use super::tools::{
    self, CALL_BUDGET_ERROR, COMPARE, COMPARE_BUDGET_ERROR, CONVERT, CONVERT_BUDGET_ERROR, Call,
    MALFORMED_CALL_ERROR, MAX_CALLS, MAX_COMPARES, MAX_CONVERTS, MAX_ROUNDS, ToolDecls, Tools,
};
use crate::chat::{ChatTemplate, Tokenizer};

/// agent/core.py's system message, verbatim, and the one addition the task makes to it: an
/// answer contract that code can grade.
pub const SYSTEM_PROMPT: &str = concat!(
    "You are a capable assistant running on a local Qwen3.5 deployment. ",
    "Use the provided tools when they give you a more accurate answer than reasoning alone; ",
    "otherwise answer directly. When documents are attached, ground your answer in them and ",
    "say so if the answer is not present.",
    "\n\n",
    "Finish your reply with one JSON object and nothing after it:\n",
    "{\"decision\": ..., \"violations\": [...], \"values\": {...}}\n",
    "- \"decision\": the direct answer to the question. \"pass\" or \"fail\" when asked whether ",
    "something meets its limits; otherwise the option or setting you recommend, written the way ",
    "the user named it; null if none of the options works.\n",
    "- \"violations\": when asked whether something meets its limits, the tool-result fields whose ",
    "limits are broken; otherwise [].\n",
    "- \"values\": the numbers the user asked for, keyed by the tool-result field they come from ",
    "as section.field, in that field's units."
);

/// What every episode shares: the tokenizer and template, the tools' declarations, the special
/// tokens, and the harness's settings.
pub struct Harness {
    pub tokenizer: Tokenizer,
    pub template: ChatTemplate,
    pub decls: ToolDecls,
    pub im_end: u32,
    /// `<|endoftext|>`, which the policy may never write.
    pub suppress: u32,
    /// What `add_generation_prompt` appends: the assistant header and, thinking off, the empty
    /// think block.
    pub prefix: String,
    /// transformers' `all_special_ids`: tokens whose text an answer's character offsets skip.
    pub special_ids: BTreeSet<u32>,
    pub max_new_tokens: usize,
    pub stop_at_json: bool,
    pieces: RefCell<HashMap<u32, String>>,
}

impl Harness {
    /// The checkpoint in `dir` and the tool declarations in `tools_json`.
    pub fn new(
        dir: &Path,
        tools_json: &Path,
        max_new_tokens: usize,
        stop_at_json: bool,
    ) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_dir(dir)?;
        let template = ChatTemplate::from_dir(dir)?;
        let decls = ToolDecls::from_file(tools_json)?;
        let id = |t: &str| tokenizer.token_id(t).ok_or_else(|| format!("no token {t}"));
        let (im_end, suppress) = (id("<|im_end|>")?, id("<|endoftext|>")?);
        let kwargs = thinking_off();
        let probe = Json::Array(vec![serde_json::json!({"role": "user", "content": "x"})]);
        let on = template.render(&probe, None, true, &kwargs)?;
        let off = template.render(&probe, None, false, &kwargs)?;
        let prefix = on
            .strip_prefix(off.as_str())
            .ok_or("the generation prompt is not an extension")?
            .to_owned();
        let special_ids = special_ids(dir, &tokenizer)?;
        Ok(Self {
            tokenizer,
            template,
            decls,
            im_end,
            suppress,
            prefix,
            special_ids,
            max_new_tokens,
            stop_at_json,
            pieces: RefCell::new(HashMap::new()),
        })
    }

    /// The question's prompt as text: the system prompt, the question, the tools.
    pub fn render_prompt(&self, case: &Value) -> Result<String, String> {
        let question = case
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or("a case without a prompt")?;
        let messages = serde_json::json!([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]);
        self.template
            .render(&messages, Some(self.decls.json()), true, &thinking_off())
    }

    /// Everything between the policy's `<|im_end|>` and its next turn.
    pub fn tool_response_text(&self, results: &[String]) -> String {
        let body: String = results
            .iter()
            .map(|r| format!("\n<tool_response>\n{r}\n</tool_response>"))
            .collect();
        format!("\n<|im_start|>user{body}<|im_end|>\n{}", self.prefix)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text).expect("tokenizer")
    }

    /// One token's text, as `tokenizer.decode([t])` writes it: a piece of a character split
    /// across tokens comes out as U+FFFD.
    pub fn piece(&self, t: u32) -> String {
        if let Some(s) = self.pieces.borrow().get(&t) {
            return s.clone();
        }
        let s = self.tokenizer.decode(&[t], false).expect("tokenizer");
        self.pieces.borrow_mut().insert(t, s.clone());
        s
    }

    /// Index of the token closing the first top-level JSON object of a turn with no
    /// `<tool_call>`, in the text the tokens' pieces spell.
    pub fn answer_close_index(&self, tokens: &[u32]) -> Option<usize> {
        let mut text = String::new();
        let mut ends = Vec::with_capacity(tokens.len());
        let mut chars = 0;
        for &t in tokens {
            let p = self.piece(t);
            chars += p.chars().count();
            text.push_str(&p);
            ends.push(chars);
        }
        if text.contains("<tool_call>") {
            return None;
        }
        let close = char_offset(&text, json_spans(&text).first()?.1);
        ends.iter().position(|&e| e >= close)
    }

    /// The generation rule `stop_at_json` adds: a sequence is done once its turn so far — the
    /// pieces of its tokens — closes a JSON object with no `<tool_call>` before it.
    pub fn answer_written(&self, turn_so_far: &str) -> bool {
        !turn_so_far.contains("<tool_call>") && !json_spans(turn_so_far).is_empty()
    }
}

fn thinking_off() -> Map<String, Json> {
    let mut kwargs = Map::new();
    kwargs.insert("enable_thinking".into(), Json::Bool(false));
    kwargs
}

/// transformers' `all_special_ids`: the named special tokens and, in a v5 tokenizer config, the
/// `extra_special_tokens` (else the `additional_special_tokens`).
fn special_ids(dir: &Path, tokenizer: &Tokenizer) -> Result<BTreeSet<u32>, String> {
    let text =
        std::fs::read_to_string(dir.join("tokenizer_config.json")).map_err(|e| e.to_string())?;
    let config: Json = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut names: Vec<String> = Vec::new();
    let token = |v: &Json| match v {
        Json::String(s) => Some(s.clone()),
        Json::Object(o) => o.get("content").and_then(Json::as_str).map(str::to_owned),
        _ => None,
    };
    for key in [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ] {
        names.extend(config.get(key).and_then(token));
    }
    match config.get("extra_special_tokens").and_then(Json::as_object) {
        Some(extra) => names.extend(extra.values().filter_map(token)),
        None => {
            if let Some(list) = config
                .get("additional_special_tokens")
                .and_then(Json::as_array)
            {
                names.extend(list.iter().filter_map(token));
            }
        }
    }
    names
        .iter()
        .map(|n| tokenizer.token_id(n).ok_or_else(|| format!("no token {n}")))
        .collect()
}

/// The character (code point) offset of a byte offset: Python counts characters.
pub fn char_offset(text: &str, byte: usize) -> usize {
    text[..byte].chars().count()
}

/// One multi-turn episode. `ids` is everything after the prompt; `policy` marks the policy's own
/// tokens (false: written by the harness — tool replies and turn headers).
#[derive(Debug, Clone)]
pub struct Episode {
    pub case: Value,
    pub prompt_ids: Vec<u32>,
    pub ids: Vec<u32>,
    pub policy: Vec<bool>,
    pub turns: Vec<String>,
    /// Index into `ids` of each turn's first token.
    pub turn_starts: Vec<usize>,
    pub calls: Vec<Call>,
    pub final_text: String,
    /// A turn hit the token budget before `<|im_end|>`, or the budget ran out between turns.
    pub truncated: bool,
    /// Calls left when the rounds ran out.
    pub unanswered: usize,
    pub answer_closed: bool,
}

impl Episode {
    pub fn policy_tokens(&self) -> usize {
        self.policy.iter().filter(|&&p| p).count()
    }
}

/// Writes the next turn of each sequence: given every live episode's tokens so far and how many
/// more tokens each may write, the tokens each wrote. A turn ends at `<|im_end|>` (included), at
/// its budget, or — under `stop_at_json` — where `Harness::answer_written` holds.
pub trait Policy {
    fn generate(&mut self, inputs: &[Vec<u32>], budgets: &[usize]) -> Vec<Vec<u32>>;
}

/// A turn the harness writes into an episode before the policy writes any: its calls with their
/// replies, already run. It is context, not the policy's tokens; its calls are the episode's calls
/// (scored, and counted against the budgets), and its round counts against `MAX_ROUNDS`.
#[derive(Debug, Clone)]
pub struct PrefilledTurn {
    /// The turn's text as the chat template writes it.
    pub text: String,
    /// The turn's tokens, its closing `<|im_end|>` included.
    pub written: Vec<u32>,
    pub calls: Vec<Call>,
}

/// Where an episode starts: the prompt alone, or the prompt and turns written before the policy's.
#[derive(Debug, Clone, Default)]
pub struct Start {
    pub turns: Vec<PrefilledTurn>,
    /// Whether the prefilled turns' tokens count against the policy's token budget (07's forced
    /// probe: no; a continued error state, whose turns the policy wrote in an earlier episode: yes).
    pub tokens_count: bool,
}

impl Harness {
    /// One call exactly as the chat template writes it in an assistant turn.
    pub fn call_block(name: &str, arguments: &Value) -> String {
        let mut params = String::new();
        for (k, v) in arguments.as_dict().unwrap_or(&[]) {
            let shown = match v {
                Value::Dict(_) | Value::List(_) => super::pyjson::dumps_unicode(v),
                other => super::pyjson::py_str(other),
            };
            params.push_str(&format!("<parameter={k}>\n{shown}\n</parameter>\n"));
        }
        format!("<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>")
    }

    /// An assistant turn of calls as the chat template writes it: the calls one per line, after the
    /// turn's own line and a blank line when it has one.
    pub fn turn_text(text: Option<&str>, calls: &[Call]) -> String {
        let blocks: Vec<String> = calls
            .iter()
            .map(|c| Self::call_block(c.name.as_deref().unwrap_or(""), c.arguments.as_ref().unwrap_or(&Value::Dict(Vec::new()))))
            .collect();
        let calls = blocks.join("\n");
        match text.filter(|t| !t.is_empty()) {
            Some(t) => format!("{t}\n\n{calls}"),
            None => calls,
        }
    }

    /// A recorded turn of a recovery file (`{"calls": [...], "results": [...], "text"?: ...}`), as
    /// the forced probe prefills it.
    pub fn recorded_turn(&self, turn: &Value) -> PrefilledTurn {
        let results: Vec<String> = turn
            .get("results")
            .and_then(Value::as_list)
            .unwrap_or(&[])
            .iter()
            .map(|r| r.as_str().unwrap_or("").to_owned())
            .collect();
        let calls: Vec<Call> = turn
            .get("calls")
            .and_then(Value::as_list)
            .unwrap_or(&[])
            .iter()
            .zip(&results)
            .map(|(c, r)| Call {
                name: c.get("name").and_then(Value::as_str).map(str::to_owned),
                arguments: c.get("arguments").cloned(),
                malformed: false,
                result: Some(r.clone()),
                turn: None,
            })
            .collect();
        let text = Self::turn_text(turn.get("text").and_then(Value::as_str), &calls);
        let mut written = self.encode(&text);
        written.push(self.im_end);
        PrefilledTurn { text, written, calls }
    }
}

/// Runs a batch of episodes with the tools in the loop.
pub fn rollout(
    h: &Harness,
    cases: &[Value],
    tools: &mut dyn Tools,
    policy: &mut dyn Policy,
) -> Result<Vec<Episode>, String> {
    let starts = vec![Start::default(); cases.len()];
    rollout_from(h, cases, &starts, tools, policy)
}

/// One episode while it runs: the episode so far, the rounds its prefilled turns took, the
/// tokens of its own they charged to its budget, and the turns it has written.
pub struct Running {
    pub ep: Episode,
    used: usize,
    spent: usize,
    round: usize,
}

impl Harness {
    /// An episode of `case` started from `start`: the prompt, and the prefilled turns with their
    /// replies.
    pub fn begin(&self, case: &Value, start: &Start) -> Result<Running, String> {
        let prompt_ids = self.encode(&self.render_prompt(case)?);
        let mut ep = Episode {
            case: case.clone(),
            prompt_ids,
            ids: Vec::new(),
            policy: Vec::new(),
            turns: Vec::new(),
            turn_starts: Vec::new(),
            calls: Vec::new(),
            final_text: String::new(),
            truncated: false,
            unanswered: 0,
            answer_closed: false,
        };
        let (mut used, mut spent) = (0, 0);
        for turn in &start.turns {
            ep.turn_starts.push(ep.ids.len());
            ep.turns.push(turn.text.clone());
            let index = ep.turns.len() - 1;
            for c in &turn.calls {
                ep.calls.push(Call {
                    turn: Some(index),
                    ..c.clone()
                });
            }
            let results: Vec<String> = turn
                .calls
                .iter()
                .map(|c| c.result.clone().unwrap_or_default())
                .collect();
            let env = self.encode(&self.tool_response_text(&results));
            if start.tokens_count {
                spent += turn.written.len();
            }
            ep.ids.extend_from_slice(&turn.written);
            ep.ids.extend_from_slice(&env);
            ep.policy
                .extend(std::iter::repeat_n(false, turn.written.len() + env.len()));
            used += 1;
        }
        Ok(Running {
            ep,
            used,
            spent,
            round: 0,
        })
    }
}

impl Running {
    /// The tokens the next turn may write; 0 ends the episode (truncated: the budget ran out
    /// between turns, no answer).
    pub fn budget(&mut self, h: &Harness) -> usize {
        let left = h
            .max_new_tokens
            .saturating_sub(self.ep.policy_tokens() + self.spent);
        if left == 0 {
            self.ep.truncated = true;
        }
        left
    }

    /// Everything the policy has seen: the prompt and the episode's tokens.
    pub fn input(&self) -> Vec<u32> {
        [self.ep.prompt_ids.as_slice(), &self.ep.ids].concat()
    }

    /// Takes the turn the policy wrote (at most `budget` tokens of `row` count). Returns its calls,
    /// each over its budget or malformed already answered with the harness's error, and the
    /// indices of those the tools must run; `None` when the turn ends the episode.
    pub fn take_turn(
        &mut self,
        h: &Harness,
        row: &[u32],
        budget: usize,
    ) -> Result<Option<(Vec<Call>, Vec<usize>)>, String> {
        let round = self.round;
        self.round += 1;
        let row = &row[..row.len().min(budget)];
        let ep = &mut self.ep;
        let cut = row.iter().position(|&t| t == h.im_end);
        let closed = if h.stop_at_json {
            h.answer_close_index(&row[..cut.unwrap_or(row.len())])
        } else {
            None
        };
        let take = match (closed, cut) {
            (Some(c), _) => {
                ep.answer_closed = true;
                &row[..=c]
            }
            (None, Some(c)) => &row[..=c],
            (None, None) => row,
        };
        ep.turn_starts.push(ep.ids.len());
        ep.ids.extend_from_slice(take);
        ep.policy.extend(std::iter::repeat_n(true, take.len()));
        let text = h.tokenizer.decode(take, true)?;
        ep.turns.push(text.clone());
        if closed.is_some() {
            ep.final_text = text;
            return Ok(None);
        }
        if cut.is_none() {
            ep.truncated = true;
            ep.final_text = text;
            return Ok(None);
        }
        let mut calls = tools::parse_calls(&text, &h.decls);
        if calls.is_empty() {
            ep.final_text = text;
            return Ok(None);
        }
        if round + self.used >= MAX_ROUNDS {
            ep.unanswered = calls.len();
            return Ok(None);
        }
        let done = &ep.calls;
        let mut sims = done
            .iter()
            .filter(|c| !tools::is_local(c.name.as_deref()))
            .count();
        let mut converts = done
            .iter()
            .filter(|c| c.name.as_deref() == Some(CONVERT))
            .count();
        let mut compares = done
            .iter()
            .filter(|c| c.name.as_deref() == Some(COMPARE))
            .count();
        let mut jobs = Vec::new();
        for (k, c) in calls.iter_mut().enumerate() {
            let (count, limit, error) = if c.malformed {
                c.result = Some(MALFORMED_CALL_ERROR.into());
                continue;
            } else if c.name.as_deref() == Some(CONVERT) {
                (&mut converts, MAX_CONVERTS, CONVERT_BUDGET_ERROR)
            } else if c.name.as_deref() == Some(COMPARE) {
                (&mut compares, MAX_COMPARES, COMPARE_BUDGET_ERROR)
            } else {
                (&mut sims, MAX_CALLS, CALL_BUDGET_ERROR)
            };
            *count += 1;
            if *count > limit {
                c.result = Some(error.into());
            } else {
                jobs.push(k);
            }
        }
        Ok(Some((calls, jobs)))
    }

    /// The turn's calls, answered: they join the episode, and their replies follow the turn.
    pub fn answer(&mut self, h: &Harness, calls: Vec<Call>) {
        let ep = &mut self.ep;
        let turn = ep.turns.len() - 1;
        let results: Vec<String> = calls
            .iter()
            .map(|c| c.result.clone().unwrap_or_default())
            .collect();
        for mut c in calls {
            c.turn = Some(turn);
            ep.calls.push(c);
        }
        let env = h.encode(&h.tool_response_text(&results));
        ep.policy.extend(std::iter::repeat_n(false, env.len()));
        ep.ids.extend(env);
    }
}

/// `rollout`, each episode starting from its `Start`: round by round, every live episode's next
/// turn in one call of the policy, every call of the round run at once.
pub fn rollout_from(
    h: &Harness,
    cases: &[Value],
    starts: &[Start],
    tools: &mut dyn Tools,
    policy: &mut dyn Policy,
) -> Result<Vec<Episode>, String> {
    let mut runs: Vec<Running> = cases
        .iter()
        .zip(starts)
        .map(|(c, s)| h.begin(c, s))
        .collect::<Result<_, _>>()?;
    let mut live: Vec<usize> = (0..runs.len()).collect();
    for _ in 0..=MAX_ROUNDS {
        let budgets: HashMap<usize, usize> = live.iter().map(|&i| (i, runs[i].budget(h))).collect();
        live.retain(|i| budgets[i] > 0);
        if live.is_empty() {
            break;
        }
        let inputs: Vec<Vec<u32>> = live.iter().map(|&i| runs[i].input()).collect();
        let rows = policy.generate(
            &inputs,
            &live.iter().map(|i| budgets[i]).collect::<Vec<_>>(),
        );
        let mut pending: Vec<(usize, Vec<Call>, Vec<usize>)> = Vec::new();
        for (row, &i) in rows.into_iter().zip(&live) {
            if let Some((calls, jobs)) = runs[i].take_turn(h, &row, budgets[&i])? {
                pending.push((i, calls, jobs));
            }
        }
        // Every pending call of the round runs at once.
        let mut jobs: Vec<(String, Value)> = Vec::new();
        let mut slots: Vec<(usize, usize)> = Vec::new();
        for (p, (_, calls, ks)) in pending.iter().enumerate() {
            for &k in ks {
                jobs.push((calls[k].name.clone().unwrap(), calls[k].arguments.clone().unwrap()));
                slots.push((p, k));
            }
        }
        for ((p, k), result) in slots.into_iter().zip(tools.call_many(&jobs)) {
            pending[p].1[k].result = Some(result);
        }
        live = Vec::with_capacity(pending.len());
        for (i, calls, _) in pending {
            runs[i].answer(h, calls);
            live.push(i);
        }
    }
    Ok(runs.into_iter().map(|r| r.ep).collect())
}
