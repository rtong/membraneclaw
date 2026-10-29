//! Prompts as transformers writes them: a checkpoint's chat template rendered the way
//! `apply_chat_template` renders it, and the checkpoint's tokenizer.
//!
//! transformers renders templates with Jinja2 configured as `trim_blocks` and
//! `lstrip_blocks`, and replaces Jinja2's `tojson` with Python's `json.dumps(x,
//! ensure_ascii=False)`. Templates are written against that, so three things are fixed
//! here rather than left to minijinja's defaults:
//! - `tojson` writes `{"a": 1, "b": [1, 2]}` — a space after each `:` and `,`, keys in
//!   the order they were given, no HTML escaping — where minijinja writes `{"a":1,...}`.
//! - Values print as Python's `str()` does, in `{{ }}` and through the `string` filter:
//!   `True`, `None`, `1e-06`. A tool call's boolean argument reaches the prompt this way.
//! - `raise_exception(message)` fails the render, as templates use it to refuse input.

use std::fmt::Write as _;
use std::path::Path;

use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Kwargs, Serde, Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{Map, Value as Json};

/// A checkpoint's chat template.
pub struct ChatTemplate {
    env: Environment<'static>,
    /// The special tokens transformers passes to every render (`bos_token`, `eos_token`, ...).
    special: Map<String, Json>,
}

impl ChatTemplate {
    /// The template of the checkpoint in `dir`: `chat_template.jinja`, else the
    /// `chat_template` string in `tokenizer_config.json`, which is also where the special
    /// tokens are read from.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let config: Json = match std::fs::read_to_string(dir.join("tokenizer_config.json")) {
            Ok(s) => serde_json::from_str(&s).map_err(|e| format!("tokenizer_config.json: {e}"))?,
            Err(_) => Json::Object(Map::new()),
        };
        let source = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(s) => s,
            Err(_) => config
                .get("chat_template")
                .and_then(Json::as_str)
                .map(str::to_owned)
                .ok_or_else(|| format!("no chat template in {}", dir.display()))?,
        };
        let mut special = Map::new();
        for name in [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
        ] {
            // Written either as the token itself or as an added-token record.
            let token = match config.get(name) {
                Some(Json::String(s)) => Some(s.clone()),
                Some(Json::Object(o)) => o.get("content").and_then(Json::as_str).map(str::to_owned),
                _ => None,
            };
            if let Some(token) = token {
                special.insert(name.to_owned(), Json::String(token));
            }
        }
        Self::new(source, special)
    }

    /// A template from its source, with the special tokens to pass to every render.
    pub fn new(source: String, special: Map<String, Json>) -> Result<Self, String> {
        let mut env = Environment::new();
        env.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| e.to_string())?,
        );
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.set_formatter(|out, _state, value| {
            match value.as_str() {
                Some(s) => out.write_str(s),
                None => out.write_str(&py_str(value)),
            }
            .map_err(|e| Error::new(ErrorKind::WriteFailure, e.to_string()))
        });
        env.add_filter("tojson", tojson);
        env.add_filter("string", |value: Value| -> String {
            match value.as_str() {
                Some(s) => s.to_owned(),
                None => py_str(&value),
            }
        });
        env.add_function(
            "raise_exception",
            |message: String| -> Result<Value, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_template_owned("chat", source)
            .map_err(|e| e.to_string())?;
        Ok(Self { env, special })
    }

    /// `messages` (and `tools`) as `apply_chat_template(messages, tools=tools,
    /// add_generation_prompt=..., tokenize=False, **kwargs)` writes them; `kwargs` holds
    /// the template's own switches, such as `enable_thinking`.
    pub fn render(
        &self,
        messages: &Json,
        tools: Option<&Json>,
        add_generation_prompt: bool,
        kwargs: &Map<String, Json>,
    ) -> Result<String, String> {
        let mut context = self.special.clone();
        context.insert("messages".into(), messages.clone());
        if let Some(tools) = tools {
            context.insert("tools".into(), tools.clone());
        }
        context.insert(
            "add_generation_prompt".into(),
            Json::Bool(add_generation_prompt),
        );
        for (k, v) in kwargs {
            context.insert(k.clone(), v.clone());
        }
        let template = self.env.get_template("chat").map_err(|e| e.to_string())?;
        template
            .render(Value::from(Serde(Json::Object(context))))
            .map_err(|e| format!("{e:#}"))
    }
}

/// Python's `json.dumps(value, ensure_ascii=False, indent=indent)`, the `tojson`
/// transformers gives templates.
fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let mut out = String::new();
    write_json(&mut out, &value, indent, 0)?;
    Ok(Value::from_safe_string(out))
}

fn write_json(
    out: &mut String,
    value: &Value,
    indent: Option<usize>,
    depth: usize,
) -> Result<(), Error> {
    // With an indent, items are separated by "," and a newline; without, by ", ".
    let (item_sep, newline) = match indent {
        Some(n) => (",", Some(n)),
        None => (", ", None),
    };
    let open_line = |out: &mut String, depth: usize| {
        if let Some(n) = newline {
            out.push('\n');
            out.push_str(&" ".repeat(n * depth));
        }
    };
    match value.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => out.push_str(&py_number(value)),
        ValueKind::String => {
            out.push_str(&serde_json::to_string(value.as_str().unwrap_or("")).unwrap())
        }
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = value.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                open_line(out, depth + 1);
                write_json(out, item, indent, depth + 1)?;
            }
            open_line(out, depth);
            out.push(']');
        }
        ValueKind::Map => {
            let keys: Vec<Value> = value.try_iter()?.collect();
            if keys.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                open_line(out, depth + 1);
                let name = match key.as_str() {
                    Some(s) => s.to_owned(),
                    None => py_str(key),
                };
                out.push_str(&serde_json::to_string(&name).unwrap());
                out.push_str(": ");
                write_json(out, &value.get_item(key)?, indent, depth + 1)?;
            }
            open_line(out, depth);
            out.push('}');
        }
        _ => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("tojson: cannot serialize a {}", value.kind()),
            ));
        }
    }
    Ok(())
}

/// Python's `str()` of a template value.
fn py_str(value: &Value) -> String {
    match value.kind() {
        ValueKind::Undefined => String::new(),
        ValueKind::None => "None".into(),
        ValueKind::Bool => (if value.is_true() { "True" } else { "False" }).into(),
        ValueKind::Number => py_number(value),
        ValueKind::String => value.as_str().unwrap_or("").to_owned(),
        _ => py_repr(value),
    }
}

/// Python's `repr()`, which `str()` of a list or dict writes for its items.
fn py_repr(value: &Value) -> String {
    match value.kind() {
        ValueKind::String => py_repr_str(value.as_str().unwrap_or("")),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<String> = value
                .try_iter()
                .map(|it| it.map(|v| py_repr(&v)).collect())
                .unwrap_or_default();
            format!("[{}]", items.join(", "))
        }
        ValueKind::Map => {
            let keys: Vec<Value> = value.try_iter().map(|it| it.collect()).unwrap_or_default();
            let items: Vec<String> = keys
                .iter()
                .map(|k| {
                    format!(
                        "{}: {}",
                        py_repr(k),
                        py_repr(&value.get_item(k).unwrap_or_default())
                    )
                })
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        _ => py_str(value),
    }
}

fn py_repr_str(s: &str) -> String {
    // Single quotes unless the text holds one and no double quote.
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// A number as Python writes it: integers in full, floats as the shortest repr that
/// reads back, positional from 1e-4 up to 1e16 and with an exponent of at least two
/// digits outside that (`1e-06`, `1e+16`).
fn py_number(value: &Value) -> String {
    if value.is_integer() {
        return value.to_string();
    }
    let x = f64::try_from(value.clone()).unwrap_or(f64::NAN);
    py_float(x)
}

fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // Rust's `{:e}` is the shortest round-tripping mantissa, as Python's repr is.
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if (-4..16).contains(&exp) {
        let s = format!("{x}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    }
}

/// A checkpoint's tokenizer: `tokenizer.json`, and the added tokens `tokenizer_config.json`
/// declares beyond it, which transformers adds when it loads the tokenizer (Qwen3.5's audio
/// and speech tokens are only there).
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let mut inner = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| e.to_string())?;
        let config: Json = match std::fs::read_to_string(dir.join("tokenizer_config.json")) {
            Ok(s) => serde_json::from_str(&s).map_err(|e| format!("tokenizer_config.json: {e}"))?,
            Err(_) => Json::Object(Map::new()),
        };
        let mut extra: Vec<(u32, &Map<String, Json>)> = config
            .get("added_tokens_decoder")
            .and_then(Json::as_object)
            .into_iter()
            .flatten()
            .filter_map(|(id, t)| Some((id.parse().ok()?, t.as_object()?)))
            .filter(|(id, _)| inner.id_to_token(*id).is_none())
            .collect();
        // One at a time, in id order: each takes the next id.
        extra.sort_by_key(|(id, _)| *id);
        for (id, t) in extra {
            let flag = |k: &str| t.get(k).and_then(Json::as_bool).unwrap_or(false);
            let content = t
                .get("content")
                .and_then(Json::as_str)
                .ok_or("an added token without content")?;
            let token = tokenizers::AddedToken::from(content, flag("special"))
                .single_word(flag("single_word"))
                .lstrip(flag("lstrip"))
                .rstrip(flag("rstrip"))
                .normalized(flag("normalized"));
            if flag("special") {
                inner.add_special_tokens(&[token]);
            } else {
                inner.add_tokens(&[token]);
            }
            if inner.token_to_id(content) != Some(id) {
                return Err(format!("added token {content} did not get id {id}"));
            }
        }
        Ok(Self { inner })
    }

    /// The token ids of `text`, with no special tokens added around it: a rendered chat
    /// template already holds its own. Special tokens written in the text are matched.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoding = self.inner.encode(text, false).map_err(|e| e.to_string())?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| e.to_string())
    }

    /// The id of a token written as text, such as `<|im_end|>`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_repr() {
        let cases = [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e-6, "1e-06"),
            (1e-4, "0.0001"),
            (123456.5, "123456.5"),
            (1e16, "1e+16"),
            (2.5e-10, "2.5e-10"),
            (-3.25, "-3.25"),
        ];
        for (x, want) in cases {
            assert_eq!(py_float(x), want, "{x}");
        }
    }

    #[test]
    fn tojson_is_json_dumps() {
        let t = ChatTemplate::new(
            "{{ x | tojson }}|{{ x | tojson(indent=2) }}".into(),
            Map::new(),
        )
        .unwrap();
        let x: Json =
            serde_json::from_str(r#"{"b": [1, 2.5, true, null], "a": "é\n\"", "c": {}}"#).unwrap();
        let mut kwargs = Map::new();
        kwargs.insert("x".into(), x);
        let got = t
            .render(&Json::Array(vec![]), None, false, &kwargs)
            .unwrap();
        let want = concat!(
            r#"{"b": [1, 2.5, true, null], "a": "é\n\"", "c": {}}"#,
            "|",
            "{\n  \"b\": [\n    1,\n    2.5,\n    true,\n    null\n  ],\n  \"a\": \"é\\n\\\"\",\n  \"c\": {}\n}"
        );
        assert_eq!(got, want);
    }

    #[test]
    fn values_print_as_python_str() {
        let t = ChatTemplate::new(
            "{{ a }} {{ b }} {{ c | string }} {{ d }}".into(),
            Map::new(),
        )
        .unwrap();
        let kwargs: Map<String, Json> =
            serde_json::from_str(r#"{"a": true, "b": null, "c": 1e-6, "d": ["x", 1]}"#).unwrap();
        let got = t
            .render(&Json::Array(vec![]), None, false, &kwargs)
            .unwrap();
        assert_eq!(got, "True None 1e-06 ['x', 1]");
    }
}
