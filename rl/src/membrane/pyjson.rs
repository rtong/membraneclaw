//! JSON as Python's `json` module reads and writes it, and Python's `str()` of what it read.
//!
//! The harness judges what a model writes with `json.loads`, and some of that is JSON only to
//! Python: `NaN` and `Infinity`, a key given twice (the last value wins, in the first one's
//! place). Numbers keep Python's split — an integer stays an integer, anything with a fraction or
//! an exponent is a float — and a string holding a raw control character is refused, as
//! `json.loads` refuses it.

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    /// An integer beyond i64, kept as written, as Python's int would hold it.
    BigInt(String),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    /// Keys in the order they were first seen.
    Dict(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Dict(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Dict(items) => Some(items),
            _ => None,
        }
    }

    /// The value of an int or a float, as Python's `float()` gives it; None for anything else.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::BigInt(s) => s.parse().ok(),
            Value::Float(x) => Some(*x),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Python's truth value.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::BigInt(_) => true,
            Value::Float(x) => *x != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(v) => !v.is_empty(),
            Value::Dict(v) => !v.is_empty(),
        }
    }

    /// Set `key`, in place if it is there, else at the end — a Python dict's assignment.
    pub fn insert(&mut self, key: String, value: Value) {
        if let Value::Dict(items) = self {
            match items.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 = value,
                None => items.push((key, value)),
            }
        }
    }
}

/// `json.loads(text)`.
pub fn loads(text: &str) -> Result<Value, String> {
    let mut p = Parser {
        s: text.as_bytes(),
        text,
        i: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("extra data at {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    text: &'a str,
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn starts(&self, lit: &str) -> bool {
        self.s[self.i..].starts_with(lit.as_bytes())
    }

    fn value(&mut self) -> Result<Value, String> {
        let Some(&c) = self.s.get(self.i) else {
            return Err("expecting value at the end".into());
        };
        let constants: [(&str, Value); 6] = [
            ("null", Value::Null),
            ("true", Value::Bool(true)),
            ("false", Value::Bool(false)),
            ("NaN", Value::Float(f64::NAN)),
            ("Infinity", Value::Float(f64::INFINITY)),
            ("-Infinity", Value::Float(f64::NEG_INFINITY)),
        ];
        for (lit, v) in constants {
            if self.starts(lit) {
                self.i += lit.len();
                return Ok(v);
            }
        }
        match c {
            b'"' => Ok(Value::Str(self.string()?)),
            b'{' => self.object(),
            b'[' => self.array(),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(format!("expecting value at {}", self.i)),
        }
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        let digit = |p: &Self, k: usize| p.s.get(k).is_some_and(|b| b.is_ascii_digit());
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        if !digit(self, self.i) {
            return Err(format!("expecting value at {start}"));
        }
        if self.s[self.i] == b'0' {
            self.i += 1;
        } else {
            while digit(self, self.i) {
                self.i += 1;
            }
        }
        let mut float = false;
        if self.s.get(self.i) == Some(&b'.') && digit(self, self.i + 1) {
            float = true;
            self.i += 1;
            while digit(self, self.i) {
                self.i += 1;
            }
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            let mut k = self.i + 1;
            if matches!(self.s.get(k), Some(b'+' | b'-')) {
                k += 1;
            }
            if digit(self, k) {
                float = true;
                while digit(self, k) {
                    k += 1;
                }
                self.i = k;
            }
        }
        let lit = &self.text[start..self.i];
        if float {
            lit.parse::<f64>()
                .map(Value::Float)
                .map_err(|e| e.to_string())
        } else {
            Ok(lit
                .parse::<i64>()
                .map(Value::Int)
                .unwrap_or_else(|_| Value::BigInt(lit.to_owned())))
        }
    }

    fn hex4(&self, at: usize) -> Option<u32> {
        let h = self.text.get(at..at + 4)?;
        if h.bytes().all(|b| b.is_ascii_hexdigit()) {
            u32::from_str_radix(h, 16).ok()
        } else {
            None
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let run = self.i;
            while self.i < self.s.len()
                && !matches!(self.s[self.i], b'"' | b'\\')
                && self.s[self.i] >= 0x20
            {
                self.i += 1;
            }
            out.push_str(&self.text[run..self.i]);
            match self.s.get(self.i) {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    let esc = *self.s.get(self.i + 1).ok_or("unterminated string")?;
                    self.i += 2;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4(self.i).ok_or("invalid \\uXXXX escape")?;
                            self.i += 4;
                            let mut code = hi;
                            if (0xd800..0xdc00).contains(&hi)
                                && self.starts("\\u")
                                && let Some(lo) = self
                                    .hex4(self.i + 2)
                                    .filter(|lo| (0xdc00..0xe000).contains(lo))
                            {
                                code = 0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                                self.i += 6;
                            }
                            // A lone surrogate, which a Python str can hold and a Rust one cannot.
                            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(format!("invalid escape at {}", self.i - 1)),
                    }
                }
                Some(_) => return Err(format!("invalid control character at {}", self.i)),
            }
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        self.i += 1;
        let mut out = Value::Dict(Vec::new());
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(out);
        }
        loop {
            if self.s.get(self.i) != Some(&b'"') {
                return Err(format!("expecting property name at {}", self.i));
            }
            let key = self.string()?;
            self.ws();
            if self.s.get(self.i) != Some(&b':') {
                return Err(format!("expecting ':' at {}", self.i));
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            out.insert(key, v);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => {
                    self.i += 1;
                    self.ws();
                }
                Some(b'}') => {
                    self.i += 1;
                    return Ok(out);
                }
                _ => return Err(format!("expecting ',' or '}}' at {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Value, String> {
        self.i += 1;
        let mut out = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Value::List(out));
        }
        loop {
            out.push(self.value()?);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => {
                    self.i += 1;
                    self.ws();
                }
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::List(out));
                }
                _ => return Err(format!("expecting ',' or ']' at {}", self.i)),
            }
        }
    }
}

/// `json.dumps(value)`: `, ` and `: ` between items, non-ASCII escaped, `NaN` and `Infinity`
/// written out; with `sort_keys`, every object's keys sorted.
pub fn dumps(value: &Value, sort_keys: bool) -> String {
    let mut out = String::new();
    write_json(&mut out, value, sort_keys, true);
    out
}

/// `json.dumps(value, ensure_ascii=False)`: as `dumps`, non-ASCII written as itself.
pub fn dumps_unicode(value: &Value) -> String {
    let mut out = String::new();
    write_json(&mut out, value, false, false);
    out
}

fn write_json(out: &mut String, value: &Value, sort_keys: bool, ascii: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Value::BigInt(s) => out.push_str(s),
        Value::Float(x) if x.is_nan() => out.push_str("NaN"),
        Value::Float(x) if x.is_infinite() => {
            out.push_str(if *x > 0.0 { "Infinity" } else { "-Infinity" })
        }
        Value::Float(x) => out.push_str(&float_repr(*x)),
        Value::Str(s) => write_string(out, s, ascii),
        Value::List(items) => {
            out.push('[');
            for (k, v) in items.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                write_json(out, v, sort_keys, ascii);
            }
            out.push(']');
        }
        Value::Dict(items) => {
            let mut items: Vec<&(String, Value)> = items.iter().collect();
            if sort_keys {
                items.sort_by(|a, b| a.0.cmp(&b.0));
            }
            out.push('{');
            for (k, (key, v)) in items.into_iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                write_string(out, key, ascii);
                out.push_str(": ");
                write_json(out, v, sort_keys, ascii);
            }
            out.push('}');
        }
    }
}

fn write_string(out: &mut String, s: &str, ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (ascii && (c as u32) > 0x7e) => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr()` of a float: the shortest digits that read back, positional from 1e-4 up
/// to 1e16 and with an exponent of at least two digits outside that (`1e-06`, `1e+16`).
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
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

/// Python's built-in `sum()` of floats, started from the int 0. Since Python 3.12 it carries
/// a Neumaier compensation term through the sum and adds it at the end, so its result can
/// differ in the last bit from adding the floats in order.
pub fn py_sum(items: impl IntoIterator<Item = f64>) -> f64 {
    let mut items = items.into_iter();
    let Some(first) = items.next() else {
        return 0.0;
    };
    let (mut f, mut c) = (0.0 + first, 0.0f64);
    for x in items {
        let t = f + x;
        c += if f.abs() >= x.abs() {
            (f - t) + x
        } else {
            (x - t) + f
        };
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// Python's `str()` of a value `json.loads` returned.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::Str(s) => s.clone(),
        _ => py_repr(value),
    }
}

/// Python's `repr()` of a value `json.loads` returned.
pub fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(b) => (if *b { "True" } else { "False" }).into(),
        Value::Int(i) => i.to_string(),
        Value::BigInt(s) => s.clone(),
        Value::Float(x) => float_repr(*x),
        Value::Str(s) => repr_str(s),
        Value::List(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Dict(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|(k, v)| format!("{}: {}", repr_str(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn repr_str(s: &str) -> String {
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

/// Python's `float(text)`: surrounding whitespace, a sign, `_` between digits, `inf`,
/// `infinity` and `nan` in any case. None where Python raises ValueError.
pub fn py_float(text: &str) -> Option<f64> {
    let t = text.trim();
    let (sign, body) = match t.as_bytes().first() {
        Some(b'-') => (-1.0, &t[1..]),
        Some(b'+') => (1.0, &t[1..]),
        _ => (1.0, t),
    };
    let lower = body.to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" {
        return Some(sign * f64::INFINITY);
    }
    if lower == "nan" {
        return Some(f64::NAN);
    }
    // Digits, at most one point, an optional exponent; `_` only between two digits.
    let bytes = body.as_bytes();
    let mut clean = String::with_capacity(body.len());
    let (mut digits, mut point, mut exp) = (false, false, false);
    for (k, &b) in bytes.iter().enumerate() {
        match b {
            b'0'..=b'9' => {
                digits = true;
                clean.push(b as char);
            }
            b'_' => {
                let between = k > 0
                    && bytes[k - 1].is_ascii_digit()
                    && bytes.get(k + 1).is_some_and(u8::is_ascii_digit);
                if !between {
                    return None;
                }
            }
            b'.' if !point && !exp => {
                point = true;
                clean.push('.');
            }
            b'e' | b'E' if digits && !exp => {
                exp = true;
                clean.push('e');
            }
            b'+' | b'-' if k > 0 && matches!(bytes[k - 1], b'e' | b'E') => clean.push(b as char),
            _ => return None,
        }
    }
    if !digits {
        return None;
    }
    clean.parse::<f64>().ok().map(|x| sign * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_what_python_reads() {
        let v = loads(r#" {"a": 1, "b": [1.5, -2e3, NaN, -Infinity], "a": "x\u00e9\ud83d\ude00", "c": null} "#).unwrap();
        let items = v.as_dict().unwrap();
        assert_eq!(items[0], ("a".into(), Value::Str("xé😀".into())));
        assert_eq!(items.len(), 3);
        let b = v.get("b").unwrap().as_list().unwrap();
        assert_eq!(b[0], Value::Float(1.5));
        assert_eq!(b[1], Value::Float(-2000.0));
        assert!(b[2].as_f64().unwrap().is_nan());
        assert_eq!(b[3], Value::Float(f64::NEG_INFINITY));
        assert_eq!(
            loads("12345678901234567890").unwrap(),
            Value::BigInt("12345678901234567890".into())
        );
        for bad in [
            "{\"a\": 1,}",
            "[1,]",
            "01",
            "{'a': 1}",
            "\"a\tb\"",
            "1 2",
            "{\"a\" 1}",
            "1e",
        ] {
            assert!(loads(bad).is_err(), "{bad}");
        }
        assert_eq!(loads("1.0").unwrap(), Value::Float(1.0));
        assert_eq!(loads("-0").unwrap(), Value::Int(0));
    }

    #[test]
    fn writes_what_python_writes() {
        let v = loads(r#"{"b": [1, 2.5, true, null, "é"], "a": 1e-06}"#).unwrap();
        assert_eq!(
            dumps(&v, false),
            r#"{"b": [1, 2.5, true, null, "\u00e9"], "a": 1e-06}"#
        );
        assert_eq!(
            dumps(&v, true),
            r#"{"a": 1e-06, "b": [1, 2.5, true, null, "\u00e9"]}"#
        );
        assert_eq!(py_str(&v), "{'b': [1, 2.5, True, None, 'é'], 'a': 1e-06}");
    }

    #[test]
    fn float_reads_as_python_float() {
        let cases = [
            ("1.e5", Some(100000.0)),
            (".5", Some(0.5)),
            ("-.5", Some(-0.5)),
            ("5.", Some(5.0)),
            ("1_000", Some(1000.0)),
            (" 2 ", Some(2.0)),
            ("+1.5", Some(1.5)),
            ("1e-3", Some(0.001)),
            ("1__0", None),
            ("_1", None),
            ("1e", None),
            ("abc", None),
            ("", None),
        ];
        for (text, want) in cases {
            assert_eq!(py_float(text), want, "{text:?}");
        }
        assert_eq!(py_float("-Infinity"), Some(f64::NEG_INFINITY));
        assert!(py_float("nan").unwrap().is_nan());
    }
}
