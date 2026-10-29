//! Notebook 07's reward for one episode, and the credits a trainer places on its tokens.
//!
//! Six components, weighted: the simulate calls' fidelity to the scenario, the decision, the
//! violations of a pass/fail question, the values asked for, the answer's format, and efficiency.
//! The decision and violations pay in full only when the simulations that prove them ran
//! (`evidence`); without, `UNGATED` of it. Three settings change it, each a flag of 07's PPO:
//! whether a simulate call the tool rejected earns call credit, where the calls component's credit
//! lands (on each call, or as each turn's change to the component), and whether an input stated
//! in another unit proves its point only when convert_units returned it in an earlier turn.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use fancy_regex::Regex;

use super::pyjson::{self, Value};
use super::tools::{self, COMPARE, CONVERT, Call, RO, SWRO};

pub const W_CALLS: f64 = 0.25;
pub const W_DECISION: f64 = 0.35;
pub const W_VIOLATIONS: f64 = 0.10;
pub const W_VALUES: f64 = 0.20;
pub const W_FORMAT: f64 = 0.05;
pub const W_EFFICIENCY: f64 = 0.05;
/// Relative tolerance on each tool argument.
pub const ARG_TOL: f64 = 0.005;
/// Relative tolerance on each reported number.
pub const VALUE_TOL: f64 = 0.015;
/// Share of the decision and violations credit paid without the simulations that prove it.
pub const UNGATED: f64 = 0.3;
const NONE_WORDS: [&str; 8] = [
    "none",
    "null",
    "n/a",
    "no option",
    "not feasible",
    "infeasible",
    "nothing",
    "no",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallCredit {
    /// Each simulate call earns its share of the calls component at its own end.
    Call,
    /// Each turn earns the change it makes to the calls component, at its last simulate call.
    Delta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settings {
    pub err_call_credit: bool,
    pub call_credit: CallCredit,
    pub provenance: bool,
}

impl Default for Settings {
    /// 03–07's reward.
    fn default() -> Self {
        Self {
            err_call_credit: true,
            call_credit: CallCredit::Call,
            provenance: false,
        }
    }
}

/// Where a credit lands.
#[derive(Debug, Clone, PartialEq)]
pub enum Anchor {
    /// The end of the episode's i-th call.
    Call(usize),
    Decision,
    Violations,
    Value(String),
    /// The answer's closing brace.
    Close,
}

#[derive(Debug, Clone)]
pub struct Score {
    pub total: f64,
    /// In the order 07 adds them, which sets the order of the sums.
    pub components: Vec<(&'static str, f64)>,
    pub evidence: bool,
    pub covered: Vec<String>,
    pub n_calls: usize,
    pub n_converts: usize,
    pub n_compares: usize,
    pub decision_correct: bool,
    pub credits: Vec<(Anchor, f64)>,
}

impl Score {
    pub fn component(&self, name: &str) -> Option<f64> {
        self.components
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| *v)
    }
}

fn weight(name: &str) -> f64 {
    match name {
        "calls" => W_CALLS,
        "decision" => W_DECISION,
        "violations" => W_VIOLATIONS,
        "values" => W_VALUES,
        "format" => W_FORMAT,
        "efficiency" => W_EFFICIENCY,
        _ => unreachable!("{name}"),
    }
}

/// mcp_watertap's defaults for each tool's arguments.
pub fn defaults(tool: &str) -> Option<Vec<(&'static str, Value)>> {
    let f = Value::Float;
    let s = |x: &str| Value::Str(x.into());
    match tool {
        RO => Some(vec![
            ("feed_flow_mass_kg_s", f(1.0)),
            ("feed_nacl_mass_frac", f(0.035)),
            ("feed_pressure_bar", f(50.0)),
            ("feed_temperature_c", f(25.0)),
            ("membrane_area_m2", f(50.0)),
            ("A_comp", f(4.2e-12)),
            ("B_comp", f(3.5e-8)),
            ("permeate_pressure_bar", f(1.01325)),
            ("pressure_drop_bar", f(3.0)),
            ("channel_height_m", f(0.002)),
            ("spacer_porosity", f(0.75)),
            ("module_length_m", f(20.0)),
            ("cp_modulus", f(1.1)),
            ("mass_transfer_coeff", f(2e-5)),
            ("concentration_polarization", s("calculated")),
            ("mass_transfer_coefficient", s("calculated")),
        ]),
        SWRO => Some(vec![
            ("erd_type", s("pressure_exchanger")),
            ("feed_flow_m3_s", f(0.3092)),
            ("feed_tds_g_L", f(35.0)),
            ("feed_tss_g_L", f(0.03)),
            ("feed_temperature_c", f(24.85)),
            ("ro_area_m2", Value::Null),
            ("A_comp", f(4.2e-12)),
            ("B_comp", f(3.5e-8)),
            ("p1_pressure_bar", f(70.0)),
            ("p1_efficiency", f(0.80)),
            ("pxr_efficiency", f(0.95)),
            ("p2_efficiency", f(0.80)),
            ("erd_efficiency", f(0.95)),
        ]),
        _ => None,
    }
}

fn lookup<'a>(table: &'a [(&'static str, Value)], key: &str) -> Option<&'a Value> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
}

static NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+(?:[eE][-+]?\d+)?").unwrap()
});

/// A number in what a model wrote: a JSON number, or the first number in a string.
pub fn parse_number(x: Option<&Value>) -> Option<f64> {
    match x? {
        Value::Int(i) => Some(*i as f64),
        Value::BigInt(s) => s.parse::<f64>().ok().filter(|v| v.is_finite()),
        Value::Float(v) => v.is_finite().then_some(*v),
        Value::Str(s) => {
            let m = NUMBER.find(s).ok()??;
            pyjson::py_float(&m.as_str().replace(',', ""))
        }
        _ => None,
    }
}

/// Whether an argument as passed (`a`, None when absent) matches what was expected (`b`).
fn same_arg(a: &Value, b: &Value) -> bool {
    match b {
        Value::Null | Value::Str(_) => {
            let got = (!a.is_null()).then(|| pyjson::py_str(a).trim().to_lowercase());
            let want = b.as_str().filter(|s| !s.is_empty()).map(str::to_lowercase);
            got == want
        }
        _ => {
            let want = b.as_f64().unwrap_or(f64::NAN);
            parse_number(Some(a)).is_some_and(|x| (x - want).abs() <= ARG_TOL * want.abs())
        }
    }
}

/// How faithfully one call reproduces a point of the scenario: a score in [0, 1] and the point
/// it covers. Scored against the closest point: the share of the point's stated arguments the
/// call got right, less 0.25 for each argument it set that the user never gave (unless it is the
/// tool's default) and for each argument the tool does not have. A call covers a point when it
/// gets every argument right, changes nothing else and ran cleanly.
pub fn call_fidelity(case: &Value, call: &Call) -> (f64, Option<String>) {
    let tool = case.get("tool").and_then(Value::as_str);
    let (Some(name), Some(Value::Dict(args))) = (call.name.as_deref(), call.arguments.as_ref())
    else {
        return (0.0, None);
    };
    if Some(name) != tool {
        return (0.0, None);
    }
    let defaults = defaults(name).expect("a simulator");
    let known: Vec<&(String, Value)> = args
        .iter()
        .filter(|(k, _)| lookup(&defaults, k).is_some())
        .collect();
    let unknown = args.len() - known.len();
    let ran = !tools::result_is_error(call.result.as_deref());
    let (mut best, mut covered) = (0.0f64, None);
    for point in case.get("points").and_then(Value::as_list).unwrap_or(&[]) {
        let expected = point.get("args").and_then(Value::as_dict).unwrap_or(&[]);
        let given = |k: &str| known.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        let right = expected
            .iter()
            .filter(|(k, v)| {
                same_arg(
                    given(k).unwrap_or_else(|| lookup(&defaults, k).unwrap_or(&Value::Null)),
                    v,
                )
            })
            .count();
        let stray = known
            .iter()
            .filter(|(k, v)| {
                !expected.iter().any(|(e, _)| e == k) && !same_arg(v, lookup(&defaults, k).unwrap())
            })
            .count();
        let score =
            (right as f64 / expected.len() as f64 - 0.25 * (stray + unknown) as f64).max(0.0);
        if right == expected.len() && stray == 0 && ran {
            covered = point.get("key").and_then(Value::as_str).map(str::to_owned);
        }
        best = best.max(score);
    }
    (best, covered)
}

/// (start, end) byte offsets of every top-level `{...}` in `text`, string-aware.
pub fn json_spans(text: &str) -> Vec<(usize, usize)> {
    let (mut spans, mut depth, mut start, mut in_str, mut esc) =
        (Vec::new(), 0usize, 0usize, false, false);
    for (i, ch) in text.char_indices() {
        if in_str {
            if esc {
                esc = false;
            } else if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                in_str = false;
            }
        } else if ch == '"' {
            in_str = depth > 0;
        } else if ch == '{' {
            if depth == 0 {
                start = i;
            }
            depth += 1;
        } else if ch == '}' && depth > 0 {
            depth -= 1;
            if depth == 0 {
                spans.push((start, i + 1));
            }
        }
    }
    spans
}

/// The first JSON object with a `"decision"` key: the object, its byte span, and whether nothing
/// but whitespace follows it.
pub fn extract_answer(text: &str) -> Option<(Value, (usize, usize), bool)> {
    for (s, e) in json_spans(text) {
        let Ok(obj) = pyjson::loads(&text[s..e]) else {
            continue;
        };
        if obj.get("decision").is_some() {
            return Some((obj, (s, e), text[e..].trim().is_empty()));
        }
    }
    None
}

fn none_like(x: Option<&Value>) -> bool {
    match x {
        None | Some(Value::Null) => true,
        Some(Value::Str(s)) => NONE_WORDS.contains(&s.trim().to_lowercase().trim_end_matches('.')),
        _ => false,
    }
}

static LABEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\A\s*(?:option|membrane|configuration|config)?\s*([A-Da-d])\b").unwrap()
});

pub fn decision_correct(case: &Value, pred: Option<&Value>) -> bool {
    let gold = case
        .get("gold")
        .and_then(|g| g.get("decision"))
        .unwrap_or(&Value::Null);
    let kind = case
        .get("decision_kind")
        .and_then(Value::as_str)
        .unwrap_or("");
    if none_like(pred) {
        return gold.is_null();
    }
    let pred = pred.unwrap();
    if gold.is_null() {
        return false;
    }
    match kind {
        "passfail" => {
            let gold = gold.as_str().unwrap_or("");
            pred.as_str().is_some_and(|p| {
                let p = p.trim().to_lowercase();
                let p = p.trim_end_matches('.');
                p == gold || p == format!("{gold}s") || p == format!("{gold}ed")
            })
        }
        "number" => {
            let Some(x) = parse_number(Some(pred)) else {
                return false;
            };
            let g = gold.as_f64().unwrap_or(f64::NAN);
            let tool = case
                .get("gold")
                .and_then(|g| g.get("decision_tool"))
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN);
            (x - g).abs() <= 0.002 * g.abs() || (x - tool).abs() <= ARG_TOL * tool.abs()
        }
        _ => match LABEL.captures(&pyjson::py_str(pred)).ok().flatten() {
            Some(m) => Some(m.get(1).unwrap().as_str().to_uppercase().as_str()) == gold.as_str(),
            None => false,
        },
    }
}

/// Nested objects' leaves keyed by their dotted paths; anything but an object gives nothing.
fn flatten(d: Option<&Value>, prefix: &str, out: &mut Value) {
    if let Some(Value::Dict(items)) = d {
        for (k, v) in items {
            let key = format!("{prefix}{k}");
            if matches!(v, Value::Dict(_)) {
                flatten(Some(v), &format!("{key}."), out);
            } else {
                out.insert(key, v.clone());
            }
        }
    }
}

/// Input units and how a stated number becomes the tool's: (quantity, unit, level, factor, offset).
/// Level 0 is the tool's own unit.
const INPUT_UNITS: &[(&str, &str, u8, f64, f64)] = &[
    ("ro_flow", "kg/s", 0, 1.0, 0.0),
    ("ro_flow", "t/h", 2, 3.6, 0.0),
    ("ro_flow", "kg/h", 2, 3600.0, 0.0),
    ("salinity", "mass fraction", 0, 1.0, 0.0),
    ("salinity", "g/kg", 1, 1000.0, 0.0),
    ("salinity", "permille", 1, 1000.0, 0.0),
    ("salinity", "pct", 1, 100.0, 0.0),
    ("salinity", "mg/kg", 1, 1e6, 0.0),
    ("temp", "C", 0, 1.0, 0.0),
    ("temp", "F", 2, 1.8, 32.0),
    ("area", "m2", 0, 1.0, 0.0),
    ("area", "ft2", 2, 1.0 / 0.09290304, 0.0),
    ("A", "SI", 0, 1e12, 0.0),
    ("A", "LMH/bar", 2, 3.6e11, 0.0),
    ("B", "SI", 0, 1e8, 0.0),
    ("B", "LMH", 2, 3.6e6, 0.0),
    ("pressure", "bar", 0, 1.0, 0.0),
    ("pressure", "kPa", 1, 100.0, 0.0),
    ("pressure", "MPa", 1, 0.1, 0.0),
    ("pressure", "psi", 2, 1e5 / 6894.757293168, 0.0),
    ("swro_flow", "m3/s", 0, 1.0, 0.0),
    ("swro_flow", "L/s", 1, 1000.0, 0.0),
    ("swro_flow", "m3/h", 2, 3600.0, 0.0),
    ("swro_flow", "m3/d", 2, 86400.0, 0.0),
    ("swro_flow", "MGD", 2, 86400.0 / 3785.411784, 0.0),
    ("tds", "g/L", 0, 1.0, 0.0),
    ("tds", "mg/L", 1, 1000.0, 0.0),
    ("tds", "kg/m3", 1, 1.0, 0.0),
    ("eff", "fraction", 0, 1.0, 0.0),
    ("eff", "pct", 1, 100.0, 0.0),
];

/// A stated unit whose number differs from the tool's (a relabel such as kg/m³ for g/L changes
/// nothing).
fn needs_conversion(quantity: &str, unit: &str) -> bool {
    INPUT_UNITS
        .iter()
        .find(|(q, u, ..)| *q == quantity && *u == unit)
        .is_some_and(|&(_, _, level, factor, offset)| level > 0 && (factor != 1.0 || offset != 0.0))
}

/// The arguments of `point` the question states in another unit than the tool's.
fn converted_inputs(case: &Value, point: &str) -> BTreeSet<String> {
    let mut need = BTreeSet::new();
    let mut add = |inputs: Option<&Value>| {
        for (k, inp) in inputs.and_then(Value::as_dict).unwrap_or(&[]) {
            let q = inp.get("q").and_then(Value::as_str).unwrap_or("");
            let unit = inp.get("unit").and_then(Value::as_str).unwrap_or("");
            if needs_conversion(q, unit) {
                need.insert(k.clone());
            }
        }
    };
    add(case.get("inputs"));
    for p in case.get("points").and_then(Value::as_list).unwrap_or(&[]) {
        if p.get("key").and_then(Value::as_str) == Some(point) {
            add(p.get("inputs"));
        }
    }
    need
}

/// The number a convert_units call returned; None for another tool or an error.
fn converted_value(call: &Call) -> Option<f64> {
    if call.name.as_deref() != Some(CONVERT) {
        return None;
    }
    let out = pyjson::loads(call.result.as_deref().unwrap_or("")).ok()?;
    match out {
        Value::Dict(_) => parse_number(out.get("value")),
        _ => None,
    }
}

/// Every input of `point` the question states in another unit, as call i passed it, is a value a
/// convert_units call of an earlier turn returned (within ARG_TOL). A call without a turn counts
/// as a turn of its own.
fn tool_converted(case: &Value, calls: &[Call], i: usize, point: &str) -> bool {
    let turn = |j: usize| calls[j].turn.unwrap_or(j);
    let got: Vec<f64> = (0..i)
        .filter(|&j| turn(j) < turn(i))
        .filter_map(|j| converted_value(&calls[j]))
        .collect();
    let defaults = defaults(calls[i].name.as_deref().unwrap_or("")).unwrap_or_default();
    let args = calls[i].arguments.as_ref();
    converted_inputs(case, point).iter().all(|k| {
        let passed = args
            .and_then(|a| a.get(k))
            .or_else(|| lookup(&defaults, k))
            .unwrap_or(&Value::Null);
        got.iter().any(|&v| same_arg(passed, &Value::Float(v)))
    })
}

/// The reward of an episode with these calls (in order, each with its result and turn) and this
/// last assistant turn.
pub fn score(case: &Value, calls: &[Call], final_text: &str, settings: &Settings) -> Score {
    let is_sim: Vec<bool> = calls
        .iter()
        .map(|c| !tools::is_local(c.name.as_deref()))
        .collect();
    let n = is_sim.iter().filter(|&&s| s).count();
    let mut fid: Vec<(f64, Option<String>)> = calls
        .iter()
        .zip(&is_sim)
        .map(|(c, &sim)| {
            if sim && (settings.err_call_credit || !tools::result_is_error(c.result.as_deref())) {
                call_fidelity(case, c)
            } else {
                (0.0, None)
            }
        })
        .collect();
    if settings.provenance {
        for (i, (_, covered)) in fid.iter_mut().enumerate() {
            if covered
                .as_deref()
                .is_some_and(|point| !tool_converted(case, calls, i, point))
            {
                *covered = None;
            }
        }
    }
    let covered: BTreeSet<String> = fid
        .iter()
        .filter_map(|(_, c)| c.clone())
        .filter(|c| !c.is_empty())
        .collect();
    let proof: BTreeSet<String> = case
        .get("proof")
        .and_then(Value::as_list)
        .unwrap_or(&[])
        .iter()
        .filter_map(|p| p.as_str().map(str::to_owned))
        .collect();
    let evidence = proof.is_subset(&covered);

    let mut comps: Vec<(&'static str, f64)> = Vec::new();
    let sim_sum = pyjson::py_sum(
        fid.iter()
            .zip(&is_sim)
            .filter(|(_, s)| **s)
            .map(|((f, _), _)| *f),
    );
    comps.push(("calls", if n > 0 { sim_sum / n as f64 } else { 0.0 }));
    let answer = extract_answer(final_text);
    let (ans, clean_end) = match &answer {
        Some((obj, _, clean)) => (Some(obj), *clean),
        None => (None, false),
    };
    let correct = ans.is_some_and(|a| a.truthy())
        && decision_correct(case, ans.and_then(|a| a.get("decision")));
    let gate = if evidence { 1.0 } else { UNGATED };
    comps.push(("decision", if correct { gate } else { 0.0 }));
    let gold = case.get("gold").unwrap_or(&Value::Null);
    if case.get("decision_kind").and_then(Value::as_str) == Some("passfail") {
        let pred = ans.and_then(|a| a.get("violations"));
        let pred_set: Option<BTreeSet<String>> = pred.and_then(Value::as_list).map(|l| {
            l.iter()
                .map(|x| pyjson::py_str(x).trim().to_owned())
                .collect()
        });
        let gold_set: BTreeSet<String> = gold
            .get("violations")
            .and_then(Value::as_list)
            .unwrap_or(&[])
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        let v = match pred_set {
            None => 0.0,
            Some(p) if gold_set.is_empty() => {
                if p.is_empty() {
                    1.0
                } else {
                    0.0
                }
            }
            Some(p) => {
                (2 * p.intersection(&gold_set).count()) as f64 / (p.len() + gold_set.len()) as f64
            }
        };
        comps.push(("violations", gate * v));
    }
    let gold_vals = gold.get("values").and_then(Value::as_dict).unwrap_or(&[]);
    let mut value_hits: Vec<(String, bool)> = Vec::new();
    if !gold_vals.is_empty() {
        let mut pred_vals = Value::Dict(Vec::new());
        flatten(ans.and_then(|a| a.get("values")), "", &mut pred_vals);
        for (path, gv) in gold_vals {
            let gv = gv.as_f64().unwrap_or(f64::NAN);
            let x = parse_number(pred_vals.get(path));
            value_hits.push((
                path.clone(),
                x.is_some_and(|x| (x - gv).abs() <= VALUE_TOL * gv.abs()),
            ));
        }
        let hits = value_hits.iter().filter(|(_, h)| *h).count();
        comps.push(("values", hits as f64 / gold_vals.len() as f64));
    }
    let format = ans.is_some_and(|a| a.truthy())
        && clean_end
        && matches!(ans.and_then(|a| a.get("violations")), Some(Value::List(_)))
        && matches!(ans.and_then(|a| a.get("values")), Some(Value::Dict(_)));
    comps.push(("format", if format { 1.0 } else { 0.0 }));
    let optimal = case
        .get("optimal_calls")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    comps.push((
        "efficiency",
        if correct && evidence && n > 0 {
            (optimal / n as f64).min(1.0)
        } else {
            0.0
        },
    ));
    let z = pyjson::py_sum(comps.iter().map(|(k, _)| weight(k)));
    let total = pyjson::py_sum(comps.iter().map(|(k, v)| weight(k) * v)) / z;
    let comp = |name: &str| {
        comps
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    };

    let mut credits: Vec<(Anchor, f64)> = match settings.call_credit {
        CallCredit::Delta => {
            // A call without a turn counts as a turn of its own.
            let turn: Vec<usize> = calls
                .iter()
                .enumerate()
                .map(|(i, c)| c.turn.unwrap_or(i))
                .collect();
            let mut amounts = vec![0.0; calls.len()];
            let (mut prev, mut s, mut k, mut last) = (0.0, 0.0, 0usize, None);
            for (i, ((f, _), &sim)) in fid.iter().zip(&is_sim).enumerate() {
                if sim {
                    s += f;
                    k += 1;
                    last = Some(i);
                }
                if let Some(l) = last
                    && (i + 1 == calls.len() || turn[i + 1] != turn[i])
                {
                    let cj = W_CALLS * s / k as f64 / z;
                    amounts[l] = cj - prev;
                    prev = cj;
                    last = None;
                }
            }
            amounts
                .into_iter()
                .enumerate()
                .map(|(i, a)| (Anchor::Call(i), a))
                .collect()
        }
        CallCredit::Call => fid
            .iter()
            .zip(&is_sim)
            .enumerate()
            .map(|(i, ((f, _), &sim))| {
                (
                    Anchor::Call(i),
                    if sim { W_CALLS * f / n as f64 / z } else { 0.0 },
                )
            })
            .collect(),
    };
    credits.push((Anchor::Decision, W_DECISION * comp("decision") / z));
    if comps.iter().any(|(k, _)| *k == "violations") {
        credits.push((Anchor::Violations, W_VIOLATIONS * comp("violations") / z));
    }
    for (path, hit) in &value_hits {
        let hit = if *hit { 1.0 } else { 0.0 };
        credits.push((
            Anchor::Value(path.clone()),
            W_VALUES * hit / gold_vals.len() as f64 / z,
        ));
    }
    credits.push((
        Anchor::Close,
        (W_FORMAT * comp("format") + W_EFFICIENCY * comp("efficiency")) / z,
    ));
    Score {
        total,
        evidence,
        covered: covered.into_iter().collect(),
        n_calls: n,
        n_converts: calls
            .iter()
            .filter(|c| c.name.as_deref() == Some(CONVERT))
            .count(),
        n_compares: calls
            .iter()
            .filter(|c| c.name.as_deref() == Some(COMPARE))
            .count(),
        decision_correct: correct,
        components: comps,
        credits,
    }
}
