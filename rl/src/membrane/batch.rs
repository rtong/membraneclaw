//! What the trainer reads off an episode: the policy's tokens compacted, the reward's credits
//! placed on them, and the masks the actor's loss uses — notebook 07's `build_batch`, for one
//! episode.
//!
//! Positions inside a turn are characters, as 07 counts them: each token's text as
//! `tokenizer.decode([t])` writes it (a piece of a character split across tokens counts as one
//! U+FFFD), special tokens counting nothing, against offsets into the turn's decoded text.

use std::collections::HashMap;
use std::sync::LazyLock;

use fancy_regex::Regex;

use super::episode::{Episode, Harness, char_offset};
use super::pyjson::Value;
use super::reward::{self, Anchor, Score};
use super::tools;

/// One episode's rows, each with one entry per policy token (at least one).
#[derive(Debug, Clone, PartialEq)]
pub struct Rows {
    /// Where each policy token sits in the episode's `ids`.
    pub index: Vec<usize>,
    pub ids: Vec<u32>,
    pub mask: Vec<f64>,
    /// The answer turn's tokens.
    pub answer: Vec<f64>,
    /// A turn's closing `<|im_end|>`.
    pub turn_end: Vec<f64>,
    /// The answer's content, as opposed to its scaffolding.
    pub content: Vec<f64>,
    pub rewards: Vec<f64>,
    /// The turn each token belongs to.
    pub turn: Vec<usize>,
}

const NUM: &str = r"-?[\d.]+(?:e[-+]?\d+)?";

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap()
}

static DECISION_KEY: LazyLock<Regex> = LazyLock::new(|| re(r#""decision"\s*:\s*"#));
static LITERAL: LazyLock<Regex> = LazyLock::new(|| re(r"\A[^,}\]]+"));
static VIOLATIONS_OPEN: LazyLock<Regex> = LazyLock::new(|| re(r#""violations"\s*:\s*\["#));
static QUOTED: LazyLock<Regex> = LazyLock::new(|| re(r#""((?:[^"\\]|\\.)*)""#));
static VALUE_NUMBER: LazyLock<Regex> = LazyLock::new(|| re(r#"": (-?[\d.eE+-]+)"#));
static COMPARISON: LazyLock<Regex> =
    LazyLock::new(|| re(&format!(r"(\S+) ({NUM}) (meets|breaks) (<=|>=) ({NUM})")));
static COSTING: LazyLock<Regex> =
    LazyLock::new(|| re(&format!(r"(costing\.\S+) ({NUM})(?=\s*->)")));
static VERDICT: LazyLock<Regex> = LazyLock::new(|| re(r"-> (pass|fail)"));
static AGGREGATION: LazyLock<Regex> = LazyLock::new(|| re(r"(?m)^passing: (.+?) -> \w+: (.+)$"));
static ANY_CONTENT: LazyLock<Regex> =
    LazyLock::new(|| re(&format!(r"{NUM}|\b(?:meets|breaks|pass|fail|null)\b")));
static DECISION_VALUE: LazyLock<Regex> =
    LazyLock::new(|| re(r#""decision"\s*:\s*("(?:[^"\\]|\\.)*"|[^,}\]]+)"#));
static VIOLATIONS_LIST: LazyLock<Regex> = LazyLock::new(|| re(r#""violations"\s*:\s*\[[^\]]*\]"#));
static CALL_END: LazyLock<Regex> = LazyLock::new(|| re(r"</tool_call>"));

/// Character spans of a match group, offset by `base` characters, in `text`.
fn group_span(
    text: &str,
    m: &fancy_regex::Captures,
    g: usize,
    base: usize,
) -> Option<(usize, usize)> {
    m.get(g).map(|g| {
        (
            base + char_offset(text, g.start()),
            base + char_offset(text, g.end()),
        )
    })
}

/// Character spans of the answer's content (`wide`: what PPO may move): the whole decision
/// literal, quotes included; in a pass/fail question the whole violations list, else the text
/// of each violation; every number among the values. None where 07's version raises.
fn content_spans(answer: &str, passfail: bool) -> Option<Vec<(usize, usize)>> {
    let (_, (s, e), _) = reward::extract_answer(answer)?;
    let s0 = char_offset(answer, s);
    let text = &answer[s..e];
    let chars: Vec<char> = text.chars().collect();
    let mut spans = Vec::new();
    let m = DECISION_KEY.find(text).ok()??;
    let p = char_offset(text, m.end());
    if chars.get(p) == Some(&'"') {
        let mut e = p + 1;
        while *chars.get(e)? != '"' {
            e += if chars[e] == '\\' { 2 } else { 1 };
        }
        spans.push((s0 + p, s0 + e + 1));
    } else if let Some(lit) = LITERAL.find(&text[m.end()..]).ok().flatten() {
        spans.push((s0 + p, s0 + p + char_offset(&text[m.end()..], lit.end())));
    }
    if let Some(vm) = VIOLATIONS_OPEN.find(text).ok().flatten()
        && let Some(close) = text[vm.end()..].find(']')
    {
        let (vs, ve) = (vm.end(), vm.end() + close);
        if passfail {
            spans.push((
                s0 + char_offset(text, vm.end()) - 1,
                s0 + char_offset(text, ve) + 1,
            ));
        } else {
            let inner = &text[vs..ve];
            let base = s0 + char_offset(text, vs);
            for im in QUOTED.captures_iter(inner) {
                spans.extend(group_span(inner, &im.ok()?, 1, base));
            }
        }
    }
    let Some(base) = text.find("\"values\"") else {
        return Some(spans);
    };
    let tail = &text[base..];
    let base_c = s0 + char_offset(text, base);
    for km in VALUE_NUMBER.captures_iter(tail) {
        spans.extend(group_span(tail, &km.ok()?, 1, base_c));
    }
    Some(spans)
}

/// The content of the comparison lines before an answer's JSON: every simulated value, limit and
/// verdict, and the aggregation line's passing set and pick.
fn reasoning_content_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    for m in COMPARISON.captures_iter(text).flatten() {
        spans.extend(
            [
                group_span(text, &m, 2, 0),
                group_span(text, &m, 5, 0),
                group_span(text, &m, 3, 0),
            ]
            .into_iter()
            .flatten(),
        );
    }
    for m in COSTING.captures_iter(text).flatten() {
        spans.extend(group_span(text, &m, 2, 0));
    }
    for m in VERDICT.captures_iter(text).flatten() {
        spans.extend(group_span(text, &m, 1, 0));
    }
    for m in AGGREGATION.captures_iter(text).flatten() {
        spans.extend(
            [group_span(text, &m, 1, 0), group_span(text, &m, 2, 0)]
                .into_iter()
                .flatten(),
        );
    }
    spans
}

/// The content of an answer turn; if its JSON does not parse, every number and verdict word.
fn answer_content_spans(text: &str, case: &Value) -> Vec<(usize, usize)> {
    let passfail = case.get("decision_kind").and_then(Value::as_str) == Some("passfail");
    let mut spans = Vec::new();
    let mut parsed = false;
    if let Some((_, (s, e), _)) = reward::extract_answer(text) {
        parsed = true;
        match content_spans(&text[..e], passfail) {
            Some(c) => spans.extend(c),
            None => parsed = false, // a malformed sampled answer falls back to every number below
        }
        spans.extend(reasoning_content_spans(&text[..s]));
    }
    if !parsed {
        for m in ANY_CONTENT.find_iter(text).flatten() {
            spans.push((char_offset(text, m.start()), char_offset(text, m.end())));
        }
    }
    spans
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Close,
    Decision,
    Violations,
    Value(String),
}

/// Character offsets, in the answer turn, where each answer credit lands.
fn answer_anchor_offsets(text: &str, case: &Value) -> HashMap<Key, usize> {
    let mut out = HashMap::new();
    let Some((_, (s, e), _)) = reward::extract_answer(text) else {
        return out;
    };
    let s0 = char_offset(text, s);
    let region = &text[s..e];
    out.insert(Key::Close, char_offset(text, e));
    if let Some(m) = DECISION_VALUE.captures(region).ok().flatten() {
        out.insert(
            Key::Decision,
            s0 + char_offset(region, m.get(1).unwrap().end()),
        );
    }
    if let Some(m) = VIOLATIONS_LIST.find(region).ok().flatten() {
        out.insert(Key::Violations, s0 + char_offset(region, m.end()));
    }
    let gold_values = case
        .get("gold")
        .and_then(|g| g.get("values"))
        .and_then(Value::as_dict)
        .unwrap_or(&[]);
    for (path, _) in gold_values {
        let m = re(&format!(
            r#""{}"\s*:\s*"?({NUM})"#,
            fancy_regex::escape(path)
        ))
        .captures(region)
        .ok()
        .flatten();
        if m.is_none()
            && let Some((sec, leaf)) = path.split_once('.')
        {
            let section = re(&format!(r#""{}"\s*:\s*\{{"#, fancy_regex::escape(sec)));
            if let Some(ms) = section.find(region).ok().flatten() {
                let rest = &region[ms.end()..];
                let leaf_re = re(&format!(
                    r#""{}"\s*:\s*"?({NUM})"#,
                    fancy_regex::escape(leaf)
                ));
                if let Some(m2) = leaf_re.captures(rest).ok().flatten() {
                    let at = s0
                        + char_offset(region, ms.end())
                        + char_offset(rest, m2.get(1).unwrap().end());
                    out.insert(Key::Value(path.clone()), at);
                    continue;
                }
            }
        }
        if let Some(m) = m {
            out.insert(
                Key::Value(path.clone()),
                s0 + char_offset(region, m.get(1).unwrap().end()),
            );
        }
    }
    out
}

impl Harness {
    /// The end character offset, in the text the pieces spell, of each token.
    fn bounds(&self, ids: &[u32]) -> Vec<usize> {
        let mut n = 0;
        ids.iter()
            .map(|t| {
                if !self.special_ids.contains(t) {
                    n += self.piece(*t).chars().count();
                }
                n
            })
            .collect()
    }
}

/// The first token whose end reaches `offset`, else the last (-1 when there are none).
fn at(bounds: &[usize], offset: usize) -> isize {
    bounds
        .iter()
        .position(|&e| e >= offset)
        .map(|i| i as isize)
        .unwrap_or(bounds.len() as isize - 1)
}

/// The episode's rows, its credits from `score` (its reward under the run's settings).
pub fn episode_rows(h: &Harness, ep: &Episode, score: &Score, pad: u32) -> Rows {
    let mine_at: Vec<usize> = (0..ep.policy.len()).filter(|&k| ep.policy[k]).collect();
    let n = mine_at.len().max(1);
    let mine: Vec<u32> = mine_at.iter().map(|&k| ep.ids[k]).collect();
    let fill = n - mine_at.len();
    let index: Vec<usize> = mine_at
        .iter()
        .copied()
        .chain(std::iter::repeat_n(0, fill))
        .collect();
    let ids: Vec<u32> = mine
        .iter()
        .copied()
        .chain(std::iter::repeat_n(pad, fill))
        .collect();
    let mask: Vec<f64> = std::iter::repeat_n(1.0, mine_at.len())
        .chain(std::iter::repeat_n(0.0, fill))
        .collect();
    // Each turn's policy tokens, in compacted coordinates.
    let mut starts: Vec<usize> = ep
        .turn_starts
        .iter()
        .map(|&s| ep.policy[..s].iter().filter(|&&p| p).count())
        .collect();
    starts.push(mine_at.len());
    let turns: Vec<(usize, usize)> = (0..ep.turns.len())
        .map(|t| (starts[t], starts[t + 1]))
        .collect();
    let (mut answer, mut turn_end, mut content, mut reward) =
        (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut turn_of = vec![0usize; n];
    for (j, &(a, b)) in turns.iter().enumerate() {
        turn_of[a..b].fill(j);
        if b > a && mine[b - 1] == h.im_end {
            turn_end[b - 1] = 1.0;
        }
    }
    let answered = !ep.turns.is_empty()
        && Some(&ep.final_text) == ep.turns.last()
        && tools::parse_calls(&ep.final_text, &h.decls).is_empty();
    let mut offsets = HashMap::new();
    if answered {
        let (a, b) = *turns.last().unwrap();
        let bounds = h.bounds(&mine[a..b]);
        answer[a..b].fill(1.0);
        let starts_c: Vec<usize> = std::iter::once(0)
            .chain(bounds.iter().copied().take(bounds.len().saturating_sub(1)))
            .collect();
        for (s, e) in answer_content_spans(&ep.final_text, &ep.case) {
            for k in a..b {
                if starts_c[k - a] < e && bounds[k - a] > s {
                    content[k] = 1.0;
                }
            }
        }
        offsets = answer_anchor_offsets(&ep.final_text, &ep.case);
    }
    let mut call_rank: HashMap<usize, usize> = HashMap::new();
    for (anchor, amount) in &score.credits {
        let pos: isize = match anchor {
            Anchor::Call(i) => {
                let t = ep.calls[*i].turn.expect("a call's turn");
                let rank = call_rank.entry(t).or_insert(0);
                let k = *rank;
                *rank += 1;
                let (a, b) = turns[t];
                let text = &ep.turns[t];
                let ends: Vec<usize> = CALL_END
                    .find_iter(text)
                    .flatten()
                    .map(|m| char_offset(text, m.end()))
                    .collect();
                if k >= ends.len() {
                    b as isize - 1
                } else {
                    a as isize + at(&h.bounds(&mine[a..b]), ends[k])
                }
            }
            _ if answered => {
                let (a, b) = *turns.last().unwrap();
                let key = match anchor {
                    Anchor::Decision => Key::Decision,
                    Anchor::Violations => Key::Violations,
                    Anchor::Value(p) => Key::Value(p.clone()),
                    _ => Key::Close,
                };
                match offsets.get(&key).or_else(|| offsets.get(&Key::Close)) {
                    None => b as isize - 1,
                    Some(&off) => a as isize + at(&h.bounds(&mine[a..b]), off),
                }
            }
            _ => mine_at.len() as isize - 1,
        };
        reward[pos.max(0) as usize] += amount;
    }
    Rows {
        index,
        ids,
        mask,
        answer,
        turn_end,
        content,
        rewards: reward,
        turn: turn_of,
    }
}
