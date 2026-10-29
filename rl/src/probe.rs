//! Notebook 10's probes, the same for its checkpoints and for 09's (re-measured on this
//! machine, LoRA unmerged): four held-out probes and the natural-error probe, each episode
//! scored with err0, two checkpoints compared episode by episode.
//!
//! - `test` greedy: the first 120 test questions.
//! - `holdout_shift` greedy: its 120 questions.
//! - `dev_forced` greedy: the 144 dev questions of 07's recovery file, each started from its
//!   slipped turns (07's forced probe).
//! - `dev_t1`: the first 120 dev questions sampled at T=1, the same draws for every checkpoint.
//! - natural errors: each checkpoint samples dev at T=1 twice; the first error state of every
//!   episode (as the training's error states) is pooled; every checkpoint continues greedily
//!   from every state.

use std::path::Path;

use serde_json::json;

use crate::generate::Generator;
use crate::grpo;
use crate::membrane::episode::{Episode, Harness, Start};
use crate::membrane::pyjson::{self, Value};
use crate::membrane::reward::{CallCredit, Settings, score};
use crate::membrane::tools::Tools;

pub const ERR0: Settings = Settings {
    err_call_credit: false,
    call_credit: CallCredit::Delta,
    provenance: false,
};

pub fn load_jsonl(path: &Path) -> Result<Vec<Value>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(pyjson::loads)
        .collect()
}

pub struct Probe {
    pub name: &'static str,
    pub cases: Vec<Value>,
    pub starts: Vec<Start>,
    pub temperature: f32,
    /// Keys episodes are paired by.
    pub keys: Vec<String>,
}

fn id(case: &Value) -> String {
    case.get("id").and_then(Value::as_str).unwrap_or("").to_owned()
}

/// The four held-out probes, from the notebooks' data directory (`smoke_test35/`).
pub fn held_out(h: &Harness, data: &Path) -> Result<Vec<Probe>, String> {
    let split = |name: &str| load_jsonl(&data.join(format!("06_direction_and_compare_data/{name}.jsonl")));
    let plain = |name: &'static str, cases: Vec<Value>, temperature: f32| Probe {
        name,
        keys: cases.iter().map(id).collect(),
        starts: vec![Start::default(); cases.len()],
        cases,
        temperature,
    };
    let test: Vec<Value> = split("test")?.into_iter().take(120).collect();
    let shift = split("holdout_shift")?;
    let dev = split("dev")?;
    let recovery = load_jsonl(&data.join("07_recover_from_errors_data/dev_recovery.jsonl"))?;
    let mut slipped = std::collections::HashMap::new();
    for r in &recovery {
        let turns: Vec<_> = r
            .get("turns")
            .and_then(Value::as_list)
            .unwrap_or(&[])
            .iter()
            .filter(|t| !matches!(t.get("label"), Some(Value::Bool(true)) | None))
            .map(|t| h.recorded_turn(t))
            .collect();
        slipped.insert(id(r), turns);
    }
    let forced_cases: Vec<Value> = dev.iter().filter(|c| slipped.contains_key(&id(c))).cloned().collect();
    let forced = Probe {
        name: "dev_forced",
        keys: forced_cases.iter().map(id).collect(),
        starts: forced_cases
            .iter()
            .map(|c| Start {
                turns: slipped[&id(c)].clone(),
                tokens_count: false,
            })
            .collect(),
        cases: forced_cases,
        temperature: 0.0,
    };
    let dev_t1: Vec<Value> = dev.into_iter().take(120).collect();
    Ok(vec![
        plain("test", test, 0.0),
        plain("holdout_shift", shift, 0.0),
        forced,
        plain("dev_t1", dev_t1, 1.0),
    ])
}

/// One probe episode's record.
pub fn record(key: &str, ep: &Episode, h: &Harness) -> serde_json::Value {
    let s = score(&ep.case, &ep.calls, &ep.final_text, &ERR0);
    let calls: Vec<serde_json::Value> = ep
        .calls
        .iter()
        .map(|c| {
            json!({
                "name": c.name,
                "arguments": c.arguments.as_ref().map(|a| pyjson::dumps(a, false)),
                "result": c.result,
                "turn": c.turn,
            })
        })
        .collect();
    let _ = h;
    json!({
        "key": key,
        "id": id(&ep.case),
        "reward": s.total,
        "components": s.components.iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
        "decision_correct": s.decision_correct,
        "evidence": s.evidence,
        "calls": calls,
        "turns": ep.turns,
        "policy_tokens": ep.policy_tokens(),
        "truncated": ep.truncated,
        "answer_closed": ep.answer_closed,
    })
}

/// Runs a probe in batches of `batch` rows, the sampler at `seed` (so that two checkpoints draw
/// alike). Returns the episodes in the probe's order.
pub fn run(
    generator: &mut Generator,
    h: &Harness,
    tools: &mut (dyn Tools + Send),
    probe: &Probe,
    batch: usize,
    seed: u64,
) -> Result<Vec<Episode>, String> {
    let saved = generator.sampling();
    generator.set_sampling(probe.temperature, seed, 0);
    generator.reset();
    let (out, _) = generator.rollout(&probe.cases, &probe.starts, tools, batch)?;
    generator.reset();
    generator.restore_sampling(saved);
    let _ = h;
    Ok(out)
}

/// The natural-error probe's states from one checkpoint's episodes: each episode's first error
/// state, keyed by the episode.
pub fn error_states(eps: &[Episode], tag: &str, max_new_tokens: usize) -> Vec<(String, Value, Start)> {
    eps.iter()
        .enumerate()
        .filter_map(|(i, ep)| {
            grpo::error_state(ep, max_new_tokens)
                .map(|s| (format!("{tag}:{i}:{}", id(&ep.case)), ep.case.clone(), s))
        })
        .collect()
}

/// Mean and 95% interval (normal) of paired differences.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Paired {
    pub n: usize,
    pub mean: f64,
    pub se: f64,
    pub lo: f64,
    pub hi: f64,
}

pub fn paired(a: &[f64], b: &[f64]) -> Paired {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let mean = d.iter().sum::<f64>() / n.max(1) as f64;
    let var = if n > 1 {
        d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let se = (var / n.max(1) as f64).sqrt();
    Paired {
        n,
        mean,
        se,
        lo: mean - 1.96 * se,
        hi: mean + 1.96 * se,
    }
}

/// The mean of several probes' paired differences and its interval (the probes independent).
pub fn mean_of(parts: &[Paired]) -> Paired {
    let k = parts.len() as f64;
    let mean = parts.iter().map(|p| p.mean).sum::<f64>() / k;
    let se = (parts.iter().map(|p| p.se * p.se).sum::<f64>()).sqrt() / k;
    Paired {
        n: parts.iter().map(|p| p.n).sum(),
        mean,
        se,
        lo: mean - 1.96 * se,
        hi: mean + 1.96 * se,
    }
}

pub fn rewards(eps: &[Episode]) -> Vec<f64> {
    eps.iter()
        .map(|ep| score(&ep.case, &ep.calls, &ep.final_text, &ERR0).total)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_interval() {
        let a = [1.0, 0.5, 0.0, 1.0];
        let b = [0.5, 0.5, 0.5, 0.0];
        let p = paired(&a, &b);
        assert!((p.mean - 0.25).abs() < 1e-12);
        // differences 0.5, 0, -0.5, 1: sample variance 0.41667
        assert!((p.se - (0.416_666_666_666_666_7f64 / 4.0).sqrt()).abs() < 1e-12);
        let m = mean_of(&[p, p]);
        assert!((m.mean - 0.25).abs() < 1e-12 && (m.se - p.se / 2f64.sqrt()).abs() < 1e-12);
    }
}
