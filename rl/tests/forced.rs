//! The forced probe's start against notebook 07's: each dev question of 07's recovery file
//! started from its slipped turns, the recorded answer written next — the same tokens, mask,
//! turns, calls and err0 reward as 07's `rollout` with its `prefill`.

use std::path::PathBuf;

use membrane_rl::hf;
use membrane_rl::membrane::episode::{Harness, Policy, Start, rollout_from};
use membrane_rl::membrane::pyjson::{self, Value};
use membrane_rl::membrane::reward::score;
use membrane_rl::membrane::tools::Tools;
use membrane_rl::probe::{self, ERR0};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_list().unwrap().iter().map(|t| t.as_f64().unwrap() as u32).collect()
}

/// Writes each row's recorded answer as its one turn.
struct Replay(Vec<Vec<u32>>);

impl Policy for Replay {
    fn generate(&mut self, inputs: &[Vec<u32>], budgets: &[usize]) -> Vec<Vec<u32>> {
        assert_eq!(inputs.len(), self.0.len(), "one turn per episode");
        self.0.iter().zip(budgets).map(|(w, &b)| w[..w.len().min(b)].to_vec()).collect()
    }
}

/// No tool runs: every call of a forced episode is in its prefilled turns.
struct NoTools;

impl Tools for NoTools {
    fn call_many(&mut self, calls: &[(String, Value)]) -> Vec<String> {
        assert!(calls.is_empty(), "a forced episode's answer makes no calls");
        Vec::new()
    }
}

#[test]
#[ignore = "needs the checkpoint in the Hugging Face cache and MEMBRANE_FORCED (export_forced_reference.py)"]
fn forced_starts_match_notebook_07() {
    let path = std::env::var_os("MEMBRANE_FORCED").expect("set MEMBRANE_FORCED");
    let recorded = probe::load_jsonl(&PathBuf::from(path)).unwrap();
    let data = repo().join("experiments/notebooks/smoke_test35");
    let h = Harness::new(
        &hf::snapshot("Qwen/Qwen3.5-0.8B").unwrap(),
        &data.join("07_error_recovery_tools.json"),
        3072,
        true,
    )
    .unwrap();
    let probes = probe::held_out(&h, &data).unwrap();
    let forced = probes.iter().find(|p| p.name == "dev_forced").unwrap();
    assert_eq!(forced.cases.len(), recorded.len());
    let written: Vec<Vec<u32>> = recorded.iter().map(|r| ids(r.get("written").unwrap())).collect();
    let starts: Vec<Start> = forced.starts.clone();
    let eps = rollout_from(&h, &forced.cases, &starts, &mut NoTools, &mut Replay(written)).unwrap();
    for (ep, rec) in eps.iter().zip(&recorded) {
        let want = rec.get("episode").unwrap();
        let name = ep.case.get("id").unwrap().as_str().unwrap();
        assert_eq!(
            ep.case.get("id"),
            rec.get("case").unwrap().get("id"),
            "the probe's order is 07's"
        );
        assert_eq!(ep.prompt_ids, ids(want.get("prompt_ids").unwrap()), "{name}: prompt");
        assert_eq!(ep.ids, ids(want.get("ids").unwrap()), "{name}: ids");
        let policy: Vec<bool> = ids(want.get("policy").unwrap()).iter().map(|&p| p == 1).collect();
        assert_eq!(ep.policy, policy, "{name}: policy mask");
        let starts: Vec<usize> = ids(want.get("turn_starts").unwrap()).iter().map(|&s| s as usize).collect();
        assert_eq!(ep.turn_starts, starts, "{name}: turn starts");
        let turns: Vec<&str> = want.get("turns").unwrap().as_list().unwrap().iter().map(|t| t.as_str().unwrap()).collect();
        assert_eq!(ep.turns, turns, "{name}: turns");
        let calls = want.get("calls").unwrap().as_list().unwrap();
        assert_eq!(ep.calls.len(), calls.len(), "{name}: calls");
        for (c, w) in ep.calls.iter().zip(calls) {
            assert_eq!(c.name.as_deref(), w.get("name").and_then(Value::as_str), "{name}: call name");
            assert_eq!(
                c.arguments.as_ref().map(|a| pyjson::dumps(a, true)),
                w.get("arguments").filter(|a| !matches!(a, Value::Null)).map(|a| pyjson::dumps(a, true)),
                "{name}: call arguments"
            );
            assert_eq!(c.result.as_deref(), w.get("result").and_then(Value::as_str), "{name}: result");
            assert_eq!(c.turn.map(|t| t as f64), w.get("turn").and_then(Value::as_f64), "{name}: call turn");
        }
        assert_eq!(ep.final_text, want.get("final_text").unwrap().as_str().unwrap(), "{name}: final text");
        let s = score(&ep.case, &ep.calls, &ep.final_text, &ERR0);
        assert_eq!(s.total, rec.get("score").unwrap().as_f64().unwrap(), "{name}: err0 reward");
    }
    println!("{} forced episodes match 07 exactly", eps.len());
}
