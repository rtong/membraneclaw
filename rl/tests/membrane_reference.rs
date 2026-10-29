//! The harness held to notebook 07's own Python, on episodes recorded in a PPO run and replayed
//! through 07's rollout by `rl/tools/export_membrane_reference.py`:
//!
//!     MEMBRANE_REFERENCE=<its out.jsonl> cargo test --release --test membrane_reference -- --ignored
//!
//! Each episode is replayed here the same way — the same turns written, the tools answering as
//! they answered there — and the episode, its score under every reward setting and its rows must
//! come out as 07's did.

use std::collections::HashMap;
use std::path::PathBuf;

use membrane_rl::hf;
use membrane_rl::membrane::batch::episode_rows;
use membrane_rl::membrane::episode::{Episode, Harness, Policy, char_offset, rollout};
use membrane_rl::membrane::pyjson::{self, Value};
use membrane_rl::membrane::reward::{self, Anchor, CallCredit, Score, Settings, score};
use membrane_rl::membrane::tools::{self, Call, Tools};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn tools_json() -> PathBuf {
    repo().join("experiments/notebooks/smoke_test35/07_error_recovery_tools.json")
}

fn settings(name: &str) -> Settings {
    let mut parts = name.split('-');
    let err = parts.next().unwrap() == "err1";
    let call_credit = if parts.next().unwrap() == "delta" {
        CallCredit::Delta
    } else {
        CallCredit::Call
    };
    let provenance = parts.next().unwrap() == "prov1";
    Settings {
        err_call_credit: err,
        call_credit,
        provenance,
    }
}

/// Writes the recorded turns again, as 07's replay did: a turn the budget cut off is written on
/// to the budget; past the recorded turns, an empty turn.
struct Replay {
    turns: Vec<Value>,
    next: usize,
    im_end: u32,
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_list()
        .unwrap()
        .iter()
        .map(|t| t.as_f64().unwrap() as u32)
        .collect()
}

impl Policy for Replay {
    fn generate(&mut self, _inputs: &[Vec<u32>], budgets: &[usize]) -> Vec<Vec<u32>> {
        let b = budgets[0];
        let turn = self.turns.get(self.next);
        self.next += 1;
        let row = match turn {
            None => vec![self.im_end],
            Some(t) => match t.get("fill") {
                Some(fill) => {
                    let fill = ids(fill);
                    let fill = if fill.is_empty() {
                        vec![self.im_end]
                    } else {
                        fill
                    };
                    fill.iter().copied().cycle().take(b).collect()
                }
                None => ids(t),
            },
        };
        vec![row]
    }
}

/// Answers every call as the run's tools answered it.
struct Recorded(HashMap<String, String>);

fn key(name: &str, args: &Value) -> String {
    pyjson::dumps(
        &Value::List(vec![Value::Str(name.into()), args.clone()]),
        true,
    )
}

impl Tools for Recorded {
    fn call_many(&mut self, calls: &[(String, Value)]) -> Vec<String> {
        calls
            .iter()
            .map(|(n, a)| {
                self.0
                    .get(&key(n, a))
                    .unwrap_or_else(|| panic!("no recorded reply to {n} {a:?}"))
                    .clone()
            })
            .collect()
    }
}

fn recorded_calls(ep: &Value) -> Vec<Call> {
    ep.get("calls")
        .unwrap()
        .as_list()
        .unwrap()
        .iter()
        .map(|c| Call {
            name: c.get("name").and_then(Value::as_str).map(str::to_owned),
            arguments: c.get("arguments").filter(|a| !a.is_null()).cloned(),
            malformed: c.get("malformed").and_then(Value::as_bool).unwrap(),
            result: c.get("result").and_then(Value::as_str).map(str::to_owned),
            turn: c.get("turn").and_then(Value::as_f64).map(|t| t as usize),
        })
        .collect()
}

fn check_episode(id: &str, got: &Episode, want: &Value) {
    let flags = |v: &Value| {
        v.as_list()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() != 0.0)
            .collect::<Vec<bool>>()
    };
    let usizes = |v: &Value| {
        v.as_list()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as usize)
            .collect::<Vec<usize>>()
    };
    assert_eq!(
        got.prompt_ids,
        ids(want.get("prompt_ids").unwrap()),
        "{id}: prompt"
    );
    assert_eq!(got.ids, ids(want.get("ids").unwrap()), "{id}: ids");
    assert_eq!(
        got.policy,
        flags(want.get("policy").unwrap()),
        "{id}: policy"
    );
    assert_eq!(
        got.turn_starts,
        usizes(want.get("turn_starts").unwrap()),
        "{id}: turn starts"
    );
    let turns: Vec<&str> = want
        .get("turns")
        .unwrap()
        .as_list()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    assert_eq!(got.turns, turns, "{id}: turns");
    let calls = recorded_calls(want);
    assert_eq!(got.calls.len(), calls.len(), "{id}: calls");
    for (k, (a, b)) in got.calls.iter().zip(&calls).enumerate() {
        assert_eq!(
            (&a.name, a.malformed, &a.result, a.turn),
            (&b.name, b.malformed, &b.result, b.turn),
            "{id}: call {k}"
        );
        let dump = |c: &Call| c.arguments.as_ref().map(|v| pyjson::dumps(v, false));
        assert_eq!(dump(a), dump(b), "{id}: call {k} arguments");
    }
    assert_eq!(
        got.final_text,
        want.get("final_text").unwrap().as_str().unwrap(),
        "{id}: final text"
    );
    assert_eq!(
        got.truncated,
        want.get("truncated").unwrap().as_bool().unwrap(),
        "{id}: truncated"
    );
    assert_eq!(
        got.unanswered,
        want.get("unanswered").unwrap().as_f64().unwrap() as usize,
        "{id}: unanswered"
    );
    assert_eq!(
        got.answer_closed,
        want.get("answer_closed").unwrap().as_bool().unwrap(),
        "{id}: answer closed"
    );
}

fn check_score(id: &str, got: &Score, want: &Value) {
    assert_eq!(
        got.total,
        want.get("total").unwrap().as_f64().unwrap(),
        "{id}: total"
    );
    let comps = want.get("components").unwrap().as_dict().unwrap();
    let names: Vec<&str> = comps.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        got.components.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        names,
        "{id}: components"
    );
    for ((k, v), (_, w)) in got.components.iter().zip(comps) {
        assert_eq!(*v, w.as_f64().unwrap(), "{id}: component {k}");
    }
    assert_eq!(
        got.evidence,
        want.get("evidence").unwrap().as_bool().unwrap(),
        "{id}: evidence"
    );
    let covered: Vec<&str> = want
        .get("covered")
        .unwrap()
        .as_list()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap())
        .collect();
    assert_eq!(got.covered, covered, "{id}: covered");
    for (field, value) in [
        ("n_calls", got.n_calls),
        ("n_converts", got.n_converts),
        ("n_compares", got.n_compares),
    ] {
        assert_eq!(
            value,
            want.get(field).unwrap().as_f64().unwrap() as usize,
            "{id}: {field}"
        );
    }
    assert_eq!(
        got.decision_correct,
        want.get("decision_correct").unwrap().as_bool().unwrap(),
        "{id}: decision"
    );
    let credits = want.get("credits").unwrap().as_list().unwrap();
    assert_eq!(got.credits.len(), credits.len(), "{id}: credits");
    for ((anchor, amount), w) in got.credits.iter().zip(credits) {
        let w = w.as_list().unwrap();
        let a = w[0].as_list().unwrap();
        let want_anchor = match a[0].as_str().unwrap() {
            "call" => Anchor::Call(a[1].as_f64().unwrap() as usize),
            "decision" => Anchor::Decision,
            "violations" => Anchor::Violations,
            "value" => Anchor::Value(a[1].as_str().unwrap().to_owned()),
            "close" => Anchor::Close,
            other => panic!("anchor {other}"),
        };
        assert_eq!(anchor, &want_anchor, "{id}: credit anchor");
        assert_eq!(
            *amount,
            w[1].as_f64().unwrap(),
            "{id}: credit at {anchor:?}"
        );
    }
}

#[test]
#[ignore = "needs MEMBRANE_REFERENCE and the checkpoint in the Hugging Face cache"]
fn harness_reward_and_rows_match_notebook_07() {
    let path = std::env::var_os("MEMBRANE_REFERENCE")
        .expect("set MEMBRANE_REFERENCE to export_membrane_reference.py's output");
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header = pyjson::loads(lines.next().unwrap()).unwrap();
    let max_new = header.get("max_new_tokens").unwrap().as_f64().unwrap() as usize;
    let stop = header.get("stop_at_json").unwrap().as_bool().unwrap();
    let h = Harness::new(
        &hf::snapshot("Qwen/Qwen3.5-0.8B").unwrap(),
        &tools_json(),
        max_new,
        stop,
    )
    .unwrap();
    let special: Vec<u32> = ids(header.get("all_special_ids").unwrap());
    assert_eq!(
        h.special_ids.iter().copied().collect::<Vec<_>>(),
        special,
        "all_special_ids"
    );
    assert_eq!(
        h.prefix,
        header.get("generation_prefix").unwrap().as_str().unwrap(),
        "generation prefix"
    );
    let pad = header.get("pad_token_id").unwrap().as_f64().unwrap() as u32;
    let names: Vec<String> = header
        .get("settings")
        .unwrap()
        .as_list()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_owned())
        .collect();
    for probe in header.get("probes").unwrap().as_list().unwrap() {
        let text = probe.get("text").unwrap().as_str().unwrap();
        let calls = tools::parse_calls(text, &h.decls);
        let want = probe.get("calls").unwrap().as_list().unwrap();
        assert_eq!(calls.len(), want.len(), "{text:?}: calls");
        for (c, w) in calls.iter().zip(want) {
            assert_eq!(
                c.name.as_deref(),
                w.get("name").and_then(Value::as_str),
                "{text:?}: name"
            );
            assert_eq!(
                c.malformed,
                w.get("malformed").and_then(Value::as_bool).unwrap_or(false),
                "{text:?}: malformed"
            );
            let dump =
                |v: Option<&Value>| v.filter(|v| !v.is_null()).map(|v| pyjson::dumps(v, false));
            assert_eq!(
                dump(c.arguments.as_ref()),
                dump(w.get("arguments")),
                "{text:?}: arguments"
            );
        }
        let answer = probe.get("answer").unwrap().as_list().unwrap();
        match reward::extract_answer(text) {
            None => assert!(answer[0].is_null(), "{text:?}: answer"),
            Some((obj, (s, e), clean)) => {
                assert_eq!(
                    pyjson::dumps(&obj, false),
                    pyjson::dumps(&answer[0], false),
                    "{text:?}: answer"
                );
                let span = answer[1].as_list().unwrap();
                let chars = (char_offset(text, s), char_offset(text, e));
                assert_eq!(
                    chars,
                    (
                        span[0].as_f64().unwrap() as usize,
                        span[1].as_f64().unwrap() as usize
                    ),
                    "{text:?}: span"
                );
                assert_eq!(clean, answer[2].as_bool().unwrap(), "{text:?}: clean end");
            }
        }
    }

    let mut episodes = 0;
    for line in lines {
        let rec = pyjson::loads(line).unwrap();
        let case = rec.get("case").unwrap().clone();
        let id = case.get("id").unwrap().as_str().unwrap().to_owned();
        let want = rec.get("episode").unwrap();
        let mut replies = HashMap::new();
        for c in recorded_calls(want) {
            let harness_error = c.malformed
                || [
                    tools::CALL_BUDGET_ERROR,
                    tools::CONVERT_BUDGET_ERROR,
                    tools::COMPARE_BUDGET_ERROR,
                ]
                .contains(&c.result.as_deref().unwrap_or(""));
            if !harness_error {
                replies.insert(
                    key(c.name.as_deref().unwrap(), c.arguments.as_ref().unwrap()),
                    c.result.clone().unwrap(),
                );
            }
        }
        let mut policy = Replay {
            turns: rec.get("written").unwrap().as_list().unwrap().to_vec(),
            next: 0,
            im_end: h.im_end,
        };
        let got = rollout(
            &h,
            std::slice::from_ref(&case),
            &mut Recorded(replies),
            &mut policy,
        )
        .unwrap()
        .remove(0);
        check_episode(&id, &got, want);

        let batch = rec.get("batch").unwrap();
        for name in &names {
            let s = score(&got.case, &got.calls, &got.final_text, &settings(name));
            check_score(
                &format!("{id} {name}"),
                &s,
                rec.get("scores").unwrap().get(name).unwrap(),
            );
            let rows = episode_rows(&h, &got, &s, pad);
            let f32s = |v: &Value| {
                v.as_list()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect::<Vec<f64>>()
            };
            let as_f32 = |v: &[f64]| v.iter().map(|&x| x as f32 as f64).collect::<Vec<f64>>();
            assert_eq!(
                as_f32(&rows.rewards),
                f32s(rec.get("rewards").unwrap().get(name).unwrap()),
                "{id} {name}: rewards"
            );
            if name == &names[0] {
                let usizes = |v: &Value| {
                    v.as_list()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap() as usize)
                        .collect::<Vec<usize>>()
                };
                assert_eq!(
                    rows.index,
                    usizes(batch.get("index").unwrap()),
                    "{id}: index"
                );
                assert_eq!(rows.ids, ids(batch.get("ids").unwrap()), "{id}: ids");
                assert_eq!(rows.turn, usizes(batch.get("turn").unwrap()), "{id}: turn");
                for (field, got) in [
                    ("mask", &rows.mask),
                    ("answer", &rows.answer),
                    ("turn_end", &rows.turn_end),
                    ("content", &rows.content),
                ] {
                    assert_eq!(
                        as_f32(got),
                        f32s(batch.get(field).unwrap()),
                        "{id}: {field}"
                    );
                }
            }
        }
        episodes += 1;
    }
    println!(
        "{episodes} episodes: episode, score under {} settings and rows match; probes match",
        names.len()
    );
}

/// The workers themselves: notebook 07's tool host under `.venv-watertap`, sent the calls of the
/// recorded episodes, answers each as the run's tools did.
#[test]
#[ignore = "needs MEMBRANE_REFERENCE and .venv-watertap"]
fn tool_workers_answer_as_the_run_did() {
    use membrane_rl::membrane::tools::ToolPool;
    let path = std::env::var_os("MEMBRANE_REFERENCE").expect("set MEMBRANE_REFERENCE");
    let text = std::fs::read_to_string(path).unwrap();
    let mut jobs: Vec<(String, Value)> = Vec::new();
    let mut want: Vec<String> = Vec::new();
    for line in text.lines().skip(1).take(12) {
        let rec = pyjson::loads(line).unwrap();
        for c in recorded_calls(rec.get("episode").unwrap()) {
            let harness_error = c.malformed
                || [
                    tools::CALL_BUDGET_ERROR,
                    tools::CONVERT_BUDGET_ERROR,
                    tools::COMPARE_BUDGET_ERROR,
                ]
                .contains(&c.result.as_deref().unwrap_or(""));
            if !harness_error {
                jobs.push((c.name.clone().unwrap(), c.arguments.clone().unwrap()));
                want.push(c.result.clone().unwrap());
            }
        }
    }
    let mut pool = ToolPool::notebook_07(&repo(), 4).unwrap();
    let got = pool.call_many(&jobs);
    for ((job, g), w) in jobs.iter().zip(&got).zip(&want) {
        assert_eq!(g, w, "{} {}", job.0, pyjson::dumps(&job.1, false));
    }
    println!(
        "{} calls ({} distinct, solved by the workers) answered as recorded",
        jobs.len(),
        pool.solved
    );
}
