//! Notebook 10 on anton: 400 steps of GRPO from seed-09-k2, a quarter of each step's episodes
//! continuing the run's own recent error states; the probes at the gate (step 120) and at the
//! end (step 400), 09's checkpoints re-measured beside notebook 10's.
//!
//!     nb10 train [--steps N] [--run NAME] [--probe-limit N]
//!
//! Python runs only the WaterTAP tool workers (07's script, JSON lines).

use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::module::Module;
use burn::tensor::{DType, Device, FloatDType};
use dash_ops::sample::SampleOptions;
use membrane_rl::generate::Generator;
use membrane_rl::grpo::{self, ErrorState, Rng};
use membrane_rl::hf;
use membrane_rl::lora;
use membrane_rl::membrane::batch::episode_rows;
use membrane_rl::membrane::episode::{Episode, Harness, Start};
use membrane_rl::membrane::pyjson::Value;
use membrane_rl::membrane::reward::score;
use membrane_rl::membrane::tools::ToolPool;
use membrane_rl::probe::{self, ERR0, Paired};
use membrane_rl::qwen3_5::{Qwen3_5ForCausalLM, RotaryTable};
use membrane_rl::update;
use serde::Serialize;
use serde_json::json;

const MODEL: &str = "Qwen/Qwen3.5-0.8B";
const NOTEBOOK: &str = "10_grpo_error_states";
const CONTROL_RUN: &str = "09_fresh_questions_runs/ppo-mlx-qwen3.5-9b-l24-n32-lr1e6c-err0-delta-seed09k2-trainall-s0";

#[derive(Debug, Clone, Serialize)]
struct Config {
    notebook: &'static str,
    start_from: String,
    split: &'static str,
    steps: usize,
    groups: usize,
    group_size: usize,
    error_groups: usize,
    error_window: usize,
    error_uses: usize,
    max_new_tokens: usize,
    stop_at_json: bool,
    temperature: f32,
    lr: f64,
    lr_schedule: &'static str,
    clip_eps: f32,
    inner_epochs: usize,
    minibatches: usize,
    adv_norm: &'static str,
    advantage: &'static str,
    max_grad_norm: f64,
    adam_betas: [f64; 2],
    adam_eps: f64,
    weight_decay: f64,
    kl_coef: f64,
    kl: &'static str,
    freeze_turn_ends: bool,
    freeze_answer_structure: bool,
    err_call_credit: bool,
    call_credit: &'static str,
    provenance: bool,
    seed: u64,
    save_every: usize,
    probe_steps: Vec<usize>,
    gate_step: usize,
    gate_upper: f64,
    probe_batch: usize,
    probe_limit: Option<usize>,
    control: Vec<(usize, String)>,
    dtype: &'static str,
    trainer: &'static str,
    generator: &'static str,
    tools: &'static str,
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn host_bytes(device: &Device) -> f64 {
    device
        .memory_pool_usage()
        .map_or(0.0, |u| u.bytes_reserved as f64 / 2f64.powi(30))
}

struct Run {
    dir: PathBuf,
    cfg: Config,
    h: Harness,
    data: PathBuf,
    table: RotaryTable,
    table_ad: RotaryTable,
    inner: Device,
    peft: dash_lora::PeftConfig,
}

fn append(path: &Path, value: &serde_json::Value) {
    let mut f = OpenOptions::new().create(true).append(true).open(path).unwrap();
    writeln!(f, "{value}").unwrap();
}

fn mean(v: impl IntoIterator<Item = f64>) -> f64 {
    let (s, n) = v.into_iter().fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 { 0.0 } else { s / n as f64 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("train") => train(&args[2..]),
        _ => {
            eprintln!("usage: nb10 train [--steps N] [--run NAME] [--probe-limit N]");
            std::process::exit(2);
        }
    }
}

fn train(args: &[String]) {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let repo = repo.canonicalize().unwrap();
    let data = repo.join("experiments/notebooks/smoke_test35");
    let steps: usize = arg(args, "--steps").map_or(400, |s| s.parse().unwrap());
    let probe_limit: Option<usize> = arg(args, "--probe-limit").map(|s| s.parse().unwrap());
    let probe_steps: Vec<usize> = arg(args, "--probe-steps")
        .map_or(vec![120, 400], |s| s.split(',').map(|x| x.parse().unwrap()).collect());
    let name = arg(args, "--run").unwrap_or_else(|| {
        "grpo-rust-qwen3.5-0.8b-g8x4-err2-lr2e6c-err0-seed09k2-trainall-s0".to_owned()
    });
    let dir = data.join(format!("{NOTEBOOK}_runs")).join(&name);
    if dir.join("metrics.jsonl").exists() {
        eprintln!("{} has a run already; give another --run", dir.display());
        std::process::exit(1);
    }
    fs::create_dir_all(&dir).unwrap();
    let seed_dir = data.join("09_fresh_questions_runs/seed-09-k2/epoch1");
    let cfg = Config {
        notebook: NOTEBOOK,
        start_from: "09_fresh_questions_runs/seed-09-k2/epoch1".into(),
        split: "train_all",
        steps,
        groups: 8,
        group_size: 4,
        error_groups: 2,
        error_window: 25,
        error_uses: 2,
        max_new_tokens: 3072,
        stop_at_json: true,
        temperature: 1.0,
        lr: 2e-6,
        lr_schedule: "constant",
        clip_eps: 0.2,
        inner_epochs: 4,
        minibatches: 4,
        adv_norm: "minibatch",
        advantage: "return minus the group's mean, no division by its spread",
        max_grad_norm: 0.5,
        adam_betas: [0.9, 0.999],
        adam_eps: 1e-5,
        weight_decay: 0.0,
        kl_coef: 0.01,
        kl: "k1 to the frozen seed adapter, summed over the policy's tokens, in the return",
        freeze_turn_ends: true,
        freeze_answer_structure: true,
        err_call_credit: false,
        call_credit: "delta",
        provenance: false,
        seed: 0,
        save_every: 20,
        gate_step: probe_steps[0],
        probe_steps,
        gate_upper: -0.05,
        probe_batch: 32,
        probe_limit,
        control: vec![
            (120, format!("{CONTROL_RUN}/adapter-step120")),
            (400, format!("{CONTROL_RUN}/adapter-step400")),
        ],
        dtype: "bf16 base, f32 LoRA factors",
        trainer: "rust: burn 0.22.1 + dashmag (prefix-tree update, exact layer-by-layer backward)",
        generator: "rust: continuous batching (each episode continues when its own tools answer), prefix states kept, sampled on the device",
        tools: "07_error_recovery.py worker (WaterTAP), 8 processes at nice 10",
    };
    fs::write(dir.join("config.json"), serde_json::to_string_pretty(&cfg).unwrap()).unwrap();

    let inner = Device::cuda(0);
    let ad = inner.clone().autodiff();
    let snapshot = hf::snapshot(MODEL).unwrap();
    let (mut model, config) = Qwen3_5ForCausalLM::from_pretrained(&snapshot, FloatDType::BF16, &ad).unwrap();
    let peft = lora::load(&mut model, &seed_dir, lora::POLICY, true, &ad).unwrap();
    lora::load(&mut model, &seed_dir, lora::REFERENCE, false, &ad).unwrap();
    for (step, path) in &cfg.control {
        lora::load(&mut model, &data.join(path), &format!("control{step}"), false, &ad).unwrap();
    }
    let mut model = lora::choose(model, lora::POLICY);
    let h = Harness::new(&snapshot, &data.join("08_prover_critic_tools.json"), cfg.max_new_tokens, cfg.stop_at_json).unwrap();
    let table = RotaryTable::new(&config, 16384, DType::BF16, &inner);
    let table_ad = RotaryTable::new(&config, 16384, DType::BF16, &ad);
    let run = Run { dir, cfg, h, data, table, table_ad, inner, peft };
    let cfg = &run.cfg;
    let h = &run.h;

    let train_cases = probe::load_jsonl(&run.data.join("09_fresh_questions_data/train_all.jsonl")).unwrap();
    assert_eq!(train_cases.len(), 3599);
    let mut tools = ToolPool::niced(
        &run.data.join("../../../.venv-watertap/bin/python"),
        &run.data.join("07_error_recovery.py"),
        8,
        10,
    )
    .unwrap();
    let options = SampleOptions {
        temperature: cfg.temperature,
        seed: cfg.seed,
        ..SampleOptions::default()
    };
    let mut generator = Generator::new(model.valid(), run.table.clone(), h, options);
    let settings = update::Settings {
        lr: cfg.lr,
        clip_eps: cfg.clip_eps,
        inner_epochs: cfg.inner_epochs,
        minibatches: cfg.minibatches,
        max_grad_norm: cfg.max_grad_norm,
        adam_betas: cfg.adam_betas,
        adam_eps: cfg.adam_eps,
        weight_decay: cfg.weight_decay,
        kl_coef: cfg.kl_coef,
        seed: cfg.seed,
    };
    let mut optim = settings.optimizer();
    let suppress = [h.suppress];
    let mut rng = Rng::new(cfg.seed);
    let mut pool: Vec<ErrorState> = Vec::new();
    let metrics = run.dir.join("metrics.jsonl");
    let samples = run.dir.join("samples.jsonl");
    println!("{NOTEBOOK} -> {}", run.dir.display());

    for step in 0..cfg.steps {
        let t0 = Instant::now();
        // The step's groups: up to `error_groups` recent error states, the rest new questions.
        pool.retain(|s| s.step + cfg.error_window > step && s.uses < cfg.error_uses);
        let mut picks: Vec<usize> = Vec::new();
        while picks.len() < cfg.error_groups && picks.len() < pool.len() {
            let i = rng.below(pool.len());
            if !picks.contains(&i) {
                picks.push(i);
            }
        }
        let mut group_cases: Vec<(Value, Start, String)> = Vec::new();
        for &i in &picks {
            pool[i].uses += 1;
            group_cases.push((pool[i].case.clone(), pool[i].start.clone(), pool[i].origin.clone()));
        }
        let n_err = group_cases.len();
        while group_cases.len() < cfg.groups {
            let c = train_cases[rng.below(train_cases.len())].clone();
            group_cases.push((c, Start::default(), String::new()));
        }
        let gs = cfg.group_size;
        let cases: Vec<Value> = group_cases.iter().flat_map(|(c, _, _)| std::iter::repeat_n(c.clone(), gs)).collect();
        let starts: Vec<Start> = group_cases.iter().flat_map(|(_, s, _)| std::iter::repeat_n(s.clone(), gs)).collect();
        let groups: Vec<Vec<usize>> = (0..cfg.groups).map(|g| (g * gs..(g + 1) * gs).collect()).collect();
        let is_err = |e: usize| e / gs < n_err;

        let before = generator.stats.clone();
        generator.set_temperature(cfg.temperature);
        let (eps, _sampled): (Vec<Episode>, _) = generator.rollout(&cases, &starts, &mut tools, cases.len()).unwrap();
        let gen_s = t0.elapsed().as_secs_f64();
        let gstats = generator.stats.clone();
        let mem_gen = host_bytes(&run.inner);
        println!(
            "  step {step}: rollout {gen_s:.0}s, {} calls, {} decode steps ({:.1} ms each), {} prefill tokens, mem {mem_gen:.1} GB",
            gstats.calls - before.calls,
            gstats.decode_steps - before.decode_steps,
            (gstats.decode_seconds - before.decode_seconds) * 1e3 / (gstats.decode_steps - before.decode_steps).max(1) as f64,
            gstats.prefill_tokens - before.prefill_tokens,
        );
        // The kept prefix states belong to the weights the update is about to change.
        generator.reset();
        run.inner.memory_cleanup();

        // Scores and rows, then the update.
        let scores: Vec<_> = eps.iter().map(|ep| score(&ep.case, &ep.calls, &ep.final_text, &ERR0)).collect();
        let prepared: Vec<grpo::Prepared> = eps
            .iter()
            .zip(&scores)
            .map(|(ep, s)| grpo::prepare(ep, &episode_rows(h, ep, s, h.suppress)))
            .collect();
        let seqs: Vec<Vec<u32>> = eps.iter().map(|ep| [ep.prompt_ids.as_slice(), &ep.ids].concat()).collect();
        let totals: Vec<f64> = scores.iter().map(|s| s.total).collect();
        let batch = update::Batch { seqs: &seqs, prepared: &prepared, scores: &totals, groups: &groups };
        let (next, out) = update::step(model, &mut optim, &settings, &run.table, &run.table_ad, &batch, &suppress, step);
        model = next;
        let fwd_s = out.before_seconds;
        println!("  step {step}: old and reference log-probs {fwd_s:.0}s over {} tree tokens (of {})", out.before.tree_tokens, seqs.iter().map(Vec::len).sum::<usize>());
        if step == 0 {
            for m in &out.minibatches {
                println!("  step 0 epoch {} minibatch {:?}: {} tree tokens, {:.1}s, ratio {:.5}, grad norm {:.4}, mem {:.1} GB", m.epoch, m.groups, m.tree_tokens, m.seconds, m.ratio, m.grad_norm, m.reserved_gb);
            }
        }
        generator.set_model(model.valid());
        let update_s = out.update_seconds;
        let update::Before { old, kl, returns, advantages: adv, .. } = &out.before;
        let counts = &out.counts;
        let ratio_first = out.minibatches.first().map(|m| m.ratio);
        let kls: Vec<f64> = out.minibatches.iter().map(|m| m.approx_kl).collect();
        let grad_norms: Vec<f64> = out.minibatches.iter().map(|m| m.grad_norm as f64).collect();
        let last_epoch = &out.last_epoch;
        let mb_stats: Vec<serde_json::Value> = out
            .minibatches
            .iter()
            .map(|m| json!({"epoch": m.epoch, "groups": m.groups, "approx_kl": m.approx_kl, "ratio": m.ratio, "grad_norm": m.grad_norm}))
            .collect();
        let mem_update = host_bytes(&run.inner);
        run.inner.memory_cleanup();

        // This step's new-question episodes leave their first error states.
        let mut made = 0;
        for (e, ep) in eps.iter().enumerate() {
            if is_err(e) {
                continue;
            }
            if let Some(start) = grpo::error_state(ep, cfg.max_new_tokens) {
                made += 1;
                pool.push(ErrorState {
                    case: ep.case.clone(),
                    start,
                    step,
                    uses: 0,
                    origin: format!("step{step}:ep{e}:{}", ep.case.get("id").and_then(Value::as_str).unwrap_or("")),
                });
            }
        }

        // The step's record.
        let total = |f: &dyn Fn(usize) -> bool| mean((0..eps.len()).filter(|&e| f(e)).map(|e| scores[e].total));
        let mask_tokens: usize = old.iter().map(Vec::len).sum();
        let actor_total: usize = counts.iter().sum();
        let entropy = -(0..eps.len())
            .flat_map(|e| batch.actor(e).into_iter().map(move |k| (e, k)))
            .map(|(e, k)| old[e][k] as f64)
            .sum::<f64>()
            / actor_total.max(1) as f64;
        let rec = json!({
            "step": step,
            "reward_mean": total(&|_| true),
            "reward_new": total(&|e| !is_err(e)),
            "reward_error_states": if n_err > 0 { json!(total(&is_err)) } else { json!(null) },
            "error_groups": n_err,
            "error_pool": pool.len(),
            "error_states_made": made,
            "decision_correct": mean(scores.iter().map(|s| s.decision_correct as u8 as f64)),
            "evidence": mean(scores.iter().map(|s| s.evidence as u8 as f64)),
            "calls_mean": mean(eps.iter().map(|ep| ep.calls.len() as f64)),
            "error_calls_mean": mean(eps.iter().map(|ep| ep.calls.iter().filter(|c| membrane_rl::membrane::tools::result_is_error(c.result.as_deref())).count() as f64)),
            "truncated": mean(eps.iter().map(|ep| ep.truncated as u8 as f64)),
            "answer_closed": mean(eps.iter().map(|ep| ep.answer_closed as u8 as f64)),
            "policy_tokens_mean": mean(eps.iter().map(|ep| ep.policy_tokens() as f64)),
            "episode_tokens_mean": mean(seqs.iter().map(|s| s.len() as f64)),
            "kl_mean": kl.iter().sum::<f64>() / mask_tokens.max(1) as f64,
            "kl_episode_mean": mean(kl.iter().copied()),
            "entropy_sampled": entropy,
            "adv_abs_mean": mean(adv.iter().map(|a| a.abs())),
            "groups_flat": groups.iter().filter(|g| g.iter().all(|&e| adv[e].abs() < 1e-12)).count(),
            "ratio_first": ratio_first,
            "approx_kl_last": kls.last(),
            "approx_kl_max": kls.iter().copied().fold(f64::NAN, f64::max),
            "clip_frac": mean(last_epoch.iter().map(|s| s.clip_frac)),
            "actor_loss": mean(last_epoch.iter().map(|s| s.loss)),
            "grad_norm_mean": mean(grad_norms.iter().copied()),
            "grad_norm_max": grad_norms.iter().copied().fold(f64::NAN, f64::max),
            "updates": grad_norms.len(),
            "lr": cfg.lr,
            "tree_tokens": out.before.tree_tokens,
            "sequence_tokens": seqs.iter().map(Vec::len).sum::<usize>(),
            "decode_steps": gstats.decode_steps - before.decode_steps,
            "prefill_tokens": gstats.prefill_tokens - before.prefill_tokens,
            "gen_seconds": gen_s,
            "fwd_seconds": fwd_s,
            "update_seconds": update_s,
            "step_seconds": t0.elapsed().as_secs_f64(),
            "solves": tools.solved,
            "mem_gen_gb": mem_gen,
            "mem_update_gb": mem_update,
            "minibatches": mb_stats,
        });
        append(&metrics, &rec);
        for (e, ep) in eps.iter().enumerate() {
            let mut r = probe::record(&format!("{step}:{e}"), ep, h);
            r["step"] = json!(step);
            r["group"] = json!(e / gs);
            r["start"] = json!(if is_err(e) { group_cases[e / gs].2.clone() } else { String::new() });
            r["kl"] = json!(kl[e]);
            r["return"] = json!(returns[e]);
            r["advantage"] = json!(adv[e]);
            append(&samples, &r);
        }
        println!(
            "step {step:>3} reward {:.3} (new {:.3}, err {}) kl {:.4} ratio1 {:.4} akl {:.5} clip {:.4} gnorm {:.3} | gen {:.0}s fwd {:.0}s upd {:.0}s | mem {:.1}/{:.1} GB",
            rec["reward_mean"].as_f64().unwrap(),
            rec["reward_new"].as_f64().unwrap(),
            rec["reward_error_states"],
            rec["kl_mean"].as_f64().unwrap(),
            ratio_first.unwrap_or(f64::NAN),
            kls.last().copied().unwrap_or(f64::NAN),
            rec["clip_frac"].as_f64().unwrap(),
            rec["grad_norm_mean"].as_f64().unwrap(),
            gen_s,
            fwd_s,
            update_s,
            mem_gen,
            mem_update,
        );

        let done = step + 1;
        if done % cfg.save_every == 0 || cfg.probe_steps.contains(&done) || done == cfg.steps {
            lora::save(&model, lora::POLICY, &run.peft, &run.dir.join(format!("adapter-step{done}"))).unwrap();
        }
        if cfg.probe_steps.contains(&done) {
            let summary = probes(&run, &mut generator, &mut tools, &model, done);
            generator.set_model(model.valid());
            run.inner.memory_cleanup();
            if done == cfg.gate_step {
                let hi = summary["mean4"]["hi"].as_f64().unwrap();
                if hi < cfg.gate_upper {
                    fs::write(
                        run.dir.join("ending.json"),
                        serde_json::to_string_pretty(&json!({"stopped_at": done, "why": "gate", "mean4": summary["mean4"]})).unwrap(),
                    )
                    .unwrap();
                    println!("gate: the 4-probe mean's upper bound {hi:.4} < {}: stopping", cfg.gate_upper);
                    return;
                }
                println!("gate passed: 4-probe mean {:.4} [{:.4}, {hi:.4}]", summary["mean4"]["mean"], summary["mean4"]["lo"]);
            }
        }
    }
    lora::save(&model, lora::POLICY, &run.peft, &run.dir.join("adapter")).unwrap();
    fs::write(
        run.dir.join("ending.json"),
        serde_json::to_string_pretty(&json!({"stopped_at": cfg.steps, "why": "steps"})).unwrap(),
    )
    .unwrap();
}

/// The probes at `step`: notebook 10's checkpoint and 09's at the same step, paired.
fn probes(
    run: &Run,
    generator: &mut Generator,
    tools: &mut ToolPool,
    model: &Qwen3_5ForCausalLM,
    step: usize,
) -> serde_json::Value {
    let t0 = Instant::now();
    let h = &run.h;
    let out = run.dir.join(format!("probes-step{step}"));
    fs::create_dir_all(&out).unwrap();
    let mut probes = probe::held_out(h, &run.data).unwrap();
    if let Some(n) = run.cfg.probe_limit {
        for p in probes.iter_mut() {
            p.cases.truncate(n);
            p.starts.truncate(n);
            p.keys.truncate(n);
        }
    }
    // 09's checkpoint at the same step (the first one for a shortened run's test).
    let control = run
        .cfg
        .control
        .iter()
        .find(|(s, _)| *s == step)
        .unwrap_or(&run.cfg.control[0])
        .0;
    let labels = [("nb10", lora::POLICY.to_owned()), ("nb09", format!("control{control}"))];
    let mut rewards: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut natural_eps: Vec<Vec<Episode>> = Vec::new();
    for (label, adapter) in &labels {
        generator.set_model(lora::choose(model.valid(), adapter));
        let mut per_probe = Vec::new();
        for (k, p) in probes.iter().enumerate() {
            let eps = probe::run(generator, h, tools, p, run.cfg.probe_batch, 1000 + k as u64).unwrap();
            let path = out.join(format!("{label}-{}.jsonl", p.name));
            let _ = fs::remove_file(&path);
            for (key, ep) in p.keys.iter().zip(&eps) {
                append(&path, &probe::record(key, ep, h));
            }
            let r = probe::rewards(&eps);
            println!("probe step {step} {label} {:<14} {:.4} over {}", p.name, mean(r.iter().copied()), r.len());
            per_probe.push(r);
        }
        rewards.push(per_probe);
        if step == *run.cfg.probe_steps.last().unwrap() {
            // The natural-error probe's sampling: dev at T=1, twice.
            let dev = probe::load_jsonl(&run.data.join("06_direction_and_compare_data/dev.jsonl")).unwrap();
            let dev: Vec<Value> = match run.cfg.probe_limit {
                Some(n) => dev.into_iter().take(n).collect(),
                None => dev,
            };
            let cases: Vec<Value> = dev.iter().chain(dev.iter()).cloned().collect();
            let sample = probe::Probe {
                name: "natural_sample",
                keys: (0..cases.len()).map(|i| i.to_string()).collect(),
                starts: vec![Start::default(); cases.len()],
                cases,
                temperature: 1.0,
            };
            let seed = if *label == "nb10" { 2000 } else { 3000 };
            natural_eps.push(probe::run(generator, h, tools, &sample, run.cfg.probe_batch, seed).unwrap());
        }
    }
    let mut summary = serde_json::Map::new();
    let mut parts: Vec<Paired> = Vec::new();
    for (k, p) in probes.iter().enumerate() {
        let d = probe::paired(&rewards[0][k], &rewards[1][k]);
        summary.insert(
            p.name.to_owned(),
            json!({"nb10": mean(rewards[0][k].iter().copied()), "nb09": mean(rewards[1][k].iter().copied()), "diff": d}),
        );
        parts.push(d);
    }
    summary.insert("mean4".into(), json!(probe::mean_of(&parts)));
    if natural_eps.len() == 2 {
        let mut states = probe::error_states(&natural_eps[0], "nb10", run.cfg.max_new_tokens);
        states.extend(probe::error_states(&natural_eps[1], "nb09", run.cfg.max_new_tokens));
        let natural = probe::Probe {
            name: "natural_errors",
            keys: states.iter().map(|s| s.0.clone()).collect(),
            cases: states.iter().map(|s| s.1.clone()).collect(),
            starts: states.iter().map(|s| s.2.clone()).collect(),
            temperature: 0.0,
        };
        let mut r = Vec::new();
        for (label, adapter) in &labels {
            generator.set_model(lora::choose(model.valid(), adapter));
            let eps = probe::run(generator, h, tools, &natural, run.cfg.probe_batch, 4000).unwrap();
            let path = out.join(format!("{label}-natural_errors.jsonl"));
            let _ = fs::remove_file(&path);
            for (key, ep) in natural.keys.iter().zip(&eps) {
                append(&path, &probe::record(key, ep, h));
            }
            r.push(probe::rewards(&eps));
        }
        let d = probe::paired(&r[0], &r[1]);
        println!("natural errors: {} states, nb10 {:.4} nb09 {:.4}, diff {:.4} [{:.4}, {:.4}]", d.n, mean(r[0].iter().copied()), mean(r[1].iter().copied()), d.mean, d.lo, d.hi);
        summary.insert(
            "natural_errors".into(),
            json!({"states": d.n, "from_nb10": natural_eps[0].len(), "from_nb09": natural_eps[1].len(), "nb10": mean(r[0].iter().copied()), "nb09": mean(r[1].iter().copied()), "diff": d}),
        );
    }
    summary.insert("seconds".into(), json!(t0.elapsed().as_secs_f64()));
    let summary = serde_json::Value::Object(summary);
    fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("probes at step {step}: {}", summary["mean4"]);
    summary
}
