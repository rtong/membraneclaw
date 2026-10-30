//! A step's batch, fixed for the update's checks: rolls out `groups` groups of 4 with one of nb10's
//! checkpoints as nb10 does, scores them, and takes the old and the reference log-probs, the
//! returns and the advantages as nb10's update does. For each episode it writes its sequence,
//! every policy token's position and target, which of them the actor's loss takes, and its score
//! (the whole step, tests/step.rs and tools/export_step_reference.py); and the actor tokens'
//! positions, targets and old log-probs, the advantage z-scored over the batch as one minibatch
//! and its loss weight (one minibatch's gradient, tests/grad.rs and
//! tools/export_grad_reference.py). With them the run's update settings (its config.json), the
//! step the checkpoint continues at and each epoch's order of the groups.
//!
//! `grad_batch <adapter-stepN of an nb10 run, under smoke_test35> <out.json> [groups] [seed]`

use std::path::PathBuf;

use burn::module::Module;
use burn::tensor::{DType, Device, FloatDType};
use dash_ops::sample::SampleOptions;
use membrane_rl::generate::Generator;
use membrane_rl::grpo::{self, Rng};
use membrane_rl::hf;
use membrane_rl::lora;
use membrane_rl::membrane::batch::episode_rows;
use membrane_rl::membrane::episode::{Harness, Start};
use membrane_rl::membrane::reward::score;
use membrane_rl::membrane::tools::ToolPool;
use membrane_rl::probe::{self, ERR0};
use membrane_rl::qwen3_5::{Qwen3_5ForCausalLM, RotaryTable};
use membrane_rl::update::{self, Settings};
use serde_json::json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap();
    let data = repo.join("experiments/notebooks/smoke_test35");
    let adapter = data.join(&args[1]);
    let out = PathBuf::from(&args[2]);
    let groups: usize = args.get(3).map_or(2, |s| s.parse().unwrap());
    let seed: u64 = args.get(4).map_or(11, |s| s.parse().unwrap());
    let gs = 4;

    let device = Device::cuda(0);
    let snapshot = hf::snapshot("Qwen/Qwen3.5-0.8B").unwrap();
    let (mut model, config) = Qwen3_5ForCausalLM::from_pretrained(&snapshot, FloatDType::BF16, &device).unwrap();
    lora::load(&mut model, &adapter, lora::POLICY, false, &device).unwrap();
    let seed_dir = data.join("09_fresh_questions_runs/seed-09-k2/epoch1");
    lora::load(&mut model, &seed_dir, lora::REFERENCE, false, &device).unwrap();
    let model = lora::choose(model, lora::POLICY);
    let h = Harness::new(&snapshot, &data.join("08_prover_critic_tools.json"), 3072, true).unwrap();
    let table = RotaryTable::new(&config, 16384, DType::BF16, &device);
    let train = probe::load_jsonl(&data.join("09_fresh_questions_data/train_all.jsonl")).unwrap();
    let mut tools = ToolPool::niced(
        &data.join("../../../.venv-watertap/bin/python"),
        &data.join("07_error_recovery.py"),
        8,
        10,
    )
    .unwrap();
    let options = SampleOptions { temperature: 1.0, seed, ..SampleOptions::default() };
    let mut generator = Generator::new(model.valid(), table.clone(), &h, options);
    let mut rng = Rng::new(seed);
    let picked: Vec<_> = (0..groups).map(|_| train[rng.below(train.len())].clone()).collect();
    let cases: Vec<_> = picked.iter().flat_map(|c| std::iter::repeat_n(c.clone(), gs)).collect();
    let starts = vec![Start::default(); cases.len()];
    let (eps, _) = generator.rollout(&cases, &starts, &mut tools, cases.len()).unwrap();
    drop(generator);
    device.memory_cleanup();

    // As nb10's step: scores, rows, the old and reference log-probs, the returns.
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(adapter.join("../config.json")).unwrap()).unwrap();
    let settings: Settings = serde_json::from_value(config).unwrap();
    let step: usize = args[1].rsplit("adapter-step").next().unwrap().parse().unwrap();
    let scores: Vec<_> = eps.iter().map(|ep| score(&ep.case, &ep.calls, &ep.final_text, &ERR0)).collect();
    let prepared: Vec<grpo::Prepared> = eps
        .iter()
        .zip(&scores)
        .map(|(ep, s)| grpo::prepare(ep, &episode_rows(&h, ep, s, h.suppress)))
        .collect();
    let seqs: Vec<Vec<u32>> = eps.iter().map(|ep| [ep.prompt_ids.as_slice(), &ep.ids].concat()).collect();
    let totals: Vec<f64> = scores.iter().map(|s| s.total).collect();
    let group_ix: Vec<Vec<usize>> = (0..groups).map(|g| (g * gs..(g + 1) * gs).collect()).collect();
    let batch = update::Batch { seqs: &seqs, prepared: &prepared, scores: &totals, groups: &group_ix };
    let limit = batch.limit();
    let start = update::before(&model.valid(), &table, &batch, &[h.suppress], settings.kl_coef);
    let (old, kl, adv) = (&start.old, &start.kl, &start.advantages);

    // One minibatch of every group: the z-score over all actor tokens, each episode weighted
    // 1 / (its actor tokens x episodes).
    let actor: Vec<Vec<usize>> = prepared
        .iter()
        .map(|p| (0..p.actor.len()).filter(|&k| p.actor[k]).collect())
        .collect();
    let counts: Vec<usize> = actor.iter().map(Vec::len).collect();
    let z = grpo::normalized(&adv, &counts);
    let episodes: Vec<_> = (0..eps.len())
        .map(|e| {
            let ks = &actor[e];
            json!({
                "seq": seqs[e],
                "reads_at": prepared[e].reads.at,
                "reads_targets": prepared[e].reads.targets,
                "actor": prepared[e].actor,
                "score": totals[e],
                "limit": limit[e],
                "at": ks.iter().map(|&k| prepared[e].reads.at[k]).collect::<Vec<_>>(),
                "targets": ks.iter().map(|&k| prepared[e].reads.targets[k]).collect::<Vec<_>>(),
                "old": ks.iter().map(|&k| old[e][k]).collect::<Vec<_>>(),
                "z": z[e],
                "weight": if counts[e] == 0 { 0.0 } else { 1.0 / (counts[e] as f32 * eps.len() as f32) },
                "reward": scores[e].total,
                "kl": kl[e],
                "advantage": adv[e],
            })
        })
        .collect();
    let batch = json!({
        "adapter": args[1],
        "seed": seed,
        "suppress": h.suppress,
        "clip_eps": settings.clip_eps,
        "settings": settings,
        "step": step,
        "orders": (0..settings.inner_epochs).map(|epoch| update::order(settings.seed, step, epoch, groups)).collect::<Vec<_>>(),
        "groups": group_ix,
        "episodes": episodes,
    });
    std::fs::write(&out, serde_json::to_string(&batch).unwrap()).unwrap();
    for e in 0..eps.len() {
        println!(
            "episode {e}: {} tokens, {} actor, reward {:.3}, kl {:.2}, advantage {:+.3}, z {:+.3}",
            seqs[e].len(),
            counts[e],
            scores[e].total,
            kl[e],
            adv[e],
            z[e]
        );
    }
    println!("wrote {}", out.display());
}
