//! Does the sampler draw from the model the trainer scores? Rolls out a batch with an adapter,
//! then recomputes every policy token's log-prob with the update's prefix-tree forward and
//! compares it with the log-prob the sampler drew the token at.
//!
//! `lp_check <adapter dir under smoke_test35> [groups] [seed]`

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
use membrane_rl::tree::{self, Reads, Tree};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap();
    let data = repo.join("experiments/notebooks/smoke_test35");
    let adapter = data.join(&args[1]);
    let groups: usize = args.get(2).map_or(8, |s| s.parse().unwrap());
    let seed: u64 = args.get(3).map_or(7, |s| s.parse().unwrap());
    let gs = 4;

    let device = Device::cuda(0);
    let snapshot = hf::snapshot("Qwen/Qwen3.5-0.8B").unwrap();
    let (mut model, config) = Qwen3_5ForCausalLM::from_pretrained(&snapshot, FloatDType::BF16, &device).unwrap();
    lora::load(&mut model, &adapter, lora::POLICY, false, &device).unwrap();
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
    let (eps, sampled) = generator.rollout(&cases, &starts, &mut tools, cases.len()).unwrap();
    generator.reset();
    device.memory_cleanup();

    let prepared: Vec<grpo::Prepared> = eps
        .iter()
        .map(|ep| grpo::prepare(ep, &episode_rows(&h, ep, &score(&ep.case, &ep.calls, &ep.final_text, &ERR0), h.suppress)))
        .collect();
    let seqs: Vec<Vec<u32>> = eps.iter().map(|ep| [ep.prompt_ids.as_slice(), &ep.ids].concat()).collect();
    let limit: Vec<usize> = prepared
        .iter()
        .zip(&seqs)
        .map(|(p, s)| p.reads.at.iter().copied().min().unwrap_or(s.len() - 1))
        .collect();
    let group_ix: Vec<Vec<usize>> = (0..groups).map(|g| (g * gs..(g + 1) * gs).collect()).collect();
    let t = Tree::build(&seqs, &limit, &group_ix, 64);
    let reads: Vec<Reads> = (0..eps.len())
        .map(|e| {
            let start = t.segments[t.leaf[e]].start;
            Reads {
                at: prepared[e].reads.at.iter().map(|&p| p - start).collect(),
                targets: prepared[e].reads.targets.clone(),
            }
        })
        .collect();
    let scored = tree::logprobs(&model.valid(), &table, &t, &reads, &[h.suppress]);

    // (|diff|, sampled, scored, episode, token index) over every policy token.
    let mut all: Vec<(f64, f32, f32, usize, usize)> = Vec::new();
    for e in 0..eps.len() {
        if sampled[e].len() != scored[e].len() {
            println!("episode {e}: {} sampled log-probs, {} policy tokens", sampled[e].len(), scored[e].len());
            continue;
        }
        for k in 0..scored[e].len() {
            let (a, b) = (sampled[e][k], scored[e][k]);
            all.push((((a - b) as f64).abs(), a, b, e, k));
        }
    }
    let n = all.len() as f64;
    let mean = all.iter().map(|x| x.0).sum::<f64>() / n;
    let signed = all.iter().map(|x| (x.1 - x.2) as f64).sum::<f64>() / n;
    let frac = |c: f64| all.iter().filter(|x| x.0 > c).count() as f64 / n;
    let mut sorted: Vec<f64> = all.iter().map(|x| x.0).collect();
    sorted.sort_by(f64::total_cmp);
    let q = |p: f64| sorted[((n - 1.0) * p) as usize];
    let ent_s = -all.iter().map(|x| x.1 as f64).sum::<f64>() / n;
    let ent_t = -all.iter().map(|x| x.2 as f64).sum::<f64>() / n;
    let rare: Vec<_> = all.iter().filter(|x| x.1 < -2.0).collect();
    let rare_mean = rare.iter().map(|x| x.0).sum::<f64>() / rare.len().max(1) as f64;
    println!("{} episodes, {} policy tokens, adapter {}", eps.len(), all.len(), adapter.display());
    println!("|sampled - scored|: mean {mean:.2e}, median {:.2e}, p99 {:.2e}, max {:.2e}; signed mean {signed:+.2e}", q(0.5), q(0.99), q(1.0));
    println!("share above 0.05 {:.4}, above 0.5 {:.4}, above 2 {:.4}", frac(0.05), frac(0.5), frac(2.0));
    println!("mean -logprob: sampler {ent_s:.4}, trainer {ent_t:.4}");
    println!("tokens the sampler drew below p=e^-2: {} ({:.4}), their mean |diff| {rare_mean:.2e}", rare.len(), rare.len() as f64 / n);
    all.sort_by(|a, b| b.0.total_cmp(&a.0));
    for &(d, a, b, e, k) in all.iter().take(12) {
        let tok = prepared[e].reads.targets[k];
        println!("  diff {d:.3}  sampled {a:.3}  scored {b:.3}  ep {e} tok {k} {:?}", h.piece(tok));
    }
}
