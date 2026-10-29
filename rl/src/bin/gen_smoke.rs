//! Times the generator on a few dev questions, greedy, with an adapter.
use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::{Device, FloatDType};
use dash_ops::sample::SampleOptions;
use membrane_rl::generate::Generator;
use membrane_rl::hf;
use membrane_rl::lora;
use membrane_rl::membrane::episode::{Harness, rollout};
use membrane_rl::membrane::pyjson;
use membrane_rl::membrane::reward::{CallCredit, Settings, score};
use membrane_rl::membrane::tools::ToolPool;
use membrane_rl::qwen3_5::{Qwen3_5ForCausalLM, RotaryTable};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map_or(8, |s| s.parse().unwrap());
    let temperature: f32 = args.get(2).map_or(0.0, |s| s.parse().unwrap());
    let nb = PathBuf::from("/home/bayan/MembraneClaw-nb10/experiments/notebooks/smoke_test35");
    let device = Device::cuda(0);
    let dir = hf::snapshot("Qwen/Qwen3.5-0.8B").unwrap();
    let (mut model, config) = Qwen3_5ForCausalLM::from_pretrained(&dir, FloatDType::BF16, &device).unwrap();
    lora::load(&mut model, &nb.join("09_fresh_questions_runs/seed-09-k2/epoch1"), lora::POLICY, false, &device).unwrap();
    let model = lora::choose(model, lora::POLICY);
    let h = Harness::new(&dir, &nb.join("08_prover_critic_tools.json"), 3072, true).unwrap();
    let table = RotaryTable::new(&config, 32768, burn::tensor::DType::BF16, &device);
    let options = SampleOptions { temperature, seed: 0, ..SampleOptions::default() };
    let mut g = Generator::new(model, table, &h, options);
    g.capacity = std::env::var("CAPACITY").map_or(0, |v| v.parse().unwrap());
    let split = args.get(3).map_or("06_direction_and_compare_data/dev.jsonl", |s| s.as_str());
    let text = std::fs::read_to_string(nb.join(split)).unwrap();
    let cases: Vec<_> = text.lines().take(n).map(|l| pyjson::loads(l).unwrap()).collect();
    let mut pool = ToolPool::niced(
        &PathBuf::from("/home/bayan/MembraneClaw/.venv-watertap/bin/python"),
        &nb.parent().unwrap().join("smoke_test35/07_error_recovery.py"),
        std::env::var("WORKERS").map_or(8, |v| v.parse().unwrap()),
        std::env::var("NICE").map_or(0, |v| v.parse().unwrap()),
    )
    .unwrap();
    let t0 = Instant::now();
    let continuous = std::env::var_os("SYNC").is_none();
    let eps = if continuous {
        let starts = vec![membrane_rl::membrane::episode::Start::default(); cases.len()];
        g.rollout(&cases, &starts, &mut pool, cases.len()).unwrap().0
    } else {
        rollout(&h, &cases, &mut pool, &mut g).unwrap()
    };
    let secs = t0.elapsed().as_secs_f64();
    let settings = Settings { err_call_credit: false, call_credit: CallCredit::Delta, provenance: false };
    let mut total = 0.0;
    for (ep, case) in eps.iter().zip(&cases) {
        let s = score(case, &ep.calls, &ep.final_text, &settings);
        total += s.total;
        println!("{:>6.3} calls {:>2} policy {:>4} turns {} trunc {}", s.total, ep.calls.len(), ep.policy_tokens(), ep.turns.len(), ep.truncated);
    }
    println!("mean reward {:.4} over {}; {:.1} s", total / eps.len() as f64, eps.len(), secs);
    println!("{:?}", g.stats);
    println!("ms per decode step {:.2}", g.stats.decode_seconds * 1e3 / g.stats.decode_steps as f64);
}
