//! The update's log-probs on recorded episodes against PyTorch's (export_logprob_reference.py):
//! in f32 against transformers' fp32 to float error, in bf16 against the fp32 truth no further
//! than PyTorch's own bf16 forward (07's training path) is from it.

use std::path::PathBuf;

use burn::module::Module;
use burn::tensor::{DType, Device, FloatDType};
use membrane_rl::hf;
use membrane_rl::lora;
use membrane_rl::probe;
use membrane_rl::qwen3_5::{Qwen3_5ForCausalLM, RotaryTable};
use membrane_rl::tree::{self, Reads, Tree};
use safetensors::SafeTensors;

fn env(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name}"))
}

fn read(path: &PathBuf) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).unwrap();
    let file = SafeTensors::deserialize(&bytes).unwrap();
    let mut out = Vec::new();
    for i in 0.. {
        let Ok(t) = file.tensor(&format!("lp.{i}")) else { break };
        out.push(
            t.data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        );
    }
    out
}

fn ids(v: &membrane_rl::membrane::pyjson::Value) -> Vec<u32> {
    v.as_list().unwrap().iter().map(|t| t.as_f64().unwrap() as u32).collect()
}

fn stats(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f64, f64) {
    let d: Vec<f64> = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .collect();
    (d.iter().sum::<f64>() / d.len() as f64, d.iter().copied().fold(0.0, f64::max))
}

#[test]
#[ignore = "needs CUDA, the checkpoint, MEMBRANE_EPISODES, MEMBRANE_ADAPTER, MEMBRANE_LP_FP32 and MEMBRANE_LP_BF16"]
fn logprobs_match_pytorch() {
    let device = Device::cuda(0);
    let fp32 = read(&env("MEMBRANE_LP_FP32"));
    let bf16_torch = read(&env("MEMBRANE_LP_BF16"));
    let n = fp32.len();
    let recorded = probe::load_jsonl(&env("MEMBRANE_EPISODES")).unwrap();
    let eps: Vec<_> = recorded.iter().filter_map(|r| r.get("episode").cloned()).take(n).collect();
    let seqs: Vec<Vec<u32>> = eps
        .iter()
        .map(|e| [ids(e.get("prompt_ids").unwrap()), ids(e.get("ids").unwrap())].concat())
        .collect();
    let reads: Vec<Reads> = eps
        .iter()
        .map(|e| {
            let p = ids(e.get("prompt_ids").unwrap()).len();
            let toks = ids(e.get("ids").unwrap());
            let policy = ids(e.get("policy").unwrap());
            let mut r = Reads::default();
            for (k, &mine) in policy.iter().enumerate() {
                if mine == 1 {
                    r.at.push(p + k - 1);
                    r.targets.push(toks[k]);
                }
            }
            r
        })
        .collect();
    let limit: Vec<usize> = reads.iter().map(|r| r.at[0]).collect();
    // All the episodes in one group: the shared system turn runs once.
    let t = Tree::build(&seqs, &limit, &[(0..n).collect()], 64);
    let local: Vec<Reads> = reads
        .iter()
        .enumerate()
        .map(|(e, r)| {
            let start = t.segments[t.leaf[e]].start;
            Reads {
                at: r.at.iter().map(|&p| p - start).collect(),
                targets: r.targets.clone(),
            }
        })
        .collect();
    let dir = hf::snapshot("Qwen/Qwen3.5-0.8B").unwrap();
    let mut results = Vec::new();
    for dtype in [FloatDType::F32, FloatDType::BF16] {
        let (mut model, config) = Qwen3_5ForCausalLM::from_pretrained(&dir, dtype, &device).unwrap();
        lora::load(&mut model, &env("MEMBRANE_ADAPTER"), lora::POLICY, false, &device).unwrap();
        let model = lora::choose(model, lora::POLICY).valid();
        let table = RotaryTable::new(&config, 16384, DType::from(dtype), &device);
        let lp = tree::logprobs(&model, &table, &t, &local, &[248044]);
        results.push(lp);
    }
    let tokens: usize = fp32.iter().map(Vec::len).sum();
    let (rust32_mean, rust32_max) = stats(&results[0], &fp32);
    let (rust16_mean, rust16_max) = stats(&results[1], &fp32);
    let (torch16_mean, torch16_max) = stats(&bf16_torch, &fp32);
    let (cross_mean, cross_max) = stats(&results[1], &bf16_torch);
    println!("{n} episodes, {tokens} policy tokens; |diff| from transformers fp32, mean / max:");
    println!("  rust f32       {rust32_mean:.3e} / {rust32_max:.3e}");
    println!("  rust bf16      {rust16_mean:.3e} / {rust16_max:.3e}");
    println!("  pytorch bf16   {torch16_mean:.3e} / {torch16_max:.3e}");
    println!("  rust bf16 against pytorch bf16: {cross_mean:.3e} / {cross_max:.3e}");
    assert!(rust32_max < 1e-3, "f32 against fp32");
    assert!(rust16_mean <= torch16_mean * 1.5, "bf16 noise beyond PyTorch's own");
}
