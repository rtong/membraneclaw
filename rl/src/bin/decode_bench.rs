//! Milliseconds per decode step: the merged decoder's layers, the logits, the sampler, at
//! batch sizes and cache lengths.
use std::time::Instant;

use burn::tensor::{DType, Device, FloatDType, Int, Tensor, TensorData};
use dash_ops::sample::{SampleOptions, sample};
use membrane_rl::hf;
use membrane_rl::lora;
use membrane_rl::qwen3_5::{Decoder, Qwen3_5ForCausalLM, RotaryTable};

fn main() {
    let device = Device::cuda(0);
    let dir = hf::snapshot("Qwen/Qwen3.5-0.8B").unwrap();
    let (mut model, config) = Qwen3_5ForCausalLM::from_pretrained(&dir, FloatDType::BF16, &device).unwrap();
    let seed = std::path::PathBuf::from("/home/bayan/MembraneClaw-nb10/experiments/notebooks/smoke_test35/09_fresh_questions_runs/seed-09-k2/epoch1");
    lora::load(&mut model, &seed, lora::POLICY, false, &device).unwrap();
    let model = lora::choose(model, lora::POLICY);
    let table = RotaryTable::new(&config, 16384, DType::BF16, &device);
    let decoder = Decoder::new(&model);
    let sync = || device.sync().unwrap();
    for &(b, len) in &[(1usize, 3000usize), (8, 3000), (32, 3000), (32, 7000), (48, 3000)] {
        let ids: Vec<i64> = (0..len as i64).map(|i| 1000 + (i * 7919) % 50000).collect();
        let mut state = model.empty_state();
        let _ = model.forward(Tensor::<2, Int>::from_data(TensorData::new(ids, [1, len]), &device), &table, &mut state);
        let rows: Vec<_> = (0..b).map(|_| &state).collect();
        let mut batch = model.batch(&rows);
        let tokens = || Tensor::<1, Int>::from_data(TensorData::new(vec![1234i64; b], [b]), &device);
        let mut h = decoder.decode(&model, tokens(), &table, &mut batch);
        sync();
        let n = 40;
        let t0 = Instant::now();
        for _ in 0..n {
            h = decoder.decode(&model, tokens(), &table, &mut batch);
        }
        sync();
        let layers = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        let t1 = Instant::now();
        for i in 0..n {
            let logits = model.logits(h.clone());
            let s = sample(logits, &SampleOptions { temperature: 1.0, seed: 0, offset: i, masked: vec![248044], ..SampleOptions::default() });
            let _ = s.tokens.into_data();
        }
        let head = t1.elapsed().as_secs_f64() * 1e3 / n as f64;
        let t2 = Instant::now();
        for _ in 0..n {
            h = decoder.decode(&model, tokens(), &table, &mut batch);
            let logits = model.logits(h.clone());
            let s = sample(logits, &SampleOptions { temperature: 1.0, ..SampleOptions::default() });
            let _ = s.tokens.into_data();
            let _ = s.sampling_logprob.into_data();
        }
        let full = t2.elapsed().as_secs_f64() * 1e3 / n as f64;
        println!("B {b:>2} cache {len:>5}: layers {layers:6.2} ms, logits+sample+read {head:6.2} ms, full step {full:6.2} ms");
    }
}
