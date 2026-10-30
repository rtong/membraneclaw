//! One whole update step as nb10 runs it (`update::step`: old and reference log-probs, returns,
//! advantages, 4 epochs of minibatches, each gradient clipped and one AdamW step), on a recorded
//! batch (`grad_batch`), against PyTorch's step on the same batch from the formulas
//! (export_step_reference.py): the step's start, each minibatch's gradient norm, and where the
//! policy's factors end up.

use std::collections::BTreeMap;
use std::path::PathBuf;

use burn::tensor::{DType, Device, FloatDType, Tensor};
use membrane_rl::grpo::Prepared;
use membrane_rl::hf;
use membrane_rl::lora;
use membrane_rl::qwen3_5::{Qwen3_5ForCausalLM, RotaryTable};
use membrane_rl::tree::Reads;
use membrane_rl::update::{self, Outcome, Settings};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

#[derive(Deserialize)]
struct Episode {
    seq: Vec<u32>,
    reads_at: Vec<usize>,
    reads_targets: Vec<u32>,
    actor: Vec<bool>,
    score: f64,
}

#[derive(Deserialize)]
struct Batch {
    suppress: u32,
    settings: Settings,
    step: usize,
    orders: Vec<Vec<usize>>,
    groups: Vec<Vec<usize>>,
    episodes: Vec<Episode>,
}

fn env(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name}"))
}

fn batch() -> Batch {
    serde_json::from_slice(&std::fs::read(env("MEMBRANE_STEP_BATCH")).unwrap()).unwrap()
}

/// Each LoRA factor by its PEFT name (`<projection>.lora_A` / `.lora_B`): shape and values.
type Factors = BTreeMap<String, (Vec<usize>, Vec<f64>)>;

fn host<const D: usize>(t: Tensor<D>) -> Vec<f64> {
    t.cast(FloatDType::F32)
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(|x| x as f64)
        .collect()
}

fn factors(m: &Qwen3_5ForCausalLM) -> Factors {
    let mut out = Factors::new();
    for (path, layer) in m.projections() {
        let Some(l) = layer.adapter(lora::POLICY) else { continue };
        for (name, p) in [("lora_A", l.a.val()), ("lora_B", l.b.val())] {
            out.insert(format!("{path}.{name}"), (p.dims().to_vec(), host(p)));
        }
    }
    out
}

/// nb10's step on the batch in `dtype`: the policy's factors before and after, and the step's record.
fn run(b: &Batch, dtype: FloatDType) -> (Factors, Factors, Outcome) {
    let device = Device::cuda(0);
    let ad = device.clone().autodiff();
    let dir = hf::snapshot("Qwen/Qwen3.5-0.8B").unwrap();
    let (mut model, config) = Qwen3_5ForCausalLM::from_pretrained(&dir, dtype, &ad).unwrap();
    lora::load(&mut model, &env("MEMBRANE_ADAPTER"), lora::POLICY, true, &ad).unwrap();
    lora::load(&mut model, &env("MEMBRANE_REFERENCE"), lora::REFERENCE, false, &ad).unwrap();
    let model = lora::choose(model, lora::POLICY);
    let table = RotaryTable::new(&config, 16384, DType::from(dtype), &device);
    let table_ad = RotaryTable::new(&config, 16384, DType::from(dtype), &ad);
    let seqs: Vec<Vec<u32>> = b.episodes.iter().map(|e| e.seq.clone()).collect();
    let prepared: Vec<Prepared> = b
        .episodes
        .iter()
        .map(|e| Prepared {
            reads: Reads { at: e.reads_at.clone(), targets: e.reads_targets.clone() },
            actor: e.actor.clone(),
        })
        .collect();
    let scores: Vec<f64> = b.episodes.iter().map(|e| e.score).collect();
    let batch = update::Batch { seqs: &seqs, prepared: &prepared, scores: &scores, groups: &b.groups };
    let theta0 = factors(&model);
    let mut optim = b.settings.optimizer();
    let (model, out) = update::step(model, &mut optim, &b.settings, &table, &table_ad, &batch, &[b.suppress], b.step);
    let per_mb = b.groups.len() / b.settings.minibatches;
    let taken: Vec<Vec<usize>> = out.minibatches.iter().map(|m| m.groups.clone()).collect();
    let recorded: Vec<Vec<usize>> = b.orders.iter().flat_map(|o| o.chunks(per_mb).map(<[usize]>::to_vec)).collect();
    assert_eq!(taken, recorded, "the step's minibatches are the recorded orders'");
    println!(
        "rust {dtype:?}: {} updates, start {:.0}s, update {:.0}s",
        out.minibatches.len(),
        out.before_seconds,
        out.update_seconds
    );
    (theta0, factors(&model), out)
}

/// PyTorch's factors after the step, and its `step.` records.
fn read_torch(path: &PathBuf) -> (Factors, BTreeMap<String, Vec<f64>>) {
    let bytes = std::fs::read(path).unwrap();
    let file = SafeTensors::deserialize(&bytes).unwrap();
    let (mut f, mut rec) = (Factors::new(), BTreeMap::new());
    for (k, t) in file.tensors() {
        let v: Vec<f64> = match t.dtype() {
            Dtype::F32 => t.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            Dtype::F64 => t.data().chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
            d => panic!("{k}: {d:?}"),
        };
        match k.strip_prefix("step.") {
            Some(name) => {
                rec.insert(name.to_owned(), v);
            }
            None => {
                f.insert(k, (t.shape().to_vec(), v));
            }
        }
    }
    (f, rec)
}

fn delta(after: &Factors, before: &Factors) -> Factors {
    after
        .iter()
        .map(|(k, (d, v))| (k.clone(), (d.clone(), v.iter().zip(&before[k].1).map(|(x, y)| x - y).collect())))
        .collect()
}

fn dot(a: &Factors, b: &Factors) -> f64 {
    a.iter()
        .filter_map(|(k, (_, x))| b.get(k).map(|(_, y)| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>()))
        .sum()
}

fn norm(a: &Factors) -> f64 {
    dot(a, a).sqrt()
}

/// Cosine and norm ratio over all factors, the worst factor's cosine and where, and the share of
/// entries whose sign agrees with `b`'s.
fn compare(a: &Factors, b: &Factors) -> (f64, f64, f64, String, f64) {
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "the same factors");
    let cos = dot(a, b) / (norm(a) * norm(b));
    let (mut worst, mut at) = (1.0f64, String::new());
    let (mut same, mut count) = (0usize, 0usize);
    for (k, (da, x)) in a {
        let (db, y) = &b[k];
        assert_eq!(da, db, "{k}: shapes");
        let xy: f64 = x.iter().zip(y).map(|(p, q)| p * q).sum();
        let nx = x.iter().map(|p| p * p).sum::<f64>().sqrt();
        let ny = y.iter().map(|q| q * q).sum::<f64>().sqrt();
        let c = if nx * ny > 0.0 { xy / (nx * ny) } else { 1.0 };
        if c < worst {
            (worst, at) = (c, k.clone());
        }
        for (p, q) in x.iter().zip(y) {
            if *q != 0.0 {
                count += 1;
                same += usize::from(p.signum() == q.signum());
            }
        }
    }
    (cos, norm(a) / norm(b), worst, at, same as f64 / count as f64)
}

fn worst_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// The step's start and its minibatches beside PyTorch's: the worst differences in the old
/// log-probs (and their mean), the KL, the advantages, and the gradient norms (relative).
fn starts(out: &Outcome, rec: &BTreeMap<String, Vec<f64>>) -> (f64, f64, f64, f64, f64) {
    let old: Vec<f64> = out.before.old.iter().flatten().map(|&x| x as f64).collect();
    let mean_old = old.iter().zip(&rec["old"]).map(|(x, y)| (x - y).abs()).sum::<f64>() / old.len() as f64;
    let norms: Vec<f64> = out.minibatches.iter().map(|m| m.grad_norm as f64).collect();
    let rel = norms.iter().zip(&rec["grad_norm"]).map(|(x, y)| (x / y - 1.0).abs()).fold(0.0, f64::max);
    println!("  grad norms   rust {}", norms.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" "));
    println!("               torch {}", rec["grad_norm"].iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" "));
    let ratio: Vec<f64> = out.minibatches.iter().map(|m| m.ratio).collect();
    let akl: Vec<f64> = out.minibatches.iter().map(|m| m.approx_kl).collect();
    println!("  ratio  worst {:.2e}; approx kl worst {:.2e} (last rust {:.3e} torch {:.3e})", worst_abs(&ratio, &rec["ratio"]), worst_abs(&akl, &rec["approx_kl"]), akl.last().unwrap(), rec["approx_kl"].last().unwrap());
    (
        worst_abs(&old, &rec["old"]),
        mean_old,
        worst_abs(&out.before.kl, &rec["kl"]),
        worst_abs(&out.before.advantages, &rec["advantages"]),
        rel,
    )
}

#[test]
#[ignore = "needs CUDA, the checkpoint, MEMBRANE_STEP_BATCH (grad_batch), MEMBRANE_ADAPTER, MEMBRANE_REFERENCE and MEMBRANE_STEP_FP32"]
fn step_matches_pytorch_in_f32() {
    let b = batch();
    let (torch, rec) = read_torch(&env("MEMBRANE_STEP_FP32"));
    let (theta0, theta1, out) = run(&b, FloatDType::F32);
    let (old, old_mean, kl, adv, norms) = starts(&out, &rec);
    println!("  old log-probs worst {old:.2e} (mean {old_mean:.2e}); KL worst {kl:.2e}; advantages worst {adv:.2e}; grad norms worst relative {norms:.2e}");
    let (cos, ratio, worst, at, sign) = compare(&delta(&theta1, &theta0), &delta(&torch, &theta0));
    println!("  the step's change of the factors: cosine {cos:.8}, norm ratio {ratio:.6}, worst factor {worst:.6} ({at}), same sign {sign:.4}; |change| {:.4e}", norm(&delta(&theta1, &theta0)));
    assert!(adv < 1e-4, "advantages");
    assert!(norms < 1e-3, "gradient norms");
    assert!(cos > 0.9999 && (ratio - 1.0).abs() < 1e-3 && worst > 0.999, "the step against PyTorch's fp32");
}

#[test]
#[ignore = "needs CUDA, the checkpoint, MEMBRANE_STEP_BATCH, MEMBRANE_ADAPTER, MEMBRANE_REFERENCE, MEMBRANE_STEP_FP32 and MEMBRANE_STEP_BF16"]
fn step_in_bf16_is_no_noisier_than_pytorchs() {
    let b = batch();
    let (torch32, _) = read_torch(&env("MEMBRANE_STEP_FP32"));
    let (torch16, rec16) = read_torch(&env("MEMBRANE_STEP_BF16"));
    let (theta0, theta1, out) = run(&b, FloatDType::BF16);
    let (old, old_mean, kl, adv, norms) = starts(&out, &rec16);
    println!("  against PyTorch's bf16: old log-probs worst {old:.2e} (mean {old_mean:.2e}); KL worst {kl:.2e}; advantages worst {adv:.2e}; grad norms worst relative {norms:.2e}");
    let d32 = delta(&torch32, &theta0);
    let rows = [
        ("rust bf16 vs pytorch fp32", compare(&delta(&theta1, &theta0), &d32)),
        ("pytorch bf16 vs pytorch fp32", compare(&delta(&torch16, &theta0), &d32)),
        ("rust bf16 vs pytorch bf16", compare(&delta(&theta1, &theta0), &delta(&torch16, &theta0))),
    ];
    println!("  the step's change of the factors: cosine, norm ratio, worst factor's cosine, same sign");
    for (label, (cos, ratio, worst, at, sign)) in &rows {
        println!("  {label:30} {cos:.6}  {ratio:.6}  {worst:.6} ({at})  {sign:.4}");
    }
    let (c16, ..) = &rows[0].1;
    let (t16, ..) = &rows[1].1;
    assert!(1.0 - c16 <= 2.0 * (1.0 - t16) + 1e-4, "bf16 further from fp32 than PyTorch's own bf16");
}
