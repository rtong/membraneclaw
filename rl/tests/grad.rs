//! The update's gradient on one real minibatch (`grad_batch`), against what it must be:
//! central differences of the loss through the forward (which is held to transformers'), PyTorch's
//! autograd on the same loss (export_grad_reference.py), and one update as nb10 makes it, whose
//! effect on the loss must follow the gradient.
//!
//! The loss is nb10's: sum_e weight_e * sum_t -min(r z_e, clip(r) z_e), r = exp(lp - old).

use std::collections::BTreeMap;
use std::path::PathBuf;

use burn::module::Module;
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::tensor::{DType, Device, FloatDType, Tensor, TensorData};
use dash_lora::Lora;
use dash_train::clip::{ClipGradNormOptions, clip_grad_norm};
use dash_train::optim::{AdamW, Beta1};
use membrane_rl::hf;
use membrane_rl::lora;
use membrane_rl::qwen3_5::{Qwen3_5ForCausalLM, RotaryTable};
use membrane_rl::tree::{self, Objective, Reads, Tree};
use safetensors::SafeTensors;
use serde::Deserialize;

const PROBE: &str = "probe";

#[derive(Deserialize)]
struct Episode {
    seq: Vec<u32>,
    limit: usize,
    at: Vec<usize>,
    targets: Vec<u32>,
    old: Vec<f32>,
    z: f32,
    weight: f32,
}

#[derive(Deserialize)]
struct Batch {
    suppress: u32,
    clip_eps: f32,
    groups: Vec<Vec<usize>>,
    episodes: Vec<Episode>,
}

fn env(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name}"))
}

fn batch() -> Batch {
    serde_json::from_slice(&std::fs::read(env("MEMBRANE_GRAD_BATCH")).unwrap()).unwrap()
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

struct Setup {
    model: Qwen3_5ForCausalLM,
    table: RotaryTable,
    table_ad: RotaryTable,
    device: Device,
}

fn setup(dtype: FloatDType) -> Setup {
    let device = Device::cuda(0);
    let ad = device.clone().autodiff();
    let dir = hf::snapshot("Qwen/Qwen3.5-0.8B").unwrap();
    let (mut model, config) = Qwen3_5ForCausalLM::from_pretrained(&dir, dtype, &ad).unwrap();
    lora::load(&mut model, &env("MEMBRANE_ADAPTER"), lora::POLICY, true, &ad).unwrap();
    let model = lora::choose(model, lora::POLICY);
    let table = RotaryTable::new(&config, 16384, DType::from(dtype), &device);
    let table_ad = RotaryTable::new(&config, 16384, DType::from(dtype), &ad);
    Setup { model, table, table_ad, device }
}

/// The tree over `groups` and each member's actor reads in its own segment's coordinates.
fn tree_of(b: &Batch, groups: &[Vec<usize>]) -> (Tree, Vec<Reads>) {
    let seqs: Vec<Vec<u32>> = b.episodes.iter().map(|e| e.seq.clone()).collect();
    let limit: Vec<usize> = b.episodes.iter().map(|e| e.limit).collect();
    let t = Tree::build(&seqs, &limit, groups, 64);
    let reads = (0..b.episodes.len())
        .map(|e| {
            if t.leaf[e] == usize::MAX {
                return Reads::default();
            }
            let start = t.segments[t.leaf[e]].start;
            Reads {
                at: b.episodes[e].at.iter().map(|&p| p - start).collect(),
                targets: b.episodes[e].targets.clone(),
            }
        })
        .collect();
    (t, reads)
}

/// The loss in f64 from the actor tokens' log-probs.
fn loss_of(b: &Batch, lp: &[Vec<f32>]) -> f64 {
    let c = b.clip_eps as f64;
    b.episodes
        .iter()
        .zip(lp)
        .filter(|(e, _)| e.weight > 0.0)
        .map(|(e, lp)| {
            let z = e.z as f64;
            let s: f64 = lp
                .iter()
                .zip(&e.old)
                .map(|(&l, &o)| {
                    let r = (l as f64 - o as f64).exp();
                    (r * z).min(r.clamp(1.0 - c, 1.0 + c) * z)
                })
                .sum();
            -s * e.weight as f64
        })
        .sum()
}

fn loss(s: &Setup, b: &Batch, model: &Qwen3_5ForCausalLM) -> f64 {
    let (t, reads) = tree_of(b, &b.groups);
    let lp = tree::logprobs(model, &s.table, &t, &reads, &[b.suppress]);
    loss_of(b, &lp)
}

/// The update's gradient of the loss, one group's tree at a time, summed on the host.
fn gradient(s: &Setup, b: &Batch) -> Factors {
    let valid = s.model.valid();
    let mut out = Factors::new();
    for g in &b.groups {
        let (t, reads) = tree_of(b, std::slice::from_ref(g));
        let objectives: Vec<Option<Objective>> = (0..b.episodes.len())
            .map(|e| {
                let ep = &b.episodes[e];
                (g.contains(&e) && ep.weight > 0.0).then(|| Objective {
                    reads: reads[e].clone(),
                    old: ep.old.clone(),
                    advantages: vec![ep.z; ep.old.len()],
                    weight: ep.weight,
                })
            })
            .collect();
        let (grads, _) = tree::gradients(
            &s.model, &valid, &s.table, &s.table_ad, &t, &objectives, &[b.suppress], b.clip_eps,
        );
        for (path, layer) in s.model.projections() {
            let Some(l) = layer.adapter(lora::POLICY) else { continue };
            for (name, p) in [("lora_A", l.a.val()), ("lora_B", l.b.val())] {
                let dims = p.dims().to_vec();
                let n = dims.iter().product();
                let g = p.grad(&grads).map_or(vec![0.0; n], host);
                let entry = out.entry(format!("{path}.{name}")).or_insert((dims, vec![0.0; n]));
                entry.1.iter_mut().zip(g).for_each(|(a, x)| *a += x);
            }
        }
        s.device.memory_cleanup();
    }
    out
}

/// The model's policy factors, moved by `eps` along `dir` (factors missing from `dir` stay), as
/// the adapter its forward runs.
fn moved(s: &Setup, dir: &Factors, eps: f64) -> Qwen3_5ForCausalLM {
    let mut v = s.model.valid();
    for (path, layer) in v.projections_mut() {
        let Some(l) = layer.adapter(lora::POLICY) else { continue };
        let shift = |t: Tensor<2>, key: String| match dir.get(&key) {
            Some((dims, d)) => {
                let d: Vec<f32> = d.iter().map(|x| (x * eps) as f32).collect();
                t.clone() + Tensor::<2>::from_data(TensorData::new(d, dims.clone()), (&t.device(), DType::F32))
            }
            None => t,
        };
        let a = shift(l.a.val(), format!("{path}.lora_A"));
        let b = shift(l.b.val(), format!("{path}.lora_B"));
        let probe = Lora::new(a, b, l.scale);
        layer.insert_adapter(PROBE, probe);
    }
    lora::choose(v, PROBE)
}

fn dot(a: &Factors, b: &Factors) -> f64 {
    a.iter()
        .filter_map(|(k, (_, x))| b.get(k).map(|(_, y)| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>()))
        .sum()
}

fn norm(a: &Factors) -> f64 {
    dot(a, a).sqrt()
}

/// `g` restricted to the factors `keep` accepts, scaled to unit norm.
fn unit(g: &Factors, keep: impl Fn(&str) -> bool) -> Factors {
    let part: Factors = g.iter().filter(|(k, _)| keep(k)).map(|(k, v)| (k.clone(), v.clone())).collect();
    let n = norm(&part);
    part.into_iter()
        .map(|(k, (d, v))| (k, (d, v.into_iter().map(|x| x / n).collect())))
        .collect()
}

#[test]
#[ignore = "needs CUDA, the checkpoint, MEMBRANE_GRAD_BATCH (grad_batch) and MEMBRANE_ADAPTER"]
fn gradient_matches_central_differences() {
    // Old log-probs taken by this f32 forward at the starting point: every ratio is 1 there, so
    // the loss is smooth until a step is large enough to clip.
    let mut b = batch();
    let s = setup(FloatDType::F32);
    let (t, reads) = tree_of(&b, &b.groups);
    let lp0 = tree::logprobs(&s.model.valid(), &s.table, &t, &reads, &[b.suppress]);
    for (e, lp) in b.episodes.iter_mut().zip(lp0) {
        e.old = lp;
    }
    let g = gradient(&s, &b);
    let l0 = loss(&s, &b, &s.model.valid());
    println!("f32: loss {l0:+.8e}, |g| {:.6e} over {} factors", norm(&g), g.len());

    // slope along d: (L(+eps) - L(-eps)) / 2 eps, against g . d = |g restricted|. The loss is
    // good to about 1e-7 in f32, so only differences well above that are held to the gradient.
    let check = |label: &str, d: &Factors, eps: f64| -> (f64, f64) {
        let up = loss(&s, &b, &moved(&s, d, eps));
        let down = loss(&s, &b, &moved(&s, d, -eps));
        s.device.memory_cleanup();
        let fd = (up - down) / (2.0 * eps);
        let analytic = dot(&g, d);
        let ratio = fd / analytic;
        println!("  {label:44} eps {eps:.0e}  change {:.1e}  differences {fd:+.6e}  gradient {analytic:+.6e}  ratio {ratio:.5}", up - down);
        (ratio, (up - down).abs())
    };
    let all = unit(&g, |_| true);
    for eps in [1e-3, 3e-4, 1e-4, 3e-5] {
        check("all factors", &all, eps);
    }
    let (mut worst, mut held) = (0.0f64, 0);
    let mut hold = |(ratio, change): (f64, f64)| {
        if change >= 2e-5 {
            worst = worst.max((ratio - 1.0).abs());
            held += 1;
        }
    };
    for i in 0..24 {
        let at = format!("model.layers.{i}.");
        let d = unit(&g, |k| k.starts_with(&at));
        for eps in [3e-4, 1e-4] {
            hold(check(&format!("layer {i}"), &d, eps));
        }
    }
    for i in [0usize, 3, 12, 23] {
        let at = format!("model.layers.{i}.");
        for k in g.keys().filter(|k| k.starts_with(&at)) {
            let d = unit(&g, |x| x == k);
            hold(check(k, &d, 3e-4));
        }
    }
    println!("worst |ratio - 1| over {held} differences above 2e-5: {worst:.2e}");
    assert!(worst < 3e-2, "the gradient is not the loss's derivative");
}

fn read_torch(path: &PathBuf) -> Factors {
    let bytes = std::fs::read(path).unwrap();
    let file = SafeTensors::deserialize(&bytes).unwrap();
    file.tensors()
        .into_iter()
        .map(|(k, t)| {
            let v = t
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                .collect();
            (k, (t.shape().to_vec(), v))
        })
        .collect()
}

/// Cosine and norm ratio over all factors, the worst factor's cosine, and the share of entries
/// whose sign agrees with `b`'s, plain and weighted by `b`'s magnitude (AdamW's first steps move
/// each entry by about lr times its gradient's sign).
fn compare(a: &Factors, b: &Factors) -> (f64, f64, f64, String, f64, f64) {
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "the same factors");
    let cos = dot(a, b) / (norm(a) * norm(b));
    let (mut worst, mut at) = (1.0f64, String::new());
    let (mut same, mut count, mut same_w, mut total_w) = (0usize, 0usize, 0.0f64, 0.0f64);
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
            if *q == 0.0 {
                continue;
            }
            let agree = p.signum() == q.signum();
            count += 1;
            same += usize::from(agree);
            total_w += q.abs();
            if agree {
                same_w += q.abs();
            }
        }
    }
    (cos, norm(a) / norm(b), worst, at, same as f64 / count as f64, same_w / total_w)
}

#[test]
#[ignore = "needs CUDA, the checkpoint, MEMBRANE_GRAD_BATCH, MEMBRANE_ADAPTER, MEMBRANE_GRAD_FP32 and MEMBRANE_GRAD_BF16"]
fn gradient_matches_pytorch() {
    let b = batch();
    let torch32 = read_torch(&env("MEMBRANE_GRAD_FP32"));
    let torch16 = read_torch(&env("MEMBRANE_GRAD_BF16"));
    let rust32 = gradient(&setup(FloatDType::F32), &b);
    let rust16 = gradient(&setup(FloatDType::BF16), &b);
    println!("{} factors; cosine, norm ratio, worst factor's cosine, entries with the same sign (plain / weighted):", rust32.len());
    let rows = [
        ("rust f32 vs pytorch fp32", compare(&rust32, &torch32)),
        ("rust bf16 vs pytorch fp32", compare(&rust16, &torch32)),
        ("pytorch bf16 vs pytorch fp32", compare(&torch16, &torch32)),
        ("rust bf16 vs pytorch bf16", compare(&rust16, &torch16)),
    ];
    for (label, (cos, ratio, worst, at, sign, sign_w)) in &rows {
        println!("  {label:30} {cos:.8}  {ratio:.6}  {worst:.6} ({at})  {sign:.4} / {sign_w:.4}");
    }
    let (c32, r32, w32, ..) = &rows[0].1;
    assert!(*c32 > 0.9999 && (r32 - 1.0).abs() < 1e-3 && *w32 > 0.999, "f32 against PyTorch's fp32");
    let (c16, ..) = &rows[1].1;
    let (t16, ..) = &rows[2].1;
    assert!(1.0 - c16 <= 2.0 * (1.0 - t16) + 1e-4, "bf16 further from fp32 than PyTorch's own bf16");
}

#[test]
#[ignore = "needs CUDA, the checkpoint, MEMBRANE_GRAD_BATCH and MEMBRANE_ADAPTER"]
fn one_update_follows_the_gradient() {
    // nb10's update in bf16 (the gradient over the minibatch's tree, clipped to 0.5, one AdamW
    // step), at three learning rates; each step then measured in f32, against the f32 gradient.
    let b = batch();
    let lrs = [2e-7, 2e-6, 2e-5];
    let (steps, g16) = {
        let s = setup(FloatDType::BF16);
        let (t, reads) = tree_of(&b, &b.groups);
        let g16 = gradient(&s, &b);
        let params = |m: &Qwen3_5ForCausalLM| -> Factors {
            let mut out = Factors::new();
            for (path, layer) in m.projections() {
                let Some(l) = layer.adapter(lora::POLICY) else { continue };
                for (name, p) in [("lora_A", l.a.val()), ("lora_B", l.b.val())] {
                    out.insert(format!("{path}.{name}"), (p.dims().to_vec(), host(p)));
                }
            }
            out
        };
        let theta0 = params(&s.model);
        let objectives: Vec<Option<Objective>> = (0..b.episodes.len())
            .map(|e| {
                let ep = &b.episodes[e];
                (ep.weight > 0.0).then(|| Objective {
                    reads: reads[e].clone(),
                    old: ep.old.clone(),
                    advantages: vec![ep.z; ep.old.len()],
                    weight: ep.weight,
                })
            })
            .collect();
        let mut steps = Vec::new();
        for lr in lrs {
            let valid = s.model.valid();
            let (grads, _) = tree::gradients(
                &s.model, &valid, &s.table, &s.table_ad, &t, &objectives, &[b.suppress], b.clip_eps,
            );
            let mut gp = GradientsParams::from_grads(grads, &s.model);
            let clip = ClipGradNormOptions { max_norm: 0.5, ..ClipGradNormOptions::default() };
            let gnorm = host(clip_grad_norm(&s.model, &mut gp, &clip))[0];
            let mut optim = ModuleOptimizer::from(AdamW {
                beta1: Beta1::new(0.9),
                beta2: 0.999,
                eps: 1e-5,
                weight_decay: 0.0,
            });
            let theta1 = params(&optim.step(lr, s.model.clone(), gp));
            let step: Factors = theta1
                .iter()
                .map(|(k, (d, v))| (k.clone(), (d.clone(), v.iter().zip(&theta0[k].1).map(|(x, y)| x - y).collect())))
                .collect();
            println!("bf16 update at lr {lr:.0e}: grad norm {gnorm:.4}, |step| {:.3e}", norm(&step));
            steps.push(step);
            s.device.memory_cleanup();
        }
        (steps, g16)
    };

    // f32, with the old log-probs this forward takes at the start (every ratio 1 there).
    let mut b = b;
    let s = setup(FloatDType::F32);
    let (t, reads) = tree_of(&b, &b.groups);
    let lp0 = tree::logprobs(&s.model.valid(), &s.table, &t, &reads, &[b.suppress]);
    for (e, lp) in b.episodes.iter_mut().zip(lp0.clone()) {
        e.old = lp;
    }
    let g32 = gradient(&s, &b);
    let l0 = loss_of(&b, &lp0);
    let mut ratios = Vec::new();
    for (lr, step) in lrs.iter().zip(&steps) {
        let after = tree::logprobs(&moved(&s, step, 1.0), &s.table, &t, &reads, &[b.suppress]);
        let change = loss_of(&b, &after) - l0;
        let (true_pred, bf16_pred) = (dot(&g32, step), dot(&g16, step));
        let cos = -true_pred / (norm(&g32) * norm(step));
        println!(
            "lr {lr:.0e}: loss change {change:+.4e}; first order with the f32 gradient {true_pred:+.4e} (ratio {:.3}), with the bf16 one {bf16_pred:+.4e}; cosine of the step with -g {cos:.4}",
            change / true_pred
        );
        let mut agree = 0;
        for (e, ep) in b.episodes.iter().enumerate() {
            let d: f64 = after[e].iter().zip(&lp0[e]).map(|(x, y)| (x - y) as f64).sum::<f64>() / lp0[e].len().max(1) as f64;
            agree += usize::from(d.signum() == (ep.z as f64).signum());
            print!("  {:+.2}:{d:+.1e}", ep.z);
        }
        println!("\n  {agree}/{} episodes' mean log-prob moved the way their advantage points", b.episodes.len());
        assert!(change < 0.0, "the update raised the loss");
        ratios.push(change / true_pred);
        s.device.memory_cleanup();
    }
    assert!((ratios[0] - 1.0).abs() < 0.1, "a small update does not move the loss as the gradient predicts");
}
