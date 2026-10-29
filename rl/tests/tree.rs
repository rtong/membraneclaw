//! The prefix tree's update passes against each episode run whole: the same log-probs, and the
//! same gradient up to the order of floating-point sums.

use std::path::PathBuf;

use burn::module::Module;
use burn::tensor::{DType, Device, FloatDType, Gradients};
use membrane_rl::hf;
use membrane_rl::lora;
use membrane_rl::qwen3_5::{Qwen3_5ForCausalLM, RotaryTable};
use membrane_rl::tree::{self, Objective, Reads, Tree};

const MODEL: &str = "Qwen/Qwen3.5-0.8B";

fn device_and_dtype() -> (Device, FloatDType) {
    let device = match std::env::var("MEMBRANE_DEVICE").as_deref() {
        #[cfg(feature = "cuda")]
        Ok("cuda") => Device::cuda(0),
        Ok("flex") | Err(_) => Device::flex(),
        Ok(other) => panic!("MEMBRANE_DEVICE={other} is not built in"),
    };
    let dtype = match std::env::var("MEMBRANE_DTYPE").as_deref() {
        Ok("bf16") => FloatDType::BF16,
        Ok("f32") | Err(_) => FloatDType::F32,
        Ok(other) => panic!("MEMBRANE_DTYPE={other}"),
    };
    (device, dtype)
}

fn seed_adapter() -> PathBuf {
    std::env::var_os("MEMBRANE_ADAPTER")
        .map(PathBuf::from)
        .expect("set MEMBRANE_ADAPTER to a PEFT adapter directory")
}

/// Every trainable LoRA factor's gradient, flattened in one order.
fn flat(model: &Qwen3_5ForCausalLM, grads: &Gradients) -> Vec<f64> {
    let mut out = Vec::new();
    for (_, layer) in model.projections() {
        if let Some(l) = layer.adapter(lora::POLICY) {
            for p in [l.a.val(), l.b.val()] {
                let n = p.dims().iter().product::<usize>();
                match p.grad(grads) {
                    Some(g) => out.extend(
                        g.cast(FloatDType::F32)
                            .into_data()
                            .convert::<f32>()
                            .try_into_vec::<f32>()
                            .unwrap()
                            .into_iter()
                            .map(|x| x as f64),
                    ),
                    None => out.extend(std::iter::repeat_n(0.0, n)),
                }
            }
        }
    }
    out
}

fn cosine(a: &[f64], b: &[f64]) -> (f64, f64) {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    (dot / (na * nb), na / nb)
}

#[test]
#[ignore = "needs the checkpoint in the Hugging Face cache and MEMBRANE_ADAPTER"]
fn tree_matches_whole_episodes() {
    let (device, dtype) = device_and_dtype();
    let ad = device.clone().autodiff();
    let dir = hf::snapshot(MODEL).unwrap();
    let (mut model, config) = Qwen3_5ForCausalLM::from_pretrained(&dir, dtype, &ad).unwrap();
    lora::load(&mut model, &seed_adapter(), lora::POLICY, true, &ad).unwrap();
    let model = lora::choose(model, lora::POLICY);
    let valid = model.valid();
    let table = RotaryTable::new(&config, 4096, DType::from(dtype), &device);
    let table_ad = RotaryTable::new(&config, 4096, DType::from(dtype), &ad);

    // Four episodes in two groups: a shared prefix, a group prefix, their own tokens.
    let tok = membrane_rl::chat::Tokenizer::from_dir(&dir).unwrap();
    let text = "The miller had moved his sacks to the loft on Tuesday, and the river rose on \
        Wednesday. Nobody in the village was surprised that the lower field flooded; it \
        floods every spring. What did surprise them was the heron that stood all morning on \
        the mill wheel, as if it had been waiting for the water. ";
    let words: Vec<u32> = tok.encode(&text.repeat(20)).unwrap();
    let root = &words[..300];
    let mut seqs = Vec::new();
    for g in 0..2 {
        for e in 0..2 {
            let mut s = root.to_vec();
            s.extend_from_slice(&words[400 + 50 * g..460 + 50 * g]);
            let off = 600 + 97 * (2 * g + e);
            s.extend_from_slice(&words[off..off + 150 + 20 * e]);
            seqs.push(s);
        }
    }
    let groups = vec![vec![0, 1], vec![2, 3]];
    // Each episode reads its last 100 positions.
    let limit: Vec<usize> = seqs.iter().map(|s| s.len() - 101).collect();
    let reads: Vec<Reads> = seqs
        .iter()
        .map(|s| {
            let n = s.len();
            (n - 101..n - 1).fold(Reads::default(), |mut r, p| {
                r.at.push(p);
                r.targets.push(s[p + 1]);
                r
            })
        })
        .collect();
    let with_tree = Tree::build(&seqs, &limit, &groups, 16);
    let whole = Tree::build(&seqs, &limit, &[vec![0], vec![1], vec![2], vec![3]], usize::MAX);
    println!(
        "tree: {} segments, {} tokens; whole: {} segments, {} tokens",
        with_tree.segments.len(),
        with_tree.tokens(),
        whole.segments.len(),
        whole.tokens()
    );
    assert_eq!(with_tree.segments.len(), 1 + 2 + 4);

    // Reads relative to each episode's own segment.
    let local = |t: &Tree| -> Vec<Reads> {
        reads
            .iter()
            .enumerate()
            .map(|(e, r)| {
                let start = t.segments[t.leaf[e]].start;
                Reads {
                    at: r.at.iter().map(|&p| p - start).collect(),
                    targets: r.targets.clone(),
                }
            })
            .collect()
    };
    let suppress = [248044u32];
    let lp_tree = tree::logprobs(&valid, &table, &with_tree, &local(&with_tree), &suppress);
    let lp_whole = tree::logprobs(&valid, &table, &whole, &local(&whole), &suppress);
    let worst = lp_tree
        .iter()
        .flatten()
        .zip(lp_whole.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("log-probs: tree against whole, max |diff| {worst:.3e}");

    let objectives = |t: &Tree| -> Vec<Option<Objective>> {
        local(t)
            .into_iter()
            .enumerate()
            .map(|(e, r)| {
                let m = r.at.len();
                Some(Objective {
                    old: lp_whole[e].iter().map(|x| x - 0.05).collect(),
                    advantages: (0..m).map(|k| if (k + e) % 3 == 0 { 1.0 } else { -0.5 }).collect(),
                    weight: 1.0 / (m as f32 * 4.0),
                    reads: r,
                })
            })
            .collect()
    };
    let (g_tree, s_tree) = tree::gradients(
        &model, &valid, &table, &table_ad, &with_tree, &objectives(&with_tree), &suppress, 0.2,
    );
    let (g_whole, s_whole) = tree::gradients(
        &model, &valid, &table, &table_ad, &whole, &objectives(&whole), &suppress, 0.2,
    );
    for (a, b) in s_tree.iter().zip(&s_whole) {
        println!("loss {:.6} / {:.6}  ratio {:.6} / {:.6}", a.loss, b.loss, a.ratio_mean, b.ratio_mean);
    }
    let (a, b) = (flat(&model, &g_tree), flat(&model, &g_whole));
    let (cos, ratio) = cosine(&a, &b);
    println!("gradient: cosine {cos:.8}, norm ratio {ratio:.6} over {} entries", a.len());
    let tol = if dtype == FloatDType::F32 { 0.9999 } else { 0.98 };
    assert!(cos >= tol, "cosine {cos}");
    assert!(worst < if dtype == FloatDType::F32 { 1e-3 } else { 0.2 });
}
