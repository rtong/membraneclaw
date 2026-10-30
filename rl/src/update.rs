//! One GRPO step's update after its rollout, as nb10 runs it: the old and the reference
//! log-probs of every policy token, the returns and the group advantages ([`before`]), then
//! `inner_epochs` passes over minibatches of whole groups, each PPO's clipped loss on the
//! minibatch's actor tokens, its gradient clipped and one optimizer step ([`step`]).
//! tests/step.rs holds a whole step to PyTorch's.

use std::time::Instant;

use burn::module::Module;
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::tensor::Device;
use dash_train::clip::{ClipGradNormOptions, clip_grad_norm};
use dash_train::optim::{AdamW, Beta1};
use serde::{Deserialize, Serialize};

use crate::grpo::{self, Prepared, Rng};
use crate::lora;
use crate::qwen3_5::{Qwen3_5ForCausalLM, RotaryTable};
use crate::tree::{self, EpisodeStats, Objective, Reads, Tree};

/// The update's settings, under the names a run's config.json gives them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub lr: f64,
    pub clip_eps: f32,
    pub inner_epochs: usize,
    pub minibatches: usize,
    pub max_grad_norm: f64,
    pub adam_betas: [f64; 2],
    pub adam_eps: f64,
    pub weight_decay: f64,
    pub kl_coef: f64,
    pub seed: u64,
}

impl Settings {
    /// AdamW under these settings, its moments empty.
    pub fn optimizer(&self) -> ModuleOptimizer {
        ModuleOptimizer::from(AdamW {
            beta1: Beta1::new(self.adam_betas[0]),
            beta2: self.adam_betas[1],
            eps: self.adam_eps,
            weight_decay: self.weight_decay,
        })
    }
}

/// A step's episodes as the update reads them.
pub struct Batch<'a> {
    /// Each episode's prompt and tokens.
    pub seqs: &'a [Vec<u32>],
    pub prepared: &'a [Prepared],
    /// Each episode's score (err0's total).
    pub scores: &'a [f64],
    /// The step's groups of episodes.
    pub groups: &'a [Vec<usize>],
}

impl Batch<'_> {
    /// Which of episode `e`'s policy tokens the actor's loss takes.
    pub fn actor(&self, e: usize) -> Vec<usize> {
        let p = &self.prepared[e];
        (0..p.actor.len()).filter(|&k| p.actor[k]).collect()
    }

    /// Each episode's place in the prefix tree may end at its first read.
    pub fn limit(&self) -> Vec<usize> {
        self.prepared
            .iter()
            .zip(self.seqs)
            .map(|(p, s)| p.reads.at.iter().copied().min().unwrap_or(s.len() - 1))
            .collect()
    }

    /// Each episode's reads `which(e)` in the coordinates of its leaf segment of `t`.
    fn local(&self, t: &Tree, which: impl Fn(usize) -> Vec<usize>) -> Vec<Reads> {
        (0..self.seqs.len())
            .map(|e| {
                if t.leaf[e] == usize::MAX {
                    return Reads::default();
                }
                let start = t.segments[t.leaf[e]].start;
                let p = &self.prepared[e];
                let mut r = Reads::default();
                for k in which(e) {
                    r.at.push(p.reads.at[k] - start);
                    r.targets.push(p.reads.targets[k]);
                }
                r
            })
            .collect()
    }
}

/// What the update starts from.
pub struct Before {
    /// Every policy token's log-prob under the policy before the update.
    pub old: Vec<Vec<f32>>,
    /// Each episode's summed k1 KL to the reference adapter.
    pub kl: Vec<f64>,
    pub returns: Vec<f64>,
    pub advantages: Vec<f64>,
    /// The tokens of the tree over every group.
    pub tree_tokens: usize,
}

/// The old and the reference log-probs over one tree of every group (`valid`: the policy without
/// autodiff, the reference adapter on it), the returns and the group advantages.
pub fn before(valid: &Qwen3_5ForCausalLM, table: &RotaryTable, b: &Batch, suppress: &[u32], kl_coef: f64) -> Before {
    assert!(b.seqs.iter().all(|s| s.len() < table.len()), "an episode is longer than the rotary table");
    let tree_all = Tree::build(b.seqs, &b.limit(), b.groups, 64);
    let reads_all = b.local(&tree_all, |e| (0..b.prepared[e].reads.at.len()).collect());
    let old = tree::logprobs(valid, table, &tree_all, &reads_all, suppress);
    let reference = lora::choose(valid.clone(), lora::REFERENCE);
    let refl = tree::logprobs(&reference, table, &tree_all, &reads_all, suppress);
    drop(reference);
    let kl: Vec<f64> = old.iter().zip(&refl).map(|(a, b)| grpo::kl_sum(a, b)).collect();
    let returns: Vec<f64> = b.scores.iter().zip(&kl).map(|(s, k)| s - kl_coef * k).collect();
    let advantages = grpo::group_advantages(&returns, b.groups);
    Before { old, kl, returns, advantages, tree_tokens: tree_all.tokens() }
}

/// The order epoch `epoch` of step `step` takes the groups in.
pub fn order(seed: u64, step: usize, epoch: usize, groups: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..groups).collect();
    Rng::new(seed ^ ((step as u64) << 20) ^ ((epoch as u64) << 8) ^ 0x5eed).shuffle(&mut order);
    order
}

/// One minibatch's update: its episodes' means before its step.
#[derive(Debug, Clone)]
pub struct Minibatch {
    pub epoch: usize,
    pub groups: Vec<usize>,
    pub approx_kl: f64,
    pub ratio: f64,
    /// The gradient's norm before clipping.
    pub grad_norm: f32,
    pub tree_tokens: usize,
    /// Since the update's first minibatch began.
    pub seconds: f64,
    /// The device's pool after the step.
    pub reserved_gb: f64,
}

pub struct Outcome {
    pub before: Before,
    /// Each episode's actor tokens.
    pub counts: Vec<usize>,
    pub minibatches: Vec<Minibatch>,
    /// The last epoch's per-episode statistics.
    pub last_epoch: Vec<EpisodeStats>,
    pub before_seconds: f64,
    pub update_seconds: f64,
}

fn mean(v: impl IntoIterator<Item = f64>) -> f64 {
    let (s, n) = v.into_iter().fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 { 0.0 } else { s / n as f64 }
}

fn reserved_gb(device: &Device) -> f64 {
    device
        .memory_pool_usage()
        .map_or(0.0, |u| u.bytes_reserved as f64 / 2f64.powi(30))
}

/// Step `step`'s update of `model` (on an autodiff device, the policy adapter chosen and
/// trainable, the reference adapter beside it) on its episodes `b`: [`before`], then for each of
/// `inner_epochs` epochs the groups in [`order`], `minibatches` minibatches of them; each
/// minibatch's advantages z-scored over its actor tokens, each episode's loss weighted one over
/// its actor tokens times the minibatch's episodes, the gradient over the minibatch's tree
/// clipped to `max_grad_norm` and one step of `optim`.
#[allow(clippy::too_many_arguments)]
pub fn step(
    mut model: Qwen3_5ForCausalLM,
    optim: &mut ModuleOptimizer,
    s: &Settings,
    table: &RotaryTable,
    table_ad: &RotaryTable,
    b: &Batch,
    suppress: &[u32],
    step: usize,
) -> (Qwen3_5ForCausalLM, Outcome) {
    let t1 = Instant::now();
    let mut valid = model.valid();
    let device = valid.model.norm.weight.device();
    let start = before(&valid, table, b, suppress, s.kl_coef);
    let before_seconds = t1.elapsed().as_secs_f64();
    device.memory_cleanup();

    let t2 = Instant::now();
    let clip = ClipGradNormOptions {
        max_norm: s.max_grad_norm,
        ..ClipGradNormOptions::default()
    };
    let limit = b.limit();
    let counts: Vec<usize> = (0..b.seqs.len()).map(|e| b.actor(e).len()).collect();
    let per_mb = b.groups.len() / s.minibatches;
    let mut minibatches = Vec::new();
    let mut last_epoch = Vec::new();
    for epoch in 0..s.inner_epochs {
        let mut epoch_stats = Vec::new();
        for mb in order(s.seed, step, epoch, b.groups.len()).chunks(per_mb) {
            let mb_groups: Vec<Vec<usize>> = mb.iter().map(|&g| b.groups[g].clone()).collect();
            let members: Vec<usize> = mb_groups.iter().flatten().copied().collect();
            let z = grpo::normalized(
                &members.iter().map(|&e| start.advantages[e]).collect::<Vec<_>>(),
                &members.iter().map(|&e| counts[e]).collect::<Vec<_>>(),
            );
            let t = Tree::build(b.seqs, &limit, &mb_groups, 64);
            let reads = b.local(&t, |e| b.actor(e));
            let mut objectives: Vec<Option<Objective>> = vec![None; b.seqs.len()];
            for (j, &e) in members.iter().enumerate() {
                if counts[e] == 0 {
                    continue;
                }
                let ks = b.actor(e);
                objectives[e] = Some(Objective {
                    old: ks.iter().map(|&k| start.old[e][k]).collect(),
                    advantages: vec![z[j]; ks.len()],
                    weight: 1.0 / (counts[e] as f32 * members.len() as f32),
                    reads: reads[e].clone(),
                });
            }
            let (grads, stats) = tree::gradients(&model, &valid, table, table_ad, &t, &objectives, suppress, s.clip_eps);
            let mut gp = GradientsParams::from_grads(grads, &model);
            let norm = clip_grad_norm(&model, &mut gp, &clip)
                .into_data()
                .convert::<f32>()
                .try_into_vec::<f32>()
                .unwrap()[0];
            model = optim.step(s.lr, model, gp);
            valid = model.valid();
            let used: Vec<&EpisodeStats> = members.iter().filter(|&&e| counts[e] > 0).map(|&e| &stats[e]).collect();
            epoch_stats.extend(used.iter().map(|s| (*s).clone()));
            minibatches.push(Minibatch {
                epoch,
                groups: mb.to_vec(),
                approx_kl: mean(used.iter().map(|s| s.approx_kl)),
                ratio: mean(used.iter().map(|s| s.ratio_mean)),
                grad_norm: norm,
                tree_tokens: t.tokens(),
                seconds: t2.elapsed().as_secs_f64(),
                reserved_gb: reserved_gb(&device),
            });
        }
        last_epoch = epoch_stats;
    }
    let outcome = Outcome {
        before: start,
        counts,
        minibatches,
        last_epoch,
        before_seconds,
        update_seconds: t2.elapsed().as_secs_f64(),
    };
    (model, outcome)
}
