//! The update's passes over a prefix tree: episodes that begin with the same tokens run those
//! tokens once. A step's 32 episodes share the system turn and the tool declarations; the four
//! of a group share their question (and, continuing an error state, everything up to that
//! state). Each episode's own tokens continue its parent's state.
//!
//! The backward pass is exact: layer by layer from the top, each segment runs its layer again
//! from the input kept by the forward pass, and backpropagates what reaches its output and its
//! final state. A child's gradient with respect to the state it started from is its parent's
//! gradient with respect to the state it ended in; those are summed (f32) over the children
//! before the parent runs its own backward. Episodes' gradients differ from running each
//! episode whole only in the order floating-point sums are taken.

use burn::tensor::{FloatDType, Gradients, Int, Tensor, TensorData};
use dash_ops::checkpoint::backward_through;
use dash_ops::logprob::{LogprobOptions, chosen_logprob};

use crate::qwen3_5::{DeltaState, LayerState, Qwen3_5ForCausalLM, RotaryTable, State};

/// One run of tokens: the root, a group's prefix, or an episode's own tokens.
#[derive(Debug, Clone)]
pub struct Segment {
    pub tokens: Vec<u32>,
    /// The position of its first token.
    pub start: usize,
    pub parent: Option<usize>,
}

/// Segments in an order where every parent comes before its children; `leaf[e]` is episode
/// `e`'s own segment.
#[derive(Debug, Clone)]
pub struct Tree {
    pub segments: Vec<Segment>,
    pub leaf: Vec<usize>,
}

fn common(seqs: &[&[u32]]) -> usize {
    let first = seqs[0];
    seqs[1..].iter().fold(first.len(), |n, s| {
        n.min(s.iter().zip(first).take_while(|(a, b)| a == b).count())
    })
}

impl Tree {
    /// The tree over `seqs` (each episode's prompt and tokens), grouped by `groups` (indices into
    /// `seqs`). `limit[e]` is the longest prefix episode `e` may share: up to the position whose
    /// hidden state predicts its first read token, which its own segment must hold. A shared
    /// prefix shorter than `min` tokens is not split off.
    pub fn build(seqs: &[Vec<u32>], limit: &[usize], groups: &[Vec<usize>], min: usize) -> Self {
        let all: Vec<usize> = groups.iter().flatten().copied().collect();
        let cut = |members: &[usize]| {
            let s: Vec<&[u32]> = members.iter().map(|&e| seqs[e].as_slice()).collect();
            members.iter().fold(common(&s), |n, &e| n.min(limit[e]))
        };
        let mut segments = Vec::new();
        let mut leaf = vec![usize::MAX; seqs.len()];
        let root_cut = if all.len() > 1 { cut(&all) } else { 0 };
        let root = (root_cut >= min).then(|| {
            segments.push(Segment {
                tokens: seqs[all[0]][..root_cut].to_vec(),
                start: 0,
                parent: None,
            });
            0
        });
        let base = root.map_or(0, |_| root_cut);
        let mut pending = Vec::new();
        for g in groups {
            let group_cut = if g.len() > 1 { cut(g) } else { 0 };
            let node = (group_cut >= base + min).then(|| {
                segments.push(Segment {
                    tokens: seqs[g[0]][base..group_cut].to_vec(),
                    start: base,
                    parent: root,
                });
                segments.len() - 1
            });
            let from = node.map_or(base, |_| group_cut);
            for &e in g {
                pending.push((e, from, node.or(root)));
            }
        }
        for (e, from, parent) in pending {
            leaf[e] = segments.len();
            segments.push(Segment {
                tokens: seqs[e][from..].to_vec(),
                start: from,
                parent,
            });
        }
        Self { segments, leaf }
    }

    pub fn tokens(&self) -> usize {
        self.segments.iter().map(|s| s.tokens.len()).sum()
    }

    fn children(&self, i: usize) -> usize {
        self.segments
            .iter()
            .filter(|s| s.parent == Some(i))
            .count()
    }
}

/// What an episode's segment reads: the hidden state at each of `at` (its own positions)
/// predicts `targets`.
#[derive(Debug, Clone, Default)]
pub struct Reads {
    pub at: Vec<usize>,
    pub targets: Vec<u32>,
}

/// `ids` `[1, n]`.
fn ids_tensor(tokens: &[u32], device: &burn::tensor::Device) -> Tensor<2, Int> {
    let v: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
    Tensor::from_data(TensorData::new(v, [1, tokens.len()]), device)
}

fn index_tensor(at: &[usize], device: &burn::tensor::Device) -> Tensor<1, Int> {
    let v: Vec<i64> = at.iter().map(|&t| t as i64).collect();
    Tensor::from_data(TensorData::new(v, [at.len()]), device)
}

/// The log-probs of every episode's reads under `model` (no autodiff), the suppressed tokens out
/// of the vocabulary: one forward pass over the tree.
pub fn logprobs(
    model: &Qwen3_5ForCausalLM,
    table: &RotaryTable,
    tree: &Tree,
    reads: &[Reads],
    suppress: &[u32],
) -> Vec<Vec<f32>> {
    let device = model.model.norm.weight.device();
    let mut states: Vec<Option<State>> = vec![None; tree.segments.len()];
    let mut left: Vec<usize> = (0..tree.segments.len()).map(|i| tree.children(i)).collect();
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); reads.len()];
    let episode_of: std::collections::HashMap<usize, usize> =
        tree.leaf.iter().enumerate().map(|(e, &s)| (s, e)).collect();
    for (i, seg) in tree.segments.iter().enumerate() {
        let mut state = match seg.parent {
            Some(p) => {
                left[p] -= 1;
                let s = if left[p] == 0 {
                    states[p].take()
                } else {
                    states[p].clone()
                };
                s.expect("parents run first")
            }
            None => model.empty_state(),
        };
        debug_assert_eq!(state.seen, seg.start);
        let hidden = model.forward(ids_tensor(&seg.tokens, &device), table, &mut state);
        match episode_of.get(&i) {
            Some(&e) => {
                let r = &reads[e];
                if r.at.is_empty() {
                    continue;
                }
                let [_, n, d] = hidden.dims();
                let h = hidden
                    .reshape([n, d])
                    .select(0, index_tensor(&r.at, &device));
                let targets = index_tensor(
                    &r.targets.iter().map(|&t| t as usize).collect::<Vec<_>>(),
                    &device,
                );
                let lp = chosen_logprob(
                    h,
                    model.head_weight(),
                    targets,
                    suppress,
                    LogprobOptions::default(),
                );
                out[e] = lp
                    .logprob
                    .into_data()
                    .convert::<f32>()
                    .try_into_vec::<f32>()
                    .unwrap();
            }
            None => states[i] = Some(state),
        }
    }
    out
}

/// An episode's part of a minibatch's loss: PPO's clipped surrogate on its actor tokens, the
/// token mean, times `weight` (one over the minibatch's episodes).
#[derive(Debug, Clone)]
pub struct Objective {
    pub reads: Reads,
    pub old: Vec<f32>,
    pub advantages: Vec<f32>,
    pub weight: f32,
}

/// Per-episode statistics of the update's first pass through it (before this minibatch's step).
#[derive(Debug, Clone, Default)]
pub struct EpisodeStats {
    pub loss: f64,
    pub ratio_mean: f64,
    pub clip_frac: f64,
    pub approx_kl: f64,
}

/// PPO's clipped surrogate for one episode: `-Σ min(r A, clip(r, 1 ± eps) A) · weight` over its
/// read tokens, `r = exp(lp - old)` (`lp` on the device, tracked when its gradient is wanted).
/// Returns the loss and the episode's statistics (token means).
pub fn ppo_loss(
    lp: Tensor<1>,
    old: &[f32],
    advantages: &[f32],
    weight: f32,
    clip_eps: f32,
) -> (Tensor<1>, EpisodeStats) {
    let m = old.len();
    let device = lp.device();
    let constant = |v: &[f32]| {
        let t = Tensor::<1>::from_data(TensorData::new(v.to_vec(), [m]), &device.clone().inner());
        if device.is_autodiff() { t.autodiff() } else { t }
    };
    let logratio = lp - constant(old);
    let ratio = logratio.clone().exp();
    let adv = constant(advantages);
    let unclipped = ratio.clone() * adv.clone();
    let clipped = ratio.clone().clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv;
    let surrogate = unclipped.clone().min_pair(clipped.clone());
    let loss = surrogate.clone().sum().neg().mul_scalar(weight);
    let n = m.max(1) as f64;
    let host = |t: Tensor<1>| -> f64 {
        t.inner().into_data().convert::<f32>().try_into_vec::<f32>().unwrap()[0] as f64
    };
    let stats = EpisodeStats {
        loss: -host(surrogate.sum()) / n,
        ratio_mean: host(ratio.clone().sum()) / n,
        clip_frac: host(unclipped.greater(clipped).float().sum()) / n,
        approx_kl: host((ratio.sub_scalar(1.0) - logratio).sum()) / n,
    };
    (loss, stats)
}

/// A layer's kept input and the state it ended in.
struct Kept {
    inputs: Vec<Tensor<3>>,
    /// Non-leaf segments: the state after each layer.
    states: Vec<Option<LayerState>>,
}

fn lift_state(s: &LayerState, grad: bool) -> LayerState {
    let lift = |t: &Tensor<4>| {
        let t = t.clone().autodiff();
        if grad { t.require_grad() } else { t }
    };
    let lift3 = |t: &Tensor<3>| {
        let t = t.clone().autodiff();
        if grad { t.require_grad() } else { t }
    };
    match s {
        LayerState::Delta(d) => LayerState::Delta(DeltaState {
            conv: lift3(&d.conv),
            recurrent: lift(&d.recurrent),
        }),
        LayerState::Attention { k, v } => LayerState::Attention {
            k: lift(k),
            v: lift(v),
        },
    }
}

/// The f32 gradient with respect to a state, as a sum over children.
#[derive(Clone)]
enum StateGrad {
    Delta { conv: Tensor<3>, recurrent: Tensor<4> },
    Attention { k: Tensor<4>, v: Tensor<4> },
}

impl StateGrad {
    fn add(self, other: StateGrad) -> StateGrad {
        match (self, other) {
            (StateGrad::Delta { conv, recurrent }, StateGrad::Delta { conv: c, recurrent: r }) => {
                StateGrad::Delta {
                    conv: conv + c,
                    recurrent: recurrent + r,
                }
            }
            (StateGrad::Attention { k, v }, StateGrad::Attention { k: k2, v: v2 }) => {
                StateGrad::Attention { k: k + k2, v: v + v2 }
            }
            _ => unreachable!("one layer's states are one kind"),
        }
    }

    /// `Σ state ⊙ grad`, the term that backpropagates `grad` into `state`.
    fn pair(&self, state: &LayerState) -> Tensor<1> {
        let f = FloatDType::F32;
        match (self, state) {
            (StateGrad::Delta { conv, recurrent }, LayerState::Delta(s)) => {
                (s.conv.clone().cast(f) * conv.clone().autodiff()).sum()
                    + (s.recurrent.clone().cast(f) * recurrent.clone().autodiff()).sum()
            }
            (StateGrad::Attention { k, v }, LayerState::Attention { k: sk, v: sv }) => {
                (sk.clone().cast(f) * k.clone().autodiff()).sum()
                    + (sv.clone().cast(f) * v.clone().autodiff()).sum()
            }
            _ => unreachable!("one layer's states are one kind"),
        }
    }

    /// The gradient a backward pass left on a lifted state, without autodiff, f32.
    fn of(state: &LayerState, grads: &Gradients) -> Option<StateGrad> {
        // A part no gradient reached contributes zeros.
        let f = FloatDType::F32;
        let g4 = |t: &Tensor<4>| t.grad(grads).map(|g| g.cast(f));
        let g3 = |t: &Tensor<3>| t.grad(grads).map(|g| g.cast(f));
        let z4 = |t: &Tensor<4>| t.clone().inner().cast(f).zeros_like();
        let z3 = |t: &Tensor<3>| t.clone().inner().cast(f).zeros_like();
        match state {
            LayerState::Delta(s) => {
                let (c, r) = (g3(&s.conv), g4(&s.recurrent));
                (c.is_some() || r.is_some()).then(|| StateGrad::Delta {
                    conv: c.unwrap_or_else(|| z3(&s.conv)),
                    recurrent: r.unwrap_or_else(|| z4(&s.recurrent)),
                })
            }
            LayerState::Attention { k, v } => {
                let (gk, gv) = (g4(k), g4(v));
                (gk.is_some() || gv.is_some()).then(|| StateGrad::Attention {
                    k: gk.unwrap_or_else(|| z4(k)),
                    v: gv.unwrap_or_else(|| z4(v)),
                })
            }
        }
    }
}

/// The gradient of a minibatch's loss (the sum of its episodes' objectives) with respect to the
/// trainable parameters of `model` (on an autodiff device), over `tree`; `objectives[e]` for each
/// episode of the tree (`None`: no part in the loss). `valid` is the same model without autodiff,
/// `table_ad` the rotary table on the autodiff device.
#[allow(clippy::too_many_arguments)]
pub fn gradients(
    model: &Qwen3_5ForCausalLM,
    valid: &Qwen3_5ForCausalLM,
    table: &RotaryTable,
    table_ad: &RotaryTable,
    tree: &Tree,
    objectives: &[Option<Objective>],
    suppress: &[u32],
    clip_eps: f32,
) -> (Gradients, Vec<EpisodeStats>) {
    let device = valid.model.norm.weight.device();
    let layers = valid.model.layers.len();
    let n_seg = tree.segments.len();
    let episode_of: std::collections::HashMap<usize, usize> =
        tree.leaf.iter().enumerate().map(|(e, &s)| (s, e)).collect();
    let is_leaf = |i: usize| episode_of.contains_key(&i);

    // Forward, no autodiff: every segment's input to every layer, and (non-leaves) the state
    // after it; the leaves' top hidden state (before the final norm).
    let mut kept: Vec<Option<Kept>> = (0..n_seg).map(|_| None).collect();
    let mut top: Vec<Option<Tensor<3>>> = vec![None; n_seg];
    for (i, seg) in tree.segments.iter().enumerate() {
        if is_leaf(i) && objectives[episode_of[&i]].is_none() {
            continue;
        }
        let rotary = table.range(seg.start, seg.tokens.len());
        let mut x = valid.embed(ids_tensor(&seg.tokens, &device));
        let mut inputs = Vec::with_capacity(layers);
        let mut states = Vec::with_capacity(layers);
        for (l, layer) in valid.model.layers.iter().enumerate() {
            inputs.push(x.clone());
            let past = seg
                .parent
                .map(|p| kept[p].as_ref().unwrap().states[l].clone().unwrap());
            let (y, s) = layer.forward(x, &rotary, past, !is_leaf(i));
            x = y;
            states.push(if is_leaf(i) { None } else { s });
        }
        if is_leaf(i) {
            top[i] = Some(x);
        }
        kept[i] = Some(Kept { inputs, states });
    }

    // Burn makes a `Gradients` only by a backward pass: start from an empty one's.
    let mut grads = Tensor::<1>::zeros([1], &device)
        .autodiff()
        .require_grad()
        .sum()
        .backward();
    let mut stats = vec![EpisodeStats::default(); objectives.len()];
    // The gradient with respect to each non-leaf segment's state after each layer, summed over
    // its children.
    let mut state_grads: Vec<Vec<Option<StateGrad>>> = (0..n_seg).map(|_| vec![None; layers]).collect();
    let one = || Tensor::<1>::ones([1], &device).autodiff();
    // Children before parents: the reverse of the tree's order.
    for i in (0..n_seg).rev() {
        let Some(k) = kept[i].take() else { continue };
        // A parent's kept states stay until its own turn; its children (later in the order)
        // have all run by then.
        let seg = &tree.segments[i];
        // The gradient with respect to this segment's output of the layer being run.
        let mut dy: Option<Tensor<3>> = None;
        if let Some(&e) = episode_of.get(&i) {
            let o = objectives[e].as_ref().unwrap();
            let x = top[i].take().unwrap().autodiff().require_grad();
            let h = model.model.norm.forward(x.clone());
            let [_, n, d] = h.dims();
            let at = index_tensor(&o.reads.at, &device);
            let h = h.reshape([n, d]).select(0, at.autodiff());
            let targets = index_tensor(
                &o.reads.targets.iter().map(|&t| t as usize).collect::<Vec<_>>(),
                &device,
            );
            let lp = chosen_logprob(
                h,
                model.head_weight(),
                targets.autodiff(),
                suppress,
                LogprobOptions::default(),
            )
            .logprob;
            let (loss, st) = ppo_loss(lp, &o.old, &o.advantages, o.weight, clip_eps);
            stats[e] = st;
            let rest = backward_through(loss, one(), model, &mut grads);
            dy = Some(x.grad(&rest).expect("the loss reads the top").cast(FloatDType::F32));
        }
        let rotary = table_ad.range(seg.start, seg.tokens.len());
        for (l, input) in k.inputs.into_iter().enumerate().rev() {
            let layer = &model.model.layers[l];
            let need_dx = l > 0;
            let x = if need_dx {
                input.autodiff().require_grad()
            } else {
                input.autodiff()
            };
            let past = seg.parent.map(|p| {
                let s = kept[p].as_ref().expect("parents run after children").states[l]
                    .as_ref()
                    .expect("a prefix keeps its states");
                lift_state(s, true)
            });
            let ds = state_grads[i][l].take();
            let (y, s) = layer.forward(x.clone(), &rotary, past.clone(), ds.is_some());
            let mut objective: Option<Tensor<1>> = dy
                .take()
                .map(|g| (y.clone().cast(FloatDType::F32) * g.autodiff()).sum());
            if let (Some(ds), Some(s)) = (&ds, &s) {
                let term = ds.pair(s);
                objective = Some(match objective {
                    Some(o) => o + term,
                    None => term,
                });
            }
            let Some(objective) = objective else {
                // Nothing reaches this layer of this segment (a prefix's top layers' outputs).
                continue;
            };
            let rest = backward_through(objective, one(), layer, &mut grads);
            if need_dx {
                dy = x.grad(&rest).map(|g| g.cast(FloatDType::F32));
            }
            if let (Some(p), Some(past)) = (seg.parent, &past) {
                if let Some(g) = StateGrad::of(past, &rest) {
                    let slot = &mut state_grads[p][l];
                    *slot = Some(match slot.take() {
                        Some(before) => before.add(g),
                        None => g,
                    });
                }
            }
        }
    }
    (grads, stats)
}
