//! Qwen3.5's text model — transformers' `Qwen3_5ForCausalLM` — as Burn modules, named so
//! that the published checkpoint loads unchanged (less the `model.language_model.` prefix
//! of the multimodal checkpoint, and its vision and multi-token-prediction weights, which
//! the text model does not use). One parameter is renamed: the checkpoint's `A_log` is
//! `a_log` here.
//!
//! Three of every four decoder layers mix tokens with a gated delta rule — a linear
//! recurrence whose state is one matrix per head — and the fourth with full attention over
//! a key/value cache. The operators are dashmag's (the delta rule chunked and one step at a
//! time, attention for training and against a cache, the causal convolution, LoRA); the
//! assembly, the states a sequence carries and the batch a decode step runs over are here.
//!
//! Layout: activations `[B, T, C]`, heads `[B, T, H, D]`, as dashmag's operators take them.
//! Recurrent and convolution states are f32; keys and values are kept in the model's dtype.
//!
//! Constants that pass every shape check and fail on numbers, fixed as transformers has them:
//! - RMSNorm is computed in f32 and scaled by `1 + weight`: weights are stored as offsets from
//!   one. The norm closing a delta-rule block is the exception: scaled by `weight`, then gated
//!   by `silu(z)` in f32.
//! - Rotary embedding turns only the first `head_dim * partial_rotary_factor` features of each
//!   attention head (64 of 256), by halves (`cat(-x2, x1)`), not interleaved pairs; theta 1e7.
//! - The attention query projection is twice as wide as the heads: per head, the query and
//!   then a gate, and the attention output is multiplied by `sigmoid(gate)` before `o_proj`.
//! - Queries and keys get RMSNorm per head before the rotation.
//! - The depthwise convolution before the delta rule is causal — each step sees itself and
//!   the three before it — and followed by SiLU.
//! - The decay is `g = -exp(A_log) * softplus(a + dt_bias)`, in f32.
//! - q and k are L2-normalised inside the delta rule (eps 1e-6 on the sum of squares) and q
//!   scaled by `Dk^-1/2`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::{Embedding, EmbeddingConfig};
use burn::store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use burn::tensor::activation::{sigmoid, silu, softplus};
use burn::tensor::{DType, Device, FloatDType, IndexingUpdateOp, Int, Tensor, TensorData};
use dash_lora::LoraLinear;
use dash_ops::attention::{AttentionMask, AttentionOptions, attention, attention_decode};
use dash_ops::conv1d::{
    Activation, CausalConv1dOptions, CausalConv1dStepOptions, causal_conv1d, causal_conv1d_update,
};
use dash_ops::gated_delta::chunk::{ChunkGatedDeltaOptions, chunk_gated_delta_rule};
use dash_ops::gated_delta::recurrent::{GatedDeltaStepOptions, gated_delta_step};
use serde::Deserialize;

/// The text model's `config.json` entries this implementation reads.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub layer_types: Vec<LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_type: String,
    pub rope_theta: f64,
    #[serde(default = "one")]
    pub partial_rotary_factor: f64,
}

fn one() -> f64 {
    1.0
}

impl Config {
    /// A checkpoint's `config.json`; a multimodal checkpoint holds the text model's under
    /// `text_config`.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let text_config = json.get("text_config").unwrap_or(&json);
        let config: Config =
            serde_json::from_value(text_config.clone()).map_err(|e| e.to_string())?;
        config.check()?;
        Ok(config)
    }

    fn check(&self) -> Result<(), String> {
        // The multimodal rotary embedding interleaves three position streams; for text the
        // three are equal and it is the plain rotation implemented here.
        if self.rope_parameters.rope_type != "default" {
            return Err(format!(
                "rope_type {} is not implemented",
                self.rope_parameters.rope_type
            ));
        }
        if !self.attn_output_gate {
            return Err("attention without the output gate is not implemented".into());
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err("layer_types does not name every layer".into());
        }
        if !self
            .linear_num_value_heads
            .is_multiple_of(self.linear_num_key_heads)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err("head counts do not divide".into());
        }
        Ok(())
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize
    }
}


/// RMSNorm as Qwen3.5 has it: computed in f32 and scaled by `1 + weight`.
#[derive(Module, Debug)]
pub struct OffsetRmsNorm {
    pub weight: Param<Tensor<1>>,
    #[module(skip)]
    eps: f64,
}

impl OffsetRmsNorm {
    fn new(dim: usize, eps: f64, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([dim], device)),
            eps,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dtype = x.dtype();
        let x = x.cast(FloatDType::F32);
        let scale = x
            .clone()
            .square()
            .mean_dim(D - 1)
            .add_scalar(self.eps)
            .sqrt()
            .recip();
        let weight = self
            .weight
            .val()
            .cast(FloatDType::F32)
            .add_scalar(1.0)
            .unsqueeze::<D>();
        (x * scale * weight).cast(dtype)
    }
}

/// The norm closing a delta-rule block: RMSNorm scaled by `weight`, then gated by `silu(z)`.
#[derive(Module, Debug)]
pub struct GatedRmsNorm {
    pub weight: Param<Tensor<1>>,
    #[module(skip)]
    eps: f64,
}

impl GatedRmsNorm {
    fn new(dim: usize, eps: f64, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            eps,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>, gate: Tensor<D>) -> Tensor<D> {
        let dtype = x.dtype();
        let x = x.cast(FloatDType::F32);
        let scale = x
            .clone()
            .square()
            .mean_dim(D - 1)
            .add_scalar(self.eps)
            .sqrt()
            .recip();
        let normed = (x * scale).cast(dtype);
        // transformers multiplies the weight in, then the gate, each in the wider of the two
        // dtypes, as torch promotes.
        let weight = self.weight.val();
        let wide = wider(weight.dtype(), dtype);
        let h = normed.cast(wide) * weight.cast(wide).unsqueeze::<D>();
        let h = h.cast(FloatDType::F32) * silu(gate.cast(FloatDType::F32));
        h.cast(dtype)
    }
}

/// The dtype torch gives a product of two float tensors.
fn wider(a: DType, b: DType) -> DType {
    if a == b { a } else { DType::F32 }
}

/// A projection with no bias, `[out, in]` as PyTorch stores it; LoRA adapters attach to it.
fn projection(input: usize, output: usize, device: &Device) -> LoraLinear {
    LoraLinear::frozen(Tensor::zeros([output, input], device))
}

#[derive(Module, Debug)]
pub struct Mlp {
    pub gate_proj: LoraLinear,
    pub up_proj: LoraLinear,
    pub down_proj: LoraLinear,
}

impl Mlp {
    fn new(hidden: usize, intermediate: usize, device: &Device) -> Self {
        Self {
            gate_proj: projection(hidden, intermediate, device),
            up_proj: projection(hidden, intermediate, device),
            down_proj: projection(intermediate, hidden, device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.down_proj
            .forward(silu(self.gate_proj.forward(x.clone())) * self.up_proj.forward(x))
    }
}

/// cos and sin of the rotary angles for positions `0..len`, `[len, rotary_dim]`, laid out as
/// transformers lays them out: the frequencies, then the same frequencies again; computed in
/// f32 as transformers computes them and cast to the activations' dtype.
#[derive(Clone)]
pub struct RotaryTable {
    cos: Tensor<2>,
    sin: Tensor<2>,
    dim: usize,
}

impl RotaryTable {
    pub fn new(config: &Config, len: usize, dtype: DType, device: &Device) -> Self {
        let dim = config.rotary_dim();
        let half = dim / 2;
        let theta = config.rope_parameters.rope_theta as f32;
        // 1 / theta ** (arange(0, dim, 2) / dim), in f32.
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(len * dim);
        let mut sin = Vec::with_capacity(len * dim);
        for p in 0..len {
            for j in 0..dim {
                let angle = p as f32 * inv_freq[j % half];
                cos.push((angle as f64).cos() as f32);
                sin.push((angle as f64).sin() as f32);
            }
        }
        let table =
            |v: Vec<f32>| Tensor::<2>::from_data(TensorData::new(v, [len, dim]), device).cast(dtype);
        Self {
            cos: table(cos),
            sin: table(sin),
            dim,
        }
    }

    pub fn len(&self) -> usize {
        self.cos.dims()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Positions `start..start + len` of every sequence of a batch.
    pub fn range(&self, start: usize, len: usize) -> Rotary {
        assert!(start + len <= self.len(), "the rotary table ends at {}", self.len());
        let at = |t: &Tensor<2>| {
            t.clone()
                .narrow(0, start, len)
                .reshape([1, len, 1, self.dim])
        };
        Rotary {
            cos: at(&self.cos),
            sin: at(&self.sin),
            dim: self.dim,
        }
    }

    /// One position per sequence, `positions` `[B]`.
    pub fn at(&self, positions: Tensor<1, Int>) -> Rotary {
        let b = positions.dims()[0];
        let at = |t: &Tensor<2>| {
            t.clone()
                .select(0, positions.clone())
                .reshape([b, 1, 1, self.dim])
        };
        Rotary {
            cos: at(&self.cos),
            sin: at(&self.sin),
            dim: self.dim,
        }
    }
}

/// The rotation of a run of positions, `[B or 1, T, 1, rotary_dim]`.
pub struct Rotary {
    cos: Tensor<4>,
    sin: Tensor<4>,
    dim: usize,
}

impl Rotary {
    /// `x` is `[B, T, H, D]`; its first `rotary_dim` features turn, the rest pass.
    pub fn apply(&self, x: Tensor<4>) -> Tensor<4> {
        let d = x.dims()[3];
        let half = self.dim / 2;
        let turned = x.clone().narrow(3, 0, self.dim);
        let passed = x.narrow(3, self.dim, d - self.dim);
        let x1 = turned.clone().narrow(3, 0, half);
        let x2 = turned.clone().narrow(3, half, half);
        let rotated = Tensor::cat(vec![x2.neg(), x1], 3);
        let out = turned * self.cos.clone() + rotated * self.sin.clone();
        Tensor::cat(vec![out, passed], 3)
    }
}

#[derive(Module, Debug)]
pub struct Attention {
    pub q_proj: LoraLinear,
    pub k_proj: LoraLinear,
    pub v_proj: LoraLinear,
    pub o_proj: LoraLinear,
    pub q_norm: OffsetRmsNorm,
    pub k_norm: OffsetRmsNorm,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

impl Attention {
    fn new(config: &Config, device: &Device) -> Self {
        let (h, kv, d, hidden) = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.hidden_size,
        );
        Self {
            q_proj: projection(hidden, h * d * 2, device),
            k_proj: projection(hidden, kv * d, device),
            v_proj: projection(hidden, kv * d, device),
            o_proj: projection(h * d, hidden, device),
            q_norm: OffsetRmsNorm::new(d, config.rms_norm_eps, device),
            k_norm: OffsetRmsNorm::new(d, config.rms_norm_eps, device),
            heads: h,
            kv_heads: kv,
            head_dim: d,
        }
    }

    fn options(&self) -> AttentionOptions {
        AttentionOptions::causal().with_scale((self.head_dim as f32).powf(-0.5))
    }

    /// The query (rotated) and the output gate, the key (rotated) and the value of `x`
    /// `[B, T, hidden]`: `[B, T, H, D]`, `[B, T, H * D]`, `[B, T, KV, D]` twice.
    fn project(&self, x: Tensor<3>, rotary: &Rotary) -> [Tensor<4>; 3] {
        let [b, t, _] = x.dims();
        let (h, kv, d) = (self.heads, self.kv_heads, self.head_dim);
        // Per head, the query and then its gate.
        let query_gate = self.q_proj.forward(x.clone()).reshape([b, t, h, 2 * d]);
        let q = rotary.apply(self.q_norm.forward(query_gate.clone().narrow(3, 0, d)));
        let k = self.k_proj.forward(x.clone()).reshape([b, t, kv, d]);
        let k = rotary.apply(self.k_norm.forward(k));
        let v = self.v_proj.forward(x).reshape([b, t, kv, d]);
        [Tensor::cat(vec![q, query_gate.narrow(3, d, d)], 3), k, v]
    }

    /// `x` is `[B, T, hidden]` at the positions `rotary` covers; `past` the keys and values
    /// of the positions before them, `[B, S, KV, D]`. Returns the output and the keys and
    /// values of all `S + T` positions.
    pub fn forward(
        &self,
        x: Tensor<3>,
        rotary: &Rotary,
        past: Option<(Tensor<4>, Tensor<4>)>,
    ) -> (Tensor<3>, (Tensor<4>, Tensor<4>)) {
        let [b, t, _] = x.dims();
        let (h, d) = (self.heads, self.head_dim);
        let [qg, k, v] = self.project(x, rotary);
        let q = qg.clone().narrow(3, 0, d);
        let gate = qg.narrow(3, d, d).reshape([b, t, h * d]);
        let (k, v) = match past {
            Some((pk, pv)) => (Tensor::cat(vec![pk, k], 1), Tensor::cat(vec![pv, v], 1)),
            None => (k, v),
        };
        // Causal, the queries aligned with the last keys: a past in front of them is seen
        // whole.
        let out = attention(
            q,
            k.clone(),
            v.clone(),
            AttentionMask::None,
            self.options(),
        );
        let out = out.reshape([b, t, h * d]) * sigmoid(gate);
        (self.o_proj.forward(out), (k, v))
    }

    /// One token per sequence, `x` `[B, hidden]`: its key and value are written into the
    /// caches `[B, capacity, KV, D]` at `positions`, and it attends to the first
    /// `positions + 1` of them.
    fn decode(
        &self,
        x: Tensor<2>,
        rotary: &Rotary,
        positions: &Tensor<1, Int>,
        cache: (Tensor<4>, Tensor<4>),
    ) -> (Tensor<2>, (Tensor<4>, Tensor<4>)) {
        let [b, hidden] = x.dims();
        let (h, kv, d) = (self.heads, self.kv_heads, self.head_dim);
        let [qg, k, v] = self.project(x.reshape([b, 1, hidden]), rotary);
        let q = qg.clone().narrow(3, 0, d).reshape([b, h, d]);
        let gate = qg.narrow(3, d, d).reshape([b, h * d]);
        let at: Tensor<4, Int> = positions
            .clone()
            .reshape([b, 1, 1, 1])
            .expand([b, 1, kv, d]);
        let (ck, cv) = cache;
        let ck = ck.scatter(1, at.clone(), k, IndexingUpdateOp::Assign);
        let cv = cv.scatter(1, at, v, IndexingUpdateOp::Assign);
        let lengths = positions.clone().add_scalar(1);
        let out = attention_decode(
            q,
            ck.clone(),
            cv.clone(),
            Some(lengths),
            AttentionOptions::default().with_scale((d as f32).powf(-0.5)),
        );
        let out = out.reshape([b, h * d]) * sigmoid(gate);
        (self.o_proj.forward(out), (ck, cv))
    }
}

/// PyTorch's `Conv1d(C, C, K, groups=C, bias=False)` used causally, weight `[C, 1, K]`.
#[derive(Module, Debug)]
pub struct DepthwiseConv1d {
    pub weight: Param<Tensor<3>>,
}

impl DepthwiseConv1d {
    fn new(channels: usize, kernel: usize, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([channels, 1, kernel], device)),
        }
    }

    fn kernel(&self) -> usize {
        self.weight.val().dims()[2]
    }

    /// `[C, K]` in `dtype`, as dashmag's operator takes it.
    fn taps(&self, dtype: DType) -> Tensor<2> {
        let [c, _, k] = self.weight.val().dims();
        self.weight.val().reshape([c, k]).cast(dtype)
    }
}

#[derive(Module, Debug)]
pub struct GatedDeltaNet {
    pub in_proj_qkv: LoraLinear,
    pub in_proj_z: LoraLinear,
    pub in_proj_b: LoraLinear,
    pub in_proj_a: LoraLinear,
    pub conv1d: DepthwiseConv1d,
    pub dt_bias: Param<Tensor<1>>,
    /// `A_log` in the checkpoint.
    pub a_log: Param<Tensor<1>>,
    pub norm: GatedRmsNorm,
    pub out_proj: LoraLinear,
    #[module(skip)]
    k_heads: usize,
    #[module(skip)]
    v_heads: usize,
    #[module(skip)]
    k_dim: usize,
    #[module(skip)]
    v_dim: usize,
}

/// What a delta-rule layer carries: the last `K - 1` inputs of the convolution `[B, K - 1, C]`
/// and the recurrent state `[B, H, Dk, Dv]`, both f32.
#[derive(Clone, Debug)]
pub struct DeltaState {
    pub conv: Tensor<3>,
    pub recurrent: Tensor<4>,
}

impl GatedDeltaNet {
    fn new(config: &Config, device: &Device) -> Self {
        let (kh, vh, kd, vd, hidden) = (
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.hidden_size,
        );
        let conv_dim = 2 * kh * kd + vh * vd;
        Self {
            in_proj_qkv: projection(hidden, conv_dim, device),
            in_proj_z: projection(hidden, vh * vd, device),
            in_proj_b: projection(hidden, vh, device),
            in_proj_a: projection(hidden, vh, device),
            conv1d: DepthwiseConv1d::new(conv_dim, config.linear_conv_kernel_dim, device),
            dt_bias: Param::from_tensor(Tensor::ones([vh], device)),
            a_log: Param::from_tensor(Tensor::zeros([vh], device)),
            norm: GatedRmsNorm::new(vd, config.rms_norm_eps, device),
            out_proj: projection(vh * vd, hidden, device),
            k_heads: kh,
            v_heads: vh,
            k_dim: kd,
            v_dim: vd,
        }
    }

    /// The decay `g` (f32) and the write strength `beta` of `x` `[..., hidden]`, `[..., H]`.
    fn gates<const D: usize>(&self, x: Tensor<D>) -> (Tensor<D>, Tensor<D>) {
        let beta = sigmoid(self.in_proj_b.forward(x.clone()));
        let a = self.in_proj_a.forward(x).cast(FloatDType::F32);
        let decay = self
            .a_log
            .val()
            .cast(FloatDType::F32)
            .exp()
            .neg()
            .unsqueeze::<D>();
        let g = decay * softplus(a + self.dt_bias.val().cast(FloatDType::F32).unsqueeze::<D>(), 1.0);
        (g, beta)
    }

    fn check_heads(&self) {
        // Qwen3.5-0.8B has as many key heads as value heads; dashmag's rule needs that.
        assert_eq!(
            self.k_heads, self.v_heads,
            "the delta rule here takes as many key heads as value heads"
        );
    }

    /// `x` is `[B, T, hidden]`; `state` is what the positions before it left (none at a
    /// sequence's start). Returns the output and, with `keep_state`, what these positions leave.
    pub fn forward(
        &self,
        x: Tensor<3>,
        state: Option<DeltaState>,
        keep_state: bool,
    ) -> (Tensor<3>, Option<DeltaState>) {
        self.check_heads();
        let [b, t, _] = x.dims();
        let (kh, vh, kd, vd) = (self.k_heads, self.v_heads, self.k_dim, self.v_dim);
        let qkv = self.in_proj_qkv.forward(x.clone());
        let z = self.in_proj_z.forward(x.clone()).reshape([b, t, vh, vd]);
        let (g, beta) = self.gates(x);
        let (conv_state, recurrent) = match state {
            Some(s) => (Some(s.conv), Some(s.recurrent)),
            None => (None, None),
        };
        let mut options = CausalConv1dOptions::default().with_activation(Activation::Silu);
        if keep_state {
            options = options.with_final_state_len(self.conv1d.kernel() - 1);
        }
        let (qkv, conv) = causal_conv1d(
            qkv.clone(),
            self.conv1d.taps(qkv.dtype()),
            None,
            conv_state,
            None,
            options,
        );
        let q = qkv.clone().narrow(2, 0, kh * kd).reshape([b, t, kh, kd]);
        let k = qkv
            .clone()
            .narrow(2, kh * kd, kh * kd)
            .reshape([b, t, kh, kd]);
        let v = qkv.narrow(2, 2 * kh * kd, vh * vd).reshape([b, t, vh, vd]);
        let (o, recurrent) = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            recurrent,
            None,
            ChunkGatedDeltaOptions::default()
                .with_qk_l2norm(true)
                .with_final_state(keep_state),
        );
        let o = self.norm.forward(o, z).reshape([b, t, vh * vd]);
        let state = match (conv, recurrent) {
            (Some(conv), Some(recurrent)) => Some(DeltaState { conv, recurrent }),
            _ => None,
        };
        (self.out_proj.forward(o), state)
    }

    /// One token per sequence, `x` `[B, hidden]`, continuing `state`.
    fn decode(&self, x: Tensor<2>, state: DeltaState) -> (Tensor<2>, DeltaState) {
        self.check_heads();
        let [b, _] = x.dims();
        let (kh, vh, kd, vd) = (self.k_heads, self.v_heads, self.k_dim, self.v_dim);
        let qkv = self.in_proj_qkv.forward(x.clone());
        let z = self.in_proj_z.forward(x.clone()).reshape([b, vh, vd]);
        let (g, beta) = self.gates(x);
        let (qkv, conv) = causal_conv1d_update(
            qkv.clone(),
            self.conv1d.taps(qkv.dtype()),
            None,
            state.conv,
            CausalConv1dStepOptions {
                activation: Activation::Silu,
            },
        );
        let q = qkv.clone().narrow(1, 0, kh * kd).reshape([b, kh, kd]);
        let k = qkv.clone().narrow(1, kh * kd, kh * kd).reshape([b, kh, kd]);
        let v = qkv.narrow(1, 2 * kh * kd, vh * vd).reshape([b, vh, vd]);
        let (o, recurrent) = gated_delta_step(
            q,
            k,
            v,
            g,
            beta,
            state.recurrent,
            None,
            GatedDeltaStepOptions {
                qk_l2norm: true,
                scale: None,
            },
        );
        let o = self.norm.forward(o, z).reshape([b, vh * vd]);
        (self.out_proj.forward(o), DeltaState { conv, recurrent })
    }
}

/// What one decoder layer carries from one call to the next.
#[derive(Clone, Debug)]
pub enum LayerState {
    Delta(DeltaState),
    /// The keys and values of every position seen, `[B, S, KV, D]`.
    Attention { k: Tensor<4>, v: Tensor<4> },
}

#[derive(Module, Debug)]
pub struct DecoderLayer {
    pub linear_attn: Option<GatedDeltaNet>,
    pub self_attn: Option<Attention>,
    pub mlp: Mlp,
    pub input_layernorm: OffsetRmsNorm,
    pub post_attention_layernorm: OffsetRmsNorm,
}

impl DecoderLayer {
    fn new(config: &Config, kind: LayerType, device: &Device) -> Self {
        let (hidden, eps) = (config.hidden_size, config.rms_norm_eps);
        Self {
            linear_attn: (kind == LayerType::LinearAttention)
                .then(|| GatedDeltaNet::new(config, device)),
            self_attn: (kind == LayerType::FullAttention).then(|| Attention::new(config, device)),
            mlp: Mlp::new(hidden, config.intermediate_size, device),
            input_layernorm: OffsetRmsNorm::new(hidden, eps, device),
            post_attention_layernorm: OffsetRmsNorm::new(hidden, eps, device),
        }
    }

    pub fn is_attention(&self) -> bool {
        self.self_attn.is_some()
    }

    /// `x` `[B, T, hidden]` continuing `state` (none at a sequence's start). Returns the
    /// output and, with `keep_state`, the state after these positions (an attention layer's
    /// keys and values are always returned: they cost nothing more).
    pub fn forward(
        &self,
        x: Tensor<3>,
        rotary: &Rotary,
        state: Option<LayerState>,
        keep_state: bool,
    ) -> (Tensor<3>, Option<LayerState>) {
        let h = self.input_layernorm.forward(x.clone());
        let (h, state) = match (&self.linear_attn, &self.self_attn, state) {
            (Some(delta), _, s) => {
                let s = s.map(|s| match s {
                    LayerState::Delta(d) => d,
                    _ => panic!("a delta-rule layer given an attention layer's state"),
                });
                let (h, s) = delta.forward(h, s, keep_state);
                (h, s.map(LayerState::Delta))
            }
            (_, Some(attn), s) => {
                let past = s.map(|s| match s {
                    LayerState::Attention { k, v } => (k, v),
                    _ => panic!("an attention layer given a delta-rule layer's state"),
                });
                let (h, (k, v)) = attn.forward(h, rotary, past);
                (h, Some(LayerState::Attention { k, v }))
            }
            _ => unreachable!("a layer is one kind or the other"),
        };
        let x = x + h;
        let h = self
            .mlp
            .forward(self.post_attention_layernorm.forward(x.clone()));
        (x + h, state)
    }
}

#[derive(Module, Debug)]
pub struct TextModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: OffsetRmsNorm,
}

/// The state of a batch of sequences that have all seen `seen` tokens: per layer, none
/// before the first token.
#[derive(Clone, Debug)]
pub struct State {
    pub layers: Vec<Option<LayerState>>,
    pub seen: usize,
}

impl State {
    pub fn empty(layers: usize) -> Self {
        Self {
            layers: vec![None; layers],
            seen: 0,
        }
    }
}

/// A batch of sequences decoding one token per step, each at its own position: the
/// delta-rule states stacked, the keys and values in caches of one capacity, a sequence's
/// first `lengths[b]` entries its own.
pub struct Batch {
    layers: Vec<BatchLayer>,
    pub lengths: Vec<usize>,
    /// Rows still decoding. A finished row keeps its slot — the batch keeps its size, so no
    /// state is copied and no kernel meets a new shape — and is fed a token it discards at
    /// position 0 of its own cache, its length frozen.
    pub active: Vec<bool>,
    capacity: usize,
}

enum BatchLayer {
    Delta(Option<DeltaState>),
    Attention(Option<(Tensor<4>, Tensor<4>)>),
}

impl Batch {
    pub fn rows(&self) -> usize {
        self.lengths.len()
    }
}

#[derive(Module, Debug)]
pub struct Qwen3_5ForCausalLM {
    pub model: TextModel,
    /// Absent when the output head is the embedding (`tie_word_embeddings`).
    pub lm_head: Option<LoraLinear>,
}

impl Qwen3_5ForCausalLM {
    /// The model's shapes, its weights not yet loaded.
    pub fn new(config: &Config, device: &Device) -> Self {
        let (hidden, eps) = (config.hidden_size, config.rms_norm_eps);
        let model = TextModel {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, hidden).init(device),
            layers: config
                .layer_types
                .iter()
                .map(|&kind| DecoderLayer::new(config, kind, device))
                .collect(),
            norm: OffsetRmsNorm::new(hidden, eps, device),
        };
        let lm_head =
            (!config.tie_word_embeddings).then(|| projection(hidden, config.vocab_size, device));
        Self { model, lm_head }
    }

    /// The checkpoint in `dir` — `config.json` and its safetensors — with every weight cast
    /// to `dtype` and frozen. Loading is strict: every parameter is filled, and the only
    /// tensors left over are the vision tower's and the multi-token-prediction head's.
    pub fn from_pretrained(
        dir: &Path,
        dtype: FloatDType,
        device: &Device,
    ) -> Result<(Self, Config), String> {
        let config = Config::from_file(&dir.join("config.json"))?;
        let mut model = Self::new(&config, device);
        let mut missing: Option<BTreeSet<String>> = None;
        for shard in shards(dir)? {
            let mut store = SafetensorsStore::from_file(&shard)
                .with_from_adapter(PyTorchToBurnAdapter)
                .with_key_remapping(r"^model\.language_model\.", "model.")
                .with_key_remapping(r"\.A_log$", ".a_log")
                .allow_partial(true);
            let result = model
                .load_from(&mut store)
                .map_err(|e| format!("{}: {e}", shard.display()))?;
            if !result.errors.is_empty() {
                return Err(format!("{}: {:?}", shard.display(), result.errors));
            }
            let unused: Vec<_> = result
                .unused
                .iter()
                .filter(|p| !(p.starts_with("model.visual.") || p.starts_with("mtp.")))
                .collect();
            if !unused.is_empty() {
                return Err(format!("tensors with no parameter: {unused:?}"));
            }
            let here: BTreeSet<String> = result.missing.into_iter().map(|(path, _)| path).collect();
            missing = Some(match missing {
                Some(before) => before.intersection(&here).cloned().collect(),
                None => here,
            });
        }
        if let Some(missing) = missing.filter(|m| !m.is_empty()) {
            return Err(format!(
                "parameters with no tensor in the checkpoint: {missing:?}"
            ));
        }
        Ok((model.map(&mut CastTo(dtype)).no_grad(), config))
    }

    /// The projections LoRA adapters may attach to, by their PEFT paths
    /// (`model.layers.3.self_attn.q_proj`).
    pub fn projections_mut(&mut self) -> Vec<(String, &mut LoraLinear)> {
        let mut out = Vec::new();
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let at = format!("model.layers.{i}");
            if let Some(a) = layer.self_attn.as_mut() {
                out.push((format!("{at}.self_attn.q_proj"), &mut a.q_proj));
                out.push((format!("{at}.self_attn.k_proj"), &mut a.k_proj));
                out.push((format!("{at}.self_attn.v_proj"), &mut a.v_proj));
                out.push((format!("{at}.self_attn.o_proj"), &mut a.o_proj));
            }
            if let Some(d) = layer.linear_attn.as_mut() {
                out.push((format!("{at}.linear_attn.in_proj_qkv"), &mut d.in_proj_qkv));
                out.push((format!("{at}.linear_attn.in_proj_z"), &mut d.in_proj_z));
                out.push((format!("{at}.linear_attn.in_proj_b"), &mut d.in_proj_b));
                out.push((format!("{at}.linear_attn.in_proj_a"), &mut d.in_proj_a));
                out.push((format!("{at}.linear_attn.out_proj"), &mut d.out_proj));
            }
            let m = &mut layer.mlp;
            out.push((format!("{at}.mlp.gate_proj"), &mut m.gate_proj));
            out.push((format!("{at}.mlp.up_proj"), &mut m.up_proj));
            out.push((format!("{at}.mlp.down_proj"), &mut m.down_proj));
        }
        out
    }

    /// `projections_mut`, read-only.
    pub fn projections(&self) -> Vec<(String, &LoraLinear)> {
        let mut out = Vec::new();
        for (i, layer) in self.model.layers.iter().enumerate() {
            let at = format!("model.layers.{i}");
            if let Some(a) = layer.self_attn.as_ref() {
                out.push((format!("{at}.self_attn.q_proj"), &a.q_proj));
                out.push((format!("{at}.self_attn.k_proj"), &a.k_proj));
                out.push((format!("{at}.self_attn.v_proj"), &a.v_proj));
                out.push((format!("{at}.self_attn.o_proj"), &a.o_proj));
            }
            if let Some(d) = layer.linear_attn.as_ref() {
                out.push((format!("{at}.linear_attn.in_proj_qkv"), &d.in_proj_qkv));
                out.push((format!("{at}.linear_attn.in_proj_z"), &d.in_proj_z));
                out.push((format!("{at}.linear_attn.in_proj_b"), &d.in_proj_b));
                out.push((format!("{at}.linear_attn.in_proj_a"), &d.in_proj_a));
                out.push((format!("{at}.linear_attn.out_proj"), &d.out_proj));
            }
            let m = &layer.mlp;
            out.push((format!("{at}.mlp.gate_proj"), &m.gate_proj));
            out.push((format!("{at}.mlp.up_proj"), &m.up_proj));
            out.push((format!("{at}.mlp.down_proj"), &m.down_proj));
        }
        out
    }

    pub fn empty_state(&self) -> State {
        State::empty(self.model.layers.len())
    }

    /// The output head's weight, `[V, hidden]`.
    pub fn head_weight(&self) -> Tensor<2> {
        match &self.lm_head {
            Some(head) => head.weight.val(),
            None => self.model.embed_tokens.weight.val(),
        }
    }

    pub fn embed(&self, ids: Tensor<2, Int>) -> Tensor<3> {
        self.model.embed_tokens.forward(ids)
    }

    /// Hidden states after the final norm, `[B, T, hidden]`, for `ids` `[B, T]` continuing
    /// `state`, which takes them.
    pub fn forward(&self, ids: Tensor<2, Int>, table: &RotaryTable, state: &mut State) -> Tensor<3> {
        let t = ids.dims()[1];
        let mut x = self.embed(ids);
        let rotary = table.range(state.seen, t);
        for (layer, s) in self.model.layers.iter().zip(state.layers.iter_mut()) {
            let (y, next) = layer.forward(x, &rotary, s.take(), true);
            x = y;
            *s = next;
        }
        state.seen += t;
        self.model.norm.forward(x)
    }

    /// Logits over the vocabulary, `[..., V]`, in the hidden states' dtype.
    pub fn logits<const D: usize>(&self, hidden: Tensor<D>) -> Tensor<D> {
        let w = self.head_weight();
        let dims = hidden.dims();
        let rows = dims[..D - 1].iter().product();
        let flat: Tensor<2> = hidden.reshape([rows, dims[D - 1]]);
        let mut out = dims;
        out[D - 1] = w.dims()[0];
        flat.matmul(w.transpose()).reshape(out)
    }

    /// The batch whose row `b` continues `rows[b]` (one sequence each); rows may share a state.
    pub fn batch(&self, rows: &[&State]) -> Batch {
        self.batch_sized(rows, 0)
    }

    /// `batch`, its key/value caches holding at least `capacity` positions from the start: a
    /// batch that never grows allocates once.
    pub fn batch_sized(&self, rows: &[&State], capacity: usize) -> Batch {
        let lengths: Vec<usize> = rows.iter().map(|s| s.seen).collect();
        let capacity = round_up(
            (lengths.iter().copied().max().unwrap_or(0) + GROW).max(capacity),
            256,
        );
        let layers = (0..self.model.layers.len())
            .map(|l| {
                let attention = self.model.layers[l].is_attention();
                let parts: Vec<&LayerState> = rows
                    .iter()
                    .map(|s| s.layers[l].as_ref().expect("a batch continues sequences already begun"))
                    .collect();
                if attention {
                    let (k0, _) = match parts[0] {
                        LayerState::Attention { k, v } => (k, v),
                        _ => unreachable!(),
                    };
                    let [_, _, kv, d] = k0.dims();
                    let zeros = || {
                        Tensor::<4>::zeros([rows.len(), capacity, kv, d], (&k0.device(), k0.dtype()))
                    };
                    let (mut ck, mut cv) = (zeros(), zeros());
                    for (b, part) in parts.iter().enumerate() {
                        let LayerState::Attention { k, v } = part else { unreachable!() };
                        let n = k.dims()[1];
                        ck = ck.slice_assign([b..b + 1, 0..n, 0..kv, 0..d], k.clone());
                        cv = cv.slice_assign([b..b + 1, 0..n, 0..kv, 0..d], v.clone());
                    }
                    BatchLayer::Attention(Some((ck, cv)))
                } else {
                    let (conv, rec): (Vec<_>, Vec<_>) = parts
                        .iter()
                        .map(|p| match p {
                            LayerState::Delta(s) => (s.conv.clone(), s.recurrent.clone()),
                            _ => unreachable!(),
                        })
                        .unzip();
                    BatchLayer::Delta(Some(DeltaState {
                        conv: Tensor::cat(conv, 0),
                        recurrent: Tensor::cat(rec, 0),
                    }))
                }
            })
            .collect();
        Batch {
            layers,
            active: vec![true; lengths.len()],
            lengths,
            capacity,
        }
    }

    /// Row `b` of a batch as a state of its own (views into the batch's tensors).
    pub fn row(&self, batch: &Batch, b: usize) -> State {
        let n = batch.lengths[b];
        let layers = batch
            .layers
            .iter()
            .map(|l| {
                Some(match l {
                    BatchLayer::Delta(Some(s)) => LayerState::Delta(DeltaState {
                        conv: s.conv.clone().narrow(0, b, 1),
                        recurrent: s.recurrent.clone().narrow(0, b, 1),
                    }),
                    BatchLayer::Attention(Some((k, v))) => LayerState::Attention {
                        k: k.clone().narrow(0, b, 1).narrow(1, 0, n),
                        v: v.clone().narrow(0, b, 1).narrow(1, 0, n),
                    },
                    _ => unreachable!("a batch's layers are filled between steps"),
                })
            })
            .collect();
        State { layers, seen: n }
    }

    /// Row `b` of a batch as a state of its own, copied out: it keeps none of the batch's
    /// buffers alive.
    pub fn row_copy(&self, batch: &Batch, b: usize) -> State {
        let n = batch.lengths[b];
        let device = self.model.norm.weight.device();
        let idx = Tensor::<1, Int>::from_data(TensorData::new(vec![b as i64], [1]), &device);
        let layers = batch
            .layers
            .iter()
            .map(|l| {
                Some(match l {
                    BatchLayer::Delta(Some(s)) => LayerState::Delta(DeltaState {
                        conv: s.conv.clone().select(0, idx.clone()),
                        recurrent: s.recurrent.clone().select(0, idx.clone()),
                    }),
                    BatchLayer::Attention(Some((k, v))) => LayerState::Attention {
                        k: k.clone().narrow(1, 0, n).select(0, idx.clone()),
                        v: v.clone().narrow(1, 0, n).select(0, idx.clone()),
                    },
                    _ => unreachable!("a batch's layers are filled between steps"),
                })
            })
            .collect();
        State { layers, seen: n }
    }

    /// Writes `state` (one sequence) into row `b` of the batch: its delta-rule states whole, its
    /// keys and values from position `from` on (those before are the row's already). The row's
    /// length becomes the state's and it decodes again.
    pub fn write_row(&self, batch: &mut Batch, b: usize, state: &State, from: usize) {
        let n = state.seen;
        while n + 1 > batch.capacity {
            self.grow(batch);
        }
        for (slot, l) in batch.layers.iter_mut().zip(&state.layers) {
            match (slot, l.as_ref().expect("a written row has begun")) {
                (BatchLayer::Delta(s), LayerState::Delta(d)) => {
                    let cur = s.take().expect("filled");
                    let [_, w, c] = cur.conv.dims();
                    let [_, hh, dk, dv] = cur.recurrent.dims();
                    *s = Some(DeltaState {
                        conv: cur.conv.slice_assign([b..b + 1, 0..w, 0..c], d.conv.clone()),
                        recurrent: cur
                            .recurrent
                            .slice_assign([b..b + 1, 0..hh, 0..dk, 0..dv], d.recurrent.clone()),
                    });
                }
                (BatchLayer::Attention(c), LayerState::Attention { k, v }) => {
                    let (ck, cv) = c.take().expect("filled");
                    let [_, _, kv, d] = ck.dims();
                    let part = |t: &Tensor<4>| t.clone().narrow(1, from, n - from);
                    *c = Some((
                        ck.slice_assign([b..b + 1, from..n, 0..kv, 0..d], part(k)),
                        cv.slice_assign([b..b + 1, from..n, 0..kv, 0..d], part(v)),
                    ));
                }
                _ => unreachable!("a batch's layers are the model's"),
            }
        }
        batch.lengths[b] = n;
        batch.active[b] = true;
    }

    /// The batch of the rows `keep`, in that order.
    pub fn keep_rows(&self, batch: Batch, keep: &[usize]) -> Batch {
        let device = self.model.norm.weight.device();
        let idx = Tensor::<1, Int>::from_data(
            TensorData::new(keep.iter().map(|&i| i as i64).collect::<Vec<_>>(), [keep.len()]),
            &device,
        );
        let layers = batch
            .layers
            .into_iter()
            .map(|l| match l {
                BatchLayer::Delta(Some(s)) => BatchLayer::Delta(Some(DeltaState {
                    conv: s.conv.select(0, idx.clone()),
                    recurrent: s.recurrent.select(0, idx.clone()),
                })),
                BatchLayer::Attention(Some((k, v))) => {
                    BatchLayer::Attention(Some((k.select(0, idx.clone()), v.select(0, idx.clone()))))
                }
                _ => unreachable!("a batch's layers are filled between steps"),
            })
            .collect();
        Batch {
            layers,
            lengths: keep.iter().map(|&i| batch.lengths[i]).collect(),
            active: keep.iter().map(|&i| batch.active[i]).collect(),
            capacity: batch.capacity,
        }
    }

    /// One token per row, `tokens` `[B]`, each at its row's next position. Returns the hidden
    /// states after the final norm, `[B, hidden]`; the batch takes the tokens.
    pub fn decode(&self, tokens: Tensor<1, Int>, table: &RotaryTable, batch: &mut Batch) -> Tensor<2> {
        let b = batch.rows();
        if batch
            .lengths
            .iter()
            .zip(&batch.active)
            .any(|(&n, &on)| on && n + 1 > batch.capacity)
        {
            self.grow(batch);
        }
        let device = tokens.device();
        let positions = Tensor::<1, Int>::from_data(
            TensorData::new(
                batch
                    .lengths
                    .iter()
                    .zip(&batch.active)
                    .map(|(&n, &on)| if on { n as i64 } else { 0 })
                    .collect::<Vec<_>>(),
                [b],
            ),
            &device,
        );
        let rotary = table.at(positions.clone());
        let mut x: Tensor<2> = self.embed(tokens.reshape([b, 1])).reshape([b, self.hidden()]);
        for (layer, slot) in self.model.layers.iter().zip(batch.layers.iter_mut()) {
            let h = layer.input_layernorm.forward(x.clone());
            let h = match (slot, &layer.linear_attn, &layer.self_attn) {
                (BatchLayer::Delta(s), Some(delta), _) => {
                    let (h, next) = delta.decode(h, s.take().expect("filled"));
                    *s = Some(next);
                    h
                }
                (BatchLayer::Attention(c), _, Some(attn)) => {
                    let (h, next) = attn.decode(h, &rotary, &positions, c.take().expect("filled"));
                    *c = Some(next);
                    h
                }
                _ => unreachable!("a batch's layers are the model's"),
            };
            x = x + h;
            let h = layer
                .mlp
                .forward(layer.post_attention_layernorm.forward(x.clone()));
            x = x + h;
        }
        for (n, &on) in batch.lengths.iter_mut().zip(&batch.active) {
            if on {
                *n += 1;
            }
        }
        self.model.norm.forward(x)
    }

    pub(crate) fn hidden(&self) -> usize {
        self.model.norm.weight.val().dims()[0]
    }

    /// Adds `GROW` positions to the batch's key/value capacity.
    pub(crate) fn grow(&self, batch: &mut Batch) {
        let more = GROW;
        for l in batch.layers.iter_mut() {
            if let BatchLayer::Attention(c) = l {
                let (k, v) = c.take().expect("filled");
                let [b, _, kv, d] = k.dims();
                let pad = |t: Tensor<4>| {
                    let z = Tensor::<4>::zeros([b, more, kv, d], (&t.device(), t.dtype()));
                    Tensor::cat(vec![t, z], 1)
                };
                *c = Some((pad(k), pad(v)));
            }
        }
        batch.capacity += more;
    }
}

/// Key/value positions a batch is given beyond its longest row, and added when one fills them:
/// a fixed step, not doubling, so that a long batch does not hold twice what it needs.
const GROW: usize = 1024;

fn round_up(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

/// The safetensors files of a checkpoint: those its index names, else `model.safetensors`.
fn shards(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let index = dir.join("model.safetensors.index.json");
    if let Ok(text) = std::fs::read_to_string(&index) {
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let files: BTreeSet<&str> = json["weight_map"]
            .as_object()
            .ok_or("model.safetensors.index.json has no weight_map")?
            .values()
            .filter_map(|v| v.as_str())
            .collect();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    let single = dir.join("model.safetensors");
    if single.is_file() {
        Ok(vec![single])
    } else {
        Err(format!("no safetensors in {}", dir.display()))
    }
}

struct CastTo(FloatDType);

impl ModuleMapper for CastTo {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let dtype = self.0;
        param.map(|t| t.cast(dtype))
    }
}

/// Projections that read one input, merged for decoding, where launches and not arithmetic
/// set the time: the weights stacked `[Σ out, in]`, the active adapters' A's stacked and B's
/// (scale folded in) block-diagonal, so that the base and the low-rank product are one each.
/// The base part is built once; the low-rank part again whenever the adapters change.
struct Fused {
    weight: Tensor<2>,
    low_rank: Option<(Tensor<2>, Tensor<2>)>,
    widths: Vec<usize>,
}

impl Fused {
    fn new(parts: &[&LoraLinear]) -> Self {
        Self {
            weight: Tensor::cat(parts.iter().map(|p| p.weight.val()).collect(), 0),
            low_rank: None,
            widths: parts.iter().map(|p| p.out_features()).collect(),
        }
    }

    /// The parts' active adapters (all or none of them have one).
    fn set_lora(&mut self, parts: &[&LoraLinear]) {
        let loras: Vec<_> = parts.iter().filter_map(|p| p.lora()).collect();
        if loras.is_empty() {
            self.low_rank = None;
            return;
        }
        assert_eq!(loras.len(), parts.len(), "merged projections all have an adapter or none");
        let rank: usize = loras.iter().map(|l| l.rank()).sum();
        let out: usize = self.widths.iter().sum();
        let device = self.weight.device();
        let u = Tensor::cat(loras.iter().map(|l| l.a.val()).collect(), 0).transpose();
        let mut v = Tensor::<2>::zeros([rank, out], (&device, DType::F32));
        let (mut r0, mut c0) = (0, 0);
        for (l, &w) in loras.iter().zip(&self.widths) {
            let r = l.rank();
            v = v.slice_assign([r0..r0 + r, c0..c0 + w], l.b.val().transpose().mul_scalar(l.scale));
            r0 += r;
            c0 += w;
        }
        self.low_rank = Some((u, v));
    }

    fn forward(&self, x: Tensor<2>) -> Vec<Tensor<2>> {
        let dtype = x.dtype();
        let base = x.clone().matmul(self.weight.clone().transpose());
        let y = match &self.low_rank {
            Some((u, v)) => dash_ops::lowrank::low_rank(
                x,
                u.clone(),
                v.clone(),
                Some(base),
                dash_ops::lowrank::LowRankOptions::default(),
            )
            .cast(dtype),
            None => base,
        };
        if self.widths.len() == 1 {
            return vec![y];
        }
        let mut start = 0;
        self.widths
            .iter()
            .map(|&w| {
                let part = y.clone().narrow(1, start, w);
                start += w;
                part
            })
            .collect()
    }
}

/// RMSNorm with its scale (`1 + weight`, f32) already formed.
fn rms_scaled<const D: usize>(x: Tensor<D>, scale: &Tensor<1>, eps: f64) -> Tensor<D> {
    let dtype = x.dtype();
    let x = x.cast(FloatDType::F32);
    let inv = x
        .clone()
        .square()
        .mean_dim(D - 1)
        .add_scalar(eps)
        .sqrt()
        .recip();
    (x * inv * scale.clone().unsqueeze::<D>()).cast(dtype)
}

enum FusedMixer {
    Delta {
        input: Fused,
        out: Fused,
        taps: Tensor<2>,
        decay: Tensor<1>,
        dt_bias: Tensor<1>,
    },
    Attention {
        qkv: Fused,
        out: Fused,
        q_norm: Tensor<1>,
        k_norm: Tensor<1>,
    },
}

struct FusedLayer {
    mixer: FusedMixer,
    gate_up: Fused,
    down: Fused,
    input_norm: Tensor<1>,
    post_norm: Tensor<1>,
}

/// The model laid out for decoding (see [`Fused`]); it reads the model's frozen weights once
/// and its adapters at every [`Decoder::set_lora`].
pub struct Decoder {
    layers: Vec<FusedLayer>,
    final_norm: Tensor<1>,
    eps: f64,
}

fn offset_scale(norm: &OffsetRmsNorm) -> Tensor<1> {
    norm.weight.val().cast(FloatDType::F32).add_scalar(1.0)
}

impl Decoder {
    pub fn new(model: &Qwen3_5ForCausalLM) -> Self {
        let layers = model
            .model
            .layers
            .iter()
            .map(|l| {
                let mixer = match (&l.linear_attn, &l.self_attn) {
                    (Some(d), _) => FusedMixer::Delta {
                        input: Fused::new(&[&d.in_proj_qkv, &d.in_proj_z, &d.in_proj_b, &d.in_proj_a]),
                        out: Fused::new(&[&d.out_proj]),
                        taps: d.conv1d.taps(d.out_proj.weight.val().dtype()),
                        decay: d.a_log.val().cast(FloatDType::F32).exp().neg(),
                        dt_bias: d.dt_bias.val().cast(FloatDType::F32),
                    },
                    (_, Some(a)) => FusedMixer::Attention {
                        qkv: Fused::new(&[&a.q_proj, &a.k_proj, &a.v_proj]),
                        out: Fused::new(&[&a.o_proj]),
                        q_norm: offset_scale(&a.q_norm),
                        k_norm: offset_scale(&a.k_norm),
                    },
                    _ => unreachable!("a layer is one kind or the other"),
                };
                FusedLayer {
                    mixer,
                    gate_up: Fused::new(&[&l.mlp.gate_proj, &l.mlp.up_proj]),
                    down: Fused::new(&[&l.mlp.down_proj]),
                    input_norm: offset_scale(&l.input_layernorm),
                    post_norm: offset_scale(&l.post_attention_layernorm),
                }
            })
            .collect();
        let mut decoder = Self {
            layers,
            final_norm: offset_scale(&model.model.norm),
            eps: model.model.norm.eps,
        };
        decoder.set_lora(model);
        decoder
    }

    /// Reads the model's active adapters again.
    pub fn set_lora(&mut self, model: &Qwen3_5ForCausalLM) {
        for (f, l) in self.layers.iter_mut().zip(&model.model.layers) {
            match (&mut f.mixer, &l.linear_attn, &l.self_attn) {
                (FusedMixer::Delta { input, out, .. }, Some(d), _) => {
                    input.set_lora(&[&d.in_proj_qkv, &d.in_proj_z, &d.in_proj_b, &d.in_proj_a]);
                    out.set_lora(&[&d.out_proj]);
                }
                (FusedMixer::Attention { qkv, out, .. }, _, Some(a)) => {
                    qkv.set_lora(&[&a.q_proj, &a.k_proj, &a.v_proj]);
                    out.set_lora(&[&a.o_proj]);
                }
                _ => unreachable!("built from the same model"),
            }
            f.gate_up.set_lora(&[&l.mlp.gate_proj, &l.mlp.up_proj]);
            f.down.set_lora(&[&l.mlp.down_proj]);
        }
    }

    /// [`Qwen3_5ForCausalLM::decode`] on the merged layout: the same arithmetic, fewer launches.
    pub fn decode(
        &self,
        model: &Qwen3_5ForCausalLM,
        tokens: Tensor<1, Int>,
        table: &RotaryTable,
        batch: &mut Batch,
    ) -> Tensor<2> {
        let b = batch.rows();
        if batch
            .lengths
            .iter()
            .zip(&batch.active)
            .any(|(&n, &on)| on && n + 1 > batch.capacity)
        {
            model.grow(batch);
        }
        let device = tokens.device();
        let positions = Tensor::<1, Int>::from_data(
            TensorData::new(
                batch
                    .lengths
                    .iter()
                    .zip(&batch.active)
                    .map(|(&n, &on)| if on { n as i64 } else { 0 })
                    .collect::<Vec<_>>(),
                [b],
            ),
            &device,
        );
        let rotary = table.at(positions.clone());
        let eps = self.eps;
        let mut x: Tensor<2> = model.embed(tokens.reshape([b, 1])).reshape([b, model.hidden()]);
        for ((f, layer), slot) in self.layers.iter().zip(&model.model.layers).zip(batch.layers.iter_mut()) {
            let h = rms_scaled(x.clone(), &f.input_norm, eps);
            let h = match (slot, &f.mixer) {
                (BatchLayer::Delta(s), FusedMixer::Delta { input, out, taps, decay, dt_bias }) => {
                    let d = layer.linear_attn.as_ref().unwrap();
                    let (kh, vh, kd, vd) = (d.k_heads, d.v_heads, d.k_dim, d.v_dim);
                    let mut parts = input.forward(h).into_iter();
                    let (qkv, z, bb, a) = (
                        parts.next().unwrap(),
                        parts.next().unwrap(),
                        parts.next().unwrap(),
                        parts.next().unwrap(),
                    );
                    let state = s.take().expect("filled");
                    let beta = sigmoid(bb);
                    let g = decay.clone().unsqueeze::<2>()
                        * softplus(a.cast(FloatDType::F32) + dt_bias.clone().unsqueeze::<2>(), 1.0);
                    let (qkv, conv) = causal_conv1d_update(
                        qkv,
                        taps.clone(),
                        None,
                        state.conv,
                        CausalConv1dStepOptions {
                            activation: Activation::Silu,
                        },
                    );
                    let q = qkv.clone().narrow(1, 0, kh * kd).reshape([b, kh, kd]);
                    let k = qkv.clone().narrow(1, kh * kd, kh * kd).reshape([b, kh, kd]);
                    let v = qkv.narrow(1, 2 * kh * kd, vh * vd).reshape([b, vh, vd]);
                    let (o, recurrent) = gated_delta_step(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        state.recurrent,
                        None,
                        GatedDeltaStepOptions {
                            qk_l2norm: true,
                            scale: None,
                        },
                    );
                    *s = Some(DeltaState { conv, recurrent });
                    let o = d.norm.forward(o, z.reshape([b, vh, vd])).reshape([b, vh * vd]);
                    out.forward(o).pop().unwrap()
                }
                (BatchLayer::Attention(c), FusedMixer::Attention { qkv, out, q_norm, k_norm }) => {
                    let a = layer.self_attn.as_ref().unwrap();
                    let (hh, kv, d) = (a.heads, a.kv_heads, a.head_dim);
                    let mut parts = qkv.forward(h).into_iter();
                    let (qg, k, v) = (parts.next().unwrap(), parts.next().unwrap(), parts.next().unwrap());
                    let qg = qg.reshape([b, 1, hh, 2 * d]);
                    let q = rotary
                        .apply(rms_scaled(qg.clone().narrow(3, 0, d), q_norm, a.q_norm.eps))
                        .reshape([b, hh, d]);
                    let gate = qg.narrow(3, d, d).reshape([b, hh * d]);
                    let k = rotary.apply(rms_scaled(k.reshape([b, 1, kv, d]), k_norm, a.k_norm.eps));
                    let v = v.reshape([b, 1, kv, d]);
                    let at: Tensor<4, Int> = positions.clone().reshape([b, 1, 1, 1]).expand([b, 1, kv, d]);
                    let (ck, cv) = c.take().expect("filled");
                    let ck = ck.scatter(1, at.clone(), k, IndexingUpdateOp::Assign);
                    let cv = cv.scatter(1, at, v, IndexingUpdateOp::Assign);
                    let o = attention_decode(
                        q,
                        ck.clone(),
                        cv.clone(),
                        Some(positions.clone().add_scalar(1)),
                        AttentionOptions::default().with_scale((d as f32).powf(-0.5)),
                    );
                    *c = Some((ck, cv));
                    out.forward(o.reshape([b, hh * d]) * sigmoid(gate)).pop().unwrap()
                }
                _ => unreachable!("a batch's layers are the model's"),
            };
            x = x + h;
            let h = rms_scaled(x.clone(), &f.post_norm, eps);
            let mut gu = f.gate_up.forward(h).into_iter();
            let (gate, up) = (gu.next().unwrap(), gu.next().unwrap());
            x = x + f.down.forward(silu(gate) * up).pop().unwrap();
        }
        for (n, &on) in batch.lengths.iter_mut().zip(&batch.active) {
            if on {
                *n += 1;
            }
        }
        rms_scaled(x, &self.final_norm, eps)
    }
}
