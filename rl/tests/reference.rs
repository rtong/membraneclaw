//! The Rust side held to transformers' numbers, from the files
//! `rl/tools/export_qwen3_5_reference.py` writes:
//!
//!     QWEN3_5_REFERENCE=<its out_dir> cargo test -- --ignored
//!
//! The checkpoint is read from the Hugging Face cache.

use std::path::PathBuf;

use burn::tensor::{Device, FloatDType, Int, Tensor, TensorData};
use membrane_rl::chat::{ChatTemplate, Tokenizer};
use membrane_rl::hf;
use membrane_rl::qwen3_5::{Config, Qwen3_5ForCausalLM, RotaryTable};
use safetensors::{Dtype, SafeTensors};
use serde_json::{Map, Value as Json};

const MODEL: &str = "Qwen/Qwen3.5-0.8B";

fn reference_dir() -> PathBuf {
    std::env::var_os("QWEN3_5_REFERENCE")
        .map(PathBuf::from)
        .expect("set QWEN3_5_REFERENCE to the directory export_qwen3_5_reference.py wrote")
}

#[test]
#[ignore = "needs the checkpoint in the Hugging Face cache and QWEN3_5_REFERENCE"]
fn chat_template_and_tokenizer_match_transformers() {
    let dir = hf::snapshot(MODEL).unwrap();
    let template = ChatTemplate::from_dir(&dir).unwrap();
    let tokenizer = Tokenizer::from_dir(&dir).unwrap();
    let file = std::fs::read_to_string(reference_dir().join("qwen3_5_chat.json")).unwrap();
    let recorded: Json = serde_json::from_str(&file).unwrap();
    let cases = recorded["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let name = format!(
            "{} (generation prompt {}, thinking {})",
            case["name"], case["add_generation_prompt"], case["enable_thinking"]
        );
        let mut kwargs = Map::new();
        kwargs.insert("enable_thinking".into(), case["enable_thinking"].clone());
        let tools = case.get("tools").filter(|t| !t.is_null());
        let generation_prompt = case["add_generation_prompt"].as_bool().unwrap();
        let text = template
            .render(&case["messages"], tools, generation_prompt, &kwargs)
            .unwrap();
        assert_eq!(text, case["text"].as_str().unwrap(), "{name}: text");
        let want: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(tokenizer.encode(&text).unwrap(), want, "{name}: token ids");
    }
    println!("{} cases match", cases.len());
}

/// The tensors `qwen3_5_reference.safetensors` holds.
struct Recorded(Vec<u8>);

impl Recorded {
    fn load() -> Self {
        Self(std::fs::read(reference_dir().join("qwen3_5_reference.safetensors")).unwrap())
    }

    fn get(&self, name: &str, dtype: Dtype) -> (Vec<u8>, Vec<usize>) {
        let file = SafeTensors::deserialize(&self.0).unwrap();
        let t = file.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(t.dtype(), dtype, "{name}");
        (t.data().to_vec(), t.shape().to_vec())
    }

    fn f32(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let (bytes, shape) = self.get(name, Dtype::F32);
        (
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            shape,
        )
    }

    fn i64(&self, name: &str) -> (Vec<i64>, Vec<usize>) {
        let (bytes, shape) = self.get(name, Dtype::I64);
        (
            bytes
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            shape,
        )
    }

    /// A recorded `[T, hidden]` as a `[1, T, hidden]` tensor.
    fn hidden(&self, name: &str, device: &Device) -> Tensor<3> {
        let (v, shape) = self.f32(name);
        Tensor::from_data(TensorData::new(v, [1, shape[0], shape[1]]), device)
    }
}

/// The largest difference, absolute and relative to the largest recorded magnitude.
fn compare(what: &str, got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let diff = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = want.iter().map(|b| b.abs()).fold(0.0f32, f32::max);
    println!(
        "{what:<22} max |diff| {diff:.3e}  (largest value {scale:.3e}, relative {:.3e})",
        diff / scale
    );
    (diff, diff / scale)
}

/// Every part of the model that runs before dashmag's delta rule lands, on Flex in f32,
/// against transformers' float32 CPU run: the embedding, each full-attention layer from the
/// hidden state recorded before it, the final norm, and the logits.
/// The device and dtype the model tests run on: `MEMBRANE_DEVICE=cuda` (with the `cuda`
/// feature) and `MEMBRANE_DTYPE=bf16`; Flex and f32 otherwise.
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

fn host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.cast(FloatDType::F32).into_data().try_to_vec::<f32>().unwrap()
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold(0, |b, (j, &x)| if x > row[b] { j } else { b })
}

fn logsumexp(row: &[f32]) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    (max + row.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln()) as f32
}

#[test]
#[ignore = "needs the checkpoint in the Hugging Face cache and QWEN3_5_REFERENCE"]
fn model_matches_transformers() {
    let (device, dtype) = device_and_dtype();
    let bf16 = dtype == FloatDType::BF16;
    // f32 holds every layer to transformers' fp32 within float error; bf16 is measured
    // against the same fp32 truth, its rounding the difference.
    let (layer_tol, logit_tol) = if bf16 { (6e-2, 0.5) } else { (1e-4, 2e-3) };
    let recorded = Recorded::load();
    let (model, config): (Qwen3_5ForCausalLM, Config) =
        Qwen3_5ForCausalLM::from_pretrained(&hf::snapshot(MODEL).unwrap(), dtype, &device).unwrap();
    let table = RotaryTable::new(&config, 512, dtype.into(), &device);

    let (ids, shape) = recorded.i64("input_ids");
    let t = shape[1];
    let input = Tensor::<2, Int>::from_data(TensorData::new(ids.clone(), [1, t]), &device);
    let (diff, _) = compare("embedding", &host(model.embed(input.clone())), &recorded.f32("hidden.0").0);
    assert!(bf16 || diff == 0.0);

    // Every layer on transformers' own input to it.
    let rotary = table.range(0, t);
    for (i, layer) in model.model.layers.iter().enumerate() {
        let x = recorded.hidden(&format!("hidden.{i}"), &device).cast(dtype);
        let (out, _) = layer.forward(x, &rotary, None, false);
        let kind = if layer.is_attention() { "attention" } else { "delta rule" };
        let (_, rel) = compare(
            &format!("layer {i:>2} ({kind})"),
            &host(out),
            &recorded.f32(&format!("hidden.{}", i + 1)).0,
        );
        assert!(rel < layer_tol, "layer {i}");
    }

    // The whole model on the ids, each layer on the one before.
    let mut state = model.empty_state();
    let hidden = model.forward(input, &table, &mut state);
    assert_eq!(state.seen, t);
    let (_, rel) = compare("final hidden (whole)", &host(hidden.clone()), &recorded.f32("hidden.final").0);
    assert!(rel < layer_tol * 4.0);
    let logits = host(model.logits(hidden.clone()));
    let v = config.vocab_size;
    let lse: Vec<f32> = logits.chunks_exact(v).map(logsumexp).collect();
    let (diff, _) = compare("logsumexp", &lse, &recorded.f32("logsumexp").0);
    assert!(diff < logit_tol);
    let target: Vec<f32> = (0..t - 1)
        .map(|i| logits[i * v + ids[i + 1] as usize] - lse[i])
        .collect();
    let (diff, _) = compare("next-token log-prob", &target, &recorded.f32("target_logprob").0);
    assert!(diff < logit_tol);
    let (top, _) = recorded.i64("top_ids");
    let same_top1 = logits
        .chunks_exact(v)
        .enumerate()
        .filter(|(i, row)| argmax(row) as i64 == top[i * 16])
        .count();
    println!("top-1 token the same at {same_top1} of {t} positions");
    assert!(same_top1 >= if bf16 { t - 3 } else { t });

    // Greedy decoding from the state, two rows forked from it and a third at another
    // position: each step's token and logits against transformers' cached generation.
    let (greedy, _) = recorded.i64("greedy_ids");
    let (g_lse, _) = recorded.f32("greedy_logsumexp");
    let short = 37;
    let mut short_state = model.empty_state();
    let short_hidden = model.forward(
        Tensor::<2, Int>::from_data(TensorData::new(ids[..short].to_vec(), [1, short]), &device),
        &table,
        &mut short_state,
    );
    let mut batch = model.batch(&[&state, &state, &short_state]);
    let last = |h: Tensor<3>| {
        let n = h.dims()[1];
        h.narrow(1, n - 1, 1).reshape([1, config.hidden_size])
    };
    let mut h = Tensor::cat(vec![last(hidden.clone()), last(hidden), last(short_hidden)], 0);
    let mut mismatched = 0;
    let mut worst_lse = 0.0f32;
    let mut short_ids: Vec<i64> = ids[..short].to_vec();
    for (step, &want) in greedy.iter().enumerate() {
        let rows = host(model.logits(h));
        let got: Vec<usize> = rows.chunks_exact(v).map(argmax).collect();
        assert_eq!(got[0], got[1], "forked rows differ at step {step}");
        mismatched += (got[0] as i64 != want) as usize;
        worst_lse = worst_lse.max((logsumexp(&rows[..v]) - g_lse[step]).abs());
        // The third row against a whole forward of its own sequence.
        short_ids.push(got[2] as i64);
        let tokens = Tensor::<1, Int>::from_data(
            TensorData::new(vec![want, want, got[2] as i64], [3]),
            &device,
        );
        h = model.decode(tokens, &table, &mut batch);
    }
    println!("greedy: {mismatched} of {} tokens differ; logsumexp within {worst_lse:.3e}", greedy.len());
    assert!(mismatched == 0 || bf16);
    assert!(worst_lse < logit_tol * 4.0);
    let n = short_ids.len();
    let mut whole = model.empty_state();
    let whole_hidden = model.forward(
        Tensor::<2, Int>::from_data(TensorData::new(short_ids, [1, n]), &device),
        &table,
        &mut whole,
    );
    let whole_last = host(model.logits(last(whole_hidden)));
    let decoded_last = host(model.logits(h.narrow(0, 2, 1)));
    let (diff, _) = compare("decoded vs whole (row 3)", &decoded_last, &whole_last);
    assert!(diff < logit_tol * 10.0);
}
