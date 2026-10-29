//! PEFT adapters on the model: the policy being trained and the frozen reference it is held to
//! are two adapters over one frozen base, read from and written to PEFT's format, so that
//! transformers, MLX and the notebooks read every checkpoint as before.

use std::path::Path;

use burn::tensor::Device;
use dash_lora::{AdapterChoice, ApplyOptions, PeftAdapter, PeftConfig};

use crate::qwen3_5::Qwen3_5ForCausalLM;

/// The adapter the policy trains.
pub const POLICY: &str = dash_lora::DEFAULT_ADAPTER;
/// The frozen adapter the KL penalty is measured against.
pub const REFERENCE: &str = "reference";

/// Reads the PEFT adapter in `dir` onto the model's projections as `name`. Every LoRA module in
/// the file must find its projection, at its shape.
pub fn load(
    model: &mut Qwen3_5ForCausalLM,
    dir: &Path,
    name: &str,
    trainable: bool,
    device: &Device,
) -> Result<PeftConfig, String> {
    let adapter = PeftAdapter::read(dir).map_err(|e| format!("{}: {e:?}", dir.display()))?;
    adapter
        .config
        .check_supported()
        .map_err(|e| format!("{}: {e:?}", dir.display()))?;
    let options = ApplyOptions {
        name: name.to_owned(),
        trainable,
    };
    adapter
        .apply_with(model.projections_mut(), device, &options)
        .map_err(|e| format!("{}: {e:?}", dir.display()))?;
    Ok(adapter.config.clone())
}

/// The model with `name` the adapter its forward passes run.
pub fn choose(model: Qwen3_5ForCausalLM, name: &str) -> Qwen3_5ForCausalLM {
    model.set_adapter(Some(name))
}

/// Writes adapter `name` to `dir` in PEFT's format, under `config` (the one it was read with).
pub fn save(
    model: &Qwen3_5ForCausalLM,
    name: &str,
    config: &PeftConfig,
    dir: &Path,
) -> Result<(), String> {
    let layers = model
        .projections()
        .into_iter()
        .filter(|(_, l)| l.adapter(name).is_some());
    let adapter = PeftAdapter::from_layers_named(config.clone(), name, layers)
        .map_err(|e| format!("{name}: {e:?}"))?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    adapter
        .write(dir)
        .map_err(|e| format!("{}: {e:?}", dir.display()))
}
