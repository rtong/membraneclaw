//! PPO training of MembraneClaw's tool-using model in Rust.
//!
//! The operators, the LoRA layer, AdamW and gradient clipping come from dashmag, on Burn.
//! What is here is everything above them: Qwen3.5's text model assembled from those operators
//! ([`qwen3_5`]) with its PEFT adapters ([`lora`]), its tokenizer and chat template
//! ([`chat`]), reading checkpoints from the Hugging Face cache ([`hf`]), the generation loop
//! ([`generate`]), and the task's environment ([`membrane`]).

pub mod chat;
pub mod generate;
pub mod grpo;
pub mod hf;
pub mod lora;
pub mod membrane;
pub mod probe;
pub mod qwen3_5;
pub mod tree;
