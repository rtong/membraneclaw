# rl

RL training of MembraneClaw's tool-using model in Rust, on [dashmag](https://github.com/bayanasar/dashmag)
and Burn. dashmag is the compute layer (operators with their backward and decode-step forms, LoRA,
recompute, losses, AdamW, clipping); everything above it lives here: the model, the generation loop,
the GRPO trainer and the task.

| Path | Purpose |
| --- | --- |
| `src/qwen3_5.rs` | Qwen3.5's text model assembled from Burn and dashmag operators (prefill, decode, batched decode state); loads the published checkpoint strictly |
| `src/lora.rs` | PEFT adapters on one frozen base: the policy, the frozen reference, controls |
| `src/generate.rs` | Sampling with continuous batching: each episode continues when its own tools answer |
| `src/tree.rs` | The update over a prefix tree of the batch's episodes: log-probs, and an exact layer-by-layer backward |
| `src/grpo.rs` | Group advantages, the minibatch z-score, error states, the step's RNG |
| `src/probe.rs` | The held-out probes and their paired comparisons |
| `src/chat.rs` | The checkpoint's chat template and tokenizer, rendered as transformers' `apply_chat_template` renders them |
| `src/hf.rs` | Checkpoints in the local Hugging Face cache (nothing is downloaded) |
| `src/membrane/` | Notebook 07's task: the harness that runs an episode, the tools, the reward, the rows the trainer reads |
| `src/bin/nb10.rs` | Notebook 10's GRPO run (`nb10 train`) |
| `tools/export_*.py` | One-time PyTorch / 07 reference records for the tests |

The simulators stay in Python: WaterTAP has no other implementation. Every tool call goes to notebook 07's
own worker processes (`07_error_recovery.py worker` under `.venv-watertap`), one JSON object per line.

## Checks

```sh
cd rl
cargo test --release                                   # unit tests, CPU
# against the reference records (see each test's `ignore` reason for its variables)
cargo test --release --features cuda -- --ignored --test-threads=1
```

The ignored tests hold the chat template and tokenizer to transformers' text and token ids; every layer
and the logits to transformers' float32 run; greedy decoding with carried state to transformers'; the
update's log-probs on recorded episodes to PyTorch's (bf16 noise no larger than PyTorch's own); the prefix
tree's gradient to each episode run whole; and, on recorded episodes, the harness, the reward and the
trainer's rows to 07's own Python, bit for bit.
