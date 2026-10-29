"""PyTorch's per-token log-probs on recorded episodes, for the Rust forward's test (tests/logprobs.rs).

    python rl/tools/export_logprob_reference.py episodes.jsonl adapter_dir out.safetensors {bf16|fp32} [n]

Reads the first n (default 8) episodes of a file of recorded episodes (each line {"episode": {"prompt_ids",
"ids", "policy"}}, as export_forced_reference.py writes them), loads Qwen3.5-0.8B with the PEFT adapter on
CUDA in the given dtype, and records the log-prob of every policy token, `<|endoftext|>` out of the
vocabulary, as notebook 07's update reads them. bf16 runs transformers' fast path (flash-linear-attention's
kernels, as 07 trained); fp32 hides those kernels so that transformers' own PyTorch code runs.
"""

import json
import sys

mode = sys.argv[4]
if mode == "fp32":
    for name in ("fla", "causal_conv1d"):
        sys.modules[name] = None

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"


def main() -> None:
    path, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL)
    suppress = tok.convert_tokens_to_ids("<|endoftext|>")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).cuda().eval()
    model = PeftModel.from_pretrained(model, adapter).eval()
    rows = [r["episode"] for r in map(json.loads, open(path)) if "episode" in r][:n]
    record = {}
    with torch.no_grad():
        for i, ep in enumerate(rows):
            seq = torch.tensor([ep["prompt_ids"] + ep["ids"]], device="cuda")
            p = len(ep["prompt_ids"])
            where = [k for k, mine in enumerate(ep["policy"]) if mine]
            pos = torch.tensor([p + k - 1 for k in where], device="cuda")
            tgt = torch.tensor([ep["ids"][k] for k in where], device="cuda")
            logits = model(input_ids=seq, use_cache=False).logits[0, pos].float()
            logits[:, suppress] = float("-inf")
            lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt[:, None])[:, 0]
            record[f"lp.{i}"] = lp.cpu()
            print(f"episode {i}: {len(where)} policy tokens of {seq.shape[1]}", flush=True)
    save_file(record, out, metadata={"mode": mode, "episodes": str(len(rows)), "adapter": adapter})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
