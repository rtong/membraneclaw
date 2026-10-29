#!/usr/bin/env python3
"""Record what transformers computes for Qwen3.5's text model, for the Rust model in rl/ to be held to.

    python rl/tools/export_qwen3_5_reference.py [out_dir] [model]   # default: fixtures Qwen/Qwen3.5-0.8B

The model runs on the CPU in float32, so the numbers are the arithmetic of transformers'
own PyTorch code paths -- the chunked and the one-step gated delta rule, the depthwise
convolution, attention -- and not of an optional GPU kernel library whose rounding
differs. The checkpoint is read from the Hugging Face cache; nothing is downloaded.

Writes, into `out_dir`:

    qwen3_5_reference.safetensors
        input_ids        [1, 100]   a fixed paragraph; 100 tokens span one full 64-token
                                    chunk of the delta rule and a partial one
        hidden.0         [100, 1024] the embeddings, the input to layer 0
        hidden.<i>       [100, 1024] the output of decoder layer i-1, for i in 1..=24
        hidden.final     [100, 1024] after the final norm
        logsumexp        [100]      of each position's logits
        target_logprob   [99]       log-probability of each next input token
        top_ids          [100, 16]  the 16 largest logits per position, and their values
        top_logits       [100, 16]
        logits_last      [2, vocab] every logit at the last two positions
        greedy_ids       [24]       greedy continuation through transformers' cache: prefill,
                                    then one token at a time
        greedy_logsumexp [24]       and, per step, its logits' logsumexp and top 16
        greedy_top_ids   [24, 16]
        greedy_top_logits [24, 16]
    qwen3_5_chat.json
        conversations rendered by the checkpoint's chat template, with the token ids the
        tokenizer gives the text
"""
import json
import os
import sys

# transformers binds flash-linear-attention's GPU kernels when it is installed, at import and
# whatever the device the model later runs on; hiding the packages keeps the CPU run on
# transformers' own PyTorch code.
for _name in ("fla", "causal_conv1d"):
    sys.modules[_name] = None

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.5-0.8B"
TOP = 16
GREEDY = 24

PARAGRAPH = (
    "The river rose through the night, and by morning the low road to the mill was under a "
    "hand's depth of brown water. Nobody in the village was surprised: the same thing had happened "
    "in each of the last three springs, always after a week of rain in the hills. The miller had "
    "moved his sacks to the loft on Tuesday. What did surprise them was the heron standing on the "
    "bridge rail, perfectly still, watching the current carry branches past as if it had been "
    "waiting for this all year."
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

CONVERSATIONS = [
    {"name": "user_only", "messages": [{"role": "user", "content": "What is 17 times 23?"}]},
    {
        "name": "system_and_turns",
        "messages": [
            {"role": "system", "content": "You answer in one sentence."},
            {"role": "user", "content": "Name a prime above 50."},
            {"role": "assistant", "content": "53 is a prime above 50."},
            {"role": "user", "content": "And one above 100?"},
        ],
    },
    {
        "name": "tool_round_trip",
        "tools": [WEATHER_TOOL],
        "messages": [
            {"role": "system", "content": "Use tools when they help."},
            {"role": "user", "content": "Is it colder in Oslo or in Lima right now?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}},
                    {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Lima", "unit": "celsius"}}},
                ],
            },
            {"role": "tool", "content": "{\"city\": \"Oslo\", \"temp_c\": -3}"},
            {"role": "tool", "content": "{\"city\": \"Lima\", \"temp_c\": 19}"},
        ],
    },
    {
        "name": "tools_no_call",
        "tools": [WEATHER_TOOL],
        "messages": [{"role": "user", "content": "Say hello in French.\nThen in German."}],
    },
]


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)

    chats = []
    for conv in CONVERSATIONS:
        for gen in (True, False):
            for thinking in (False, True):
                kwargs = {"enable_thinking": thinking}
                text = tok.apply_chat_template(conv["messages"], tools=conv.get("tools"), tokenize=False,
                                               add_generation_prompt=gen, **kwargs)
                ids = tok(text, add_special_tokens=False)["input_ids"]
                chats.append({"name": conv["name"], "messages": conv["messages"], "tools": conv.get("tools"),
                              "add_generation_prompt": gen, "enable_thinking": thinking,
                              "text": text, "ids": ids})
    with open(os.path.join(OUT, "qwen3_5_chat.json"), "w") as f:
        json.dump({"model": MODEL, "cases": chats}, f, indent=1, ensure_ascii=False)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    ids = tok(PARAGRAPH, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :100]
    assert ids.shape[1] == 100, ids.shape

    hidden = {}
    layers = model.model.layers

    def keep(name):
        def hook(_module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            hidden[name] = out[0].detach().clone()
        return hook

    layers[0].register_forward_pre_hook(lambda _m, args: hidden.__setitem__("hidden.0", args[0][0].detach().clone()))
    for i, layer in enumerate(layers):
        layer.register_forward_hook(keep(f"hidden.{i + 1}"))
    model.model.norm.register_forward_hook(keep("hidden.final"))

    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0]
    record = dict(hidden)
    hidden.clear()
    logp = torch.log_softmax(logits, dim=-1)
    top = logits.topk(TOP, dim=-1)
    record.update({
        "input_ids": ids,
        "logsumexp": torch.logsumexp(logits, dim=-1),
        "target_logprob": logp[:-1].gather(-1, ids[0, 1:, None])[:, 0],
        "top_ids": top.indices,
        "top_logits": top.values,
        "logits_last": logits[-2:].clone(),
    })

    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=GREEDY,
                             min_new_tokens=GREEDY, do_sample=False, output_scores=True, output_logits=True,
                             return_dict_in_generate=True, use_cache=True)
    step_logits = torch.stack([s[0] for s in out.logits])
    step_top = step_logits.topk(TOP, dim=-1)
    record.update({
        "greedy_ids": out.sequences[0, ids.shape[1]:],
        "greedy_logsumexp": torch.logsumexp(step_logits, dim=-1),
        "greedy_top_ids": step_top.indices,
        "greedy_top_logits": step_top.values,
    })

    record = {k: v.contiguous() for k, v in record.items()}
    save_file(record, os.path.join(OUT, "qwen3_5_reference.safetensors"),
              metadata={"model": MODEL, "torch": torch.__version__})
    print(f"wrote {len(record)} tensors and {len(chats)} chat cases to {OUT}")
    print("greedy:", repr(tok.decode(record["greedy_ids"])))


if __name__ == "__main__":
    main()
