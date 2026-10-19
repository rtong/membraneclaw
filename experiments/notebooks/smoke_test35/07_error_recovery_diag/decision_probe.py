"""Step 1 of the A diagnosis: the two fragile decisions read off every saved checkpoint, teacher-forced on fixed
contexts (no sampling noise), plus where each checkpoint's KL to its seed sits.
  (a) route: on fixed prompts (diag_common.probe_set), log-odds of a first call to convert_units over the simulate
      tool, at the token after "<tool_call>\\n<function" ("=" then "convert" vs "=s") -- and P(convert first);
  (b) the aggregation line's exit: P(line break) right after "passing: X -> <objective>: X" on 60 healthy contexts
      (step 195-205 of the original run, as agg_exit_probe.py);
  (c) on a run's own training samples next to the checkpoint (steps N..N+2, 24 episodes), the per-token log ratio
      to its seed (the trainer's KL estimate) by turn kind: route tokens, rest of the first turn, local-tool turns,
      simulate turns, the answer turn's trainable content and its frozen scaffolding.
usage: python decision_probe.py <out.jsonl> <run>=<dir>[:<steps>] ...   (steps: comma list for (c), default none)
Each checkpoint appends one JSON line to out.jsonl; checkpoints already there are skipped."""
import json, math, os, re, statistics as st, sys, time
from collections import defaultdict
import torch
from diag_common import *

out_path = Path(sys.argv[1])
specs = []
for a in sys.argv[2:]:
    run, rest = a.split("=", 1)
    d, _, kl_steps = rest.partition(":")
    specs.append((run, d, {int(s) for s in kl_steps.split(",") if s}))
done = {(j["run"], j["step"]) for j in map(json.loads, out_path.read_text().splitlines())} if out_path.exists() else set()

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained(nb.MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
IDS = route_ids(tok)
SUP = tok.convert_tokens_to_ids("<|endoftext|>")
prefix = nb.generation_prefix(tok); im_end = tok.convert_tokens_to_ids("<|im_end|>")
NL = {tok("\n\n", add_special_tokens=False).input_ids[0], tok("\n", add_special_tokens=False).input_ids[0]}
pool = CachedPool(6)
torch.cuda.set_per_process_memory_fraction(min(1.0, 6000 * 2**20 / torch.cuda.get_device_properties(0).total_memory))
base_model = AutoModelForCausalLM.from_pretrained(nb.MODEL, dtype=torch.bfloat16).cuda().eval()
model, current = None, None


def use(adapter: Path):
    global model, current
    name = "a" + "".join(ch if ch.isalnum() else "_" for ch in str(adapter.relative_to(R)))
    if name == current:
        return
    if model is None:
        model = PeftModel.from_pretrained(base_model, str(adapter), adapter_name=name).eval()
    else:
        model.load_adapter(str(adapter), adapter_name=name)
        model.set_adapter(name)
        if current:
            model.delete_adapter(current)
    current = name


def next_logprobs(texts: list[str], back: int, batch: int = 2) -> list[torch.Tensor]:
    """log-softmax (endoftext suppressed, as the trainer) at the last `back` positions of each text."""
    out = []
    base = model.get_base_model()
    for b in range(0, len(texts), batch):
        enc = [tok(t, add_special_tokens=False).input_ids for t in texts[b:b + batch]]
        w = max(map(len, enc))
        ids = torch.tensor([[tok.pad_token_id] * (w - len(e)) + e for e in enc], device="cuda")
        attn = torch.tensor([[0] * (w - len(e)) + [1] * len(e) for e in enc], device="cuda")
        with torch.no_grad():
            h = base.model(input_ids=ids, attention_mask=attn, position_ids=nb.padded_position_ids(attn), use_cache=False).last_hidden_state
            lg = base.lm_head(h[:, -back:]).float()
            lg[..., SUP] = nb.SUPPRESSED
            out += list(torch.log_softmax(lg, -1).cpu())
    return out


# (a) the route prompts
PROBES = probe_set()
route_texts = [nb.render_prompt(tok, c) + OPEN for c in PROBES]

# (b) the aggregation contexts (agg_exit_probe.py's, re-run calls cached)
agg = re.compile(r"^(passing|lowest|cheapest|highest)[^\n]*->[^\n]*$", re.M)
rows = [json.loads(l) for l in (R / "ppo-n300-turn-lr2e6-s0-to400" / "samples.jsonl").read_text().splitlines()]
pick = [r for r in rows if 195 <= r["step"] <= 205 and not r["truncated"] and r["reward"] >= 0.5
        and not any(c.get("error") for c in r["calls"]) and not any(c.get("malformed") for c in r["calls"])][:70]
agg_texts = []
for r in pick:
    t = r["turns"][-1]
    m = list(agg.finditer(t))
    if not m or "{" not in t[m[-1].end():]:
        continue
    ep = rebuild(r, tok, pool, prefix, im_end)
    ctx = tok.decode(ep.prompt_ids + ep.ids[:ep.turn_starts[-1]]) + t[: m[-1].end()]
    agg_texts.append(ctx)
agg_texts = agg_texts[:60]
print(f"{len(PROBES)} route prompts, {len(agg_texts)} aggregation contexts", flush=True)


def kl_by_kind(run_dir: Path, seed_dir: Path, step: int, adapter: Path) -> dict:
    """Per-token log ratio checkpoint / seed on the run's own samples of steps step..step+2, by token kind."""
    rows = [r for r in map(json.loads, (run_dir / "samples.jsonl").read_text().splitlines()) if step <= r["step"] < step + 3]
    if not rows:
        return None
    eps = [rebuild(r, tok, pool, prefix, im_end) for r in rows]
    B = nb.build_batch(eps, tok, "cuda")
    lps = {}
    for which, ad in (("ckpt", adapter), ("seed", seed_dir)):
        use(ad)
        lp = []
        with torch.no_grad():
            for i in range(len(eps)):
                lp.append(policy_logprobs(model, B["seqs"][i:i + 1], B["attn"][i:i + 1], B["span"], B["index"][i:i + 1], suppress=SUP)[0])
        lps[which] = torch.cat(lp)
    ratio = (lps["ckpt"] - lps["seed"]) * B["mask"]
    sums, counts = defaultdict(float), defaultdict(int)
    for i, ep in enumerate(eps):
        route = set(route_positions(ep, tok, IDS))
        n = int(B["mask"][i].sum())
        for k in range(n):
            j = int(B["turn"][i][k])
            kind = turn_kind(ep, j)
            if k in route:
                key = "route"
            elif j == 0:
                key = "first turn, rest"
            elif kind == "answer":
                key = "answer content" if B["content"][i][k] > 0 else "answer scaffolding (frozen)"
            else:
                key = kind + " turns"
            sums[key] += float(ratio[i][k]); counts[key] += 1
    tot = sum(sums.values())
    return {k: {"tokens_per_ep": counts[k] / len(eps), "mean_log_ratio": sums[k] / counts[k], "share_of_kl": sums[k] / tot if tot else None}
            for k in sums} | {"_episodes": len(eps), "_kl_per_token": tot / sum(counts.values())}


for run, d, kl_steps in specs:
    d = R / d
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    seed_dir = HERE / cfg["start_from"] if cfg else d
    ckpts = sorted(((int(p.name.split("step")[1]), p) for p in d.glob("adapter-step*")), key=lambda x: x[0]) if cfg else [(0, d)]
    only = {int(x) for x in os.environ.get("ONLY_STEPS", "").split(",") if x}
    for step, adapter in ckpts:
        if (run, step) in done or (only and step not in only):
            continue
        t0 = time.time()
        use(adapter)
        lp = next_logprobs(route_texts, 2)
        res = {"run": run, "step": step, "adapter": str(adapter.relative_to(R))}
        groups = defaultdict(list)
        for c, l in zip(PROBES, lp):
            p_eq, p_eqs = l[0, IDS["eq"]].item(), l[0, IDS["eqs"]].item()
            conv = p_eq + l[1, IDS["convert"]].item()
            g = ("train " if c.get("probe_split") else "dev ") + route_group(c["id"])
            groups[g].append((conv - p_eqs, math.exp(conv)))
        res["route"] = {g: {"n": len(v), "log_odds": st.mean(x for x, _ in v), "p_convert": st.mean(p for _, p in v)} for g, v in sorted(groups.items())}
        la = next_logprobs(agg_texts, 1)
        pe = sorted(sum(math.exp(l[0, t].item()) for t in NL) for l in la)
        res["agg_exit"] = {"mean": st.mean(pe), "median": pe[len(pe) // 2], "below_half": sum(x < 0.5 for x in pe)}
        if step in kl_steps:
            if (kl := kl_by_kind(d, seed_dir, step, adapter)) is not None:
                res["kl"] = kl
        with out_path.open("a") as fh:
            fh.write(json.dumps(res) + "\n")
        r2 = res["route"]
        print(f"{run:6} {step:3}  route log-odds " + "  ".join(f"{g}: {v['log_odds']:+.2f}" for g, v in r2.items())
              + f" | agg exit {res['agg_exit']['mean']:.3f}" + (f" | kl/token {res['kl']['_kl_per_token']:.4f}" if "kl" in res else "")
              + f"  ({time.time() - t0:.0f}s)", flush=True)
