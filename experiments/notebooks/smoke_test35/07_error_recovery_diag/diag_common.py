"""Shared helpers for the diagnosis of why A (and the other runs) lost "convert first" after step 200:
the fixed probe prompts, the route decision tokens, a disk-cached tool pool, and rebuilding a stored training
sample (samples.jsonl) into the Episode the trainer built, so build_batch gives its exact rewards and masks."""
import importlib.util, json, random, sys
from pathlib import Path

HERE = Path("/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35")
spec = importlib.util.spec_from_file_location("nb07", HERE / "07_error_recovery.py")
nb = importlib.util.module_from_spec(spec); sys.modules["nb07"] = nb; spec.loader.exec_module(nb)
R = HERE / "07_error_recovery_runs"
SC = Path(__file__).resolve().parent
CASES = {c["id"]: c for split in ("train", "dev") for c in nb.load_split(split)}
NEED = {q: {t[1] for p in nb.convert_plan(c, limits=False) for t in p["targets"] if t[0] == "arg"} for q, c in CASES.items()}
# A first call is "<tool_call>\n<function=convert_units>" or "...<function=simulate_ro>": after "<function" the
# tokenizer writes "=" then "convert" (or "check"), or "=s" then "im" "ulate". The route is decided at the token
# after "function" ("=" vs "=s") and, after "=", at "convert" vs "check".
OPEN = "<tool_call>\n<function="


def route_group(q: str) -> str | None:
    c = CASES[q]
    if c["level"] == 0:
        return "L0"
    if not NEED[q]:
        return None
    if c["level"] == 1:
        return "L1 one unit" if len(NEED[q]) == 1 else "L1 2+ units"
    return "L2"


def probe_set() -> list[dict]:
    """Fixed prompts for P(convert first): every dev L1/L2 question that needs a conversion, 60 dev L0 questions
    (they need none), and 96 train L1/L2 questions (the first 64 of them are the attribution's probe)."""
    rng = random.Random(0)
    dev = nb.load_split("dev")
    out = [c for c in dev if route_group(c["id"]) not in (None, "L0")]
    out += rng.sample([c for c in dev if c["level"] == 0], 60)
    train = [c for c in nb.load_split("train") if route_group(c["id"]) not in (None, "L0")]
    return out + [dict(c, probe_split="train") for c in rng.sample(train, 96)]


def route_ids(tok) -> dict[str, int]:
    ids = {name: tok(OPEN[:-1] + s, add_special_tokens=False).input_ids for name, s in
           (("eq", "=convert_units"), ("eqs", "=simulate_ro"))}
    base = tok(OPEN[:-1], add_special_tokens=False).input_ids
    eq, eqs = ids["eq"][len(base)], ids["eqs"][len(base)]
    conv = tok(OPEN + "convert_units", add_special_tokens=False).input_ids[len(tok(OPEN, add_special_tokens=False).input_ids)]
    check = tok(OPEN + "check_limits", add_special_tokens=False).input_ids[len(tok(OPEN, add_special_tokens=False).input_ids)]
    assert tok.decode([eq]) == "=" and tok.decode([eqs]) == "=s" and tok.decode([conv]) == "convert", (eq, eqs, conv)
    return {"eq": eq, "eqs": eqs, "convert": conv, "check": check}


class CachedPool:
    """nb.ToolPool behind a disk cache (the simulators are deterministic): rebuilt episodes re-run their calls."""

    def __init__(self, workers: int = 6):
        self.path = SC / "tool_cache.jsonl"
        self.cache = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                k, v = json.loads(line)
                self.cache[k] = v
        self.pool, self.workers = None, workers

    def call_many(self, jobs):
        keys = [nb.ToolPool.key(n, a) for n, a in jobs]
        miss = sorted({k: i for i, k in enumerate(keys) if k not in self.cache}.values())
        if miss:
            if self.pool is None:
                self.pool = nb.ToolPool(self.workers)
            res = self.pool.call_many([jobs[i] for i in miss])
            with self.path.open("a") as fh:
                for i, r in zip(miss, res):
                    self.cache[keys[i]] = r
                    fh.write(json.dumps([keys[i], r]) + "\n")
        return [self.cache[k] for k in keys]


def rebuild(r: dict, tok, pool: CachedPool, prefix: str, im_end: int) -> "nb.Episode":
    """A stored training sample as the Episode rollout built: its turns re-tokenized (the stored text is the
    decoded turn), each executed turn's calls re-run (with rollout's budget rules), the tool responses between."""
    case = CASES[r["id"]]
    ep = nb.Episode(case, tok(nb.render_prompt(tok, case), add_special_tokens=False).input_ids)
    executed = {c["turn"] for c in r["calls"]}
    last = len(r["turns"]) - 1
    cut_in_turn = r["truncated"] and r["final_text"] == r["turns"][-1]
    n_sims = n_conv = n_comp = 0
    for j, text in enumerate(r["turns"]):
        ids = tok(text, add_special_tokens=False).input_ids
        ends = not (j == last and (r["answer_closed"] or cut_in_turn))
        take = ids + ([im_end] if ends else [])
        ep.turn_starts.append(len(ep.ids))
        ep.ids += take
        ep.policy += [True] * len(take)
        ep.turns.append(text)
        if j not in executed:
            continue
        calls = nb.parse_calls(text)
        jobs, where = [], []
        for c in calls:
            if c.get("malformed"):
                c["result"] = nb.MALFORMED_CALL_ERROR
            elif c.get("name") == nb.CONVERT:
                n_conv += 1
                if n_conv > nb.MAX_CONVERTS:
                    c["result"] = nb.CONVERT_BUDGET_ERROR
                else:
                    jobs.append((c["name"], c["arguments"])); where.append(c)
            elif c.get("name") == nb.COMPARE:
                n_comp += 1
                if n_comp > nb.MAX_COMPARES:
                    c["result"] = nb.COMPARE_BUDGET_ERROR
                else:
                    jobs.append((c["name"], c["arguments"])); where.append(c)
            else:
                n_sims += 1
                if n_sims > nb.MAX_CALLS:
                    c["result"] = nb.CALL_BUDGET_ERROR
                else:
                    jobs.append((c["name"], c["arguments"])); where.append(c)
        for c, res in zip(where, pool.call_many(jobs)):
            c["result"] = res
        for c in calls:
            c["turn"] = j
            ep.calls.append(c)
        env = tok(nb.tool_response_text([c["result"] for c in calls], prefix), add_special_tokens=False).input_ids
        ep.ids += env
        ep.policy += [False] * len(env)
    ep.final_text, ep.truncated = r["final_text"], r["truncated"]
    ep.unanswered, ep.answer_closed = r["unanswered"], r["answer_closed"]
    return ep


def route_positions(ep: "nb.Episode", tok, ids: dict[str, int]) -> list[int]:
    """Compacted (policy-token) indices of the route decision in the first turn: the token after "function"
    and, when it is "=", the next one ("convert" / "check")."""
    if not ep.turns or not ep.turns[0].startswith(OPEN[:-1]):
        return []
    head = tok(OPEN[:-1], add_special_tokens=False).input_ids
    mine = [t for t, p in zip(ep.ids, ep.policy) if p]
    if mine[:len(head)] != head:
        return []
    k = len(head)
    return [k, k + 1] if mine[k] == ids["eq"] else [k]


def turn_kind(ep: "nb.Episode", j: int) -> str:
    names = {c.get("name") for c in ep.calls if c.get("turn") == j}
    if not names:
        return "answer" if not nb.parse_calls(ep.turns[j]) else "unexecuted calls"
    return "simulate" if names - set(nb.LOCAL_TOOLS) else "local tools"


def policy_logprobs(model, seqs, attn, span: int, index, *, suppress: int, hidden_layer: int | None = None, chunk: int = 256):
    """nb.forward_policy's log-probs (and critic features) with the vocabulary log-softmax taken in chunks of
    positions, each recomputed in backward (torch.utils.checkpoint): a whole episode's (n x 248k) softmax is ~3 GB."""
    import torch
    from torch.utils.checkpoint import checkpoint
    base = model.get_base_model()
    out = base.model(input_ids=seqs, attention_mask=attn, position_ids=nb.padded_position_ids(attn),
                     output_hidden_states=hidden_layer is not None, use_cache=False)
    offset = seqs.shape[1] - span
    pos = offset + index - 1
    h = nb._take(out.last_hidden_state, pos)
    targets = nb._take(seqs[:, offset:], index)

    def piece(hc, tc):
        lg = base.lm_head(hc).float()
        lg[..., suppress] = nb.SUPPRESSED
        return torch.log_softmax(lg, dim=-1).gather(-1, tc.unsqueeze(-1)).squeeze(-1)

    parts = []
    for s in range(0, h.shape[1], chunk):
        hc, tc = h[:, s:s + chunk], targets[:, s:s + chunk]
        parts.append(checkpoint(piece, hc, tc, use_reentrant=False) if torch.is_grad_enabled() else piece(hc, tc))
    lp = torch.cat(parts, dim=1)
    hidden = nb._take(out.hidden_states[hidden_layer], pos).detach() if hidden_layer is not None else None
    return lp, hidden
