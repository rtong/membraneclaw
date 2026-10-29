#!/usr/bin/env python3
"""Record what notebook 07's harness computes on real episodes, for rl/'s environment to be held to.

    python rl/tools/export_membrane_reference.py <samples.jsonl> <out.jsonl> [n] [seed]

Run it with a Python that has torch and transformers (the series' training venv); the simulators run in
07's own tool workers under .venv-watertap. Nothing is trained or sampled. n episodes recorded in a PPO
run's samples.jsonl (answered, unanswered and truncated ones alike) are replayed through 07's `rollout`
by a policy that writes each episode's recorded turns again, so the tools, the budgets, the rounds and
the stop at the answer's closing brace act on them as they did in the run. A turn the budget cut off is
written on to the budget, as sampling would.

The first line holds what every episode shares: the tokenizer's special ids, the text 07 appends after a
tool response, the reward settings -- and probes: turns written by hand where recorded ones are thin (calls
that do not parse, parameters typed against the schema, answers that are almost JSON), each with what
07's `parse_calls` and `extract_answer` make of it. Each further line is one episode: its case, the turns as the policy
wrote them, the episode that came out, its score under every reward setting 07 has (err_call_credit x
call_credit x provenance), and build_batch's rows for it alone, the rewards per setting.
"""
import importlib.util
import json
import random
import sys
from pathlib import Path

NB = Path(__file__).resolve().parents[2] / "experiments" / "notebooks" / "smoke_test35" / "07_error_recovery.py"
SETTINGS = [(e, c, p) for e in (True, False) for c in ("call", "delta") for p in (False, True)]
PROBES = [
    "<tool_call>\n<function=simulate_ro>\n<parameter=feed_pressure_bar>\n55\n</parameter>\n</tool_call>",
    "<tool_call>\n<function=simulate_ro>\n<parameter=feed_pressure_bar>\n55\n</parameter>\n</function>\n</tool_call>",
    "<tool_call><function=simulate_ro></function></tool_call> and <tool_call>junk</tool_call>",
    "<tool_call>\n<function=simulate_ro>\n<parameter=membrane_area_m2>\n1_000\n</parameter>\n"
    "<parameter=membrane_area_m2>\n 2e3 \n</parameter>\n<parameter=feed_temperature_c>\nNone\n</parameter>\n"
    "<parameter=A_comp>\nabout 4e-12\n</parameter>\n<parameter=concentration_polarization>\n  fixed \n</parameter>\n"
    "<parameter=colour>\n{\"a\": 1}\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=check_limits>\n<parameter=limits>\n{\"permeate.nacl_ppm\": \"<= 500 mg/L\", "
    "\"flux.water_LMH\": \"<= 20 gfd\"}\n</parameter>\n<parameter=values>\n{\"A\": {\"permeate.nacl_ppm\": 212.5}}\n"
    "</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=check_limits>\n<parameter=limits>\n{bad json\n"
    "</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=convert_units>\n<parameter=value>\ninf\n</parameter>\n<parameter=from_unit>\n°F\n"
    "</parameter>\n<parameter=to_unit>\nC\n</parameter>\n</function>\n</tool_call>",
    "Thinking.\n<tool_call>\n<function=no_such_tool>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\ntrailing",
    'The pick is B.\n{"decision": "B", "violations": [], "values": {"permeate": {"flow_kg_s": "1.2 kg/s"}}}',
    '{"note": "no decision"} then {"decision": null, "violations": [], "values": {}} trailing words',
    '{"decision": "pass", "values": {"x": NaN}, "violations": ["a", "a"], "decision": "fail"}',
    '{"decision": "a \\"quoted\\" {brace}", "violations": []}\n',
    '{"decision": "pass", "violations": [],}',
    'no json here at all, just 3.5 bar',
    '{"decision": "°F → 20 m³/h ‰", "violations": [], "values": {}}  ',
]


def setting_name(e: bool, c: str, p: bool) -> str:
    return f"err{int(e)}-{c}-prov{int(p)}"


def load_notebook():
    spec = importlib.util.spec_from_file_location("nb07", NB)
    nb = importlib.util.module_from_spec(spec)
    sys.modules["nb07"] = nb
    spec.loader.exec_module(nb)
    return nb


class Replay:
    """A policy that writes recorded turns: for each episode in the batch, its next turn."""

    def __init__(self, written: dict[tuple, list], im_end: int):
        self.written, self.im_end = written, im_end
        self.next = {k: 0 for k in written}
        self.prompts = list(written)

    def parameters(self):
        import torch
        yield torch.zeros(1)

    def generate(self, input_ids, attention_mask, max_new_tokens, pad_token_id, **_):
        import torch
        rows = []
        for ids, att in zip(input_ids.tolist(), attention_mask.tolist()):
            seq = [t for t, a in zip(ids, att) if a]
            key = next(p for p in self.prompts if tuple(seq[: len(p)]) == p)
            turns, k = self.written[key], self.next[key]
            self.next[key] += 1
            if k >= len(turns):          # the replay asks for more than the run wrote: an empty turn
                new = [self.im_end]
            elif isinstance(turns[k], dict):  # cut off by the budget: written on to it
                fill = turns[k]["fill"] or [self.im_end]
                new = (fill * (max_new_tokens // len(fill) + 1))[:max_new_tokens]
            else:
                new = turns[k][:max_new_tokens]
            rows.append(ids + new + [pad_token_id] * (max_new_tokens - len(new)))
        return torch.tensor(rows)


def written_turns(rec: dict, tok, im_end: int) -> list:
    last = len(rec["turns"]) - 1
    cut_off = rec["truncated"] and not any(c["turn"] == last for c in rec["calls"])
    out = []
    for j, text in enumerate(rec["turns"]):
        ids = tok(text, add_special_tokens=False).input_ids
        out.append({"fill": ids} if j == last and cut_off else ids + [im_end])
    return out


def pick(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Answered, unanswered and truncated episodes in the proportions 5:2:1, across the run's steps."""
    kinds = {"answered": [r for r in rows if not r["truncated"] and not r["unanswered"]],
             "unanswered": [r for r in rows if r["unanswered"]],
             "truncated": [r for r in rows if r["truncated"]]}
    want = {"answered": n * 5 // 8, "unanswered": n * 2 // 8}
    want["truncated"] = n - sum(want.values())
    out = []
    for kind, k in want.items():
        out += rng.sample(kinds[kind], min(k, len(kinds[kind])))
    rng.shuffle(out)
    return out


def main() -> None:
    samples, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 160
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    nb = load_notebook()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(nb.MODEL, padding_side="left")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    cases = {c["id"]: c for split in ("train", "dev", "test", "holdout_shift") for c in nb.load_split(split)}
    rows = [json.loads(line) for line in samples.read_text().splitlines()]
    chosen = pick(rows, n, random.Random(seed))
    run_cfg = json.loads((samples.parent / "config.json").read_text())
    max_new = run_cfg.get("max_new_tokens", 3072)
    stop_at_json = bool(run_cfg.get("stop_at_json", 1))

    pool = nb.ToolPool(4)
    lines = [json.dumps({"all_special_ids": sorted(tok.all_special_ids), "pad_token_id": tok.pad_token_id,
                         "generation_prefix": nb.generation_prefix(tok), "max_new_tokens": max_new,
                         "stop_at_json": stop_at_json, "settings": [setting_name(*s) for s in SETTINGS],
                         "probes": [{"text": t, "calls": nb.parse_calls(t), "answer": list(nb.extract_answer(t))}
                                    for t in PROBES],
                         "run_settings": setting_name(bool(run_cfg.get("err_call_credit", 1)),
                                                      run_cfg.get("call_credit", "call"),
                                                      bool(run_cfg.get("provenance", 0)))})]
    same_reward = 0
    try:
        for b in range(0, len(chosen), 8):
            recs = chosen[b: b + 8]
            batch = [cases[r["id"]] for r in recs]
            prompts = [tuple(tok(nb.render_prompt(tok, c), add_special_tokens=False).input_ids) for c in batch]
            written = {p: written_turns(r, tok, im_end) for p, r in zip(prompts, recs)}
            eps = nb.rollout(Replay(written, im_end), tok, batch, pool, temperature=1.0, max_new_tokens=max_new,
                             stop_at_json=stop_at_json)
            for rec, p, ep in zip(recs, prompts, eps):
                scores, rewards = {}, {}
                for e, c, pv in SETTINGS:
                    nb.ERR_CALL_CREDIT, nb.CALL_CREDIT, nb.PROVENANCE = e, c, pv
                    name = setting_name(e, c, pv)
                    scores[name] = nb.score_episode(ep)
                    B = nb.build_batch([ep], tok, "cpu")
                    rewards[name] = B["rewards"][0].tolist()
                keep = {k: B[k][0].tolist() for k in ("index", "ids", "mask", "answer", "turn_end", "content", "turn")}
                run = json.loads(lines[0])["run_settings"]
                same_reward += abs(scores[run]["total"] - rec["reward"]) < 1e-12
                lines.append(json.dumps({
                    "case": ep.case, "written": written[p], "recorded_reward": rec["reward"],
                    "episode": {"prompt_ids": ep.prompt_ids, "ids": ep.ids, "policy": [int(x) for x in ep.policy],
                                "turn_starts": ep.turn_starts, "turns": ep.turns,
                                "calls": [{"name": c.get("name"), "arguments": c.get("arguments"), "result": c["result"],
                                           "turn": c.get("turn"), "malformed": bool(c.get("malformed"))}
                                          for c in ep.calls],
                                "final_text": ep.final_text, "truncated": ep.truncated, "unanswered": ep.unanswered,
                                "answer_closed": ep.answer_closed},
                    "scores": scores, "batch": keep, "rewards": rewards}, ensure_ascii=False))
            print(f"{min(b + 8, len(chosen))}/{len(chosen)} episodes", flush=True)
    finally:
        pool.close()
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(chosen)} episodes to {out_path}; the replay's reward equals the run's on {same_reward} "
          f"({len(chosen) - same_reward} differ: re-tokenized turns can spend the token budget differently)")


if __name__ == "__main__":
    main()
