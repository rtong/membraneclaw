"""Records notebook 07's forced probe start for the Rust harness's test (tests/forced.rs).

    python rl/tools/export_forced_reference.py out.jsonl [recovery.jsonl]

For every dev question of 07's recovery file (default: 07_recover_from_errors_data/dev_recovery.jsonl),
07's own ``rollout`` starts the episode from the question's slipped turns (``forced_prefill``) and a replayed
policy writes the recorded answer as the next turn. Written per question: the answer's tokens, the episode
(ids, policy mask, turns, calls) and its err0 delta score, provenance off.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_membrane_reference import NB, Replay, load_notebook  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1])
    recovery = Path(sys.argv[2]) if len(sys.argv) > 2 else NB.parent / "07_recover_from_errors_data" / "dev_recovery.jsonl"
    nb = load_notebook()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(nb.MODEL, padding_side="left")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    prefill = nb.forced_prefill(recovery)
    answers = {r["id"]: r["answer"] for r in map(json.loads, recovery.read_text().splitlines())}
    cases = [c for c in nb.load_split("dev") if c["id"] in prefill]
    nb.ERR_CALL_CREDIT, nb.CALL_CREDIT, nb.PROVENANCE = False, "delta", False
    pool = nb.ToolPool(1)
    lines = []
    try:
        for b in range(0, len(cases), 16):
            batch = cases[b: b + 16]
            prompts = [tuple(tok(nb.render_prompt(tok, c), add_special_tokens=False).input_ids) for c in batch]
            written = {p: [tok(answers[c["id"]], add_special_tokens=False).input_ids + [im_end]]
                       for p, c in zip(prompts, batch)}
            eps = nb.rollout(Replay(written, im_end), tok, batch, pool, temperature=1.0, max_new_tokens=3072,
                             stop_at_json=True, prefill=prefill)
            for p, ep in zip(prompts, eps):
                sc = nb.score_episode(ep)
                lines.append(json.dumps({
                    "case": ep.case, "written": written[p][0],
                    "episode": {"prompt_ids": ep.prompt_ids, "ids": ep.ids, "policy": [int(x) for x in ep.policy],
                                "turn_starts": ep.turn_starts, "turns": ep.turns,
                                "calls": [{"name": c.get("name"), "arguments": c.get("arguments"), "result": c["result"],
                                           "turn": c.get("turn")} for c in ep.calls],
                                "final_text": ep.final_text, "truncated": ep.truncated,
                                "answer_closed": ep.answer_closed},
                    "score": sc["total"]}, ensure_ascii=False))
            print(f"{min(b + 16, len(cases))}/{len(cases)}", flush=True)
    finally:
        pool.close()
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} forced episodes to {out}")


if __name__ == "__main__":
    main()
