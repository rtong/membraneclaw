"""08.5: re-score saved probe episodes offline under both reward settings -- the default (err_call_credit 1, 03-08's
reward) and err0 (err_call_credit 0: a simulate call the tool rejected earns no call credit and is no evidence; the
9B-critic run's scoring). The episode total does not depend on call_credit (call vs delta only moves credit between
turns). Each call's error flag stands in for its result text. Validation: the default re-score must equal the stored
score total of every episode (default_mismatches 0). Pure Python, the training venv:
  ../../membrane_grpo/.venv/bin/python 08.5_mlx_stack_tools/rescore_probes.py <probe dir> [<probe dir> ...]
Probe dirs are relative to experiments/notebooks/smoke_test35 (or absolute). One JSON line per dir; a dir without
probes prints {}."""
import importlib.util
import json
import sys
from pathlib import Path

ST = Path(__file__).resolve().parent.parent  # experiments/notebooks/smoke_test35
ERR_RESULT, OK_RESULT = '{"error": "recorded error"}', '{"ok": true}'
TAGS = ["test-T0-n120-stop", "holdout_shift-T0-stop", "dev-T0-stop-forced", "dev-T1-n120-stop"]


def load_module():
    spec = importlib.util.spec_from_file_location("pc085_rescore", ST / "08.5_mlx_stack.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pc085_rescore"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(dirs):
    pc = load_module()
    cases = {}
    for split in ("test", "holdout_shift", "dev"):
        for c in pc.load_split(split):
            cases[c["id"]] = c
    for d in dirs:
        d = Path(d) if Path(d).is_absolute() else ST / d
        row = {}
        for tag in TAGS:
            f = d / f"{tag}.episodes.jsonl"
            if not f.exists():
                continue
            eps = [json.loads(line) for line in open(f)]
            tot1 = tot0 = 0.0
            bad = 0
            for e in eps:
                calls = [{"name": c["name"], "arguments": c["arguments"], "turn": c.get("turn"),
                          "result": ERR_RESULT if c.get("error") else OK_RESULT} for c in e["calls"]]
                case = cases[e["id"]]
                pc.ERR_CALL_CREDIT = True
                t1 = pc.score(case, calls, e["final_text"])["total"]
                pc.ERR_CALL_CREDIT = False
                t0 = pc.score(case, calls, e["final_text"])["total"]
                bad += abs(t1 - e["score"]["total"]) > 1e-9
                tot1 += t1
                tot0 += t0
            stored = json.load(open(d / f"{tag}.json"))["overall"]["reward"]
            row[tag] = {"n": len(eps), "stored": round(stored, 4), "default": round(tot1 / len(eps), 4),
                        "err0": round(tot0 / len(eps), 4), "default_mismatches": bad}
        pc.ERR_CALL_CREDIT = True
        if row:
            have = [t for t in TAGS if t in row]
            row["mean4_default"] = round(sum(row[t]["default"] for t in have) / len(have), 4)
            row["mean4_err0"] = round(sum(row[t]["err0"] for t in have) / len(have), 4)
        print(d.relative_to(ST) if d.is_relative_to(ST) else d, json.dumps(row), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
