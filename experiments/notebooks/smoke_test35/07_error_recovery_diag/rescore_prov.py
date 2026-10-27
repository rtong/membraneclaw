"""Re-score a saved probe under --err-call-credit 0 and --provenance 1 (an input the question states in another unit
proves its point only when a convert_units call of an earlier turn returned it): writes <tag>-err0-prov.episodes.jsonl
and <tag>-err0-prov.json next to it (reward-independent summary fields copied). The tool results are rebuilt as the
harness gave them: convert_units recomputed (deterministic), a rejected call an error, any other call accepted; the
stored score reproduces before the two flags are set.
usage: python rescore_prov.py <run dir> [tag, default dev-T1-x2-stop]"""
import importlib.util, json, sys
from pathlib import Path
HERE = Path("/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35")
spec = importlib.util.spec_from_file_location("nb07", HERE / "07_error_recovery.py")
nb = importlib.util.module_from_spec(spec); sys.modules["nb07"] = nb; spec.loader.exec_module(nb)
d = Path(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else "dev-T1-x2-stop"
cases = {c["id"]: c for split in ("train", "dev", "test", "holdout_shift") for c in nb.load_split(split)}
rows = [json.loads(l) for l in (d / f"{tag}.episodes.jsonl").read_text().splitlines()]


def result(c):
    a = c["arguments"] if isinstance(c["arguments"], dict) else {}
    if c.get("error"):
        return '{"error": "rejected"}'
    if c["name"] == nb.CONVERT:
        return nb.convert_units(a.get("value"), str(a.get("from_unit", "")), str(a.get("to_unit", "")))
    return '{"ok": 1}'


out, old, new = [], [], []
for r in rows:
    calls = [dict(c, result=result(c)) for c in r["calls"]]
    nb.ERR_CALL_CREDIT, nb.PROVENANCE = True, False
    a = nb.score(cases[r["id"]], calls, r["final_text"])
    assert abs(a["total"] - r["score"]["total"]) < 1e-9, r["id"]  # the stored score reproduces
    nb.ERR_CALL_CREDIT, nb.PROVENANCE = False, True
    b = nb.score(cases[r["id"]], calls, r["final_text"])
    old.append(a["total"]); new.append(b["total"])
    out.append(r | {"score": {k: b[k] for k in ("total", "components", "evidence", "decision_correct")}})
nb.PROVENANCE = False
(d / f"{tag}-err0-prov.episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
s = json.loads((d / f"{tag}.json").read_text())
s["overall"]["reward"] = sum(new) / len(new)
s["meta"] = s.get("meta", {}) | {"err_call_credit": False, "provenance": True, "rescored_from": f"{tag}.episodes.jsonl"}
s.pop("families", None)
(d / f"{tag}-err0-prov.json").write_text(json.dumps(s, indent=1) + "\n")
print(f"{d}/{tag}: {len(rows)} episodes, reward {sum(old) / len(old):.4f} -> err0 + provenance {sum(new) / len(new):.4f}")
