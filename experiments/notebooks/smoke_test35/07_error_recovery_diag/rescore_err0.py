"""Re-score a saved probe under --err-call-credit 0 (a simulate call the tool rejected earns no call credit):
writes <tag>-err0.episodes.jsonl and <tag>-err0.json next to it (reward-independent summary fields copied).
usage: python rescore_err0.py <run dir> [tag, default dev-T1-x2-stop]"""
import importlib.util, json, sys
from pathlib import Path
HERE = Path("/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35")
spec = importlib.util.spec_from_file_location("nb07", HERE / "07_error_recovery.py")
nb = importlib.util.module_from_spec(spec); sys.modules["nb07"] = nb; spec.loader.exec_module(nb)
d = Path(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else "dev-T1-x2-stop"
cases = {c["id"]: c for split in ("train", "dev", "test", "holdout_shift") for c in nb.load_split(split)}
rows = [json.loads(l) for l in (d / f"{tag}.episodes.jsonl").read_text().splitlines()]
out, old, new = [], [], []
for r in rows:
    calls = [dict(c, result='{"error": 1}' if c.get("error") else "{}") for c in r["calls"]]
    nb.ERR_CALL_CREDIT = True
    a = nb.score(cases[r["id"]], calls, r["final_text"])
    assert abs(a["total"] - r["score"]["total"]) < 1e-9, r["id"]  # the stored score reproduces
    nb.ERR_CALL_CREDIT = False
    b = nb.score(cases[r["id"]], calls, r["final_text"])
    old.append(a["total"]); new.append(b["total"])
    out.append(r | {"score": {k: b[k] for k in ("total", "components", "evidence", "decision_correct")}})
(d / f"{tag}-err0.episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
s = json.loads((d / f"{tag}.json").read_text())
s["overall"]["reward"] = sum(new) / len(new)
s["meta"] = s.get("meta", {}) | {"err_call_credit": False, "rescored_from": f"{tag}.episodes.jsonl"}
s.pop("families", None)
(d / f"{tag}-err0.json").write_text(json.dumps(s, indent=1) + "\n")
print(f"{d}/{tag}: {len(rows)} episodes, reward {sum(old) / len(old):.4f} -> err0 {sum(new) / len(new):.4f}")
