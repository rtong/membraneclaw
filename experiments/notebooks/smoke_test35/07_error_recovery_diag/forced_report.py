"""Forced probe (PR #33's 144 dev unit slips, greedy): PR #33's measures plus how each power-of-ten retry was made.
usage: python forced_report.py <label=run dir> ...   (each dir holds dev-T0-stop-forced.episodes.jsonl)"""
import importlib.util, json, math, os, sys
from collections import Counter
from pathlib import Path

HERE = Path("/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35")
spec = importlib.util.spec_from_file_location("nb07", HERE / "07_error_recovery.py")
nb = importlib.util.module_from_spec(spec); sys.modules["nb07"] = nb; spec.loader.exec_module(nb)
nb.result_is_error = lambda r: r == "ERR"
dev = {c["id"]: c for c in nb.load_split("dev")}
STEM = os.environ.get("STEM", "dev-T0-stop-forced")
need = {q: {t[1] for p in nb.convert_plan(c, limits=False) for t in p["targets"] if t[0] == "arg"} for q, c in dev.items()}


def pr33_stats(calls):  # PR #33's recovery_stats, on the recorded error flag
    sims = [(c, bool(c["error"])) for c in calls if c["name"] not in nb.LOCAL_TOOLS]
    errs = [e for _, e in sims]
    first = errs.index(True) if True in errs else None
    seen, repeats, streak, run = set(), 0, 0, 0
    for c, e in sims:
        key = json.dumps([c.get("name"), c.get("arguments")], sort_keys=True, default=str)
        repeats += key in seen
        seen.add(key)
        run = run + 1 if e else 0
        streak = max(streak, run)
    return {"erred": first is not None, "recovered": first is not None and not all(errs[first:]), "repeats": repeats, "streak": streak}


def pow10_retries(case, calls):
    out = []
    idx = [i for i, c in enumerate(calls) if c["name"] == case["tool"] and isinstance(c["arguments"], dict)]
    for a, i in enumerate(idx):
        e = calls[i]
        if not e["error"]:
            continue
        for j in idx[a + 1:]:
            c = calls[j]
            A, B = e["arguments"], c["arguments"]
            diff = [k for k in set(A) | set(B) if A.get(k) != B.get(k)]
            if len(diff) != 1:
                continue
            k = diff[0]
            try:
                n = math.log10(float(B[k]) / float(A[k]))
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                break
            if round(n) != 0 and abs(n - round(n)) < 0.01:
                via = any(x["name"] == nb.CONVERT and isinstance(x["arguments"], dict)
                          and abs(float(x["arguments"].get("value", 0) or 0) - float(A[k])) <= 1e-9 * max(1, abs(float(A[k])))
                          for x in calls[i + 1:j])
                cov = nb.call_fidelity(case, {**c, "result": "ERR" if c["error"] else "OK"})[1]
                out.append(("convert" if via else "direct", "right" if cov else "clean-wrong" if not c["error"] else "errs"))
            break
    return out


print(f"{'':30}{'n':>4} {'reward':>7} {'decision':>8} {'evidence':>8} {'recovered':>9} {'repeats/ep':>10} {'3+ err run':>10} | "
      f"10^k retries: direct (right/clean-wrong/errs)  via convert (right/clean-wrong/errs)")
for arg in sys.argv[1:]:
    label, d = arg.split("=", 1)
    eps = [json.loads(l) for l in (Path(d) / f"{STEM}.episodes.jsonl").read_text().splitlines()]
    n = len(eps)
    st = [pr33_stats(e["calls"]) for e in eps]
    er = [s for s in st if s["erred"]]
    rt = Counter(x for e in eps for x in pow10_retries(dev[e["id"]], e["calls"]))
    f = lambda route: f"{sum(v for (r, _), v in rt.items() if r == route):3} ({rt[(route, 'right')]}/{rt[(route, 'clean-wrong')]}/{rt[(route, 'errs')]})"
    print(f"{label:30}{n:4} {sum(e['score']['total'] for e in eps) / n:7.3f} {sum(e['score']['decision_correct'] for e in eps) / n:8.3f} "
          f"{sum(bool(e['score']['evidence']) for e in eps) / n:8.3f} {sum(s['recovered'] for s in er) / max(1, len(er)):9.3f} "
          f"{sum(s['repeats'] for s in st) / n:10.2f} {sum(s['streak'] >= 3 for s in st) / n:10.3f} | "
          f"direct {f('direct')}   convert {f('convert')}")
print("\nPR #33's own forced table (their machine): control (06's 300 retrained) recovered 0.285, reward 0.401, decision 0.292, "
      "repeats 4.97, 3+ run 0.868; seed-rt (effa2e4's recipe retrained) 0.562 / 0.551 / 0.444 / 4.06 / 0.785; "
      "diagnosis-line seed 0.812 / 0.686 / 0.632 / 0.67 / 0.576")
