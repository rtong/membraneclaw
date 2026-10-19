"""Two probes of the same dev questions, paired per question (B - A). The pre-registered activation test: PPO's
step-200 checkpoint against its seed, dev 240 x 2 at T=1; it counts only if the 95% interval excludes 0.
usage: python compare_runs.py <A dir> <B dir> [probe stem, default dev-T1-x2-stop] [A label] [B label]"""
import importlib.util, json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path

HERE = Path("/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35")
spec = importlib.util.spec_from_file_location("nb07", HERE / "07_error_recovery.py")
nb = importlib.util.module_from_spec(spec); sys.modules["nb07"] = nb; spec.loader.exec_module(nb)

a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
stem = sys.argv[3] if len(sys.argv) > 3 else "dev-T1-x2-stop"
names = [sys.argv[4] if len(sys.argv) > 4 else "A", sys.argv[5] if len(sys.argv) > 5 else "B"]
runs = dict(zip(names, (a_dir, b_dir)))
cases = {c["id"]: c for c in nb.load_split(stem.split("-T")[0])}   # the probe stem starts with its split
rows = {k: [json.loads(l) for l in (d / f"{stem}.episodes.jsonl").read_text().splitlines()] for k, d in runs.items()}
overall = {k: json.loads((d / f"{stem}.json").read_text())["overall"] for k, d in runs.items()}


def exact(ep):
    s, c = ep["score"], ep["score"]["components"]
    return bool(s["decision_correct"]) and c.get("format", 0) == 1 and c.get("violations", 1) == 1 and c.get("values", 1) == 1


reward = lambda r: r["score"]["total"]
dec = lambda r: float(bool(r["score"]["decision_correct"]))
whole = lambda r: float(exact(r))
sim_err = lambda r: any(x["name"] == cases[r["id"]]["tool"] and x["error"] for x in r["calls"])
rec = lambda r: float(nb.recovered(cases[r["id"]], r["calls"]))
conv_loop = lambda r: sum(x["name"] == "convert_units" and bool(x["error"]) for x in r["calls"]) >= 5

print(f"== {stem}: {names[0]} = {a_dir}\n   {names[1]} = {b_dir}\n   {len(rows[names[0]])} vs {len(rows[names[1]])} episodes")
print(f"   {'':44}" + "".join(f"{n:>14}" for n in names))
for label, f in [("reward", reward), ("decision right", dec), ("whole answer right", whole),
                 ("episodes, a simulate call erring", lambda r: float(sim_err(r))),
                 ("recovered (erring candidate re-simulated)", rec), ("convert loops (>=5 erroring converts)", lambda r: float(conv_loop(r)))]:
    print(f"   {label:44}" + "".join(f"{sum(map(f, rows[n])) / len(rows[n]):14.3f}" for n in names))
for label, keep in [("decision right, a simulate erring", sim_err), ("decision right, none erring", lambda r: not sim_err(r))]:
    print(f"   {label:44}" + "".join(
        f"{sum(map(dec, rs)) / max(len(rs), 1):9.3f} n={len(rs):3}" for rs in ([r for r in rows[n] if keep(r)] for n in names)))
print(f"   {'recovered, among episodes with an erring sim':44}" + "".join(
    f"{sum(map(rec, rs)) / max(len(rs), 1):14.3f}" for rs in ([r for r in rows[n] if sim_err(r)] for n in names)))
for k, label in [("answer_found", "answers"), ("converts_errored", "share of converts erroring"),
                 ("calls_errored", "share of simulate calls erroring"), ("truncated", "cut at the budget"), ("policy_tokens", "policy tokens")]:
    print(f"   {label:44}" + "".join(f"{overall[n][k]:14.3f}" for n in names))
print()


def by_q(rs, f):
    q = defaultdict(list)
    for r in rs:
        q[r["id"]].append(f(r))
    return {k: sum(v) / len(v) for k, v in q.items()}


for label, f in [("reward", reward), ("decision right", dec), ("whole answer", whole), ("simulate erring", lambda r: float(sim_err(r)))]:
    a, b = by_q(rows[names[0]], f), by_q(rows[names[1]], f)
    d = [b[k] - a[k] for k in a]
    m, se = sum(d) / len(d), st.pstdev(d) / math.sqrt(len(d))
    up, down = sum(x > 0 for x in d), sum(x < 0 for x in d)
    print(f"   {names[1]} - {names[0]}, {label:16} {m:+.3f}  95% interval {m - 1.96 * se:+.3f} to {m + 1.96 * se:+.3f}  "
          f"(questions up {up}, down {down}, same {len(d) - up - down})")
p = {}
for n in names:
    rs = [r for r in rows[n] if sim_err(r)]
    p[n] = (sum(map(dec, rs)) / len(rs), len(rs))
(pa, na), (pb, nb_) = p[names[0]], p[names[1]]
se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb_)
print(f"   {names[1]} - {names[0]}, decision right among erring episodes {pb - pa:+.3f}  95% interval {pb - pa - 1.96 * se:+.3f} to "
      f"{pb - pa + 1.96 * se:+.3f}  (n {na} vs {nb_}, unpaired)")
print()
# by family and level: where the gain sits
for key in ("family", "level"):
    print(f"   by {key}: reward {names[0]} -> {names[1]}  (decision)")
    groups = sorted({cases[r["id"]][key] for r in rows[names[0]]})
    for g in groups:
        vals = [[r for r in rows[n] if cases[r["id"]][key] == g] for n in names]
        print(f"     {str(g):18} " + "  ->  ".join(f"{sum(map(reward, v)) / len(v):.3f} ({sum(map(dec, v)) / len(v):.3f})" for v in vals)
              + f"   n={len(vals[0])}")
