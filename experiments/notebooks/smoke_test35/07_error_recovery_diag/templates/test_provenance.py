"""CPU check of --provenance (PROVENANCE: a simulate call proves its point only with inputs convert_units converted).
usage: python test_provenance.py <trainer with the flag> <trainer before it>
  1. Off, score() is bit-identical to the trainer before the flag (every field: total, components, evidence, covered,
     credits) on the PPO runs' training samples (err0 as trained; both call credits) and the step-200 probe episodes,
     and it reproduces each sample's stored reward -- so the rebuilt tool results below are the ones the runs saw.
  2. On, the 2,400 reference episodes (06's clean ones, 07's recovery ones) score exactly as off: every conversion
     they pass came from the tool a turn earlier.
  3. Hand-built calls: an input converted in the head proves nothing; a tool conversion of an earlier turn does
     (also rounded within ARG_TOL); one of the same turn, an erroring one, or a later one does not; a question with
     nothing to convert is unchanged; calls without a turn count as a turn each. A bad value is refused."""
import importlib.util, json, sys, types
from pathlib import Path

HERE = Path("/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35")
R = HERE / "07_error_recovery_runs"


def load(path, name):
    m = types.ModuleType(name); m.__file__ = str(HERE / "07_error_recovery.py")   # its data paths resolve from here
    sys.modules[name] = m
    exec(compile(Path(path).read_text(), name, "exec"), m.__dict__)
    return m


nb, old = load(sys.argv[1], "nb_new"), load(sys.argv[2], "nb_old")
assert hasattr(nb, "PROVENANCE") and not hasattr(old, "PROVENANCE")
CASES = {c["id"]: c for s in ("train", "dev", "test", "holdout_shift") for c in nb.load_split(s)}


def rebuilt(calls):
    """A stored episode's calls with tool results: convert_units recomputed (deterministic), the others by outcome."""
    out = []
    for c in calls:
        a = c["arguments"]
        if c["error"]:
            res = '{"error": "rejected"}'
        elif c["name"] == nb.CONVERT:
            res = nb.convert_units(a.get("value"), str(a.get("from_unit", "")), str(a.get("to_unit", "")))   # as the harness
        else:
            res = '{"ok": 1}'
        out.append({"name": c["name"], "arguments": a, "result": res, "turn": c.get("turn")})
    return out


def both(fn):
    for m in (nb, old):
        m.ERR_CALL_CREDIT = False
    nb.PROVENANCE = False
    return fn(nb), fn(old)


# 1. off == before, on the runs' samples and probes
RUNS = ["ppo-n300-maskend-turn-lr2e6-err0-s0-to400", "ppo-n300-maskend-turn-lr2e6-err0-delta-td0-s0",
        "ppo-n300-maskend-turn-lr2e6-err0-delta-td05-s0", "ppo-n300-maskend-turn-lr2e6-err0-delta-alam05-s0"]
n = bad_stored = bad_convert = 0
for run in RUNS:
    for line in (R / run / "samples.jsonl").read_text().splitlines():
        r = json.loads(line); case = CASES[r["id"]]; calls = rebuilt(r["calls"])
        bad_convert += sum(c["name"] == nb.CONVERT and not s["error"] and nb.result_is_error(c["result"]) for c, s in zip(calls, r["calls"]))
        for mode in ("call", "delta"):
            nb.CALL_CREDIT = old.CALL_CREDIT = mode
            a, b = both(lambda m: m.score(case, calls, r["final_text"]))
            assert a == b, (run, r["step"], r["id"], mode)
        bad_stored += abs(a["total"] - r["reward"]) > 1e-12
        n += 1
nb.CALL_CREDIT = old.CALL_CREDIT = "call"
print(f"  1. training samples: {n} episodes x 2 call credits, off == before (every field); stored reward reproduced on "
      f"{n - bad_stored}/{n}; recomputed convert results erroring where the run's did not: {bad_convert}")
assert bad_stored == 0 and bad_convert == 0
m = 0
for d in ("ppo-n300-maskend-turn-lr2e6-err0-s0/step200", "ppo-n300-turn-lr2e6-s0/step200",
          "ppo-n300-maskend-turn-lr2e6-err0-delta-alam05-s0/step200"):
    for p in sorted((R / d).glob("*-err0.episodes.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line); case = CASES[r["id"]]; calls = rebuilt(r["calls"])
            a, b = both(lambda mod: mod.score(case, calls, r["final_text"]))
            assert a == b and abs(a["total"] - r["score"]["total"]) < 1e-12, (p.name, r["id"])
            m += 1
print(f"  1. probe episodes (dev, test, holdout_shift, forced; err0): {m}, off == before and == stored")

# 2. references: on == off
k = 0
for path in (nb.DATA / "train_reference.jsonl", nb.recovery_path()):
    for line in path.read_text().splitlines():
        e = json.loads(line); case = CASES[e["id"]]
        calls = [c | {"result": res, "turn": t} for t, turn in enumerate(e["turns"]) for c, res in zip(turn["calls"], turn["results"])]
        for err in (True, False):
            nb.ERR_CALL_CREDIT = err
            nb.PROVENANCE = False; off = nb.score(case, calls, e["answer"])
            nb.PROVENANCE = True; on = nb.score(case, calls, e["answer"])
            assert on == off, (path.name, e["id"], err)
            assert on["evidence"] and on["decision_correct"], (path.name, e["id"])
        k += 1
nb.PROVENANCE = False
print(f"  2. reference episodes: {k} (06 clean + 07 recovery), on == off under both call scorings, all proved")

# 3. hand-built
nb.ERR_CALL_CREDIT = False
L1 = [c for c in nb.load_split("dev") if c["level"] == 1 and nb.convert_plan(c, limits=False)]
case = min(L1, key=lambda c: (len(c["points"]), len(nb.convert_plan(c, limits=False))))
plan = nb.convert_plan(case, limits=False)
convs =[{"name": nb.CONVERT, "arguments": {"value": p["value"], "from_unit": p["from_unit"], "to_unit": p["to_unit"]},
          "result": nb.convert_units(p["value"], p["from_unit"], p["to_unit"])} for p in plan]
sims = [{"name": case["tool"], "arguments": dict(pt["args"]), "result": '{"ok": 1}'} for pt in case["points"]]
answer = json.dumps(nb.gold_answer(case))
T = lambda calls, turns: [c | {"turn": t} for c, t in zip(calls, turns)]   # noqa: E731


def sc(calls, prov):
    nb.PROVENANCE = prov
    out = nb.score(case, calls, answer)
    nb.PROVENANCE = False
    return out


tool_first = T(convs + sims, [0] * len(convs) + [1] * len(sims))
head = T(sims, [0] * len(sims))
same_turn = T(convs + sims, [0] * (len(convs) + len(sims)))
later = T(sims + convs, [0] * len(sims) + [1] * len(convs))
err_conv = T([c | {"result": '{"error": "x"}'} for c in convs] + sims, [0] * len(convs) + [1] * len(sims))
rounded = T(convs + [s | {"arguments": {k: (v * 1.004 if isinstance(v, float) else v) for k, v in s["arguments"].items()}}
                     for s in sims], [0] * len(convs) + [1] * len(sims))
no_turn = [{k: v for k, v in c.items() if k != "turn"} for c in tool_first]
for name, calls, proved in (("tool, then simulate", tool_first, True), ("converted in the head", head, False),
                            ("tool in the same turn", same_turn, False), ("tool after the simulate", later, False),
                            ("tool erred", err_conv, False), ("tool, then simulate rounded 0.4%", rounded, True),
                            ("tool, then simulate, no turns", no_turn, True)):
    off, on = sc(calls, False), sc(calls, True)
    assert off["evidence"], name
    assert on["evidence"] == proved, (name, on["covered"])
    assert on["components"]["calls"] == off["components"]["calls"], name          # fidelity unchanged
    if proved:
        assert on == off, name
    else:
        assert on["components"]["decision"] == nb.UNGATED * off["components"]["decision"] and on["components"]["efficiency"] == 0
    print(f"  3. {name:34} evidence off {off['evidence']!s:5} on {on['evidence']!s:5} | reward {off['total']:.3f} -> {on['total']:.3f}")
L0 = next(c for c in nb.load_split("dev") if c["level"] == 0)
calls0 = [{"name": L0["tool"], "arguments": dict(pt["args"]), "result": '{"ok": 1}', "turn": 0} for pt in L0["points"]]
a0 = json.dumps(nb.gold_answer(L0))
nb.PROVENANCE = True; on0 = nb.score(L0, calls0, a0); nb.PROVENANCE = False
assert on0 == nb.score(L0, calls0, a0) and on0["evidence"]
print(f"  3. nothing to convert (level 0)        unchanged, reward {on0['total']:.3f}")
try:
    nb.train_ppo({**nb.PPO_DEFAULTS, "provenance": 2, "adv_unit": "turn"}, None)
    raise AssertionError("accepted provenance 2")
except SystemExit as e:
    print("  bad value: provenance 2 ->", e)
print("ok")
