"""09: the targeted SFT doses -- k episodes per unit 07's seed never converts right (rate <= 0.061 over its 800 T=1
warm-up episodes). Units with train2 recovery episodes get k/2 references + k/2 recoveries; F and t/h have no recovery
episodes (a raw value there does not err), so k references. Writes 09_fresh_questions_data/dose_k{2,4,8}.jsonl."""
import json, random
from pathlib import Path

ST = Path(__file__).resolve().parent.parent
DATA09 = ST / "09_fresh_questions_data"
TARGETS = [("feed_flow_mass_kg_s", "t/h"), ("feed_flow_m3_s", "m3/h"), ("A_comp", "LMH/bar"), ("B_comp", "LMH"),
           ("feed_temperature_c", "F"), ("feed_flow_mass_kg_s", "kg/h"), ("membrane_area_m2", "ft2"), ("feed_tds_g_L", "mg/L")]
cases = {c["id"]: c for c in map(json.loads, (DATA09 / "train2.jsonl").read_text().splitlines())}
refs = [json.loads(l) for l in (DATA09 / "train2_reference.jsonl").read_text().splitlines()]
recs = [json.loads(l) for l in (DATA09 / "train2_recovery.jsonl").read_text().splitlines()]


def stated(c):
    u = {(k, v.get("unit")) for k, v in c["inputs"].items()}
    for pt in c["points"]:
        u |= {(k, v.get("unit")) for k, v in (pt.get("inputs") or {}).items()}
    return u


for k in (2, 4, 8):
    used, dose = set(), []
    for param, unit in TARGETS:
        rng = random.Random(f"09:{param}:{unit}")
        has_rec = any((r.get("slip") or {}).get("param") == param and r["slip"]["unit"] == unit for r in recs)
        n_rec = k // 2 if has_rec else 0
        pool_ref = [r for r in refs if (param, unit) in stated(cases[r["id"]]) and r["id"] not in used]
        pool_rec = [r for r in recs if (r.get("slip") or {}).get("param") == param and r["slip"]["unit"] == unit
                    and r["id"] not in used]
        rng.shuffle(pool_ref); rng.shuffle(pool_rec)
        pick = pool_ref[:k - n_rec] + pool_rec[:n_rec]
        for r in pick:
            used.add(r["id"])
            dose.append(dict(r, dose=f"{param} [{unit}]"))
    (DATA09 / f"dose_k{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dose))
    print(f"k={k}: {len(dose)} episodes ({sum(1 for r in dose if 'slip' in r)} recoveries)")
