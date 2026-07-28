"""Validate the RO wrapper against WaterTAP's own published reference values.

The expected numbers are taken from WaterTAP's unit test
`TestReverseOsmosis0D_kf_calculated` in
watertap/unit_models/tests/test_reverse_osmosis_0D.py, so this checks that our
wrapper reproduces the upstream model rather than merely that it runs.

Run: .venv-watertap/bin/python mcp_watertap/test_ro_model.py
"""
from __future__ import annotations

import sys

from ro_model import ROSimulationError, simulate

# WaterTAP reference case: 1 kg/s of 3.5 wt% NaCl, 50 bar, 50 m2, 3 bar drop.
REFERENCE = {
    "flux_water_kg_m2_s": 0.00456244,
    "flux_salt_kg_m2_s": 1.5926761e-6,
    "permeate_water_kg_s": 0.22812202,
    "permeate_nacl_kg_s": 7.96338071e-5,
    "feed_nacl_g_L": 35.7511,
    "wall_nacl_g_L_inlet": 41.95562266,
}
RTOL = 1e-4


def check(label: str, got: float, expected: float) -> bool:
    ok = abs(got - expected) <= RTOL * abs(expected)
    print(f"  {label:24s} {got:>16.8g} {expected:>16.8g}  {'ok' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    failures = 0

    print("1. WaterTAP reference case (kf calculated)")
    r = simulate()
    passed = all([
        check("flux water kg/m2/s", r["flux"]["water_kg_m2_s"], REFERENCE["flux_water_kg_m2_s"]),
        check("flux salt kg/m2/s", r["flux"]["salt_kg_m2_s"], REFERENCE["flux_salt_kg_m2_s"]),
        check("permeate water kg/s", r["permeate"]["water_kg_s"], REFERENCE["permeate_water_kg_s"]),
        check("permeate NaCl kg/s", r["permeate"]["nacl_kg_s"], REFERENCE["permeate_nacl_kg_s"]),
        check("feed NaCl g/L", r["feed"]["nacl_g_L"], REFERENCE["feed_nacl_g_L"]),
        check("wall NaCl g/L inlet",
              r["concentration_polarization"]["wall_nacl_g_L_inlet"],
              REFERENCE["wall_nacl_g_L_inlet"]),
    ])
    failures += 0 if passed else 1
    print(f"  recovery {r['performance']['water_recovery_pct']:.2f}% | "
          f"rejection {r['performance']['salt_rejection_pct']:.3f}% | "
          f"flux {r['flux']['water_LMH']:.2f} LMH")

    print("\n2. Every valid CP / mass-transfer combination solves")
    combos = [
        ("none", "none"),
        ("fixed", "none"),
        ("calculated", "fixed"),
        ("calculated", "calculated"),
    ]
    for cp, kf in combos:
        try:
            out = simulate(concentration_polarization=cp, mass_transfer_coefficient=kf)
            print(f"  cp={cp:<11s} kf={kf:<11s} -> "
                  f"{out['flux']['water_LMH']:6.2f} LMH, "
                  f"{out['performance']['water_recovery_pct']:5.2f}% recovery")
        except ROSimulationError as exc:
            print(f"  cp={cp:<11s} kf={kf:<11s} -> FAILED: {exc}")
            failures += 1

    print("\n3. Physical trend: higher feed pressure raises recovery")
    rec = [simulate(feed_pressure_bar=p)["performance"]["water_recovery_pct"]
           for p in (40, 50, 60)]
    print(f"  40/50/60 bar -> {rec[0]:.2f}% / {rec[1]:.2f}% / {rec[2]:.2f}%")
    if not (rec[0] < rec[1] < rec[2]):
        print("  MISMATCH: recovery should increase monotonically with pressure")
        failures += 1

    print("\n4. Invalid input is rejected, not silently wrong")
    for kwargs, why in [
        ({"concentration_polarization": "calculated", "mass_transfer_coefficient": "none"}, "incompatible combo"),
        ({"feed_nacl_mass_frac": 1.5}, "mass fraction > 1"),
        ({"feed_pressure_bar": 0.5}, "feed below permeate pressure"),
        ({"nonexistent_param": 1}, "unknown parameter"),
    ]:
        try:
            simulate(**kwargs)
            print(f"  {why:32s} -> NOT REJECTED")
            failures += 1
        except ROSimulationError:
            print(f"  {why:32s} -> rejected")

    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
