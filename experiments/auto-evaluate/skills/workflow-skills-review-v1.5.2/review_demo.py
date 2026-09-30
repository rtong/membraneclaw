"""Offline unit checks using synthetic fixtures, not solver or model results."""
from pathlib import Path
import hashlib
import json
import math
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from auto_evaluate.fixed_input_guard import require_fixed_inputs
from auto_evaluate.public_program import metric_values, checks
import auto_evaluate.public_program as implementation

class ReviewChecks(unittest.TestCase):
    def test_source_snapshot(self):
        self.assertTrue(Path(implementation.__file__).resolve().is_relative_to(ROOT))
        for entry in json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))["files"]:
            self.assertEqual(hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest(), entry["sha256"])

    def test_fixed_inputs_and_free_variable(self):
        fixed = {"membrane_area": 137, "composition": {"Na": 0.12}}
        for pressure in (45, 52, 61):
            require_fixed_inputs(fixed, {**fixed, "pressure": pressure})
        for args in ({"membrane_area": 137}, {"membrane_area": 138, "composition": {"Na": 0.12}}):
            with self.assertRaises(ValueError):
                require_fixed_inputs(fixed, args)
        with self.assertRaises(ValueError):
            require_fixed_inputs(fixed, fixed, required_fields=("pressure",))

    def test_metric_conventions(self):
        result = {"permeate": {"flow_kg_s": 2.5}, "performance": {"ro_recovery_pct": 48, "system_recovery_pct": 41}}
        values = metric_values(result, {})
        self.assertAlmostEqual(values["Qp_m3_h"], 9)
        self.assertEqual(values["ro_recovery_pct"], 48)
        self.assertEqual(values["system_recovery_pct"], 41)

    def test_constraints(self):
        rules = [{"metric": "x", "op": "<", "threshold": 1}, {"metric": "x", "op": "<=", "threshold": 1}]
        self.assertEqual([r["passed"] for r in checks({"x": 1}, rules)], [False, True])
        for value in (None, math.nan, math.inf):
            self.assertFalse(any(r["passed"] for r in checks({"x": value}, rules)))
        self.assertFalse(any(r["passed"] for r in checks({}, rules)))

if __name__ == "__main__":
    unittest.main(verbosity=2)
