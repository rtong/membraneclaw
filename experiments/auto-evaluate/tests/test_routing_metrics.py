from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from auto_evaluate.routing_metrics import binary_route_metrics  # noqa: E402


class RoutingMetricsTests(unittest.TestCase):
    def test_mixed_r0_r2_metrics_use_rag_as_positive_class(self):
        rows = [
            {"expected": "skip_rag", "predicted": "skip_rag"},
            {"expected": "skip_rag", "predicted": "use_rag"},
            {"expected": "use_rag", "predicted": "use_rag"},
            {"expected": "use_rag", "predicted": "skip_rag"},
        ]
        metrics = binary_route_metrics(
            rows,
            expected_key="expected",
            predicted_key="predicted",
        )
        self.assertEqual(0.5, metrics["accuracy"])
        self.assertEqual(0.5, metrics["use_rag_precision"])
        self.assertEqual(0.5, metrics["use_rag_recall"])
        self.assertEqual(0.5, metrics["use_rag_f1"])
        self.assertEqual(0.5, metrics["balanced_accuracy"])


if __name__ == "__main__":
    unittest.main()
