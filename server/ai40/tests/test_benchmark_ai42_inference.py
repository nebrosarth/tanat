from __future__ import annotations

import unittest

from tanat_ai40.benchmark_ai42_inference import _timing


class AI42InferenceBenchmarkTests(unittest.TestCase):
    def test_timing_reports_batch_latency_and_row_throughput(self) -> None:
        result = _timing(2.0, iterations=20, batch=10)
        self.assertEqual(result["milliseconds_per_batch"], 100.0)
        self.assertEqual(result["rows_per_second"], 100.0)

    def test_timing_rejects_non_positive_values(self) -> None:
        for values in ((0.0, 1, 1), (1.0, 0, 1), (1.0, 1, 0)):
            with self.assertRaises(ValueError):
                _timing(*values)


if __name__ == "__main__":
    unittest.main()
