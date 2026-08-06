import json
import tempfile
import unittest
from pathlib import Path

from chartjudge_bench.adapters.base import ModelAdapter
from chartjudge_bench.runner import run_evaluation


class AlwaysCorrectCPAAdapter(ModelAdapter):
    thread_safe = True

    def generate(self, request):
        choice = "Image A" if request.pass_name == "Forward" else "Image B"
        return rf"<think>test</think>\boxed{{{choice}}}"


class RunnerTests(unittest.TestCase):
    def test_cpa_run_and_resume(self):
        rows = [
            {
                "pair_id": "1",
                "category": "Data_Fidelity",
                "subcategory": "Axis_Scaling",
                "chart_type": "bar",
                "better_image": "GOOD",
                "worse_image": "BAD",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            first = run_evaluation(
                task="CPA",
                adapter=AlwaysCorrectCPAAdapter(),
                rows=rows,
                output_path=output,
                workers=1,
            )
            second = run_evaluation(
                task="CPA",
                adapter=AlwaysCorrectCPAAdapter(),
                rows=rows,
                output_path=output,
                workers=1,
            )
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(first["newly_written"], 1)
            self.assertEqual(second["newly_written"], 0)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "Consistent_Correct")


if __name__ == "__main__":
    unittest.main()

