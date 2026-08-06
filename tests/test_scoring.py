import unittest

from chartjudge_bench.scoring import score_predictions


class ScoringTests(unittest.TestCase):
    def test_cpa_official_primary_and_diagnostics(self):
        truth = [
            {"pair_id": "1", "category": "Data_Fidelity", "subcategory": "Axis_Scaling", "chart_type": "bar"},
            {"pair_id": "2", "category": "Data_Fidelity", "subcategory": "Axis_Scaling", "chart_type": "line"},
            {"pair_id": "3", "category": "Visual_Effects", "subcategory": "Visual_Style", "chart_type": "pie"},
            {"pair_id": "4", "category": "Visual_Effects", "subcategory": "Visual_Style", "chart_type": "area"},
        ]
        predictions = [
            {"sample_id": "1", "status": "Consistent_Correct"},
            {"sample_id": "2", "status": "Bias_Position_A"},
            {"sample_id": "3", "status": "Parse_Fail"},
        ]
        metrics = score_predictions("CPA", predictions, truth, require_official_size=False)
        self.assertEqual(metrics["overall_accuracy"], 0.25)
        self.assertEqual(metrics["dimension_accuracy"]["Axis_Scaling"]["accuracy"], 0.5)
        self.assertEqual(metrics["dimension_accuracy"]["Visual_Style"]["accuracy"], 0.0)
        self.assertEqual(metrics["diagnostics"]["position_a_bias_rate"], 0.25)
        self.assertEqual(metrics["diagnostics"]["parse_fail_rate"], 0.25)
        self.assertEqual(metrics["diagnostics"]["coverage"], 0.75)

    def test_accept_reject_primary_includes_parse_fail_and_missing(self):
        truth = [
            {"example_id": "1", "decision": "accept", "chart_type": "bar"},
            {"example_id": "2", "decision": "accept", "chart_type": "line"},
            {"example_id": "3", "decision": "reject", "chart_type": "pie"},
            {"example_id": "4", "decision": "reject", "chart_type": "area"},
            {"example_id": "5", "decision": "reject", "chart_type": "scatter"},
        ]
        predictions = [
            {"sample_id": "1", "model_decision": "Accept"},
            {"sample_id": "2", "model_decision": "Reject"},
            {"sample_id": "3", "model_decision": "Reject"},
            {"sample_id": "4", "model_decision": "Unknown"},
        ]
        metrics = score_predictions(
            "ChartEditing", predictions, truth, require_official_size=False
        )
        self.assertEqual(metrics["overall_accuracy"], 0.4)
        self.assertEqual(metrics["diagnostics"]["valid_precision"], 1.0)
        self.assertEqual(metrics["diagnostics"]["valid_recall"], 0.5)
        self.assertEqual(metrics["diagnostics"]["valid_specificity"], 1.0)
        self.assertEqual(metrics["diagnostics"]["parse_fail_rate"], 0.2)
        self.assertEqual(metrics["diagnostics"]["coverage"], 0.8)

    def test_duplicate_predictions_are_rejected(self):
        truth = [{"example_id": "1", "decision": "accept", "chart_type": "bar"}]
        predictions = [
            {"sample_id": "1", "model_decision": "Accept"},
            {"sample_id": "1", "model_decision": "Reject"},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate prediction IDs"):
            score_predictions(
                "ChartReproduction", predictions, truth, require_official_size=False
            )


if __name__ == "__main__":
    unittest.main()

