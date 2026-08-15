import unittest

from chartjudge_bench.parsing import cpa_status, parse_accept_reject, parse_cpa_choice
from chartjudge_bench.prompts import verify_all_prompts
from chartjudge_bench.protocols import (
    CPAProtocol,
    ChartEditingProtocol,
    ChartReproductionProtocol,
)


EXPECTED_HASHES = {
    "chart_reproduction_system": "edd6825266a513281e3a32063cc50e782a5b322956fae548db6a30dfbe3e3be8",
    "chart_editing_system": "d8c6aee9ef4c33bd67c1d0c25ce4160174165834df53036cf4b8b8c4593348ed",
    "cpa_system_instruction": "2302f7ce41f73904c573d006f0df9e747e5fb93f3dde858bc09e90d68a7cb17e",
}


class PromptAndProtocolTests(unittest.TestCase):
    def test_paper_prompts_are_immutable(self):
        self.assertEqual(verify_all_prompts(), EXPECTED_HASHES)

    def test_original_parsers(self):
        self.assertEqual(
            parse_accept_reject("[Answer]: Accept\n[Reason]: ok")[0], "Accept"
        )
        self.assertEqual(
            parse_accept_reject("<think>x</think>[Answer]: Reject")[0], "Reject"
        )
        self.assertEqual(
            parse_accept_reject("Accept and Reject are both possible")[0], "Unknown"
        )
        self.assertEqual(
            parse_cpa_choice(r"<think>x</think>\boxed{Image B}")[0], "Image B"
        )
        self.assertEqual(cpa_status("Image A", "Image B"), "Consistent_Correct")

    def test_cpa_has_original_two_pass_order_and_text(self):
        sample = {
            "sample_id": "1",
            "chart_type": "bar",
            "category": "Data_Fidelity",
            "subcategory": "Axis_Scaling",
            "better_image": "GOOD",
            "worse_image": "BAD",
        }
        requests = CPAProtocol().build_requests(sample)
        self.assertEqual(
            [request.pass_name for request in requests], ["Forward", "Reverse"]
        )
        self.assertEqual(requests[0].messages[0]["content"][1]["image"], "GOOD")
        self.assertEqual(requests[0].messages[0]["content"][3]["image"], "BAD")
        self.assertEqual(requests[1].messages[0]["content"][1]["image"], "BAD")
        self.assertEqual(requests[1].messages[0]["content"][3]["image"], "GOOD")
        self.assertEqual(
            requests[0].messages[0]["content"][-1]["text"],
            "\nWhich image is better? Remember to put the choice in \\boxed{}.",
        )

    def test_editing_message_keeps_original_instruction_and_order(self):
        sample = {
            "sample_id": "1",
            "chart_type": "line",
            "ground_truth": "Accept",
            "reference_image": "REF",
            "edited_image": "EDIT",
            "instruction": "Change only the title.",
        }
        request = ChartEditingProtocol().build_requests(sample)[0]
        blocks = request.messages[0]["content"]
        self.assertIn(
            '**User Instruction**: "Change only the title."', blocks[0]["text"]
        )
        self.assertEqual(blocks[1]["image"], "REF")
        self.assertEqual(blocks[3]["image"], "EDIT")

    def test_reproduction_keeps_ground_truth_first(self):
        sample = {
            "sample_id": "1",
            "chart_type": "bar",
            "ground_truth": "Reject",
            "generated_image": "GENERATED",
            "target_image": "TARGET",
        }
        request = ChartReproductionProtocol().build_requests(sample)[0]
        blocks = request.messages[1]["content"]
        self.assertEqual(blocks[2]["image"], "TARGET")
        self.assertEqual(blocks[4]["image"], "GENERATED")


if __name__ == "__main__":
    unittest.main()
