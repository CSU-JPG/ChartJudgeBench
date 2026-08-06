# Paper-v1 prompts

These prompt assets reproduce the prompts in the original ChartJudgeBench evaluation scripts. Runtime SHA-256 verification prevents accidental edits.

The complete multimodal message sequence is assembled in `chartjudge_bench/protocols.py`, including the original image order and text fragments. CPA always runs twice: first with the better image as Image A, then with the better image as Image B.

Do not edit `paper_v1`. If the benchmark protocol changes, create a new version such as `canonical_v2`.

