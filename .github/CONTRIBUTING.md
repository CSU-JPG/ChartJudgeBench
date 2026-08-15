# Contributing

Issues and pull requests are welcome.

Before submitting a change:

1. Do not modify files under `chartjudge_bench/prompt_assets/paper_v1`.
2. Run `chartjudge verify-prompts`.
3. Run `python -m unittest discover -s tests -v`.
4. Do not commit API keys, local absolute paths, model weights, datasets, or generated outputs.

Protocol changes must use a new version rather than overwriting `paper_v1`, so published results remain reproducible.
