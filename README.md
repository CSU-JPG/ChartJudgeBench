# ChartJudgeBench

Official data and evaluation toolkit for **ChartJudgeBench**, a benchmark for judging chart quality with multimodal models.

> Paper title, authors, and citation: **Coming soon**.

## Benchmark tasks

| Task | Samples | Model decision | Primary metric |
|---|---:|---|---|
| CPA | 1,003 pairs | Select the better chart in two reversed passes | Overall Accuracy |
| ChartEditing | 335 cases | Accept / Reject | Overall Accuracy |
| ChartReproduction | 315 cases | Accept / Reject | Overall Accuracy |

Dataset: [Lijian9/ChartJudgeBench](https://huggingface.co/datasets/Lijian9/ChartJudgeBench)

The image data stays on Hugging Face. This repository contains the official prompts, inference protocols, model adapters, strict scorer, tests, and documentation.

## Installation

```bash
git clone https://github.com/hush699/ChartJudgeBench.git
cd ChartJudgeBench
```

Choose the backend you need:

```bash
# Core: dataset loading and offline scoring
pip install -e .

# OpenAI-compatible APIs
pip install -e ".[api]"

# Local Qwen/ThinkLite models
pip install -e ".[qwen]"

# Every supported backend
pip install -e ".[all]"
```

Equivalent dependency lists are provided in `requirements.txt`, `requirements-api.txt`, `requirements-qwen.txt`, and `requirements-all.txt`. If you install through one of those files, finish with `pip install -e . --no-deps` to register the `chartjudge` command.

## Quick start

### 1. Verify the official prompts

```bash
chartjudge verify-prompts
```

The `paper_v1` prompts are copied exactly from the original evaluation scripts. Their SHA-256 hashes are checked at runtime and in CI. The task protocol also preserves the original text fragments, image order, parsers, and CPA two-pass procedure.

### 2. Run an OpenAI-compatible model

Copy `.env.example` to `.env` or export the variables in your shell:

```text
OPENAI_API_KEY=your_api_key
OPENAI_API_URL=https://your-openai-compatible-endpoint/v1
```

Copy and edit `configs/models/openai_compatible.example.json`, then run:

```bash
chartjudge run \
  --task ChartReproduction \
  --config configs/models/openai_compatible.example.json \
  --output outputs/my_model_reproduction.jsonl
```

The runner saves each completed case immediately and resumes from an existing JSONL file by default.

### 3. Run a local model

Set the model path in one of the paper configurations:

```bash
chartjudge run \
  --task ChartEditing \
  --config configs/models/qwen3_vl_32b_editing.paper_v1.json \
  --output outputs/qwen3_vl_32b_editing.jsonl
```

The repository includes paper-v1 configurations for:

- Gemini ChartReproduction through an OpenAI-compatible API
- Qwen3-VL-32B ChartEditing through Transformers
- ThinkLite-VL-7B CPA through Transformers

These preserve the generation settings from the supplied original scripts. The remaining paper models can be added as configuration files after their details are finalized.

### 4. Score predictions

```bash
chartjudge score \
  --task ChartEditing \
  --predictions outputs/my_model_editing.jsonl \
  --output outputs/my_model_editing_metrics.json
```

The scorer aligns predictions against Hugging Face dataset IDs. Missing predictions and parse failures count as incorrect in Overall Accuracy. Duplicate IDs and unknown IDs are rejected instead of being silently ignored.

## Official metrics

All three tasks use **Overall Accuracy** as the ranking metric.

### CPA

The model sees every pair twice:

1. Forward: better chart = Image A, worse chart = Image B.
2. Reverse: worse chart = Image A, better chart = Image B.

A sample is correct only if both passes select the better image.

- `overall_accuracy = consistently_correct / 1003`
- Accuracy for each of the seven CPA dimensions
- Diagnostics: Parse Fail Rate, Position A Bias Rate, Position B Bias Rate, Consistent Wrong Rate, and Coverage

CPA does not report Precision, Recall, or F1.

### ChartEditing and ChartReproduction

- `overall_accuracy = correct / official_dataset_size`
- Overall Accuracy includes parse failures and missing predictions as incorrect
- Valid Precision, Recall, Specificity, and F1 are calculated only over responses successfully parsed as `Accept` or `Reject`
- Parse Fail Rate and Coverage are diagnostic metrics

Ground-truth labels always come from the dataset, never from fields copied into prediction files.

Metric values in JSON output use the range `[0, 1]`.

## Use your own model

Users only need to implement:

```python
response = adapter.generate(request)
```

The benchmark owns the prompts, image order, parsing, two-pass logic, and scoring. See [docs/CUSTOM_MODELS.md](docs/CUSTOM_MODELS.md) and [examples/custom_model.py](examples/custom_model.py).

## Repository structure

```text
ChartJudgeBench/
├── chartjudge_bench/
│   ├── adapters/          # API, Transformers, and custom model interfaces
│   ├── prompt_assets/     # Immutable paper-v1 prompts
│   ├── data.py            # Hugging Face dataset loader
│   ├── protocols.py       # Task prompts, image order, and parsers
│   ├── runner.py          # Batch inference and resume support
│   └── scoring.py         # Official ID-aligned metrics
├── configs/models/        # Model and generation configurations
├── docs/                  # User documentation
├── examples/              # Custom model example
├── results/               # Paper results (coming soon)
├── tests/                 # Protocol and metric regression tests
├── requirements.txt       # Core dependencies
├── requirements-api.txt   # OpenAI-compatible API dependencies
├── requirements-qwen.txt  # Qwen/ThinkLite dependencies
└── requirements-all.txt   # Every supported backend
```

## Reproducibility notes

- Never modify `paper_v1`. Create a new protocol version for future changes.
- Record the model identifier and revision, package versions, generation settings, prompt version, and prediction JSONL.
- The supplied Qwen3-VL-32B paper run used sampling (`temperature=0.1`, `top_p=0.9`) without a fixed seed. Exact reruns may therefore vary.
- Do not commit API keys, server paths, model weights, or generated outputs.

## Results

The final 26-model table will be added after verification. See [results/README.md](results/README.md).

## Citation

Coming soon.

## License

- Code: [Apache License 2.0](LICENSE)
- Dataset: [CC BY 4.0](DATA_LICENSE.md)
