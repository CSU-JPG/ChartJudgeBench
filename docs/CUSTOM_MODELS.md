# Connect a custom model

ChartJudgeBench keeps task rules separate from model-specific code. A custom model only needs to implement one method:

```python
from chartjudge_bench.adapters import ModelAdapter

class MyModelAdapter(ModelAdapter):
    def generate(self, request):
        # request.messages contains ordered text and image blocks.
        raw_text = call_my_model(request.messages)
        return raw_text
```

The benchmark runner owns the prompts, image order, CPA two-pass protocol, parsing, and scoring. The adapter must not rebuild or alter them.

## Closed-source API models

If the endpoint supports OpenAI-compatible multimodal chat completions, no
Python code is required. Copy `configs/models/openai_compatible.example.json`,
set the API key and base URL through environment variables, and pass the model
identifier with `--model-name`.

```bash
chartjudge run \
  --task ChartReproduction \
  --config configs/models/openai_compatible.example.json \
  --model-name YOUR_MODEL_NAME \
  --output outputs/my_api_model_reproduction.jsonl
```

Never put an API key directly in a JSON configuration or Python source file.

## Other SDKs or local runtimes

Create a JSON config:

```json
{
  "adapter": "custom",
  "class_path": "my_package.my_adapter:MyModelAdapter",
  "adapter_kwargs": {"model_name": "my-model"},
  "runner": {"workers": 1}
}
```

Then run:

```bash
chartjudge run --task CPA --config my_model.json --output outputs/my_model_cpa.jsonl
```

Set `thread_safe = True` on the adapter only if it is safe to call the same instance concurrently. Local Transformers adapters should normally use one worker.

`request.messages` is already the complete benchmark request. Return only the
raw response text; do not parse the answer, change image order, or copy task
prompts into the adapter.
