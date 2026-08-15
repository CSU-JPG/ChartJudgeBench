# Model configurations

The benchmark protocol owns the prompts, image order, parsing, and scoring. A
model configuration only selects a backend, checkpoint path, and generation
settings. Local paths in the bundled JSON files are placeholders; pass the real
path with `--model-path` or edit a copied configuration.

## Closed-source models

Gemini-3-Pro, GPT-5.2, Seed1.6-VL, and Claude-Sonnet-4.5 use the single
`openai_compatible.example.json` entry point when their endpoint implements the
OpenAI chat-completions schema. For another provider SDK, implement the small
custom adapter interface documented in `docs/CUSTOM_MODELS.md`. No historical
API keys or private endpoints are included.

## Local model configurations

| Paper label | Configuration | Adapter |
|---|---|---|
| MiMo-VL-SFT | `local/mimo_vl_sft.json` | Qwen-style Transformers |
| MiMo-VL-SFT (NoThink) | `local/mimo_vl_sft_nothink.json` | Qwen-style Transformers |
| MiMo-VL-RL | `local/mimo_vl_rl.json` | Qwen-style Transformers |
| MiMo-VL-RL (NoThink) | `local/mimo_vl_rl_nothink.json` | Qwen-style Transformers |
| Qwen3-VL-30B | `local/qwen3_vl_30b.json` | Qwen-style Transformers |
| Qwen3-VL-8B | `local/qwen3_vl_8b.json` | Qwen-style Transformers |
| Qwen3-VL-32B | `local/qwen3_vl_32b.json` | Qwen-style Transformers |
| LLaVA-Critic-R1 | `local/llava_critic_r1.json` | Qwen-style Transformers |
| ThinkLite-VL-7B | `local/thinklite_vl_7b.json` | Qwen2.5-VL Transformers |
| Qwen2-VL-7B | `local/qwen2_vl_7b.json` | Qwen-style Transformers |
| Qwen2-VL-72B | `local/qwen2_vl_72b.json` | Qwen-style Transformers |
| Qwen2.5-VL-7B | `local/qwen2_5_vl_7b.json` | Qwen-style Transformers |
| Qwen2.5-VL-72B | `local/qwen2_5_vl_72b.json` | Qwen-style Transformers |
| DeepSeek-VL | `local/deepseek_vl.json` | DeepSeek-VL native |
| GLM-4V | `local/glm_4v.json` | Transformers inline images |
| Kimi-VL | `local/kimi_vl.json` | Transformers separate images |
| InternVL2.5-8B | `local/internvl2_5_8b.json` | InternVL native chat |
| InternVL2.5-38B | `local/internvl2_5_38b.json` | InternVL native chat |
| InternVL3.5-8B | `local/internvl3_5_8b.json` | InternVL native chat |
| InternVL3.5-38B | `local/internvl3_5_38b.json` | InternVL native chat |
| Molmo | `local/molmo.json` | vLLM multimodal |
| Molmo2 | `local/molmo2.json` | Transformers inline images |

The table has 22 local configurations. Together with the four API model labels,
the paper table contains 26 evaluation settings. MiMo Think and NoThink share
the same two checkpoints but intentionally use different control suffixes and
generation limits.

Example:

```bash
chartjudge run \
  --task ChartEditing \
  --config configs/models/local/qwen3_vl_8b.json \
  --model-path /models/Qwen3-VL-8B-Instruct \
  --output outputs/qwen3_vl_8b_editing.jsonl
```

The same model configuration can be used for CPA, ChartEditing, and
ChartReproduction. Task-specific prompts must not be copied into or modified in
model adapters.

## Environments

Use a separate environment when model requirements conflict:

| Backend | Starting point |
|---|---|
| Qwen, MiMo, ThinkLite, LLaVA-Critic | The checkpoint's official environment, plus `requirements/qwen.txt` |
| InternVL2.5/3.5 | The checkpoint's official environment, or `requirements/internvl.txt` as a minimal starting point |
| DeepSeek-VL | Install the official DeepSeek-VL package/repository, then `pip install -e . --no-deps` |
| Molmo | Install the checkpoint-compatible vLLM version; `requirements/vllm.txt` is only a minimal starting point |
| GLM-4V, Kimi-VL, Molmo2 | Use the Transformers version required by the checkpoint |

The project does not claim that one frozen environment can load all model
families. For a paper rerun, record the exact checkpoint revision, CUDA/PyTorch
stack, package versions, and GPU layout with the prediction file.
