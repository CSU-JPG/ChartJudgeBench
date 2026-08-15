import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIGS = ROOT / "configs" / "models" / "local"
EXPECTED_LOCAL_LABELS = {
    "MiMo-VL-SFT",
    "Qwen3-VL-30B",
    "MiMo-VL-RL",
    "LLaVA-Critic-R1",
    "ThinkLite-VL-7B",
    "Qwen2-VL-7B",
    "Qwen2-VL-72B",
    "DeepSeek-VL",
    "Qwen2.5-VL-7B",
    "Qwen2.5-VL-72B",
    "GLM-4V",
    "MiMo-VL-SFT (NoThink)",
    "MiMo-VL-RL (NoThink)",
    "Kimi-VL",
    "InternVL2.5-38B",
    "InternVL2.5-8B",
    "Molmo",
    "Molmo2",
    "Qwen3-VL-8B",
    "Qwen3-VL-32B",
    "InternVL3.5-8B",
    "InternVL3.5-38B",
}


class ModelConfigTests(unittest.TestCase):
    def test_final_local_model_config_count(self):
        self.assertEqual(len(list(LOCAL_CONFIGS.glob("*.json"))), 22)

    def test_local_configs_use_supported_adapters_and_placeholder_paths(self):
        supported = {
            "qwen_transformers",
            "thinklite_transformers",
            "transformers_chat",
            "internvl",
            "deepseek_vl",
            "molmo_vllm",
        }
        labels = set()
        for path in LOCAL_CONFIGS.glob("*.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            labels.add(config["display_name"])
            self.assertIn(config["adapter"], supported)
            self.assertTrue(
                config["adapter_kwargs"]["model_path"].startswith("/path/to/")
            )
            self.assertEqual(config["runner"]["workers"], 1)
        self.assertEqual(labels, EXPECTED_LOCAL_LABELS)

    def test_qwen35_is_not_in_final_configs(self):
        names = {path.name.lower() for path in LOCAL_CONFIGS.glob("*.json")}
        self.assertFalse(any("qwen3_5" in name or "qwen3.5" in name for name in names))

    def test_mimo_modes_are_explicit(self):
        expected = {
            "mimo_vl_sft.json",
            "mimo_vl_sft_nothink.json",
            "mimo_vl_rl.json",
            "mimo_vl_rl_nothink.json",
        }
        self.assertTrue(
            expected.issubset({path.name for path in LOCAL_CONFIGS.glob("*.json")})
        )


if __name__ == "__main__":
    unittest.main()
