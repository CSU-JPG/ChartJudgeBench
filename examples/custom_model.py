"""Minimal user-defined adapter example.

Replace the body of generate() with your own multimodal model call, then point a
custom config's class_path to examples.custom_model:MyModelAdapter.
"""

from chartjudge_bench.adapters import ModelAdapter


class MyModelAdapter(ModelAdapter):
    def __init__(self, model_name: str = "my-model") -> None:
        self.model_name = model_name

    def generate(self, request) -> str:
        # request.messages contains the exact benchmark prompt and ordered images.
        raise NotImplementedError("Connect your model here and return its raw text")
