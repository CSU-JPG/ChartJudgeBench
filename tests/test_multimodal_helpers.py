import unittest

from PIL import Image

from chartjudge_bench.adapters._multimodal import (
    close_images,
    flatten_messages,
    messages_with_image_placeholders,
)


class MultimodalHelperTests(unittest.TestCase):
    def setUp(self):
        self.first = Image.new("RGB", (4, 4), "red")
        self.second = Image.new("RGB", (4, 4), "blue")
        self.messages = [
            {"role": "system", "content": "SYSTEM"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A:"},
                    {"type": "image", "image": self.first},
                    {"type": "text", "text": " B:"},
                    {"type": "image", "image": self.second},
                ],
            },
        ]

    def tearDown(self):
        self.first.close()
        self.second.close()

    def test_flatten_preserves_text_and_image_order(self):
        prompt, images = flatten_messages(self.messages, image_marker="<image>")
        try:
            self.assertEqual(prompt, "SYSTEM\n\nA:<image> B:<image>")
            self.assertEqual(images[0].getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(images[1].getpixel((0, 0)), (0, 0, 255))
        finally:
            close_images(images)

    def test_chat_placeholders_do_not_mutate_protocol_messages(self):
        prepared, images = messages_with_image_placeholders(
            self.messages, inline_pil=False
        )
        try:
            self.assertEqual(prepared[1]["content"][1], {"type": "image"})
            self.assertIs(self.messages[1]["content"][1]["image"], self.first)
        finally:
            close_images(images)


if __name__ == "__main__":
    unittest.main()
