import unittest

from src.config import load_prompt


class PromptTest(unittest.TestCase):
    def test_prompt_templates_are_available(self):
        self.assertIn("{question}", load_prompt("classify"))
        self.assertIn("{question}", load_prompt("direct_generate"))
        self.assertIn("{context}", load_prompt("rag_generate"))


if __name__ == "__main__":
    unittest.main()
