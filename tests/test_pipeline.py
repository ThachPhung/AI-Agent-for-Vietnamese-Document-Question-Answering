import unittest

from src.pipeline import parse_answer


class ParseAnswerTest(unittest.TestCase):
    def test_extracts_answer_marker(self):
        self.assertEqual(parse_answer("Answer: Nội dung"), "Nội dung")

    def test_returns_plain_output(self):
        self.assertEqual(parse_answer("Nội dung"), "Nội dung")


if __name__ == "__main__":
    unittest.main()
