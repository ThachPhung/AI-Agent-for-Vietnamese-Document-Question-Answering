import unittest
from types import SimpleNamespace

from src.agent import create_agent, run_agent


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)

    def invoke(self, _prompt):
        return next(self.outputs)


class FakeRetriever:
    def __init__(self, content="Nội dung tài liệu"):
        self.content = content
        self.questions = []

    def invoke(self, question):
        self.questions.append(question)
        return [SimpleNamespace(page_content=self.content)]


class AgentTest(unittest.TestCase):
    def test_answers_directly_when_search_is_not_needed(self):
        llm = FakeLLM(
            '{"needs_search": false, "reason": "Câu chào"}',
            "Xin chào!",
        )
        retriever = FakeRetriever()

        result = run_agent(create_agent(llm, retriever), "Xin chào")

        self.assertEqual(result["answer"], "Xin chào!")
        self.assertFalse(result["needs_search"])
        self.assertEqual(retriever.questions, [])

    def test_retrieves_context_when_search_is_needed(self):
        llm = FakeLLM(
            '{"needs_search": true, "reason": "Cần dữ kiện"}',
            "Câu trả lời từ tài liệu",
        )
        retriever = FakeRetriever()

        result = run_agent(create_agent(llm, retriever), "Tài liệu nói gì?")

        self.assertEqual(result["answer"], "Câu trả lời từ tài liệu")
        self.assertTrue(result["needs_search"])
        self.assertEqual(retriever.questions, ["Tài liệu nói gì?"])

    def test_defaults_to_search_when_classification_is_invalid(self):
        llm = FakeLLM("invalid", "Answer: Câu trả lời dự phòng")
        retriever = FakeRetriever()

        result = run_agent(create_agent(llm, retriever), "Câu hỏi")

        self.assertEqual(result["answer"], "Câu trả lời dự phòng")
        self.assertTrue(result["needs_search"])
        self.assertEqual(retriever.questions, ["Câu hỏi"])


if __name__ == "__main__":
    unittest.main()
