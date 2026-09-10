"""Định nghĩa luồng xử lý câu hỏi bằng LangGraph.

Workflow:
  classify → (conditional) → retrieve → generate_with_context → END
                            → generate_direct → END
"""

import json
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.config import load_prompt
from src.pipeline import parse_answer, retrieve_docs


class AgentState(TypedDict):
    question: str
    needs_search: bool
    reasoning: str
    context: str
    answer: str


def create_agent(llm, retriever):
    """Tạo agent phân loại, truy xuất và trả lời câu hỏi."""

    def classify(state):
        prompt_template = load_prompt("classify")
        prompt = prompt_template.format(question=state["question"])
        output = llm.invoke(prompt)

        try:
            match = re.search(r"\{.*?\}", output, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return {
                    "needs_search": bool(result.get("needs_search", True)),
                    "reasoning": result.get("reason", ""),
                }
        except (json.JSONDecodeError, AttributeError):
            pass

        return {
            "needs_search": True,
            "reasoning": "Mặc định tìm kiếm trong tài liệu.",
        }

    def retrieve(state):
        context = retrieve_docs(state["question"], retriever)
        return {"context": context}

    def generate_with_context(state):
        prompt_template = load_prompt("rag_generate")
        prompt = prompt_template.format(
            context=state["context"], question=state["question"]
        )
        output = llm.invoke(prompt)
        return {"answer": parse_answer(output)}

    def generate_direct(state):
        prompt_template = load_prompt("direct_generate")
        prompt = prompt_template.format(question=state["question"])
        output = llm.invoke(prompt)
        return {"answer": parse_answer(output)}

    def route(state):
        return "retrieve" if state["needs_search"] else "generate_direct"

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_with_context", generate_with_context)
    graph.add_node("generate_direct", generate_direct)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route,
        {
            "retrieve": "retrieve",
            "generate_direct": "generate_direct",
        },
    )
    graph.add_edge("retrieve", "generate_with_context")
    graph.add_edge("generate_with_context", END)
    graph.add_edge("generate_direct", END)

    return graph.compile()


def run_agent(agent, question):
    """Chạy agent và trả về câu trả lời cùng kết quả phân loại."""
    result = agent.invoke(
        {
            "question": question,
            "needs_search": False,
            "reasoning": "",
            "context": "",
            "answer": "",
        }
    )
    return {
        "answer": result["answer"],
        "needs_search": result["needs_search"],
        "reasoning": result["reasoning"],
    }
