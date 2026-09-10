"""PDF RAG Agent — Phiên bản Q&A đơn giản."""

import streamlit as st

from src.agent import create_agent, run_agent
from src.models import init_model_state, load_embeddings, load_llm
from src.pipeline import process_pdf

st.set_page_config(page_title="PDF RAG Agent", layout="wide")
init_model_state()


def main():
    st.title("🤖 PDF RAG Agent")
    st.markdown("""
    **AI Agent hỏi đáp tài liệu PDF** — tự quyết định có cần tìm kiếm
    trong tài liệu hay trả lời trực tiếp.

    1. **Upload PDF** → Chọn file và nhấn "Xử lý PDF"
    2. **Đặt câu hỏi** → Agent tự phân loại và trả lời
    ---
    """)

    if not st.session_state.models_loaded:
        st.info("Đang tải models...")
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.models_loaded = True
        st.success("Models đã sẵn sàng!")
        st.rerun()

    uploaded_file = st.file_uploader("Upload file PDF", type="pdf")
    if uploaded_file and st.button("Xử lý PDF"):
        with st.spinner("Đang xử lý..."):
            retriever, num_chunks = process_pdf(
                uploaded_file, st.session_state.embeddings
            )
            st.session_state.retriever = retriever
            st.session_state.agent = create_agent(st.session_state.llm, retriever)
            st.success(f"Hoàn thành! {num_chunks} chunks")

    if st.session_state.agent:
        question = st.text_input("Đặt câu hỏi:")
        if question:
            with st.spinner("Agent đang suy nghĩ..."):
                result = run_agent(st.session_state.agent, question)

                with st.expander("🤔 Quá trình suy nghĩ"):
                    if result["needs_search"]:
                        st.write(f"🔍 **Tìm kiếm tài liệu**: {result['reasoning']}")
                    else:
                        st.write(f"💬 **Trả lời trực tiếp**: {result['reasoning']}")

                st.write("**Trả lời:**")
                st.write(result["answer"])


if __name__ == "__main__":
    main()
