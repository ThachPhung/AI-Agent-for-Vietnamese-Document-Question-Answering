"""PDF RAG Agent — Phiên bản Chatbot đầy đủ."""

import streamlit as st

from src.agent import create_agent, run_agent
from src.models import init_model_state, load_embeddings, load_llm
from src.pipeline import process_pdf
from src.ui import add_message, clear_chat, display_chat, init_chat_state

st.set_page_config(
    page_title="PDF RAG Agent Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_model_state()
init_chat_state()


def main():
    st.title("🤖 PDF RAG Agent")

    with st.sidebar:
        st.title("⚙️ Cài đặt")

        if not st.session_state.models_loaded:
            st.warning("⏳ Đang tải models...")
            with st.spinner("Đang tải AI models..."):
                st.session_state.embeddings = load_embeddings()
                st.session_state.llm = load_llm()
                st.session_state.models_loaded = True
            st.success("✅ Models đã sẵn sàng!")
            st.rerun()
        else:
            st.success("✅ Models đã sẵn sàng!")

        st.markdown("---")

        st.subheader("📄 Upload tài liệu")
        uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")

        if uploaded_file:
            if st.button("🔄 Xử lý PDF", use_container_width=True):
                with st.spinner("Đang xử lý PDF..."):
                    retriever, num_chunks = process_pdf(
                        uploaded_file, st.session_state.embeddings
                    )
                    st.session_state.retriever = retriever
                    st.session_state.agent = create_agent(
                        st.session_state.llm, retriever
                    )
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    clear_chat()
                    add_message(
                        "assistant",
                        f"✅ Đã xử lý thành công file "
                        f"**{uploaded_file.name}**!\n\n"
                        f"📊 Tài liệu được chia thành {num_chunks} phần.\n\n"
                        f"🤖 Agent sẵn sàng — tôi sẽ tự quyết định "
                        "khi nào cần tìm trong tài liệu.",
                    )
                st.rerun()

        if st.session_state.pdf_processed:
            st.success(f"📄 Đã tải: {st.session_state.pdf_name}")
        else:
            st.info("📄 Chưa có tài liệu")

        st.markdown("---")
        st.subheader("💬 Điều khiển Chat")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            clear_chat()
            st.rerun()

        st.markdown("---")
        st.subheader("📋 Hướng dẫn")
        st.markdown("""
        **Cách sử dụng:**
        1. **Upload PDF** - Chọn file và nhấn "Xử lý PDF"
        2. **Đặt câu hỏi** - Nhập câu hỏi trong ô chat
        3. **Agent tự quyết định** - Tìm tài liệu hoặc trả lời trực tiếp
        """)

    st.markdown(
        "*🤖 Agent tự động phân loại câu hỏi — "
        "tìm kiếm tài liệu khi cần, trả lời trực tiếp khi không cần*"
    )
    with st.container():
        display_chat()

    if st.session_state.models_loaded:
        if st.session_state.pdf_processed:
            user_input = st.chat_input("Nhập câu hỏi của bạn...")
            if user_input:
                add_message("user", user_input)
                with st.chat_message("user"):
                    st.write(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("🤖 Agent đang suy nghĩ..."):
                        try:
                            result = run_agent(st.session_state.agent, user_input)

                            if result["needs_search"]:
                                st.caption(
                                    f"🔍 *Tìm kiếm tài liệu: {result['reasoning']}*"
                                )
                            else:
                                st.caption(
                                    f"💬 *Trả lời trực tiếp: {result['reasoning']}*"
                                )

                            st.write(result["answer"])
                            add_message("assistant", result["answer"])

                        except Exception as error:
                            error_msg = f"Xin lỗi, đã có lỗi xảy ra: {error}"
                            st.error(error_msg)
                            add_message("assistant", error_msg)
        else:
            st.info("🔄 Vui lòng upload và xử lý file PDF trước khi bắt đầu chat!")
            st.chat_input("Nhập câu hỏi của bạn...", disabled=True)
    else:
        st.info("⏳ Đang tải AI models, vui lòng đợi...")
        st.chat_input("Nhập câu hỏi của bạn...", disabled=True)


if __name__ == "__main__":
    main()
