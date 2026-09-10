import time

import streamlit as st


def init_chat_state():
    defaults = {"chat_history": [], "pdf_processed": False, "pdf_name": ""}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_message(role, content):
    st.session_state.chat_history.append(
        {"role": role, "content": content, "timestamp": time.time()}
    )


def clear_chat():
    st.session_state.chat_history = []


def display_chat():
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.write("Xin chào! Hãy upload file PDF và đặt câu hỏi nhé! 😊")
        return

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
