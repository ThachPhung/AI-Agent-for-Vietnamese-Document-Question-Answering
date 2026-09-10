from importlib.util import find_spec

import streamlit as st
import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from src.config import EMBEDDING_MODEL, LLM_MODEL, MAX_NEW_TOKENS


def init_model_state():
    defaults = {
        "retriever": None,
        "agent": None,
        "models_loaded": False,
        "embeddings": None,
        "llm": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _load_model():
    if torch.cuda.is_available():
        if find_spec("bitsandbytes") is None:
            raise RuntimeError(
                "Thiếu bitsandbytes. Hãy cài requirements-gpu.txt để chạy CUDA."
            )

        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        return AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            quantization_config=config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    return AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )


def _create_pipeline(model):
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=False)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


@st.cache_resource
def load_llm():
    """Tải Vicuna với cấu hình phù hợp cho GPU hoặc CPU."""
    return _create_pipeline(_load_model())
