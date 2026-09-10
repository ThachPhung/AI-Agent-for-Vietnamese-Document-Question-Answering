from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
LLM_MODEL = "lmsys/vicuna-7b-v1.5"
MAX_NEW_TOKENS = 512

CHUNK_BUFFER_SIZE = 1
CHUNK_THRESHOLD_TYPE = "percentile"
CHUNK_THRESHOLD_AMOUNT = 95
CHUNK_MIN_SIZE = 500


def load_prompt(name):
    """Đọc prompt template theo tên."""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
