# PDF RAG Agent

Ứng dụng hỏi đáp tài liệu PDF bằng tiếng Việt, kết hợp Retrieval-Augmented
Generation (RAG) với một agent LangGraph. Agent phân loại từng câu hỏi để quyết
định truy xuất tài liệu hoặc trả lời trực tiếp.

## Luồng xử lý

```text
PDF → PyPDFLoader → SemanticChunker → Embeddings → Chroma → Retriever

Question → Classify
             ├── cần tài liệu → Retrieve → RAG prompt → Vicuna → Answer
             └── không cần    → Direct prompt ───────→ Vicuna → Answer
```

- Embedding: `bkai-foundation-models/vietnamese-bi-encoder`
- LLM: `lmsys/vicuna-7b-v1.5`
- Vector store: Chroma
- Orchestration: LangGraph
- Giao diện: Streamlit
- GPU inference: quantization 4-bit bằng bitsandbytes

## Yêu cầu

- Python 3.11
- Khoảng 20 GB dung lượng trống cho môi trường và model
- Kết nối Internet trong lần chạy đầu
- GPU NVIDIA với ít nhất 8 GB VRAM được khuyến nghị
- Nếu chạy CPU: nên có ít nhất 24 GB RAM và chấp nhận tốc độ chậm

Model Vicuna có dung lượng khoảng 13.5 GB trước khi quantization. Lần chạy đầu
cũng tải embedding model khoảng 540 MB; các lần sau sử dụng cache của
Hugging Face.

## Cài đặt

### NVIDIA GPU

Cấu hình khuyến nghị là Linux hoặc WSL2 với NVIDIA driver hoạt động bình thường.

```bash
git clone https://github.com/ThachPhung/Personal-Chatbot-with-RAG.git
cd Personal-Chatbot-with-RAG

conda create -n aio-rag python=3.11 -y
conda activate aio-rag

python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-gpu.txt
```

Lệnh trên dùng PyTorch cho CUDA 12.6. Nếu driver yêu cầu bản CUDA khác, chọn
lệnh tương ứng tại [PyTorch Start Locally](https://pytorch.org/get-started/locally/),
sau đó cài `requirements-gpu.txt`.

Kiểm tra CUDA trước khi chạy:

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

Lệnh phải in ra tên GPU mà không báo lỗi.

### CPU hoặc macOS

```bash
git clone https://github.com/ThachPhung/Personal-Chatbot-with-RAG.git
cd Personal-Chatbot-with-RAG

conda create -n aio-rag python=3.11 -y
conda activate aio-rag

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Chạy ứng dụng

Giao diện hỏi đáp đơn giản:

```bash
python -m streamlit run app.py
```

Giao diện chatbot có lịch sử hội thoại:

```bash
python -m streamlit run chatbot_app.py
```

Mở <http://localhost:8501> nếu trình duyệt không tự khởi động.

## Cách sử dụng

1. Chọn một file PDF.
2. Nhấn **Xử lý PDF** và chờ hệ thống tạo vector store.
3. Nhập câu hỏi.
4. Agent tự chọn trả lời trực tiếp hoặc tìm ngữ cảnh trong PDF.

Vector store hiện được tạo trong bộ nhớ cho từng phiên chạy. Khởi động lại ứng
dụng sẽ cần xử lý lại PDF.

## Cấu trúc dự án

```text
Personal-Chatbot-with-RAG/
├── .github/workflows/
│   └── ci.yml
├── notebooks/
│   └── [Code]_Project_RAG_Chatbot.ipynb
├── prompts/
│   ├── classify.txt
│   ├── direct_generate.txt
│   └── rag_generate.txt
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── models.py
│   ├── pipeline.py
│   └── ui.py
├── app.py
├── chatbot_app.py
├── pyproject.toml
├── requirements-dev.txt
├── requirements-gpu.txt
├── requirements.txt
├── .gitignore
└── README.md
```

Repository không kèm tài liệu PDF mẫu. Người dùng upload tài liệu của mình qua
giao diện.

## Kiểm tra code

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py chatbot_app.py src tests
```

Để chạy formatter và linter:

```bash
python -m pip install -r requirements-dev.txt
ruff format --check .
ruff check .
```

## Xử lý lỗi thường gặp

### `torch.cuda.is_available()` trả về `False`

PyTorch đang dùng bản CPU hoặc NVIDIA driver chưa tương thích. Cài lại PyTorch
theo đúng CUDA tại trang hướng dẫn chính thức rồi chạy lại bước kiểm tra CUDA.

### Báo thiếu `bitsandbytes`

Môi trường được cài bằng `requirements.txt` thay vì `requirements-gpu.txt`:

```bash
python -m pip install -r requirements-gpu.txt
```

### Terminal chạy nhầm Streamlit

Luôn chạy qua Python của môi trường Conda:

```bash
python -m streamlit run app.py
```

## Lưu ý

- Không cần API key để chạy ứng dụng.
- Model được tải trực tiếp từ Hugging Face.
- Vicuna 7B v1.5 sử dụng giấy phép Llama 2; cần kiểm tra điều khoản trước khi
  triển khai thương mại.
