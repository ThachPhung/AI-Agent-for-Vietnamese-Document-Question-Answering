# **PDF RAG Assistant (Chatbot hỏi đáp tài liệu)**

Dự án này xây dựng một chatbot hỏi đáp tài liệu PDF bằng tiếng Việt dựa trên kiến trúc **Retrieval-Augmented Generation (RAG)**. Hệ thống cho phép người dùng upload file PDF và đặt câu hỏi trực tiếp về nội dung tài liệu, chatbot sẽ truy xuất thông tin liên quan và sinh câu trả lời chính xác dựa trên nội dung đó.

## **Kiến trúc RAG Pipeline**

```
PDF → PyPDFLoader → SemanticChunker → Chroma Vector DB → Retriever
                                                              ↓
User Question → RAG Chain (Retriever + Prompt + LLM) → Answer
```

1. **Đọc PDF**: Sử dụng `PyPDFLoader` để trích xuất văn bản từ file PDF
2. **Semantic Chunking**: Chia tài liệu thành các phần theo ngữ nghĩa (không cắt theo độ dài cố định)
3. **Vector Embedding**: Chuyển các chunks thành vector sử dụng `vietnamese-bi-encoder`
4. **Vector Database**: Lưu trữ embeddings vào ChromaDB
5. **RAG Chain**: Kết hợp Retriever + Prompt Template + LLM để sinh câu trả lời

## **Công nghệ sử dụng**

| Công nghệ | Mục đích |
|---|---|
| **LangChain** | Xây dựng pipeline RAG |
| **ChromaDB** | Lưu trữ vector embeddings |
| **HuggingFace Embeddings** | `bkai-foundation-models/vietnamese-bi-encoder` |
| **Vicuna 7B v1.5** | LLM sinh câu trả lời (quantization 4-bit) |
| **Semantic Chunking** | Chia tài liệu theo ngữ nghĩa |
| **Streamlit** | Xây dựng giao diện web tương tác |
| **BitsAndBytes** | Quantization 4-bit cho LLM |

## **Cài đặt**

### 1. Tạo môi trường Conda

```bash
conda create -n aio-rag python=3.11
conda activate aio-rag
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. Cấu hình (tùy chọn)

```bash
cp .env.example .env
# Chỉnh sửa .env nếu cần
```

## **Chạy ứng dụng**

### Phiên bản đơn giản (Q&A)

```bash
streamlit run app.py
```

Giao diện bao gồm:
- Upload PDF và nhấn "Xử lý PDF"
- Nhập câu hỏi → Nhận câu trả lời ngay lập tức

### Phiên bản Chatbot (khung chat đầy đủ)

```bash
streamlit run chatbot_app.py
```

Giao diện mở rộng bao gồm:
- **Sidebar**: Upload PDF, điều khiển chat, hướng dẫn sử dụng
- **Khung chat**: Lịch sử hội thoại liên tục
- **Tương tác**: Đặt nhiều câu hỏi mà không cần xử lý lại PDF

## **Cấu trúc dự án**

```
Personal-Chatbot-with-RAG/
├── src/                                    # Source modules
│   ├── __init__.py
│   ├── config.py                           # Constants (model names, params)
│   ├── models.py                           # Embedding + LLM loading
│   ├── pipeline.py                         # PDF processing + RAG chain
│   └── ui.py                               # Chat UI helpers
├── app.py                                  # Entry: Q&A đơn giản
├── chatbot_app.py                          # Entry: Chatbot đầy đủ
├── [Code]_Project_RAG_Chatbot.ipynb        # Notebook gốc (Colab)
├── Personal Chatbot.pdf                    # Tài liệu hướng dẫn
├── requirements.txt                        # Thư viện Python
├── .env.example                            # Template biến môi trường
├── .gitignore                              # Git ignore rules
└── README.md                               # File này
```

## **Yêu cầu phần cứng**

- **GPU mode** (khuyến nghị): NVIDIA GPU với ≥ 6GB VRAM → sử dụng quantization 4-bit
- **CPU mode**: ≥ 16GB RAM → sử dụng bfloat16 (chậm hơn đáng kể)

> **Lưu ý**: Chương trình tự động phát hiện GPU. Nếu có CUDA sẽ dùng quantization 4-bit, nếu không sẽ fallback về CPU mode.

## **Quy trình hoạt động**

1. Người dùng upload file PDF
2. Hệ thống đọc nội dung PDF và chia nhỏ văn bản bằng Semantic Chunker
3. Các đoạn văn được chuyển thành vector embeddings và lưu vào vector database
4. Khi người dùng đặt câu hỏi:
    - Retriever tìm các đoạn văn liên quan
    - LLM (Vicuna) sinh câu trả lời dựa trên context truy xuất
5. Kết quả được hiển thị trực tiếp trên giao diện chat
