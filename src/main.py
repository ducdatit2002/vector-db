# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# Một số môi trường macOS/Conda có thể crash ở native OpenMP thread runtime
# khi FAISS và PyTorch/SentenceTransformers cùng được load. Các biến này cần
# được set trước khi import faiss, numpy, sentence_transformers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import faiss
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Không tìm thấy module 'faiss'. Hãy cài package bằng lệnh:\n"
        "    pip install -r requirements.txt\n"
        "Nếu đang dùng nhiều môi trường Python, kiểm tra interpreter bằng:\n"
        "    python -m pip show faiss-cpu"
    ) from exc

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@dataclass
class Chunk:
    """Một đoạn văn bản nhỏ kèm metadata để hiển thị kết quả tìm kiếm."""

    text: str
    source: str
    chunk_id: int

def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    """Đọc toàn bộ file .txt trong thư mục data."""
    documents: list[tuple[str, str]] = []

    for file_path in sorted(data_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            documents.append((file_path.name, text))

    return documents

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Chia văn bản thành các chunk khoảng chunk_size ký tự, có overlap.

    Overlap giúp câu ở ranh giới giữa hai chunk không bị mất ngữ cảnh.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap phải nhỏ hơn chunk_size")

    chunks: list[str] = []
    start = 0

    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Ưu tiên kết thúc chunk tại khoảng trắng để không cắt giữa chữ.
        if end < text_length:
            split_at = text.rfind(" ", start + chunk_size // 2, end)
            if split_at != -1:
                end = split_at

        chunk_start = start
        if chunk_start > 0 and not text[chunk_start].isspace() and not text[chunk_start - 1].isspace():
            previous_space = text.rfind(" ", max(0, chunk_start - 30), chunk_start)
            if previous_space != -1:
                chunk_start = previous_space + 1

        chunk = text[chunk_start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks

def build_chunks(documents: list[tuple[str, str]]) -> list[Chunk]:
    """Tạo danh sách chunk từ nhiều tài liệu."""
    chunks: list[Chunk] = []

    for source, text in documents:
        for index, chunk in enumerate(chunk_text(text)):
            chunks.append(Chunk(text=chunk, source=source, chunk_id=index))

    return chunks


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Chuẩn hóa vector để inner product tương đương cosine similarity."""
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Tạo FAISS IndexFlatIP và nạp embeddings vào index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def semantic_search(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    top_k: int = 3,
    threshold: float = 0.35,
) -> list[dict[str, object]]:
    """Tìm các chunk gần nghĩa nhất với câu hỏi bằng vector search."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = normalize_embeddings(query_embedding)

    scores, ids = index.search(query_embedding, top_k)

    results: list[dict[str, object]] = []
    for score, chunk_index in zip(scores[0], ids[0]):
        if chunk_index == -1 or float(score) < threshold:
            continue

        chunk = chunks[int(chunk_index)]
        results.append(
            {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "score": float(score),
            }
        )

    return results

def keyword_search(query: str, chunks: list[Chunk], top_k: int = 3) -> list[dict[str, object]]:
    """Demo tìm kiếm keyword đơn giản để so sánh với semantic search."""
    keywords = {word.lower().strip(".,!?;:()[]") for word in query.split()}
    scored_results: list[dict[str, object]] = []

    for chunk in chunks:
        chunk_words = {word.lower().strip(".,!?;:()[]") for word in chunk.text.split()}
        matched = keywords.intersection(chunk_words)
        if matched:
            scored_results.append(
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "score": len(matched),
                }
            )

    return sorted(scored_results, key=lambda item: item["score"], reverse=True)[:top_k]

def print_results(title: str, results: list[dict[str, object]], no_result_message: str) -> None:
    print(f"\n=== {title} ===")

    if not results:
        print(no_result_message)
        return

    for order, result in enumerate(results, start=1):
        print(f"\nKết quả {order}")
        print(f"Source: {result['source']} | Chunk: {result['chunk_id']} | Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VectorDB Company FAQ semantic search demo")
    parser.add_argument(
        "--query",
        "-q",
        default="Nhân viên được làm việc từ xa mấy ngày mỗi tuần?",
        help="Câu hỏi cần tìm trong tài liệu công ty.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Số kết quả gần nghĩa cần trả về.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Ngưỡng similarity tối thiểu. Thấp hơn ngưỡng sẽ bị bỏ qua.",
    )
    parser.add_argument(
        "--compare-keyword",
        action="store_true",
        help="Hiển thị thêm kết quả keyword search để so sánh.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    faiss.omp_set_num_threads(1)

    documents = load_documents(DATA_DIR)
    if not documents:
        raise RuntimeError(f"Không tìm thấy file .txt nào trong {DATA_DIR}")

    chunks = build_chunks(documents)
    print(f"Đã load {len(documents)} tài liệu và tạo {len(chunks)} chunks.")

    print(f"Đang tải embedding model: {DEFAULT_MODEL_NAME}")
    model = SentenceTransformer(DEFAULT_MODEL_NAME)

    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize_embeddings(embeddings)

    index = build_faiss_index(embeddings)
    print(f"Đã tạo FAISS IndexFlatIP với {index.ntotal} vectors.")

    semantic_results = semantic_search(
        query=args.query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=args.top_k,
        threshold=args.threshold,
    )

    print(f"\nQuery: {args.query}")
    print_results("Vector Search", semantic_results, "Không tìm thấy thông tin")

    if args.compare_keyword:
        keyword_results = keyword_search(args.query, chunks, top_k=args.top_k)
        print_results("Keyword Search", keyword_results, "Keyword search không tìm thấy kết quả")


if __name__ == "__main__":
    main()
