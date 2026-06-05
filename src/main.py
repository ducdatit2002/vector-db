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

