# -*- coding: utf-8 -*-
"""Kiểm tra nhanh dependency native để debug lỗi môi trường.

Chạy:
    python src/check_env.py
"""

from __future__ import annotations

import os
import platform
import sys


def print_step(message: str) -> None:
    print(f"\n[check] {message}", flush=True)


def main() -> None:
    print_step("Python runtime")
    print(sys.version)
    print(sys.executable)
    print(platform.platform())

    for name in [
        "OMP_NUM_THREADS",
        "KMP_DUPLICATE_LIB_OK",
        "KMP_INIT_AT_FORK",
        "USE_TF",
        "TRANSFORMERS_NO_TF",
    ]:
        print(f"{name}={os.environ.get(name)}")

    print_step("Import numpy")
    import numpy as np

    print(f"numpy {np.__version__}")

    print_step("Import faiss")
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Không tìm thấy module 'faiss'. Cài dependency bằng:\n"
            "    pip install -r requirements.txt\n"
            "Sau đó kiểm tra lại bằng:\n"
            "    python -m pip show faiss-cpu"
        ) from exc

    print(f"faiss loaded: {faiss.__file__}")
    faiss.omp_set_num_threads(1)

    print_step("Import sentence_transformers")
    import sentence_transformers

    print(f"sentence-transformers {sentence_transformers.__version__}")

    print_step("All imports OK")


if __name__ == "__main__":
    main()
