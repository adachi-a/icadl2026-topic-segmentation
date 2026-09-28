#!/usr/bin/env python3
"""RapidOCR wrapper used by the production on-screen text stage."""

from __future__ import annotations

import ctypes
import os
import site
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _preload_cuda_libs() -> None:
    candidates: list[str] = []
    for site_packages in site.getsitepackages():
        candidates.extend(
            [
                f"{site_packages}/nvidia/cudnn/lib",
                f"{site_packages}/nvidia/cu13/lib",
                f"{site_packages}/nvidia/cublas/lib",
            ]
        )
    for directory in candidates:
        if not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if not (filename.startswith("lib") and ".so" in filename):
                continue
            try:
                ctypes.CDLL(os.path.join(directory, filename), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


class RapidOCRBackend:
    name = "rapidocr"

    def __init__(self) -> None:
        _preload_cuda_libs()
        import onnxruntime as ort
        from rapidocr_onnxruntime import RapidOCR

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(f"CUDAExecutionProvider is unavailable: {providers}")
        self.engine = RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True)

    def warmup(self) -> None:
        self.engine(np.zeros((100, 100, 3), dtype=np.uint8))

    def run(self, image_path: Path) -> dict[str, Any]:
        image = np.array(Image.open(image_path).convert("RGB"))
        started = time.perf_counter()
        result, _ = self.engine(image)
        elapsed_ms = (time.perf_counter() - started) * 1000
        texts: list[str] = []
        boxes: list[Any] = []
        for box, text, _score in result or []:
            texts.append(text)
            boxes.append([[float(x), float(y)] for x, y in box])
        return {"texts": texts, "boxes": boxes, "elapsed_ms": elapsed_ms}
