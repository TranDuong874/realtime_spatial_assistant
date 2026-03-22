from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence
import tempfile
import re

import cv2
import numpy as np
from PIL import Image

import config


class PaddleOCRService:
    def __init__(
        self,
        *,
        language: str = config.PADDLEOCR_LANGUAGE,
        ocr_version: str = config.PADDLEOCR_OCR_VERSION,
        use_gpu: bool = config.PADDLEOCR_USE_GPU,
        use_textline_orientation: bool = config.PADDLEOCR_USE_TEXTLINE_ORIENTATION,
        text_detection_model_name: str | None = config.PADDLEOCR_TEXT_DETECTION_MODEL_NAME,
        text_recognition_model_name: str | None = config.PADDLEOCR_TEXT_RECOGNITION_MODEL_NAME,
        text_det_thresh: float = config.PADDLEOCR_TEXT_DET_THRESH,
        text_det_box_thresh: float = config.PADDLEOCR_TEXT_DET_BOX_THRESH,
        text_rec_score_thresh: float = config.PADDLEOCR_TEXT_REC_SCORE_THRESH,
        min_text_chars: int = config.PADDLEOCR_MIN_TEXT_CHARS,
    ) -> None:
        self.language = language
        self.ocr_version = ocr_version
        self.use_gpu = use_gpu
        self.use_textline_orientation = use_textline_orientation
        self.text_detection_model_name = text_detection_model_name
        self.text_recognition_model_name = text_recognition_model_name
        self.text_det_thresh = text_det_thresh
        self.text_det_box_thresh = text_det_box_thresh
        self.text_rec_score_thresh = text_rec_score_thresh
        self.min_text_chars = min_text_chars

        self._ocr = None

    def recognize_frame(self, frame: np.ndarray | Image.Image | str | Path) -> list[dict[str, Any]]:
        return self.recognize_frames([frame])[0]

    def recognize_frames(
        self,
        frames: Sequence[np.ndarray | Image.Image | str | Path],
    ) -> list[list[dict[str, Any]]]:
        if not frames:
            return []

        ocr = self._get_ocr()
        results: list[list[dict[str, Any]]] = []
        temp_paths: list[Path] = []
        try:
            for frame in frames:
                source_path = self._prepare_input(frame, temp_paths)
                raw_result = self._run_ocr(ocr, source_path)
                results.append(self._format_result(raw_result))
        finally:
            for path in temp_paths:
                if path.exists():
                    path.unlink()

        return results

    def merge_text(self, detections: Sequence[dict[str, Any]]) -> str:
        texts = [str(item["text"]).strip() for item in detections if str(item.get("text", "")).strip()]
        return " ".join(texts)

    def _get_ocr(self) -> Any:
        if self._ocr is not None:
            return self._ocr

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "paddleocr is required for PaddleOCRService. Install it with `pip install paddleocr` "
                "and install a matching `paddlepaddle-gpu` wheel for GPU inference."
            ) from exc

        try:
            import paddle
        except ImportError as exc:
            raise ImportError(
                "paddle is required for PaddleOCRService. Install `paddlepaddle-gpu` for GPU inference "
                "or `paddlepaddle` for CPU-only use."
            ) from exc

        resolved_use_gpu = self.use_gpu and paddle.device.is_compiled_with_cuda()
        self.use_gpu = resolved_use_gpu

        init_kwargs: dict[str, Any] = {
            "device": "gpu:0" if resolved_use_gpu else "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": self.use_textline_orientation,
            "text_det_thresh": self.text_det_thresh,
            "text_det_box_thresh": self.text_det_box_thresh,
            "text_rec_score_thresh": self.text_rec_score_thresh,
        }
        if self.text_detection_model_name is not None:
            init_kwargs["text_detection_model_name"] = self.text_detection_model_name
        if self.text_recognition_model_name is not None:
            init_kwargs["text_recognition_model_name"] = self.text_recognition_model_name
        if self.text_detection_model_name is None and self.text_recognition_model_name is None:
            init_kwargs["lang"] = self.language
            init_kwargs["ocr_version"] = self.ocr_version

        self._ocr = PaddleOCR(**init_kwargs)
        return self._ocr

    def _prepare_input(
        self,
        frame: np.ndarray | Image.Image | str | Path,
        temp_paths: list[Path],
    ) -> str:
        if isinstance(frame, (str, Path)):
            return str(Path(frame).expanduser().resolve())

        rgb_frame = self._to_rgb_array(frame)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = Path(handle.name)
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(temp_path), bgr_frame):
            raise ValueError("Failed to write temporary image for PaddleOCR inference.")
        temp_paths.append(temp_path)
        return str(temp_path)

    def _run_ocr(self, ocr: Any, source_path: str) -> Any:
        if hasattr(ocr, "predict"):
            result = ocr.predict(
                source_path,
                text_det_thresh=self.text_det_thresh,
                text_det_box_thresh=self.text_det_box_thresh,
                text_rec_score_thresh=self.text_rec_score_thresh,
            )
            if isinstance(result, list):
                return result
            return [result]

        if hasattr(ocr, "ocr"):
            result = ocr.ocr(source_path, cls=self.use_textline_orientation)
            return [] if result is None else result

        raise AttributeError("Unsupported PaddleOCR instance: expected `predict` or `ocr`.")

    def _format_result(self, raw_result: Any) -> list[dict[str, Any]]:
        if raw_result is None:
            return []

        if isinstance(raw_result, list) and raw_result and hasattr(raw_result[0], "res"):
            return self._format_predict_result([item.res for item in raw_result])

        if isinstance(raw_result, list) and raw_result and hasattr(raw_result[0], "get"):
            return self._format_predict_result(raw_result)

        if isinstance(raw_result, list) and len(raw_result) == 1 and isinstance(raw_result[0], list):
            return self._format_legacy_result(raw_result[0])

        if isinstance(raw_result, list):
            return self._format_legacy_result(raw_result)

        raise TypeError(f"Unsupported PaddleOCR result type: {type(raw_result)!r}")

    def _format_predict_result(self, raw_result: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for page in raw_result:
            polygons = page.get("rec_polys")
            texts = page.get("rec_texts")
            scores = page.get("rec_scores")
            if polygons is None or texts is None or scores is None:
                continue

            polygon_list = np.asarray(polygons).tolist()
            score_list = np.asarray(scores).tolist()
            for polygon, text, score in zip(polygon_list, list(texts), score_list):
                normalized_text = self._normalize_text(text)
                if not self._keep_detection(normalized_text, score):
                    continue
                formatted.append(
                    {
                        "text": normalized_text,
                        "confidence": float(score),
                        "polygon": [[float(x), float(y)] for x, y in polygon],
                    }
                )
        return formatted

    def _format_legacy_result(self, raw_result: list[Any]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for item in raw_result:
            if not isinstance(item, list) or len(item) != 2:
                continue

            polygon, text_info = item
            if not isinstance(text_info, (list, tuple)) or len(text_info) != 2:
                continue
            text, score = text_info
            normalized_text = self._normalize_text(text)
            if not self._keep_detection(normalized_text, score):
                continue
            formatted.append(
                {
                    "text": normalized_text,
                    "confidence": float(score),
                    "polygon": [[float(x), float(y)] for x, y in polygon],
                }
            )
        return formatted

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text).strip())

    def _keep_detection(self, text: str, score: Any) -> bool:
        score_value = float(score)
        if score_value < self.text_rec_score_thresh:
            return False
        if not text:
            return False
        alnum_chars = [char for char in text if char.isalnum()]
        if len(alnum_chars) < self.min_text_chars:
            return False
        if any(char.isalpha() for char in alnum_chars):
            alpha_ratio = sum(char.isalpha() for char in alnum_chars) / len(alnum_chars)
            if len(alnum_chars) <= 3 and alpha_ratio < 0.5:
                return False
        return True

    def _to_rgb_array(self, frame: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(frame, Image.Image):
            return np.asarray(frame.convert("RGB"))

        if isinstance(frame, np.ndarray):
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Expected frame array with shape (H, W, 3).")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        raise TypeError("Expected a PIL image, numpy image array, or filesystem path.")
