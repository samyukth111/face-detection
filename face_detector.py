from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FaceBox = tuple[int, int, int, int]
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"


def load_face_detector(confidence_threshold: float = 0.85) -> cv2.FaceDetectorYN:
    if not 0 < confidence_threshold < 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YuNet face detection model not found: {MODEL_PATH}")

    return cv2.FaceDetectorYN.create(
        model=str(MODEL_PATH),
        config="",
        input_size=(320, 320),
        score_threshold=confidence_threshold,
        nms_threshold=0.3,
        top_k=5000,
    )


def detect_faces(
    image: Image.Image,
    detector: cv2.FaceDetectorYN,
    min_face_size: int = 30,
    confidence_threshold: float = 0.85,
) -> tuple[Image.Image, list[FaceBox]]:
    if min_face_size < 1:
        raise ValueError("min_face_size must be greater than 0.")
    if not 0 < confidence_threshold < 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    rgb_image = image.convert("RGB")
    pixels = np.array(rgb_image, copy=True)
    bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    detector.setInputSize((rgb_image.width, rgb_image.height))
    _, detections = detector.detect(bgr_pixels)

    face_data = [] if detections is None else [
        detection
        for detection in detections
        if detection[-1] >= confidence_threshold
        and detection[2] >= min_face_size
        and detection[3] >= min_face_size
    ]
    face_data.sort(key=lambda detection: (detection[1], detection[0]))
    face_boxes: list[FaceBox] = []

    for index, detection in enumerate(face_data, start=1):
        x, y, width, height = (int(round(float(value))) for value in detection[:4])
        confidence = float(detection[-1])
        face_boxes.append((x, y, width, height))
        cv2.rectangle(
            pixels,
            (x, y),
            (x + width, y + height),
            (37, 190, 130),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            pixels,
            f"Face {index}: {confidence:.0%}",
            (x, max(y - 10, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (37, 190, 130),
            2,
            cv2.LINE_AA,
        )

    return Image.fromarray(pixels), face_boxes
