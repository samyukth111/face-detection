import unittest

import numpy as np
from PIL import Image

from face_detector import detect_faces, load_face_detector


class FakeDetector:
    def __init__(self, detections: list[tuple[float, ...]] | None = None) -> None:
        self.detections = (
            np.asarray(detections, dtype=np.float32) if detections else None
        )
        self.input_size: tuple[int, int] | None = None

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.input_size = size

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        return 1, self.detections


class FaceDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = load_face_detector()

    def test_blank_image_returns_rgb_image_with_original_size(self) -> None:
        image = Image.new("RGB", (320, 240), color="white")

        annotated, faces = detect_faces(image, self.detector)

        self.assertEqual(faces, [])
        self.assertEqual(annotated.size, image.size)
        self.assertEqual(annotated.mode, "RGB")

    def test_only_confident_faces_above_minimum_size_are_counted(self) -> None:
        detector = FakeDetector(
            [
                (10, 20, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.98),
                (80, 20, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.75),
                (140, 20, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.99),
            ]
        )
        image = Image.new("RGB", (200, 100), color="white")

        annotated, faces = detect_faces(image, detector)

        self.assertEqual(faces, [(10, 20, 40, 40)])
        self.assertEqual(detector.input_size, (200, 100))
        self.assertEqual(image.getpixel((10, 30)), (255, 255, 255))
        self.assertNotEqual(annotated.getpixel((10, 30)), (255, 255, 255))

    def test_no_detections_returns_zero_faces(self) -> None:
        _, faces = detect_faces(Image.new("RGB", (80, 80)), FakeDetector())

        self.assertEqual(faces, [])

    def test_invalid_face_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            detect_faces(
                Image.new("RGB", (100, 100)),
                FakeDetector(),
                min_face_size=0,
            )

    def test_invalid_confidence_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            load_face_detector(confidence_threshold=1.0)


if __name__ == "__main__":
    unittest.main()
