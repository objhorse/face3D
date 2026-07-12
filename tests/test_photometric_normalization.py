import unittest

import cv2
import numpy as np

from src.appearance.photometric_normalization import normalize_skin_chroma


class PhotometricNormalizationTests(unittest.TestCase):
    def test_spatial_chroma_cast_is_reduced_without_changing_luminance_much(self):
        image = np.full((128, 128, 3), [170, 135, 120], dtype=np.uint8)
        image[:, 48:80, 1] = np.clip(image[:, 48:80, 1].astype(int) + 18, 0, 255)
        mask = np.full((128, 128), 255, dtype=np.uint8)
        corrected, report, _ = normalize_skin_chroma(
            image, mask, np.zeros_like(mask), field_size=64, sigma=7.0
        )
        before_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        after_lab = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB).astype(np.float32)
        before = np.linalg.norm(
            before_lab[:, 56:72, 1:3].mean((0, 1)) - before_lab[:, :32, 1:3].mean((0, 1))
        )
        after = np.linalg.norm(
            after_lab[:, 56:72, 1:3].mean((0, 1)) - after_lab[:, :32, 1:3].mean((0, 1))
        )
        self.assertTrue(report["applied"])
        self.assertLess(after, before)
        self.assertLess(abs(float(corrected.mean()) - float(image.mean())), 6.0)


if __name__ == "__main__":
    unittest.main()
