import unittest

import numpy as np

from src.module3_texture import (
    _feather_view_weight,
    _multiband_blend,
    _small_connected_regions,
    _texture_alpha_from_observation,
)


class TextureBlendingTests(unittest.TestCase):
    def test_multiband_blend_is_finite_and_bounded(self):
        left = np.full((64, 64, 3), [180, 120, 100], dtype=np.float32)
        right = np.full((64, 64, 3), [150, 145, 125], dtype=np.float32)
        x = np.linspace(1.0, 0.0, 64, dtype=np.float32)[None, :]
        first_weight = np.repeat(x, 64, axis=0)
        blended = _multiband_blend([left, right], [first_weight, 1.0 - first_weight], levels=3)
        self.assertTrue(np.isfinite(blended).all())
        self.assertGreaterEqual(float(blended.min()), 99.0)
        self.assertLessEqual(float(blended.max()), 181.0)

    def test_only_small_missing_regions_are_selected_for_inpaint(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        mask[20:60, 20:60] = 1
        selected = _small_connected_regions(mask, max_area=100)
        self.assertEqual(int(selected.sum()), 16)

    def test_unobserved_uv_stays_transparent(self):
        valid = np.ones((64, 64), dtype=bool)
        observed = np.zeros((64, 64), dtype=bool)
        observed[20:44, 20:44] = True
        alpha = _texture_alpha_from_observation(valid, observed)
        self.assertEqual(int(alpha[0, 0]), 0)
        self.assertEqual(int(alpha[32, 32]), 255)

    def test_geometry_keep_mask_hides_observed_non_face_region(self):
        valid = np.ones((64, 64), dtype=bool)
        observed = np.ones((64, 64), dtype=bool)
        keep = np.ones((64, 64), dtype=bool)
        keep[48:, :] = False
        alpha = _texture_alpha_from_observation(valid, observed, geometry_keep=keep)
        self.assertEqual(int(alpha[20, 20]), 255)
        self.assertEqual(int(alpha[56, 20]), 0)

    def test_view_weight_is_feathered_at_visibility_boundary(self):
        weight = np.zeros((128, 128), dtype=np.float32)
        weight[16:112, 16:112] = 1.0
        feathered = _feather_view_weight(weight, radius_px=24.0)
        self.assertEqual(float(feathered[0, 0]), 0.0)
        self.assertLess(float(feathered[17, 64]), 0.1)
        self.assertGreater(float(feathered[64, 64]), 0.95)


if __name__ == "__main__":
    unittest.main()
