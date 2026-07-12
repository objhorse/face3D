import unittest

import numpy as np

from src.appearance.texture_registration import build_sampling_warp


class TextureRegistrationTests(unittest.TestCase):
    def test_warp_reduces_control_residual_and_anchors_edges(self):
        model = np.array(
            [[35, 35], [50, 35], [65, 35], [35, 50], [50, 50], [65, 50], [40, 65], [60, 65]],
            dtype=np.float32,
        )
        observed = model + np.array([4.0, -3.0], dtype=np.float32)
        warp = build_sampling_warp(model, observed, (100, 100), smoothing=0.1)
        self.assertLess(warp.control_residual_after_px, warp.control_residual_before_px)
        edge = warp.apply(np.array([[0, 0], [99, 99]], dtype=np.float32), (100, 100))
        np.testing.assert_allclose(edge, np.array([[0, 0], [99, 99]]), atol=0.75)

    def test_warp_scales_to_higher_resolution(self):
        model = np.array(
            [[30, 30], [50, 30], [70, 30], [30, 50], [50, 50], [70, 50], [40, 70], [60, 70]],
            dtype=np.float32,
        )
        observed = model + np.array([5.0, 0.0], dtype=np.float32)
        warp = build_sampling_warp(model, observed, (100, 100), smoothing=0.1)
        point = warp.apply(np.array([[100, 100]], dtype=np.float32), (200, 200))
        self.assertGreater(point[0, 0], 106.0)

    def test_rejects_extreme_control_outlier(self):
        model = np.array(
            [[20, 20], [40, 20], [60, 20], [20, 40], [40, 40], [60, 40], [30, 60], [50, 60]],
            dtype=np.float32,
        )
        observed = model + np.array([3.0, 2.0], dtype=np.float32)
        observed[-1] += 80.0
        warp = build_sampling_warp(model, observed, (100, 100), smoothing=1.0)
        self.assertLessEqual(warp.max_displacement_px, 28.01)


if __name__ == "__main__":
    unittest.main()
