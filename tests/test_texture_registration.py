import unittest

import numpy as np

from src.appearance.texture_registration import (
    LocalFeatureSpec,
    build_layered_feature_warp,
    build_multi_feature_warp,
    build_sampling_warp,
    displacement_field_metrics,
)


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

    def test_layered_warp_bounds_global_similarity(self):
        global_model = np.array(
            [[20, 20], [80, 20], [20, 80], [80, 80], [50, 50], [35, 50]],
            dtype=np.float32,
        )
        global_observed = global_model + np.array([30.0, -20.0], dtype=np.float32)
        local_model = np.array(
            [[40, 55], [45, 60], [50, 62], [55, 60], [60, 55], [50, 52]],
            dtype=np.float32,
        )
        local_observed = local_model.copy()

        warp = build_layered_feature_warp(
            global_model,
            global_observed,
            local_model,
            local_observed,
            (100, 100),
            max_translation_px=8.0,
            max_rotation_degrees=1.0,
            max_scale_delta=0.015,
        )

        center = warp.apply(np.array([[50, 50]], dtype=np.float32), (100, 100))[0]
        self.assertLessEqual(abs(center[0] - 50), 8.1)
        self.assertLessEqual(abs(center[1] - 50), 8.1)
        self.assertLessEqual(float(np.linalg.norm(center - [50, 50])), 8.1)
        self.assertLessEqual(
            float(
                np.hypot(
                    warp.diagnostics["translation_x_px"],
                    warp.diagnostics["translation_y_px"],
                )
            ),
            8.01,
        )

    def test_layered_warp_local_field_is_confined_and_improves_nose_controls(self):
        global_model = np.array(
            [[20, 20], [80, 20], [20, 80], [80, 80], [50, 35], [50, 75]],
            dtype=np.float32,
        )
        local_model = np.array(
            [[40, 55], [44, 60], [50, 63], [56, 60], [60, 55], [50, 52]],
            dtype=np.float32,
        )
        local_observed = local_model + np.array([4.0, 2.0], dtype=np.float32)

        warp = build_layered_feature_warp(
            global_model,
            global_model,
            local_model,
            local_observed,
            (100, 100),
            local_max_displacement_px=10.0,
        )

        before = np.linalg.norm(local_model - local_observed, axis=1).mean()
        after = np.linalg.norm(warp.apply(local_model, (100, 100)) - local_observed, axis=1).mean()
        self.assertLess(after, before * 0.5)
        outside = warp.apply(np.array([[5, 5], [95, 95]], dtype=np.float32), (100, 100))
        np.testing.assert_allclose(outside, np.array([[5, 5], [95, 95]]), atol=0.4)

    def test_displacement_metrics_report_nonfolding_identity(self):
        field = np.zeros((32, 32, 2), dtype=np.float32)
        metrics = displacement_field_metrics(field, (100, 100))
        self.assertAlmostEqual(metrics["min_jacobian"], 1.0, places=5)
        self.assertAlmostEqual(metrics["displacement_p95_px"], 0.0, places=5)

    def test_layered_warp_scales_local_field_instead_of_dropping_it(self):
        global_model = np.array(
            [[10, 10], [90, 10], [10, 90], [90, 90], [50, 20], [50, 80]],
            dtype=np.float32,
        )
        local_model = np.array(
            [[38, 48], [42, 55], [48, 60], [52, 60], [58, 55], [62, 48]],
            dtype=np.float32,
        )
        local_observed = local_model.copy()
        local_observed[:3, 0] += 18.0
        local_observed[3:, 0] -= 18.0

        warp = build_layered_feature_warp(
            global_model,
            global_model,
            local_model,
            local_observed,
            (100, 100),
            local_max_displacement_px=18.0,
            min_jacobian=0.35,
        )

        self.assertGreaterEqual(warp.min_jacobian, 0.35)
        self.assertGreaterEqual(warp.diagnostics["local_scale_applied"], 0.0)
        self.assertLessEqual(warp.diagnostics["local_scale_applied"], 1.0)
        if warp.diagnostics["local_scale_applied"] > 0.0:
            moved = np.linalg.norm(warp.apply(local_model, (100, 100)) - local_model, axis=1)
            self.assertGreater(float(moved.max()), 0.1)

    def test_multi_feature_warp_aligns_eyes_independently(self):
        global_model = np.array(
            [[20, 20], [80, 20], [20, 80], [80, 80], [50, 25], [50, 75]],
            dtype=np.float32,
        )
        right_eye = np.array(
            [[20, 38], [24, 36], [29, 36], [34, 38], [29, 39], [24, 39]],
            dtype=np.float32,
        )
        left_eye = np.array(
            [[66, 38], [71, 36], [76, 36], [80, 38], [76, 39], [71, 39]],
            dtype=np.float32,
        )
        right_observed = right_eye + np.array([0.0, 5.0], dtype=np.float32)
        right_observed[[0, 3], 0] += np.array([-2.0, 2.0], dtype=np.float32)
        left_observed = left_eye + np.array([0.0, 3.0], dtype=np.float32)

        warp = build_multi_feature_warp(
            global_model,
            global_model,
            (
                LocalFeatureSpec(
                    "subject_right_eye",
                    right_eye,
                    right_observed,
                    radius_x_scale=1.2,
                    radius_y_scale=2.2,
                ),
                LocalFeatureSpec(
                    "subject_left_eye",
                    left_eye,
                    left_observed,
                    radius_x_scale=1.2,
                    radius_y_scale=2.2,
                ),
            ),
            (100, 100),
            local_max_displacement_px=10.0,
            min_jacobian=0.35,
        )

        right_after = warp.apply(right_eye, (100, 100))
        left_after = warp.apply(left_eye, (100, 100))
        self.assertLess(
            np.linalg.norm(right_after - right_observed, axis=1).mean(),
            1.0,
        )
        self.assertLess(
            np.linalg.norm(left_after - left_observed, axis=1).mean(),
            1.0,
        )
        outside = warp.apply(
            np.array([[5, 5], [95, 95], [50, 90]], dtype=np.float32),
            (100, 100),
        )
        np.testing.assert_allclose(
            outside,
            np.array([[5, 5], [95, 95], [50, 90]], dtype=np.float32),
            atol=0.4,
        )
        self.assertGreaterEqual(warp.min_jacobian, 0.35)
        self.assertEqual(
            set(warp.diagnostics["features"]),
            {"subject_right_eye", "subject_left_eye"},
        )


if __name__ == "__main__":
    unittest.main()
