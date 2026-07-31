import numpy as np

from src.appearance.nasal_local_texture import (
    build_nasal_uv_alpha,
    composite_pixel_locked_texture,
)
from src.appearance.texture_registration import (
    LocalFeatureSpec,
    build_multi_feature_warp,
)
from src.appearance.semantic_feature_registration import (
    NostrilObservations,
    SemanticFeatureControls,
)
from src.appearance.stable_texture_registration import (
    replace_nostril_model_controls,
)


def _ordered_nose_controls():
    left_model = np.column_stack(
        (np.linspace(70.0, 100.0, 6), np.full(6, 112.0))
    )
    right_model = np.column_stack(
        (np.linspace(100.0, 130.0, 6), np.full(6, 112.0))
    )
    left_observed = np.column_stack(
        (
            np.array([60.0, 70.0, 80.0, 89.0, 96.0, 100.0]),
            np.full(6, 113.0),
        )
    )
    right_observed = np.column_stack(
        (
            np.array([100.0, 104.0, 111.0, 120.0, 130.0, 140.0]),
            np.full(6, 113.0),
        )
    )
    model = np.vstack(
        (
            np.array([[100.0, 82.0]]),
            left_model,
            right_model,
            np.array([[84.0, 108.0], [116.0, 108.0]]),
        )
    ).astype(np.float32)
    observed = np.vstack(
        (
            np.array([[100.0, 83.0]]),
            left_observed,
            right_observed,
            np.array([[77.0, 109.0], [123.0, 109.0]]),
        )
    ).astype(np.float32)
    groups = (
        "tip_anchor",
        *(("lower_left",) * 6),
        *(("lower_right",) * 6),
        "nostril_left",
        "nostril_right",
    )
    return model, observed, groups


def test_ordered_nasal_field_expands_nostrils_without_horizontal_folding():
    model, observed, groups = _ordered_nose_controls()
    global_controls = np.array(
        [[75.0, 145.0], [90.0, 138.0], [110.0, 138.0], [125.0, 145.0]],
        dtype=np.float32,
    )
    warp = build_multi_feature_warp(
        global_controls,
        global_controls.copy(),
        (
            LocalFeatureSpec(
                "nose",
                model,
                observed,
                strategy="ordered_nasal",
                groups=groups,
                max_displacement_px=28.0,
            ),
        ),
        (200, 200),
        grid_size=128,
        min_jacobian=0.35,
        independent_feature_backtracking=True,
    )

    corrected = warp.apply(model, (200, 200))
    residual = np.linalg.norm(corrected - observed, axis=1)
    row = np.column_stack(
        (np.linspace(45.0, 155.0, 221), np.full(221, 110.0))
    ).astype(np.float32)
    mapped_row = warp.apply(row, (200, 200))

    assert float(residual.mean()) < 5.0
    assert np.all(np.diff(mapped_row[:, 0]) > 0.0)
    assert warp.min_jacobian >= 0.35
    assert warp.diagnostics["local_scale_applied"] == 1.0
    assert warp.diagnostics["feature_scale_applied"]["nose"] == 1.0
    assert (
        warp.diagnostics["features"]["nose"]["strategy"]
        == "ordered_nasal"
    )


def test_pixel_locked_composite_preserves_every_unowned_byte():
    baseline = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
    candidate = np.full_like(baseline, 240)
    alpha = np.zeros((12, 10), dtype=np.float32)
    alpha[3:9, 2:8] = 0.5
    alpha[5:7, 4:6] = 1.0

    result, report = composite_pixel_locked_texture(
        baseline,
        candidate,
        alpha,
    )

    outside = alpha == 0.0
    assert np.array_equal(result[outside], baseline[outside])
    assert np.any(result[~outside] != baseline[~outside])
    assert report["outside_exact"] is True
    assert report["owned_texels"] == int(np.count_nonzero(~outside))


def test_nasal_uv_alpha_only_owns_front_projected_mask():
    vertices = np.array(
        [
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [1.0, 1.0, 2.0],
            [-1.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uv_vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    camera = {
        "K": np.array(
            [[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "R": np.eye(3, dtype=np.float32),
        "t": np.zeros(3, dtype=np.float32),
    }
    nose_mask = np.zeros((32, 32), dtype=np.uint8)
    nose_mask[12:21, 12:21] = 255

    alpha, report = build_nasal_uv_alpha(
        vertices,
        faces,
        uv_vertices,
        faces,
        camera,
        nose_mask,
        texture_size=32,
        source_mask_dilate_px=0,
        feather_px=2.0,
        depth_tolerance_ratio=0.02,
    )

    assert alpha.shape == (32, 32)
    assert 0 < np.count_nonzero(alpha) < alpha.size
    assert np.all(alpha[:3] == 0)
    assert np.all(alpha[-3:] == 0)
    assert report["owned_texels"] == int(np.count_nonzero(alpha))


def test_rendered_nostril_controls_replace_only_landmark_proxies():
    controls = SemanticFeatureControls(
        model_points=np.array(
            [[10.0, 2.0], [4.0, 8.0], [16.0, 8.0]],
            dtype=np.float32,
        ),
        observed_points=np.array(
            [[10.0, 3.0], [2.0, 9.0], [18.0, 9.0]],
            dtype=np.float32,
        ),
        groups=("tip_anchor", "nostril_left", "nostril_right"),
        confidence=0.9,
        diagnostics={},
    )
    rendered = NostrilObservations(
        centers=np.array([[5.5, 8.5], [14.5, 8.6]], dtype=np.float32),
        confidence=0.92,
        diagnostics={"source": "render"},
    )

    replaced = replace_nostril_model_controls(controls, rendered)

    assert np.array_equal(replaced.model_points[0], controls.model_points[0])
    assert np.array_equal(replaced.model_points[1:], rendered.centers)
    assert np.array_equal(replaced.observed_points, controls.observed_points)
    assert (
        replaced.diagnostics["model_nostril_source"]
        == "rendered_reference_texture"
    )
