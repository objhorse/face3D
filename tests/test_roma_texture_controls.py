import numpy as np
from types import SimpleNamespace

from src.appearance.roma_texture_controls import (
    build_roma_texture_controls,
    work_pixels_to_letterbox_canvas,
)
from src.appearance.stable_texture_registration import (
    _roma_nasal_displacement_limit,
)


def test_work_pixels_map_to_square_letterbox_canvas() -> None:
    mapped = work_pixels_to_letterbox_canvas(
        np.asarray([[0.0, 0.0], [640.0, 480.0]]),
        work_size=(640, 480),
        canvas_shape=(1024, 1024),
    )
    assert np.allclose(mapped, [[0.0, 128.0], [1024.0, 896.0]])


def test_roma_displacement_limit_tracks_trusted_cross_view_disparity() -> None:
    assert _roma_nasal_displacement_limit(None) == 28.0
    controls = SimpleNamespace(
        metadata={"maximum_selected_displacement_px": 55.25}
    )
    assert _roma_nasal_displacement_limit(controls) == 59.25
    excessive = SimpleNamespace(
        metadata={"maximum_selected_displacement_px": 100.0}
    )
    assert _roma_nasal_displacement_limit(excessive) == 72.0


def test_controls_deduplicate_front_vertices_and_preserve_both_sides() -> None:
    vertices = np.asarray(
        [[-0.4, -0.4, 1.0], [0.4, -0.4, 1.0], [0.0, 0.4, 1.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    camera = {
        "K": np.asarray([[500.0, 0.0, 512.0], [0.0, 500.0, 512.0], [0.0, 0.0, 1.0]]),
        "R": np.eye(3),
        "t": np.zeros(3),
    }
    trusted = []
    canvas_points = np.asarray(
        [[450.0, 400.0], [512.0, 400.0], [575.0, 400.0], [512.0, 500.0]]
    )
    for side in ("subject-left", "subject-right"):
        for index, model in enumerate(canvas_points):
            raw = (model - np.asarray([0.0, 128.0])) / 1.6
            trusted.append(
                {
                    "pair_match": {
                        "side_view": side,
                        "front_pixel": raw.tolist(),
                        "side_pixel": raw.tolist(),
                        "semantic_region": "alar_dome",
                        "confidence": 0.8,
                        "diagnostics": {"baseline_vertex_index": float(index)},
                    },
                    "weight": 0.8,
                }
            )
    controls = build_roma_texture_controls(
        {"observations": {"trusted": trusted}},
        vertices,
        faces,
        {"front": camera, "left": camera, "right": camera},
        minimum_spacing_px=1.0,
    )
    assert set(controls) == {"front", "left", "right"}
    assert len(controls["front"].model_points) == 4
    assert len(controls["left"].model_points) == 4
    assert len(controls["right"].model_points) == 4
    assert np.allclose(controls["front"].model_points, controls["front"].observed_points)
