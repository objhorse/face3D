import numpy as np
import pytest

from src.module3_texture import bake_texture, rasterize_uv_map


def _tiny_bake_inputs(tex_size=16):
    vertices = np.array(
        [
            [-0.5, -0.5, 2.0],
            [0.0, 0.5, 2.0],
            [0.5, -0.5, 2.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv_verts = np.array(
        [[0.1, 0.1], [0.5, 0.9], [0.9, 0.1]],
        dtype=np.float32,
    )
    uv_faces = np.array([[0, 1, 2]], dtype=np.int32)
    tri_map, bary_map = rasterize_uv_map(uv_verts, uv_faces, tex_size)
    camera = {
        "K": np.array(
            [[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "R": np.eye(3, dtype=np.float32),
        "t": np.zeros(3, dtype=np.float32),
    }
    image = np.full((16, 16, 3), [180, 120, 90], dtype=np.uint8)
    mask = np.full((16, 16), 255, dtype=np.uint8)
    return (
        vertices,
        faces,
        uv_verts,
        uv_faces,
        tri_map,
        bary_map,
        {"front": camera},
        {"front": image},
        {"front": mask},
    )


def test_strict_texture_mode_rejects_sampling_warps_before_bake():
    inputs = _tiny_bake_inputs()

    with pytest.raises(ValueError, match="rejects sampling_warps"):
        bake_texture(
            *inputs[:8],
            tex_size=16,
            face_masks=inputs[8],
            sampling_warps={"front": object()},
            sampling_mode="strict_projective",
        )


def test_strict_texture_bake_records_zero_coordinate_displacement():
    inputs = _tiny_bake_inputs()
    diagnostics = {}
    sampling_debug = {}

    texture = bake_texture(
        *inputs[:8],
        tex_size=16,
        face_masks=inputs[8],
        diagnostics=diagnostics,
        sampling_mode="strict_projective",
        sampling_debug_out=sampling_debug,
    )

    assert texture.shape == (16, 16, 3)
    assert diagnostics["sampling_mode"] == "strict_projective"
    metrics = diagnostics["sampling_coordinates"]["front"]
    assert metrics["exact_coordinate_ratio"] == 1.0
    assert metrics["max_displacement_px"] == pytest.approx(0.0)
    assert sampling_debug["view_names"] == ("front",)
    assert sampling_debug["observed"].any()
