from __future__ import annotations

import numpy as np

from src.geometry.semantic_eyelid_rig import build_semantic_eyelid_rig


def _two_eye_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for eye_offset in (0.0, 10.0):
        start = len(vertices)
        for y in range(3):
            for x in range(3):
                vertices.append([eye_offset + x, y, 0.1 * (1 - abs(x - 1))])
        for y in range(2):
            for x in range(2):
                a = start + y * 3 + x
                faces.extend(([a, a + 1, a + 3], [a + 1, a + 4, a + 3]))
    vertices.append([100.0, 100.0, 100.0])
    mapping = np.zeros((68, 3), dtype=np.int64)
    right_triangles = {
        36: [0, 3, 1], 37: [3, 4, 1], 38: [4, 5, 2],
        39: [2, 5, 4], 40: [4, 8, 5], 41: [3, 7, 4],
    }
    left_triangles = {
        42: [9, 12, 10], 43: [12, 13, 10], 44: [13, 14, 11],
        45: [11, 14, 13], 46: [13, 17, 14], 47: [12, 16, 13],
    }
    for index, triangle in {**right_triangles, **left_triangles}.items():
        mapping[index] = triangle
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
        mapping,
    )


def test_builds_eight_compact_semantic_controls() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    rig = build_semantic_eyelid_rig(vertices, faces, mapping, support_rings=2)

    assert len(rig.control_names) == 8
    assert rig.weights.shape == (len(vertices), 8)
    assert np.isfinite(rig.weights).all()
    assert np.all(rig.weights >= 0.0)
    assert 18 not in rig.active_vertices
    assert np.all(rig.weights[18] == 0.0)


def test_left_and_right_eye_supports_do_not_cross() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    rig = build_semantic_eyelid_rig(vertices, faces, mapping, support_rings=2)
    right_columns = [i for i, name in enumerate(rig.control_names) if "right" in name]
    left_columns = [i for i, name in enumerate(rig.control_names) if "left" in name]

    assert np.all(rig.weights[:9, left_columns] == 0.0)
    assert np.all(rig.weights[9:18, right_columns] == 0.0)


def test_applying_offsets_keeps_every_vertex_outside_support_exact() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    rig = build_semantic_eyelid_rig(vertices, faces, mapping, support_rings=2)
    offsets = np.zeros((8, 3), dtype=np.float32)
    offsets[:, 2] = 0.25
    candidate = rig.apply(vertices, offsets)
    outside = np.setdiff1d(np.arange(len(vertices)), rig.active_vertices)

    np.testing.assert_array_equal(candidate[outside], vertices[outside])
    assert np.linalg.norm(candidate[rig.core_vertices] - vertices[rig.core_vertices]) > 0.0


def test_topology_reuses_same_semantic_indices_for_new_identity() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    first = build_semantic_eyelid_rig(vertices, faces, mapping, support_rings=2)
    second_vertices = vertices * np.array([1.2, 0.9, 1.1], dtype=np.float32)
    second = build_semantic_eyelid_rig(second_vertices, faces, mapping, support_rings=2)

    np.testing.assert_array_equal(first.active_vertices, second.active_vertices)
    np.testing.assert_array_equal(first.core_vertices, second.core_vertices)


def test_tiny_edges_receive_nearly_rigid_control_weights() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    vertices = vertices.copy()
    vertices[3] = vertices[0] + np.array([0.0, 1e-3, 0.0], dtype=np.float32)
    rig = build_semantic_eyelid_rig(
        vertices,
        faces,
        mapping,
        support_rings=2,
        core_rings=1,
        sigma_rings=1.0,
    )

    tiny_edge_jump = np.linalg.norm(rig.weights[3] - rig.weights[0])
    regular_edge_jumps = [
        np.linalg.norm(rig.weights[b] - rig.weights[a])
        for a, b in ((0, 1), (3, 4), (1, 4))
    ]

    assert tiny_edge_jump < 0.25 * float(np.median(regular_edge_jumps))
