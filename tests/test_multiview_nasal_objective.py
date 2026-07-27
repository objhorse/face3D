from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from src.cross_view_geometry import Camera
from src.geometry.multiview_nasal_objective import (
    MultiviewNasalSamplingConfig,
    build_candidate_nasal_mesh,
    project_multiview_nasal_boundaries,
)
from src.geometry.nasal_observations import (
    NasalObservationBundle,
    NasalViewObservation,
)
from src.geometry.observable_flame_subspace import ProjectionView


VIEW_NAMES = ("front", "subject-left", "subject-right")


def _rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )


def _mesh():
    vertices = np.array(
        [
            [-1.0, -1.0, 4.0],
            [0.0, -1.0, 3.5],
            [1.0, -1.0, 4.0],
            [-1.0, 1.0, 4.0],
            [0.0, 1.0, 3.5],
            [1.0, 1.0, 4.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _bases(vertex_count: int):
    flame_vectors = np.zeros((vertex_count, 3, 2), dtype=np.float64)
    flame_vectors[:, 0, 0] = np.linspace(0.01, 0.06, vertex_count)
    flame_vectors[:, 2, 1] = np.linspace(-0.03, 0.02, vertex_count)
    semantic_vectors = np.zeros((8, vertex_count, 3), dtype=np.float64)
    semantic_vectors[0, :, 1] = np.linspace(0.02, 0.08, vertex_count)
    semantic_vectors[4, :, 2] = 0.04

    support = np.ones(vertex_count, dtype=bool)
    bridge = np.zeros(vertex_count, dtype=bool)
    tip = np.zeros(vertex_count, dtype=bool)
    subject_left = np.zeros(vertex_count, dtype=bool)
    subject_right = np.zeros(vertex_count, dtype=bool)
    transition = np.ones(vertex_count, dtype=bool)
    if vertex_count == 6:
        bridge[[3, 4, 5]] = True
        tip[[1, 4]] = True
        subject_left[[2, 5]] = True
        subject_right[[0, 3]] = True
    else:
        tip[-3:] = True
        subject_left[4:7] = True
        subject_right[-3:] = True
    observable = SimpleNamespace(vertex_basis=flame_vectors)
    semantic = SimpleNamespace(
        vectors=semantic_vectors,
        support_mask=support,
        protected_mask=np.zeros(vertex_count, dtype=bool),
        mode_support_masks=np.ones((8, vertex_count), dtype=bool),
        region_masks={
            "nose_bridge": bridge,
            "nose_tip": tip,
            "subject_left_nose_wing": subject_left,
            "subject_right_nose_wing": subject_right,
            "tip_alar_transition": transition,
        },
    )
    return observable, semantic


def _camera(name: str, camera_view: str, K: np.ndarray) -> Camera:
    return Camera(
        name=name,
        view=camera_view,
        image_size=(100, 100),
        K=K.copy(),
        dist=np.zeros(5, dtype=np.float64),
        R_rig_to_camera=np.eye(3, dtype=np.float64),
        t_rig_to_camera=np.zeros(3, dtype=np.float64),
    )


def _observation(
    semantic_view: str,
    camera: Camera,
    boundary_names: tuple[str, ...],
) -> NasalViewObservation:
    height = width = 100
    confidence = np.add.outer(
        np.arange(height, dtype=np.float64) * 0.003,
        np.arange(width, dtype=np.float64) * 0.002,
    )
    confidence /= float(confidence.max())
    boundary = np.zeros((height, width), dtype=bool)
    boundary[20:80, 50] = True
    curves = {
        name: np.array([[50.0, 20.0], [50.0, 79.0]], dtype=np.float64)
        for name in boundary_names
    }
    fields = {
        name: np.zeros((height, width), dtype=np.float32)
        for name in boundary_names
    }
    return NasalViewObservation(
        semantic_view=semantic_view,
        camera=camera,
        original_size=(100, 100),
        mask_canvas_shape=(100, 100),
        work_size=(100, 100),
        roi_work_xyxy=(0.0, 0.0, 100.0, 100.0),
        boundaries_work=curves,
        boundary=boundary,
        distance_fields=fields,
        distance_field=np.zeros((height, width), dtype=np.float32),
        confidence=confidence,
        variant_boundaries_work={"base": curves},
        variant_boundaries={"base": boundary},
    )


def _observations(K: np.ndarray) -> NasalObservationBundle:
    return NasalObservationBundle(
        front=_observation(
            "front",
            _camera("camera2", "front", K),
            ("subject-left-alar", "nose-tip", "subject-right-alar"),
        ),
        subject_left=_observation(
            "subject-left",
            _camera("camera1", "left", K),
            ("nasal-profile",),
        ),
        subject_right=_observation(
            "subject-right",
            _camera("camera3", "right", K),
            ("nasal-profile",),
        ),
    )


def _views(K: np.ndarray, *, reverse_image_x: bool = False):
    rotations = (
        np.eye(3),
        _rotation_y(18.0),
        _rotation_y(-18.0),
    )
    if reverse_image_x:
        rotations = (
            np.diag([-1.0, -1.0, 1.0]),
            rotations[1],
            rotations[2],
        )
    return tuple(
        ProjectionView(name, K, rotation, np.zeros(3))
        for name, rotation in zip(VIEW_NAMES, rotations)
    )


def _problem(*, reverse_image_x: bool = False):
    K = np.array(
        [[40.0, 0.0, 50.0], [0.0, 42.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    vertices, faces = _mesh()
    observable, semantic = _bases(len(vertices))
    candidate = build_candidate_nasal_mesh(
        vertices,
        faces,
        observable,
        semantic,
        np.zeros(2),
        np.zeros(8),
    )
    return (
        candidate,
        semantic,
        _observations(K),
        _views(K, reverse_image_x=reverse_image_x),
    )


def _config() -> MultiviewNasalSamplingConfig:
    return MultiviewNasalSamplingConfig(
        front_samples_per_region=5,
        side_samples_per_view=12,
    )


def test_known_camera_projects_samples_in_declared_work_frame():
    candidate, semantic, observations, views = _problem()

    result = project_multiview_nasal_boundaries(
        candidate,
        semantic,
        observations,
        views,
        config=_config(),
    )

    for view_result, view in zip(result.per_view, views):
        camera_points = (
            view_result.model_points @ view.R_model_to_camera.T
            + view.t_model_to_camera
        )
        homogeneous = camera_points @ view.K.T
        expected = homogeneous[:, :2] / homogeneous[:, 2, None]
        np.testing.assert_allclose(view_result.pixel_xy, expected, atol=2e-6)
        assert np.all(view_result.pixel_xy[:, 0] < 100.0)
        assert np.all(view_result.pixel_xy[:, 1] < 100.0)
        assert np.all(view_result.depth > 0.0)
        provenance_points = np.sum(
            candidate.vertices[view_result.source_vertex_indices]
            * view_result.source_weights[:, :, None],
            axis=1,
        )
        np.testing.assert_allclose(
            view_result.model_points,
            provenance_points,
            atol=1e-12,
        )
        expected_confidence = (
            0.002 * view_result.pixel_xy[:, 0]
            + 0.003 * view_result.pixel_xy[:, 1]
        ) / (0.005 * 99.0)
        np.testing.assert_allclose(
            view_result.confidence,
            expected_confidence,
            atol=1e-12,
        )


def test_zero_and_nonzero_flame_and_semantic_coefficients_compose_exactly():
    vertices, faces = _mesh()
    observable, semantic = _bases(len(vertices))

    zero = build_candidate_nasal_mesh(
        vertices,
        faces,
        observable,
        semantic,
        np.zeros(2),
        np.zeros(8),
    )
    flame_coefficients = np.array([0.5, -0.25])
    semantic_coefficients = np.array([1.25, 0, 0, 0, -0.5, 0, 0, 0])
    changed = build_candidate_nasal_mesh(
        vertices,
        faces,
        observable,
        semantic,
        flame_coefficients,
        semantic_coefficients,
    )
    expected = (
        vertices
        + np.einsum(
            "vcr,r->vc",
            observable.vertex_basis,
            flame_coefficients,
        )
        + np.einsum(
            "mvc,m->vc",
            semantic.vectors,
            semantic_coefficients,
        )
    )

    np.testing.assert_array_equal(zero.vertices, vertices)
    np.testing.assert_allclose(changed.vertices, expected, atol=1e-15)
    np.testing.assert_array_equal(changed.faces, faces)
    assert changed.faces.dtype == faces.dtype


def test_inputs_remain_unchanged_and_candidate_api_cannot_change_fixed_state():
    candidate, semantic, observations, views = _problem()
    vertices, faces = _mesh()
    observable, semantic_for_build = _bases(len(vertices))
    arrays = [
        vertices,
        faces,
        observable.vertex_basis,
        semantic_for_build.vectors,
        semantic.support_mask,
        *semantic.region_masks.values(),
        observations.front.confidence,
    ]
    arrays.extend(
        array
        for view in views
        for array in (view.K, view.R_model_to_camera, view.t_model_to_camera)
    )
    snapshots = [array.copy() for array in arrays]

    rebuilt = build_candidate_nasal_mesh(
        vertices,
        faces,
        observable,
        semantic_for_build,
        np.ones(2),
        np.ones(8),
    )
    project_multiview_nasal_boundaries(
        candidate,
        semantic,
        observations,
        views,
        config=_config(),
    )

    for array, snapshot in zip(arrays, snapshots):
        np.testing.assert_array_equal(array, snapshot)
    np.testing.assert_array_equal(rebuilt.faces, faces)
    parameters = inspect.signature(build_candidate_nasal_mesh).parameters
    assert "cameras" not in parameters
    assert "expression" not in parameters
    with pytest.raises(TypeError):
        build_candidate_nasal_mesh(
            vertices,
            faces,
            observable,
            semantic_for_build,
            np.zeros(2),
            np.zeros(8),
            expression=np.ones(3),
        )


def test_occluded_back_side_wing_is_rejected_by_facing_and_sparse_depth():
    vertices = np.array(
        [
            [-1.0, -1.0, 3.0],
            [1.0, -1.0, 3.0],
            [1.0, 1.0, 3.0],
            [-1.0, 1.0, 3.0],
            [-0.3, -0.3, 5.0],
            [0.3, -0.3, 5.0],
            [0.0, 0.3, 5.0],
            [1.2, -0.4, 4.0],
            [1.8, -0.4, 4.0],
            [1.5, 0.4, 4.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=np.int64,
    )
    observable, semantic = _bases(len(vertices))
    semantic.support_mask[:] = False
    semantic.support_mask[4:] = True
    semantic.region_masks["subject_left_nose_wing"][:] = False
    semantic.region_masks["subject_left_nose_wing"][4:7] = True
    semantic.region_masks["nose_tip"][:] = False
    semantic.region_masks["nose_tip"][7:] = True
    semantic.region_masks["tip_alar_transition"][:] = False
    semantic.region_masks["tip_alar_transition"][4:] = True
    candidate = build_candidate_nasal_mesh(
        vertices,
        faces,
        observable,
        semantic,
        np.zeros(2),
        np.zeros(8),
    )
    K = np.array(
        [[40.0, 0.0, 50.0], [0.0, 40.0, 50.0], [0.0, 0.0, 1.0]]
    )
    views = tuple(
        ProjectionView(name, K, np.eye(3), np.zeros(3))
        for name in VIEW_NAMES
    )

    result = project_multiview_nasal_boundaries(
        candidate,
        semantic,
        _observations(K),
        views,
        config=_config(),
    )

    for samples in result.per_view[1:]:
        assert "nose_wing_subject_left" not in samples.source_labels
        assert "nose_tip" in samples.source_labels
        assert np.all(samples.visible)


def test_subject_labels_do_not_follow_image_x_ordering():
    normal = _problem(reverse_image_x=False)
    reversed_x = _problem(reverse_image_x=True)

    normal_result = project_multiview_nasal_boundaries(
        *normal,
        config=_config(),
    )
    reversed_result = project_multiview_nasal_boundaries(
        *reversed_x,
        config=_config(),
    )
    normal_front = normal_result.per_view[0]
    reversed_front = reversed_result.per_view[0]

    for samples in (normal_front, reversed_front):
        assert "nose_wing_subject_left" in samples.source_labels
        assert "nose_wing_subject_right" in samples.source_labels
    normal_left_x = np.mean(
        normal_front.pixel_xy[
            np.asarray(normal_front.source_labels) == "nose_wing_subject_left",
            0,
        ]
    )
    normal_right_x = np.mean(
        normal_front.pixel_xy[
            np.asarray(normal_front.source_labels) == "nose_wing_subject_right",
            0,
        ]
    )
    reversed_left_x = np.mean(
        reversed_front.pixel_xy[
            np.asarray(reversed_front.source_labels)
            == "nose_wing_subject_left",
            0,
        ]
    )
    reversed_right_x = np.mean(
        reversed_front.pixel_xy[
            np.asarray(reversed_front.source_labels)
            == "nose_wing_subject_right",
            0,
        ]
    )
    assert normal_left_x > normal_right_x
    assert reversed_left_x < reversed_right_x


def test_invalid_work_frame_intrinsics_and_nonfinite_inputs_fail_clearly():
    candidate, semantic, observations, views = _problem()
    wrong_K = views[0].K.copy()
    wrong_K[0, 2] = 75.0
    bad_views = (
        ProjectionView("front", wrong_K, np.eye(3), np.zeros(3)),
        views[1],
        views[2],
    )

    with pytest.raises(ValueError, match="work.*K|K.*work"):
        project_multiview_nasal_boundaries(
            candidate,
            semantic,
            observations,
            bad_views,
            config=_config(),
        )

    vertices, faces = _mesh()
    observable, build_semantic = _bases(len(vertices))
    vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_candidate_nasal_mesh(
            vertices,
            faces,
            observable,
            build_semantic,
            np.zeros(2),
            np.zeros(8),
        )
    vertices[0, 0] = 0.0
    faces[0, 0] = len(vertices)
    with pytest.raises(ValueError, match="face|topology|indices"):
        build_candidate_nasal_mesh(
            vertices,
            faces,
            observable,
            build_semantic,
            np.zeros(2),
            np.zeros(8),
        )
    with pytest.raises(ValueError, match="coefficients.*finite|finite.*coefficients"):
        build_candidate_nasal_mesh(
            _mesh()[0],
            _mesh()[1],
            observable,
            build_semantic,
            np.array([np.nan, 0.0]),
            np.zeros(8),
        )


def test_returned_arrays_are_deeply_immutable():
    candidate, semantic, observations, views = _problem()
    result = project_multiview_nasal_boundaries(
        candidate,
        semantic,
        observations,
        views,
        config=_config(),
    )
    arrays = [result.candidate_vertices, result.faces]
    for samples in result.per_view:
        arrays.extend(
            [
                samples.pixel_xy,
                samples.model_points,
                samples.source_vertex_indices,
                samples.source_weights,
                samples.confidence,
                samples.visible,
                samples.depth,
            ]
        )

    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE|writeable"):
            array.setflags(write=True)


def test_calls_are_deterministic_with_identical_sample_order_and_confidence():
    args = _problem()
    first = project_multiview_nasal_boundaries(*args, config=_config())
    second = project_multiview_nasal_boundaries(*args, config=_config())

    np.testing.assert_array_equal(first.candidate_vertices, second.candidate_vertices)
    for left, right in zip(first.per_view, second.per_view):
        assert left.semantic_view == right.semantic_view
        assert left.source_labels == right.source_labels
        assert left.boundary_names == right.boundary_names
        for name in (
            "pixel_xy",
            "model_points",
            "source_vertex_indices",
            "source_weights",
            "confidence",
            "visible",
            "depth",
        ):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
