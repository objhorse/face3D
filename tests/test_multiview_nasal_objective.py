from __future__ import annotations

import inspect
import time
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import least_squares

import src.geometry.multiview_nasal_objective as nasal_objective
from src.appearance.projective_sampling import project_points_strict
from src.cross_view_geometry import Camera, scale_intrinsics
from src.geometry.multiview_nasal_objective import (
    MultiviewNasalObjectiveConfig,
    MultiviewNasalObjectiveContext,
    MultiviewNasalResidualEvaluation,
    MultiviewNasalSoftProjection,
    MultiviewNasalProjection,
    MultiviewNasalSamplingConfig,
    PreparedNasalProjectionContext,
    ProjectedNasalSamples,
    _edge_adjacency,
    _external_region_boundary_edges,
    _front_external_edge_groups,
    _silhouette_edges,
    _sample_edges,
    _sparse_visibility,
    build_candidate_nasal_mesh,
    build_candidate_nasal_mesh_prepared,
    evaluate_multiview_nasal_objective,
    evaluate_multiview_nasal_objective_residuals,
    prepare_multiview_nasal_objective_context,
    prepare_nasal_projection_context,
    project_multiview_nasal_boundaries,
    project_multiview_nasal_boundaries_prepared,
)
from src.geometry.nasal_observations import (
    NasalObservationBundle,
    NasalViewObservation,
    _rasterize_curves,
    _unsigned_distance_field,
)
from src.geometry.nasal_semantic_basis import (
    NASAL_SEMANTIC_MODE_NAMES,
    build_nasal_semantic_basis,
)
from src.geometry.observable_flame_subspace import (
    ObservableFlameSubspaceConfig,
    ProjectionView,
    build_observable_flame_subspace,
)
from tests.test_nasal_semantic_basis import _synthetic_face as _real_basis_mesh


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
    vertices, faces = _grid_mesh(3)
    vertices[:, 0] -= 1.0
    vertices[:, 1] -= 1.0
    vertices[:, 2] = np.array(
        [4.0, 3.8, 4.0, 4.0, 3.5, 4.0, 4.0, 3.8, 4.0]
    )
    return vertices, faces.astype(np.int32)


def _grid_mesh(size: int = 7):
    xx, yy = np.meshgrid(
        np.arange(size, dtype=np.float64),
        np.arange(size, dtype=np.float64),
    )
    vertices = np.column_stack(
        (xx.ravel(), yy.ravel(), np.full(size * size, 4.0))
    )
    faces = []
    for row in range(size - 1):
        for column in range(size - 1):
            first = row * size + column
            right = first + 1
            below = first + size
            diagonal = below + 1
            faces.extend(((first, diagonal, right), (first, below, diagonal)))
    return vertices, np.asarray(faces, dtype=np.int64)


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
    if vertex_count == 9:
        bridge[[1, 4, 7]] = True
        tip[[1, 4, 7]] = True
        subject_left[[2, 5, 8]] = True
        subject_right[[0, 3, 6]] = True
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


def _camera(
    name: str,
    camera_view: str,
    K: np.ndarray,
    image_size: tuple[int, int] = (200, 200),
) -> Camera:
    return Camera(
        name=name,
        view=camera_view,
        image_size=image_size,
        K=K.copy(),
        dist=np.zeros(5, dtype=np.float64),
        R_rig_to_camera=np.eye(3, dtype=np.float64),
        t_rig_to_camera=np.zeros(3, dtype=np.float64),
    )


def _observation(
    semantic_view: str,
    camera: Camera,
    boundary_names: tuple[str, ...],
    *,
    coordinate_K: np.ndarray,
    work_size: tuple[int, int] = (100, 100),
    camera_metadata_overrides=None,
) -> NasalViewObservation:
    width, height = work_size
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
    camera_metadata = {
        "camera_name": camera.name,
        "camera_view": camera.view,
        "subject_relative_view": semantic_view,
        "image_size_wh": list(camera.image_size),
        "intrinsics": np.asarray(camera.K, dtype=float).tolist(),
        "distortion_coefficients": np.asarray(
            camera.dist,
            dtype=float,
        ).reshape(-1).tolist(),
        "rig_to_camera_rotation": np.asarray(
            camera.R_rig_to_camera,
            dtype=float,
        ).tolist(),
        "rig_to_camera_translation": np.asarray(
            camera.t_rig_to_camera,
            dtype=float,
        ).reshape(3).tolist(),
    }
    if camera_metadata_overrides:
        camera_metadata.update(camera_metadata_overrides)
    coordinate_metadata = {
        "source_pixel_frame": "undistorted_original_px",
        "observation_pixel_frame": "undistorted_work_px",
        "original_size_wh": list(camera.image_size),
        "work_size_wh": list(work_size),
        "intrinsics": np.asarray(coordinate_K, dtype=float).tolist(),
        "distortion_coefficients": np.zeros(5).tolist(),
        "conversion_source": "src.geometry.observation_coordinates",
    }
    return NasalViewObservation(
        semantic_view=semantic_view,
        camera=camera,
        original_size=tuple(camera.image_size),
        mask_canvas_shape=(height, width),
        work_size=work_size,
        roi_work_xyxy=(0.0, 0.0, float(width), float(height)),
        boundaries_work=curves,
        boundary=boundary,
        distance_fields=fields,
        distance_field=np.zeros((height, width), dtype=np.float32),
        confidence=confidence,
        variant_boundaries_work={"base": curves},
        variant_boundaries={"base": boundary},
        camera_metadata=camera_metadata,
        coordinate_metadata=coordinate_metadata,
    )


def _original_K(
    work_K: np.ndarray,
    original_size: tuple[int, int] = (200, 200),
    work_size: tuple[int, int] = (100, 100),
) -> np.ndarray:
    result = np.asarray(work_K, dtype=np.float64).copy()
    result[0] *= original_size[0] / float(work_size[0])
    result[1] *= original_size[1] / float(work_size[1])
    return result


def _observations(
    work_K: np.ndarray,
    *,
    camera_K: np.ndarray = None,
    coordinate_K: np.ndarray = None,
    front_names=("subject-left-alar", "subject-right-alar"),
    side_names=("nasal-profile",),
    camera_metadata_overrides=None,
) -> NasalObservationBundle:
    camera_intrinsics = (
        _original_K(work_K)
        if camera_K is None
        else np.asarray(camera_K, dtype=np.float64)
    )
    coordinate_intrinsics = (
        _original_K(work_K)
        if coordinate_K is None
        else np.asarray(coordinate_K, dtype=np.float64)
    )
    return NasalObservationBundle(
        front=_observation(
            "front",
            _camera("camera2", "front", camera_intrinsics),
            tuple(front_names),
            coordinate_K=coordinate_intrinsics,
            camera_metadata_overrides=camera_metadata_overrides,
        ),
        subject_left=_observation(
            "subject-left",
            _camera("camera1", "left", camera_intrinsics),
            tuple(side_names),
            coordinate_K=coordinate_intrinsics,
        ),
        subject_right=_observation(
            "subject-right",
            _camera("camera3", "right", camera_intrinsics),
            tuple(side_names),
            coordinate_K=coordinate_intrinsics,
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
        np.testing.assert_allclose(view_result.pixel_xy, expected, atol=6e-6)
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
    assert set(result.per_view[0].boundary_names) == {
        "subject-left-alar",
        "subject-right-alar",
    }
    assert set(result.per_view[0].source_labels) == {
        "subject_left_nose_wing",
        "subject_right_nose_wing",
    }
    assert all(
        set(samples.boundary_names).issubset(
            observations.by_view[samples.semantic_view].distance_fields
        )
        for samples in result.per_view
    )
    for samples in result.per_view:
        for label, edge in zip(
            samples.source_labels,
            samples.source_vertex_indices,
        ):
            assert label in result.semantic_region_masks
            assert np.all(result.semantic_region_masks[label][edge])


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


def test_adjacent_coplanar_faces_must_have_opposite_shared_edge_winding():
    vertices = np.array(
        [
            [-1.0, -1.0, 4.0],
            [1.0, -1.0, 4.0],
            [1.0, 1.0, 4.0],
            [-1.0, 1.0, 4.0],
        ]
    )
    consistent = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    inconsistent = np.array([[0, 2, 1], [0, 2, 3]], dtype=np.int64)
    observable, semantic = _bases(len(vertices))

    candidate = build_candidate_nasal_mesh(
        vertices,
        consistent,
        observable,
        semantic,
        np.zeros(2),
        np.zeros(8),
    )

    np.testing.assert_array_equal(candidate.faces, consistent)
    with pytest.raises(ValueError, match="winding|opposite directions"):
        build_candidate_nasal_mesh(
            vertices,
            inconsistent,
            observable,
            semantic,
            np.zeros(2),
            np.zeros(8),
        )


def test_real_semantic_and_observable_dataclasses_integrate_with_scaled_work_frame():
    vertices, faces, landmark_faces, barycentric = _real_basis_mesh()
    semantic = build_nasal_semantic_basis(
        vertices,
        faces,
        landmark_faces,
        barycentric,
        model_to_front_camera=np.eye(3),
    )
    work_K = np.array(
        [[46.0, 1.0, 50.0], [0.0, 44.0, 50.0], [0.0, 0.0, 1.0]]
    )
    base_rotation = np.diag([-1.0, 1.0, -1.0])
    views = (
        ProjectionView("front", work_K, base_rotation, np.array([0.0, 0.0, 4.0])),
        ProjectionView(
            "subject-left",
            work_K,
            base_rotation @ _rotation_y(72.0),
            np.array([0.0, 0.0, 4.0]),
        ),
        ProjectionView(
            "subject-right",
            work_K,
            base_rotation @ _rotation_y(-72.0),
            np.array([0.0, 0.0, 4.0]),
        ),
    )
    shape_basis = semantic.vectors[0][:, :, None]
    observable = build_observable_flame_subspace(
        vertices,
        shape_basis,
        semantic.support_mask,
        semantic.protected_mask,
        views,
        config=ObservableFlameSubspaceConfig(
            min_nasal_response_ratio=0.0,
            max_protected_to_nasal_energy_ratio=1.0,
            relative_nasal_energy_floor=0.0,
            relative_singular_value_threshold=1e-6,
            max_rank=1,
        ),
    )
    candidate = build_candidate_nasal_mesh(
        vertices,
        faces,
        observable,
        semantic,
        np.zeros(observable.retained_rank),
        np.zeros(8),
    )

    result = project_multiview_nasal_boundaries(
        candidate,
        semantic,
        _observations(work_K),
        views,
        config=MultiviewNasalSamplingConfig(
            front_samples_per_region=6,
            side_samples_per_view=8,
        ),
    )

    assert observable.vertex_basis.shape == (
        len(vertices),
        3,
        observable.retained_rank,
    )
    assert tuple(samples.semantic_view for samples in result.per_view) == VIEW_NAMES
    assert all(len(samples.pixel_xy) for samples in result.per_view)


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


def test_back_facing_open_boundary_is_not_a_silhouette_candidate():
    vertices = np.array(
        [
            [-0.5, -0.5, 4.0],
            [0.5, -0.5, 4.0],
            [0.0, 0.5, 4.0],
        ],
        dtype=np.float64,
    )
    back_faces = np.array([[0, 1, 2]], dtype=np.int64)
    front_faces = np.array([[0, 2, 1]], dtype=np.int64)

    back = _silhouette_edges(
        _edge_adjacency(back_faces),
        vertices,
        back_faces,
    )
    front = _silhouette_edges(
        _edge_adjacency(front_faces),
        vertices,
        front_faces,
    )

    assert back == ()
    assert set(front) == {(0, 1), (0, 2), (1, 2)}


def test_front_facing_open_boundary_is_rejected_only_when_depth_occluded():
    vertices = np.array(
        [
            [-1.0, -1.0, 3.0],
            [1.0, -1.0, 3.0],
            [1.0, 1.0, 3.0],
            [-1.0, 1.0, 3.0],
            [-0.3, -0.3, 5.0],
            [0.3, -0.3, 5.0],
            [0.0, 0.3, 5.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 6, 5]],
        dtype=np.int64,
    )
    K = np.array(
        [[40.0, 0.0, 50.0], [0.0, 40.0, 50.0], [0.0, 0.0, 1.0]]
    )
    projection = project_points_strict(
        vertices,
        K,
        np.eye(3),
        np.zeros(3),
    )
    silhouettes = _silhouette_edges(
        _edge_adjacency(faces),
        projection.camera_points,
        faces,
    )
    rear_edge = (4, 5)
    rear_point = np.mean(vertices[list(rear_edge)], axis=0, keepdims=True)
    rear_projection = project_points_strict(
        rear_point,
        K,
        np.eye(3),
        np.zeros(3),
    )

    visible = _sparse_visibility(
        rear_projection.pixel_xy,
        rear_projection.depth,
        projection.pixel_xy,
        projection.depth,
        faces,
        _config(),
    )

    assert rear_edge in silhouettes
    assert visible.tolist() == [False]


def test_external_front_contours_exclude_holes_and_protected_islands_with_overlap():
    size = 9
    vertices, faces = _grid_mesh(size)
    projected = vertices[:, :2] * 8.0 + 20.0
    adjacency = _edge_adjacency(faces)
    rows, columns = np.divmod(np.arange(len(vertices)), size)
    protected = (rows == 4) & (columns == 4)
    left = (columns <= 6) & ~protected
    right = (columns >= 2) & ~protected

    left_edges = _external_region_boundary_edges(
        adjacency,
        faces,
        left,
        protected,
        projected,
    )
    right_edges = _external_region_boundary_edges(
        adjacency,
        faces,
        right,
        protected,
        projected,
    )

    assert left_edges
    assert right_edges
    assert np.any(left & right)
    for edges, minimum_column, maximum_column in (
        (left_edges, 0, 6),
        (right_edges, 2, 8),
    ):
        used = np.unique(np.asarray(edges))
        used_rows, used_columns = np.divmod(used, size)
        assert np.all(
            (used_rows == 0)
            | (used_rows == size - 1)
            | (used_columns == minimum_column)
            | (used_columns == maximum_column)
        )
        assert not np.any(protected[used])


def test_front_classification_uses_one_composite_exterior_and_excludes_medial_edges():
    size = 9
    vertices, faces = _grid_mesh(size)
    projected = vertices[:, :2] * 8.0 + 20.0
    adjacency = _edge_adjacency(faces)
    rows, columns = np.divmod(np.arange(len(vertices)), size)
    protected = (rows == 4) & (columns == 4)
    regions = {
        "nose_bridge": (columns == 4) & ~protected,
        "nose_tip": (columns >= 3) & (columns <= 5) & ~protected,
        "subject_left_nose_wing": (columns >= 4) & ~protected,
        "subject_right_nose_wing": (columns <= 4) & ~protected,
        "tip_alar_transition": (columns >= 2) & (columns <= 6) & ~protected,
    }
    composite = (
        regions["subject_left_nose_wing"]
        | regions["subject_right_nose_wing"]
        | regions["nose_tip"]
        | regions["tip_alar_transition"]
    )
    external = set(
        _external_region_boundary_edges(
            adjacency,
            faces,
            composite,
            protected,
            projected,
        )
    )

    groups = _front_external_edge_groups(
        adjacency,
        faces,
        regions,
        protected,
        projected,
    )
    classified = {
        edge
        for _source, _target, edges in groups
        for edge in edges
    }
    medial_edge = tuple(
        sorted(
            (
                2 * size + 4,
                3 * size + 4,
            )
        )
    )

    assert {source for source, _target, _edges in groups} == {
        "subject_left_nose_wing",
        "subject_right_nose_wing",
    }
    assert classified
    assert classified.issubset(external)
    assert medial_edge not in external
    assert medial_edge not in classified


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
        assert "subject_left_nose_wing" in samples.source_labels
        assert "subject_right_nose_wing" in samples.source_labels
    normal_left_x = np.mean(
        normal_front.pixel_xy[
            np.asarray(normal_front.source_labels) == "subject_left_nose_wing",
            0,
        ]
    )
    normal_right_x = np.mean(
        normal_front.pixel_xy[
            np.asarray(normal_front.source_labels) == "subject_right_nose_wing",
            0,
        ]
    )
    reversed_left_x = np.mean(
        reversed_front.pixel_xy[
            np.asarray(reversed_front.source_labels)
            == "subject_left_nose_wing",
            0,
        ]
    )
    reversed_right_x = np.mean(
        reversed_front.pixel_xy[
            np.asarray(reversed_front.source_labels)
            == "subject_right_nose_wing",
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


def test_authoritative_coordinate_metadata_controls_nonidentity_work_K():
    candidate, semantic, _observations_unused, _views_unused = _problem()
    camera_work_K = np.array(
        [[36.0, 0.0, 48.0], [0.0, 38.0, 52.0], [0.0, 0.0, 1.0]]
    )
    authoritative_work_K = np.array(
        [[43.0, 1.5, 54.0], [0.0, 41.0, 47.0], [0.0, 0.0, 1.0]]
    )
    observations = _observations(
        authoritative_work_K,
        camera_K=_original_K(camera_work_K),
        coordinate_K=_original_K(authoritative_work_K),
    )
    authoritative_views = _views(authoritative_work_K)

    result = project_multiview_nasal_boundaries(
        candidate,
        semantic,
        observations,
        authoritative_views,
        config=_config(),
    )

    assert all(len(samples.pixel_xy) for samples in result.per_view)
    with pytest.raises(ValueError, match="coordinate.*work|work.*contract"):
        project_multiview_nasal_boundaries(
            candidate,
            semantic,
            observations,
            _views(camera_work_K),
            config=_config(),
        )


def test_camera_metadata_must_match_the_immutable_observation_camera():
    candidate, semantic, _observations_unused, views = _problem()
    observations = _observations(
        views[0].K,
        camera_metadata_overrides={"camera_name": "wrong-camera"},
    )

    with pytest.raises(ValueError, match="camera_metadata.*camera"):
        project_multiview_nasal_boundaries(
            candidate,
            semantic,
            observations,
            views,
            config=_config(),
        )


def _reachable_arrays(value, seen=None):
    visited = set() if seen is None else seen
    if id(value) in visited:
        return []
    visited.add(id(value))
    if isinstance(value, np.ndarray):
        return [value]
    if is_dataclass(value):
        return [
            array
            for field in fields(value)
            for array in _reachable_arrays(getattr(value, field.name), visited)
        ]
    if isinstance(value, Mapping):
        return [
            array
            for item in value.values()
            for array in _reachable_arrays(item, visited)
        ]
    if isinstance(value, (tuple, list)):
        return [
            array
            for item in value
            for array in _reachable_arrays(item, visited)
        ]
    return []


def test_prepared_context_performs_static_work_once_for_multiple_evaluations(
    monkeypatch,
):
    vertices, faces = _mesh()
    observable, semantic = _bases(len(vertices))
    K = np.array(
        [[40.0, 0.0, 50.0], [0.0, 42.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    observations = _observations(K)
    views = _views(K)
    calls = {
        "_validate_faces": 0,
        "_semantic_vectors": 0,
        "_validate_work_intrinsics": 0,
        "_build_edge_adjacency_arrays": 0,
        "_prepare_front_composite_loops": 0,
    }

    for name in calls:
        original = getattr(nasal_objective, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(nasal_objective, name, counted)

    prepared = prepare_nasal_projection_context(
        faces,
        semantic,
        observations,
        views,
        config=_config(),
    )
    static_calls = calls.copy()
    assert static_calls == {name: 1 for name in calls}

    results = []
    for semantic_scale in (0.0, 0.25):
        semantic_coefficients = np.zeros(8, dtype=np.float64)
        semantic_coefficients[0] = semantic_scale
        candidate = build_candidate_nasal_mesh_prepared(
            vertices,
            observable,
            prepared,
            np.zeros(2, dtype=np.float64),
            semantic_coefficients,
        )
        result = project_multiview_nasal_boundaries_prepared(
            candidate,
            prepared,
        )
        assert candidate.faces is prepared.faces
        assert result.faces is prepared.faces
        results.append(result)

    assert calls == static_calls
    assert not np.array_equal(
        results[0].candidate_vertices,
        results[1].candidate_vertices,
    )
    assert prepared.edge_vertices.ndim == 2
    assert prepared.edge_faces.ndim == 2
    assert prepared.front_loop_offsets.ndim == 1
    for array in _reachable_arrays(prepared):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE|writeable"):
            array.setflags(write=True)


def test_fractional_roi_matches_one_shot_and_prepared_near_sample_boundary():
    vertices, faces = _mesh()
    observable, semantic = _bases(len(vertices))
    K = np.array(
        [[40.0, 0.0, 50.0], [0.0, 42.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    observations = _observations(K)
    fractional_roi = (52.8, 0.25, 99.75, 99.5)
    observations = NasalObservationBundle(
        front=observations.front,
        subject_left=replace(
            observations.subject_left,
            roi_work_xyxy=fractional_roi,
        ),
        subject_right=observations.subject_right,
    )
    views = _views(K)
    candidate = build_candidate_nasal_mesh(
        vertices,
        faces,
        observable,
        semantic,
        np.zeros(2, dtype=np.float64),
        np.zeros(8, dtype=np.float64),
    )
    one_shot = project_multiview_nasal_boundaries(
        candidate,
        semantic,
        observations,
        views,
        config=_config(),
    )
    prepared = prepare_nasal_projection_context(
        faces,
        semantic,
        observations,
        views,
        config=_config(),
    )
    prepared_candidate = build_candidate_nasal_mesh_prepared(
        vertices,
        observable,
        prepared,
        np.zeros(2, dtype=np.float64),
        np.zeros(8, dtype=np.float64),
    )
    repeated = project_multiview_nasal_boundaries_prepared(
        prepared_candidate,
        prepared,
    )

    assert prepared.view_contracts[1].roi_work_xyxy == fractional_roi
    assert np.all(repeated.per_view[1].pixel_xy[:, 0] >= fractional_roi[0])
    assert not np.any(
        np.isclose(repeated.per_view[1].pixel_xy[:, 0], 52.77164841)
    )
    for expected, actual in zip(one_shot.per_view, repeated.per_view):
        assert expected.semantic_view == actual.semantic_view
        assert expected.source_labels == actual.source_labels
        assert expected.boundary_names == actual.boundary_names
        for name in (
            "pixel_xy",
            "model_points",
            "source_vertex_indices",
            "source_weights",
            "confidence",
            "visible",
            "depth",
        ):
            np.testing.assert_array_equal(
                getattr(expected, name),
                getattr(actual, name),
            )


def _reconstruct_prepared_context(prepared, **overrides):
    values = {
        "vertex_count": prepared.vertex_count,
        "faces": prepared.faces,
        "semantic_vectors": prepared.semantic_vectors,
        "support_mask": prepared.support_mask,
        "protected_mask": prepared.protected_mask,
        "region_masks": prepared.region_masks,
        "views": prepared.views,
        "view_contracts": prepared.view_contracts,
        "config": prepared.config,
    }
    values.update(overrides)
    return PreparedNasalProjectionContext(**values)


def test_direct_prepared_context_rejects_regions_outside_support():
    candidate, semantic, observations, views = _problem()
    prepared = prepare_nasal_projection_context(
        candidate.faces,
        semantic,
        observations,
        views,
        config=_config(),
    )
    support = prepared.support_mask.copy()
    support[0] = False

    with pytest.raises(ValueError, match="region.*outside.*support"):
        _reconstruct_prepared_context(prepared, support_mask=support)


def test_direct_prepared_context_rejects_regions_on_protected_vertices():
    candidate, semantic, observations, views = _problem()
    prepared = prepare_nasal_projection_context(
        candidate.faces,
        semantic,
        observations,
        views,
        config=_config(),
    )
    support = prepared.support_mask.copy()
    protected = prepared.protected_mask.copy()
    support[0] = False
    protected[0] = True

    with pytest.raises(ValueError, match="region.*protected"):
        _reconstruct_prepared_context(
            prepared,
            support_mask=support,
            protected_mask=protected,
        )


def test_direct_prepared_context_rejects_support_protected_overlap():
    candidate, semantic, observations, views = _problem()
    prepared = prepare_nasal_projection_context(
        candidate.faces,
        semantic,
        observations,
        views,
        config=_config(),
    )
    protected = prepared.protected_mask.copy()
    protected[0] = True

    with pytest.raises(ValueError, match="support.*protected.*disjoint"):
        _reconstruct_prepared_context(
            prepared,
            protected_mask=protected,
        )


def test_perspective_correct_slanted_edge_has_uniform_projected_samples():
    vertices = np.array(
        [
            [-1.0, 0.0, 2.0],
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 4.0],
        ],
        dtype=np.float64,
    )
    K = np.array(
        [[100.0, 0.0, 100.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projection = project_points_strict(
        vertices,
        K,
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
    count = 8

    points, indices, weights = _sample_edges(
        ((0, 1),),
        count,
        vertices,
        projection.pixel_xy,
        projection.depth,
    )
    sampled = project_points_strict(
        points,
        K,
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
    screen_fraction = (np.arange(count, dtype=np.float64) + 0.5) / count
    expected_pixels = (
        (1.0 - screen_fraction[:, None]) * projection.pixel_xy[0]
        + screen_fraction[:, None] * projection.pixel_xy[1]
    )
    expected_alpha = (
        screen_fraction * projection.depth[0]
        / (
            (1.0 - screen_fraction) * projection.depth[1]
            + screen_fraction * projection.depth[0]
        )
    )

    analytical_pixels = np.column_stack(
        (
            K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
            K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2],
        )
    )
    np.testing.assert_allclose(analytical_pixels, expected_pixels, atol=1e-12)
    np.testing.assert_allclose(
        sampled.pixel_xy,
        expected_pixels,
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.diff(analytical_pixels[:, 0]),
        np.full(count - 1, 7.5),
        atol=1e-12,
    )
    np.testing.assert_array_equal(indices, np.tile((0, 1), (count, 1)))
    np.testing.assert_allclose(weights[:, 1], expected_alpha, atol=1e-12)
    np.testing.assert_allclose(
        points,
        np.sum(vertices[indices] * weights[:, :, None], axis=1),
        atol=1e-12,
    )


def test_every_reachable_result_array_is_deeply_immutable_and_isolated():
    candidate, semantic, observations, views = _problem()
    result = project_multiview_nasal_boundaries(
        candidate,
        semantic,
        observations,
        views,
        config=_config(),
    )
    observation_arrays = _reachable_arrays(observations)
    result_arrays = _reachable_arrays(result)
    region_snapshot = {
        name: mask.copy()
        for name, mask in result.semantic_region_masks.items()
    }

    assert not hasattr(result, "observations")
    assert result_arrays
    assert not any(
        result_array is observation_array
        for result_array in result_arrays
        for observation_array in observation_arrays
    )
    for array in result_arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE|writeable"):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        result.semantic_region_masks["new"] = np.zeros(
            len(result.candidate_vertices),
            dtype=bool,
        )

    for mask in semantic.region_masks.values():
        mask[:] = False
    for name, snapshot in region_snapshot.items():
        np.testing.assert_array_equal(result.semantic_region_masks[name], snapshot)


def test_result_rejects_empty_targets_and_invalid_provenance():
    candidate, semantic, observations, views = _problem()
    result = project_multiview_nasal_boundaries(
        candidate,
        semantic,
        observations,
        views,
        config=_config(),
    )
    front = result.per_view[0]
    empty_values = {
        "pixel_xy": np.empty((0, 2)),
        "model_points": np.empty((0, 3)),
        "source_vertex_indices": np.empty((0, 2), dtype=np.int64),
        "source_weights": np.empty((0, 2)),
        "confidence": np.empty(0),
        "source_labels": (),
        "boundary_names": (),
        "visible": np.empty(0, dtype=bool),
        "depth": np.empty(0),
    }
    with pytest.raises(ValueError, match="empty|at least one"):
        replace(front, **empty_values)

    bad_indices = front.source_vertex_indices.copy()
    bad_indices[0, 0] = len(result.candidate_vertices)
    with pytest.raises(ValueError, match="source.*range|indices"):
        MultiviewNasalProjection(
            result.candidate_vertices,
            result.faces,
            (replace(front, source_vertex_indices=bad_indices),)
            + result.per_view[1:],
            result.observation_target_names,
            result.semantic_region_masks,
        )

    bad_points = front.model_points.copy()
    bad_points[0, 0] += 0.25
    with pytest.raises(ValueError, match="reconstruct|provenance"):
        MultiviewNasalProjection(
            result.candidate_vertices,
            result.faces,
            (replace(front, model_points=bad_points),) + result.per_view[1:],
            result.observation_target_names,
            result.semantic_region_masks,
        )

    bad_targets = tuple("not-an-observation-target" for _ in front.boundary_names)
    with pytest.raises(ValueError, match="target.*observation|boundary"):
        MultiviewNasalProjection(
            result.candidate_vertices,
            result.faces,
            (replace(front, boundary_names=bad_targets),) + result.per_view[1:],
            result.observation_target_names,
            result.semantic_region_masks,
        )

    bad_weights = front.source_weights.copy()
    bad_weights[0] = (-0.5, 1.5)
    with pytest.raises(ValueError, match="provenance|weight"):
        replace(front, source_weights=bad_weights)


def test_result_rejects_unknown_and_mislabeled_semantic_sources():
    candidate, semantic, observations, views = _problem()
    result = project_multiview_nasal_boundaries(
        candidate,
        semantic,
        observations,
        views,
        config=_config(),
    )
    front = result.per_view[0]
    unknown = ("not-a-semantic-region",) + front.source_labels[1:]
    with pytest.raises(ValueError, match="canonical.*source|semantic region"):
        replace(front, source_labels=unknown)

    labels = list(front.source_labels)
    left_index = labels.index("subject_left_nose_wing")
    labels[left_index] = "subject_right_nose_wing"
    mislabeled = replace(front, source_labels=tuple(labels))
    with pytest.raises(ValueError, match="source.*region|represented"):
        MultiviewNasalProjection(
            result.candidate_vertices,
            result.faces,
            (mislabeled,) + result.per_view[1:],
            result.observation_target_names,
            result.semantic_region_masks,
        )


def test_missing_mandatory_front_alar_group_fails_clearly():
    candidate, semantic, observations, views = _problem()
    semantic.region_masks["subject_left_nose_wing"][:] = False

    with pytest.raises(ValueError, match="mandatory.*subject-left-alar"):
        project_multiview_nasal_boundaries(
            candidate,
            semantic,
            observations,
            views,
            config=_config(),
        )


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


def _objective_semantic_basis():
    vertices, _faces = _mesh()
    _observable, semantic = _bases(len(vertices))
    vectors = np.zeros((8, len(vertices), 3), dtype=np.float64)
    subject_left = semantic.region_masks["subject_left_nose_wing"]
    subject_right = semantic.region_masks["subject_right_nose_wing"]
    tip = semantic.region_masks["nose_tip"]

    vectors[0, [2, 5, 8], 0] = (0.08, 0.14, 0.10)
    vectors[0, [0, 3, 6], 0] = (-0.08, -0.14, -0.10)
    vectors[1, subject_left, 0] = 0.10
    vectors[1, subject_right, 0] = 0.10
    vectors[2, subject_left | subject_right, 2] = -0.10
    vectors[3, subject_left, 2] = -0.10
    vectors[3, subject_right, 2] = 0.10
    vectors[4, tip, 2] = -0.12
    vectors[5, tip, 1] = 0.10
    vectors[6, tip, 2] = -0.10
    vectors[6, 4, 2] = -0.18
    vectors[6, 1, 1] = -0.12
    vectors[6, 7, 1] = 0.12
    vectors[7, subject_left | subject_right, 1] = 0.08
    values = dict(vars(semantic))
    values.update(vectors=vectors, names=NASAL_SEMANTIC_MODE_NAMES)
    return SimpleNamespace(**values)


def _semantic_with_vectors(semantic, vectors):
    values = dict(vars(semantic))
    values["vectors"] = np.asarray(vectors, dtype=np.float64)
    return SimpleNamespace(**values)


def _single_face_flip_semantic():
    semantic = _objective_semantic_basis()
    vectors = np.zeros_like(semantic.vectors)
    _vertices, faces = _mesh()
    vectors[0, faces[0, 2]] = (-1.0, 1.0, 0.0)
    return _semantic_with_vectors(semantic, vectors)


def _objective_observable(vertex_count: int):
    observable, _semantic = _bases(vertex_count)
    return SimpleNamespace(
        vertex_basis=observable.vertex_basis,
        coefficient_basis=np.eye(2, dtype=np.float64),
        retained_rank=2,
    )


def _constant_confidence_observations(
    observations: NasalObservationBundle,
    values,
) -> NasalObservationBundle:
    supplied = dict(values)

    def changed(observation):
        value = supplied.get(observation.semantic_view, 1.0)
        confidence = np.asarray(value, dtype=np.float64)
        if confidence.ndim == 0:
            confidence = np.full(
                observation.boundary.shape,
                float(confidence),
                dtype=np.float64,
            )
        return replace(observation, confidence=confidence)

    return NasalObservationBundle(
        front=changed(observations.front),
        subject_left=changed(observations.subject_left),
        subject_right=changed(observations.subject_right),
    )


def _target_observations_from_current_projection(
    observations: NasalObservationBundle,
    target_projection: MultiviewNasalProjection,
    *,
    constant_distance: float = None,
) -> NasalObservationBundle:
    changed = {}
    for target_samples in target_projection.per_view:
        observation = observations.by_view[target_samples.semantic_view]
        curves = {}
        fields = {}
        for boundary_name in sorted(set(target_samples.boundary_names)):
            selected = np.asarray(
                [
                    name == boundary_name
                    for name in target_samples.boundary_names
                ],
                dtype=bool,
            )
            points = target_samples.pixel_xy[selected]
            order = np.lexsort((points[:, 0], points[:, 1]))
            curves[boundary_name] = points[order]
            if constant_distance is None:
                named_boundary = _rasterize_curves(
                    {boundary_name: curves[boundary_name]},
                    observation.work_size,
                )
                fields[boundary_name] = _unsigned_distance_field(
                    named_boundary,
                    np.hypot(*observation.work_size),
                )
            else:
                fields[boundary_name] = np.full(
                    observation.boundary.shape,
                    float(constant_distance),
                    dtype=np.float64,
                )
        boundary = _rasterize_curves(curves, observation.work_size)
        aggregate = np.minimum.reduce(tuple(fields.values()))
        changed[target_samples.semantic_view] = replace(
            observation,
            boundaries_work=curves,
            boundary=boundary,
            distance_fields=fields,
            distance_field=aggregate,
            variant_boundaries_work={"base": curves},
            variant_boundaries={"base": boundary},
        )
    return NasalObservationBundle(
        front=changed["front"],
        subject_left=changed["subject-left"],
        subject_right=changed["subject-right"],
    )


def _observations_with_manual_curves(
    observations: NasalObservationBundle,
    curves_by_view,
) -> NasalObservationBundle:
    changed = {}
    for semantic_view, supplied_curves in dict(curves_by_view).items():
        observation = observations.by_view[semantic_view]
        curves = {
            str(name): np.asarray(curve, dtype=np.float64)
            for name, curve in dict(supplied_curves).items()
        }
        fields = {}
        for name, curve in curves.items():
            named_boundary = _rasterize_curves(
                {name: curve},
                observation.work_size,
            )
            fields[name] = _unsigned_distance_field(
                named_boundary,
                np.hypot(*observation.work_size),
            )
        boundary = _rasterize_curves(curves, observation.work_size)
        changed[semantic_view] = replace(
            observation,
            boundaries_work=curves,
            boundary=boundary,
            distance_fields=fields,
            distance_field=np.minimum.reduce(tuple(fields.values())),
            variant_boundaries_work={"base": curves},
            variant_boundaries={"base": boundary},
        )
    return NasalObservationBundle(
        front=changed.get("front", observations.front),
        subject_left=changed.get(
            "subject-left",
            observations.subject_left,
        ),
        subject_right=changed.get(
            "subject-right",
            observations.subject_right,
        ),
    )


def _named_curve_confidence_observations(
    observations: NasalObservationBundle,
    named_values,
) -> NasalObservationBundle:
    supplied = {
        view: dict(values)
        for view, values in dict(named_values).items()
    }

    def changed(observation):
        if observation.semantic_view not in supplied:
            return observation
        yy, xx = np.indices(observation.boundary.shape, dtype=np.float64)
        best_distance = np.full(observation.boundary.shape, np.inf)
        confidence = np.ones(observation.boundary.shape, dtype=np.float64)
        for name, value in supplied[observation.semantic_view].items():
            curve = np.asarray(
                observation.boundaries_work[name],
                dtype=np.float64,
            )
            dx = xx[:, :, None] - curve[None, None, :, 0]
            dy = yy[:, :, None] - curve[None, None, :, 1]
            distance = np.min(dx * dx + dy * dy, axis=2)
            replace_mask = distance < best_distance
            confidence[replace_mask] = float(value)
            best_distance[replace_mask] = distance[replace_mask]
        return replace(observation, confidence=confidence)

    return NasalObservationBundle(
        front=changed(observations.front),
        subject_left=changed(observations.subject_left),
        subject_right=changed(observations.subject_right),
    )


def _objective_problem(
    *,
    target_semantic=None,
    confidence=None,
    sampling_config=None,
    semantic=None,
    observable=None,
    flame_standard_deviations=None,
    constant_distance: float = None,
    named_confidence=None,
    return_observations: bool = False,
):
    vertices, faces = _mesh()
    semantic = _objective_semantic_basis() if semantic is None else semantic
    observable = (
        _objective_observable(len(vertices))
        if observable is None
        else observable
    )
    work_K = np.array(
        [[40.0, 0.0, 50.0], [0.0, 42.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    observations = _observations(work_K)
    sampling = (
        MultiviewNasalSamplingConfig(
            front_samples_per_region=10,
            side_samples_per_view=16,
        )
        if sampling_config is None
        else sampling_config
    )
    prepared = prepare_nasal_projection_context(
        faces,
        semantic,
        observations,
        _views(work_K),
        config=sampling,
    )
    semantic_values = (
        np.zeros(8, dtype=np.float64)
        if target_semantic is None
        else np.asarray(target_semantic, dtype=np.float64)
    )
    target_candidate = build_candidate_nasal_mesh_prepared(
        vertices,
        observable,
        prepared,
        np.zeros(observable.retained_rank),
        semantic_values,
    )
    target_projection = project_multiview_nasal_boundaries_prepared(
        target_candidate,
        prepared,
    )
    target_observations = _target_observations_from_current_projection(
        observations,
        target_projection,
        constant_distance=constant_distance,
    )
    target_observations = _constant_confidence_observations(
        target_observations,
        {} if confidence is None else confidence,
    )
    if named_confidence is not None:
        target_observations = _named_curve_confidence_observations(
            target_observations,
            named_confidence,
        )
    prepared_target = prepare_nasal_projection_context(
        faces,
        semantic,
        target_observations,
        _views(work_K),
        config=sampling,
    )
    context = prepare_multiview_nasal_objective_context(
        vertices,
        observable,
        semantic,
        target_observations,
        projection_context=prepared_target,
        flame_mode_standard_deviations=flame_standard_deviations,
    )
    if return_observations:
        return context, semantic_values, target_observations
    return context, semantic_values


def _manual_objective_context(
    curves_by_view,
    *,
    semantic=None,
    confidence=None,
    sampling_config=None,
):
    vertices, faces = _mesh()
    semantic = _objective_semantic_basis() if semantic is None else semantic
    observable = _objective_observable(len(vertices))
    work_K = np.array(
        [[40.0, 0.0, 50.0], [0.0, 42.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    observations = _observations_with_manual_curves(
        _observations(work_K),
        curves_by_view,
    )
    observations = _constant_confidence_observations(
        observations,
        {
            "front": 1.0,
            "subject-left": 1.0,
            "subject-right": 1.0,
        } if confidence is None else confidence,
    )
    sampling = (
        MultiviewNasalSamplingConfig(
            front_samples_per_region=10,
            side_samples_per_view=16,
        )
        if sampling_config is None
        else sampling_config
    )
    prepared = prepare_nasal_projection_context(
        faces,
        semantic,
        observations,
        _views(work_K),
        config=sampling,
    )
    return prepare_multiview_nasal_objective_context(
        vertices,
        observable,
        semantic,
        observations,
        projection_context=prepared,
    )


def _image_focused_config(**overrides):
    values = {
        "front_image_weight": 4.0,
        "side_image_weight": 4.0,
        "flame_prior_weight": 0.01,
        "semantic_prior_weight": 0.01,
        "smoothness_weight": 0.01,
        "symmetry_weight": 0.01,
        "robust_f_scale": 10.0,
    }
    values.update(overrides)
    return MultiviewNasalObjectiveConfig(**values)


def test_unified_objective_prefers_known_shared_alar_widening():
    target = np.zeros(8)
    target[0] = 0.8
    context, _ = _objective_problem(target_semantic=target)
    config = _image_focused_config()

    zero = evaluate_multiview_nasal_objective(
        np.zeros(context.parameter_count),
        context,
        config,
    )
    correct = evaluate_multiview_nasal_objective(
        np.r_[np.zeros(context.observable_rank), target],
        context,
        config,
    )
    wrong = evaluate_multiview_nasal_objective(
        np.r_[np.zeros(context.observable_rank), -target],
        context,
        config,
    )

    assert correct.total_robust_cost < zero.total_robust_cost
    assert correct.total_robust_cost < wrong.total_robust_cost


def test_tip_roundness_target_prefers_roundness_over_pure_tip_depth():
    target = np.zeros(8)
    target[6] = 0.9
    context, _, observations = _objective_problem(
        target_semantic=target,
        return_observations=True,
    )
    config = _image_focused_config()
    roundness = np.r_[np.zeros(context.observable_rank), target]
    rounded = evaluate_multiview_nasal_objective(roundness, context, config)
    depth_results = []
    for coefficient in np.linspace(-2.0, 2.0, 161):
        tip_depth = np.zeros(context.parameter_count)
        tip_depth[context.observable_rank + 4] = coefficient
        depth_results.append(
            evaluate_multiview_nasal_objective(
                tip_depth,
                context,
                config,
            )
        )
    best_depth = min(
        depth_results,
        key=lambda result: result.total_robust_cost,
    )

    for observation in observations.by_view.values():
        expected_boundary = _rasterize_curves(
            observation.boundaries_work,
            observation.work_size,
        )
        np.testing.assert_array_equal(
            observation.boundary,
            expected_boundary,
        )
        np.testing.assert_array_equal(
            observation.variant_boundaries["base"],
            expected_boundary,
        )
        assert set(observation.variant_boundaries_work["base"]) == set(
            observation.boundaries_work
        )
        for name, curve in observation.boundaries_work.items():
            named_boundary = _rasterize_curves(
                {name: curve},
                observation.work_size,
            )
            expected_distance = _unsigned_distance_field(
                named_boundary,
                np.hypot(*observation.work_size),
            )
            np.testing.assert_allclose(
                observation.distance_fields[name],
                expected_distance,
                atol=1e-6,
                rtol=0.0,
            )
    assert rounded.total_robust_cost < best_depth.total_robust_cost


def test_zero_confidence_image_rows_and_finite_difference_gradient_are_exactly_zero():
    target = np.zeros(8)
    target[0] = 0.7
    context, _ = _objective_problem(
        target_semantic=target,
        confidence={
            "front": 0.0,
            "subject-left": 0.0,
            "subject-right": 0.0,
        },
    )
    theta = np.zeros(context.parameter_count)
    base = evaluate_multiview_nasal_objective(theta, context)
    perturbed_theta = theta.copy()
    perturbed_theta[context.observable_rank] = 1e-6
    perturbed = evaluate_multiview_nasal_objective(
        perturbed_theta,
        context,
    )

    for name in context.image_term_names:
        np.testing.assert_array_equal(
            base.term_residuals[name],
            np.zeros_like(base.term_residuals[name]),
        )
        np.testing.assert_array_equal(
            perturbed.term_residuals[name],
            np.zeros_like(perturbed.term_residuals[name]),
        )
        np.testing.assert_array_equal(
            (
                perturbed.term_residuals[name]
                - base.term_residuals[name]
            )
            / 1e-6,
            np.zeros_like(base.term_residuals[name]),
        )


def test_per_view_confidence_normalization_is_duplicate_invariant():
    per_view_terms = {
        "front": (
            (np.array([1.0, 2.0]), np.array([0.25, 1.0])),
            (np.array([3.0, 4.0]), np.array([0.5, 0.75])),
        ),
        "subject-left": (
            (np.array([1.5, 2.5, 3.5]), np.array([0.2, 0.4, 0.8])),
        ),
        "subject-right": (
            (np.array([0.5, 1.0, 2.0]), np.array([0.3, 0.6, 0.9])),
        ),
    }
    for terms in per_view_terms.values():
        confidence_sum = sum(float(np.sum(confidence)) for _, confidence in terms)
        original = [
            nasal_objective._normalized_image_residual(
                distance,
                confidence,
                confidence_sum,
                1.7,
            )
            for distance, confidence in terms
        ]
        duplicated = [
            nasal_objective._normalized_image_residual(
                np.repeat(distance, 2),
                np.repeat(confidence, 2),
                2.0 * confidence_sum,
                1.7,
            )
            for distance, confidence in terms
        ]
        original_cost = sum(0.5 * np.dot(value, value) for value in original)
        duplicated_cost = sum(
            0.5 * np.dot(value, value) for value in duplicated
        )
        assert duplicated_cost == pytest.approx(
            original_cost,
            rel=1e-15,
            abs=1e-15,
        )


def test_front_alar_terms_share_one_view_denominator_with_unequal_evidence():
    context, _ = _objective_problem(named_confidence={
        "front": {
            "subject-left-alar": 0.25,
            "subject-right-alar": 1.0,
        }
    })
    result = evaluate_multiview_nasal_objective(
        np.zeros(context.parameter_count),
        context,
    )
    left_name = "front_subject_left_alar"
    right_name = "front_subject_right_alar"
    confidence_ratio = (
        result.effective_confidence_sums[left_name]
        / result.effective_confidence_sums[right_name]
    )

    distances = np.array([1.0, 2.0, 3.0])
    left_confidence = np.full(3, 0.25)
    right_confidence = np.ones(3)
    view_sum = float(
        np.sum(left_confidence) + np.sum(right_confidence)
    )
    left = nasal_objective._normalized_image_residual(
        distances,
        left_confidence,
        view_sum,
        1.0,
    )
    right = nasal_objective._normalized_image_residual(
        distances,
        right_confidence,
        view_sum,
        1.0,
    )
    assert np.dot(left, left) / np.dot(right, right) == pytest.approx(0.25)
    assert result.per_view_effective_confidence_sums["front"] == (
        pytest.approx(
            result.effective_confidence_sums[left_name]
            + result.effective_confidence_sums[right_name]
        )
    )
    assert result.report_data["per_view_effective_confidence_sums"] == (
        result.per_view_effective_confidence_sums
    )


def test_symmetry_evidence_factor_is_continuous_and_suppresses_single_side_asymmetry():
    target = np.zeros(8)
    target[3] = 0.9
    factors = []
    best_coefficients = []
    grid = np.linspace(0.0, target[3], 10)
    for right_confidence in (0.0, 0.25, 0.5, 1.0):
        context, _ = _objective_problem(
            target_semantic=target,
            confidence={
                "front": 1.0,
                "subject-left": 1.0,
                "subject-right": right_confidence,
            },
        )
        config = _image_focused_config(
            semantic_prior_weight=0.02,
            symmetry_weight=1.5,
            symmetry_evidence_floor=0.1,
            symmetry_evidence_ceiling=1.0,
        )
        results = []
        for coefficient in grid:
            theta = np.zeros(context.parameter_count)
            theta[context.observable_rank + 3] = coefficient
            results.append(
                evaluate_multiview_nasal_objective(theta, context, config)
            )
        factors.append(
            results[0].symmetry_evidence_factors[
                "alar_depth_asymmetry"
            ]
        )
        best_coefficients.append(grid[np.argmin(
            [result.total_robust_cost for result in results]
        )])

    assert factors[0] > factors[1] > factors[2] > factors[3]
    assert best_coefficients[0] < best_coefficients[-1]
    assert factors[0] <= 1.0
    assert factors[-1] >= 0.1


def test_symmetric_and_asymmetric_targets_prefer_supported_parameters():
    config = _image_focused_config(symmetry_weight=0.05)
    for mode_index in (0, 1):
        target = np.zeros(8)
        target[mode_index] = 0.75
        context, _ = _objective_problem(target_semantic=target)
        correct_theta = np.r_[np.zeros(context.observable_rank), target]
        competing = np.zeros(8)
        competing[1 - mode_index] = target[mode_index]
        competing_theta = np.r_[
            np.zeros(context.observable_rank),
            competing,
        ]

        correct = evaluate_multiview_nasal_objective(
            correct_theta,
            context,
            config,
        )
        wrong = evaluate_multiview_nasal_objective(
            competing_theta,
            context,
            config,
        )

        assert correct.total_robust_cost < wrong.total_robust_cost


def test_smoothness_is_zero_continuous_and_penalizes_rough_local_basis():
    base_semantic = _objective_semantic_basis()
    smooth_vectors = np.array(base_semantic.vectors, copy=True)
    rough_vectors = np.array(base_semantic.vectors, copy=True)
    smooth_vectors[0] = 0.0
    smooth_vectors[0, :, 0] = 0.12
    rough_vectors[0] = 0.0
    rough_vectors[0, 4, 0] = 0.12
    smooth_context, _ = _objective_problem(
        semantic=_semantic_with_vectors(base_semantic, smooth_vectors),
    )
    rough_context, _ = _objective_problem(
        semantic=_semantic_with_vectors(base_semantic, rough_vectors),
    )
    zero = evaluate_multiview_nasal_objective(
        np.zeros(smooth_context.parameter_count),
        smooth_context,
    )
    small_theta = np.zeros(smooth_context.parameter_count)
    small_theta[smooth_context.observable_rank] = 1e-6
    small = evaluate_multiview_nasal_objective(
        small_theta,
        smooth_context,
    )
    large_theta = small_theta * 2.0
    large = evaluate_multiview_nasal_objective(
        large_theta,
        smooth_context,
    )
    rough_theta = np.zeros(rough_context.parameter_count)
    rough_theta[rough_context.observable_rank] = 1.0
    smooth_theta = np.zeros(smooth_context.parameter_count)
    smooth_theta[smooth_context.observable_rank] = 1.0

    np.testing.assert_array_equal(
        zero.term_residuals["surface_smoothness"],
        np.zeros_like(zero.term_residuals["surface_smoothness"]),
    )
    assert np.linalg.norm(
        large.term_residuals["surface_smoothness"]
    ) == pytest.approx(
        2.0 * np.linalg.norm(
            small.term_residuals["surface_smoothness"]
        ),
        rel=1e-10,
        abs=1e-16,
    )
    assert np.linalg.norm(
        evaluate_multiview_nasal_objective(
            rough_theta,
            rough_context,
        ).term_residuals["surface_smoothness"]
    ) > np.linalg.norm(
        evaluate_multiview_nasal_objective(
            smooth_theta,
            smooth_context,
        ).term_residuals["surface_smoothness"]
    )


def test_surface_orientation_barrier_is_near_zero_at_baseline_and_strong_on_flip():
    context, _ = _objective_problem(semantic=_single_face_flip_semantic())
    baseline = evaluate_multiview_nasal_objective(
        np.zeros(context.parameter_count),
        context,
    )
    flipped_theta = np.zeros(context.parameter_count)
    flipped_theta[context.observable_rank] = 1.0
    flipped = evaluate_multiview_nasal_objective(
        flipped_theta,
        context,
    )

    assert np.linalg.norm(
        baseline.term_residuals["surface_orientation_barrier"]
    ) < 1e-4
    assert baseline.report_data["surface_orientation_barrier"][
        "min_signed_area_ratio"
    ] == pytest.approx(1.0, abs=1e-14)
    baseline_orientation = baseline.report_data[
        "surface_orientation_barrier"
    ]
    assert baseline_orientation[
        "global_feasibility_signed_area_ratio"
    ] == baseline_orientation["actual_min_signed_area_ratio"]
    assert flipped.report_data["surface_orientation_barrier"][
        "min_signed_area_ratio"
    ] < 0.0
    flipped_orientation = flipped.report_data[
        "surface_orientation_barrier"
    ]
    assert flipped_orientation[
        "global_feasibility_signed_area_ratio"
    ] == flipped_orientation["actual_min_signed_area_ratio"]
    assert flipped_orientation[
        "diagnostic_normalized_log_mean_exp_signed_area_ratio"
    ] >= flipped_orientation["actual_min_signed_area_ratio"]
    assert np.linalg.norm(
        flipped.term_residuals["surface_orientation_barrier"]
    ) > 50.0
    assert len(
        flipped.term_residuals["surface_orientation_barrier"]
    ) == context.orientation_active_face_count + 1


def test_orientation_context_selects_basis_or_support_faces_and_is_immutable():
    context, _ = _objective_problem(semantic=_single_face_flip_semantic())
    faces = context.projection_context.faces
    expected_vertices = (
        np.any(context.observable_vertex_basis != 0.0, axis=(1, 2))
        | np.any(
            context.projection_context.semantic_vectors != 0.0,
            axis=(0, 2),
        )
        | context.projection_context.support_mask
    )
    expected_faces = np.flatnonzero(
        np.any(expected_vertices[faces], axis=1)
    )

    np.testing.assert_array_equal(
        context.orientation_face_indices,
        expected_faces,
    )
    np.testing.assert_array_equal(
        context.orientation_face_vertices,
        faces[expected_faces],
    )
    assert context.orientation_active_face_count == len(expected_faces)
    for array in (
        context.orientation_face_indices,
        context.orientation_face_vertices,
        context.orientation_reference_cross,
        context.orientation_reference_inverse_squared_norm,
    ):
        assert not array.flags.writeable
        assert _is_bytes_backed(array)


def test_orientation_active_faces_and_ratios_are_uniform_scale_invariant():
    context, _ = _objective_problem()
    baseline = context.baseline_vertices
    displacement = (
        0.25 * context.projection_context.semantic_vectors[0]
    )
    expected_indices = None
    expected_ratios = None

    for scale in (1e-6, 1.0, 1e6):
        (
            face_indices,
            face_vertices,
            reference_cross,
            inverse_squared_norm,
        ) = nasal_objective._orientation_reference_data(
            baseline * scale,
            context.projection_context,
            context.observable_vertex_basis * scale,
        )
        ratio_context = SimpleNamespace(
            orientation_face_vertices=face_vertices,
            orientation_reference_cross=reference_cross,
            orientation_reference_inverse_squared_norm=(
                inverse_squared_norm
            ),
            orientation_active_face_count=len(face_indices),
        )
        ratios = nasal_objective._orientation_signed_area_ratios(
            (baseline + displacement) * scale,
            ratio_context,
        )
        if expected_indices is None:
            expected_indices = face_indices
            expected_ratios = ratios
        else:
            np.testing.assert_array_equal(
                face_indices,
                expected_indices,
            )
            np.testing.assert_allclose(
                ratios,
                expected_ratios,
                rtol=1e-13,
                atol=1e-13,
            )


def test_orientation_context_rejects_degenerate_movable_baseline_faces():
    semantic = _single_face_flip_semantic()
    observable = _objective_observable(len(_mesh()[0]))
    context, _, observations = _objective_problem(
        semantic=semantic,
        observable=observable,
        return_observations=True,
    )
    face_index = int(context.orientation_face_indices[0])
    face = context.projection_context.faces[face_index]
    degenerate = np.array(context.baseline_vertices, copy=True)
    degenerate[face[2]] = degenerate[face[0]]

    with pytest.raises(
        ValueError,
        match="degenerate.*potentially deformable.*total=",
    ) as caught:
        prepare_multiview_nasal_objective_context(
            degenerate,
            observable,
            semantic,
            observations,
            projection_context=context.projection_context,
        )

    assert f"indices=[{face_index}" in str(caught.value)


def test_orientation_soft_minimum_is_normalized_for_equal_face_ratios():
    temperature = 0.01
    expected = 0.37

    for count in (1, 1000):
        result = nasal_objective._soft_minimum_orientation_ratio(
            np.full(count, expected, dtype=np.float64),
            temperature,
        )
        assert result == pytest.approx(expected, abs=1e-14)

    varied = np.array([0.11, 0.43, 0.82], dtype=np.float64)
    reference = nasal_objective._soft_minimum_orientation_ratio(
        varied,
        temperature,
    )
    duplicated = nasal_objective._soft_minimum_orientation_ratio(
        np.tile(varied, 1000),
        temperature,
    )
    assert duplicated == pytest.approx(reference, abs=1e-14)


def test_orientation_global_barrier_uses_exact_min_without_face_dilution():
    config = MultiviewNasalObjectiveConfig()
    sparse_bad_face = np.r_[0.19, np.ones(1000)]
    dense_bad_face = np.r_[0.19, np.ones(100_000)]

    sparse_residuals, sparse_minimum = (
        nasal_objective._surface_orientation_barrier_residuals(
            sparse_bad_face,
            config,
        )
    )
    dense_residuals, dense_minimum = (
        nasal_objective._surface_orientation_barrier_residuals(
            dense_bad_face,
            config,
        )
    )
    sparse_diagnostic = (
        nasal_objective._soft_minimum_orientation_ratio(
            sparse_bad_face,
            config.orientation_barrier_softmin_temperature,
        )
    )
    dense_diagnostic = (
        nasal_objective._soft_minimum_orientation_ratio(
            dense_bad_face,
            config.orientation_barrier_softmin_temperature,
        )
    )
    expected_global = (
        config.orientation_barrier_weight
        * nasal_objective._stable_softplus(
            np.asarray(
                [
                    (
                        config.orientation_barrier_margin
                        - 0.19
                    )
                    / config.orientation_barrier_scale
                ]
            )
        )[0]
    )

    assert sparse_minimum == pytest.approx(0.19)
    assert dense_minimum == pytest.approx(0.19)
    assert sparse_minimum == min(sparse_bad_face)
    assert dense_minimum == min(dense_bad_face)
    assert sparse_residuals[-1] == pytest.approx(expected_global)
    assert dense_residuals[-1] == pytest.approx(expected_global)
    assert sparse_residuals[-1] == pytest.approx(dense_residuals[-1])
    assert sparse_diagnostic > sparse_minimum
    assert dense_diagnostic > dense_minimum


def test_prior_scales_parameter_order_and_zero_theta_reproduce_baseline():
    context, _ = _objective_problem(
        flame_standard_deviations=(2.0, 4.0),
        confidence={
            "front": 0.0,
            "subject-left": 0.0,
            "subject-right": 0.0,
        },
    )
    semantic_stddevs = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    config = MultiviewNasalObjectiveConfig(
        flame_prior_weight=2.0,
        semantic_prior_weight=3.0,
        semantic_prior_standard_deviations=semantic_stddevs,
        smoothness_weight=0.0,
        symmetry_weight=0.0,
    )
    zero = evaluate_multiview_nasal_objective(
        np.zeros(context.parameter_count),
        context,
        config,
    )
    theta = np.r_[np.array([2.0, 4.0]), np.arange(1.0, 9.0)]
    result = evaluate_multiview_nasal_objective(theta, context, config)

    np.testing.assert_array_equal(
        zero.candidate.vertices,
        context.baseline_vertices,
    )
    np.testing.assert_array_equal(
        result.term_residuals["flame_prior"],
        np.array([2.0, 2.0]),
    )
    np.testing.assert_array_equal(
        result.term_residuals["semantic_prior"],
        np.full(8, 3.0),
    )
    assert result.parameter_ordering == (
        "observable_flame_0",
        "observable_flame_1",
    ) + NASAL_SEMANTIC_MODE_NAMES


def test_objective_residual_layout_is_fixed_and_calls_are_deterministic():
    context, _ = _objective_problem()
    theta = np.zeros(context.parameter_count)
    first = evaluate_multiview_nasal_objective(theta, context)
    second = evaluate_multiview_nasal_objective(theta, context)
    changed_theta = theta.copy()
    changed_theta[context.observable_rank + 6] = 1e-7
    changed = evaluate_multiview_nasal_objective(changed_theta, context)

    assert first.term_slices == second.term_slices == changed.term_slices
    assert len(first.residuals) == len(second.residuals) == len(changed.residuals)
    np.testing.assert_array_equal(first.residuals, second.residuals)
    assert tuple(first.term_slices) == (
        context.image_term_names
        + (
            "projection_depth_barrier",
            "flame_prior",
            "semantic_prior",
            "surface_smoothness",
            "surface_orientation_barrier",
            "weak_symmetry",
        )
    )


def test_soft_semantic_slots_change_continuously_with_candidate_geometry():
    context, _ = _objective_problem()
    zero_theta = np.zeros(context.parameter_count)
    widened_theta = zero_theta.copy()
    widened_theta[context.observable_rank] = 0.8
    rounded_theta = zero_theta.copy()
    rounded_theta[context.observable_rank + 6] = 0.9
    baseline = evaluate_multiview_nasal_objective(zero_theta, context)
    widened = evaluate_multiview_nasal_objective(widened_theta, context)
    rounded = evaluate_multiview_nasal_objective(rounded_theta, context)

    assert baseline.term_slices == widened.term_slices == rounded.term_slices
    assert len(baseline.residuals) == len(widened.residuals) == len(
        rounded.residuals
    )
    assert isinstance(baseline.projection, MultiviewNasalSoftProjection)
    assert set(baseline.report_data["soft_profile_summaries"]) == set(
        context.image_term_names
    )
    for slots in baseline.projection.per_term:
        assert slots.front_depth.shape == (len(slots.pixel_xy),)
        assert slots.soft_visibility.shape == slots.soft_weights.shape
        assert np.all(
            (slots.soft_visibility >= 0.0)
            & (slots.soft_visibility <= 1.0)
        )
    before_left = baseline.projection.by_term["front_subject_left_alar"]
    after_left = widened.projection.by_term["front_subject_left_alar"]
    np.testing.assert_array_equal(
        before_left.support_vertex_indices,
        after_left.support_vertex_indices,
    )
    assert not np.array_equal(before_left.soft_weights, after_left.soft_weights)
    assert not np.array_equal(before_left.pixel_xy, after_left.pixel_xy)
    for name in (
        "subject_left_nasal_profile",
        "subject_right_nasal_profile",
    ):
        before = baseline.projection.by_term[name]
        after = rounded.projection.by_term[name]
        np.testing.assert_array_equal(
            before.support_vertex_indices,
            after.support_vertex_indices,
        )
        assert not np.array_equal(before.soft_weights, after.soft_weights)
        assert not np.array_equal(before.pixel_xy, after.pixel_xy)
        assert not hasattr(before, "visible")
        assert not hasattr(before, "source_vertex_indices")


def test_out_of_frame_slots_keep_observation_weight_and_increase_image_cost():
    semantic = _objective_semantic_basis()
    vectors = np.array(semantic.vectors, copy=True)
    vectors[5] = 0.0
    vectors[5, :, 1] = 20.0
    context, _ = _objective_problem(
        semantic=_semantic_with_vectors(semantic, vectors),
    )
    zero_theta = np.zeros(context.parameter_count)
    moved_theta = zero_theta.copy()
    moved_theta[context.observable_rank + 5] = 1.0
    baseline = evaluate_multiview_nasal_objective(zero_theta, context)
    moved = evaluate_multiview_nasal_objective(moved_theta, context)
    baseline_image_cost = sum(
        baseline.raw_costs[name] for name in context.image_term_names
    )
    moved_image_cost = sum(
        moved.raw_costs[name] for name in context.image_term_names
    )

    assert moved_image_cost > baseline_image_cost
    assert moved_image_cost > 0.0
    assert moved.effective_confidence_sums == (
        baseline.effective_confidence_sums
    )
    assert moved.symmetry_evidence_factors == (
        baseline.symmetry_evidence_factors
    )
    assert any(
        np.any(
            (samples.pixel_xy[:, 1] < 0.0)
            | (samples.pixel_xy[:, 1] > 99.0)
        )
        for samples in moved.projection.per_term
    )
    assert all(not hasattr(samples, "visible") for samples in moved.projection.per_term)


def test_depth_barrier_is_finite_and_continuous_across_min_depth():
    semantic = _objective_semantic_basis()
    vectors = np.array(semantic.vectors, copy=True)
    vectors[5] = 0.0
    vectors[5, :, 2] = -2.0
    context, _ = _objective_problem(
        semantic=_semantic_with_vectors(semantic, vectors),
    )
    crossing = float(
        np.min(context.baseline_vertices[:, 2])
        - context.projection_context.config.min_depth
    ) / 2.0
    coefficients = crossing + np.linspace(-1e-5, 1e-5, 9)
    barriers = []
    residuals = []
    for coefficient in coefficients:
        theta = np.zeros(context.parameter_count)
        theta[context.observable_rank + 5] = coefficient
        result = evaluate_multiview_nasal_objective_residuals(theta, context)
        barriers.append(
            np.linalg.norm(
                result.term_residuals["projection_depth_barrier"]
            )
        )
        residuals.append(result.residuals)
    assert np.isfinite(np.asarray(residuals)).all()
    assert np.isfinite(barriers).all()
    assert barriers[-1] > barriers[0]
    slopes = np.diff(np.asarray(residuals), axis=0) / np.diff(
        coefficients
    )[:, None]
    assert np.isfinite(slopes).all()

    baseline = evaluate_multiview_nasal_objective(
        np.zeros(context.parameter_count),
        context,
    )
    adversarial_theta = np.zeros(context.parameter_count)
    adversarial_theta[context.observable_rank + 5] = 2.0
    adversarial = evaluate_multiview_nasal_objective(
        adversarial_theta,
        context,
    )
    assert np.any(
        adversarial.projection.by_term[
            "front_subject_left_alar"
        ].depth <= 0.0
    )
    assert adversarial.total_robust_cost > baseline.total_robust_cost
    assert adversarial.robust_costs["projection_depth_barrier"] > (
        baseline.total_robust_cost
    )
    np.testing.assert_array_equal(
        context.parameter_lower_bounds,
        np.full(context.parameter_count, -3.0),
    )
    np.testing.assert_array_equal(
        context.parameter_upper_bounds,
        np.full(context.parameter_count, 3.0),
    )
    for bounded_probe in (
        context.parameter_lower_bounds,
        context.parameter_upper_bounds,
    ):
        assert np.isfinite(
            evaluate_multiview_nasal_objective_residuals(
                bounded_probe,
                context,
            ).residuals
        ).all()
    assert baseline.report_data["parameter_bounds"] == {
        "lower": tuple(context.parameter_lower_bounds),
        "upper": tuple(context.parameter_upper_bounds),
        "solver_requirement": "C3 must enforce these fixed bounds",
    }


def test_residual_only_matches_full_and_avoids_c1_face_scans(monkeypatch):
    context, _ = _objective_problem()
    theta = np.zeros(context.parameter_count)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("C2 residual evaluation scanned C1 topology")

    monkeypatch.setattr(nasal_objective, "_sparse_visibility", forbidden)
    monkeypatch.setattr(
        nasal_objective,
        "_silhouette_edges_from_arrays",
        forbidden,
    )
    monkeypatch.setattr(
        nasal_objective,
        "_current_prepared_edge_groups",
        forbidden,
    )
    minimal = evaluate_multiview_nasal_objective_residuals(theta, context)
    full = evaluate_multiview_nasal_objective(theta, context)

    assert isinstance(minimal, MultiviewNasalResidualEvaluation)
    np.testing.assert_array_equal(minimal.residuals, full.residuals)
    assert minimal.term_slices == full.term_slices
    assert not hasattr(minimal, "candidate")
    assert not hasattr(minimal, "projection")
    assert not hasattr(minimal, "report_data")


def test_soft_profile_continuity_through_prior_edge_transition_value():
    context, _ = _objective_problem()
    center = 1.8883812427520774
    coefficients = np.linspace(center - 1e-6, center + 1e-6, 41)
    residuals = []
    for coefficient in coefficients:
        theta = np.zeros(context.parameter_count)
        theta[context.observable_rank + 6] = coefficient
        residuals.append(
            evaluate_multiview_nasal_objective_residuals(
                theta,
                context,
            ).residuals
        )
    residuals = np.asarray(residuals)
    steps = np.diff(coefficients)
    jumps = np.max(np.abs(np.diff(residuals, axis=0)), axis=1)
    slopes = jumps / steps

    assert np.max(jumps) < 1e-7
    assert np.max(slopes) < 2.0
    assert np.isfinite(slopes).all()


def test_directional_soft_profile_suppresses_deep_interior_vertex():
    slot_y = np.array([40.0, 50.0, 60.0])
    exterior = np.column_stack((np.full(3, 500.0), slot_y))
    interior = np.column_stack((np.full(3, -500.0), slot_y))
    projected = np.vstack((exterior, interior))
    depth = np.r_[np.ones(3), np.full(3, 10.0)]
    config = MultiviewNasalObjectiveConfig()
    baseline, _depth, _weights, _front_depth, _visibility = (
        nasal_objective._soft_profile_slots(
        projected,
        depth,
        slot_y,
        1,
        config,
        1e-4,
        work_size=(100, 100),
    )
    )
    moved_interior = projected.copy()
    moved_interior[3:, 0] += 10.0
    interior_result, _depth, _weights, _front_depth, _visibility = (
        nasal_objective._soft_profile_slots(
        moved_interior,
        depth,
        slot_y,
        1,
        config,
        1e-4,
        work_size=(100, 100),
    )
    )
    moved_exterior = projected.copy()
    moved_exterior[:3, 0] += 10.0
    exterior_result, _depth, _weights, _front_depth, _visibility = (
        nasal_objective._soft_profile_slots(
        moved_exterior,
        depth,
        slot_y,
        1,
        config,
        1e-4,
        work_size=(100, 100),
    )
    )

    assert np.max(np.abs(interior_result - baseline)) < 1e-12
    assert np.max(np.abs(exterior_result - baseline)) > 9.9


def test_soft_z_buffer_suppresses_rear_point_in_same_image_footprint():
    config = MultiviewNasalObjectiveConfig()
    slot_y = np.array([50.0])
    projected = np.array([[50.0, 50.0], [50.5, 50.0]])
    depth = np.array([1.0, 100.0])
    baseline = nasal_objective._soft_profile_slots(
        projected,
        depth,
        slot_y,
        1,
        config,
        1e-4,
        work_size=(100, 100),
    )
    pixels, _slot_depth, weights, front_depth, visibility = baseline
    moved_rear = projected.copy()
    moved_rear[1, 0] += 0.25
    rear_pixels = nasal_objective._soft_profile_slots(
        moved_rear,
        depth,
        slot_y,
        1,
        config,
        1e-4,
        work_size=(100, 100),
    )[0]
    moved_foreground = projected.copy()
    moved_foreground[0, 0] += 0.25
    foreground_pixels = nasal_objective._soft_profile_slots(
        moved_foreground,
        depth,
        slot_y,
        1,
        config,
        1e-4,
        work_size=(100, 100),
    )[0]

    assert weights[0, 1] / weights[0, 0] < 1e-8
    assert visibility[0, 1] / visibility[0, 0] < 1e-8
    assert 1.0 <= front_depth[0] < 1.1
    assert np.max(np.abs(rear_pixels - pixels)) < 1e-8
    assert np.max(np.abs(foreground_pixels - pixels)) > 0.24


@pytest.mark.parametrize("rear_x", [50.5, 500.0, 1000.0, 5000.0])
def test_soft_z_buffer_suppresses_deep_rear_point_at_extreme_x(rear_x):
    config = MultiviewNasalObjectiveConfig()
    projected = np.array([[50.0, 50.0], [rear_x, 50.0]])
    depth = np.array([1.0, 100.0])

    _pixels, _depth, weights, _front, _visibility = (
        nasal_objective._soft_profile_slots(
            projected,
            depth,
            np.array([50.0]),
            1,
            config,
            1e-4,
            work_size=(100, 100),
        )
    )

    assert weights[0, 1] / weights[0, 0] < 1e-8


@pytest.mark.parametrize("direction_sign", [-1, 1])
def test_direction_score_cannot_capture_locality_anchor_at_work_width(
    direction_sign,
):
    config = MultiviewNasalObjectiveConfig()
    rear_x = 5000.0 if direction_sign > 0 else -4361.0

    pixels, _depth, weights, front_depth, _visibility = (
        nasal_objective._soft_profile_slots(
            np.array([[320.0, 240.0], [rear_x, 240.0]]),
            np.array([1.0, 20.0]),
            np.array([240.0]),
            direction_sign,
            config,
            1e-4,
            work_size=(640, 480),
        )
    )

    assert weights[0, 1] / weights[0, 0] < 1e-8
    assert abs(pixels[0, 0] - 320.0) < 1e-6
    assert 1.0 <= front_depth[0] < 1.1


def test_soft_profile_keeps_near_depth_exterior_point_observable():
    config = MultiviewNasalObjectiveConfig()
    projected = np.array([[50.0, 50.0], [80.0, 50.0]])

    pixels, _depth, weights, _front, _visibility = (
        nasal_objective._soft_profile_slots(
            projected,
            np.array([1.0, 1.0]),
            np.array([50.0]),
            1,
            config,
            1e-4,
            work_size=(100, 100),
        )
    )

    assert weights[0, 1] > weights[0, 0]
    assert 65.0 < pixels[0, 0] < 80.0


def test_bounded_directional_score_is_continuous_through_saturation():
    config = MultiviewNasalObjectiveConfig()
    moving_x = np.linspace(450.0, 550.0, 1001)
    slot_x = []
    for x_value in moving_x:
        pixels = nasal_objective._soft_profile_slots(
            np.array([[50.0, 50.0], [x_value, 50.0]]),
            np.array([1.0, 1.0]),
            np.array([50.0]),
            1,
            config,
            1e-4,
            work_size=(100, 100),
        )[0]
        slot_x.append(float(pixels[0, 0]))
    slot_x = np.asarray(slot_x)
    first_derivative = np.diff(slot_x) / np.diff(moving_x)
    derivative_change = np.diff(first_derivative)

    assert np.isfinite(slot_x).all()
    assert np.isfinite(first_derivative).all()
    assert np.max(np.abs(np.diff(slot_x))) < 0.11
    assert np.max(np.abs(derivative_change)) < 1e-6


def test_soft_z_buffer_depth_sweep_has_no_residual_jump():
    config = MultiviewNasalObjectiveConfig()
    projected = np.array([[50.0, 50.0], [50.5, 50.0]])
    slot_y = np.array([50.0])
    rear_depths = np.linspace(0.5, 2.0, 601)
    residuals = []
    rear_weights = []
    for rear_depth in rear_depths:
        pixels, _depth, weights, _front, _visibility = (
            nasal_objective._soft_profile_slots(
                projected,
                np.array([1.0, rear_depth]),
                slot_y,
                1,
                config,
                1e-4,
                work_size=(100, 100),
            )
        )
        residuals.append(float(pixels[0, 0] - 49.0))
        rear_weights.append(float(weights[0, 1]))
    residuals = np.asarray(residuals)
    derivatives = np.diff(residuals) / np.diff(rear_depths)

    assert np.isfinite(residuals).all()
    assert np.isfinite(derivatives).all()
    assert np.max(np.abs(np.diff(residuals))) < 0.02
    assert np.max(np.abs(np.diff(derivatives))) < 0.5
    assert rear_weights[0] > rear_weights[-1]


def test_global_depth_feasibility_is_not_diluted_by_large_support():
    config = MultiviewNasalObjectiveConfig()
    depths = np.full(160_000, 4.0)
    depths[0] = 0.0
    soft_minimum = nasal_objective._soft_minimum_depth(
        depths,
        config.depth_barrier_softmin_temperature,
    )
    global_residual = (
        config.depth_barrier_weight
        * nasal_objective._stable_softplus(
            np.array(
                [
                    (
                        1e-4 - soft_minimum
                    ) / config.depth_barrier_scale
                ]
            )
        )[0]
    )

    assert soft_minimum < 0.002
    assert global_residual > 60.0


def test_exact_polyline_distance_remains_analytic_outside_image():
    semantic = _objective_semantic_basis()
    vectors = np.array(semantic.vectors, copy=True)
    vectors[5] = 0.0
    vectors[5, :, 1] = 0.1
    curves = {
        "front": {
            "subject-left-alar": np.array([[60.0, 20.0], [60.0, 80.0]]),
            "subject-right-alar": np.array([[40.0, 20.0], [40.0, 80.0]]),
        },
        "subject-left": {
            "nasal-profile": np.array([[65.0, 20.0], [65.0, 80.0]]),
        },
        "subject-right": {
            "nasal-profile": np.array([[35.0, 20.0], [35.0, 80.0]]),
        },
    }
    context = _manual_objective_context(
        curves,
        semantic=_semantic_with_vectors(semantic, vectors),
        confidence={
            "front": 1.0,
            "subject-left": 0.0,
            "subject-right": 0.0,
        },
    )
    baseline = evaluate_multiview_nasal_objective(
        np.zeros(context.parameter_count),
        context,
    )
    base_y = baseline.projection.by_term[
        "front_subject_left_alar"
    ].pixel_xy[-1, 1]
    assert base_y < 99.0

    def last_slot_y(coefficient):
        theta = np.zeros(context.parameter_count)
        theta[context.observable_rank + 5] = coefficient
        return evaluate_multiview_nasal_objective(
            theta,
            context,
        ).projection.by_term["front_subject_left_alar"].pixel_xy[-1, 1]

    lower, upper = 0.0, 100.0
    assert last_slot_y(upper) > 99.0
    for _iteration in range(60):
        midpoint = 0.5 * (lower + upper)
        if last_slot_y(midpoint) < 99.0:
            lower = midpoint
        else:
            upper = midpoint
    crossing = 0.5 * (lower + upper)
    coefficients = crossing + np.array([-1e-4, 0.0, 1e-4])
    recovered_distances = []
    for coefficient in coefficients:
        theta = np.zeros(context.parameter_count)
        theta[context.observable_rank + 5] = coefficient
        result = evaluate_multiview_nasal_objective(theta, context)
        term = context.image_terms[0]
        pixels = result.projection.by_term[term.name].pixel_xy
        expected = np.hypot(
            pixels[:, 0] - 60.0,
            np.maximum.reduce(
                (
                    20.0 - pixels[:, 1],
                    pixels[:, 1] - 80.0,
                    np.zeros(len(pixels)),
                )
            ),
        )
        recovered = (
            result.term_residuals[term.name]
            * np.sqrt(
                result.per_view_effective_confidence_sums["front"]
            )
            / np.sqrt(term.confidence)
        )
        np.testing.assert_allclose(recovered, expected, atol=1e-10, rtol=0.0)
        recovered_distances.append(recovered[-1])
        assert pixels[-1, 1] > 98.9
    derivatives = np.diff(recovered_distances) / np.diff(coefficients)
    assert derivatives[0] == pytest.approx(
        derivatives[1],
        rel=1e-7,
        abs=1e-7,
    )


def test_scipy_least_squares_converges_to_manual_shared_widening():
    semantic = _objective_semantic_basis()
    vectors = np.array(semantic.vectors, copy=True)
    vectors[0] = 0.0
    vectors[0, semantic.region_masks["subject_left_nose_wing"], 0] = 0.1
    vectors[0, semantic.region_masks["subject_right_nose_wing"], 0] = -0.1
    curves = {
        "front": {
            "subject-left-alar": np.array([[61.0, 20.0], [61.0, 80.0]]),
            "subject-right-alar": np.array([[39.0, 20.0], [39.0, 80.0]]),
        },
    }
    context = _manual_objective_context(
        curves,
        semantic=_semantic_with_vectors(semantic, vectors),
        confidence={
            "front": 1.0,
            "subject-left": 0.0,
            "subject-right": 0.0,
        },
    )
    config = MultiviewNasalObjectiveConfig(
        side_image_weight=0.0,
        flame_prior_weight=0.0,
        semantic_prior_weight=0.0,
        smoothness_weight=0.0,
        symmetry_weight=0.0,
        robust_f_scale=1.0,
    )

    def residual(coefficient):
        theta = np.zeros(context.parameter_count)
        theta[context.observable_rank] = coefficient[0]
        return evaluate_multiview_nasal_objective_residuals(
            theta,
            context,
            config,
        ).residuals.copy()

    solved = least_squares(
        residual,
        np.array([0.0]),
        bounds=(-2.0, 2.0),
        loss="soft_l1",
        f_scale=config.robust_f_scale,
    )

    assert solved.success
    assert solved.x[0] == pytest.approx(1.0, abs=1e-6)
    theta = np.zeros(context.parameter_count)
    theta[context.observable_rank] = solved.x[0]
    evaluated = evaluate_multiview_nasal_objective(
        theta,
        context,
        config,
    )
    non_orientation = np.concatenate(
        [
            residual
            for name, residual in evaluated.term_residuals.items()
            if name != "surface_orientation_barrier"
        ]
    )
    assert np.linalg.norm(non_orientation) < 1e-8
    assert np.linalg.norm(
        evaluated.term_residuals["surface_orientation_barrier"]
    ) < 1e-6


def test_residual_only_realistic_mesh_avoids_face_scale_visibility_cost():
    size = 285
    vertices, faces = _grid_mesh(size)
    center = 0.5 * float(size - 1)
    vertices[:, :2] = (vertices[:, :2] - center) / center
    vertices[:, 2] = 4.0
    columns = np.tile(np.arange(size), size)
    rows = np.repeat(np.arange(size), size)
    vertex_count = len(vertices)
    support = (
        (np.abs(columns - center) <= 6)
        & (np.abs(rows - center) <= 24)
    )
    subject_left = support & (columns >= size // 2)
    subject_right = support & (columns < size // 2)
    tip = (
        (np.abs(columns - center) <= size * 0.08)
        & (np.abs(rows - center) <= size * 0.2)
        & support
    )
    semantic = SimpleNamespace(
        vectors=np.zeros((8, vertex_count, 3), dtype=np.float64),
        names=NASAL_SEMANTIC_MODE_NAMES,
        support_mask=support,
        protected_mask=np.zeros(vertex_count, dtype=bool),
        mode_support_masks=np.broadcast_to(
            support,
            (8, vertex_count),
        ).copy(),
        region_masks={
            "nose_bridge": (
                (np.abs(columns - center) <= 3)
                & support
            ),
            "nose_tip": tip,
            "subject_left_nose_wing": subject_left,
            "subject_right_nose_wing": subject_right,
            "tip_alar_transition": support,
        },
    )
    observable = SimpleNamespace(
        vertex_basis=np.zeros((vertex_count, 3, 0), dtype=np.float64),
        coefficient_basis=np.zeros((0, 0), dtype=np.float64),
        retained_rank=0,
    )
    work_K = np.array(
        [[40.0, 0.0, 50.0], [0.0, 42.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    observations = _constant_confidence_observations(
        _observations(work_K),
        {"front": 1.0, "subject-left": 1.0, "subject-right": 1.0},
    )
    context = prepare_multiview_nasal_objective_context(
        vertices,
        observable,
        semantic,
        observations,
        faces=faces.astype(np.int32),
        views=_views(work_K),
        sampling_config=MultiviewNasalSamplingConfig(
            front_samples_per_region=10,
            side_samples_per_view=16,
        ),
    )
    theta = np.zeros(context.parameter_count)
    evaluate_multiview_nasal_objective_residuals(theta, context)
    durations = []
    for _repeat in range(5):
        started = time.perf_counter()
        evaluate_multiview_nasal_objective_residuals(theta, context)
        durations.append(time.perf_counter() - started)

    assert len(faces) > 159_000
    assert context.orientation_active_face_count < 2_000
    assert np.median(durations) < 0.45


def _is_bytes_backed(array: np.ndarray) -> bool:
    value = array
    seen = set()
    while isinstance(value, np.ndarray) and id(value) not in seen:
        seen.add(id(value))
        value = value.base
    return isinstance(value, bytes)


def test_objective_validation_reports_and_arrays_are_deeply_immutable():
    context, _ = _objective_problem()
    result = evaluate_multiview_nasal_objective(
        np.zeros(context.parameter_count),
        context,
    )
    minimal = evaluate_multiview_nasal_objective_residuals(
        np.zeros(context.parameter_count),
        context,
    )

    with pytest.raises(ValueError, match="theta.*shape|length"):
        evaluate_multiview_nasal_objective(
            np.zeros(context.parameter_count + 1),
            context,
        )
    invalid_theta = np.zeros(context.parameter_count)
    invalid_theta[0] = np.nan
    with pytest.raises(ValueError, match="theta.*finite"):
        evaluate_multiview_nasal_objective(invalid_theta, context)
    with pytest.raises(ValueError, match="front_image_weight.*finite"):
        MultiviewNasalObjectiveConfig(front_image_weight=np.nan)
    with pytest.raises(ValueError, match="profile_vertical_sigma_px"):
        MultiviewNasalObjectiveConfig(profile_vertical_sigma_px=0.0)
    with pytest.raises(
        ValueError,
        match="profile_visibility_spatial_sigma_px",
    ):
        MultiviewNasalObjectiveConfig(
            profile_visibility_spatial_sigma_px=0.0
        )
    with pytest.raises(ValueError, match="depth_barrier_weight"):
        MultiviewNasalObjectiveConfig(depth_barrier_weight=0.0)
    with pytest.raises(ValueError, match="depth_barrier_scale"):
        MultiviewNasalObjectiveConfig(depth_barrier_scale=1.0)
    with pytest.raises(ValueError, match="orientation_barrier_weight"):
        MultiviewNasalObjectiveConfig(orientation_barrier_weight=0.0)
    with pytest.raises(ValueError, match="orientation_barrier_margin"):
        MultiviewNasalObjectiveConfig(orientation_barrier_margin=0.1)
    with pytest.raises(ValueError, match="robust_f_scale"):
        MultiviewNasalObjectiveConfig(robust_f_scale=0.1)
    with pytest.raises(ValueError, match="semantic_coefficient_bound"):
        MultiviewNasalObjectiveConfig(semantic_coefficient_bound=4.0)
    with pytest.raises(ValueError, match="baseline.*finite"):
        replace(
            context,
            baseline_vertices=np.full_like(
                context.baseline_vertices,
                np.nan,
            ),
        )
    with pytest.raises(ValueError, match="depth support"):
        replace(
            context,
            depth_support_vertex_indices=(
                context.depth_support_vertex_indices[:-1]
            ),
        )
    with pytest.raises(ValueError, match="parameter bounds"):
        replace(
            context,
            parameter_upper_bounds=np.full(
                context.parameter_count,
                4.0,
            ),
        )
    with pytest.raises(ValueError, match="orientation.*faces"):
        replace(
            context,
            orientation_face_indices=context.orientation_face_indices[:-1],
        )
    with pytest.raises(ValueError, match="orientation.*reference"):
        replace(
            context,
            orientation_reference_cross=np.zeros_like(
                context.orientation_reference_cross
            ),
        )
    with pytest.raises(ValueError, match="context"):
        evaluate_multiview_nasal_objective(
            np.zeros(context.parameter_count),
            object(),
        )

    assert isinstance(context, MultiviewNasalObjectiveContext)
    for value in (context, result, minimal):
        arrays = _reachable_arrays(value)
        assert arrays
        for array in arrays:
            assert not array.flags.writeable
            assert _is_bytes_backed(array)
            with pytest.raises(ValueError, match="WRITEABLE|writeable"):
                array.setflags(write=True)
    with pytest.raises(TypeError):
        result.term_residuals["new"] = np.zeros(1)
    with pytest.raises(TypeError):
        result.per_view_sample_counts["front"] = 0
    with pytest.raises(TypeError):
        result.report_data["new"] = 1

    assert result.report_data["robust_loss"] == "soft_l1"
    assert result.report_data["parameter_ordering"] == result.parameter_ordering
    assert set(result.report_data["raw_costs"]) == set(result.term_slices)
    assert set(result.report_data["robust_costs"]) == set(result.term_slices)
    assert result.report_data["effective_observation_counts"]
    assert result.report_data["effective_confidence_sums"]
    assert result.report_data["per_view_effective_observation_counts"]
    assert result.report_data["per_view_effective_confidence_sums"]
    assert result.report_data["per_view_sample_counts"]
    assert result.report_data["symmetry_evidence_factors"]
    orientation_report = result.report_data[
        "surface_orientation_barrier"
    ]
    assert orientation_report["active_face_count"] == (
        context.orientation_active_face_count
    )
    assert set(orientation_report["signed_area_ratio_quantiles"]) == {
        "p01",
        "p05",
        "p50",
        "p95",
        "p99",
    }
    assert (
        orientation_report[
            "baseline_local_triangle_quality_threshold"
        ]
        == pytest.approx(
            nasal_objective._ORIENTATION_BASELINE_LOCAL_QUALITY_THRESHOLD
        )
    )


def test_real_semantic_and_observable_results_prepare_and_evaluate_objective():
    vertices, faces, landmark_faces, barycentric = _real_basis_mesh()
    semantic = build_nasal_semantic_basis(
        vertices,
        faces,
        landmark_faces,
        barycentric,
        model_to_front_camera=np.eye(3),
    )
    work_K = np.array(
        [[46.0, 1.0, 50.0], [0.0, 44.0, 50.0], [0.0, 0.0, 1.0]]
    )
    base_rotation = np.diag([-1.0, 1.0, -1.0])
    views = (
        ProjectionView(
            "front",
            work_K,
            base_rotation,
            np.array([0.0, 0.0, 4.0]),
        ),
        ProjectionView(
            "subject-left",
            work_K,
            base_rotation @ _rotation_y(72.0),
            np.array([0.0, 0.0, 4.0]),
        ),
        ProjectionView(
            "subject-right",
            work_K,
            base_rotation @ _rotation_y(-72.0),
            np.array([0.0, 0.0, 4.0]),
        ),
    )
    observable = build_observable_flame_subspace(
        vertices,
        semantic.vectors[0][:, :, None],
        semantic.support_mask,
        semantic.protected_mask,
        views,
        config=ObservableFlameSubspaceConfig(
            min_nasal_response_ratio=0.0,
            max_protected_to_nasal_energy_ratio=1.0,
            relative_nasal_energy_floor=0.0,
            relative_singular_value_threshold=1e-6,
            max_rank=1,
        ),
    )
    observations = _constant_confidence_observations(
        _observations(work_K),
        {"front": 1.0, "subject-left": 1.0, "subject-right": 1.0},
    )
    context = prepare_multiview_nasal_objective_context(
        vertices,
        observable,
        semantic,
        observations,
        faces=faces,
        views=views,
        sampling_config=MultiviewNasalSamplingConfig(
            front_samples_per_region=6,
            side_samples_per_view=8,
        ),
    )
    result = evaluate_multiview_nasal_objective(
        np.zeros(context.parameter_count),
        context,
    )

    assert context.observable_rank == observable.retained_rank
    assert result.residuals.ndim == 1
    np.testing.assert_array_equal(result.candidate.vertices, vertices)
