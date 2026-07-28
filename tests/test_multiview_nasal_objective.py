from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

import src.geometry.multiview_nasal_objective as nasal_objective
from src.appearance.projective_sampling import project_points_strict
from src.cross_view_geometry import Camera, scale_intrinsics
from src.geometry.multiview_nasal_objective import (
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
    prepare_nasal_projection_context,
    project_multiview_nasal_boundaries,
    project_multiview_nasal_boundaries_prepared,
)
from src.geometry.nasal_observations import (
    NasalObservationBundle,
    NasalViewObservation,
)
from src.geometry.nasal_semantic_basis import build_nasal_semantic_basis
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
