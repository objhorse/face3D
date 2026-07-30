from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.geometry.observable_flame_subspace import ProjectionView
from src.geometry.rig_consistent_pose import (
    RigConsistentPoseContext,
    compose_rig_consistent_views,
    fit_rig_consistent_pose,
    project_68_landmarks,
)
from tests.test_nasal_semantic_basis import _synthetic_face


def _views() -> tuple[ProjectionView, ...]:
    intrinsic = np.asarray(
        ((640.0, 0.0, 320.0), (0.0, 640.0, 240.0), (0.0, 0.0, 1.0))
    )
    angle = np.deg2rad(28.0)
    left = Rotation.from_rotvec((0.0, angle, 0.0)).as_matrix()
    right = left.T
    translation = np.asarray((0.0, 0.0, 4.0))
    return (
        ProjectionView("front", intrinsic, np.eye(3), translation),
        ProjectionView("subject-left", intrinsic, left, left @ translation),
        ProjectionView("subject-right", intrinsic, right, right @ translation),
    )


def test_shared_pose_fit_reduces_three_view_error_and_preserves_rig():
    vertices, _faces, triangles, barycentric = _synthetic_face()
    initial_views = _views()
    target_front_rotation = (
        Rotation.from_rotvec(np.deg2rad((1.2, -1.8, 0.7))).as_matrix()
        @ initial_views[0].R_model_to_camera
    )
    target_front_translation = (
        initial_views[0].t_model_to_camera + np.asarray((0.01, -0.008, 0.04))
    )
    target_views = compose_rig_consistent_views(
        initial_views,
        target_front_rotation,
        target_front_translation,
    )
    observations = {
        view.name: project_68_landmarks(
            vertices,
            triangles,
            barycentric,
            view,
        )
        for view in target_views
    }
    context = RigConsistentPoseContext(
        vertices=vertices,
        landmark_triangles=triangles,
        landmark_barycentric=barycentric,
        initial_views=initial_views,
        observed_landmarks_68=observations,
    )

    result = fit_rig_consistent_pose(context)

    assert result.success
    assert result.final_cost < result.initial_cost * 0.02
    front_initial = initial_views[0]
    front_solved = result.views[0]
    for before, after in zip(initial_views[1:], result.views[1:]):
        relative_before = (
            before.R_model_to_camera @ front_initial.R_model_to_camera.T
        )
        relative_after = (
            after.R_model_to_camera @ front_solved.R_model_to_camera.T
        )
        np.testing.assert_allclose(relative_after, relative_before, atol=1e-10)
        offset_before = (
            before.t_model_to_camera
            - relative_before @ front_initial.t_model_to_camera
        )
        offset_after = (
            after.t_model_to_camera
            - relative_after @ front_solved.t_model_to_camera
        )
        np.testing.assert_allclose(offset_after, offset_before, atol=1e-10)


def test_pose_fit_does_not_mutate_geometry_or_observations():
    vertices, _faces, triangles, barycentric = _synthetic_face()
    views = _views()
    observations = {
        view.name: project_68_landmarks(
            vertices,
            triangles,
            barycentric,
            view,
        )
        for view in views
    }
    original_vertices = vertices.copy()
    original_observations = {
        name: points.copy() for name, points in observations.items()
    }
    context = RigConsistentPoseContext(
        vertices=vertices,
        landmark_triangles=triangles,
        landmark_barycentric=barycentric,
        initial_views=views,
        observed_landmarks_68=observations,
    )

    result = fit_rig_consistent_pose(context)

    assert result.success
    np.testing.assert_array_equal(vertices, original_vertices)
    for name, points in observations.items():
        np.testing.assert_array_equal(points, original_observations[name])
