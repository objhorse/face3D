from __future__ import annotations

import numpy as np

from src.geometry.nasal_base_observations import (
    NasalBaseObservationBundle,
    NasalBaseViewObservation,
)
from src.geometry.nasal_base_optimizer import (
    NASAL_BASE_OPTIMIZATION_STAGES,
    NasalBaseOptimizationContext,
    fit_staged_nasal_base,
    project_nasal_base_landmarks,
)
from src.geometry.nasal_base_semantic_basis import (
    NASAL_BASE_MODE_NAMES,
    apply_nasal_base_semantic_basis,
    build_nasal_base_semantic_basis,
)
from src.geometry.observable_flame_subspace import ProjectionView
from tests.test_nasal_semantic_basis import _synthetic_face


def _views() -> tuple[ProjectionView, ...]:
    intrinsic = np.asarray(
        ((120.0, 0.0, 32.0), (0.0, 120.0, 32.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    angle = np.deg2rad(32.0)
    rotate_left = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    rotate_right = rotate_left.T
    translation = np.asarray((0.0, 0.0, 4.0))
    return (
        ProjectionView("front", intrinsic, np.eye(3), translation),
        ProjectionView("subject-left", intrinsic, rotate_left, translation),
        ProjectionView("subject-right", intrinsic, rotate_right, translation),
    )


def _observation(
    semantic_view: str,
    projected: np.ndarray,
) -> NasalBaseViewObservation:
    if semantic_view == "front":
        selection = np.asarray((0, 1, 2, 3, 4))
    elif semantic_view == "subject-left":
        selection = np.asarray((2, 3, 4))
    else:
        selection = np.asarray((0, 1, 2))
    names = (
        "subject_right_outer",
        "subject_right_inner",
        "columella",
        "subject_left_inner",
        "subject_left_outer",
    )
    return NasalBaseViewObservation(
        semantic_view=semantic_view,
        anchor_names=tuple(names[index] for index in selection),
        mediapipe_indices=np.asarray((75, 97, 2, 326, 305))[selection],
        landmark_68_indices=np.asarray((31, 32, 33, 34, 35))[selection],
        source_xy=projected[selection],
        target_xy=projected[selection],
        confidence=np.ones(len(selection)),
    )


def _problem(target_coefficients: np.ndarray):
    vertices, faces, triangles, barycentric = _synthetic_face()
    basis = build_nasal_base_semantic_basis(
        vertices,
        faces,
        triangles,
        barycentric,
        np.eye(3),
    )
    views = _views()
    target_vertices = apply_nasal_base_semantic_basis(
        vertices,
        basis,
        target_coefficients,
    )
    projected = {
        view.name: project_nasal_base_landmarks(
            target_vertices,
            triangles,
            barycentric,
            view,
        )
        for view in views
    }
    observations = {
        name: _observation(name, points)
        for name, points in projected.items()
    }
    bundle = NasalBaseObservationBundle(
        front=observations["front"],
        subject_left=observations["subject-left"],
        subject_right=observations["subject-right"],
    )
    return NasalBaseOptimizationContext(
        baseline_vertices=vertices,
        landmark_triangles=triangles,
        landmark_barycentric=barycentric,
        basis=basis,
        views=views,
        observations=bundle,
    )


def test_staged_optimizer_uses_semantic_parameter_ownership():
    target = np.asarray((0.4, -0.3, 0.5, 0.15, -0.25, 0.2, 0.35, -0.1))
    context = _problem(target)

    result = fit_staged_nasal_base(context)

    assert result.success
    assert tuple(result.stage_results) == NASAL_BASE_OPTIMIZATION_STAGES
    assert result.stage_results["frontal"].active_parameter_names == (
        "columella_vertical",
        "nostril_width_shared",
        "nostril_width_asymmetry",
        "nostril_height_shared",
        "nostril_height_asymmetry",
    )
    assert result.stage_results["oblique"].active_parameter_names == (
        "columella_depth",
        "alar_rim_curvature_shared",
        "alar_rim_curvature_asymmetry",
    )
    assert result.final_cost < result.initial_cost * 0.05


def test_fit_does_not_move_any_vertex_outside_nasal_base_support():
    target = np.asarray((0.2, 0.2, -0.2, 0.1, 0.25, -0.1, 0.2, 0.1))
    context = _problem(target)
    result = fit_staged_nasal_base(context)

    assert result.success
    np.testing.assert_array_equal(
        result.candidate_vertices[~context.basis.support_mask],
        context.baseline_vertices[~context.basis.support_mask],
    )
    np.testing.assert_array_equal(
        result.candidate_vertices[context.basis.protected_mask],
        context.baseline_vertices[context.basis.protected_mask],
    )
    assert result.coefficients.shape == (len(NASAL_BASE_MODE_NAMES),)


def test_local_objective_is_invariant_to_per_view_image_translation():
    target = np.asarray((0.25, -0.2, 0.3, 0.1, -0.2, 0.15, 0.2, -0.05))
    context = _problem(target)
    offsets = {
        "front": np.asarray((18.0, -7.0)),
        "subject-left": np.asarray((-12.0, 9.0)),
        "subject-right": np.asarray((15.0, 5.0)),
    }
    translated = {}
    for name, observation in context.observations.by_view.items():
        translated[name] = NasalBaseViewObservation(
            semantic_view=name,
            anchor_names=observation.anchor_names,
            mediapipe_indices=observation.mediapipe_indices,
            landmark_68_indices=observation.landmark_68_indices,
            source_xy=observation.source_xy + offsets[name],
            target_xy=observation.target_xy + offsets[name],
            confidence=observation.confidence,
        )
    translated_context = NasalBaseOptimizationContext(
        baseline_vertices=context.baseline_vertices,
        landmark_triangles=context.landmark_triangles,
        landmark_barycentric=context.landmark_barycentric,
        basis=context.basis,
        views=context.views,
        observations=NasalBaseObservationBundle(
            front=translated["front"],
            subject_left=translated["subject-left"],
            subject_right=translated["subject-right"],
        ),
    )

    result = fit_staged_nasal_base(translated_context)

    assert result.success
    assert result.final_cost < result.initial_cost * 0.05
