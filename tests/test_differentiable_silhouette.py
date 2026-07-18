import numpy as np
import pytest
import torch

from src.geometry.differentiable_silhouette import (
    build_silhouette_target,
    camera_vertices_to_clip,
    create_cuda_raster_context,
    evaluate_geometry_candidate,
    make_interior_landmark_weights,
    render_soft_silhouette,
    silhouette_metrics,
    weighted_silhouette_loss,
)


def test_camera_clip_projection_matches_pixel_projection():
    vertices = torch.tensor(
        [[0.0, 0.0, 2.0], [0.2, -0.1, 2.5], [-0.1, 0.15, 1.8]],
        dtype=torch.float32,
    )
    K = torch.tensor(
        [[800.0, 0.0, 100.0], [0.0, 820.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.zeros(3, dtype=torch.float32)

    clip = camera_vertices_to_clip(vertices, K, R, t, image_shape=(100, 200))
    ndc = clip[:, :2] / clip[:, 3:4]
    recovered_u = (ndc[:, 0] + 1.0) * 0.5 * 199.0
    recovered_v = (1.0 - ndc[:, 1]) * 0.5 * 99.0

    camera = vertices @ R.T + t
    projected = camera @ K.T
    expected = projected[:, :2] / projected[:, 2:3]
    recovered = torch.stack([recovered_u, recovered_v], dim=1)
    assert torch.allclose(recovered, expected, atol=1e-4)


def test_fixed_contour_landmarks_have_zero_weight():
    weights = make_interior_landmark_weights(
        point_count=68,
        base_weight=0.25,
        stable_indices=np.arange(27, 68),
        stable_weight=1.5,
        device="cpu",
    )

    assert torch.count_nonzero(weights[:17]).item() == 0
    assert torch.all(weights[17:27] == 0.25)
    assert torch.all(weights[27:] == 1.75)


def test_side_target_prefers_profile_boundary_near_image_center():
    mask = np.zeros((128, 128), dtype=np.uint8)
    # A side-view face patch: the profile is the left edge near image center,
    # while the far edge represents hair/ear territory.
    for y in range(20, 111):
        left = 54 - int(8 * np.sin((y - 20) / 90.0 * np.pi))
        mask[y, left:118] = 255

    target = build_silhouette_target(mask, view_name="left", resolution=128)
    reliability = target.reliability_np

    profile_weight = float(reliability[64, 46:60].max())
    far_edge_weight = float(reliability[64, 112:122].max())
    top_weight = float(reliability[20:25].max())

    assert profile_weight > 0.8
    assert far_edge_weight < 0.3
    assert top_weight < 0.3


def test_front_target_trusts_both_cheek_boundaries():
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv = np.indices(mask.shape)
    ellipse = ((cv[1] - 64.0) / 36.0) ** 2 + ((cv[0] - 68.0) / 48.0) ** 2 <= 1.0
    mask[ellipse] = 255

    target = build_silhouette_target(mask, view_name="front", resolution=128)
    reliability = target.reliability_np

    assert float(reliability[68, 25:35].max()) > 0.8
    assert float(reliability[68, 93:103].max()) > 0.8
    assert float(reliability[18:23].max()) < 0.3


def test_trusted_boundary_metric_is_normalized_by_face_width():
    mask = np.zeros((128, 128), dtype=np.uint8)
    yy, xx = np.indices(mask.shape)
    mask[((xx - 64.0) / 34.0) ** 2 + ((yy - 68.0) / 46.0) ** 2 <= 1.0] = 255

    target_128 = build_silhouette_target(mask, view_name="front", resolution=128)
    target_256 = build_silhouette_target(mask, view_name="front", resolution=256)
    pred_128 = np.roll(target_128.target_np, shift=2, axis=1)
    pred_256 = np.roll(target_256.target_np, shift=4, axis=1)

    metric_128 = silhouette_metrics(pred_128, target_128)
    metric_256 = silhouette_metrics(pred_256, target_256)
    assert metric_128["trusted_boundary_face_width_pct"] == pytest.approx(
        metric_256["trusted_boundary_face_width_pct"], abs=0.2
    )


def test_side_far_boundary_does_not_change_trusted_contour_metric():
    mask = np.zeros((128, 128), dtype=np.uint8)
    for y in range(20, 111):
        left = 54 - int(8 * np.sin((y - 20) / 90.0 * np.pi))
        mask[y, left:118] = 255
    target = build_silhouette_target(mask, view_name="left", resolution=128)
    baseline = silhouette_metrics(target.target_np, target)
    altered = target.target_np.copy()
    altered[45:80, 120:126] = 1.0
    candidate = silhouette_metrics(altered, target)

    assert candidate["trusted_boundary_face_width_pct"] == pytest.approx(
        baseline["trusted_boundary_face_width_pct"], abs=1e-6
    )


def _candidate_record(
    view, boundary_pct, interior_px=10.0, dice=0.8, legacy=10.0, render_tol=0.0
):
    return {
        "view": view,
        "interior_mean_px": interior_px,
        "legacy_fixed_contour_diagnostic_px": legacy,
        "silhouette": {
            "trusted_boundary_face_width_pct": boundary_pct,
            "trusted_region_dice": dice,
            "face_width_source_px": 1000.0,
            "render_pixel_face_width_pct": render_tol,
        },
    }


def test_candidate_gate_ignores_legacy_fixed_contour():
    views = ("left", "front", "right")
    before = [_candidate_record(view, 10.0, legacy=5.0) for view in views]
    after = [_candidate_record(view, 9.0, legacy=500.0) for view in views]

    decision = evaluate_geometry_candidate(
        before, after, mesh_quality_gate={"passed": True}
    )
    assert decision["accepted"]
    assert decision["legacy_fixed_contour_excluded"]


def test_candidate_gate_rejects_large_single_view_worsening():
    views = ("left", "front", "right")
    before = [_candidate_record(view, 10.0) for view in views]
    after = [
        _candidate_record("left", 8.0),
        _candidate_record("front", 8.0),
        _candidate_record("right", 10.3),
    ]

    decision = evaluate_geometry_candidate(
        before, after, mesh_quality_gate={"passed": True}
    )
    assert not decision["accepted"]
    assert "view_consistency" in decision["failed_gates"]


def test_candidate_gate_treats_subpixel_view_change_as_preserved():
    views = ("left", "front", "right")
    before = [
        _candidate_record(view, 10.0, render_tol=0.8) for view in views
    ]
    after = [
        _candidate_record("left", 10.5, render_tol=0.8),
        _candidate_record("front", 8.0, render_tol=0.8),
        _candidate_record("right", 10.45, render_tol=0.8),
    ]

    decision = evaluate_geometry_candidate(
        before, after, mesh_quality_gate={"passed": True}
    )
    assert decision["accepted"]
    assert decision["metrics"]["improved_views"] == 1
    assert decision["metrics"]["preserved_views"] == 3


def test_candidate_gate_rejects_interior_landmark_damage():
    views = ("left", "front", "right")
    before = [_candidate_record(view, 10.0, interior_px=10.0) for view in views]
    after = [_candidate_record(view, 9.0, interior_px=13.0) for view in views]

    decision = evaluate_geometry_candidate(
        before, after, mesh_quality_gate={"passed": True}
    )
    assert not decision["accepted"]
    assert "interior_mean_preserved" in decision["failed_gates"]
    assert "interior_views_preserved" in decision["failed_gates"]


def test_candidate_gate_rejects_mesh_quality_failure():
    views = ("left", "front", "right")
    before = [_candidate_record(view, 10.0) for view in views]
    after = [_candidate_record(view, 9.0) for view in views]

    decision = evaluate_geometry_candidate(
        before, after, mesh_quality_gate={"passed": False, "comparison": {}}
    )
    assert not decision["accepted"]
    assert "mesh_quality" in decision["failed_gates"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_soft_silhouette_has_finite_vertex_gradients():
    pytest.importorskip("nvdiffrast")
    vertices = torch.tensor(
        [[-0.4, -0.3, 2.0], [0.4, -0.3, 2.0], [0.0, 0.45, 2.0]],
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    K = torch.tensor(
        [[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]],
        device="cuda",
    )
    context = create_cuda_raster_context()
    prediction = render_soft_silhouette(
        vertices,
        faces,
        K,
        torch.eye(3, device="cuda"),
        torch.zeros(3, device="cuda"),
        image_shape=(64, 64),
        render_shape=(64, 64),
        context=context,
    )
    target = torch.roll(prediction.detach(), shifts=2, dims=1)
    loss = weighted_silhouette_loss(prediction, target, torch.ones_like(prediction))
    loss.backward()

    assert torch.isfinite(vertices.grad).all()
    assert float(vertices.grad.norm()) > 0.0
