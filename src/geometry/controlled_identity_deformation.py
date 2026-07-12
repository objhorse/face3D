"""Bounded 3D observation fitting over a low-frequency deformation graph."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from src.geometry.deformation_graph import DeformationGraph, apply_node_translations, smooth_vertex_displacements


@dataclass(frozen=True)
class ControlledIdentityProfile:
    name: str
    region_displacement_limits_m: Mapping[str, float]
    observation_weight: float = 1.0
    smooth_weight: float = 10.0
    control_weight: float = 0.01
    edge_weight: float = 0.0000001
    max_iterations: int = 240
    learning_rate: float = 0.025


def default_controlled_identity_profiles() -> tuple[ControlledIdentityProfile, ...]:
    base = {"nose": 0.0025, "mouth": 0.0015, "eye": 0.0010, "chin_or_jaw": 0.004, "face": 0.003}
    return (
        ControlledIdentityProfile("conservative", base),
        ControlledIdentityProfile("balanced", {key: value * 1.5 for key, value in base.items()}),
        ControlledIdentityProfile("likeness", {key: value * 2.0 for key, value in base.items()}),
    )


def _profile_limit(profile: ControlledIdentityProfile, region: str) -> float:
    return float(profile.region_displacement_limits_m.get(region, profile.region_displacement_limits_m.get("face", 0.003)))


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, singular, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    denominator = float(np.square(source_zero).sum())
    scale = float(singular.sum() / max(denominator, 1e-12))
    scale = float(np.clip(scale, 0.85, 1.15))
    translation = target_center - scale * (rotation @ source_center)
    return rotation, scale, translation


def normalize_observation_groups(
    observations: Sequence[object],
    min_group_size: int = 6,
) -> tuple[list[object], dict]:
    """Remove pair-specific similarity bias before non-rigid fitting.

    Pairwise triangulation can have excellent reprojection error while the two
    side-camera point clouds disagree in metric depth. Registering each pair to
    its current visible surface prevents that calibration residual from being
    interpreted as identity shape.
    """
    normalized = [SimpleNamespace(**vars(item)) for item in observations]
    report: dict[str, dict] = {}
    for side in ("left", "right"):
        indices = [
            index
            for index, item in enumerate(observations)
            if side in item.pixels_by_view and len(item.pixels_by_view) == 2
        ]
        if len(indices) < int(min_group_size):
            report[side] = {"count": len(indices), "applied": False, "reason": "too_few_observations"}
            continue
        source = np.asarray([observations[index].target_model_point for index in indices], dtype=np.float64)
        target = np.asarray([observations[index].surface_point for index in indices], dtype=np.float64)
        inliers = np.ones(len(indices), dtype=bool)
        for _ in range(3):
            rotation, scale, translation = _fit_similarity(source[inliers], target[inliers])
            aligned = (scale * (rotation @ source.T)).T + translation
            residual = np.linalg.norm(aligned - target, axis=1)
            median = float(np.median(residual[inliers]))
            mad = float(np.median(np.abs(residual[inliers] - median)))
            threshold = max(median + 3.0 * 1.4826 * mad, 0.0015)
            updated = residual <= threshold
            if updated.sum() < int(min_group_size) or np.array_equal(updated, inliers):
                break
            inliers = updated
        rotation, scale, translation = _fit_similarity(source[inliers], target[inliers])
        aligned = (scale * (rotation @ source.T)).T + translation
        before = np.linalg.norm(source - target, axis=1)
        after = np.linalg.norm(aligned - target, axis=1)
        for local_index, observation_index in enumerate(indices):
            normalized[observation_index].target_model_point = aligned[local_index].astype(np.float64)
        angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))))
        report[side] = {
            "count": len(indices),
            "inlier_count": int(inliers.sum()),
            "applied": True,
            "scale": scale,
            "rotation_deg": angle,
            "translation_m": translation.tolist(),
            "before_median_m": float(np.median(before)),
            "after_median_m": float(np.median(after)),
            "after_p90_m": float(np.percentile(after, 90)),
        }
    return normalized, report


def _attachments(observations: Sequence[object], faces: np.ndarray):
    face_ids = np.asarray([int(item.face_index) for item in observations], dtype=np.int64)
    bary = np.asarray([item.bary_coords for item in observations], dtype=np.float32)
    targets = np.asarray([item.target_model_point for item in observations], dtype=np.float32)
    weights = np.asarray([float(item.weight) for item in observations], dtype=np.float32)
    return np.asarray(faces, dtype=np.int64)[face_ids], bary, targets, weights


def optimize_controlled_identity_candidate(
    baseline_vertices: np.ndarray,
    faces: np.ndarray,
    graph: DeformationGraph,
    observations: Sequence[object],
    profile: ControlledIdentityProfile,
    device: str = "cpu",
) -> tuple[np.ndarray, dict]:
    if not observations:
        return np.asarray(baseline_vertices, dtype=np.float32).copy(), {"accepted": False, "reason": "no_observations"}
    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    baseline = torch.tensor(np.asarray(baseline_vertices, dtype=np.float32), device=dev)
    vertex_indices = torch.tensor(graph.vertex_node_indices, dtype=torch.long, device=dev)
    vertex_weights = torch.tensor(graph.vertex_node_weights, dtype=torch.float32, device=dev)
    triangle_indices, bary, targets, target_weights = _attachments(observations, faces)
    triangles = torch.tensor(triangle_indices, dtype=torch.long, device=dev)
    bary_t = torch.tensor(bary, dtype=torch.float32, device=dev)
    targets_t = torch.tensor(targets, dtype=torch.float32, device=dev)
    weights_t = torch.tensor(target_weights, dtype=torch.float32, device=dev)
    node_edges = torch.tensor(graph.node_edges, dtype=torch.long, device=dev)
    faces_np = np.asarray(faces, dtype=np.int64)
    mesh_edges_np = np.vstack((faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]))
    mesh_edges_np.sort(axis=1)
    mesh_edges_np = np.unique(mesh_edges_np, axis=0)
    mesh_edges = torch.tensor(mesh_edges_np, dtype=torch.long, device=dev)
    baseline_edge_lengths = torch.linalg.norm(
        baseline[mesh_edges[:, 1]] - baseline[mesh_edges[:, 0]], dim=1
    ).clamp_min(1e-4)
    limits = torch.tensor([_profile_limit(profile, region) for region in graph.node_regions], dtype=torch.float32, device=dev)
    controls = torch.zeros((len(graph.node_vertex_indices), 3), dtype=torch.float32, device=dev, requires_grad=True)
    optimizer = torch.optim.Adam([controls], lr=float(profile.learning_rate))
    history = []
    for step in range(int(profile.max_iterations)):
        optimizer.zero_grad()
        delta = torch.zeros_like(baseline)
        for column in range(vertex_indices.shape[1]):
            idx = vertex_indices[:, column]
            valid = idx >= 0
            if valid.any():
                delta[valid] = delta[valid] + vertex_weights[valid, column, None] * controls[idx[valid]]
        candidate = baseline + delta
        predicted = (candidate[triangles] * bary_t[:, :, None]).sum(dim=1)
        residual = predicted - targets_t
        observation_loss = (torch.nn.functional.huber_loss(predicted, targets_t, delta=0.003, reduction="none").mean(dim=1) * weights_t).sum() / weights_t.sum().clamp_min(1e-6)
        smooth_loss = torch.tensor(0.0, device=dev)
        if len(node_edges):
            smooth_loss = (controls[node_edges[:, 0]] - controls[node_edges[:, 1]]).square().mean()
        control_loss = controls.square().mean()
        candidate_edge_lengths = torch.linalg.norm(
            candidate[mesh_edges[:, 1]] - candidate[mesh_edges[:, 0]], dim=1
        )
        edge_loss = ((candidate_edge_lengths / baseline_edge_lengths) - 1.0).square().mean()
        loss = (
            profile.observation_weight * observation_loss
            + profile.smooth_weight * smooth_loss
            + profile.control_weight * control_loss
            + profile.edge_weight * edge_loss
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            norms = torch.linalg.norm(controls, dim=1).clamp_min(1e-12)
            controls.mul_(torch.minimum(torch.ones_like(norms), limits / norms)[:, None])
        if step % 20 == 0 or step == profile.max_iterations - 1:
            history.append({"step": step, "loss": float(loss.detach().cpu()), "observation_loss": float(observation_loss.detach().cpu()), "max_control_m": float(torch.linalg.norm(controls, dim=1).max().detach().cpu())})
    baseline_np = baseline.detach().cpu().numpy()
    raw_result = apply_node_translations(baseline_np, graph, controls.detach().cpu().numpy())
    smoothed_delta = smooth_vertex_displacements(
        faces, raw_result - baseline_np, iterations=60, relaxation=0.5
    )
    result = baseline_np + smoothed_delta
    before = np.linalg.norm(np.asarray([baseline_vertices[np.asarray(faces)[item.face_index]] * item.bary_coords[:, None] for item in observations]).sum(axis=1) - targets, axis=1)
    after = np.linalg.norm(np.asarray([result[np.asarray(faces)[item.face_index]] * item.bary_coords[:, None] for item in observations]).sum(axis=1) - targets, axis=1)
    return result, {"accepted": True, "before_median_m": float(np.median(before)), "after_median_m": float(np.median(after)), "improvement_ratio": float(1.0 - np.median(after) / max(np.median(before), 1e-9)), "history": history, "control_translations_m": controls.detach().cpu().numpy().tolist()}
