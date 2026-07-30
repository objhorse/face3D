"""Bounded 3D observation fitting over a low-frequency deformation graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from src.geometry.deformation_graph import DeformationGraph, apply_node_translations, smooth_vertex_displacements
from src.geometry.differentiable_silhouette import IDENTITY_SILHOUETTE_REGIONS


@dataclass(frozen=True)
class LowFrequencyBasisConfig:
    """Geometry-relative limits for the stable identity control basis."""

    width_sigma: float = 0.24
    protection_core_ratio: float = 0.035
    protection_outer_ratio: float = 0.12
    width_displacement_ratio: float = 0.040
    depth_displacement_ratio: float = 0.025
    chin_displacement_ratio: float = 0.030
    max_vertex_displacement_ratio: float = 0.060


@dataclass(frozen=True)
class LowFrequencyIdentityBasis:
    """Small canonical deformation basis shared by every camera and expression."""

    names: tuple[str, ...]
    vectors: np.ndarray
    symmetry_pairs: tuple[tuple[int, int], ...]
    protected_mask: np.ndarray
    editable_mask: np.ndarray
    face_center: np.ndarray
    face_width: float
    face_height: float
    max_vertex_displacement: float

    def validate(self, vertex_count: int) -> None:
        expected = (len(self.names), int(vertex_count), 3)
        if self.vectors.shape != expected:
            raise ValueError(f"basis vectors have shape {self.vectors.shape}, expected {expected}")
        if self.protected_mask.shape != (int(vertex_count),):
            raise ValueError("protected mask does not match vertex count")
        if self.editable_mask.shape != (int(vertex_count),):
            raise ValueError("editable mask does not match vertex count")
        if not np.isfinite(self.vectors).all():
            raise ValueError("basis contains NaN or Inf")
        if not np.isfinite(self.face_width) or self.face_width <= 0.0:
            raise ValueError("face width must be finite and positive")


@dataclass(frozen=True)
class LowFrequencyOptimizationConfig:
    max_iterations: int = 120
    learning_rate: float = 0.05
    checkpoint_interval: int = 5
    observation_weight: float = 1.0
    coefficient_weight: float = 0.005
    symmetry_weight: float = 0.02
    edge_weight: float = 0.20
    laplacian_weight: float = 0.50
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class LowFrequencySafetyThresholds:
    max_vertex_displacement_ratio: float = 0.060
    max_mean_displacement_ratio: float = 0.025
    max_protected_displacement_ratio: float = 0.001
    max_new_normal_flips: int = 0


@dataclass(frozen=True)
class LowFrequencyAcceptanceThresholds:
    min_front_boundary_improvement_ratio: float = 0.30
    max_front_region_worsen_ratio: float = 0.10
    max_profile_boundary_worsen_ratio: float = 0.10
    max_interior_landmark_worsen_ratio: float = 0.10
    max_overlap_drop: float = 0.01


_WIDTH_BANDS = (
    ("temple", 0.72),
    ("cheekbone", 0.30),
    ("cheek", -0.05),
    ("jaw", -0.50),
    ("chin", -0.88),
)
_DEPTH_BANDS = (
    ("cheekbone", 0.24),
    ("jaw", -0.50),
)


def _smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        raise ValueError("smoothstep requires edge1 > edge0")
    t = np.clip((np.asarray(values, dtype=np.float64) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def stable_feature_landmark_points(landmarks: np.ndarray) -> np.ndarray:
    """Return eye, nose, mouth, and philtrum points from a 68-point face."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("landmarks must have shape (N, 3)")
    if len(points) < 68:
        raise ValueError("68 landmarks are required when protected points are not supplied")
    stable = [points[36:48], points[27:36], points[48:68]]
    philtrum = np.linspace(points[33], points[51], num=5, dtype=np.float64)[1:-1]
    stable.append(philtrum)
    return np.concatenate(stable, axis=0)


def _canonical_face_frame(landmarks: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("face landmarks must have shape (N, 3) with at least three points")
    x_low, x_high = np.percentile(points[:, 0], [1.0, 99.0])
    y_low, y_high = np.percentile(points[:, 1], [1.0, 99.0])
    face_width = float(x_high - x_low)
    face_height = float(y_high - y_low)
    if face_width <= 1e-8 or face_height <= 1e-8:
        raise ValueError("face landmarks do not span a usable face frame")
    center = np.array([(x_low + x_high) * 0.5, (y_low + y_high) * 0.5, np.median(points[:, 2])])
    if len(points) >= 68:
        upper_y = float(np.mean(points[19:25, 1]))
        lower_y = float(np.mean(points[6:11, 1]))
        nose_z = float(np.mean(points[27:36, 2]))
        contour_z = float(np.median(points[:17, 2]))
    else:
        upper_y = float(y_high)
        lower_y = float(y_low)
        nose_z = float(np.percentile(points[:, 2], 90.0))
        contour_z = float(np.percentile(points[:, 2], 20.0))
    up_sign = 1.0 if upper_y >= lower_y else -1.0
    front_sign = 1.0 if nose_z >= contour_z else -1.0
    return center, face_width, face_height, up_sign, front_sign


def restrict_low_frequency_identity_basis(
    basis: LowFrequencyIdentityBasis,
    allowed_prefixes: Sequence[str],
) -> tuple[LowFrequencyIdentityBasis, tuple[str, ...]]:
    """Disable controls outside trusted observation regions without changing topology."""
    prefixes = tuple(str(value) for value in allowed_prefixes)
    active = tuple(
        name for name in basis.names if any(name.startswith(prefix) for prefix in prefixes)
    )
    vectors = basis.vectors.copy()
    for index, name in enumerate(basis.names):
        if name not in active:
            vectors[index] = 0.0
    return replace(basis, vectors=vectors), active


def build_low_frequency_identity_basis(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_landmarks: np.ndarray,
    protected_points: Optional[np.ndarray] = None,
    config: Optional[LowFrequencyBasisConfig] = None,
) -> LowFrequencyIdentityBasis:
    """Construct a deterministic 16-parameter semantic basis in FLAME space."""
    cfg = config or LowFrequencyBasisConfig()
    verts = np.asarray(vertices, dtype=np.float64)
    faces_np = np.asarray(faces, dtype=np.int64)
    landmarks = np.asarray(face_landmarks, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if len(faces_np) and (faces_np.min() < 0 or faces_np.max() >= len(verts)):
        raise ValueError("faces contain invalid vertex indices")

    center, face_width, face_height, up_sign, front_sign = _canonical_face_frame(landmarks)
    x_norm = (verts[:, 0] - center[0]) / max(face_width * 0.5, 1e-8)
    y_norm = up_sign * (verts[:, 1] - center[1]) / max(face_height * 0.5, 1e-8)
    anchor_front = front_sign * landmarks[:, 2]
    vertex_front = front_sign * verts[:, 2]

    x_envelope = 1.0 - _smoothstep(1.00, 1.18, np.abs(x_norm))
    lower_envelope = _smoothstep(-1.16, -1.02, y_norm)
    upper_envelope = 1.0 - _smoothstep(1.08, 1.24, y_norm)
    front_floor = float(np.percentile(anchor_front, 3.0) - 0.08 * face_width)
    front_envelope = _smoothstep(front_floor, front_floor + 0.12 * face_width, vertex_front)
    face_support = x_envelope * lower_envelope * upper_envelope * front_envelope

    if protected_points is None:
        protected = stable_feature_landmark_points(landmarks)
    else:
        protected = np.asarray(protected_points, dtype=np.float64)
        if protected.ndim != 2 or protected.shape[1] != 3 or not len(protected):
            raise ValueError("protected points must have shape (N, 3)")
    distances = np.linalg.norm(verts[:, None, :] - protected[None, :, :], axis=2).min(axis=1)
    distance_ratio = distances / face_width
    protection = _smoothstep(
        float(cfg.protection_core_ratio),
        float(cfg.protection_outer_ratio),
        distance_ratio,
    )
    protected_mask = distance_ratio <= float(cfg.protection_core_ratio)
    support = face_support * protection
    side_transition = _smoothstep(0.02, 0.22, np.abs(x_norm))
    xneg = side_transition * (x_norm < 0.0)
    xpos = side_transition * (x_norm > 0.0)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_axis = np.array([0.0, up_sign, 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, front_sign], dtype=np.float64)

    names: list[str] = []
    vectors: list[np.ndarray] = []
    symmetry_pairs: list[tuple[int, int]] = []

    def add_pair(name: str, vertical_center: float, direction: np.ndarray, amplitude: float) -> None:
        vertical = np.exp(-0.5 * ((y_norm - vertical_center) / float(cfg.width_sigma)) ** 2)
        base_weight = support * vertical
        left_index = len(vectors)
        names.append(f"{name}_xneg")
        vectors.append((base_weight * xneg)[:, None] * (-direction)[None, :] * amplitude)
        right_index = len(vectors)
        names.append(f"{name}_xpos")
        vectors.append((base_weight * xpos)[:, None] * direction[None, :] * amplitude)
        symmetry_pairs.append((left_index, right_index))

    width_amplitude = float(cfg.width_displacement_ratio) * face_width
    for band_name, band_center in _WIDTH_BANDS:
        add_pair(f"{band_name}_width", band_center, x_axis, width_amplitude)

    depth_amplitude = float(cfg.depth_displacement_ratio) * face_width
    for band_name, band_center in _DEPTH_BANDS:
        add_pair(f"{band_name}_depth", band_center, z_axis, depth_amplitude)

    chin_vertical = np.exp(-0.5 * ((y_norm + 0.90) / 0.20) ** 2)
    chin_center = 1.0 - _smoothstep(0.45, 0.92, np.abs(x_norm))
    chin_weight = support * chin_vertical * chin_center
    chin_amplitude = float(cfg.chin_displacement_ratio) * face_width
    names.append("chin_depth")
    vectors.append(chin_weight[:, None] * z_axis[None, :] * chin_amplitude)
    names.append("chin_length")
    vectors.append(chin_weight[:, None] * (-y_axis)[None, :] * chin_amplitude)

    vector_array = np.asarray(vectors, dtype=np.float32)
    vector_array[:, protected_mask, :] = 0.0
    basis = LowFrequencyIdentityBasis(
        names=tuple(names),
        vectors=vector_array,
        symmetry_pairs=tuple(symmetry_pairs),
        protected_mask=protected_mask.astype(bool),
        editable_mask=(support > 1e-4),
        face_center=center.astype(np.float32),
        face_width=float(face_width),
        face_height=float(face_height),
        max_vertex_displacement=float(cfg.max_vertex_displacement_ratio) * face_width,
    )
    basis.validate(len(verts))
    return basis


def apply_low_frequency_identity_numpy(
    baseline_vertices: np.ndarray,
    basis: LowFrequencyIdentityBasis,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply bounded dimensionless coefficients and return candidate and displacement."""
    baseline = np.asarray(baseline_vertices, dtype=np.float32)
    basis.validate(len(baseline))
    coeff = np.asarray(coefficients, dtype=np.float32).reshape(-1)
    if len(coeff) != len(basis.names):
        raise ValueError("coefficient count does not match basis")
    coeff = np.clip(coeff, -1.0, 1.0)
    displacement = np.einsum("p,pvc->vc", coeff, basis.vectors, optimize=True)
    norms = np.linalg.norm(displacement, axis=1)
    scale = np.minimum(1.0, float(basis.max_vertex_displacement) / np.maximum(norms, 1e-12))
    displacement = displacement * scale[:, None]
    displacement[basis.protected_mask] = 0.0
    return baseline + displacement, displacement


def apply_low_frequency_identity_torch(
    baseline_vertices: torch.Tensor,
    basis_vectors: torch.Tensor,
    coefficients: torch.Tensor,
    max_vertex_displacement: float,
    protected_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable counterpart used by the multi-view optimizer."""
    if basis_vectors.ndim != 3 or basis_vectors.shape[1:] != baseline_vertices.shape:
        raise ValueError("basis tensor and baseline vertices are incompatible")
    if coefficients.numel() != basis_vectors.shape[0]:
        raise ValueError("coefficient count does not match basis")
    displacement = torch.einsum("p,pvc->vc", coefficients.reshape(-1), basis_vectors)
    norms = torch.linalg.norm(displacement, dim=1).clamp_min(1e-12)
    limit = torch.as_tensor(max_vertex_displacement, dtype=displacement.dtype, device=displacement.device)
    displacement = displacement * torch.minimum(torch.ones_like(norms), limit / norms)[:, None]
    if protected_mask is not None:
        displacement = displacement.masked_fill(protected_mask[:, None], 0.0)
    return baseline_vertices + displacement, displacement


def _unique_mesh_edges(faces: np.ndarray) -> np.ndarray:
    face_idx = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate(
        [face_idx[:, [0, 1]], face_idx[:, [1, 2]], face_idx[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _uniform_laplacian_torch(vertices: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    vertex_count = vertices.shape[0]
    neighbor_sum = torch.zeros_like(vertices)
    degree = torch.zeros(vertex_count, dtype=vertices.dtype, device=vertices.device)
    a, b = edges[:, 0], edges[:, 1]
    neighbor_sum.index_add_(0, a, vertices[b])
    neighbor_sum.index_add_(0, b, vertices[a])
    ones = torch.ones(len(edges), dtype=vertices.dtype, device=vertices.device)
    degree.index_add_(0, a, ones)
    degree.index_add_(0, b, ones)
    neighbor_mean = neighbor_sum / degree.clamp_min(1.0)[:, None]
    return vertices - neighbor_mean


def optimize_low_frequency_identity(
    baseline_vertices: np.ndarray,
    faces: np.ndarray,
    basis: LowFrequencyIdentityBasis,
    observation_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    config: Optional[LowFrequencyOptimizationConfig] = None,
    device: str = "cpu",
) -> dict:
    """Optimize bounded basis coefficients and return auditable checkpoints."""
    cfg = config or LowFrequencyOptimizationConfig()
    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    baseline_np = np.asarray(baseline_vertices, dtype=np.float32)
    basis.validate(len(baseline_np))
    baseline = torch.tensor(baseline_np, dtype=torch.float32, device=dev)
    vectors = torch.tensor(basis.vectors, dtype=torch.float32, device=dev)
    protected = torch.tensor(basis.protected_mask, dtype=torch.bool, device=dev)
    edges = torch.tensor(_unique_mesh_edges(faces), dtype=torch.long, device=dev)
    baseline_edge_lengths = torch.linalg.norm(
        baseline[edges[:, 1]] - baseline[edges[:, 0]], dim=1
    ).clamp_min(1e-8)
    baseline_laplacian = _uniform_laplacian_torch(baseline, edges).detach()
    raw_coefficients = torch.zeros(len(basis.names), dtype=torch.float32, device=dev, requires_grad=True)
    optimizer = torch.optim.Adam([raw_coefficients], lr=float(cfg.learning_rate))
    checkpoints = []
    history = []

    for step in range(int(cfg.max_iterations)):
        optimizer.zero_grad()
        coefficients = torch.tanh(raw_coefficients)
        candidate, displacement = apply_low_frequency_identity_torch(
            baseline,
            vectors,
            coefficients,
            basis.max_vertex_displacement,
            protected,
        )
        observation_loss = observation_loss_fn(candidate)
        if observation_loss.ndim != 0:
            observation_loss = observation_loss.mean()
        coefficient_loss = coefficients.square().mean()
        if basis.symmetry_pairs:
            pair_idx = torch.tensor(basis.symmetry_pairs, dtype=torch.long, device=dev)
            symmetry_loss = (coefficients[pair_idx[:, 0]] - coefficients[pair_idx[:, 1]]).square().mean()
        else:
            symmetry_loss = torch.zeros((), dtype=baseline.dtype, device=dev)
        edge_lengths = torch.linalg.norm(
            candidate[edges[:, 1]] - candidate[edges[:, 0]], dim=1
        )
        edge_loss = ((edge_lengths / baseline_edge_lengths) - 1.0).square().mean()
        laplacian_delta = _uniform_laplacian_torch(candidate, edges) - baseline_laplacian
        laplacian_loss = (laplacian_delta / float(basis.face_width)).square().mean()
        total_loss = (
            float(cfg.observation_weight) * observation_loss
            + float(cfg.coefficient_weight) * coefficient_loss
            + float(cfg.symmetry_weight) * symmetry_loss
            + float(cfg.edge_weight) * edge_loss
            + float(cfg.laplacian_weight) * laplacian_loss
        )
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite low-frequency optimization loss at step {step + 1}")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_coefficients], float(cfg.gradient_clip_norm))
        optimizer.step()

        should_save = (
            step == 0
            or (step + 1) % max(int(cfg.checkpoint_interval), 1) == 0
            or step + 1 == int(cfg.max_iterations)
        )
        if should_save:
            with torch.no_grad():
                coeff_np = torch.tanh(raw_coefficients).detach().cpu().numpy().astype(np.float32)
                candidate_np, displacement_np = apply_low_frequency_identity_numpy(
                    baseline_np, basis, coeff_np
                )
                record = {
                    "step": int(step + 1),
                    "total_loss": float(total_loss.detach().cpu()),
                    "observation_loss": float(observation_loss.detach().cpu()),
                    "coefficient_loss": float(coefficient_loss.detach().cpu()),
                    "symmetry_loss": float(symmetry_loss.detach().cpu()),
                    "edge_loss": float(edge_loss.detach().cpu()),
                    "laplacian_loss": float(laplacian_loss.detach().cpu()),
                    "coefficient_norm": float(np.linalg.norm(coeff_np)),
                    "max_displacement": float(np.linalg.norm(displacement_np, axis=1).max()),
                }
                history.append(record)
                checkpoints.append(
                    {
                        "step": int(step + 1),
                        "vertices": candidate_np,
                        "displacement": displacement_np,
                        "coefficients": coeff_np,
                        "losses": record,
                    }
                )

    return {
        "parameter_count": len(basis.names),
        "parameter_names": list(basis.names),
        "history": history,
        "checkpoints": checkpoints,
    }


def count_new_normal_flips(
    baseline_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
) -> int:
    face_idx = np.asarray(faces, dtype=np.int64)
    baseline_tri = np.asarray(baseline_vertices, dtype=np.float64)[face_idx]
    candidate_tri = np.asarray(candidate_vertices, dtype=np.float64)[face_idx]
    baseline_normals = np.cross(
        baseline_tri[:, 1] - baseline_tri[:, 0], baseline_tri[:, 2] - baseline_tri[:, 0]
    )
    candidate_normals = np.cross(
        candidate_tri[:, 1] - candidate_tri[:, 0], candidate_tri[:, 2] - candidate_tri[:, 0]
    )
    baseline_length = np.linalg.norm(baseline_normals, axis=1)
    candidate_length = np.linalg.norm(candidate_normals, axis=1)
    valid = (baseline_length > 1e-12) & (candidate_length > 1e-12)
    dots = np.einsum("ij,ij->i", baseline_normals, candidate_normals)
    return int(np.count_nonzero(valid & (dots <= 0.0)))


def evaluate_low_frequency_safety(
    baseline_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    basis: LowFrequencyIdentityBasis,
    *,
    thresholds: Optional[LowFrequencySafetyThresholds] = None,
) -> dict:
    """Independent displacement and normal-orientation gates."""
    limits = thresholds or LowFrequencySafetyThresholds()
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    candidate = np.asarray(candidate_vertices, dtype=np.float64)
    displacement = candidate - baseline
    norms = np.linalg.norm(displacement, axis=1)
    max_ratio = float(norms.max() / basis.face_width)
    mean_ratio = float(norms[basis.editable_mask].mean() / basis.face_width) if np.any(basis.editable_mask) else 0.0
    protected_ratio = float(norms[basis.protected_mask].max() / basis.face_width) if np.any(basis.protected_mask) else 0.0
    normal_flips = count_new_normal_flips(baseline, candidate, faces)
    gates = {
        "finite": bool(np.isfinite(candidate).all()),
        "max_displacement": max_ratio <= float(limits.max_vertex_displacement_ratio) + 1e-9,
        "mean_displacement": mean_ratio <= float(limits.max_mean_displacement_ratio) + 1e-9,
        "protected_displacement": protected_ratio <= float(limits.max_protected_displacement_ratio) + 1e-9,
        "normal_orientation": normal_flips <= int(limits.max_new_normal_flips),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "max_displacement_ratio": max_ratio,
        "mean_editable_displacement_ratio": mean_ratio,
        "max_protected_displacement_ratio": protected_ratio,
        "new_normal_flips": normal_flips,
        "thresholds": {
            "max_vertex_displacement_ratio": float(limits.max_vertex_displacement_ratio),
            "max_mean_displacement_ratio": float(limits.max_mean_displacement_ratio),
            "max_protected_displacement_ratio": float(limits.max_protected_displacement_ratio),
            "max_new_normal_flips": int(limits.max_new_normal_flips),
        },
    }


def select_low_frequency_checkpoint(
    baseline_vertices: np.ndarray,
    checkpoints: Sequence[dict],
    evaluator: Callable[[dict], dict],
) -> tuple[np.ndarray, Optional[dict], list[dict]]:
    """Select the best accepted checkpoint or return the exact baseline."""
    trials = []
    accepted = []
    for checkpoint in checkpoints:
        decision = dict(evaluator(checkpoint))
        trial = {
            "step": int(checkpoint["step"]),
            "accepted": bool(decision.get("accepted", False)),
            "rank_score": float(decision.get("rank_score", np.inf)),
            "decision": decision,
            "checkpoint": checkpoint,
        }
        trials.append(trial)
        if trial["accepted"] and np.isfinite(trial["rank_score"]):
            accepted.append(trial)
    if not accepted:
        public_trials = [{key: value for key, value in row.items() if key != "checkpoint"} for row in trials]
        return np.asarray(baseline_vertices, dtype=np.float32).copy(), None, public_trials
    selected = min(
        accepted,
        key=lambda row: (
            row["rank_score"],
            float(row["checkpoint"]["losses"].get("coefficient_norm", np.inf)),
            int(row["step"]),
        ),
    )
    public_trials = [{key: value for key, value in row.items() if key != "checkpoint"} for row in trials]
    return np.asarray(selected["checkpoint"]["vertices"], dtype=np.float32).copy(), selected, public_trials


def evaluate_low_frequency_observations(
    before_records: Sequence[dict],
    after_records: Sequence[dict],
    *,
    safety_gate: Optional[dict] = None,
    mesh_quality_gate: Optional[dict] = None,
    thresholds: Optional[LowFrequencyAcceptanceThresholds] = None,
) -> dict:
    """Evaluate front, profiles, interior anchors, and geometry independently."""
    limits = thresholds or LowFrequencyAcceptanceThresholds()
    before_by_view = {str(item["view"]): item for item in before_records}
    after_by_view = {str(item["view"]): item for item in after_records}
    shared_views = sorted(set(before_by_view) & set(after_by_view))
    per_view = []
    for view in shared_views:
        before = before_by_view[view]
        after = after_by_view[view]
        before_silhouette = before.get("silhouette") or {}
        after_silhouette = after.get("silhouette") or {}
        before_boundary = float(before_silhouette.get(
            "identity_balanced_boundary_mean_px",
            before_silhouette.get(
                "regional_balanced_boundary_mean_px",
                before_silhouette.get("trusted_boundary_mean_px", np.inf),
            ),
        ))
        after_boundary = float(after_silhouette.get(
            "identity_balanced_boundary_mean_px",
            after_silhouette.get(
                "regional_balanced_boundary_mean_px",
                after_silhouette.get("trusted_boundary_mean_px", np.inf),
            ),
        ))
        before_interior = float(before.get("interior_mean_px", np.inf))
        after_interior = float(after.get("interior_mean_px", np.inf))
        before_overlap = float(before_silhouette.get("trusted_region_dice", 0.0))
        after_overlap = float(after_silhouette.get("trusted_region_dice", 0.0))
        if not np.isfinite(before_boundary) or not np.isfinite(after_boundary):
            continue
        boundary_change = (after_boundary - before_boundary) / max(before_boundary, 1e-6)
        interior_change = (after_interior - before_interior) / max(before_interior, 1e-6)
        region_changes = {}
        before_regions = before_silhouette.get("regions") or {}
        after_regions = after_silhouette.get("regions") or {}
        for region in sorted(set(before_regions) & set(after_regions)):
            before_region = float(before_regions[region].get("boundary_mean_px", np.inf))
            after_region = float(after_regions[region].get("boundary_mean_px", np.inf))
            if not np.isfinite(before_region) or not np.isfinite(after_region):
                continue
            change = (after_region - before_region) / max(before_region, 1.0)
            region_changes[region] = {
                "before_boundary_px": before_region,
                "after_boundary_px": after_region,
                "boundary_change_ratio": float(change),
                "boundary_improvement_ratio": float(-change),
                "before_signed_width_error_px": float(
                    before_regions[region].get("signed_width_error_px", np.nan)
                ),
                "after_signed_width_error_px": float(
                    after_regions[region].get("signed_width_error_px", np.nan)
                ),
            }
        per_view.append(
            {
                "view": view,
                "before_boundary_px": before_boundary,
                "after_boundary_px": after_boundary,
                "boundary_change_ratio": boundary_change,
                "boundary_improvement_ratio": -boundary_change,
                "before_interior_px": before_interior,
                "after_interior_px": after_interior,
                "interior_worsen_ratio": interior_change,
                "overlap_drop": before_overlap - after_overlap,
                "regions": region_changes,
            }
        )

    front = next((item for item in per_view if item["view"] == "front"), None)
    profiles = [item for item in per_view if item["view"] != "front"]
    finite_interior = [
        item for item in per_view if np.isfinite(item["interior_worsen_ratio"])
    ]
    mean_interior_worsen = (
        float(np.mean([item["interior_worsen_ratio"] for item in finite_interior]))
        if finite_interior
        else float("inf")
    )
    max_profile_worsen = (
        float(max(item["boundary_change_ratio"] for item in profiles))
        if profiles
        else float("inf")
    )
    max_overlap_drop = (
        float(max(item["overlap_drop"] for item in per_view))
        if per_view
        else float("inf")
    )
    front_region_changes = [
        values
        for name, values in (front or {}).get("regions", {}).items()
        if name in IDENTITY_SILHOUETTE_REGIONS
    ]
    max_front_region_worsen = (
        float(max(item["boundary_change_ratio"] for item in front_region_changes))
        if front_region_changes
        else float("-inf")
    )
    safety_passed = bool((safety_gate or {"passed": True}).get("passed", False))
    mesh_passed = bool((mesh_quality_gate or {"passed": True}).get("passed", False))
    gates = {
        "front_available": front is not None,
        "profiles_available": len(profiles) >= 2,
        "front_boundary_improved": bool(
            front is not None
            and front["boundary_improvement_ratio"]
            >= float(limits.min_front_boundary_improvement_ratio)
        ),
        "front_regions_preserved": bool(
            not front_region_changes
            or max_front_region_worsen
            <= float(limits.max_front_region_worsen_ratio)
        ),
        "profiles_preserved": bool(
            len(profiles) >= 2
            and max_profile_worsen <= float(limits.max_profile_boundary_worsen_ratio)
        ),
        "interior_landmarks_preserved": bool(
            mean_interior_worsen <= float(limits.max_interior_landmark_worsen_ratio)
        ),
        "overlap_preserved": bool(max_overlap_drop <= float(limits.max_overlap_drop)),
        "displacement_safety": safety_passed,
        "mesh_quality": mesh_passed,
    }
    accepted = all(gates.values())
    rank_score = (
        float(np.mean([item["after_boundary_px"] for item in per_view]))
        if per_view
        else float("inf")
    )
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "accepted": accepted,
        "reason": "accepted by controlled identity gates" if accepted else "rejected: " + ", ".join(failed),
        "failed_gates": failed,
        "gates": gates,
        "rank_score": rank_score,
        "metrics": {
            "front_boundary_improvement_ratio": (
                float(front["boundary_improvement_ratio"]) if front is not None else float("-inf")
            ),
            "max_front_region_worsen_ratio": max_front_region_worsen,
            "max_profile_boundary_worsen_ratio": max_profile_worsen,
            "mean_interior_landmark_worsen_ratio": mean_interior_worsen,
            "max_overlap_drop": max_overlap_drop,
        },
        "per_view": per_view,
        "safety_gate": safety_gate or {"passed": True},
        "mesh_quality_gate": mesh_quality_gate or {"passed": True},
        "thresholds": {
            "min_front_boundary_improvement_ratio": float(limits.min_front_boundary_improvement_ratio),
            "max_front_region_worsen_ratio": float(limits.max_front_region_worsen_ratio),
            "max_profile_boundary_worsen_ratio": float(limits.max_profile_boundary_worsen_ratio),
            "max_interior_landmark_worsen_ratio": float(limits.max_interior_landmark_worsen_ratio),
            "max_overlap_drop": float(limits.max_overlap_drop),
        },
        "texture_metrics_excluded": True,
    }


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
