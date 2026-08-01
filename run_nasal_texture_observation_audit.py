"""Run hash-locked, observation-only cross-view nasal texture audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from run_expression_depth_experiment import _export_with_baseline_texture
from run_multiview_alar_surface_experiment import _load_a2_low_resolution
from run_multiview_nasal_shape_experiment import (
    _subdivide_candidate,
    _validate_embedded_textured_glb,
    _write_viewer,
    build_model_projection_views,
)
from run_nasal_observation_audit import (
    _load_capture_images,
    assert_file_tree_unchanged,
    build_undistorted_observation_rig,
    file_tree_hashes,
    validate_separate_output,
)
from src.cross_view_geometry import restore_mask_to_work_frame, scale_intrinsics
from src.geometry.alar_surface_basis import apply_alar_surface_basis, build_alar_surface_basis
from src.geometry.cross_view_surface_observations import sparse_zbuffer_attachments
from src.geometry.nasal_semantic_basis import build_nasal_semantic_basis
from src.geometry.nasal_texture_observations import (
    NasalCoordinateProvenance,
    NasalEpipolarMatchResult,
    NasalEpipolarSeed,
    NasalPairMatch,
    NasalTextureObservationBundle,
    NasalTextureObservationConfig,
    build_nasal_texture_confidence_maps,
    fixed_rig_fundamental,
    match_model_guided_nasal_pair,
    triangulate_nasal_pair_matches,
)
from src.geometry.nasal_view_registration import (
    BaselineNasalRegistrationSurface,
    NasalViewRegistrationConfig,
    ViewRegistrationSample,
    estimate_fixed_nasal_view_offsets,
)
from src.geometry.profile_triangulation import load_profile_rig
from src.geometry.roma_nasal_fitter import fit_roma_nasal_surface
from src.geometry.roma_nasal_matcher import (
    run_roma_nasal_pair,
    select_roma_nasal_matches,
)
from src.learned_cross_view_geometry import load_loftr_matcher, run_loftr_matches
from src.module2_geometry import export_mesh_glb, export_mesh_obj
from src.module3_texture import load_mesh_obj
from src.reports.nasal_geometry_report import (
    render_nasal_geometry_screenshots,
    validate_minimal_nasal_candidate,
)
from src.reports.nasal_observation_io import load_nasal_observation_bundle
from src.reports.nasal_texture_observation_report import (
    write_nasal_texture_observation_report,
)


DEFAULT_V10_OBJ_SHA256 = "053e4495c7c18240f27c44d67fa338ac2775c4a2a886156bc703a8f5fcea0376"
DEFAULT_V10_REPORT_SHA256 = "9ee414b86d47ce8960671b1c4335e5d04d02dc8a15ed506e083d7dc7bf1f8527"
DEFAULT_RIG_SHA256 = "c7f439c2216714f1a5c564ad6c7a6eef62944e03c2e9dc36cd3a6dbb8a710ea9"
DEFAULT_ROMA_PYTHON = Path(
    os.environ.get(
        "FACE3D_ROMA_PYTHON",
        r"D:\Anaconda\envs\pytorch\python.exe",
    )
)
DEFAULT_ROMA_TORCH_HOME = Path(
    os.environ.get(
        "FACE3D_ROMA_TORCH_HOME",
        str(Path(__file__).resolve().parent / "models" / "roma-cache"),
    )
)
DEFAULT_ROMA_WORKER = (
    Path(__file__).resolve().parent / "src" / "geometry" / "roma_nasal_worker.py"
)
SEMANTIC_TO_RIG_VIEW = {
    "front": "front",
    "subject-left": "left",
    "subject-right": "right",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash_locked_input(path: str | Path, expected_sha256: str, label: str) -> str:
    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing: {source}")
    actual = _sha256(source)
    if actual != str(expected_sha256).lower():
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def load_locked_v10_surface(
    obj_path: str | Path,
    baseline: Any,
    low_resolution_vertices: np.ndarray,
    *,
    max_export_rounding_error_m: float = 1.5e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Load the locked OBJ and verify that it is the exported v10 subdivision."""
    from src.module3_texture import load_mesh_obj

    locked_vertices, locked_faces, _uv, _uv_faces = load_mesh_obj(Path(obj_path))
    expected_vertices, expected_faces, _expected_uv, _expected_uv_faces = (
        _subdivide_candidate(baseline, np.asarray(low_resolution_vertices))
    )
    locked_vertices = np.asarray(locked_vertices, dtype=np.float64)
    locked_faces = np.asarray(locked_faces, dtype=np.int64)
    expected_vertices = np.asarray(expected_vertices, dtype=np.float64)
    expected_faces = np.asarray(expected_faces, dtype=np.int64)
    if locked_vertices.shape != expected_vertices.shape:
        raise RuntimeError("locked v10 OBJ vertex count disagrees with v10 lineage")
    if locked_faces.shape != expected_faces.shape or not np.array_equal(
        locked_faces,
        expected_faces,
    ):
        raise RuntimeError("locked v10 OBJ topology disagrees with v10 lineage")
    deviation = np.linalg.norm(locked_vertices - expected_vertices, axis=1)
    maximum = float(np.max(deviation)) if len(deviation) else float("inf")
    if not np.isfinite(maximum) or maximum > float(max_export_rounding_error_m):
        raise RuntimeError(
            "locked v10 OBJ vertices disagree with reconstructed v10 lineage: "
            f"maximum deviation {maximum:.9f} m"
        )
    return locked_vertices, locked_faces, {
        "vertex_count": int(len(locked_vertices)),
        "face_count": int(len(locked_faces)),
        "lineage_max_deviation_m": maximum,
    }


def subdivide_low_resolution_vertex_masks(
    faces: np.ndarray,
    masks_by_name: Mapping[str, np.ndarray],
    *,
    iterations: int = 2,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Propagate semantic masks through the exact Loop-subdivision topology."""
    import trimesh

    topology = np.asarray(faces, dtype=np.int64)
    if topology.ndim != 2 or topology.shape[1] != 3 or len(topology) == 0:
        raise ValueError("faces must have shape (F, 3)")
    vertex_count = int(np.max(topology)) + 1
    masks = {
        str(name): np.asarray(mask, dtype=bool).copy()
        for name, mask in masks_by_name.items()
    }
    if not masks or any(mask.shape != (vertex_count,) for mask in masks.values()):
        raise ValueError("semantic masks must match the low-resolution vertices")
    for _iteration in range(int(iterations)):
        edges, _edge_faces = trimesh.geometry.faces_to_edges(
            topology,
            return_index=True,
        )
        edges.sort(axis=1)
        unique, inverse = trimesh.grouping.unique_rows(edges)
        unique_edges = edges[unique]
        masks = {
            name: np.concatenate((mask, np.any(mask[unique_edges], axis=1)))
            for name, mask in masks.items()
        }
        odd_indices = inverse.reshape((-1, 3)) + vertex_count
        topology = np.column_stack(
            [
                topology[:, 0],
                odd_indices[:, 0],
                odd_indices[:, 2],
                odd_indices[:, 0],
                topology[:, 1],
                odd_indices[:, 1],
                odd_indices[:, 2],
                odd_indices[:, 1],
                topology[:, 2],
                odd_indices[:, 0],
                odd_indices[:, 1],
                odd_indices[:, 2],
            ]
        ).reshape((-1, 3))
        vertex_count = len(next(iter(masks.values())))
    return masks, np.asarray(topology, dtype=np.int64)


def evaluate_release_a_gate(
    bundle: NasalTextureObservationBundle,
    *,
    min_per_side: int = 6,
    min_regions_per_side: int = 2,
    max_median_reprojection_px: float = 1.5,
    max_p90_reprojection_px: float = 2.5,
    cluster_radius_px: float = 18.0,
    min_spatial_diameter_px: float = 18.0,
) -> dict[str, Any]:
    trusted_by_side = {
        side: tuple(bundle.by_side[side])
        for side in ("subject-left", "subject-right")
    }
    counts = {side: len(values) for side, values in trusted_by_side.items()}
    regions = {
        side: sorted({value.pair_match.semantic_region for value in values})
        for side, values in trusted_by_side.items()
    }
    reprojection = np.asarray(
        [
            error
            for observation in bundle.trusted
            for error in observation.reprojection_errors_px.values()
        ],
        dtype=np.float64,
    )
    p50 = float(np.percentile(reprojection, 50.0)) if len(reprojection) else float("inf")
    p90 = float(np.percentile(reprojection, 90.0)) if len(reprojection) else float("inf")
    cluster_fraction: dict[str, float] = {}
    spatial_diameter: dict[str, float] = {}
    unique_anchor_count: dict[str, int] = {}
    for side, values in trusted_by_side.items():
        points = np.asarray(
            [value.pair_match.front_pixel for value in values],
            dtype=np.float64,
        ).reshape(-1, 2)
        if not len(points):
            cluster_fraction[side] = 1.0
            spatial_diameter[side] = 0.0
            unique_anchor_count[side] = 0
            continue
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        spatial_diameter[side] = float(np.max(distances))
        cluster_fraction[side] = float(
            np.max(np.count_nonzero(distances <= float(cluster_radius_px), axis=1))
            / len(points)
        )
        anchor_keys = {
            int(value.pair_match.diagnostics["baseline_vertex_index"])
            if "baseline_vertex_index" in value.pair_match.diagnostics
            else tuple(np.rint(value.pair_match.front_pixel).astype(int))
            for value in values
        }
        unique_anchor_count[side] = len(anchor_keys)
    gates = {
        "minimum_trusted_per_side": all(value >= int(min_per_side) for value in counts.values()),
        "semantic_region_coverage": all(
            len(value) >= int(min_regions_per_side) for value in regions.values()
        ),
        "median_reprojection": p50 <= float(max_median_reprojection_px),
        "p90_reprojection": p90 <= float(max_p90_reprojection_px),
        "spatial_dispersion": all(
            spatial_diameter[side] >= float(min_spatial_diameter_px)
            and unique_anchor_count[side] >= int(min_per_side)
            for side in trusted_by_side
        ),
    }
    passed = bool(all(gates.values()))
    return {
        "passed": passed,
        "status": "ready_for_geometry" if passed else "insufficient_texture_evidence",
        "gates": gates,
        "trusted_by_side": counts,
        "regions_by_side": regions,
        "reprojection_error_px": {"p50": p50, "p90": p90},
        "max_cluster_fraction_by_side": cluster_fraction,
        "spatial_diameter_px_by_side": spatial_diameter,
        "unique_anchor_count_by_side": unique_anchor_count,
        "thresholds": {
            "min_per_side": int(min_per_side),
            "min_regions_per_side": int(min_regions_per_side),
            "max_median_reprojection_px": float(max_median_reprojection_px),
            "max_p90_reprojection_px": float(max_p90_reprojection_px),
            "cluster_radius_px": float(cluster_radius_px),
            "min_spatial_diameter_px": float(min_spatial_diameter_px),
        },
    }


def _project_vertices(
    vertices: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera = np.asarray(vertices, dtype=np.float64) @ np.asarray(rotation).T
    camera += np.asarray(translation, dtype=np.float64).reshape(3)
    homogeneous = camera @ np.asarray(intrinsics, dtype=np.float64).T
    pixels = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-12)
    return pixels, camera[:, 2]


def _rasterize_vertex_support(
    projected: np.ndarray,
    faces: np.ndarray,
    vertex_mask: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    topology = np.asarray(faces, dtype=np.int64)
    supported_faces = topology[np.all(np.asarray(vertex_mask, dtype=bool)[topology], axis=1)]
    for triangle in supported_faces:
        polygon = np.rint(projected[triangle]).astype(np.int32)
        if np.isfinite(projected[triangle]).all():
            cv2.fillConvexPoly(mask, polygon, 1, lineType=cv2.LINE_AA)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, kernel, iterations=1)


def _provenance(
    semantic_view: str,
    source_size: tuple[int, int],
    work_size: tuple[int, int],
) -> NasalCoordinateProvenance:
    sx = work_size[0] / float(source_size[0])
    sy = work_size[1] / float(source_size[1])
    return NasalCoordinateProvenance(
        semantic_view=semantic_view,
        source_size=source_size,
        work_size=work_size,
        source_to_work=np.asarray(
            [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        undistorted=True,
    )


def _loftr_seed_pairs(
    matcher: Any,
    device: Any,
    images: Mapping[str, np.ndarray],
    confidence_maps: Mapping[str, Any],
    provenance: Mapping[str, NasalCoordinateProvenance],
    side_view: str,
) -> tuple[NasalPairMatch, ...]:
    front_points, side_points, scores = run_loftr_matches(
        matcher,
        device,
        cv2.cvtColor(images["front"], cv2.COLOR_RGB2BGR),
        cv2.cvtColor(images[side_view], cv2.COLOR_RGB2BGR),
    )
    pairs = []
    for front, side, score in zip(front_points, side_points, scores):
        fx, fy = np.rint(front).astype(int)
        sx, sy = np.rint(side).astype(int)
        if (
            score < 0.45
            or fx < 0
            or fy < 0
            or sx < 0
            or sy < 0
            or fy >= images["front"].shape[0]
            or fx >= images["front"].shape[1]
            or sy >= images[side_view].shape[0]
            or sx >= images[side_view].shape[1]
            or confidence_maps["front"].final_confidence[fy, fx] <= 0.0
            or confidence_maps[side_view].final_confidence[sy, sx] <= 0.0
        ):
            continue
        pairs.append(
            NasalPairMatch(
                side_view=side_view,
                front_pixel=front,
                side_pixel=side,
                confidence=float(score),
                semantic_region="protected_nasal_skin_seed",
                source_matcher="loftr_seed_only",
                provenance_by_view={
                    "front": provenance["front"],
                    side_view: provenance[side_view],
                },
                diagnostics={"loftr_confidence": float(score)},
            )
        )
    return tuple(pairs)


def _face_and_bary_for_vertex(
    vertex_index: int,
    faces: np.ndarray,
    allowed_faces: np.ndarray,
) -> tuple[int, np.ndarray] | None:
    allowed = np.asarray(allowed_faces, dtype=np.int64)
    topology = np.asarray(faces, dtype=np.int64)
    for face_index in allowed:
        positions = np.flatnonzero(topology[face_index] == int(vertex_index))
        if len(positions):
            bary = np.zeros(3, dtype=np.float64)
            bary[int(positions[0])] = 1.0
            return int(face_index), bary
    return None


def _build_registration(
    loftr_by_side: Mapping[str, Sequence[NasalPairMatch]],
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    region_masks: Mapping[str, np.ndarray],
    views_by_semantic: Mapping[str, Any],
    rig: Any,
    provenance: Mapping[str, NasalCoordinateProvenance],
) -> Any:
    if set(region_masks) != {
        "bridge",
        "peri_nasal_skin",
        "subject_left_peri_nasal",
        "subject_right_peri_nasal",
    }:
        raise ValueError("registration region masks are incomplete")
    region_masks = {
        name: np.asarray(mask, dtype=bool)
        for name, mask in region_masks.items()
    }
    if any(mask.shape != (len(candidate_vertices),) for mask in region_masks.values()):
        raise ValueError("registration region masks must match locked v10 vertices")
    bridge = region_masks["bridge"]
    left_peri = region_masks["subject_left_peri_nasal"]
    right_peri = region_masks["subject_right_peri_nasal"]
    face_map = {
        name: np.flatnonzero(np.any(mask[np.asarray(faces, dtype=np.int64)], axis=1))
        for name, mask in region_masks.items()
    }
    surface = BaselineNasalRegistrationSurface(
        vertices=candidate_vertices,
        faces=faces,
        protected_face_indices_by_region=face_map,
        rig=rig,
        provenance_by_view=provenance,
        model_to_front_rotation=views_by_semantic["front"].R_model_to_camera,
        model_to_front_translation=views_by_semantic["front"].t_model_to_camera,
        rig_view_by_semantic=SEMANTIC_TO_RIG_VIEW,
    )
    projection_matrices = surface.projection_matrices_by_view
    samples: list[ViewRegistrationSample] = []
    front_view = views_by_semantic["front"]
    for side_view, pairs in loftr_by_side.items():
        bundle = triangulate_nasal_pair_matches(
            pairs,
            rig,
            observed_work_size_by_semantic_view={
                semantic: value.work_size for semantic, value in provenance.items()
            },
            rig_view_by_semantic=SEMANTIC_TO_RIG_VIEW,
            config=NasalTextureObservationConfig(min_match_confidence=0.45),
        )
        for observation in bundle.trusted:
            point_front = observation.point_reference_m
            point_model = (
                np.asarray(front_view.R_model_to_camera).T
                @ (point_front - np.asarray(front_view.t_model_to_camera))
            )
            eligible = bridge | (left_peri if side_view == "subject-left" else right_peri)
            indices = np.flatnonzero(eligible)
            if not len(indices):
                continue
            nearest = int(indices[np.argmin(np.linalg.norm(candidate_vertices[indices] - point_model, axis=1))])
            region = "bridge" if bridge[nearest] else (
                "subject_left_peri_nasal"
                if side_view == "subject-left"
                else "subject_right_peri_nasal"
            )
            attached = _face_and_bary_for_vertex(nearest, faces, face_map[region])
            if attached is None:
                continue
            face_index, bary = attached
            for semantic_view, observed_pixel in (
                ("front", observation.pair_match.front_pixel),
                (side_view, observation.pair_match.side_pixel),
            ):
                projected_h = projection_matrices[semantic_view] @ np.append(
                    candidate_vertices[nearest], 1.0
                )
                projected = projected_h[:2] / projected_h[2]
                samples.append(
                    ViewRegistrationSample(
                        semantic_view=semantic_view,
                        projected_pixel=projected,
                        observed_pixel=observed_pixel,
                        confidence=observation.pair_match.confidence,
                        semantic_region=region,
                        source="protected_surface_correspondence",
                        baseline_face_index=face_index,
                        baseline_bary_coords=bary,
                        matched_point_3d=point_model,
                    )
                )
    return estimate_fixed_nasal_view_offsets(
        tuple(samples),
        surface,
        NasalViewRegistrationConfig(min_samples_per_view=6),
    )


def _visible_seed_vertices(
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    region_mask: np.ndarray,
    front_view: Any,
    side_view: Any,
    front_k: np.ndarray,
    side_k: np.ndarray,
    *,
    max_count: int = 90,
) -> np.ndarray:
    indices = np.flatnonzero(region_mask)
    if not len(indices):
        return indices
    front_pixels, front_depth = _project_vertices(
        candidate_vertices, front_view.R_model_to_camera, front_view.t_model_to_camera, front_k
    )
    side_pixels, side_depth = _project_vertices(
        candidate_vertices, side_view.R_model_to_camera, side_view.t_model_to_camera, side_k
    )
    front_faces, _front_bary, front_hit_depth = sparse_zbuffer_attachments(
        candidate_vertices,
        faces,
        front_view.R_model_to_camera,
        front_view.t_model_to_camera,
        front_k,
        front_pixels[indices],
    )
    side_faces, _side_bary, side_hit_depth = sparse_zbuffer_attachments(
        candidate_vertices,
        faces,
        side_view.R_model_to_camera,
        side_view.t_model_to_camera,
        side_k,
        side_pixels[indices],
    )
    visible = (
        (front_faces >= 0)
        & (side_faces >= 0)
        & (np.abs(front_hit_depth - front_depth[indices]) <= 0.006)
        & (np.abs(side_hit_depth - side_depth[indices]) <= 0.006)
    )
    candidates = indices[visible]
    if len(candidates) <= max_count:
        return candidates
    chosen = []
    order = candidates[np.argsort(front_pixels[candidates, 1], kind="stable")]
    for index in order:
        if not chosen:
            chosen.append(int(index))
        else:
            front_distance = np.linalg.norm(front_pixels[chosen] - front_pixels[index], axis=1)
            side_distance = np.linalg.norm(side_pixels[chosen] - side_pixels[index], axis=1)
            if np.min(front_distance) >= 3.0 and np.min(side_distance) >= 3.0:
                chosen.append(int(index))
        if len(chosen) >= max_count:
            break
    return np.asarray(chosen, dtype=np.int64)


def _semantic_regions_for_vertices(
    vertices: np.ndarray,
    indices: np.ndarray,
    semantic_frame: Any,
) -> dict[int, str]:
    local = (
        np.asarray(vertices)[indices] - semantic_frame.origin
    ) @ semantic_frame.matrix.T
    if not len(local):
        return {}
    low, high = np.percentile(local[:, 1], [33.0, 67.0])
    result = {}
    for index, height in zip(indices, local[:, 1]):
        if height >= high:
            region = "soft_triangle"
        elif height <= low:
            region = "alar_groove"
        else:
            region = "alar_dome"
        result[int(index)] = region
    return result


def _local_projection_jacobians(
    faces: np.ndarray,
    front_projection: np.ndarray,
    side_projection: np.ndarray,
    indices: np.ndarray,
) -> dict[int, np.ndarray]:
    topology = np.asarray(faces, dtype=np.int64)
    incident: dict[int, set[int]] = {int(index): set() for index in indices}
    requested = set(incident)
    for triangle in topology:
        members = requested.intersection(int(value) for value in triangle)
        if not members:
            continue
        neighbors = {int(value) for value in triangle}
        for member in members:
            incident[member].update(neighbors - {member})
    result: dict[int, np.ndarray] = {}
    for index in indices:
        key = int(index)
        neighbors = np.asarray(sorted(incident[key]), dtype=np.int64)
        jacobian = np.eye(2, dtype=np.float64)
        if len(neighbors) >= 3:
            front_delta = front_projection[neighbors] - front_projection[key]
            side_delta = side_projection[neighbors] - side_projection[key]
            valid = (
                np.isfinite(front_delta).all(axis=1)
                & np.isfinite(side_delta).all(axis=1)
                & (np.linalg.norm(front_delta, axis=1) > 0.1)
            )
            if np.count_nonzero(valid) >= 3:
                mapping, _residuals, rank, _singular = np.linalg.lstsq(
                    front_delta[valid],
                    side_delta[valid],
                    rcond=None,
                )
                candidate = mapping.T
                scales = np.linalg.svd(candidate, compute_uv=False)
                if (
                    rank == 2
                    and np.isfinite(candidate).all()
                    and float(scales[-1]) >= 0.15
                    and float(scales[0]) <= 6.0
                ):
                    jacobian = candidate
        result[key] = jacobian
    return result


def point_to_supported_surface_distances(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_mask: np.ndarray,
) -> np.ndarray:
    """Compute exact point-to-triangle distances on the supported v10 surface."""
    import trimesh

    query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mesh_vertices = np.asarray(vertices, dtype=np.float64)
    topology = np.asarray(faces, dtype=np.int64)
    support = np.asarray(vertex_mask, dtype=bool)
    if support.shape != (len(mesh_vertices),):
        raise ValueError("surface support mask must match mesh vertices")
    supported_faces = topology[np.all(support[topology], axis=1)]
    if len(supported_faces) == 0:
        raise ValueError("surface support mask contains no complete triangles")
    triangles = mesh_vertices[supported_faces]
    distances = []
    for point in query:
        repeated = np.broadcast_to(point, (len(triangles), 3))
        closest = trimesh.triangles.closest_point(triangles, repeated)
        distances.append(float(np.min(np.linalg.norm(closest - point, axis=1))))
    return np.asarray(distances, dtype=np.float64)


def _subdivide_alar_basis_vectors(
    baseline: Any,
    alar_basis: Any,
    expected_faces: np.ndarray,
) -> np.ndarray:
    base_vertices, base_faces, _base_uv, _base_uv_faces = _subdivide_candidate(
        baseline,
        baseline.vertices,
    )
    if not np.array_equal(np.asarray(base_faces, dtype=np.int64), expected_faces):
        raise RuntimeError("subdivided basis topology disagrees with locked v10")
    vectors = []
    for mode in np.asarray(alar_basis.vectors, dtype=np.float64):
        mode_vertices, mode_faces, _uv, _uv_faces = _subdivide_candidate(
            baseline,
            np.asarray(baseline.vertices, dtype=np.float64) + mode,
        )
        if not np.array_equal(np.asarray(mode_faces, dtype=np.int64), expected_faces):
            raise RuntimeError("subdivided alar mode topology changed")
        vectors.append(np.asarray(mode_vertices, dtype=np.float64) - base_vertices)
    return np.asarray(vectors, dtype=np.float64)


def _relative_artifact_paths(value: Any, root: Path) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _relative_artifact_paths(child, root)
            for key, child in value.items()
        }
    return Path(value).relative_to(root).as_posix()


def _fit_and_export_roma_candidate(
    output: Path,
    source_v10: Path,
    observations: NasalTextureObservationBundle,
    baseline: Any,
    alar_basis: Any,
    locked_vertices: np.ndarray,
    locked_faces: np.ndarray,
    alar_support: np.ndarray,
    alar_protected: np.ndarray,
    front_view: Any,
    front_intrinsics: np.ndarray,
    *,
    viewer_template: str | Path | None = None,
) -> dict[str, Any]:
    trusted = tuple(observations.trusted)
    front_pixels = np.asarray(
        [value.pair_match.front_pixel for value in trusted],
        dtype=np.float64,
    )
    face_indices, barycentric, _depth = sparse_zbuffer_attachments(
        locked_vertices,
        locked_faces,
        front_view.R_model_to_camera,
        front_view.t_model_to_camera,
        front_intrinsics,
        front_pixels,
    )
    valid = face_indices >= 0
    if np.any(valid):
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] &= np.all(
            np.asarray(alar_support, dtype=bool)[
                np.asarray(locked_faces, dtype=np.int64)[face_indices[valid_indices]]
            ],
            axis=1,
        )
    if np.count_nonzero(valid) < 6:
        raise RuntimeError("RoMa observations do not attach to six alar triangles")
    trusted_valid = tuple(value for value, keep in zip(trusted, valid) if keep)
    target_front = np.asarray(
        [value.point_reference_m for value in trusted_valid],
        dtype=np.float64,
    )
    rotation = np.asarray(front_view.R_model_to_camera, dtype=np.float64)
    translation = np.asarray(front_view.t_model_to_camera, dtype=np.float64)
    target_model = (target_front - translation) @ rotation
    valid_triangles = np.asarray(locked_faces, dtype=np.int64)[face_indices[valid]]
    attached_model = np.sum(
        np.asarray(locked_vertices, dtype=np.float64)[valid_triangles]
        * barycentric[valid, :, None],
        axis=1,
    )
    raw_offsets = target_model - attached_model
    median_offset = np.median(raw_offsets, axis=0)
    centered_offsets = raw_offsets - median_offset
    raw_offset_diagnostics = {
        "median_mm": (median_offset * 1000.0).tolist(),
        "raw_rmse_mm": float(
            np.sqrt(np.mean(np.sum(raw_offsets**2, axis=1))) * 1000.0
        ),
        "translation_centered_rmse_mm": float(
            np.sqrt(np.mean(np.sum(centered_offsets**2, axis=1))) * 1000.0
        ),
        "axis_std_mm": (np.std(centered_offsets, axis=0) * 1000.0).tolist(),
    }
    weights = np.asarray(
        [max(float(value.weight), 1e-4) for value in trusted_valid],
        dtype=np.float64,
    )
    high_resolution_basis = _subdivide_alar_basis_vectors(
        baseline,
        alar_basis,
        np.asarray(locked_faces, dtype=np.int64),
    )
    movable = np.asarray(alar_support, dtype=bool) & ~np.asarray(
        alar_protected,
        dtype=bool,
    )
    high_resolution_basis[:, ~movable, :] = 0.0
    fit = fit_roma_nasal_surface(
        locked_vertices,
        locked_faces,
        high_resolution_basis,
        face_indices[valid],
        barycentric[valid],
        target_model,
        weights,
    )
    if not fit.success:
        raise RuntimeError(f"RoMa nasal fitting failed: {fit.optimizer_message}")
    if not np.array_equal(
        fit.candidate_vertices[~movable],
        np.asarray(locked_vertices)[~movable],
    ):
        raise RuntimeError("RoMa fitting changed vertices outside movable alar support")

    source_obj = source_v10 / "meshes" / "face_mesh.obj"
    _source_vertices, source_faces, uv_vertices, uv_faces = load_mesh_obj(source_obj)
    if not np.array_equal(np.asarray(source_faces, dtype=np.int64), locked_faces):
        raise RuntimeError("v10 UV topology disagrees with locked geometry")
    quality = validate_minimal_nasal_candidate(
        parameters=fit.coefficients,
        baseline_vertices=locked_vertices,
        candidate_vertices=fit.candidate_vertices,
        baseline_faces=locked_faces,
        candidate_faces=locked_faces,
        baseline_uv_vertices=uv_vertices,
        candidate_uv_vertices=uv_vertices,
        baseline_uv_faces=uv_faces,
        candidate_uv_faces=uv_faces,
    )
    if not quality["passed"]:
        raise RuntimeError(
            "RoMa candidate failed geometry quality: "
            + ", ".join(str(value) for value in quality["issues"])
            + "; coefficients="
            + np.array2string(fit.coefficients, precision=4, separator=",")
            + f"; rmse_mm={fit.initial_rmse_mm:.4f}->{fit.final_rmse_mm:.4f}"
            + "; target_offset="
            + json.dumps(raw_offset_diagnostics, ensure_ascii=True)
            + "; quality_comparison="
            + json.dumps(
                quality.get("quality", {}).get("comparison", {}),
                ensure_ascii=True,
            )
            + "; face_flips="
            + json.dumps(quality.get("face_flips", {}), ensure_ascii=True)
        )

    meshes = output / "meshes"
    textures = output / "textures"
    meshes.mkdir(parents=True, exist_ok=True)
    textures.mkdir(parents=True, exist_ok=True)
    baseline_textured_source = source_v10 / "meshes" / "candidate_textured.glb"
    baseline_geometry_source = source_v10 / "meshes" / "candidate_geometry.glb"
    texture_source = source_v10 / "textures" / "albedo_white.png"
    baseline_textured = meshes / "baseline_textured.glb"
    baseline_geometry = meshes / "baseline_geometry.glb"
    texture = textures / "albedo_white.png"
    shutil.copy2(baseline_textured_source, baseline_textured)
    shutil.copy2(baseline_geometry_source, baseline_geometry)
    shutil.copy2(texture_source, texture)
    candidate_obj = meshes / "candidate.obj"
    candidate_geometry = meshes / "candidate_geometry.glb"
    candidate_textured = meshes / "candidate_textured.glb"
    export_mesh_obj(
        fit.candidate_vertices,
        locked_faces,
        uv_vertices,
        uv_faces,
        candidate_obj,
    )
    export_mesh_glb(
        fit.candidate_vertices,
        locked_faces,
        uv_vertices,
        uv_faces,
        candidate_geometry,
    )
    _export_with_baseline_texture(
        mesh_path=candidate_obj,
        texture_path=texture,
        output_path=candidate_textured,
    )
    glb_validation = _validate_embedded_textured_glb(candidate_textured)
    viewer = _write_viewer(
        template=Path(viewer_template).resolve() if viewer_template else None,
        baseline_glb=baseline_textured,
        candidate_glb=candidate_textured,
        output=output / "roma_nasal_compare.html",
        dataset_label="captures_20260612_135253 | RoMa fixed-rig nasal v1",
    )
    renders = render_nasal_geometry_screenshots(
        baseline_glb=baseline_textured,
        candidate_glb=candidate_textured,
        output_dir=output / "debug" / "roma_model_renders",
    )
    return {
        "fit": fit.to_dict(),
        "quality": quality,
        "attachment_count": int(np.count_nonzero(valid)),
        "movable_vertex_count": int(np.count_nonzero(movable)),
        "baseline_textured_glb": baseline_textured.relative_to(output).as_posix(),
        "baseline_geometry_glb": baseline_geometry.relative_to(output).as_posix(),
        "candidate_obj": candidate_obj.relative_to(output).as_posix(),
        "candidate_geometry_glb": candidate_geometry.relative_to(output).as_posix(),
        "candidate_textured_glb": candidate_textured.relative_to(output).as_posix(),
        "texture": texture.relative_to(output).as_posix(),
        "viewer": viewer.relative_to(output).as_posix(),
        "renders": _relative_artifact_paths(renders, output),
        "glb_validation": glb_validation,
    }


def run_nasal_texture_observation_audit(
    capture_dir: str | Path,
    source_v10: str | Path,
    output: str | Path,
    *,
    rig_calibration: str | Path,
    work_size: tuple[int, int] = (640, 480),
    expected_v10_obj_sha256: str = DEFAULT_V10_OBJ_SHA256,
    expected_v10_report_sha256: str = DEFAULT_V10_REPORT_SHA256,
    expected_rig_sha256: str = DEFAULT_RIG_SHA256,
    matcher_loader: Any = load_loftr_matcher,
    roma_pair_runner: Any = run_roma_nasal_pair,
    roma_python: str | Path = DEFAULT_ROMA_PYTHON,
    roma_torch_home: str | Path = DEFAULT_ROMA_TORCH_HOME,
    roma_worker: str | Path = DEFAULT_ROMA_WORKER,
) -> Path:
    """Audit whether v10 has enough cross-view skin evidence for Release B."""
    captures = Path(capture_dir).resolve()
    source = Path(source_v10).resolve()
    target = Path(output).resolve()
    calibration = Path(rig_calibration).resolve()
    if not captures.is_dir():
        raise FileNotFoundError(f"capture directory does not exist: {captures}")
    if not source.is_dir():
        raise FileNotFoundError(f"v10 source does not exist: {source}")
    validate_separate_output(target, source, captures)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {target}")
    obj_path = source / "meshes" / "face_mesh.obj"
    report_path = source / "alar_surface_report.json"
    locked_hashes = {
        "v10_obj": verify_hash_locked_input(obj_path, expected_v10_obj_sha256, "v10 OBJ"),
        "v10_report": verify_hash_locked_input(report_path, expected_v10_report_sha256, "v10 report"),
        "rig": verify_hash_locked_input(calibration, expected_rig_sha256, "fixed rig"),
    }
    source_hashes = file_tree_hashes(source)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.partial-",
            dir=str(target.parent),
        )
    )
    immutable_inputs_verified = False

    try:
        rig = load_profile_rig(calibration, max_stereo_rms_px=10.0)
        raw_images, capture_paths = _load_capture_images(captures, rig)
        from src import config as cfg
        from src.module0_intrinsics import undistort_images_with_calibration

        undistorted, new_intrinsics = undistort_images_with_calibration(
            raw_images,
            calibration_path=calibration,
            alpha=float(cfg.UNDISTORT_ALPHA),
        )
        if new_intrinsics is None:
            raise RuntimeError("undistortion did not return calibrated new K")
        observation_rig = build_undistorted_observation_rig(rig, new_intrinsics)
        images = {
            "front": cv2.resize(undistorted["front"], work_size, interpolation=cv2.INTER_AREA),
            "subject-left": cv2.resize(undistorted["left"], work_size, interpolation=cv2.INTER_AREA),
            "subject-right": cv2.resize(undistorted["right"], work_size, interpolation=cv2.INTER_AREA),
        }
        provenance = {
            semantic: _provenance(
                semantic,
                observation_rig.cameras_by_view[rig_view].image_size,
                work_size,
            )
            for semantic, rig_view in SEMANTIC_TO_RIG_VIEW.items()
        }

        source_a2 = Path(report["baseline_contract"]["source_a2"]).resolve()
        a2_report = json.loads((source_a2 / "nasal_base_report.json").read_text(encoding="utf-8"))
        source_v4 = Path(a2_report["source_v4"]["path"]).resolve()
        source_observations = load_nasal_observation_bundle(source_v4 / "nasal_observations.json")
        baseline, source_views, _a2_report, _source_v4 = _load_a2_low_resolution(
            source_a2,
            source_observations,
        )
        views = build_model_projection_views(
            source_observations,
            baseline.front_rotation,
            baseline.front_translation,
        )
        views_by_semantic = {view.name: view for view in views}
        projection_basis = build_nasal_semantic_basis(
            baseline.vertices,
            baseline.faces,
            baseline.landmark_triangles,
            baseline.landmark_barycentric,
            views_by_semantic["front"].R_model_to_camera,
        )
        alar_basis = build_alar_surface_basis(
            baseline.vertices,
            baseline.faces,
            baseline.landmark_triangles,
            baseline.landmark_barycentric,
            views_by_semantic["front"].R_model_to_camera,
            projection_basis=projection_basis,
        )
        coefficients = np.asarray(report["optimization"]["coefficients"], dtype=np.float64)
        low_resolution_candidate_vertices = apply_alar_surface_basis(
            baseline.vertices,
            alar_basis,
            coefficients,
        )
        candidate_vertices, candidate_faces, locked_surface_info = (
            load_locked_v10_surface(
                obj_path,
                baseline,
                low_resolution_candidate_vertices,
            )
        )
        transferred_masks, transferred_faces = subdivide_low_resolution_vertex_masks(
            baseline.faces,
            {
                "broad_support": projection_basis.support_mask,
                "nose_bridge": projection_basis.region_masks["nose_bridge"],
                "nose_tip": projection_basis.region_masks["nose_tip"],
                "alar_support": alar_basis.support_mask,
                "alar_protected": alar_basis.protected_mask,
                "subject_left_alar": alar_basis.region_masks["subject_left_alar"],
                "subject_right_alar": alar_basis.region_masks["subject_right_alar"],
            },
        )
        if not np.array_equal(transferred_faces, candidate_faces):
            raise RuntimeError(
                "semantic mask subdivision topology disagrees with locked v10 OBJ"
            )
        semantic_transfer_info = {
            "method": "exact_two_iteration_loop_topology",
            "topology_verified": True,
            "vertex_count": int(len(candidate_vertices)),
            "selected_vertices_by_mask": {
                name: int(np.count_nonzero(mask))
                for name, mask in transferred_masks.items()
            },
        }
        intrinsics = {
            semantic: scale_intrinsics(
                observation_rig.cameras_by_view[rig_view].K,
                observation_rig.cameras_by_view[rig_view].image_size,
                work_size,
            )
            for semantic, rig_view in SEMANTIC_TO_RIG_VIEW.items()
        }
        projections = {
            semantic: _project_vertices(
                candidate_vertices,
                views_by_semantic[semantic].R_model_to_camera,
                views_by_semantic[semantic].t_model_to_camera,
                intrinsics[semantic],
            )[0]
            for semantic in SEMANTIC_TO_RIG_VIEW
        }
        broad_support = transferred_masks["broad_support"]
        projected_support = {
            semantic: _rasterize_vertex_support(
                projections[semantic],
                candidate_faces,
                broad_support,
                images[semantic].shape[:2],
            )
            for semantic in SEMANTIC_TO_RIG_VIEW
        }
        confidence_maps = {}
        preprocess_root = source_v4 / "debug" / "preprocess"
        for semantic, rig_view in SEMANTIC_TO_RIG_VIEW.items():
            nose_path = preprocess_root / f"{rig_view}_nose_mask.png"
            face_path = preprocess_root / f"{rig_view}_face_mask.png"
            nose = restore_mask_to_work_frame(
                cv2.imread(str(nose_path), cv2.IMREAD_GRAYSCALE),
                target_size=work_size,
            )
            face = restore_mask_to_work_frame(
                cv2.imread(str(face_path), cv2.IMREAD_GRAYSCALE),
                target_size=work_size,
            )
            confidence_maps[semantic] = build_nasal_texture_confidence_maps(
                images[semantic],
                nose,
                face_mask=face,
                projected_support=projected_support[semantic],
            )

        matcher, device = matcher_loader()
        loftr_by_side = {
            side: _loftr_seed_pairs(
                matcher,
                device,
                images,
                confidence_maps,
                provenance,
                side,
            )
            for side in ("subject-left", "subject-right")
        }
        locked_local = (
            candidate_vertices - projection_basis.semantic_frame.origin
        ) @ projection_basis.semantic_frame.matrix.T
        bridge = transferred_masks["nose_bridge"]
        peri = broad_support & ~transferred_masks["alar_support"]
        peri &= ~transferred_masks["nose_tip"]
        registration_masks = {
            "bridge": bridge,
            "peri_nasal_skin": peri,
            "subject_left_peri_nasal": peri & (locked_local[:, 0] >= 0.0),
            "subject_right_peri_nasal": peri & (locked_local[:, 0] < 0.0),
        }
        registration = _build_registration(
            loftr_by_side,
            candidate_vertices,
            candidate_faces,
            registration_masks,
            views_by_semantic,
            observation_rig,
            provenance,
        )
        del matcher
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        match_results: dict[str, NasalEpipolarMatchResult] = {}
        roma_metadata: dict[str, Any] = {}
        for side in ("subject-left", "subject-right"):
            side_region = np.asarray(
                transferred_masks[
                    "subject_left_alar" if side == "subject-left" else "subject_right_alar"
                ],
                dtype=bool,
            )
            indices = _visible_seed_vertices(
                candidate_vertices,
                candidate_faces,
                side_region,
                views_by_semantic["front"],
                views_by_semantic[side],
                intrinsics["front"],
                intrinsics[side],
            )
            region_by_index = _semantic_regions_for_vertices(
                candidate_vertices,
                indices,
                alar_basis.semantic_frame,
            )
            jacobian_by_index = _local_projection_jacobians(
                candidate_faces,
                projections["front"],
                projections[side],
                indices,
            )
            seeds = tuple(
                NasalEpipolarSeed(
                    front_pixel=projections["front"][index]
                    + registration.offsets_by_view["front"],
                    predicted_side_pixel=projections[side][index]
                    + registration.offsets_by_view[side],
                    semantic_region=region_by_index[int(index)],
                    baseline_vertex_index=int(index),
                    side_from_front_jacobian=jacobian_by_index[int(index)],
                )
                for index in indices
            )
            roma_batch = roma_pair_runner(
                images["front"],
                images[side],
                projected_support["front"],
                projected_support[side],
                python_executable=roma_python,
                worker_script=roma_worker,
                torch_home=roma_torch_home,
            )
            fundamental = fixed_rig_fundamental(
                observation_rig,
                side_view=side,
                provenance_by_view={
                    "front": provenance["front"],
                    side: provenance[side],
                },
                rig_view_by_semantic=SEMANTIC_TO_RIG_VIEW,
            )
            match_results[side] = select_roma_nasal_matches(
                roma_batch,
                side_view=side,
                seeds=seeds,
                front_confidence=confidence_maps["front"],
                side_confidence=confidence_maps[side],
                provenance_by_view={
                    "front": provenance["front"],
                    side: provenance[side],
                },
                fundamental=fundamental,
            )
            roma_metadata[side] = {
                **dict(roma_batch.metadata),
                "dense_candidate_count": int(len(roma_batch.front_pixels)),
                "selected_match_count": int(len(match_results[side].matches)),
                "rejected_seed_count": int(len(match_results[side].rejected)),
                "front_crop": roma_batch.front_crop.to_dict(),
                "side_crop": roma_batch.side_crop.to_dict(),
            }

        side_bundles = {
            side: triangulate_nasal_pair_matches(
                result.matches,
                observation_rig,
                observed_work_size_by_semantic_view={
                    semantic: value.work_size for semantic, value in provenance.items()
                },
                rig_view_by_semantic=SEMANTIC_TO_RIG_VIEW,
                config=NasalTextureObservationConfig(min_match_confidence=0.10),
            )
            for side, result in match_results.items()
        }
        observations = NasalTextureObservationBundle(
            trusted=tuple(
                value
                for side in ("subject-left", "subject-right")
                for value in side_bundles[side].trusted
            ),
            rejected=tuple(
                value
                for side in ("subject-left", "subject-right")
                for value in side_bundles[side].rejected
            ),
            metadata={
                "source": "roma_dense_with_fixed_rig_validation",
                "geometry_modified": False,
            },
        )
        front_view = views_by_semantic["front"]
        observation_model_points = np.asarray(
            [
                np.asarray(front_view.R_model_to_camera).T
                @ (value.point_reference_m - np.asarray(front_view.t_model_to_camera))
                for value in observations.trusted
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        model_distances = point_to_supported_surface_distances(
            observation_model_points,
            candidate_vertices,
            candidate_faces,
            broad_support,
        ).tolist()
        gate = evaluate_release_a_gate(observations)
        candidate_artifacts = None
        if gate["passed"]:
            candidate_artifacts = _fit_and_export_roma_candidate(
                staging,
                source,
                observations,
                baseline,
                alar_basis,
                candidate_vertices,
                candidate_faces,
                transferred_masks["alar_support"],
                transferred_masks["alar_protected"],
                views_by_semantic["front"],
                intrinsics["front"],
            )
        for semantic, maps in confidence_maps.items():
            cv2.imwrite(
                str(staging / f"confidence_{semantic}.png"),
                np.uint8(np.clip(maps.final_confidence, 0.0, 1.0) * 255),
            )
        staged_report = write_nasal_texture_observation_report(
            staging,
            images_by_view=images,
            matches_by_side=match_results,
            observations=observations,
            registration=registration,
            gate=gate,
            model_distances_m=model_distances,
            metadata={
                "capture_dir": str(captures),
                "capture_paths": {key: str(value) for key, value in capture_paths.items()},
                "source_v10": str(source),
                "rig_calibration": str(calibration),
                "locked_hashes": locked_hashes,
                "locked_v10_surface": locked_surface_info,
                "semantic_label_transfer": semantic_transfer_info,
                "work_size": list(work_size),
                "geometry_modified": candidate_artifacts is not None,
                "deformed_mesh_created": candidate_artifacts is not None,
                "candidate_artifacts": candidate_artifacts,
                "loftr_seed_counts": {side: len(value) for side, value in loftr_by_side.items()},
                "roma": roma_metadata,
                "model_seed_counts": {
                    side: len(result.matches) + len(result.rejected)
                    for side, result in match_results.items()
                },
            },
        )
        assert_file_tree_unchanged(source, source_hashes)
        verify_hash_locked_input(obj_path, expected_v10_obj_sha256, "v10 OBJ after audit")
        verify_hash_locked_input(
            report_path,
            expected_v10_report_sha256,
            "v10 report after audit",
        )
        verify_hash_locked_input(calibration, expected_rig_sha256, "fixed rig after audit")
        immutable_inputs_verified = True
        relative_report = staged_report.relative_to(staging)
        staging.rename(target)
        return target / relative_report
    finally:
        try:
            if not immutable_inputs_verified:
                assert_file_tree_unchanged(source, source_hashes)
                verify_hash_locked_input(
                    obj_path,
                    expected_v10_obj_sha256,
                    "v10 OBJ after failed audit",
                )
                verify_hash_locked_input(
                    report_path,
                    expected_v10_report_sha256,
                    "v10 report after failed audit",
                )
                verify_hash_locked_input(
                    calibration,
                    expected_rig_sha256,
                    "fixed rig after failed audit",
                )
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-v10", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rig-calibration", type=Path, required=True)
    parser.add_argument("--roma-python", type=Path, default=DEFAULT_ROMA_PYTHON)
    parser.add_argument(
        "--roma-torch-home",
        type=Path,
        default=DEFAULT_ROMA_TORCH_HOME,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_nasal_texture_observation_audit(
        args.capture_dir,
        args.source_v10,
        args.output,
        rig_calibration=args.rig_calibration,
        roma_python=args.roma_python,
        roma_torch_home=args.roma_torch_home,
    )
    print(report)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_RIG_SHA256",
    "DEFAULT_V10_OBJ_SHA256",
    "DEFAULT_V10_REPORT_SHA256",
    "evaluate_release_a_gate",
    "run_nasal_texture_observation_audit",
    "verify_hash_locked_input",
]
