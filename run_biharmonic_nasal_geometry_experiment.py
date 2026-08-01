"""Replace the hard-truncated RoMa nasal fit with a smooth surface solve."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from run_multiview_nasal_shape_experiment import _validate_embedded_textured_glb
from run_nasal_observation_audit import assert_file_tree_unchanged, file_tree_hashes
from src.appearance.roma_texture_controls import work_pixels_to_letterbox_canvas
from src.geometry.biharmonic_nasal_surface import (
    BiharmonicNasalConfig,
    VectorBiharmonicNasalResult,
    expand_vertex_mask,
    fit_vector_biharmonic_nasal_surface,
)
from src.geometry.cross_view_surface_observations import sparse_zbuffer_attachments
from src.module2_geometry import export_mesh_glb, export_mesh_obj
from src.module3_texture import (
    export_glb,
    load_cameras,
    load_mesh_obj,
    transparent_bottom_face_mask,
)
from src.reports.nasal_geometry_report import validate_minimal_nasal_candidate
from src.reports.offline_glb_compare import write_offline_glb_compare_viewer


DEFAULT_BENDING_WEIGHTS = (100.0, 300.0, 1000.0, 3000.0)
DEFAULT_PRIOR_WEIGHTS = (0.5, 2.0, 8.0, 20.0)
DEFAULT_ROBUST_DELTAS_MM = (1.5,)


def _percentile(values: np.ndarray, percentile: float) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.percentile(array, percentile)) if len(array) else 0.0


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    topology = np.asarray(faces, dtype=np.int64)
    edges = np.vstack(
        (topology[:, [0, 1]], topology[:, [1, 2]], topology[:, [2, 0]])
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _unit_face_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    topology = np.asarray(faces, dtype=np.int64)
    normals = np.cross(
        vertices[topology[:, 1]] - vertices[topology[:, 0]],
        vertices[topology[:, 2]] - vertices[topology[:, 0]],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    unit = np.zeros_like(normals)
    unit[valid] = normals[valid] / lengths[valid, None]
    return unit, valid


def mesh_change_metrics(
    baseline_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, Any]:
    """Measure deformation smoothness without using image or texture evidence."""
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    candidate = np.asarray(candidate_vertices, dtype=np.float64)
    topology = np.asarray(faces, dtype=np.int64)
    displacement = candidate - baseline
    magnitude_mm = np.linalg.norm(displacement, axis=1) * 1000.0
    moved = magnitude_mm > 1e-4

    edges = _unique_edges(topology)
    edge_crossing = moved[edges[:, 0]] ^ moved[edges[:, 1]]
    transition_vertices = np.zeros(len(baseline), dtype=bool)
    if np.any(edge_crossing):
        transition_vertices[np.unique(edges[edge_crossing])] = True

    neighbor_sum = np.zeros_like(displacement)
    degree = np.zeros(len(baseline), dtype=np.float64)
    np.add.at(neighbor_sum, edges[:, 0], displacement[edges[:, 1]])
    np.add.at(neighbor_sum, edges[:, 1], displacement[edges[:, 0]])
    np.add.at(degree, edges[:, 0], 1.0)
    np.add.at(degree, edges[:, 1], 1.0)
    laplacian = displacement.copy()
    connected = degree > 0.0
    laplacian[connected] -= neighbor_sum[connected] / degree[connected, None]
    laplacian_mm = np.linalg.norm(laplacian, axis=1) * 1000.0

    baseline_normals, baseline_valid = _unit_face_normals(baseline, topology)
    candidate_normals, candidate_valid = _unit_face_normals(candidate, topology)
    comparable = baseline_valid & candidate_valid
    dots = np.sum(baseline_normals * candidate_normals, axis=1)
    angles = np.zeros(len(topology), dtype=np.float64)
    angles[comparable] = np.degrees(np.arccos(np.clip(dots[comparable], -1.0, 1.0)))
    active_faces = comparable & np.any(moved[topology], axis=1)
    transition_faces = comparable & np.any(moved[topology], axis=1) & ~np.all(
        moved[topology], axis=1
    )

    base_lengths = np.linalg.norm(
        baseline[edges[:, 1]] - baseline[edges[:, 0]], axis=1
    )
    candidate_lengths = np.linalg.norm(
        candidate[edges[:, 1]] - candidate[edges[:, 0]], axis=1
    )
    valid_edges = base_lengths > 1e-12
    active_edges = valid_edges & np.any(moved[edges], axis=1)
    edge_ratios = np.ones(len(edges), dtype=np.float64)
    edge_ratios[valid_edges] = candidate_lengths[valid_edges] / base_lengths[valid_edges]

    return {
        "moved_vertex_count": int(np.count_nonzero(moved)),
        "transition_vertex_count": int(np.count_nonzero(transition_vertices)),
        "maximum_displacement_mm": float(np.max(magnitude_mm)),
        "displacement_p95_mm": _percentile(magnitude_mm[moved], 95.0),
        "laplacian_displacement_p95_mm": _percentile(laplacian_mm[moved], 95.0),
        "transition_laplacian_p95_mm": _percentile(
            laplacian_mm[transition_vertices], 95.0
        ),
        "active_face_normal_change_p95_deg": _percentile(angles[active_faces], 95.0),
        "active_face_normal_change_p99_deg": _percentile(angles[active_faces], 99.0),
        "active_face_normal_change_max_deg": _percentile(angles[active_faces], 100.0),
        "transition_face_normal_change_p95_deg": _percentile(
            angles[transition_faces], 95.0
        ),
        "transition_face_normal_change_p99_deg": _percentile(
            angles[transition_faces], 99.0
        ),
        "active_edge_length_ratio_p01": _percentile(edge_ratios[active_edges], 1.0),
        "active_edge_length_ratio_p99": _percentile(edge_ratios[active_edges], 99.0),
        "active_edge_length_ratio_max": _percentile(edge_ratios[active_edges], 100.0),
        "face_flip_count": int(np.count_nonzero(comparable & (dots < 0.0))),
        "new_degenerate_face_count": int(
            np.count_nonzero(baseline_valid & ~candidate_valid)
        ),
    }


def _smoothness_score(metrics: dict[str, Any]) -> float:
    return float(
        metrics["active_face_normal_change_p95_deg"]
        + 2.0 * metrics["transition_face_normal_change_p95_deg"]
        + 0.25 * metrics["active_face_normal_change_p99_deg"]
        + 8.0 * metrics["laplacian_displacement_p95_mm"]
        + 2.0 * metrics["maximum_displacement_mm"]
    )


def _distance_metrics(
    first: np.ndarray,
    second: np.ndarray,
    *,
    vertex_mask: np.ndarray | None = None,
) -> dict[str, float]:
    distance_mm = np.linalg.norm(
        np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64),
        axis=1,
    ) * 1000.0
    if vertex_mask is not None:
        mask = np.asarray(vertex_mask, dtype=bool).reshape(-1)
        if mask.shape != distance_mm.shape or not np.any(mask):
            raise ValueError("distance vertex mask must select at least one vertex")
        distance_mm = distance_mm[mask]
    return {
        "mean_mm": float(np.mean(distance_mm)),
        "p95_mm": _percentile(distance_mm, 95.0),
        "maximum_mm": float(np.max(distance_mm)),
    }


def _candidate_configs(
    bending_weights: Iterable[float],
    prior_weights: Iterable[float],
    robust_deltas_mm: Iterable[float],
) -> tuple[BiharmonicNasalConfig, ...]:
    return tuple(
        BiharmonicNasalConfig(
            fixed_boundary_rings=2,
            bending_weight=float(bending),
            displacement_prior_weight=float(prior),
            robust_delta_mm=float(delta),
            robust_iterations=5,
        )
        for bending in bending_weights
        for prior in prior_weights
        for delta in robust_deltas_mm
    )


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read texture: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_biharmonic_nasal_geometry_experiment(
    source_v8: str | Path,
    source_v9: str | Path,
    source_v10: str | Path,
    output: str | Path,
    *,
    support_rings: int = 4,
    viewer_vendor_root: str | Path | None = None,
) -> Path:
    v8 = Path(source_v8).resolve()
    v9 = Path(source_v9).resolve()
    v10 = Path(source_v10).resolve()
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {target}")
    for source in (v8, v9, v10):
        if not source.is_dir():
            raise FileNotFoundError(source)

    v8_metrics = _load_json(v8 / "metrics.json")
    if (
        v8_metrics.get("schema") != "nasal-texture-observation-audit-v1"
        or v8_metrics.get("status") != "ready_for_geometry"
    ):
        raise RuntimeError("source v8 is not a successful RoMa geometry experiment")
    v9_report = _load_json(v9 / "roma_nasal_texture_report.json")
    if v9_report.get("status") != "success":
        raise RuntimeError("source v9 is not a successful texture experiment")

    v10_obj = v10 / "meshes" / "face_mesh.obj"
    v8_obj = v8 / "meshes" / "candidate.obj"
    v9_obj = v9 / "meshes" / "face_mesh.obj"
    camera_path = v9 / "meshes" / "cameras.json"
    texture_path = v9 / "textures" / "albedo_white.png"
    baseline_glb_path = v9 / "meshes" / "candidate_textured_v9.glb"
    for path in (v10_obj, v8_obj, v9_obj, camera_path, texture_path, baseline_glb_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)

    source_hashes = {source: file_tree_hashes(source) for source in (v8, v9, v10)}
    baseline_vertices, faces, uv_vertices, uv_faces = load_mesh_obj(v10_obj)
    old_vertices, old_faces, _old_uv, _old_uv_faces = load_mesh_obj(v8_obj)
    _v9_vertices, v9_faces, v9_uv_vertices, v9_uv_faces = load_mesh_obj(v9_obj)
    if not np.array_equal(faces, old_faces) or not np.array_equal(faces, v9_faces):
        raise RuntimeError("v8, v9, and v10 geometry topology do not match")
    if not np.array_equal(uv_faces, v9_uv_faces) or not np.allclose(
        uv_vertices, v9_uv_vertices, atol=0.0, rtol=0.0
    ):
        raise RuntimeError("v9 UV contract differs from v10")

    cameras_document = _load_json(camera_path)
    contract = dict(cameras_document.get("coordinate_contract", {}))
    work_size = tuple(map(int, contract.get("source_work_size_wh", (640, 480))))
    canvas_shape = tuple(map(int, contract.get("target_canvas_shape_hw", (1024, 1024))))
    cameras = load_cameras(camera_path)
    front_camera = cameras["front"]
    trusted = tuple(dict(v8_metrics.get("observations", {})).get("trusted", ()))
    if len(trusted) < 6:
        raise RuntimeError("v8 does not contain enough trusted surface observations")
    front_work_pixels = np.asarray(
        [item["pair_match"]["front_pixel"] for item in trusted], dtype=np.float64
    )
    front_canvas_pixels = work_pixels_to_letterbox_canvas(
        front_work_pixels,
        work_size=work_size,
        canvas_shape=canvas_shape,
    )
    face_indices, barycentric, _depth = sparse_zbuffer_attachments(
        baseline_vertices,
        faces,
        front_camera["R"],
        front_camera["t"],
        front_camera["K"],
        front_canvas_pixels,
    )
    valid = face_indices >= 0
    if np.count_nonzero(valid) < 6:
        raise RuntimeError("trusted observations do not attach to the v10 mesh")

    target_front = np.asarray(
        [item["point_reference_m"] for item, keep in zip(trusted, valid) if keep],
        dtype=np.float64,
    )
    rotation = np.asarray(front_camera["R"], dtype=np.float64)
    translation = np.asarray(front_camera["t"], dtype=np.float64).reshape(3)
    target_model = (target_front - translation) @ rotation
    weights = np.asarray(
        [max(float(item["weight"]), 1e-4) for item, keep in zip(trusted, valid) if keep],
        dtype=np.float64,
    )

    old_displacement = np.linalg.norm(old_vertices - baseline_vertices, axis=1)
    seed_support = old_displacement > 1e-9
    support = expand_vertex_mask(seed_support, faces, rings=int(support_rings))
    reference_displacement = old_vertices - baseline_vertices
    old_geometry_metrics = mesh_change_metrics(baseline_vertices, old_vertices, faces)

    candidates: list[tuple[VectorBiharmonicNasalResult, dict[str, Any]]] = []
    sweep_records: list[dict[str, Any]] = []
    for config in _candidate_configs(
        DEFAULT_BENDING_WEIGHTS,
        DEFAULT_PRIOR_WEIGHTS,
        DEFAULT_ROBUST_DELTAS_MM,
    ):
        result = fit_vector_biharmonic_nasal_surface(
            baseline_vertices,
            faces,
            support,
            face_indices[valid],
            barycentric[valid],
            target_model,
            weights,
            reference_displacement_m=reference_displacement,
            config=config,
        )
        geometry = mesh_change_metrics(
            baseline_vertices, result.candidate_vertices, faces
        )
        record = {
            "config": {
                "bending_weight": config.bending_weight,
                "displacement_prior_weight": config.displacement_prior_weight,
                "robust_delta_mm": config.robust_delta_mm,
            },
            "fit": result.to_dict(),
            "geometry": geometry,
            "distance_to_accepted_v8": _distance_metrics(
                result.candidate_vertices,
                old_vertices,
                vertex_mask=seed_support,
            ),
            "smoothness_score": _smoothness_score(geometry),
        }
        record["selection_score"] = float(
            record["smoothness_score"]
            + 6.0 * record["distance_to_accepted_v8"]["p95_mm"]
        )
        candidates.append((result, record))
        sweep_records.append(record)

    initial_rmse = candidates[0][0].initial_rmse_mm
    target_rmse = initial_rmse * 0.80
    feasible = [
        item
        for item in candidates
        if item[0].final_rmse_mm <= target_rmse
        and item[1]["geometry"]["face_flip_count"] == 0
        and item[1]["geometry"]["new_degenerate_face_count"] == 0
    ]
    if not feasible:
        feasible = [
            item
            for item in candidates
            if item[1]["geometry"]["face_flip_count"] == 0
            and item[1]["geometry"]["new_degenerate_face_count"] == 0
        ]
    if not feasible:
        raise RuntimeError("all biharmonic candidates changed mesh orientation or degeneracy")
    selected_result, selected_record = min(
        feasible,
        key=lambda item: (
            item[1]["selection_score"],
            item[0].final_rmse_mm,
        ),
    )
    selected_record["selected"] = True
    selected_record["selection_reason"] = (
        "minimum smoothness plus accepted-v8 identity-anchor score among candidates "
        "retaining at least 20 percent of the observation RMSE improvement"
    )

    quality = validate_minimal_nasal_candidate(
        parameters=np.asarray(
            [
                selected_result.metadata["bending_weight"],
                selected_result.metadata["displacement_prior_weight"],
                selected_result.metadata["robust_delta_mm"],
            ],
            dtype=np.float64,
        ),
        baseline_vertices=baseline_vertices,
        candidate_vertices=selected_result.candidate_vertices,
        baseline_faces=faces,
        candidate_faces=faces,
        baseline_uv_vertices=uv_vertices,
        candidate_uv_vertices=uv_vertices,
        baseline_uv_faces=uv_faces,
        candidate_uv_faces=uv_faces,
    )
    if not quality["passed"]:
        raise RuntimeError("selected biharmonic candidate failed topology quality")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=str(target.parent))
    )
    try:
        mesh_dir = staging / "meshes"
        texture_dir = staging / "textures"
        mesh_dir.mkdir()
        texture_dir.mkdir()
        baseline_glb = mesh_dir / "baseline_v9_textured.glb"
        texture = texture_dir / "albedo_white.png"
        shutil.copy2(baseline_glb_path, baseline_glb)
        shutil.copy2(texture_path, texture)
        shutil.copy2(camera_path, mesh_dir / "cameras.json")

        candidate_obj = mesh_dir / "candidate_biharmonic.obj"
        candidate_geometry = mesh_dir / "candidate_biharmonic_geometry.glb"
        candidate_textured = mesh_dir / "candidate_biharmonic_textured.glb"
        export_mesh_obj(
            selected_result.candidate_vertices,
            faces,
            uv_vertices,
            uv_faces,
            candidate_obj,
        )
        export_mesh_glb(
            selected_result.candidate_vertices,
            faces,
            uv_vertices,
            uv_faces,
            candidate_geometry,
        )
        texture_rgb = _read_rgb(texture)
        transparent_faces, transparent_y_floor = transparent_bottom_face_mask(
            selected_result.candidate_vertices,
            faces,
            0.05,
        )
        export_glb(
            selected_result.candidate_vertices,
            faces,
            uv_vertices,
            uv_faces,
            texture_rgb,
            candidate_textured,
            lighting_type="white",
            lighting_display_name="white",
            smooth_geometry=False,
            transparent_face_mask=transparent_faces,
        )
        shutil.copy2(candidate_textured, mesh_dir / "face.glb")
        glb_validation = dict(_validate_embedded_textured_glb(candidate_textured))
        glb_validation["valid"] = True

        viewer = write_offline_glb_compare_viewer(
            output_path=staging / "biharmonic_nasal_compare.html",
            left_model=baseline_glb,
            right_model=candidate_textured,
            vendor_root=(
                Path(viewer_vendor_root).resolve()
                if viewer_vendor_root is not None
                else Path(__file__).resolve().parent / "frontend" / "vendor"
            ),
            title="captures_20260612_135253: smooth RoMa nasal geometry",
            left_label="Baseline: v8 geometry with accepted v9 texture",
            right_label="Candidate: biharmonic geometry with the same v9 texture",
        )
        report = {
            "schema": "biharmonic-nasal-geometry-experiment-v1",
            "status": "success",
            "sources": {
                "v8_geometry": str(v8),
                "v9_texture": str(v9),
                "v10_baseline": str(v10),
                "unchanged_after_run": True,
            },
            "observation_contract": {
                "trusted_count": len(trusted),
                "front_attachment_count": int(np.count_nonzero(valid)),
                "source_work_size_wh": list(work_size),
                "target_canvas_shape_hw": list(canvas_shape),
            },
            "support": {
                "seed_vertex_count": int(np.count_nonzero(seed_support)),
                "expanded_vertex_count": int(np.count_nonzero(support)),
                "expansion_rings": int(support_rings),
                "boundary_mode": "fixed_outer_two_topology_rings",
                "identity_distance_scope": "v8_moved_seed_support",
            },
            "old_v8_hard_boundary_geometry": old_geometry_metrics,
            "parameter_sweep": sweep_records,
            "selection": selected_record,
            "quality": quality,
            "texture": {
                "source": str(texture_path),
                "geometry_only_experiment": True,
                "rebaked": False,
                "transparent_bottom_face_count": int(np.count_nonzero(transparent_faces)),
                "transparent_bottom_y_floor": float(transparent_y_floor),
            },
            "artifacts": {
                "viewer": viewer.relative_to(staging).as_posix(),
                "baseline_glb": baseline_glb.relative_to(staging).as_posix(),
                "candidate_obj": candidate_obj.relative_to(staging).as_posix(),
                "candidate_geometry_glb": candidate_geometry.relative_to(staging).as_posix(),
                "candidate_textured_glb": candidate_textured.relative_to(staging).as_posix(),
                "face_glb": "meshes/face.glb",
                "texture": texture.relative_to(staging).as_posix(),
                "glb_validation": glb_validation,
            },
        }
        report_path = staging / "biharmonic_nasal_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for source, hashes in source_hashes.items():
            assert_file_tree_unchanged(source, hashes)
        staging.rename(target)
        return target / report_path.relative_to(staging)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v8", type=Path, required=True)
    parser.add_argument("--source-v9", type=Path, required=True)
    parser.add_argument("--source-v10", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-rings", type=int, default=4)
    parser.add_argument("--viewer-vendor-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_biharmonic_nasal_geometry_experiment(
        args.source_v8,
        args.source_v9,
        args.source_v10,
        args.output,
        support_rings=args.support_rings,
        viewer_vendor_root=args.viewer_vendor_root,
    )
    print(report)


if __name__ == "__main__":
    main()
