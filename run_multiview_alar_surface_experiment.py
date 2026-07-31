"""Run the hash-locked three-view outer-alar surface experiment."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from run_expression_depth_experiment import _export_with_baseline_texture
from run_multiview_nasal_shape_experiment import (
    _load_observation_work_images,
    _subdivide_candidate,
    _validate_embedded_textured_glb,
    _write_viewer,
    build_model_projection_views,
)
from run_nasal_base_shape_experiment import (
    _load_v4_low_resolution_baseline,
)
from src.geometry.alar_surface_basis import (
    apply_alar_surface_basis,
    build_alar_surface_basis,
)
from src.geometry.alar_surface_observations import (
    build_alar_surface_observations,
)
from src.geometry.alar_surface_optimizer import (
    AlarSurfaceOptimizationConfig,
    fit_multiview_alar_surface,
    prepare_alar_surface_optimization_context,
    project_alar_targets,
)
from src.geometry.nasal_base_semantic_basis import (
    apply_nasal_base_semantic_basis,
    build_nasal_base_semantic_basis,
)
from src.geometry.nasal_semantic_basis import build_nasal_semantic_basis
from src.module2_geometry import export_mesh_glb, export_mesh_obj
from src.reports.nasal_geometry_report import (
    render_nasal_geometry_screenshots,
    validate_minimal_nasal_candidate,
)
from src.reports.nasal_observation_io import load_nasal_observation_bundle


DEFAULT_A2_OBJ_SHA256 = (
    "50051dd20c973ed43f6511eca7b2a5ab4592438901e31da9a74a4ad3bc60201b"
)
DEFAULT_V2_GLB_SHA256 = (
    "219fde410ddc99abf2367315ffe7407969f17b355e654b74f2a21f0e9fd3c7da"
)
DEFAULT_V2_TEXTURE_SHA256 = (
    "b4402401937e5df8a5dc7fe1755be75c02482424ee807463b613485ca6f8e620"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = _sha256(path)
    if actual != str(expected).lower():
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_a2_low_resolution(
    source_a2: Path,
    observations,
):
    report_path = source_a2 / "nasal_base_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "nasal-base-semantic-a2-v1"
        or report.get("status") != "success"
    ):
        raise RuntimeError("source A2 report is not a successful nasal-base result")
    source_v4 = Path(report["source_v4"]["path"]).resolve()
    v4_report = json.loads(
        (source_v4 / "fit_report.json").read_text(encoding="utf-8")
    )
    baseline = _load_v4_low_resolution_baseline(source_v4, v4_report)
    views = build_model_projection_views(
        observations,
        baseline.front_rotation,
        baseline.front_translation,
    )
    a2_basis = build_nasal_base_semantic_basis(
        baseline.vertices,
        baseline.faces,
        baseline.landmark_triangles,
        baseline.landmark_barycentric,
        views[0].R_model_to_camera,
    )
    coefficients = np.asarray(
        report["optimization"]["coefficients"],
        dtype=np.float64,
    )
    a2_vertices = apply_nasal_base_semantic_basis(
        baseline.vertices,
        a2_basis,
        coefficients,
    )
    return (
        dataclasses.replace(baseline, vertices=a2_vertices),
        views,
        report,
        source_v4,
    )


def _export_candidate(
    baseline,
    candidate_vertices: np.ndarray,
    texture_source: Path,
    output: Path,
) -> dict[str, Any]:
    meshes = output / "meshes"
    textures = output / "textures"
    meshes.mkdir(parents=True)
    textures.mkdir(parents=True)
    vertices, faces, uv_vertices, uv_faces = _subdivide_candidate(
        baseline,
        candidate_vertices,
    )
    obj = meshes / "face_mesh.obj"
    geometry_glb = meshes / "candidate_geometry.glb"
    export_mesh_obj(vertices, faces, uv_vertices, uv_faces, obj)
    export_mesh_glb(vertices, faces, uv_vertices, uv_faces, geometry_glb)
    texture = textures / "albedo_white.png"
    shutil.copy2(texture_source, texture)
    if _sha256(texture) != _sha256(texture_source):
        raise RuntimeError("copied accepted v5 texture changed")
    textured_glb = meshes / "candidate_textured.glb"
    _export_with_baseline_texture(
        mesh_path=obj,
        texture_path=texture,
        output_path=textured_glb,
    )
    validation = _validate_embedded_textured_glb(textured_glb)
    return {
        "candidate_obj": obj,
        "candidate_geometry_glb": geometry_glb,
        "candidate_textured_glb": textured_glb,
        "texture": texture,
        "texture_sha256": _sha256(texture),
        "subdivided_vertex_count": int(len(vertices)),
        "subdivided_face_count": int(len(faces)),
        "glb_validation": validation,
    }


def _draw_projection_diagnostics(
    output: Path,
    work_images: Mapping[str, np.ndarray],
    observations,
    baseline_projection: Mapping[str, np.ndarray],
    candidate_projection: Mapping[str, np.ndarray],
) -> dict[str, Path]:
    debug = output / "debug" / "alar_projection"
    debug.mkdir(parents=True)
    paths = {}
    for view in ("front", "subject-left", "subject-right"):
        image = np.asarray(work_images[view])
        if image.ndim == 2:
            canvas = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            canvas = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        for target in observations.by_view[view]:
            curve = np.rint(target.curve_work).astype(np.int32)
            cv2.polylines(
                canvas,
                [curve.reshape(-1, 1, 2)],
                False,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
            for point in baseline_projection[target.name]:
                cv2.circle(
                    canvas,
                    tuple(np.rint(point).astype(int)),
                    2,
                    (255, 255, 0),
                    -1,
                    cv2.LINE_AA,
                )
            for point in candidate_projection[target.name]:
                cv2.circle(
                    canvas,
                    tuple(np.rint(point).astype(int)),
                    2,
                    (0, 255, 0),
                    -1,
                    cv2.LINE_AA,
                )
        path = debug / f"{view.replace('-', '_')}.png"
        cv2.imwrite(str(path), canvas)
        paths[view] = path

    cards = "\n".join(
        (
            "<figure><img src=\"../alar_projection/"
            + path.name
            + "\"><figcaption>"
            + view
            + "</figcaption></figure>"
        )
        for view, path in paths.items()
    )
    observation_dir = output / "debug" / "alar_observations"
    observation_dir.mkdir(parents=True)
    (observation_dir / "index.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Alar observations</title>
<style>body{background:#111820;color:#e8eef5;font:15px system-ui;margin:20px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
figure{margin:0}img{width:100%;height:auto}figcaption{padding:8px 0}
.legend{margin-bottom:16px}.o{color:#ff5cff}.b{color:#58ffff}.c{color:#56ff7d}</style>
<h1>Three-view outer-alar evidence</h1>
<p class="legend"><span class="o">Observed curve</span> ·
<span class="b">A2 baseline</span> · <span class="c">Candidate</span></p>
<main>"""
        + cards
        + "</main>",
        encoding="utf-8",
    )
    return paths


def _metric_acceptance(result) -> dict[str, Any]:
    baseline = result.baseline_metrics["views"]
    candidate = result.candidate_metrics["views"]
    changes = {}
    improved = 0
    all_within_tolerance = True
    for view in ("front", "subject-left", "subject-right"):
        before = float(baseline[view]["combined_mean_px"])
        after = float(candidate[view]["combined_mean_px"])
        improvement = (before - after) / before if before > 1e-9 else 0.0
        changes[view] = {
            "baseline_px": before,
            "candidate_px": after,
            "improvement_ratio": improvement,
            "within_regression_tolerance": after <= before + 0.25,
        }
        improved += int(improvement >= 0.10)
        all_within_tolerance &= after <= before + 0.25
    objective_improvement = (
        (result.initial_cost - result.final_cost) / result.initial_cost
        if result.initial_cost > 1e-9
        else 0.0
    )
    passed = (
        all_within_tolerance
        and improved >= 2
        and objective_improvement >= 0.10
    )
    return {
        "passed": bool(passed),
        "views_improved_at_least_10_percent": int(improved),
        "all_views_within_0_25_px": bool(all_within_tolerance),
        "objective_improvement_ratio": float(objective_improvement),
        "per_view": changes,
    }


def run_multiview_alar_surface_experiment(
    capture_dir: str | Path,
    source_a2: str | Path,
    source_v5: str | Path,
    output: str | Path,
    *,
    rig_calibration: str | Path,
    expected_a2_obj_sha256: str = DEFAULT_A2_OBJ_SHA256,
    expected_v2_glb_sha256: str = DEFAULT_V2_GLB_SHA256,
    expected_v2_texture_sha256: str = DEFAULT_V2_TEXTURE_SHA256,
    viewer_template: str | Path | None = None,
) -> Path:
    """Fit outer nasal wings and publish an immutable A2/v5 comparison."""
    captures = Path(capture_dir).resolve()
    a2 = Path(source_a2).resolve()
    v5 = Path(source_v5).resolve()
    target = Path(output).resolve()
    rig = Path(rig_calibration).resolve()
    for path, label in (
        (captures, "capture directory"),
        (a2, "A2 source"),
        (v5, "v5 source"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not rig.is_file():
        raise FileNotFoundError(f"rig calibration does not exist: {rig}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {target}")

    a2_obj = a2 / "meshes" / "face_mesh.obj"
    a2_hash = _verify_hash(a2_obj, expected_a2_obj_sha256, "accepted A2 OBJ")
    v5_report_path = v5 / "nasal_local_texture_report.json"
    v5_report = json.loads(v5_report_path.read_text(encoding="utf-8"))
    source_hashes = v5_report.get("source_v2", {}).get("hashes_before", {})
    if (
        source_hashes.get("glb") != expected_v2_glb_sha256
        or source_hashes.get("texture") != expected_v2_texture_sha256
        or source_hashes.get("obj") != expected_a2_obj_sha256
    ):
        raise RuntimeError("v5 report does not descend from the accepted v2/A2 hashes")
    baseline_glb = v5 / "meshes" / "face.glb"
    texture_source = v5 / "textures" / "albedo_white.png"
    baseline_glb_hash = _sha256(baseline_glb)
    texture_hash = _sha256(texture_source)
    source_snapshots = {
        "a2_obj": a2_hash,
        "v5_glb": baseline_glb_hash,
        "v5_texture": texture_hash,
        "v5_report": _sha256(v5_report_path),
        "rig": _sha256(rig),
    }

    target.mkdir(parents=True)
    observation_path = Path(
        json.loads((a2 / "nasal_base_report.json").read_text(encoding="utf-8"))[
            "source_v4"
        ]["path"]
    ).resolve() / "nasal_observations.json"
    source_observations = load_nasal_observation_bundle(observation_path)
    baseline, views, a2_report, source_v4 = _load_a2_low_resolution(
        a2,
        source_observations,
    )
    projection_basis = build_nasal_semantic_basis(
        baseline.vertices,
        baseline.faces,
        baseline.landmark_triangles,
        baseline.landmark_barycentric,
        views[0].R_model_to_camera,
    )
    basis = build_alar_surface_basis(
        baseline.vertices,
        baseline.faces,
        baseline.landmark_triangles,
        baseline.landmark_barycentric,
        views[0].R_model_to_camera,
        projection_basis=projection_basis,
    )
    observations = build_alar_surface_observations(source_observations)
    context = prepare_alar_surface_optimization_context(
        baseline.vertices,
        baseline.faces,
        basis,
        projection_basis,
        observations,
        views,
    )
    config = AlarSurfaceOptimizationConfig()
    result = fit_multiview_alar_surface(context, config)
    if not result.success:
        failed = result.stage_results.get(str(result.failure_stage))
        detail = failed.message if failed is not None else "unknown solver failure"
        raise RuntimeError(
            f"outer-alar optimization failed at {result.failure_stage}: {detail}"
        )

    unchanged_outside = np.array_equal(
        result.candidate_vertices[~basis.support_mask],
        baseline.vertices[~basis.support_mask],
    )
    unchanged_protected = np.array_equal(
        result.candidate_vertices[basis.protected_mask],
        baseline.vertices[basis.protected_mask],
    )
    quality = validate_minimal_nasal_candidate(
        parameters=result.coefficients,
        baseline_vertices=baseline.vertices,
        candidate_vertices=result.candidate_vertices,
        baseline_faces=baseline.faces,
        candidate_faces=baseline.faces,
        baseline_uv_vertices=baseline.uv_vertices,
        candidate_uv_vertices=baseline.uv_vertices,
        baseline_uv_faces=baseline.uv_faces,
        candidate_uv_faces=baseline.uv_faces,
    )
    if not unchanged_outside:
        quality["issues"].append("vertices_changed_outside_outer_alar_support")
        quality["passed"] = False
    if not unchanged_protected:
        quality["issues"].append("protected_vertices_changed")
        quality["passed"] = False
    metric_acceptance = _metric_acceptance(result)
    parameter_names = list(basis.names) + [
        f"{view}_{axis}_translation_px"
        for view in ("front", "subject-left", "subject-right")
        for axis in ("x", "y")
    ]
    bounds = np.r_[
        np.full(6, float(config.coefficient_bound)),
        np.full(6, float(config.nuisance_translation_bound_px)),
    ]
    saturated_parameters = [
        name
        for name, value, bound in zip(
            parameter_names,
            result.parameters,
            bounds,
        )
        if abs(float(value)) >= float(bound) - 1e-4
    ]
    metric_acceptance["saturated_parameters"] = saturated_parameters
    metric_acceptance["saturation_count"] = len(saturated_parameters)
    metric_acceptance["saturation_limit"] = 3
    if len(saturated_parameters) > 3:
        metric_acceptance["passed"] = False
        metric_acceptance["saturation_review_required"] = True

    artifacts = _export_candidate(
        baseline,
        result.candidate_vertices,
        texture_source,
        target,
    )
    baseline_copy = target / "meshes" / "baseline_textured.glb"
    shutil.copy2(baseline_glb, baseline_copy)
    baseline_geometry_source = v5 / "meshes" / "face_stable_geometry.glb"
    baseline_geometry = target / "meshes" / "baseline_geometry.glb"
    shutil.copy2(baseline_geometry_source, baseline_geometry)
    viewer = _write_viewer(
        template=Path(viewer_template).resolve() if viewer_template else None,
        baseline_glb=baseline_copy,
        candidate_glb=Path(artifacts["candidate_textured_glb"]),
        output=target / "alar_surface_compare.html",
        dataset_label=f"{captures.name} | multiview outer-alar surface v1",
    )
    model_renders = render_nasal_geometry_screenshots(
        baseline_glb=baseline_copy,
        candidate_glb=Path(artifacts["candidate_textured_glb"]),
        output_dir=target / "debug" / "model_renders",
    )

    work_images = _load_observation_work_images(
        observation_path,
        source_observations,
        rig,
    )
    baseline_projection = project_alar_targets(
        np.zeros(12, dtype=np.float64),
        context,
        config,
    )
    candidate_projection = project_alar_targets(
        result.parameters,
        context,
        config,
    )
    overlays = _draw_projection_diagnostics(
        target,
        work_images,
        observations,
        baseline_projection,
        candidate_projection,
    )

    source_after = {
        "a2_obj": _sha256(a2_obj),
        "v5_glb": _sha256(baseline_glb),
        "v5_texture": _sha256(texture_source),
        "v5_report": _sha256(v5_report_path),
        "rig": _sha256(rig),
    }
    if source_after != source_snapshots:
        raise RuntimeError("an accepted baseline changed during the experiment")
    status = (
        "success"
        if quality["passed"] and metric_acceptance["passed"]
        else "review_required"
    )
    report_payload = {
        "schema": "multiview-alar-surface-experiment-v1",
        "status": status,
        "dataset": captures.name,
        "baseline_contract": {
            "source_a2": str(a2),
            "source_v5": str(v5),
            "source_v4": str(source_v4),
            "hashes_before": source_snapshots,
            "hashes_after": source_after,
            "unchanged": source_after == source_snapshots,
        },
        "semantic_basis": {
            "parameter_ordering": list(basis.names),
            "support_vertex_count": int(np.count_nonzero(basis.support_mask)),
            "protected_vertex_count": int(np.count_nonzero(basis.protected_mask)),
            "unit_scale": float(basis.unit_scale),
            "outside_support_unchanged": bool(unchanged_outside),
            "protected_vertices_unchanged": bool(unchanged_protected),
            "metadata": dict(basis.metadata),
        },
        "optimization": result.to_report(),
        "metric_acceptance": metric_acceptance,
        "geometry_quality": quality,
        "texture": {
            "strategy": "accepted_v5_same_uv_for_geometry_review",
            "source_sha256": texture_hash,
            "candidate_sha256": artifacts["texture_sha256"],
            "non_texture_geometry_objective": True,
        },
        "artifacts": {
            **artifacts,
            "baseline_geometry_glb": baseline_geometry,
            "baseline_textured_glb": baseline_copy,
            "viewer": viewer,
            "observation_report": target / "debug" / "alar_observations" / "index.html",
            "projection_overlays": overlays,
            "model_renders": model_renders,
        },
        "a2_report_source_hash": _sha256(a2 / "nasal_base_report.json"),
    }
    report = target / "alar_surface_report.json"
    report.write_text(
        json.dumps(_jsonable(report_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit fixed-rig outer nasal-wing surface evidence."
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-a2", type=Path, required=True)
    parser.add_argument("--source-v5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rig-calibration", type=Path, required=True)
    parser.add_argument("--viewer-template", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_multiview_alar_surface_experiment(
        args.capture_dir,
        args.source_a2,
        args.source_v5,
        args.output,
        rig_calibration=args.rig_calibration,
        viewer_template=args.viewer_template,
    )
    print(f"Outer-alar report: {report}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_A2_OBJ_SHA256",
    "DEFAULT_V2_GLB_SHA256",
    "DEFAULT_V2_TEXTURE_SHA256",
    "run_multiview_alar_surface_experiment",
]
