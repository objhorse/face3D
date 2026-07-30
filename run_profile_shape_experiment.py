"""Build textured A/B candidates from trusted silhouette profile targets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from run_expression_depth_experiment import (
    _default_viewer_template,
    _export_with_baseline_texture,
    _load_fit_payload,
    _write_embedded_compare_viewer,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--silhouette-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewer-template", type=Path)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _region_displacements(
    baseline_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    landmark_triangles: np.ndarray,
    barycentric: np.ndarray,
    rotation: np.ndarray,
) -> dict[str, Any]:
    baseline_landmarks = np.sum(
        baseline_vertices[landmark_triangles] * barycentric[:, :, None],
        axis=1,
    )
    candidate_landmarks = np.sum(
        candidate_vertices[landmark_triangles] * barycentric[:, :, None],
        axis=1,
    )
    displacement = (candidate_landmarks - baseline_landmarks) @ rotation.T
    magnitude = np.linalg.norm(displacement, axis=1) * 1000.0
    groups = {
        "nose": np.arange(27, 36),
        "eyes": np.arange(36, 48),
        "mouth": np.arange(48, 68),
        "chin": np.arange(7, 10),
        "jaw": np.arange(0, 17),
    }
    return {
        name: {
            "mean_mm": float(np.mean(magnitude[indices])),
            "p95_mm": float(np.percentile(magnitude[indices], 95.0)),
            "max_mm": float(np.max(magnitude[indices])),
        }
        for name, indices in groups.items()
    }


def _export_candidate(
    *,
    source_mesh_dir: Path,
    silhouette_report: dict[str, Any],
    output_mesh_dir: Path,
    cfg: Any,
) -> dict[str, Any]:
    import torch
    import trimesh

    from src.geometry.identity_quality import (
        IdentityDriftThresholds,
        compute_identity_drift,
        make_identity_drift_gate,
    )
    from src.geometry.mesh_quality import (
        MeshQualityThresholds,
        compare_mesh_quality,
        compute_mesh_quality,
    )
    from src.geometry.profile_shape_fit import (
        fit_flame_nose_profile_shape,
        profile_target_from_silhouette_report,
    )
    from src.module2_geometry import (
        FLAMEModel,
        _get_flame_uv,
        export_mesh_glb,
        export_mesh_obj,
        load_flame_landmark_mapping,
    )

    fit_payload, optimized = _load_fit_payload(source_mesh_dir)
    shape = np.asarray(optimized["shape_params"], dtype=np.float32)
    expression = np.asarray(optimized["expression_params"], dtype=np.float32)
    mica_anchor = np.asarray(
        optimized["mica_identity_anchor"],
        dtype=np.float32,
    )
    front_pose = optimized["per_view"]["front"]
    front_rotation = np.asarray(front_pose["R"], dtype=np.float64)
    front_translation = np.asarray(front_pose["t"], dtype=np.float64)
    target = profile_target_from_silhouette_report(silhouette_report)

    flame = FLAMEModel(
        Path(cfg.FLAME_MODEL_PATH),
        n_shape=len(shape),
        n_exp=len(expression),
    ).cpu()
    faces = flame.faces.detach().cpu().numpy()
    landmark_data = load_flame_landmark_mapping(Path(cfg.FLAME_LANDMARK_PATH))
    if landmark_data is None:
        raise RuntimeError("FLAME landmark embedding is required")
    landmark_triangles = faces[
        np.asarray(landmark_data["face_idx"], dtype=np.int64)
    ]
    barycentric = np.asarray(
        landmark_data["bary_coords"],
        dtype=np.float64,
    )
    fit = fit_flame_nose_profile_shape(
        template_vertices=flame.v_template.detach().cpu().numpy(),
        shape_basis=flame.shape_basis.detach().cpu().numpy(),
        baseline_shape=shape,
        faces=faces,
        landmark_triangles=landmark_triangles,
        landmark_barycentric=barycentric,
        front_rotation=front_rotation,
        front_translation=front_translation,
        target_nose_to_upper_lip_m=target,
    )
    candidate_shape = np.asarray(fit["candidate_shape"], dtype=np.float32)
    zero_expression = torch.zeros(flame.n_exp, dtype=torch.float32)
    with torch.no_grad():
        baseline_neutral = flame(
            torch.from_numpy(shape),
            zero_expression,
        ).cpu().numpy()
        candidate_neutral = flame(
            torch.from_numpy(candidate_shape),
            zero_expression,
        ).cpu().numpy()
        mica_neutral = flame(
            torch.from_numpy(mica_anchor),
            zero_expression,
        ).cpu().numpy()
        candidate_vertices = flame(
            torch.from_numpy(candidate_shape),
            torch.from_numpy(expression),
        ).cpu().numpy()

    absolute_identity_gate = make_identity_drift_gate(
        anchor_shape=mica_anchor,
        candidate_shape=candidate_shape,
        anchor_vertices=mica_neutral,
        candidate_vertices=candidate_neutral,
        thresholds=IdentityDriftThresholds(
            max_coefficient_l2=8.0,
            max_mean_displacement_pct=3.0,
            max_p95_displacement_pct=5.0,
            max_displacement_pct=8.0,
        ),
    )
    baseline_identity = compute_identity_drift(
        anchor_shape=mica_anchor,
        candidate_shape=shape,
        anchor_vertices=mica_neutral,
        candidate_vertices=baseline_neutral,
    )
    candidate_identity = absolute_identity_gate["metrics"]
    relative_limits = {
        "coefficient_delta_l2": 1.0,
        "mean_displacement_pct": 0.25,
        "p95_displacement_pct": 0.75,
        # The intended nose-tip correction may own the single largest MICA
        # displacement. Global mean/P95 and the explicit outside-nose gate
        # remain the stronger non-worsening checks.
        "max_displacement_pct": 1.5,
    }
    relative_change = {
        name: float(candidate_identity[name] - baseline_identity[name])
        for name in relative_limits
    }
    relative_issues = [
        f"{name}_worsening_exceeded"
        for name, limit in relative_limits.items()
        if relative_change[name] > float(limit)
    ]
    identity_gate = {
        "passed": bool(absolute_identity_gate["passed"] and not relative_issues),
        "issues": list(absolute_identity_gate["issues"]) + relative_issues,
        "baseline_metrics": baseline_identity,
        "candidate_metrics": candidate_identity,
        "relative_change": relative_change,
        "relative_limits": relative_limits,
        "absolute_gate": absolute_identity_gate,
    }
    baseline_quality = compute_mesh_quality(
        baseline_neutral,
        faces,
        label="profile_shape_baseline",
    )
    candidate_quality = compute_mesh_quality(
        candidate_neutral,
        faces,
        label="profile_shape_candidate",
    )
    mesh_gate = compare_mesh_quality(
        baseline_quality,
        candidate_quality,
        thresholds=MeshQualityThresholds(
            min_face_ratio=1.0,
            max_new_degenerate_faces=0,
            max_new_nonmanifold_edges=0,
            max_new_boundary_edges=0,
        ),
        region_name="profile_shape",
    )
    accepted = bool(fit["passed"] and identity_gate["passed"] and mesh_gate["passed"])
    if not accepted:
        failed = []
        if not fit["passed"]:
            failed.append("protected_profile_shape")
        if not identity_gate["passed"]:
            failed.append("identity_drift")
        if not mesh_gate["passed"]:
            failed.append("mesh_quality")
        raise RuntimeError(f"profile shape candidate rejected: {', '.join(failed)}")

    uv_vertices, uv_faces = _get_flame_uv(Path(cfg.FLAME_MODEL_PATH), faces)
    vertices_sub, faces_sub = trimesh.remesh.subdivide_loop(
        candidate_vertices,
        faces,
        iterations=2,
    )
    neutral_sub, neutral_faces = trimesh.remesh.subdivide_loop(
        candidate_neutral,
        faces,
        iterations=2,
    )
    if not np.array_equal(faces_sub, neutral_faces):
        raise RuntimeError("neutral and expression subdivision topology diverged")
    uv_sub, uv_faces_sub = uv_vertices, uv_faces
    for _iteration in range(2):
        uv_sub, uv_faces_sub = trimesh.remesh.subdivide(
            uv_sub,
            uv_faces_sub,
        )
    if len(faces_sub) != len(uv_faces_sub):
        raise RuntimeError("geometry and UV subdivision topology diverged")

    output_mesh_dir.mkdir(parents=True, exist_ok=True)
    export_mesh_obj(
        vertices_sub,
        faces_sub,
        uv_sub,
        uv_faces_sub,
        output_mesh_dir / "face_mesh.obj",
    )
    export_mesh_glb(
        vertices_sub,
        faces_sub,
        uv_sub,
        uv_faces_sub,
        output_mesh_dir / "face_mesh.glb",
    )
    export_mesh_glb(
        neutral_sub,
        faces_sub,
        uv_sub,
        uv_faces_sub,
        output_mesh_dir / "face_stable_neutral.glb",
    )
    shutil.copy2(
        output_mesh_dir / "face_mesh.glb",
        output_mesh_dir / "face_stable_geometry.glb",
    )
    optimized["shape_params"] = candidate_shape.tolist()
    optimized["profile_shape_constraint"] = _jsonable(fit)
    fit_payload["parameters"]["optimized_parameters"] = optimized
    (output_mesh_dir / "stable_fit_meta.json").write_text(
        json.dumps(fit_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "accepted": accepted,
        "fit": _jsonable(fit),
        "identity_gate": _jsonable(identity_gate),
        "mesh_gate": _jsonable(mesh_gate),
        "region_displacements": _region_displacements(
            baseline_neutral,
            candidate_neutral,
            landmark_triangles,
            barycentric,
            front_rotation,
        ),
        "vertices": int(len(vertices_sub)),
        "faces": int(len(faces_sub)),
    }


def main() -> None:
    from src import config as cfg

    args = _parse_args()
    capture_dir = args.capture_dir.resolve()
    source_output = args.source_output.resolve()
    output = args.output.resolve()
    source_mesh_dir = source_output / "meshes"
    output_mesh_dir = output / "meshes"
    output_texture_dir = output / "textures"
    output.mkdir(parents=True, exist_ok=True)
    silhouette_report = json.loads(
        args.silhouette_report.resolve().read_text(encoding="utf-8")
    )
    geometry = _export_candidate(
        source_mesh_dir=source_mesh_dir,
        silhouette_report=silhouette_report,
        output_mesh_dir=output_mesh_dir,
        cfg=cfg,
    )
    for name in ("cameras.json", "stable_semantic_regions.json"):
        source = source_mesh_dir / name
        if source.exists():
            shutil.copy2(source, output_mesh_dir / name)

    texture_candidates = (
        source_output / "textures" / "albedo_baseline_locked.png",
        source_output / "textures" / "albedo_white.png",
    )
    baseline_texture = next(
        (path for path in texture_candidates if path.exists()),
        None,
    )
    if baseline_texture is None:
        raise FileNotFoundError("source output has no baseline texture")
    output_texture_dir.mkdir(parents=True, exist_ok=True)
    output_texture = output_texture_dir / "albedo_baseline_locked.png"
    shutil.copy2(baseline_texture, output_texture)
    candidate_textured = output_mesh_dir / "face_same_texture.glb"
    _export_with_baseline_texture(
        mesh_path=output_mesh_dir / "face_mesh.obj",
        texture_path=output_texture,
        output_path=candidate_textured,
    )

    baseline_textured = source_mesh_dir / "face_same_texture.glb"
    if not baseline_textured.exists():
        baseline_textured = source_mesh_dir / "face.glb"
    template = (
        args.viewer_template.resolve()
        if args.viewer_template
        else _default_viewer_template()
    )
    viewer = _write_embedded_compare_viewer(
        template,
        baseline_textured,
        candidate_textured,
        output / "profile_shape_compare.html",
        baseline_label="Baseline: protected expression, original identity shape",
        candidate_label="Candidate: silhouette-guided FLAME nose shape",
        title="Silhouette-guided parametric shape A/B",
    )
    report = {
        "capture_dir": str(capture_dir),
        "source_output": str(source_output),
        "silhouette_report": str(args.silhouette_report.resolve()),
        "geometry": geometry,
        "baseline_glb": str(baseline_textured),
        "candidate_glb": str(candidate_textured),
        "viewer": str(viewer),
    }
    report_path = output / "profile_shape_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fit = geometry["fit"]
    print(
        "Nose-to-upper-lip: "
        f"{fit['baseline_nose_to_upper_lip_m'] * 1000.0:.2f} -> "
        f"{fit['candidate_nose_to_upper_lip_m'] * 1000.0:.2f} mm "
        f"(target {fit['target_nose_to_upper_lip_m'] * 1000.0:.2f} mm)"
    )
    print(f"A/B viewer: {viewer}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
