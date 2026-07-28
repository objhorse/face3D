"""Minimal mesh validity and offline diagnostics for nasal shape candidates."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.geometry.mesh_quality import (
    MeshQualityThresholds,
    compare_mesh_quality,
    compute_mesh_quality,
)


_SEMANTIC_YAWS = {
    "front": 0.0,
    "subject-left": 42.0,
    "subject-right": -42.0,
}


def _face_vectors(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )


def validate_minimal_nasal_candidate(
    *,
    parameters: Any,
    baseline_vertices: Any,
    candidate_vertices: Any,
    baseline_faces: Any,
    candidate_faces: Any,
    baseline_uv_vertices: Any,
    candidate_uv_vertices: Any,
    baseline_uv_faces: Any,
    candidate_uv_faces: Any,
    degenerate_area_epsilon: float = 1e-10,
    flip_dot_tolerance: float = -1e-8,
) -> dict[str, Any]:
    """Return the Batch D validity gate without anatomical or fit thresholds."""
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    candidate = np.asarray(candidate_vertices, dtype=np.float64)
    faces = np.asarray(baseline_faces)
    candidate_face_values = np.asarray(candidate_faces)
    baseline_uv = np.asarray(baseline_uv_vertices)
    candidate_uv = np.asarray(candidate_uv_vertices)
    uv_faces = np.asarray(baseline_uv_faces)
    candidate_uv_face_values = np.asarray(candidate_uv_faces)
    try:
        coefficient_values = np.asarray(parameters, dtype=np.float64)
    except (TypeError, ValueError):
        coefficient_values = np.array([np.nan], dtype=np.float64)

    issues: list[str] = []
    finite_parameters = bool(
        coefficient_values.ndim == 1
        and np.isfinite(coefficient_values).all()
    )
    finite_vertices = bool(
        baseline.ndim == 2
        and baseline.shape[1:] == (3,)
        and candidate.shape == baseline.shape
        and np.isfinite(baseline).all()
        and np.isfinite(candidate).all()
    )
    if not finite_parameters:
        issues.append("parameters_contain_nan_or_inf")
    if not finite_vertices:
        issues.append("vertices_are_nonfinite_or_have_changed_count")

    same_faces = bool(
        faces.ndim == 2
        and faces.shape[1:] == (3,)
        and candidate_face_values.shape == faces.shape
        and np.array_equal(candidate_face_values, faces)
    )
    same_uv_vertices = bool(
        baseline_uv.ndim == 2
        and baseline_uv.shape[1:] == (2,)
        and candidate_uv.shape == baseline_uv.shape
        and np.array_equal(candidate_uv, baseline_uv)
    )
    same_uv_faces = bool(
        uv_faces.ndim == 2
        and uv_faces.shape[1:] == (3,)
        and candidate_uv_face_values.shape == uv_faces.shape
        and np.array_equal(candidate_uv_face_values, uv_faces)
    )
    if not same_faces:
        issues.append("face_topology_changed")
    if not same_uv_vertices or not same_uv_faces:
        issues.append("uv_topology_or_coordinates_changed")

    thresholds = MeshQualityThresholds(
        min_face_ratio=1.0,
        degenerate_area_eps=float(degenerate_area_epsilon),
        max_new_degenerate_faces=0,
        max_new_nonmanifold_edges=0,
        max_new_boundary_edges=0,
    )
    quality_baseline = compute_mesh_quality(
        baseline,
        faces,
        label="baseline",
        thresholds=thresholds,
    )
    quality_candidate = compute_mesh_quality(
        candidate,
        candidate_face_values,
        label="candidate",
        thresholds=thresholds,
    )
    quality_comparison = compare_mesh_quality(
        quality_baseline,
        quality_candidate,
        thresholds=thresholds,
        region_name="full_mesh",
    )
    for issue in quality_comparison["issues"]:
        if issue not in issues:
            issues.append(str(issue))

    flip_indices: list[int] = []
    comparable_face_count = 0
    if finite_vertices and same_faces:
        baseline_vectors = _face_vectors(baseline, faces.astype(np.int64))
        candidate_vectors = _face_vectors(candidate, faces.astype(np.int64))
        baseline_lengths = np.linalg.norm(baseline_vectors, axis=1)
        candidate_lengths = np.linalg.norm(candidate_vectors, axis=1)
        valid_baseline = baseline_lengths > 2.0 * float(degenerate_area_epsilon)
        valid_candidate = candidate_lengths > 0.0
        comparable = valid_baseline & valid_candidate
        comparable_face_count = int(np.count_nonzero(comparable))
        normalized_dot = np.ones(len(faces), dtype=np.float64)
        normalized_dot[comparable] = np.einsum(
            "ij,ij->i",
            baseline_vectors[comparable],
            candidate_vectors[comparable],
        ) / (
            baseline_lengths[comparable]
            * candidate_lengths[comparable]
        )
        flip_indices = [
            int(index)
            for index in np.flatnonzero(
                comparable & (normalized_dot < float(flip_dot_tolerance))
            )
        ]
        if flip_indices:
            issues.append("new_face_flips")

    return {
        "passed": not issues,
        "issues": issues,
        "checks": {
            "finite_parameters": finite_parameters,
            "finite_vertices": finite_vertices,
            "same_vertex_count": bool(candidate.shape == baseline.shape),
            "same_faces": same_faces,
            "same_uv_vertices": same_uv_vertices,
            "same_uv_faces": same_uv_faces,
        },
        "quality": {
            "baseline": quality_baseline,
            "candidate": quality_candidate,
            "comparison": quality_comparison,
        },
        "face_flips": {
            "count": len(flip_indices),
            "indices": flip_indices,
            "comparable_face_count": comparable_face_count,
            "baseline_degenerate_faces_ignored": int(
                quality_baseline.get("degenerate_faces") or 0
            ),
            "normalized_dot_threshold": float(flip_dot_tolerance),
            "degenerate_area_epsilon": float(degenerate_area_epsilon),
        },
    }


def _render_glb(source: Path, output: Path, yaw_degrees: float) -> None:
    import pyrender
    import trimesh
    from PIL import Image

    trimesh_scene = trimesh.load(source, force="scene")
    if not trimesh_scene.geometry:
        raise ValueError(f"GLB contains no geometry: {source}")
    scene = pyrender.Scene.from_trimesh_scene(
        trimesh_scene,
        bg_color=np.array([15, 20, 27, 255]),
        ambient_light=np.array([0.55, 0.55, 0.55]),
    )
    bounds = np.asarray(trimesh_scene.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError(f"GLB bounds are invalid: {source}")
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    magnitude = max(float(max(extent[0], extent[1]) * 0.62), 1e-4)
    camera = pyrender.OrthographicCamera(xmag=magnitude, ymag=magnitude)
    pose = trimesh.transformations.rotation_matrix(
        np.deg2rad(float(yaw_degrees)),
        [0, 1, 0],
    )
    pose[:3, 3] = center + pose[:3, 2] * max(float(max(extent) * 3.0), 1e-3)
    scene.add(camera, pose=pose)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.2),
        pose=pose,
    )
    renderer = pyrender.OffscreenRenderer(768, 768)
    try:
        color, _depth = renderer.render(scene)
    finally:
        renderer.delete()
    if color.size == 0 or not np.any(color != color.reshape(-1, color.shape[-1])[0]):
        raise RuntimeError(f"headless renderer produced a blank image for {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(output)


def render_nasal_geometry_screenshots(
    *,
    baseline_glb: str | Path,
    candidate_glb: str | Path,
    output_dir: str | Path,
) -> dict[str, dict[str, Path]]:
    """Render matching front/subject-left/subject-right GLB diagnostics."""
    sources = {
        "baseline": Path(baseline_glb),
        "candidate": Path(candidate_glb),
    }
    target = Path(output_dir)
    for role, source in sources.items():
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"{role} GLB is missing or empty: {source}")
    result: dict[str, dict[str, Path]] = {}
    try:
        for role, source in sources.items():
            result[role] = {}
            for semantic_view, yaw in _SEMANTIC_YAWS.items():
                output = target / f"{role}_{semantic_view}.png"
                _render_glb(source, output, yaw)
                result[role][semantic_view] = output
    except Exception as exc:
        raise RuntimeError(
            f"headless nasal geometry rendering failed: {type(exc).__name__}: {exc}"
        ) from exc
    return result


def _objective_rows(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> str:
    baseline_raw = _mapping_or_empty(baseline.get("raw_costs"))
    baseline_robust = _mapping_or_empty(baseline.get("robust_costs"))
    candidate_raw = _mapping_or_empty(candidate.get("raw_costs"))
    candidate_robust = _mapping_or_empty(candidate.get("robust_costs"))
    names = tuple(
        dict.fromkeys(
            tuple(baseline_raw)
            + tuple(baseline_robust)
            + tuple(candidate_raw)
            + tuple(candidate_robust)
        )
    )
    rows = []
    for name in names:
        cells = [
            name,
            baseline_raw.get(name),
            baseline_robust.get(name),
            candidate_raw.get(name),
            candidate_robust.get(name),
        ]
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(_format_cell(value))}</td>"
                for value in cells
            )
            + "</tr>"
        )
    return "\n".join(rows)


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    return str(value)


def write_nasal_geometry_report(
    output_dir: str | Path,
    *,
    dataset_label: str,
    screenshots: Mapping[str, Mapping[str, str | Path]],
    baseline_objective: Mapping[str, Any],
    candidate_objective: Mapping[str, Any],
    evidence: Mapping[str, Any],
    validity: Mapping[str, Any],
) -> Path:
    """Write a static, file-URL-compatible pure-geometry diagnostic report."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    screenshot_cells = []
    for semantic_view in _SEMANTIC_YAWS:
        baseline = Path(screenshots["baseline"][semantic_view])
        candidate = Path(screenshots["candidate"][semantic_view])
        for path in (baseline, candidate):
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"geometry screenshot missing: {path}")
        screenshot_cells.append(
            f"""
            <section>
              <h2>{html.escape(semantic_view)}</h2>
              <div class="pair">
                <figure><img src="{html.escape(baseline.name)}"><figcaption>Baseline</figcaption></figure>
                <figure><img src="{html.escape(candidate.name)}"><figcaption>New candidate</figcaption></figure>
              </div>
            </section>
            """
        )
    confidence_sums = _mapping_or_empty(
        evidence.get("per_view_effective_confidence_sums")
    )
    low_confidence = [
        str(view)
        for view, value in confidence_sums.items()
        if float(value) <= 0.0
    ]
    warning = (
        "Low-confidence evidence: " + ", ".join(low_confidence)
        if low_confidence
        else "All three views contain positive-confidence geometric evidence."
    )
    quality = _mapping_or_empty(validity.get("quality"))
    comparison = _mapping_or_empty(quality.get("comparison"))
    flips = _mapping_or_empty(validity.get("face_flips"))
    report_data = {
        "dataset": dataset_label,
        "geometry_only": True,
        "texture_scoring_included": False,
        "validity": validity,
        "evidence": evidence,
    }
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(dataset_label)} nasal geometry</title>
  <style>
    body {{ margin: 0; background: #0f141b; color: #edf1f5; font: 15px/1.45 Arial, sans-serif; }}
    main {{ max-width: 1180px; margin: auto; padding: 24px; }}
    h1 {{ font-size: 26px; margin: 0 0 6px; }}
    h2 {{ font-size: 18px; margin-top: 26px; }}
    .note {{ color: #b8c3cf; }}
    .pair {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    figure {{ margin: 0; background: #171e27; border: 1px solid #33404d; padding: 8px; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 7px 2px 1px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ padding: 7px 8px; border: 1px solid #33404d; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    pre {{ overflow: auto; background: #171e27; padding: 12px; border: 1px solid #33404d; }}
    @media (max-width: 720px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body><main>
  <h1>{html.escape(dataset_label)}: nasal geometry</h1>
  <p class="note">Pure geometry report. Texture scoring is not part of the objective or validity gate.</p>
  <p><strong>Evidence:</strong> {html.escape(warning)}</p>
  {''.join(screenshot_cells)}
  <h2>Unified objective</h2>
  <table>
    <thead><tr><th>Term</th><th>Baseline raw</th><th>Baseline robust</th><th>Candidate raw</th><th>Candidate robust</th></tr></thead>
    <tbody>{_objective_rows(baseline_objective, candidate_objective)}</tbody>
  </table>
  <h2>Minimal validity</h2>
  <p>Passed: <strong>{html.escape(str(bool(validity.get('passed'))))}</strong>;
     degenerate delta: {html.escape(_format_cell(comparison.get('degenerate_delta')))};
     nonmanifold edge delta: {html.escape(_format_cell(comparison.get('nonmanifold_edge_delta')))};
     boundary edge delta: {html.escape(_format_cell(comparison.get('boundary_edge_delta')))};
     new face flips: {html.escape(_format_cell(flips.get('count')))}.</p>
  <details><summary>Machine-readable diagnostics</summary><pre>{html.escape(json.dumps(report_data, ensure_ascii=False, indent=2))}</pre></details>
</main></body></html>
"""
    output = target / "index.html"
    output.write_text(page, encoding="utf-8")
    return output


__all__ = [
    "render_nasal_geometry_screenshots",
    "validate_minimal_nasal_candidate",
    "write_nasal_geometry_report",
]
