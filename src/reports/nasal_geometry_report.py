"""Minimal mesh validity and offline diagnostics for nasal shape candidates."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

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


def _array_or_empty(
    value: Any,
    *,
    dtype: Any | None,
    empty_shape: tuple[int, ...],
) -> tuple[np.ndarray, bool]:
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError):
        return np.empty(empty_shape, dtype=dtype or np.float64), False
    return result, True


def _mesh_input_is_safe(vertices: np.ndarray, faces: np.ndarray) -> bool:
    return bool(
        vertices.ndim == 2
        and vertices.shape[1:] == (3,)
        and len(vertices) > 0
        and np.isfinite(vertices).all()
        and faces.ndim == 2
        and faces.shape[1:] == (3,)
        and len(faces) > 0
        and np.issubdtype(faces.dtype, np.integer)
        and int(faces.min()) >= 0
        and int(faces.max()) < len(vertices)
    )


def _index_array_is_valid(indices: np.ndarray, vertex_count: int) -> bool:
    return bool(
        vertex_count > 0
        and indices.ndim == 2
        and indices.shape[1:] == (3,)
        and len(indices) > 0
        and np.issubdtype(indices.dtype, np.integer)
        and int(indices.min()) >= 0
        and int(indices.max()) < vertex_count
    )


def _invalid_mesh_quality(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    valid_indices = _index_array_is_valid(
        faces,
        len(vertices) if vertices.ndim == 2 else 0,
    )
    face_count = int(len(faces)) if faces.ndim >= 1 else 0
    return {
        "label": label,
        "vertex_count": int(len(vertices)) if vertices.ndim >= 1 else 0,
        "face_count": face_count,
        "finite_vertices": bool(
            vertices.ndim == 2
            and vertices.shape[1:] == (3,)
            and len(vertices) > 0
            and np.isfinite(vertices).all()
        ),
        "finite_faces": bool(
            faces.ndim == 2
            and faces.shape[1:] == (3,)
            and len(faces) > 0
            and np.issubdtype(faces.dtype, np.integer)
        ),
        "valid_face_indices": valid_indices,
        "degenerate_faces": face_count,
        "tiny_faces": face_count,
        "extreme_aspect_faces": face_count,
        "boundary_edges": None,
        "nonmanifold_edges": None,
        "component_count": None,
        "watertight": False,
    }


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
    baseline, baseline_converted = _array_or_empty(
        baseline_vertices,
        dtype=np.float64,
        empty_shape=(0, 3),
    )
    candidate, candidate_converted = _array_or_empty(
        candidate_vertices,
        dtype=np.float64,
        empty_shape=(0, 3),
    )
    faces, faces_converted = _array_or_empty(
        baseline_faces,
        dtype=None,
        empty_shape=(0, 3),
    )
    candidate_face_values, candidate_faces_converted = _array_or_empty(
        candidate_faces,
        dtype=None,
        empty_shape=(0, 3),
    )
    baseline_uv, baseline_uv_converted = _array_or_empty(
        baseline_uv_vertices,
        dtype=np.float64,
        empty_shape=(0, 2),
    )
    candidate_uv, candidate_uv_converted = _array_or_empty(
        candidate_uv_vertices,
        dtype=np.float64,
        empty_shape=(0, 2),
    )
    uv_faces, uv_faces_converted = _array_or_empty(
        baseline_uv_faces,
        dtype=None,
        empty_shape=(0, 3),
    )
    candidate_uv_face_values, candidate_uv_faces_converted = _array_or_empty(
        candidate_uv_faces,
        dtype=None,
        empty_shape=(0, 3),
    )
    coefficient_values, parameters_converted = _array_or_empty(
        parameters,
        dtype=np.float64,
        empty_shape=(0,),
    )
    baseline_vertex_count = (
        len(baseline)
        if baseline.ndim == 2 and baseline.shape[1:] == (3,)
        else 0
    )
    candidate_vertex_count = (
        len(candidate)
        if candidate.ndim == 2 and candidate.shape[1:] == (3,)
        else 0
    )
    baseline_uv_vertex_count = (
        len(baseline_uv)
        if baseline_uv.ndim == 2 and baseline_uv.shape[1:] == (2,)
        else 0
    )
    candidate_uv_vertex_count = (
        len(candidate_uv)
        if candidate_uv.ndim == 2 and candidate_uv.shape[1:] == (2,)
        else 0
    )

    issues: list[str] = []
    finite_parameters = bool(
        parameters_converted
        and coefficient_values.ndim == 1
        and np.isfinite(coefficient_values).all()
    )
    finite_vertices = bool(
        baseline_converted
        and candidate_converted
        and baseline.ndim == 2
        and baseline.shape[1:] == (3,)
        and len(baseline) > 0
        and candidate.shape == baseline.shape
        and np.isfinite(baseline).all()
        and np.isfinite(candidate).all()
    )
    if not finite_parameters:
        issues.append("parameters_contain_nan_or_inf")
    if not finite_vertices:
        issues.append("vertices_are_nonfinite_or_have_changed_count")

    same_faces = bool(
        faces_converted
        and candidate_faces_converted
        and faces.ndim == 2
        and faces.shape[1:] == (3,)
        and np.issubdtype(faces.dtype, np.integer)
        and np.issubdtype(candidate_face_values.dtype, np.integer)
        and candidate_face_values.shape == faces.shape
        and np.array_equal(candidate_face_values, faces)
        and _index_array_is_valid(faces, baseline_vertex_count)
        and _index_array_is_valid(
            candidate_face_values,
            candidate_vertex_count,
        )
    )
    same_uv_vertices = bool(
        baseline_uv_converted
        and candidate_uv_converted
        and baseline_uv.ndim == 2
        and baseline_uv.shape[1:] == (2,)
        and len(baseline_uv) > 0
        and candidate_uv.shape == baseline_uv.shape
        and np.isfinite(baseline_uv).all()
        and np.isfinite(candidate_uv).all()
        and np.array_equal(candidate_uv, baseline_uv)
    )
    same_uv_faces = bool(
        uv_faces_converted
        and candidate_uv_faces_converted
        and uv_faces.ndim == 2
        and uv_faces.shape[1:] == (3,)
        and np.issubdtype(uv_faces.dtype, np.integer)
        and np.issubdtype(candidate_uv_face_values.dtype, np.integer)
        and candidate_uv_face_values.shape == uv_faces.shape
        and np.array_equal(candidate_uv_face_values, uv_faces)
        and _index_array_is_valid(uv_faces, baseline_uv_vertex_count)
        and _index_array_is_valid(
            candidate_uv_face_values,
            candidate_uv_vertex_count,
        )
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
    baseline_mesh_safe = _mesh_input_is_safe(baseline, faces)
    candidate_mesh_safe = _mesh_input_is_safe(
        candidate,
        candidate_face_values,
    )
    quality_baseline = (
        compute_mesh_quality(
            baseline,
            faces,
            label="baseline",
            thresholds=thresholds,
        )
        if baseline_mesh_safe
        else _invalid_mesh_quality(baseline, faces, label="baseline")
    )
    quality_candidate = (
        compute_mesh_quality(
            candidate,
            candidate_face_values,
            label="candidate",
            thresholds=thresholds,
        )
        if candidate_mesh_safe
        else _invalid_mesh_quality(
            candidate,
            candidate_face_values,
            label="candidate",
        )
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
    if finite_vertices and same_faces and baseline_mesh_safe:
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
            "same_vertex_count": bool(
                baseline_converted
                and candidate_converted
                and baseline.ndim == 2
                and candidate.shape == baseline.shape
            ),
            "same_faces": same_faces,
            "same_uv_vertices": same_uv_vertices,
            "same_uv_faces": same_uv_faces,
            "nonempty_vertices": bool(
                baseline.ndim == 2
                and candidate.ndim == 2
                and len(baseline) > 0
                and len(candidate) > 0
            ),
            "nonempty_faces": bool(
                faces.ndim == 2
                and candidate_face_values.ndim == 2
                and len(faces) > 0
                and len(candidate_face_values) > 0
            ),
            "valid_face_indices": bool(
                _index_array_is_valid(faces, baseline_vertex_count)
                and _index_array_is_valid(
                    candidate_face_values,
                    candidate_vertex_count,
                )
            ),
            "nonempty_uv_vertices": bool(
                baseline_uv.ndim == 2
                and candidate_uv.ndim == 2
                and len(baseline_uv) > 0
                and len(candidate_uv) > 0
            ),
            "nonempty_uv_faces": bool(
                uv_faces.ndim == 2
                and candidate_uv_face_values.ndim == 2
                and len(uv_faces) > 0
                and len(candidate_uv_face_values) > 0
            ),
            "valid_uv_face_indices": bool(
                _index_array_is_valid(
                    uv_faces,
                    baseline_uv_vertex_count,
                )
                and _index_array_is_valid(
                    candidate_uv_face_values,
                    candidate_uv_vertex_count,
                )
            ),
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


def _uniform_geometry_scene(trimesh_scene: Any, pyrender: Any) -> Any:
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.72, 0.76, 0.80, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.85,
    )
    scene = pyrender.Scene(
        bg_color=np.array([15, 20, 27, 255]),
        ambient_light=np.array([0.55, 0.55, 0.55]),
    )
    for node_name in trimesh_scene.graph.nodes_geometry:
        transform, geometry_name = trimesh_scene.graph[node_name]
        mesh = trimesh_scene.geometry[geometry_name]
        scene.add(
            pyrender.Mesh.from_trimesh(
                mesh,
                material=material,
                smooth=True,
            ),
            pose=np.asarray(transform, dtype=np.float64),
        )
    return scene


def _render_glb(source: Path, output: Path, yaw_degrees: float) -> None:
    import pyrender
    import trimesh
    from PIL import Image

    trimesh_scene = trimesh.load(source, force="scene")
    if not trimesh_scene.geometry:
        raise ValueError(f"GLB contains no geometry: {source}")
    scene = _uniform_geometry_scene(trimesh_scene, pyrender)
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


def write_nasal_evidence_overlays(
    output_dir: str | Path,
    *,
    work_images_by_view: Mapping[str, Any],
    observation_curves_by_view: Mapping[str, Mapping[str, Any]],
    baseline_projection_by_view: Mapping[str, Any],
    candidate_projection_by_view: Mapping[str, Any],
) -> dict[str, Path]:
    """Overlay observed and projected geometry on the three work-frame images."""
    from PIL import Image, ImageDraw

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for semantic_view in _SEMANTIC_YAWS:
        if semantic_view not in work_images_by_view:
            raise ValueError(f"missing original-image context for {semantic_view}")
        image = np.asarray(work_images_by_view[semantic_view])
        if (
            image.ndim != 3
            or image.shape[2] not in (3, 4)
            or not np.isfinite(image).all()
        ):
            raise ValueError(f"{semantic_view} work image is invalid")
        canvas = Image.fromarray(np.asarray(image[:, :, :3], dtype=np.uint8))
        draw = ImageDraw.Draw(canvas)
        curves = observation_curves_by_view.get(semantic_view, {})
        for points in curves.values():
            values = np.asarray(points, dtype=np.float64)
            if values.ndim != 2 or values.shape[1:] != (2,) or not np.isfinite(values).all():
                raise ValueError(f"{semantic_view} observation curve is invalid")
            if len(values) >= 2:
                draw.line(
                    [tuple(point) for point in values],
                    fill=(55, 217, 138),
                    width=3,
                )
        for projected, color in (
            (baseline_projection_by_view, (88, 200, 255)),
            (candidate_projection_by_view, (255, 111, 174)),
        ):
            values = np.asarray(projected.get(semantic_view, ()), dtype=np.float64)
            if values.ndim != 2 or values.shape[1:] != (2,) or not np.isfinite(values).all():
                raise ValueError(f"{semantic_view} projected points are invalid")
            for x, y in values:
                radius = 2.5
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=color,
                )
        output = target / f"evidence_{semantic_view}.png"
        canvas.save(output)
        result[semantic_view] = output
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
    evidence_overlays: Mapping[str, str | Path],
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
        evidence_overlay = Path(evidence_overlays[semantic_view])
        for path in (baseline, candidate, evidence_overlay):
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"geometry evidence image missing: {path}")
        screenshot_cells.append(
            f"""
            <section>
              <h2>{html.escape(semantic_view)}</h2>
              <h3>Original-image projection evidence</h3>
              <figure class="evidence">
                <img src="{html.escape(evidence_overlay.name)}">
                <figcaption>
                  <span class="observation">Observation</span>
                  <span class="baseline">Baseline projection</span>
                  <span class="candidate">Candidate projection</span>
                </figcaption>
              </figure>
              <h3>Uniform-material geometry renders</h3>
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
        "geometry_render_material": "uniform_untextured",
        "original_images_used_for_context_only": True,
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
    h3 {{ font-size: 15px; margin: 16px 0 8px; }}
    .note {{ color: #b8c3cf; }}
    .pair {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    figure {{ margin: 0; background: #171e27; border: 1px solid #33404d; padding: 8px; }}
    .evidence {{ max-width: 760px; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 7px 2px 1px; }}
    figcaption span {{ margin-right: 16px; font-weight: bold; }}
    .observation {{ color: #55d98a; }}
    .baseline {{ color: #58c8ff; }}
    .candidate {{ color: #ff6fae; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ padding: 7px 8px; border: 1px solid #33404d; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    pre {{ overflow: auto; background: #171e27; padding: 12px; border: 1px solid #33404d; }}
    @media (max-width: 720px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body><main>
  <h1>{html.escape(dataset_label)}: nasal geometry</h1>
  <p class="note">Geometry renders use one untextured material. Original images appear only behind projection evidence; texture scoring is not part of the objective or validity gate.</p>
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
    "write_nasal_evidence_overlays",
    "write_nasal_geometry_report",
]
