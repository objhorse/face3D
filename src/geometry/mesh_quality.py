"""Mesh quality metrics and hard gates for stable reconstruction.

The stable pipeline uses these checks as a guardrail: texture or visibility
problems may be reported, but they must not be solved by deleting central face
geometry or by exporting a broken mesh as successful.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class MeshQualityThresholds:
    min_face_ratio: float = 0.98
    degenerate_area_eps: float = 1e-10
    tiny_area_eps: float = 1e-8
    extreme_aspect_ratio: float = 50.0
    max_new_degenerate_faces: int = 0
    max_new_nonmanifold_edges: int = 0
    max_new_boundary_edges: int = 0


def _as_array(data: Any, dtype: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=dtype)
    return arr.copy() if not arr.flags.c_contiguous else arr


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros(0, dtype=np.float64)
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def _face_aspect_ratios(vertices: np.ndarray, faces: np.ndarray, area: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros(0, dtype=np.float64)
    tri = vertices[faces]
    e0 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    e1 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    e2 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    longest = np.maximum(np.maximum(e0, e1), e2)
    shortest_altitude = np.divide(2.0 * area, longest, out=np.zeros_like(area), where=longest > 0)
    return np.divide(longest, shortest_altitude, out=np.full_like(area, np.inf), where=shortest_altitude > 0)


def _edge_counts(faces: np.ndarray) -> Dict[tuple[int, int], int]:
    counts: Dict[tuple[int, int], int] = {}
    if len(faces) == 0:
        return counts
    for a, b, c in faces.astype(np.int64, copy=False):
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (int(u), int(v)) if u < v else (int(v), int(u))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def _connected_components(faces: np.ndarray) -> int:
    if len(faces) == 0:
        return 0
    parent = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, c in faces.astype(np.int64, copy=False):
        union(int(a), int(b))
        union(int(b), int(c))
    used = set(int(x) for x in faces.reshape(-1))
    return len({find(x) for x in used})


def compute_mesh_quality(
    vertices: Any,
    faces: Any,
    *,
    label: str = "",
    thresholds: MeshQualityThresholds | None = None,
) -> Dict[str, Any]:
    thresholds = thresholds or MeshQualityThresholds()
    verts = _as_array(vertices, np.float64)
    face_idx = _as_array(faces, np.int64)

    report: Dict[str, Any] = {
        "label": label,
        "vertex_count": int(len(verts)),
        "face_count": int(len(face_idx)),
        "finite_vertices": bool(np.isfinite(verts).all()),
        "finite_faces": bool(np.isfinite(face_idx).all()) if face_idx.size else True,
        "valid_face_indices": True,
    }

    if face_idx.size:
        report["valid_face_indices"] = bool(face_idx.min() >= 0 and face_idx.max() < len(verts))

    if (not report["finite_vertices"]) or (not report["finite_faces"]) or (not report["valid_face_indices"]):
        report.update({
            "degenerate_faces": int(len(face_idx)),
            "tiny_faces": int(len(face_idx)),
            "extreme_aspect_faces": int(len(face_idx)),
            "boundary_edges": None,
            "nonmanifold_edges": None,
            "component_count": None,
            "watertight": False,
        })
        return report

    area = _face_areas(verts, face_idx)
    aspect = _face_aspect_ratios(verts, face_idx, area)
    edge_counts = _edge_counts(face_idx)
    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    nonmanifold_edges = sum(1 for count in edge_counts.values() if count > 2)

    report.update({
        "degenerate_faces": int(np.count_nonzero(area <= thresholds.degenerate_area_eps)),
        "tiny_faces": int(np.count_nonzero(area <= thresholds.tiny_area_eps)),
        "min_face_area": float(area.min()) if len(area) else 0.0,
        "mean_face_area": float(area.mean()) if len(area) else 0.0,
        "max_aspect_ratio": float(np.nanmax(aspect)) if len(aspect) else 0.0,
        "extreme_aspect_faces": int(np.count_nonzero(aspect > thresholds.extreme_aspect_ratio)),
        "boundary_edges": int(boundary_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
        "component_count": int(_connected_components(face_idx)),
        "watertight": bool(boundary_edges == 0 and nonmanifold_edges == 0),
    })
    return report


def _load_trimesh(path: Path):
    import trimesh

    loaded = trimesh.load(str(path), force="mesh", process=False)
    if hasattr(loaded, "geometry"):
        meshes = [g for g in loaded.geometry.values() if hasattr(g, "faces") and len(g.faces)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    return loaded


def load_mesh_quality(
    mesh_path: str | Path,
    *,
    label: str | None = None,
    thresholds: MeshQualityThresholds | None = None,
) -> Dict[str, Any]:
    path = Path(mesh_path)
    mesh = _load_trimesh(path)
    raw_vertex_count = int(len(mesh.vertices))
    raw_face_count = int(len(mesh.faces))
    try:
        mesh = mesh.copy()
        try:
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
        except TypeError:
            mesh.merge_vertices()
    except Exception:
        pass
    report = compute_mesh_quality(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        label=label or path.name,
        thresholds=thresholds,
    )
    report["path"] = str(path)
    report["raw_vertex_count"] = raw_vertex_count
    report["raw_face_count"] = raw_face_count
    report["topology_vertex_count_after_merge"] = int(len(mesh.vertices))
    return report


def compare_mesh_quality(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    thresholds: MeshQualityThresholds | None = None,
    region_name: str = "full_mesh",
) -> Dict[str, Any]:
    thresholds = thresholds or MeshQualityThresholds()
    baseline_faces = int(baseline.get("face_count") or 0)
    candidate_faces = int(candidate.get("face_count") or 0)
    face_ratio = (candidate_faces / baseline_faces) if baseline_faces else 0.0

    issues = []
    warnings = []

    if not candidate.get("finite_vertices", False):
        issues.append("candidate_vertices_contain_nan_or_inf")
    if not candidate.get("finite_faces", False):
        issues.append("candidate_faces_contain_nan_or_inf")
    if not candidate.get("valid_face_indices", False):
        issues.append("candidate_faces_reference_missing_vertices")
    if baseline_faces and face_ratio < thresholds.min_face_ratio:
        issues.append("face_count_dropped_below_threshold")

    base_degen = int(baseline.get("degenerate_faces") or 0)
    cand_degen = int(candidate.get("degenerate_faces") or 0)
    if cand_degen > base_degen + thresholds.max_new_degenerate_faces:
        issues.append("degenerate_faces_increased")

    base_nonmanifold = int(baseline.get("nonmanifold_edges") or 0)
    cand_nonmanifold = int(candidate.get("nonmanifold_edges") or 0)
    if cand_nonmanifold > base_nonmanifold + thresholds.max_new_nonmanifold_edges:
        issues.append("nonmanifold_edges_increased")

    base_boundary = baseline.get("boundary_edges")
    cand_boundary = candidate.get("boundary_edges")
    if base_boundary is not None and cand_boundary is not None:
        if int(cand_boundary) > int(base_boundary) + thresholds.max_new_boundary_edges:
            issues.append("boundary_edges_increased")

    base_extreme = int(baseline.get("extreme_aspect_faces") or 0)
    cand_extreme = int(candidate.get("extreme_aspect_faces") or 0)
    if cand_extreme > max(base_extreme + 50, int(base_extreme * 1.10) + 1):
        warnings.append("extreme_aspect_faces_increased")

    return {
        "region": region_name,
        "passed": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "face_ratio": float(face_ratio),
        "face_count_delta": int(candidate_faces - baseline_faces),
        "degenerate_delta": int(cand_degen - base_degen),
        "nonmanifold_edge_delta": int(cand_nonmanifold - base_nonmanifold),
        "boundary_edge_delta": (
            int(cand_boundary) - int(base_boundary)
            if base_boundary is not None and cand_boundary is not None
            else None
        ),
    }


def make_quality_gate(
    *,
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    thresholds: MeshQualityThresholds | None = None,
    region_name: str = "full_mesh",
) -> Dict[str, Any]:
    comparison = compare_mesh_quality(
        baseline,
        candidate,
        thresholds=thresholds,
        region_name=region_name,
    )
    return {
        "passed": comparison["passed"],
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }


def assert_quality_gate(gate: Dict[str, Any], *, context: str = "mesh quality") -> None:
    if gate.get("passed"):
        return
    issues: Iterable[str] = gate.get("comparison", {}).get("issues", [])
    raise RuntimeError(f"{context} failed: {', '.join(issues)}")
