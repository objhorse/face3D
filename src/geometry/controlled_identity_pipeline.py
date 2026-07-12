"""Optional session integration for controlled cross-view identity deformation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


def _scaled_matches(matcher, device, image_a, image_b, size=832):
    from src.learned_cross_view_geometry import run_loftr_matches

    a = cv2.resize(image_a, (size, size), interpolation=cv2.INTER_AREA)
    b = cv2.resize(image_b, (size, size), interpolation=cv2.INTER_AREA)
    pa, pb, confidence = run_loftr_matches(matcher, device, a, b)
    scale_a = np.array([image_a.shape[1] / size, image_a.shape[0] / size])
    scale_b = np.array([image_b.shape[1] / size, image_b.shape[0] / size])
    return pa * scale_a, pb * scale_b, confidence


def _write_obj(template: Path, output: Path, vertices: np.ndarray) -> None:
    lines = template.read_text(encoding="utf-8", errors="replace").splitlines()
    result, index = [], 0
    for line in lines:
        if line.startswith("v "):
            result.append("v %.9f %.9f %.9f" % tuple(vertices[index]))
            index += 1
        else:
            result.append(line)
    if index != len(vertices):
        raise ValueError("candidate/template vertex count mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(result) + "\n", encoding="utf-8")


def run_controlled_identity_stage(
    preprocessed_views: dict,
    intrinsics: dict,
    mesh_dir: Path,
    calibration_path: Path,
    artifact_dir: Path,
    device: str = "cuda",
    apply_selected: bool = False,
    selected_profile: str = "likeness",
) -> dict:
    from scipy.spatial import cKDTree
    from src.cross_view_geometry import load_calibration
    from src.geometry.controlled_identity_deformation import default_controlled_identity_profiles, normalize_observation_groups, optimize_controlled_identity_candidate
    from src.geometry.cross_view_surface_observations import attach_observations_to_mesh, build_front_centered_tracks, build_observation_summary, build_trusted_semantic_map, filter_reciprocal_pair_matches, triangulate_and_filter_tracks, write_observation_audit
    from src.geometry.deformation_graph import build_deformation_graph
    from src.geometry.identity_deformation_quality import evaluate_attachment_similarity, select_identity_backtrack
    from src.learned_cross_view_geometry import load_loftr_matcher
    from src.module3_texture import load_cameras, load_mesh_obj

    artifact_dir.mkdir(parents=True, exist_ok=True)
    matcher, match_device = load_loftr_matcher()
    semantic = {name: build_trusted_semantic_map(view) for name, view in preprocessed_views.items()}
    pair_matches, pair_audit = {}, []
    for side in ("left", "right"):
        forward = _scaled_matches(matcher, match_device, preprocessed_views["front"]["image"], preprocessed_views[side]["image"])
        reverse = _scaled_matches(matcher, match_device, preprocessed_views[side]["image"], preprocessed_views["front"]["image"])
        pair, audit = filter_reciprocal_pair_matches(side, forward[0], forward[1], forward[2], reverse[0], reverse[1], reverse[2], semantic["front"], semantic[side])
        pair_matches[side] = pair
        pair_audit.extend(audit)
    tracks = build_front_centered_tracks(pair_matches)
    calibrated = load_calibration(calibration_path)
    cameras_by_view = {camera.view: camera for camera in calibrated.values()}
    fitted = load_cameras(mesh_dir / "cameras.json")
    triangulated, triangulation_audit = triangulate_and_filter_tracks(tracks, cameras_by_view, intrinsics)
    vertices, faces, _uv, _uv_faces = load_mesh_obj(mesh_dir / "face_mesh.obj")
    front = fitted["front"]
    observations, attachment_audit = attach_observations_to_mesh(triangulated, vertices, faces, front["R"], front["t"], front["K"])
    observations, registration_report = normalize_observation_groups(observations)
    summary = build_observation_summary(observations, attachment_audit, pair_audit, triangulation_audit)
    write_observation_audit(artifact_dir, {k: v["image"] for k, v in preprocessed_views.items()}, pair_audit, triangulation_audit, attachment_audit, observations, summary)
    metadata = {"pipeline": "controlled_identity", "m1": summary, "observation_registration": registration_report, "accepted_candidates": {}, "applied": False}
    if not summary["m1_passed"]:
        (artifact_dir / "reconstruction_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata
    points = np.asarray([item.surface_point for item in observations])
    labels_at_points = np.asarray([item.semantic_region for item in observations], dtype=object)
    _distance, nearest = cKDTree(points).query(vertices, k=1)
    regions = labels_at_points[nearest].astype(str)
    graph = build_deformation_graph(vertices, faces, np.arange(len(vertices)), regions, node_count=220, influences=6)
    rows = []
    for profile in default_controlled_identity_profiles():
        raw, optimization = optimize_controlled_identity_candidate(vertices, faces, graph, observations, profile, device=device)
        gate = lambda current: evaluate_attachment_similarity(
            vertices,
            current,
            faces,
            observations,
            min_improvement_ratio=0.03,
            max_after_median_m=0.001,
        )
        candidate, quality, alpha = select_identity_backtrack(vertices, raw, faces, similarity_gate=gate)
        row = {"name": profile.name, "accepted": quality["accepted"], "alpha": alpha, "optimization": optimization, "quality": quality}
        rows.append(row)
        if quality["accepted"]:
            path = artifact_dir / "geometry" / f"face_controlled_{profile.name}.obj"
            _write_obj(mesh_dir / "face_mesh.obj", path, candidate)
            metadata["accepted_candidates"][profile.name] = str(path)
    metadata["profiles"] = rows
    if apply_selected and selected_profile in metadata["accepted_candidates"]:
        shutil.copy2(metadata["accepted_candidates"][selected_profile], mesh_dir / "face_mesh.obj")
        metadata["applied"] = True
        metadata["selected_profile"] = selected_profile
    (artifact_dir / "reconstruction_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
