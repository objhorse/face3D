"""Run cross-view identity observations against an existing stable reconstruction.

This entry point deliberately uses the fitted cameras stored beside the mesh as a
single shared coordinate system. It does not use the currently unreliable hard-rig
extrinsics and never overwrites the source reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def write_obj_with_vertices(template: Path, output: Path, vertices: np.ndarray) -> None:
    result, vertex_index = [], 0
    for line in template.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            result.append("v %.9f %.9f %.9f" % tuple(vertices[vertex_index]))
            vertex_index += 1
        else:
            result.append(line)
    if vertex_index != len(vertices):
        raise ValueError("candidate/template vertex count mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(result) + "\n", encoding="utf-8")


def estimate_relative_pose_candidate(front_points, side_points, front_camera, side_camera):
    from src.cross_view_geometry import relative_camera_transform
    from src.learned_cross_view_geometry import (
        rotation_delta_degrees,
        translation_direction_delta_degrees,
    )

    norm_front = cv2.undistortPoints(
        np.asarray(front_points, np.float64).reshape(-1, 1, 2), front_camera.K, None
    ).reshape(-1, 2)
    norm_side = cv2.undistortPoints(
        np.asarray(side_points, np.float64).reshape(-1, 1, 2), side_camera.K, None
    ).reshape(-1, 2)
    essential, mask = cv2.findEssentialMat(
        norm_front, norm_side, np.eye(3), cv2.RANSAC, 0.999, 0.004
    )
    if essential is None or mask is None or int(mask.sum()) < 8:
        return {"available": False, "inliers": int(mask.sum()) if mask is not None else 0}
    count, rotation, translation, pose_mask = cv2.recoverPose(
        essential, norm_front, norm_side, np.eye(3), mask=mask
    )
    current_rotation, current_translation = relative_camera_transform(front_camera, side_camera)
    direction = translation.reshape(3)
    if np.dot(direction, current_translation) < 0:
        direction = -direction
    direction *= np.linalg.norm(current_translation) / max(np.linalg.norm(direction), 1e-12)
    return {
        "available": True,
        "inliers": int(count),
        "rotation_delta_deg": float(rotation_delta_degrees(rotation, current_rotation)),
        "translation_direction_delta_deg": float(
            translation_direction_delta_degrees(direction, current_translation)
        ),
        "candidate_relative_R": rotation.tolist(),
        "candidate_relative_t": direction.tolist(),
        "current_relative_R": current_rotation.tolist(),
        "current_relative_t": current_translation.tolist(),
    }


def fitted_camera_records(camera_path: Path):
    from src.cross_view_geometry import Camera
    from src.module3_texture import load_cameras

    fitted = load_cameras(camera_path)
    records = {}
    for view, values in fitted.items():
        records[view] = Camera(
            name=view,
            view=view,
            image_size=(1024, 1024),
            K=np.asarray(values["K"], dtype=np.float64),
            dist=np.zeros(5, dtype=np.float64),
            R_rig_to_camera=np.asarray(values["R"], dtype=np.float64),
            t_rig_to_camera=np.asarray(values["t"], dtype=np.float64),
        )
    return records, fitted


def write_joint_camera_candidate(output_path: Path, fitted, pose_candidates):
    front_rotation = np.asarray(fitted["front"]["R"], dtype=np.float64)
    front_translation = np.asarray(fitted["front"]["t"], dtype=np.float64)
    views = {}
    for view, values in fitted.items():
        rotation = np.asarray(values["R"], dtype=np.float64)
        translation = np.asarray(values["t"], dtype=np.float64)
        candidate = pose_candidates.get(view)
        if candidate and candidate.get("available"):
            relative_rotation = np.asarray(candidate["candidate_relative_R"], dtype=np.float64)
            relative_translation = np.asarray(candidate["candidate_relative_t"], dtype=np.float64)
            rotation = relative_rotation @ front_rotation
            translation = relative_rotation @ front_translation + relative_translation
        views[view] = {
            "K": np.asarray(values["K"], dtype=np.float64).tolist(),
            "R": rotation.tolist(),
            "t": translation.tolist(),
        }
    output_path.write_text(json.dumps({"views": views}, indent=2), encoding="utf-8")


def load_capture_images(capture_dir: Path):
    mapping = {"left": "camera1", "front": "camera2", "right": "camera3"}
    images = {}
    paths = {}
    for view, stem in mapping.items():
        candidates = sorted(capture_dir.glob(f"{stem}*.jpg"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected exactly one {stem}*.jpg in {capture_dir}, found {len(candidates)}"
            )
        path = candidates[0]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        images[view] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        paths[view] = path
    return images, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-size", type=int, default=832)
    parser.add_argument("--refresh-matches", action="store_true")
    args = parser.parse_args()

    from src.geometry.cross_view_surface_observations import (
        ObservationFilterConfig,
        attach_observations_to_mesh,
        build_front_centered_tracks,
        build_observation_summary,
        build_trusted_semantic_map,
        filter_reciprocal_pair_matches,
        triangulate_and_filter_tracks,
        write_observation_audit,
    )
    from src.learned_cross_view_geometry import load_loftr_matcher, run_loftr_matches
    from src.module1_preprocess import preprocess_all_views
    from src.module3_texture import load_mesh_obj

    baseline = args.baseline_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh_path = baseline / "meshes" / "face_mesh.obj"
    camera_path = baseline / "meshes" / "cameras.json"
    if not mesh_path.exists() or not camera_path.exists():
        raise FileNotFoundError("baseline must contain meshes/face_mesh.obj and cameras.json")

    images, capture_paths = load_capture_images(args.capture_dir.resolve())
    preprocessed = preprocess_all_views(images, debug_dir=output / "preprocess", target_size=1024)
    working = {view: item["image"] for view, item in preprocessed.items()}
    semantic = {view: build_trusted_semantic_map(item) for view, item in preprocessed.items()}
    matcher, device = load_loftr_matcher()

    pair_matches = {}
    pair_audit = []
    for side in ("left", "right"):
        cache = output / f"loftr_front_{side}_{args.match_size}.npz"
        if cache.exists() and not args.refresh_matches:
            data = np.load(cache)
            raw = tuple(data[key] for key in ("ff", "fs", "fc", "rs", "rf", "rc"))
        else:
            front_small = cv2.resize(working["front"], (args.match_size, args.match_size))
            side_small = cv2.resize(working[side], (args.match_size, args.match_size))
            ff, fs, fc = run_loftr_matches(matcher, device, front_small, side_small)
            rs, rf, rc = run_loftr_matches(matcher, device, side_small, front_small)
            scale = 1024.0 / args.match_size
            raw = (ff * scale, fs * scale, fc, rs * scale, rf * scale, rc)
            np.savez_compressed(cache, ff=raw[0], fs=raw[1], fc=raw[2], rs=raw[3], rf=raw[4], rc=raw[5])
        pair, audit = filter_reciprocal_pair_matches(
            side, *raw, front_semantic_map=semantic["front"], side_semantic_map=semantic[side]
        )
        pair_matches[side] = pair
        pair_audit.extend(audit)

    cameras, fitted = fitted_camera_records(camera_path)
    pose_candidates = {
        side: estimate_relative_pose_candidate(
            pair_matches[side].front_points,
            pair_matches[side].side_points,
            cameras["front"],
            cameras[side],
        )
        for side in ("left", "right")
    }
    joint_camera_path = output / "cameras_joint_candidate.json"
    write_joint_camera_candidate(joint_camera_path, fitted, pose_candidates)
    tracks = build_front_centered_tracks(pair_matches)
    triangulated, triangulation_audit = triangulate_and_filter_tracks(
        tracks, cameras, {view: item["K"] for view, item in fitted.items()}
    )
    vertices, faces, _uv, _uv_faces = load_mesh_obj(mesh_path)
    front = fitted["front"]
    attached, attachment_audit = attach_observations_to_mesh(
        triangulated, vertices, faces, front["R"], front["t"], front["K"], ObservationFilterConfig()
    )
    summary = build_observation_summary(attached, attachment_audit, pair_audit, triangulation_audit)
    write_observation_audit(output, working, pair_audit, triangulation_audit, attachment_audit, attached, summary)

    joint_cameras, joint_fitted = fitted_camera_records(joint_camera_path)
    joint_triangulated, joint_triangulation_audit = triangulate_and_filter_tracks(
        tracks, joint_cameras, {view: item["K"] for view, item in joint_fitted.items()}
    )
    joint_front = joint_fitted["front"]
    joint_attached, joint_attachment_audit = attach_observations_to_mesh(
        joint_triangulated, vertices, faces, joint_front["R"], joint_front["t"],
        joint_front["K"], ObservationFilterConfig()
    )
    joint_summary = build_observation_summary(
        joint_attached, joint_attachment_audit, pair_audit, joint_triangulation_audit
    )
    write_observation_audit(
        output / "joint_camera_audit", working, pair_audit, joint_triangulation_audit,
        joint_attachment_audit, joint_attached, joint_summary
    )

    candidate_report = {"generated": False, "reason": "joint observation gate failed", "profiles": []}
    if joint_summary["m1_passed"]:
        from scipy.spatial import cKDTree
        from src.geometry.controlled_identity_deformation import (
            default_controlled_identity_profiles,
            normalize_observation_groups,
            optimize_controlled_identity_candidate,
        )
        from src.geometry.deformation_graph import build_deformation_graph
        from src.geometry.identity_deformation_quality import (
            evaluate_attachment_similarity,
            select_identity_backtrack,
        )

        normalized, registration = normalize_observation_groups(joint_attached)
        surface_points = np.asarray([item.surface_point for item in normalized])
        labels = np.asarray([item.semantic_region for item in normalized], dtype=object)
        _distance, nearest = cKDTree(surface_points).query(vertices, k=1)
        regions = labels[nearest].astype(str)
        graph = build_deformation_graph(
            vertices, faces, np.arange(len(vertices)), regions, node_count=220, influences=6
        )
        baseline_similarity = evaluate_attachment_similarity(
            vertices, vertices, faces, normalized,
            min_improvement_ratio=0.0, max_after_median_m=float("inf"),
        )
        baseline_attachment_median = float(baseline_similarity["after_median_m"])
        candidate_report = {"generated": True, "observation_registration": registration, "profiles": []}
        for profile in default_controlled_identity_profiles():
            raw, optimization = optimize_controlled_identity_candidate(
                vertices, faces, graph, normalized, profile, device="cuda"
            )
            gate = lambda current: evaluate_attachment_similarity(
                vertices, current, faces, normalized,
                min_improvement_ratio=0.03,
                max_after_median_m=baseline_attachment_median,
            )
            candidate, quality, alpha = select_identity_backtrack(
                vertices, raw, faces, similarity_gate=gate
            )
            row = {"name": profile.name, "accepted": quality["accepted"], "alpha": alpha,
                   "optimization": optimization, "quality": quality}
            if quality["accepted"]:
                path = output / "geometry" / f"face_identity_{profile.name}.obj"
                write_obj_with_vertices(mesh_path, path, candidate)
                row["path"] = str(path)
            candidate_report["profiles"].append(row)
    (output / "experiment_meta.json").write_text(json.dumps({
        "baseline": str(baseline), "capture_dir": str(args.capture_dir.resolve()),
        "capture_paths": {view: str(path) for view, path in capture_paths.items()},
        "camera_source": str(camera_path), "camera_mode": "fixed_fitted_shared_world",
        "joint_camera_candidate": str(joint_camera_path),
        "relative_pose_candidates": pose_candidates,
        "summary": summary,
        "joint_summary": joint_summary,
        "geometry_candidates": candidate_report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
