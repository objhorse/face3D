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


def bundle_adjust_three_view_cameras(tracks, cameras, fitted):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from src.cross_view_geometry import relative_camera_transform, triangulate_correspondences

    if len(tracks) < 6:
        return fitted, {"accepted": False, "reason": "too_few_three_view_tracks"}
    sides = ("left", "right")
    initial_rvecs, initial_translations = {}, {}
    for side in sides:
        rotation, translation = relative_camera_transform(cameras["front"], cameras[side])
        initial_rvecs[side] = Rotation.from_matrix(rotation).as_rotvec()
        initial_translations[side] = np.asarray(translation, dtype=np.float64)

    initial_points = []
    for track in tracks:
        estimates = []
        for side in sides:
            rotation = Rotation.from_rotvec(initial_rvecs[side]).as_matrix()
            xyz, _positive = triangulate_correspondences(
                np.asarray(track.pixels_by_view["front"])[None],
                np.asarray(track.pixels_by_view[side])[None],
                cameras["front"].K, cameras[side].K,
                rotation, initial_translations[side],
            )
            if np.isfinite(xyz[0]).all():
                estimates.append(xyz[0])
        initial_points.append(np.mean(estimates, axis=0) if estimates else np.array([0.0, 0.0, 0.3]))
    initial_points = np.asarray(initial_points, dtype=np.float64)
    x0 = np.concatenate([
        initial_rvecs["left"], initial_translations["left"],
        initial_rvecs["right"], initial_translations["right"],
        initial_points.reshape(-1),
    ])

    def unpack(values):
        poses = {
            "left": (Rotation.from_rotvec(values[0:3]).as_matrix(), values[3:6]),
            "right": (Rotation.from_rotvec(values[6:9]).as_matrix(), values[9:12]),
        }
        return poses, values[12:].reshape(-1, 3)

    def project(point, view, poses):
        camera_point = point if view == "front" else poses[view][0] @ point + poses[view][1]
        homogeneous = cameras[view].K @ camera_point
        return homogeneous[:2] / max(float(homogeneous[2]), 1e-8)

    def residual(values):
        poses, points = unpack(values)
        terms = []
        for track, point in zip(tracks, points):
            for view in ("front", "left", "right"):
                terms.extend((project(point, view, poses) - track.pixels_by_view[view]) / 2.0)
            terms.append((point[2] - np.clip(point[2], 0.12, 1.5)) / 0.01)
        for side in sides:
            rotation, translation = poses[side]
            current_rvec = Rotation.from_matrix(rotation).as_rotvec()
            terms.extend((current_rvec - initial_rvecs[side]) / np.deg2rad(3.0))
            terms.extend((translation - initial_translations[side]) / 0.008)
            terms.append((np.linalg.norm(translation) - np.linalg.norm(initial_translations[side])) / 0.002)
        return np.asarray(terms, dtype=np.float64)

    before = residual(x0)
    result = least_squares(residual, x0, loss="soft_l1", f_scale=1.0, max_nfev=300)
    poses, _points = unpack(result.x)
    front_rotation = np.asarray(fitted["front"]["R"], dtype=np.float64)
    front_translation = np.asarray(fitted["front"]["t"], dtype=np.float64)
    adjusted = {view: {key: np.asarray(value).copy() for key, value in data.items()} for view, data in fitted.items()}
    pose_changes = {}
    for side in sides:
        relative_rotation, relative_translation = poses[side]
        adjusted[side]["R"] = relative_rotation @ front_rotation
        adjusted[side]["t"] = relative_rotation @ front_translation + relative_translation
        pose_changes[side] = {
            "rotation_delta_deg": float(np.degrees(np.linalg.norm(
                Rotation.from_matrix(relative_rotation @ Rotation.from_rotvec(initial_rvecs[side]).as_matrix().T).as_rotvec()
            ))),
            "translation_delta_m": float(np.linalg.norm(relative_translation - initial_translations[side])),
        }
    return adjusted, {
        "accepted": bool(result.success), "status": int(result.status), "message": result.message,
        "tracks": len(tracks), "cost_before": float(np.mean(before ** 2)),
        "cost_after": float(np.mean(residual(result.x) ** 2)), "pose_changes": pose_changes,
    }


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
        CrossViewTrack,
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

    lr_cache = output / f"loftr_left_right_{args.match_size}.npz"
    if lr_cache.exists() and not args.refresh_matches:
        data = np.load(lr_cache)
        lr_raw = tuple(data[key] for key in ("ff", "fs", "fc", "rs", "rf", "rc"))
    else:
        left_small = cv2.resize(working["left"], (args.match_size, args.match_size))
        right_small = cv2.resize(working["right"], (args.match_size, args.match_size))
        lp, rp, lc = run_loftr_matches(matcher, device, left_small, right_small)
        rp2, lp2, rc = run_loftr_matches(matcher, device, right_small, left_small)
        scale = 1024.0 / args.match_size
        lr_raw = (lp * scale, rp * scale, lc, rp2 * scale, lp2 * scale, rc)
        np.savez_compressed(
            lr_cache, ff=lr_raw[0], fs=lr_raw[1], fc=lr_raw[2],
            rs=lr_raw[3], rf=lr_raw[4], rc=lr_raw[5]
        )
    left_right, left_right_audit = filter_reciprocal_pair_matches(
        "left_right", *lr_raw,
        front_semantic_map=semantic["left"], side_semantic_map=semantic["right"]
    )
    pair_audit.extend(left_right_audit)

    from scipy.spatial import cKDTree
    left_pair = pair_matches["left"]
    right_pair = pair_matches["right"]
    left_tree = cKDTree(left_pair.side_points)
    right_tree = cKDTree(right_pair.side_points)
    left_dist, left_index = left_tree.query(left_right.front_points, k=1)
    right_dist, right_index = right_tree.query(left_right.side_points, k=1)
    loop_tracks = []
    loop_audit = []
    for index in range(len(left_right.front_points)):
        li, ri = int(left_index[index]), int(right_index[index])
        front_delta = float(np.linalg.norm(left_pair.front_points[li] - right_pair.front_points[ri]))
        accepted_loop = bool(left_dist[index] <= 3.0 and right_dist[index] <= 3.0 and front_delta <= 3.0)
        loop_audit.append({
            "index": index, "accepted": accepted_loop,
            "left_link_px": float(left_dist[index]), "right_link_px": float(right_dist[index]),
            "front_closure_px": front_delta,
        })
        if accepted_loop:
            loop_tracks.append(CrossViewTrack(
                pixels_by_view={
                    "front": 0.5 * (left_pair.front_points[li] + right_pair.front_points[ri]),
                    "left": 0.5 * (left_pair.side_points[li] + left_right.front_points[index]),
                    "right": 0.5 * (right_pair.side_points[ri] + left_right.side_points[index]),
                },
                confidence_by_pair={
                    "front_left": float(left_pair.confidence[li]),
                    "front_right": float(right_pair.confidence[ri]),
                    "left_right": float(left_right.confidence[index]),
                },
                semantic_region=left_pair.semantic_regions[li],
            ))

    mediapipe_regions = {
        "nose": [1, 2, 4, 5, 6, 19, 94, 168, 195, 197],
        "mouth": [0, 13, 14, 17, 61, 78, 82, 87, 91, 95, 291, 308, 312, 317, 321, 324],
        "upper_face": [33, 133, 263, 362],
        "chin_or_jaw": [152, 175, 199],
    }
    landmark_tracks = []
    landmark_arrays = {}
    for view in ("left", "front", "right"):
        values = np.asarray(preprocessed[view]["landmarks"], dtype=np.float64)
        landmark_arrays[view] = values.reshape(len(values), -1)[:, :2]
    for region, indices in mediapipe_regions.items():
        for landmark_index in indices:
            if any(landmark_index >= len(landmark_arrays[view]) for view in landmark_arrays):
                continue
            pixels = {view: landmark_arrays[view][landmark_index] for view in landmark_arrays}
            trusted = True
            for view, point in pixels.items():
                xy = np.rint(point).astype(int)
                if not (0 <= xy[0] < semantic[view].shape[1] and 0 <= xy[1] < semantic[view].shape[0]):
                    trusted = False
                    break
                if semantic[view][xy[1], xy[0]] == 0:
                    trusted = False
                    break
            if trusted:
                landmark_tracks.append(CrossViewTrack(
                    pixels_by_view=pixels,
                    confidence_by_pair={"semantic_landmark": 0.75},
                    semantic_region=region,
                ))
    loop_tracks.extend(landmark_tracks)
    (output / "three_view_loop_audit.json").write_text(
        json.dumps({
            "loftr_loops": len(loop_tracks) - len(landmark_tracks),
            "semantic_landmark_tracks": len(landmark_tracks),
            "total_tracks": len(loop_tracks),
            "records": loop_audit,
        }, indent=2), encoding="utf-8"
    )

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
    candidate_camera_records, candidate_fitted = fitted_camera_records(joint_camera_path)
    bundle_fitted, bundle_report = bundle_adjust_three_view_cameras(
        loop_tracks, candidate_camera_records, candidate_fitted
    )
    bundle_camera_path = output / "cameras_bundle_adjusted.json"
    bundle_camera_path.write_text(json.dumps({"views": {
        view: {key: np.asarray(value, dtype=np.float64).tolist() for key, value in data.items()}
        for view, data in bundle_fitted.items()
    }}, indent=2), encoding="utf-8")
    pair_tracks = build_front_centered_tracks(pair_matches)
    tracks = loop_tracks
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

    joint_cameras, joint_fitted = fitted_camera_records(bundle_camera_path)
    loop_config = ObservationFilterConfig(
        min_total_observations=12,
        min_observations_per_side=12,
        min_required_regions=2,
    )
    joint_triangulated, joint_triangulation_audit = triangulate_and_filter_tracks(
        tracks, joint_cameras, {view: item["K"] for view, item in joint_fitted.items()},
        loop_config,
    )
    joint_front = joint_fitted["front"]
    joint_attached, joint_attachment_audit = attach_observations_to_mesh(
        joint_triangulated, vertices, faces, joint_front["R"], joint_front["t"],
        joint_front["K"], loop_config
    )
    joint_summary = build_observation_summary(
        joint_attached, joint_attachment_audit, pair_audit, joint_triangulation_audit,
        loop_config,
    )
    write_observation_audit(
        output / "joint_camera_audit", working, pair_audit, joint_triangulation_audit,
        joint_attachment_audit, joint_attached, joint_summary
    )

    candidate_report = {"generated": False, "reason": "three-view loop gate failed", "profiles": []}
    three_view_ready = len(loop_tracks) >= 12 and joint_summary["m1_passed"]
    if three_view_ready:
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

        normalized = list(joint_attached)
        registration = {"mode": "disabled", "reason": "preserve one metric coordinate frame"}
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
        "bundle_adjusted_cameras": str(bundle_camera_path),
        "bundle_adjustment": bundle_report,
        "relative_pose_candidates": pose_candidates,
        "summary": summary,
        "joint_summary": joint_summary,
        "pair_track_count": len(pair_tracks),
        "three_view_loop_count": len(loop_tracks),
        "geometry_candidates": candidate_report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
