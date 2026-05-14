from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


CAMERA_TO_VIEW = {
    "camera1": "left",
    "camera2": "front",
    "camera3": "right",
}


def _json_array(value):
    return np.asarray(value, dtype=float).tolist()


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((11, 8), 0.015, 0.011, dictionary)
    board.setLegacyPattern(True)
    return board, cv2.aruco.CharucoDetector(board)


def collect_detections(captures_dir: Path, min_corners: int):
    _board, detector = make_board()
    detections = {}
    stats = {}

    for image_path in sorted(captures_dir.rglob("*.jpg")):
        camera_name = image_path.parent.name
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
        marker_count = 0 if marker_ids is None else len(marker_ids)
        corner_count = 0 if ids is None else len(ids)
        image_size = (gray.shape[1], gray.shape[0])
        stats.setdefault(camera_name, []).append(
            {
                "file": image_path.name,
                "markers": marker_count,
                "charuco_corners": corner_count,
                "usable": corner_count >= min_corners,
            }
        )
        if corner_count >= min_corners:
            detections[(camera_name, image_path.name)] = {
                "corners": corners,
                "ids": ids,
                "image_size": image_size,
            }

    return detections, stats


def calibrate_intrinsics(board, detections, camera_name):
    names = sorted(name for cam, name in detections if cam == camera_name)
    corners = [detections[(camera_name, name)]["corners"] for name in names]
    ids = [detections[(camera_name, name)]["ids"] for name in names]
    image_size = detections[(camera_name, names[0])]["image_size"]
    rms, k_mat, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        corners,
        ids,
        board,
        image_size,
        None,
        None,
    )
    return {
        "rms": float(rms),
        "K": k_mat,
        "dist": dist.reshape(-1),
        "image_size": image_size,
        "frames": names,
        "rvecs": rvecs,
        "tvecs": tvecs,
    }


def shared_points(board, detections, cam_a, cam_b, min_corners):
    board_points = board.getChessboardCorners()
    names_a = {name for cam, name in detections if cam == cam_a}
    names_b = {name for cam, name in detections if cam == cam_b}
    object_points = []
    image_points_a = []
    image_points_b = []
    used_names = []

    for name in sorted(names_a & names_b):
        det_a = detections[(cam_a, name)]
        det_b = detections[(cam_b, name)]
        pts_a = {int(idx): det_a["corners"][i, 0] for i, idx in enumerate(det_a["ids"].ravel())}
        pts_b = {int(idx): det_b["corners"][i, 0] for i, idx in enumerate(det_b["ids"].ravel())}
        shared = sorted(set(pts_a) & set(pts_b))
        if len(shared) < min_corners:
            continue
        object_points.append(np.array([board_points[idx] for idx in shared], dtype=np.float32))
        image_points_a.append(np.array([pts_a[idx] for idx in shared], dtype=np.float32))
        image_points_b.append(np.array([pts_b[idx] for idx in shared], dtype=np.float32))
        used_names.append(name)

    return object_points, image_points_a, image_points_b, used_names


def stereo_pair(board, detections, intrinsics, cam_a, cam_b, min_corners):
    obj, img_a, img_b, names = shared_points(board, detections, cam_a, cam_b, min_corners)
    if len(obj) < 5:
        raise RuntimeError(f"not enough shared frames for {cam_a}-{cam_b}")

    k_a = intrinsics[cam_a]["K"]
    d_a = intrinsics[cam_a]["dist"]
    k_b = intrinsics[cam_b]["K"]
    d_b = intrinsics[cam_b]["dist"]
    image_size = intrinsics[cam_a]["image_size"]
    rms, _k1, _d1, _k2, _d2, r_mat, t_vec, _e, _f = cv2.stereoCalibrate(
        obj,
        img_a,
        img_b,
        k_a,
        d_a,
        k_b,
        d_b,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    return {
        "rms": float(rms),
        "R_a_to_b": r_mat,
        "t_a_to_b": t_vec.reshape(3),
        "frames": names,
    }


def invert_transform(r_mat, t_vec):
    inv_r = r_mat.T
    inv_t = -inv_r @ np.asarray(t_vec).reshape(3)
    return inv_r, inv_t


def estimate_board_poses(board, detections, intrinsics):
    poses = {}
    obj_points = board.getChessboardCorners()
    for (camera_name, frame_name), det in detections.items():
        k_mat = intrinsics[camera_name]["K"]
        dist = intrinsics[camera_name]["dist"]
        ids = det["ids"].ravel().astype(int)
        object_points = np.array([obj_points[idx] for idx in ids], dtype=np.float32)
        image_points = det["corners"].reshape(-1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            k_mat,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        r_mat, _ = cv2.Rodrigues(rvec)
        poses[(camera_name, frame_name)] = {
            "R_board_to_camera": r_mat,
            "t_board_to_camera": tvec.reshape(3),
        }
    return poses


def rotation_angle_deg(r_mat):
    value = (np.trace(r_mat) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def relative_pose_samples(poses, reference, camera_name):
    names_ref = {name for cam, name in poses if cam == reference}
    names_cam = {name for cam, name in poses if cam == camera_name}
    samples = []
    for frame_name in sorted(names_ref & names_cam):
        ref_pose = poses[(reference, frame_name)]
        cam_pose = poses[(camera_name, frame_name)]
        ref_r = ref_pose["R_board_to_camera"]
        ref_t = ref_pose["t_board_to_camera"]
        cam_r = cam_pose["R_board_to_camera"]
        cam_t = cam_pose["t_board_to_camera"]
        rig_to_cam_r = cam_r @ ref_r.T
        rig_to_cam_t = cam_t - rig_to_cam_r @ ref_t
        samples.append((frame_name, rig_to_cam_r, rig_to_cam_t))
    return samples


def summarize_pose_samples(samples):
    if not samples:
        return {}
    ref_name, ref_r, ref_t = samples[0]
    trans = np.array([sample[2] for sample in samples])
    rot_delta = [
        rotation_angle_deg(sample[1] @ ref_r.T)
        for sample in samples
    ]
    return {
        "reference_sample": ref_name,
        "translation_mean_m": trans.mean(axis=0),
        "translation_std_m": trans.std(axis=0),
        "rotation_delta_mean_deg": float(np.mean(rot_delta)),
        "rotation_delta_max_deg": float(np.max(rot_delta)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", type=Path, default=Path("captures"))
    parser.add_argument("--output", type=Path, default=Path("config/camera_calibration.json"))
    parser.add_argument("--min-corners", type=int, default=12)
    args = parser.parse_args()

    board, _detector = make_board()
    detections, stats = collect_detections(args.captures, args.min_corners)
    cameras = sorted({camera for camera, _name in detections})
    if not cameras:
        raise RuntimeError(f"No usable ChArUco detections in {args.captures}")

    intrinsics = {}
    for camera_name in cameras:
        intrinsics[camera_name] = calibrate_intrinsics(board, detections, camera_name)

    reference = "camera2" if "camera2" in cameras else cameras[0]
    board_poses = estimate_board_poses(board, detections, intrinsics)
    rig_extrinsics = {
        reference: {
            "rig_to_camera": {"R": np.eye(3), "t": np.zeros(3)},
            "camera_to_rig": {"R": np.eye(3), "t": np.zeros(3)},
            "source": "reference",
            "stereo_rms": 0.0,
        }
    }
    stereo = {}
    pose_consistency = {}
    for camera_name in cameras:
        if camera_name == reference:
            continue
        samples = relative_pose_samples(board_poses, reference, camera_name)
        pose_consistency[f"{reference}_{camera_name}"] = summarize_pose_samples(samples)
        pair = stereo_pair(board, detections, intrinsics, reference, camera_name, args.min_corners)
        stereo[f"{reference}_{camera_name}"] = pair
        rig_to_cam_r = pair["R_a_to_b"]
        rig_to_cam_t = pair["t_a_to_b"]
        cam_to_rig_r, cam_to_rig_t = invert_transform(rig_to_cam_r, rig_to_cam_t)
        rig_extrinsics[camera_name] = {
            "rig_to_camera": {"R": rig_to_cam_r, "t": rig_to_cam_t},
            "camera_to_rig": {"R": cam_to_rig_r, "t": cam_to_rig_t},
            "source": f"stereo:{reference}->{camera_name}",
            "stereo_rms": pair["rms"],
        }

    result = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "captures_dir": str(args.captures),
        "reference_camera": reference,
        "units": "meters",
        "board": {
            "type": "ChArUco",
            "squares_x": 11,
            "squares_y": 8,
            "square_size_m": 0.015,
            "marker_size_m": 0.011,
            "dictionary": "DICT_4X4_50",
            "legacy_pattern": True,
            "source_pdf": "calib.io_charuco_200x150_8x11_15_11_DICT_4X4.pdf",
        },
        "view_aliases": CAMERA_TO_VIEW,
        "cameras": {},
        "stereo_pairs": {},
        "pose_consistency": {},
        "detection_summary": stats,
    }

    for camera_name in cameras:
        data = intrinsics[camera_name]
        ext = rig_extrinsics[camera_name]
        result["cameras"][camera_name] = {
            "view": CAMERA_TO_VIEW.get(camera_name, camera_name),
            "image_size": list(data["image_size"]),
            "rms": data["rms"],
            "K": _json_array(data["K"]),
            "dist_coeffs": _json_array(data["dist"]),
            "usable_frames": len(data["frames"]),
            "rig_to_camera": {
                "R": _json_array(ext["rig_to_camera"]["R"]),
                "t": _json_array(ext["rig_to_camera"]["t"]),
            },
            "camera_to_rig": {
                "R": _json_array(ext["camera_to_rig"]["R"]),
                "t": _json_array(ext["camera_to_rig"]["t"]),
            },
            "extrinsic_source": ext["source"],
            "stereo_rms": ext["stereo_rms"],
        }

    for pair_name, pair in stereo.items():
        result["stereo_pairs"][pair_name] = {
            "rms": pair["rms"],
            "frames": pair["frames"],
            "R_a_to_b": _json_array(pair["R_a_to_b"]),
            "t_a_to_b": _json_array(pair["t_a_to_b"]),
        }
    for pair_name, summary in pose_consistency.items():
        result["pose_consistency"][pair_name] = {
            key: (_json_array(value) if isinstance(value, np.ndarray) else value)
            for key, value in summary.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    for camera_name, data in result["cameras"].items():
        k_mat = np.asarray(data["K"])
        print(
            f"{camera_name}/{data['view']}: rms={data['rms']:.4f}, "
            f"fx={k_mat[0,0]:.2f}, fy={k_mat[1,1]:.2f}, "
            f"cx={k_mat[0,2]:.2f}, cy={k_mat[1,2]:.2f}, "
            f"frames={data['usable_frames']}"
        )
    for pair_name, data in result["stereo_pairs"].items():
        print(f"{pair_name}: stereo_rms={data['rms']:.4f}, frames={len(data['frames'])}")
    for pair_name, data in result["pose_consistency"].items():
        print(
            f"{pair_name}: pose_std_m={data.get('translation_std_m')}, "
            f"rot_delta_max_deg={data.get('rotation_delta_max_deg')}"
        )


if __name__ == "__main__":
    main()
