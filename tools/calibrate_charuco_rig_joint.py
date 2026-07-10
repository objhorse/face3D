from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


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


def load_calibration(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_detections(captures_dir: Path, min_corners: int):
    _board, detector = make_board()
    detections = {}
    stats = {}

    for image_path in sorted(captures_dir.rglob("*.jpg")):
        camera_name = image_path.parent.name
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        corners, ids, _marker_corners, marker_ids = detector.detectBoard(gray)
        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        corner_count = 0 if ids is None else int(len(ids))
        image_size = (gray.shape[1], gray.shape[0])
        stats.setdefault(camera_name, []).append(
            {
                "file": image_path.name,
                "markers": marker_count,
                "charuco_corners": corner_count,
                "usable": corner_count >= min_corners,
            }
        )
        if ids is not None and corner_count >= min_corners:
            detections[(camera_name, image_path.name)] = {
                "corners": corners.reshape(-1, 2).astype(np.float64),
                "ids": ids.ravel().astype(np.int32),
                "image_size": image_size,
            }

    return detections, stats


def camera_intrinsics(calibration: dict, camera_name: str):
    data = calibration["cameras"][camera_name]
    return {
        "K": np.asarray(data["K"], dtype=np.float64),
        "dist": np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1),
        "image_size": tuple(data["image_size"]),
    }


def shared_frame_names(
    detections: dict,
    cameras: Iterable[str],
    min_corners: int,
) -> List[str]:
    sets = []
    for camera_name in cameras:
        names = {
            frame_name
            for cam, frame_name in detections
            if cam == camera_name and len(detections[(cam, frame_name)]["ids"]) >= min_corners
        }
        sets.append(names)
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def solve_board_pose(board_points, detection, intrinsics):
    ids = detection["ids"]
    obj = board_points[ids].astype(np.float64)
    img = detection["corners"].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        img,
        intrinsics["K"],
        intrinsics["dist"],
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return rvec.reshape(3), tvec.reshape(3)


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    r_mat, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    return r_mat


def rotation_angle_deg(r_mat: np.ndarray) -> float:
    value = (np.trace(r_mat) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def medoid_relative_pose(
    board_poses: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    reference: str,
    camera_name: str,
    frame_names: List[str],
):
    samples = []
    for frame_name in frame_names:
        ref_pose = board_poses.get((reference, frame_name))
        cam_pose = board_poses.get((camera_name, frame_name))
        if ref_pose is None or cam_pose is None:
            continue
        ref_r = rodrigues(ref_pose[0])
        cam_r = rodrigues(cam_pose[0])
        ref_t = ref_pose[1]
        cam_t = cam_pose[1]
        rel_r = cam_r @ ref_r.T
        rel_t = cam_t - rel_r @ ref_t
        rel_rvec, _ = cv2.Rodrigues(rel_r)
        samples.append(
            {
                "frame": frame_name,
                "R": rel_r,
                "rvec": rel_rvec.reshape(3),
                "t": rel_t.reshape(3),
            }
        )
    if not samples:
        raise RuntimeError(f"No relative pose samples for {reference}->{camera_name}")

    n = len(samples)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            rot_delta = rotation_angle_deg(samples[i]["R"] @ samples[j]["R"].T)
            trans_delta_mm = float(np.linalg.norm(samples[i]["t"] - samples[j]["t"]) * 1000.0)
            value = rot_delta + trans_delta_mm / 5.0
            dist[i, j] = dist[j, i] = value
    medoid_idx = int(np.argmin(np.median(dist, axis=1)))
    return samples[medoid_idx], samples


def project_camera_points(points_cam: np.ndarray, intrinsics: dict) -> np.ndarray:
    k = intrinsics["K"]
    dist = intrinsics["dist"]
    k1 = dist[0] if len(dist) > 0 else 0.0
    k2 = dist[1] if len(dist) > 1 else 0.0
    p1 = dist[2] if len(dist) > 2 else 0.0
    p2 = dist[3] if len(dist) > 3 else 0.0
    k3 = dist[4] if len(dist) > 4 else 0.0

    z = points_cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    x = points_cam[:, 0] / safe_z
    y = points_cam[:, 1] / safe_z
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    u = k[0, 0] * x_dist + k[0, 2]
    v = k[1, 1] * y_dist + k[1, 2]
    return np.stack([u, v], axis=1)


class JointCalibrationProblem:
    def __init__(
        self,
        calibration: dict,
        detections: dict,
        board_points: np.ndarray,
        reference: str,
        side_cameras: List[str],
        frame_names: List[str],
    ):
        self.calibration = calibration
        self.detections = detections
        self.board_points = np.asarray(board_points, dtype=np.float64)
        self.reference = reference
        self.side_cameras = side_cameras
        self.cameras = [reference] + side_cameras
        self.frame_names = list(frame_names)
        self.intrinsics = {
            camera_name: camera_intrinsics(calibration, camera_name)
            for camera_name in self.cameras
        }
        self.side_param_index = {
            camera_name: idx * 6
            for idx, camera_name in enumerate(side_cameras)
        }
        self.frame_param_start = len(side_cameras) * 6
        self.observations = []
        for frame_idx, frame_name in enumerate(self.frame_names):
            for camera_name in self.cameras:
                det = detections.get((camera_name, frame_name))
                if det is None:
                    continue
                ids = det["ids"]
                obj = self.board_points[ids].astype(np.float64)
                img = det["corners"].astype(np.float64)
                self.observations.append(
                    {
                        "frame_idx": frame_idx,
                        "frame": frame_name,
                        "camera": camera_name,
                        "object_points": obj,
                        "image_points": img,
                    }
                )

    @property
    def n_params(self) -> int:
        return len(self.side_cameras) * 6 + len(self.frame_names) * 6

    @property
    def n_residuals(self) -> int:
        return sum(obs["image_points"].shape[0] * 2 for obs in self.observations)

    def pack_initial(self, board_poses: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        values = np.zeros(self.n_params, dtype=np.float64)
        for camera_name in self.side_cameras:
            medoid, _samples = medoid_relative_pose(
                board_poses,
                self.reference,
                camera_name,
                self.frame_names,
            )
            start = self.side_param_index[camera_name]
            values[start : start + 3] = medoid["rvec"]
            values[start + 3 : start + 6] = medoid["t"]

        for frame_idx, frame_name in enumerate(self.frame_names):
            pose = board_poses.get((self.reference, frame_name))
            if pose is None:
                raise RuntimeError(f"Missing reference board pose for {frame_name}")
            start = self.frame_param_start + frame_idx * 6
            values[start : start + 3] = pose[0]
            values[start + 3 : start + 6] = pose[1]
        return values

    def unpack_side_transform(self, x: np.ndarray, camera_name: str):
        if camera_name == self.reference:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        start = self.side_param_index[camera_name]
        rvec = x[start : start + 3]
        t = x[start + 3 : start + 6]
        return rodrigues(rvec), t

    def unpack_frame_pose(self, x: np.ndarray, frame_idx: int):
        start = self.frame_param_start + frame_idx * 6
        rvec = x[start : start + 3]
        t = x[start + 3 : start + 6]
        return rodrigues(rvec), t

    def residuals(self, x: np.ndarray) -> np.ndarray:
        chunks = []
        for obs in self.observations:
            frame_r, frame_t = self.unpack_frame_pose(x, obs["frame_idx"])
            points_ref = (frame_r @ obs["object_points"].T).T + frame_t.reshape(1, 3)
            cam_r, cam_t = self.unpack_side_transform(x, obs["camera"])
            points_cam = (cam_r @ points_ref.T).T + cam_t.reshape(1, 3)
            projected = project_camera_points(points_cam, self.intrinsics[obs["camera"]])
            chunks.append((projected - obs["image_points"]).reshape(-1))
        return np.concatenate(chunks, axis=0)

    def jac_sparsity(self):
        matrix = lil_matrix((self.n_residuals, self.n_params), dtype=np.int8)
        row = 0
        for obs in self.observations:
            n = obs["image_points"].shape[0] * 2
            frame_start = self.frame_param_start + obs["frame_idx"] * 6
            matrix[row : row + n, frame_start : frame_start + 6] = 1
            if obs["camera"] != self.reference:
                side_start = self.side_param_index[obs["camera"]]
                matrix[row : row + n, side_start : side_start + 6] = 1
            row += n
        return matrix.tocsr()

    def frame_error_rows(self, x: np.ndarray):
        by_frame = {frame_name: [] for frame_name in self.frame_names}
        by_camera = {}
        cursor = 0
        residuals = self.residuals(x)
        for obs in self.observations:
            n_pts = obs["image_points"].shape[0]
            arr = residuals[cursor : cursor + n_pts * 2].reshape(-1, 2)
            err = np.linalg.norm(arr, axis=1)
            by_frame[obs["frame"]].extend(err.tolist())
            by_camera.setdefault(obs["camera"], []).extend(err.tolist())
            cursor += n_pts * 2

        rows = []
        for frame_name, errors in by_frame.items():
            if not errors:
                continue
            arr = np.asarray(errors, dtype=np.float64)
            rows.append(
                {
                    "file": frame_name,
                    "points": int(arr.size),
                    "mean_px": float(arr.mean()),
                    "median_px": float(np.median(arr)),
                    "rms_px": float(np.sqrt(np.mean(arr * arr))),
                    "p90_px": float(np.percentile(arr, 90)),
                    "max_px": float(arr.max()),
                }
            )
        camera_summary = {}
        for camera_name, errors in by_camera.items():
            arr = np.asarray(errors, dtype=np.float64)
            camera_summary[camera_name] = {
                "points": int(arr.size),
                "mean_px": float(arr.mean()),
                "median_px": float(np.median(arr)),
                "rms_px": float(np.sqrt(np.mean(arr * arr))),
                "p90_px": float(np.percentile(arr, 90)),
                "max_px": float(arr.max()),
            }
        return rows, camera_summary

    def extrinsics_from_solution(self, x: np.ndarray):
        extrinsics = {
            self.reference: {
                "R": np.eye(3, dtype=np.float64),
                "t": np.zeros(3, dtype=np.float64),
                "rvec": np.zeros(3, dtype=np.float64),
            }
        }
        for camera_name in self.side_cameras:
            start = self.side_param_index[camera_name]
            rvec = x[start : start + 3]
            t = x[start + 3 : start + 6]
            extrinsics[camera_name] = {
                "R": rodrigues(rvec),
                "t": t.copy(),
                "rvec": rvec.copy(),
            }
        return extrinsics


def aggregate_rows(rows):
    if not rows:
        return {"frames": 0}
    means = np.asarray([row["mean_px"] for row in rows], dtype=np.float64)
    rms = np.asarray([row["rms_px"] for row in rows], dtype=np.float64)
    maxes = np.asarray([row["max_px"] for row in rows], dtype=np.float64)
    return {
        "frames": len(rows),
        "mean_of_frame_mean_px": float(means.mean()),
        "median_of_frame_mean_px": float(np.median(means)),
        "max_frame_mean_px": float(means.max()),
        "mean_of_frame_rms_px": float(rms.mean()),
        "max_frame_rms_px": float(rms.max()),
        "max_point_error_px": float(maxes.max()),
        "frames_mean_lt_5px": int(np.sum(means < 5.0)),
        "frames_mean_lt_3px": int(np.sum(means < 3.0)),
        "frames_mean_lt_2px": int(np.sum(means < 2.0)),
        "frames_rms_lt_5px": int(np.sum(rms < 5.0)),
        "frames_rms_lt_3px": int(np.sum(rms < 3.0)),
        "frames_rms_lt_2px": int(np.sum(rms < 2.0)),
    }


def solve_problem(problem: JointCalibrationProblem, x0: np.ndarray, max_nfev: int, f_scale: float):
    return least_squares(
        problem.residuals,
        x0,
        jac_sparsity=problem.jac_sparsity(),
        method="trf",
        loss="soft_l1",
        f_scale=f_scale,
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=0,
    )


def invert_transform(r_mat, t_vec):
    inv_r = r_mat.T
    inv_t = -inv_r @ np.asarray(t_vec).reshape(3)
    return inv_r, inv_t


def build_candidate(calibration: dict, reference: str, side_cameras: List[str], selected: dict):
    candidate = json.loads(json.dumps(calibration))
    candidate["created_at"] = datetime.now().isoformat(timespec="seconds")
    candidate["source_note"] = (
        "diagnostic three-camera joint calibration candidate; "
        "fixed intrinsics; official config was not overwritten"
    )
    candidate["reference_camera"] = reference
    candidate["stereo_pairs"] = {}
    candidate.setdefault("joint_calibration", {})
    candidate["joint_calibration"].update(
        {
            "filter": selected["filter"],
            "frames": selected["frames"],
            "summary": selected["summary"],
            "camera_summary": selected["camera_summary"],
        }
    )

    for camera_name in [reference] + side_cameras:
        ext = selected["extrinsics"][camera_name]
        r_mat = np.asarray(ext["R"], dtype=np.float64)
        t_vec = np.asarray(ext["t"], dtype=np.float64)
        cam_to_rig_r, cam_to_rig_t = invert_transform(r_mat, t_vec)
        candidate["cameras"][camera_name]["rig_to_camera"] = {
            "R": _json_array(r_mat),
            "t": _json_array(t_vec),
        }
        candidate["cameras"][camera_name]["camera_to_rig"] = {
            "R": _json_array(cam_to_rig_r),
            "t": _json_array(cam_to_rig_t),
        }
        if camera_name == reference:
            candidate["cameras"][camera_name]["extrinsic_source"] = "joint_reference"
            candidate["cameras"][camera_name]["stereo_rms"] = 0.0
        else:
            candidate["cameras"][camera_name]["extrinsic_source"] = (
                f"joint_three_camera:{reference}->{camera_name}:{selected['filter']}"
            )
            candidate["cameras"][camera_name]["stereo_rms"] = selected["summary"]["global_rms_px"]
            pair_name = f"{reference}_{camera_name}"
            candidate["stereo_pairs"][pair_name] = {
                "rms": selected["summary"]["global_rms_px"],
                "frames": selected["frames"],
                "R_a_to_b": _json_array(r_mat),
                "t_a_to_b": _json_array(t_vec),
                "source": "three_camera_joint",
            }
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", type=Path, default=Path("captures"))
    parser.add_argument("--calibration", type=Path, default=Path("config/camera_calibration.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/debug/calibration_check"))
    parser.add_argument("--reference", default="camera2")
    parser.add_argument("--cameras", nargs="+", default=["camera1", "camera2", "camera3"])
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--f-scale", type=float, default=3.0)
    parser.add_argument("--min-frames", type=int, default=6)
    parser.add_argument("--target-max-frame-mean-px", type=float, default=5.0)
    parser.add_argument("--target-mean-px", type=float, default=5.0)
    args = parser.parse_args()

    board, _detector = make_board()
    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    calibration = load_calibration(args.calibration)
    detections, stats = collect_detections(args.captures, args.min_corners)

    cameras = list(args.cameras)
    if args.reference not in cameras:
        raise RuntimeError(f"Reference camera {args.reference} is not in {cameras}")
    side_cameras = [name for name in cameras if name != args.reference]
    frame_names = shared_frame_names(detections, cameras, args.min_corners)
    if len(frame_names) < args.min_frames:
        raise RuntimeError(f"Only {len(frame_names)} shared frames; need at least {args.min_frames}")

    intrinsics = {
        camera_name: camera_intrinsics(calibration, camera_name)
        for camera_name in cameras
    }
    board_poses = {}
    for camera_name in cameras:
        for frame_name in frame_names:
            pose = solve_board_pose(
                board_points,
                detections[(camera_name, frame_name)],
                intrinsics[camera_name],
            )
            if pose is not None:
                board_poses[(camera_name, frame_name)] = pose

    history = []
    current_frames = list(frame_names)
    selected = None
    previous_x = None
    iteration = 0
    while len(current_frames) >= args.min_frames:
        problem = JointCalibrationProblem(
            calibration=calibration,
            detections=detections,
            board_points=board_points,
            reference=args.reference,
            side_cameras=side_cameras,
            frame_names=current_frames,
        )
        if previous_x is None:
            x0 = problem.pack_initial(board_poses)
        else:
            x0 = problem.pack_initial(board_poses)
            # Carry side-camera extrinsics forward when the frame set changes.
            keep_side = len(side_cameras) * 6
            x0[:keep_side] = previous_x[:keep_side]

        result = solve_problem(problem, x0, args.max_nfev, args.f_scale)
        previous_x = result.x.copy()
        residuals = problem.residuals(result.x)
        global_rms = float(np.sqrt(np.mean(residuals * residuals)))
        frame_rows, camera_summary = problem.frame_error_rows(result.x)
        frame_rows_sorted = sorted(frame_rows, key=lambda row: row["mean_px"], reverse=True)
        agg = aggregate_rows(frame_rows)
        extrinsics_np = problem.extrinsics_from_solution(result.x)
        extrinsics = {
            camera_name: {
                "R": _json_array(data["R"]),
                "t": _json_array(data["t"]),
                "rvec": _json_array(data["rvec"]),
                "translation_norm_m": float(np.linalg.norm(data["t"])),
            }
            for camera_name, data in extrinsics_np.items()
        }
        step = {
            "iteration": iteration,
            "frames": list(current_frames),
            "frame_count": len(current_frames),
            "global_rms_px": global_rms,
            "cost": float(result.cost),
            "status": int(result.status),
            "message": result.message,
            "nfev": int(result.nfev),
            "summary": {
                **agg,
                "global_rms_px": global_rms,
            },
            "camera_summary": camera_summary,
            "worst_frames": frame_rows_sorted[:10],
            "extrinsics": extrinsics,
        }
        history.append(step)
        print(
            f"iter={iteration:02d} frames={len(current_frames):02d} "
            f"global_rms={global_rms:.3f}px "
            f"mean_frame_mean={agg.get('mean_of_frame_mean_px', float('nan')):.3f}px "
            f"max_frame_mean={agg.get('max_frame_mean_px', float('nan')):.3f}px "
            f"mean<5={agg.get('frames_mean_lt_5px', 0)}"
        )

        if selected is None and agg.get("max_frame_mean_px", np.inf) < args.target_max_frame_mean_px:
            selected = {
                "filter": f"joint_iter_{iteration}_all_frames_under_{args.target_max_frame_mean_px:g}px",
                "frames": list(current_frames),
                "summary": step["summary"],
                "camera_summary": camera_summary,
                "extrinsics": extrinsics,
            }
            break
        if len(current_frames) <= args.min_frames:
            break
        worst = frame_rows_sorted[0]["file"]
        current_frames = [name for name in current_frames if name != worst]
        iteration += 1

    if selected is None:
        viable = [
            step
            for step in history
            if step["frame_count"] >= args.min_frames
        ]
        # Prefer more frames, but penalize high error enough that heavy filtering can win.
        best = min(
            viable,
            key=lambda step: (
                step["summary"].get("mean_of_frame_mean_px", np.inf)
                + 0.25 * step["summary"].get("max_frame_mean_px", np.inf)
                - 0.05 * step["frame_count"]
            ),
        )
        selected = {
            "filter": f"joint_best_score_iter_{best['iteration']}",
            "frames": best["frames"],
            "summary": best["summary"],
            "camera_summary": best["camera_summary"],
            "extrinsics": best["extrinsics"],
        }

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": (
            "three-camera joint extrinsic calibration with fixed intrinsics; "
            "optimizes side-camera rig extrinsics and one board pose per shared frame"
        ),
        "captures": str(args.captures),
        "calibration": str(args.calibration),
        "reference_camera": args.reference,
        "cameras": cameras,
        "side_cameras": side_cameras,
        "min_corners": args.min_corners,
        "shared_frames_initial": frame_names,
        "detection_summary": stats,
        "history": history,
        "selected_candidate": selected,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "joint_calibration_report.json"
    candidate_path = args.output_dir / "camera_calibration_joint_candidate.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    candidate = build_candidate(calibration, args.reference, side_cameras, selected)
    candidate["joint_calibration"]["report"] = str(report_path)
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {candidate_path}")


if __name__ == "__main__":
    main()
