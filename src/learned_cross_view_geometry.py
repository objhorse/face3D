"""LoFTR-based cross-view geometry diagnostics for fixed RGB rigs."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.cross_view_geometry import (
    Camera,
    load_calibration,
    relative_camera_transform,
    restore_mask_to_work_frame,
    scale_intrinsics,
    triangulate_correspondences,
    write_ply,
)


WORK_SIZE = (640, 480)
MIN_FACE_DEPTH_M = 0.12
MAX_FACE_DEPTH_M = 1.50
PAIR_SPECS = (
    ("camera2_camera1", "camera2", "camera1"),
    ("camera2_camera3", "camera2", "camera3"),
    ("camera1_camera3", "camera1", "camera3"),
)
CALIBRATION_PROBES = (
    ("camera2_camera1", "camera2", "camera1"),
    ("camera2_camera3", "camera2", "camera3"),
)
MAX_REFINED_ROTATION_DELTA_DEG = 10.0
MAX_REFINED_TRANSLATION_DELTA_DEG = 5.0
MIN_REFINED_INLIERS = 100


def percentile_summary(
    values: np.ndarray,
    percentiles: Sequence[float] = (50.0, 75.0, 90.0, 95.0),
) -> Dict[str, Optional[float]]:
    if len(values) == 0:
        return {f"p{int(p):02d}": None for p in percentiles}
    return {
        f"p{int(p):02d}": float(np.percentile(values, p))
        for p in percentiles
    }


def rotation_delta_degrees(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = np.asarray(rotation_a, dtype=np.float64) @ np.asarray(
        rotation_b, dtype=np.float64
    ).T
    cosine = (float(np.trace(delta)) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def translation_direction_delta_degrees(
    translation_a: np.ndarray,
    translation_b: np.ndarray,
) -> float:
    a = np.asarray(translation_a, dtype=np.float64).reshape(-1)
    b = np.asarray(translation_b, dtype=np.float64).reshape(-1)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return float("nan")
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _read_capture(captures_dir: Path, camera_name: str) -> np.ndarray:
    candidates = sorted(captures_dir.glob(f"{camera_name}_*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No capture found for {camera_name}")
    image = cv2.imread(str(candidates[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {candidates[0]}")
    return cv2.resize(image, WORK_SIZE, interpolation=cv2.INTER_AREA)


def _read_mask(root: Path, camera: Camera) -> np.ndarray:
    mask_path = root / "output" / "debug" / f"{camera.view}_face_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Missing preprocessing mask: {mask_path}")
    restored = restore_mask_to_work_frame(mask, target_size=(1024, 768))
    return cv2.resize(restored, WORK_SIZE, interpolation=cv2.INTER_NEAREST)


def _sample_mask(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    xy = np.rint(points).astype(np.int32)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < mask.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < mask.shape[0])
    )
    accepted = np.zeros(len(points), dtype=bool)
    accepted[inside] = mask[xy[inside, 1], xy[inside, 0]] > 127
    return accepted


def load_loftr_matcher():
    import torch
    from kornia.feature import LoFTR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained="outdoor").to(device).eval()
    return matcher, device


def _image_tensor(image: np.ndarray, device: object):
    import torch

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray).float()[None, None] / 255.0
    return tensor.to(device)


def run_loftr_matches(
    matcher: object,
    device: object,
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    if image_a.shape[:2] != image_b.shape[:2]:
        raise ValueError("LoFTR input images must have the same height and width")
    with torch.no_grad():
        output = matcher(
            {
                "image0": _image_tensor(image_a, device),
                "image1": _image_tensor(image_b, device),
            }
        )
    points_a = np.asarray(
        output["keypoints0"].detach().cpu().numpy(), dtype=np.float64
    ).reshape(-1, 2)
    points_b = np.asarray(
        output["keypoints1"].detach().cpu().numpy(), dtype=np.float64
    ).reshape(-1, 2)
    confidence = np.asarray(
        output["confidence"].detach().cpu().numpy(), dtype=np.float64
    ).reshape(-1)
    if not (len(points_a) == len(points_b) == len(confidence)):
        raise ValueError("LoFTR returned inconsistent match array lengths")
    if not (
        np.isfinite(points_a).all()
        and np.isfinite(points_b).all()
        and np.isfinite(confidence).all()
    ):
        raise ValueError("LoFTR returned non-finite matches")
    confidence = np.clip(confidence, 0.0, 1.0)
    return points_a, points_b, confidence


def _load_loftr():
    """Compatibility wrapper for the existing diagnostic pipeline."""
    return load_loftr_matcher()


def _run_loftr(
    matcher: object,
    device: object,
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper for the existing diagnostic pipeline."""
    return run_loftr_matches(matcher, device, image_a, image_b)


def rectified_vertical_error(
    points_a: np.ndarray,
    points_b: np.ndarray,
    camera_a: Camera,
    camera_b: Camera,
    work_size: Tuple[int, int] = WORK_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_a = scale_intrinsics(camera_a.K, camera_a.image_size, work_size)
    K_b = scale_intrinsics(camera_b.K, camera_b.image_size, work_size)
    R_ba, t_ba = relative_camera_transform(camera_a, camera_b)
    R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
        K_a,
        camera_a.dist,
        K_b,
        camera_b.dist,
        work_size,
        R_ba,
        t_ba,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1.0,
    )
    rect_a = cv2.undistortPoints(
        points_a.reshape(-1, 1, 2), K_a, camera_a.dist, R=R1, P=P1
    ).reshape(-1, 2)
    rect_b = cv2.undistortPoints(
        points_b.reshape(-1, 1, 2), K_b, camera_b.dist, R=R2, P=P2
    ).reshape(-1, 2)
    return np.abs(rect_a[:, 1] - rect_b[:, 1]), rect_a, rect_b


def rectified_vertical_error_for_pose(
    points_a: np.ndarray,
    points_b: np.ndarray,
    camera_a: Camera,
    camera_b: Camera,
    rotation_ba: np.ndarray,
    translation_ba: np.ndarray,
    work_size: Tuple[int, int] = WORK_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    K_a = scale_intrinsics(camera_a.K, camera_a.image_size, work_size)
    K_b = scale_intrinsics(camera_b.K, camera_b.image_size, work_size)
    R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
        K_a,
        camera_a.dist,
        K_b,
        camera_b.dist,
        work_size,
        rotation_ba,
        translation_ba,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1.0,
    )
    rect_a = cv2.undistortPoints(
        points_a.reshape(-1, 1, 2), K_a, camera_a.dist, R=R1, P=P1
    ).reshape(-1, 2)
    rect_b = cv2.undistortPoints(
        points_b.reshape(-1, 1, 2), K_b, camera_b.dist, R=R2, P=P2
    ).reshape(-1, 2)
    signed = rect_a[:, 1] - rect_b[:, 1]
    return signed, np.abs(signed)


def _estimate_pose(
    points_a: np.ndarray,
    points_b: np.ndarray,
    camera_a: Camera,
    camera_b: Camera,
) -> Dict[str, Optional[float]]:
    if len(points_a) < 20:
        return {
            "essential_inliers": 0,
            "rotation_delta_deg": None,
            "translation_direction_delta_deg": None,
        }
    K_a = scale_intrinsics(camera_a.K, camera_a.image_size, WORK_SIZE)
    K_b = scale_intrinsics(camera_b.K, camera_b.image_size, WORK_SIZE)
    norm_a = cv2.undistortPoints(
        points_a.reshape(-1, 1, 2), K_a, camera_a.dist
    ).reshape(-1, 2)
    norm_b = cv2.undistortPoints(
        points_b.reshape(-1, 1, 2), K_b, camera_b.dist
    ).reshape(-1, 2)
    essential, inliers = cv2.findEssentialMat(
        norm_a,
        norm_b,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.004,
    )
    if essential is None or inliers is None or int(inliers.sum()) < 8:
        return {
            "essential_inliers": int(inliers.sum()) if inliers is not None else 0,
            "rotation_delta_deg": None,
            "translation_direction_delta_deg": None,
        }
    _count, rotation, translation, _pose_mask = cv2.recoverPose(
        essential, norm_a, norm_b, np.eye(3), mask=inliers
    )
    rig_rotation, rig_translation = relative_camera_transform(camera_a, camera_b)
    translation_delta = min(
        translation_direction_delta_degrees(translation.reshape(3), rig_translation),
        translation_direction_delta_degrees(-translation.reshape(3), rig_translation),
    )
    return {
        "essential_inliers": int(inliers.sum()),
        "rotation_delta_deg": rotation_delta_degrees(rotation, rig_rotation),
        "translation_direction_delta_deg": translation_delta,
    }


def _estimate_refined_pose_candidate(
    points_a: np.ndarray,
    points_b: np.ndarray,
    camera_a: Camera,
    camera_b: Camera,
) -> Tuple[Dict[str, object], Optional[np.ndarray], Optional[np.ndarray]]:
    if len(points_a) < 20:
        return (
            {
                "accepted": False,
                "reason": "not enough matches",
                "essential_inliers": 0,
            },
            None,
            None,
        )
    K_a = scale_intrinsics(camera_a.K, camera_a.image_size, WORK_SIZE)
    K_b = scale_intrinsics(camera_b.K, camera_b.image_size, WORK_SIZE)
    norm_a = cv2.undistortPoints(
        points_a.reshape(-1, 1, 2), K_a, camera_a.dist
    ).reshape(-1, 2)
    norm_b = cv2.undistortPoints(
        points_b.reshape(-1, 1, 2), K_b, camera_b.dist
    ).reshape(-1, 2)
    essential, inliers = cv2.findEssentialMat(
        norm_a,
        norm_b,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.004,
    )
    if essential is None or inliers is None:
        return (
            {
                "accepted": False,
                "reason": "essential matrix failed",
                "essential_inliers": 0,
            },
            None,
            None,
        )
    inlier_count = int(inliers.sum())
    if inlier_count < 8:
        return (
            {
                "accepted": False,
                "reason": "too few essential inliers",
                "essential_inliers": inlier_count,
            },
            None,
            None,
        )
    _count, rotation, translation, _pose_mask = cv2.recoverPose(
        essential, norm_a, norm_b, np.eye(3), mask=inliers
    )
    rig_rotation, rig_translation = relative_camera_transform(camera_a, camera_b)
    translation = translation.reshape(3)
    if float(np.dot(translation, rig_translation)) < 0.0:
        translation = -translation
    scaled_translation = (
        translation
        / max(float(np.linalg.norm(translation)), 1e-9)
        * float(np.linalg.norm(rig_translation))
    )
    rotation_delta = rotation_delta_degrees(rotation, rig_rotation)
    translation_delta = translation_direction_delta_degrees(
        scaled_translation, rig_translation
    )
    signed_error, abs_error = rectified_vertical_error_for_pose(
        points_a, points_b, camera_a, camera_b, rotation, scaled_translation
    )
    accepted = (
        inlier_count >= MIN_REFINED_INLIERS
        and rotation_delta <= MAX_REFINED_ROTATION_DELTA_DEG
        and translation_delta <= MAX_REFINED_TRANSLATION_DELTA_DEG
        and float(np.percentile(abs_error, 75)) <= 3.5
    )
    reason = (
        "accepted bounded LoFTR pose candidate"
        if accepted
        else "candidate outside bounded quality gates"
    )
    return (
        {
            "accepted": bool(accepted),
            "reason": reason,
            "essential_inliers": inlier_count,
            "rotation_delta_deg": float(rotation_delta),
            "translation_direction_delta_deg": float(translation_delta),
            "rig_y_error_px": percentile_summary(abs_error, (50.0, 75.0, 90.0, 95.0)),
            "signed_y_error_px": percentile_summary(
                signed_error, (5.0, 25.0, 50.0, 75.0, 95.0)
            ),
            "matches_within_3px": int((abs_error <= 3.0).sum()),
        },
        rotation,
        scaled_translation,
    )


def _triangulate_good_matches(
    points_a: np.ndarray,
    points_b: np.ndarray,
    image_a: np.ndarray,
    camera_a: Camera,
    camera_b: Camera,
    good: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(good.sum()) == 0:
        empty_xyz = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        return empty_xyz, empty_colors, np.zeros(len(points_a), dtype=bool)
    K_a = scale_intrinsics(camera_a.K, camera_a.image_size, WORK_SIZE)
    K_b = scale_intrinsics(camera_b.K, camera_b.image_size, WORK_SIZE)
    undist_a = cv2.undistortPoints(
        points_a[good].reshape(-1, 1, 2), K_a, camera_a.dist, P=K_a
    ).reshape(-1, 2)
    undist_b = cv2.undistortPoints(
        points_b[good].reshape(-1, 1, 2), K_b, camera_b.dist, P=K_b
    ).reshape(-1, 2)
    rotation, translation = relative_camera_transform(camera_a, camera_b)
    xyz, positive = triangulate_correspondences(
        undist_a, undist_b, K_a, K_b, rotation, translation
    )
    plausible = (
        positive
        & (xyz[:, 2] >= MIN_FACE_DEPTH_M)
        & (xyz[:, 2] <= MAX_FACE_DEPTH_M)
        & np.isfinite(xyz).all(axis=1)
    )
    xy = np.rint(points_a[good]).astype(np.int32)
    xy[:, 0] = np.clip(xy[:, 0], 0, image_a.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, image_a.shape[0] - 1)
    colors = image_a[xy[:, 1], xy[:, 0]]
    final_good = np.zeros(len(points_a), dtype=bool)
    good_indices = np.flatnonzero(good)
    final_good[good_indices[plausible]] = True
    return xyz[plausible], colors[plausible], final_good


def _draw_match_overlay(
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    accepted: np.ndarray,
    good: np.ndarray,
    max_matches: int = 260,
) -> np.ndarray:
    height = max(image_a.shape[0], image_b.shape[0])
    width_a = image_a.shape[1]
    canvas = np.zeros((height, width_a + image_b.shape[1], 3), dtype=np.uint8)
    canvas[: image_a.shape[0], :width_a] = image_a
    canvas[: image_b.shape[0], width_a:] = image_b
    indices = np.flatnonzero(accepted)
    if len(indices) > max_matches:
        order = np.linspace(0, len(indices) - 1, max_matches).astype(np.int64)
        indices = indices[order]
    for index in indices:
        color = (70, 220, 120) if good[index] else (60, 170, 255)
        point_a = tuple(np.rint(points_a[index]).astype(int))
        raw_b = np.rint(points_b[index]).astype(int)
        point_b = (int(raw_b[0] + width_a), int(raw_b[1]))
        cv2.line(canvas, point_a, point_b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, point_a, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point_b, 2, color, -1, cv2.LINE_AA)
    return canvas


def _probe_calibration_options(
    points_a: np.ndarray,
    points_b: np.ndarray,
    cameras: Mapping[str, Camera],
) -> List[Dict[str, object]]:
    probes: List[Dict[str, object]] = []
    for label, camera_a_name, camera_b_name in CALIBRATION_PROBES:
        camera_a = cameras[camera_a_name]
        camera_b = cameras[camera_b_name]
        y_error, _rect_a, _rect_b = rectified_vertical_error(
            points_a, points_b, camera_a, camera_b
        )
        summary = percentile_summary(y_error, (50.0, 90.0))
        probes.append(
            {
                "calibration_pair": label,
                "median_rectified_vertical_error_px": summary["p50"],
                "p90_rectified_vertical_error_px": summary["p90"],
                "matches_within_3px": int((y_error <= 3.0).sum()),
            }
        )
    return probes


def _quality_label(metrics: Mapping[str, object]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    if int(metrics["face_confident_matches"]) < 100:
        failures.append("face_confident_matches")
    yerr = metrics["rig_y_error_px"]
    if yerr["p50"] is None or float(yerr["p50"]) > 3.0:
        failures.append("rig_y_error_p50")
    if yerr["p90"] is None or float(yerr["p90"]) > 8.0:
        failures.append("rig_y_error_p90")
    if int(metrics["rig_matches_within_3px"]) < 100:
        failures.append("rig_matches_within_3px")
    pose_delta = metrics["pose_estimate"]["rotation_delta_deg"]
    if pose_delta is None or float(pose_delta) > 10.0:
        failures.append("pose_rotation_delta")
    return not failures, failures


def _json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _invert_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    inverse_rotation = np.asarray(rotation, dtype=np.float64).T
    inverse_translation = -inverse_rotation @ np.asarray(
        translation, dtype=np.float64
    ).reshape(3)
    return inverse_rotation, inverse_translation


def write_refined_calibration_candidate(
    calibration_path: Path,
    output_path: Path,
    cameras: Mapping[str, Camera],
    pair_results: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    source = json.loads(calibration_path.read_text(encoding="utf-8"))
    updates: List[Dict[str, object]] = []
    for result in pair_results:
        candidate = result.get("refined_pose_candidate", {})
        if not isinstance(candidate, Mapping) or not candidate.get("accepted"):
            continue
        rotation_ba = result.get("_refined_rotation_ba")
        translation_ba = result.get("_refined_translation_ba")
        if rotation_ba is None or translation_ba is None:
            continue
        camera_a_name = str(result["camera_a"])
        camera_b_name = str(result["camera_b"])
        camera_a = cameras[camera_a_name]
        rotation_ba = np.asarray(rotation_ba, dtype=np.float64)
        translation_ba = np.asarray(translation_ba, dtype=np.float64).reshape(3)
        new_rotation = rotation_ba @ camera_a.R_rig_to_camera
        new_translation = (
            rotation_ba @ camera_a.t_rig_to_camera.reshape(3)
            + translation_ba
        )
        inverse_rotation, inverse_translation = _invert_transform(
            new_rotation, new_translation
        )
        camera_data = source["cameras"][camera_b_name]
        camera_data["rig_to_camera"] = {
            "R": new_rotation.tolist(),
            "t": new_translation.tolist(),
        }
        camera_data["camera_to_rig"] = {
            "R": inverse_rotation.tolist(),
            "t": inverse_translation.tolist(),
        }
        previous_source = str(camera_data.get("extrinsic_source", ""))
        camera_data["extrinsic_source"] = (
            f"loftr_refined:{camera_a_name}->{camera_b_name};"
            f"{previous_source}"
        ).rstrip(";")
        camera_data["loftr_refinement"] = {
            "source_pair": str(result["pair"]),
            "essential_inliers": int(candidate["essential_inliers"]),
            "rotation_delta_deg": float(candidate["rotation_delta_deg"]),
            "translation_direction_delta_deg": float(
                candidate["translation_direction_delta_deg"]
            ),
            "rig_y_error_px": candidate["rig_y_error_px"],
            "matches_within_3px": int(candidate["matches_within_3px"]),
        }
        pair_name = f"{camera_a_name}_{camera_b_name}"
        if pair_name in source.get("stereo_pairs", {}):
            source["stereo_pairs"][pair_name]["R_a_to_b"] = rotation_ba.tolist()
            source["stereo_pairs"][pair_name]["t_a_to_b"] = (
                translation_ba.tolist()
            )
            previous_pair_source = str(
                source["stereo_pairs"][pair_name].get("source", "")
            )
            source["stereo_pairs"][pair_name]["source"] = (
                f"loftr_refined;{previous_pair_source}"
            ).rstrip(";")
        updates.append(
            {
                "pair": pair_name,
                "updated_camera": camera_b_name,
                "rotation_delta_deg": float(candidate["rotation_delta_deg"]),
                "translation_direction_delta_deg": float(
                    candidate["translation_direction_delta_deg"]
                ),
                "candidate_median_y_error_px": candidate["rig_y_error_px"]["p50"],
                "candidate_matches_within_3px": int(
                    candidate["matches_within_3px"]
                ),
            }
        )
    source["loftr_refined_candidate"] = {
        "created_by": "tools/validate_learned_cross_view_geometry.py",
        "note": (
            "Diagnostic candidate only. Review before replacing the official "
            "camera_calibration.json."
        ),
        "updates": updates,
    }
    output_path.write_text(
        json.dumps(_json_ready(source), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return updates


def _metric_card(label: str, value: object) -> str:
    return f"<div><b>{html.escape(str(value))}</b><span>{html.escape(label)}</span></div>"


def write_report(
    output_dir: Path,
    pair_results: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    cards: List[str] = []
    for result in pair_results:
        pair = str(result["pair"])
        status_class = "pass" if result["passed"] else "fail"
        status_text = "pass" if result["passed"] else "not ready"
        yerr = result["rig_y_error_px"]
        pose = result["pose_estimate"]
        refined = result.get("refined_pose_candidate", {})
        failures = ", ".join(str(v) for v in result["failed_gates"]) or "none"
        probes = "".join(
            f"<li>{html.escape(str(p['calibration_pair']))}: "
            f"median {float(p['median_rectified_vertical_error_px']):.2f}px, "
            f"p90 {float(p['p90_rectified_vertical_error_px']):.2f}px, "
            f"<=3px {int(p['matches_within_3px'])}</li>"
            for p in result["calibration_probe"]
        )
        cards.append(
            f"""
            <section class="pair">
              <div class="pair-head">
                <h2>{html.escape(pair)}</h2>
                <span class="status {status_class}">{status_text}</span>
              </div>
              <div class="metrics">
                {_metric_card("raw LoFTR matches", result["raw_matches"])}
                {_metric_card("face + confidence matches", result["face_confident_matches"])}
                {_metric_card("rig <= 3px matches", result["rig_matches_within_3px"])}
                {_metric_card("rig median y error", f"{float(yerr['p50']):.2f}px" if yerr["p50"] is not None else "n/a")}
                {_metric_card("rig p90 y error", f"{float(yerr['p90']):.2f}px" if yerr["p90"] is not None else "n/a")}
                {_metric_card("triangulated points", result["triangulated_points"])}
                {_metric_card("essential inliers", pose["essential_inliers"])}
                {_metric_card("pose rotation delta", f"{float(pose['rotation_delta_deg']):.1f} deg" if pose["rotation_delta_deg"] is not None else "n/a")}
                {_metric_card("median y offset", f"{float(result['median_signed_y_offset_px']):.2f}px" if result["median_signed_y_offset_px"] is not None else "n/a")}
                {_metric_card("offset-corrected <=3px", result["offset_corrected_matches_within_3px"])}
                {_metric_card("refined <=3px", refined.get("matches_within_3px", "n/a"))}
                {_metric_card("refined median y", f"{float(refined['rig_y_error_px']['p50']):.2f}px" if refined.get("rig_y_error_px", {}).get("p50") is not None else "n/a")}
              </div>
              <p>Refined candidate: {html.escape(str(refined.get("reason", "n/a")))}</p>
              <p>Failed gates: {html.escape(failures)}</p>
              <p>Calibration probe with the same LoFTR points:</p>
              <ul>{probes}</ul>
              <figure>
                <img src="{pair}_loftr_matches.jpg" alt="{html.escape(pair)} matches">
                <figcaption>Green lines pass current rig epipolar check; orange lines are face matches that do not.</figcaption>
              </figure>
            </section>
            """
        )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Learned Cross-View Geometry Diagnostic</title>
<style>
:root{{--bg:#0d1117;--panel:#151b23;--line:#303a46;--text:#f1f5f9;--muted:#9aa7b4;--green:#55c982;--red:#f08a8a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:"Segoe UI",Arial,sans-serif;letter-spacing:0}}
main{{width:min(1240px,calc(100% - 28px));margin:auto;padding:26px 0 44px}} h1,h2,p{{margin-top:0}} h1{{font-size:28px;margin-bottom:8px}}
.lead{{color:var(--muted);line-height:1.6}} .summary,.pair{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:20px;margin-top:18px}}
.pair-head{{display:flex;justify-content:space-between;align-items:center;gap:16px}} .status{{border-radius:4px;padding:5px 10px;font-weight:700}}
.status.pass{{background:var(--green);color:#06120b}} .status.fail{{background:var(--red);color:#190707}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}} .metrics div{{background:#0f151d;border:1px solid var(--line);padding:12px;min-width:0}}
.metrics b{{display:block;font-size:18px;overflow-wrap:anywhere}} .metrics span{{display:block;color:var(--muted);font-size:12px;margin-top:4px}}
figure{{margin:14px 0 0;background:#080c12;border:1px solid var(--line)}} img{{display:block;width:100%;height:auto}} figcaption{{padding:9px 11px;color:var(--muted);font-size:13px}}
li{{margin:4px 0;color:var(--muted)}} code{{color:#8ad0df}} @media(max-width:850px){{.metrics{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body><main>
<h1>Learned Cross-View Geometry Diagnostic</h1>
<p class="lead">This report uses LoFTR only as an independent geometry diagnostic. It does not update calibration, FLAME, texture, or GLB outputs.</p>
<section class="summary">
  <div class="pair-head"><h2>Summary</h2><span class="status {'pass' if summary['passed'] else 'fail'}">{'pass' if summary['passed'] else 'not ready'}</span></div>
  <p>{html.escape(str(summary['conclusion']))}</p>
  <p>Raw metrics are saved in <code>quality.json</code>; sparse point clouds are saved beside this report.</p>
</section>
{''.join(cards)}
</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def process_pair(
    pair_name: str,
    camera_a_name: str,
    camera_b_name: str,
    cameras: Mapping[str, Camera],
    images: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    matcher: object,
    device: object,
    output_dir: Path,
) -> Dict[str, object]:
    camera_a = cameras[camera_a_name]
    camera_b = cameras[camera_b_name]
    points_a, points_b, confidence = _run_loftr(
        matcher, device, images[camera_a_name], images[camera_b_name]
    )
    if len(confidence):
        confidence_floor = max(0.10, float(np.percentile(confidence, 25.0)))
    else:
        confidence_floor = 1.0
    accepted = (
        _sample_mask(masks[camera_a_name], points_a)
        & _sample_mask(masks[camera_b_name], points_b)
        & (confidence >= confidence_floor)
    )
    accepted_points_a = points_a[accepted]
    accepted_points_b = points_b[accepted]
    accepted_confidence = confidence[accepted]
    y_error, _rect_a, _rect_b = rectified_vertical_error(
        accepted_points_a, accepted_points_b, camera_a, camera_b
    )
    signed_y_error, _abs_y_error = rectified_vertical_error_for_pose(
        accepted_points_a,
        accepted_points_b,
        camera_a,
        camera_b,
        *relative_camera_transform(camera_a, camera_b),
    )
    median_signed_offset = (
        float(np.median(signed_y_error)) if len(signed_y_error) else float("nan")
    )
    offset_corrected = (
        np.abs(signed_y_error - median_signed_offset)
        if len(signed_y_error)
        else np.empty((0,), dtype=np.float64)
    )
    good_rig = np.zeros(len(points_a), dtype=bool)
    accepted_indices = np.flatnonzero(accepted)
    good_in_accepted = y_error <= 3.0
    good_rig[accepted_indices[good_in_accepted]] = True
    xyz, colors, triangulated_mask = _triangulate_good_matches(
        points_a,
        points_b,
        images[camera_a_name],
        camera_a,
        camera_b,
        good_rig,
    )
    refined_pose, refined_rotation, refined_translation = (
        _estimate_refined_pose_candidate(
            accepted_points_a, accepted_points_b, camera_a, camera_b
        )
    )
    write_ply(output_dir / f"{pair_name}_loftr_points.ply", xyz, colors)
    cv2.imwrite(
        str(output_dir / f"{pair_name}_loftr_matches.jpg"),
        _draw_match_overlay(
            images[camera_a_name],
            images[camera_b_name],
            points_a,
            points_b,
            accepted,
            good_rig,
        ),
    )

    metrics: Dict[str, object] = {
        "pair": pair_name,
        "camera_a": camera_a_name,
        "camera_b": camera_b_name,
        "view_a": camera_a.view,
        "view_b": camera_b.view,
        "raw_matches": int(len(points_a)),
        "face_confident_matches": int(accepted.sum()),
        "confidence_floor": float(confidence_floor),
        "raw_confidence": percentile_summary(confidence, (50.0, 75.0, 90.0, 95.0, 99.0)),
        "accepted_confidence": percentile_summary(accepted_confidence, (50.0, 75.0, 90.0, 95.0)),
        "rig_y_error_px": percentile_summary(y_error, (50.0, 75.0, 90.0, 95.0)),
        "signed_y_error_px": percentile_summary(
            signed_y_error, (5.0, 25.0, 50.0, 75.0, 95.0)
        ),
        "median_signed_y_offset_px": median_signed_offset,
        "offset_corrected_y_error_px": percentile_summary(
            offset_corrected, (50.0, 75.0, 90.0, 95.0)
        ),
        "offset_corrected_matches_within_3px": int((offset_corrected <= 3.0).sum()),
        "rig_matches_within_3px": int(good_in_accepted.sum()),
        "triangulated_points": int(len(xyz)),
        "triangulated_from_current_rig": int(triangulated_mask.sum()),
        "pose_estimate": _estimate_pose(
            accepted_points_a, accepted_points_b, camera_a, camera_b
        ),
        "refined_pose_candidate": refined_pose,
        "calibration_probe": _probe_calibration_options(
            accepted_points_a, accepted_points_b, cameras
        )
        if camera_a_name == "camera2"
        else [],
    }
    if refined_rotation is not None and refined_translation is not None:
        metrics["_refined_rotation_ba"] = refined_rotation
        metrics["_refined_translation_ba"] = refined_translation
    passed, failed_gates = _quality_label(metrics)
    metrics["passed"] = passed
    metrics["failed_gates"] = failed_gates
    return metrics


def _alias_warning(pair_results: Sequence[Mapping[str, object]]) -> Optional[str]:
    by_pair = {str(result["pair"]): result for result in pair_results}
    first = by_pair.get("camera2_camera1")
    if not first:
        return None
    probes = {
        str(item["calibration_pair"]): item
        for item in first.get("calibration_probe", [])
    }
    current = probes.get("camera2_camera1")
    alternate = probes.get("camera2_camera3")
    if not current or not alternate:
        return None
    current_median = current["median_rectified_vertical_error_px"]
    alternate_median = alternate["median_rectified_vertical_error_px"]
    if (
        current_median is not None
        and alternate_median is not None
        and float(alternate_median) + 2.0 < float(current_median)
    ):
        return (
            "The camera2-camera1 face matches fit the camera2-camera3 calibration "
            "better than their named calibration. Review capture-to-calibration "
            "mapping and the rig extrinsics before using learned matches for shape."
        )
    return None


def run_validation(
    root: Path,
    captures_dir: Path,
    calibration_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = load_calibration(calibration_path)
    images = {name: _read_capture(captures_dir, name) for name in cameras}
    masks = {name: _read_mask(root, camera) for name, camera in cameras.items()}
    matcher, device = _load_loftr()
    pair_results = [
        process_pair(
            pair_name,
            camera_a_name,
            camera_b_name,
            cameras,
            images,
            masks,
            matcher,
            device,
            output_dir,
        )
        for pair_name, camera_a_name, camera_b_name in PAIR_SPECS
    ]
    candidate_path = output_dir / "camera_calibration_loftr_candidate.json"
    candidate_updates = write_refined_calibration_candidate(
        calibration_path, candidate_path, cameras, pair_results
    )
    public_pair_results: List[Dict[str, object]] = [
        {key: value for key, value in result.items() if not key.startswith("_")}
        for result in pair_results
    ]
    warning = _alias_warning(pair_results)
    passed = all(
        result["passed"]
        for result in public_pair_results
        if result["pair"] in {"camera2_camera1", "camera2_camera3"}
    )
    if passed and warning is None:
        conclusion = (
            "LoFTR produced enough cross-view matches and the current rig explains "
            "them within the strict epipolar gates. These points can be considered "
            "for a bounded FLAME fitting experiment."
        )
    else:
        details = [
            "LoFTR produced enough matches, but the current rig does not yet pass "
            "strict epipolar gates for direct shape optimization."
        ]
        if candidate_updates:
            details.append(
                "A bounded LoFTR refined calibration candidate was exported for "
                "review because it improves at least one failing pair."
            )
        if warning:
            details.append(warning)
        details.append(
            "Keep this diagnostic as evidence; do not pull FLAME vertices with these "
            "matches until the mapping/calibration issue is resolved."
        )
        conclusion = " ".join(details)
    summary: Dict[str, object] = {
        "passed": bool(passed and warning is None),
        "conclusion": conclusion,
        "input": str(captures_dir),
        "calibration": str(calibration_path),
        "work_size": list(WORK_SIZE),
        "device": str(device),
        "alias_warning": warning,
        "candidate_calibration": str(candidate_path),
        "candidate_updates": candidate_updates,
        "pairs": public_pair_results,
    }
    json_summary = _json_ready(summary)
    (output_dir / "quality.json").write_text(
        json.dumps(json_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(output_dir, json_summary["pairs"], json_summary)
    return json_summary
