"""Calibrated cross-view geometry diagnostics for synchronized RGB captures."""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


WORK_SIZE = (1024, 768)
MIN_FACE_DEPTH_M = 0.12
MAX_FACE_DEPTH_M = 1.50


@dataclass(frozen=True)
class Camera:
    name: str
    view: str
    image_size: Tuple[int, int]
    K: np.ndarray
    dist: np.ndarray
    R_rig_to_camera: np.ndarray
    t_rig_to_camera: np.ndarray


def load_calibration(path: Path) -> Dict[str, Camera]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cameras: Dict[str, Camera] = {}
    for name, item in data["cameras"].items():
        cameras[name] = Camera(
            name=name,
            view=str(item["view"]),
            image_size=tuple(int(v) for v in item["image_size"]),
            K=np.asarray(item["K"], dtype=np.float64),
            dist=np.asarray(item["dist_coeffs"], dtype=np.float64).reshape(-1),
            R_rig_to_camera=np.asarray(
                item["rig_to_camera"]["R"], dtype=np.float64
            ),
            t_rig_to_camera=np.asarray(
                item["rig_to_camera"]["t"], dtype=np.float64
            ).reshape(3),
        )
    return cameras


def scale_intrinsics(
    K: np.ndarray,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> np.ndarray:
    source_w, source_h = source_size
    target_w, target_h = target_size
    scaled = np.asarray(K, dtype=np.float64).copy()
    scaled[0, :] *= target_w / float(source_w)
    scaled[1, :] *= target_h / float(source_h)
    scaled[2, :] = (0.0, 0.0, 1.0)
    return scaled


def relative_camera_transform(
    camera_a: Camera,
    camera_b: Camera,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return X_b = R_ba X_a + t_ba."""
    R_ba = camera_b.R_rig_to_camera @ camera_a.R_rig_to_camera.T
    t_ba = camera_b.t_rig_to_camera - R_ba @ camera_a.t_rig_to_camera
    return R_ba, t_ba


def triangulate_correspondences(
    points_a: np.ndarray,
    points_b: np.ndarray,
    K_a: np.ndarray,
    K_b: np.ndarray,
    R_ba: np.ndarray,
    t_ba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    projection_a = K_a @ np.column_stack(
        [np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
    )
    projection_b = K_b @ np.column_stack([R_ba, t_ba.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(
        projection_a,
        projection_b,
        np.asarray(points_a, dtype=np.float64).T,
        np.asarray(points_b, dtype=np.float64).T,
    )
    xyz = (homogeneous[:3] / homogeneous[3:4]).T
    depth_a = xyz[:, 2]
    depth_b = (xyz @ R_ba.T + t_ba)[:, 2]
    positive = np.isfinite(xyz).all(axis=1) & (depth_a > 0) & (depth_b > 0)
    return xyz, positive


def restore_mask_to_work_frame(
    mask: np.ndarray,
    target_size: Tuple[int, int] = WORK_SIZE,
) -> np.ndarray:
    """Undo the project's square letterbox for a 4:3 source image."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    height, width = mask.shape
    if height == width:
        source_ratio = target_size[0] / float(target_size[1])
        crop_h = int(round(width / source_ratio))
        y0 = max(0, (height - crop_h) // 2)
        mask = mask[y0 : y0 + crop_h]
    restored = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    restored = np.where(restored > 127, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.erode(restored, kernel, iterations=1)


def _normalized_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _ratio_matches(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    ratio: float = 0.76,
) -> Dict[int, int]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    result: Dict[int, int] = {}
    for candidates in matcher.knnMatch(descriptors_a, descriptors_b, k=2):
        if len(candidates) != 2:
            continue
        best, second = candidates
        if best.distance < ratio * second.distance:
            result[int(best.queryIdx)] = int(best.trainIdx)
    return result


def reciprocal_sift_matches(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.015)
    keypoints_a, descriptors_a = sift.detectAndCompute(
        _normalized_gray(image_a), mask_a
    )
    keypoints_b, descriptors_b = sift.detectAndCompute(
        _normalized_gray(image_b), mask_b
    )
    if descriptors_a is None or descriptors_b is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, np.empty((0,), dtype=np.float32)

    forward = _ratio_matches(descriptors_a, descriptors_b)
    backward = _ratio_matches(descriptors_b, descriptors_a)
    pairs = [
        (idx_a, idx_b)
        for idx_a, idx_b in forward.items()
        if backward.get(idx_b) == idx_a
    ]
    if not pairs:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, np.empty((0,), dtype=np.float32)

    points_a = np.asarray(
        [keypoints_a[idx_a].pt for idx_a, _ in pairs], dtype=np.float32
    )
    points_b = np.asarray(
        [keypoints_b[idx_b].pt for _, idx_b in pairs], dtype=np.float32
    )
    vertical_error = np.abs(points_a[:, 1] - points_b[:, 1])
    return points_a, points_b, vertical_error


def estimate_disparity_search(
    points_a: np.ndarray,
    points_b: np.ndarray,
    vertical_error: np.ndarray,
    mask_a: Optional[np.ndarray] = None,
    mask_b: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    if len(points_a):
        reliable = vertical_error <= 6.0
        disparities = points_a[reliable, 0] - points_b[reliable, 0]
    else:
        disparities = np.empty((0,), dtype=np.float32)

    if len(disparities) >= 20:
        low, high = np.percentile(disparities, [1, 99])
        low -= 32.0
        high += 32.0
    elif mask_a is not None and mask_b is not None:
        _ya, xa = np.where(mask_a > 0)
        _yb, xb = np.where(mask_b > 0)
        if len(xa) and len(xb):
            center = float(np.median(xa) - np.median(xb))
            low, high = center - 320.0, center + 320.0
        else:
            low, high = -320.0, 320.0
    else:
        low, high = -320.0, 320.0

    min_disparity = int(math.floor(low / 16.0) * 16)
    max_disparity = int(math.ceil(high / 16.0) * 16)
    if max_disparity - min_disparity < 64:
        max_disparity = min_disparity + 64
    if max_disparity - min_disparity > 640:
        center = 0.5 * (max_disparity + min_disparity)
        min_disparity = int(math.floor((center - 320.0) / 16.0) * 16)
        max_disparity = min_disparity + 640
    return min_disparity, max_disparity


def _compute_disparity(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    min_disparity: int,
    max_disparity: int,
) -> np.ndarray:
    num_disparities = int(math.ceil(
        (max_disparity - min_disparity) / 16.0
    ) * 16)
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=5,
        P1=8 * 5 * 5,
        P2=32 * 5 * 5,
        disp12MaxDiff=2,
        preFilterCap=31,
        uniquenessRatio=8,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(gray_a, gray_b).astype(np.float32) / 16.0


def _shift_horizontally(
    image: np.ndarray,
    shift_px: int,
    interpolation: int,
) -> np.ndarray:
    transform = np.array([[1.0, 0.0, float(shift_px)], [0.0, 1.0, 0.0]])
    return cv2.warpAffine(
        image,
        transform,
        (image.shape[1], image.shape[0]),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _shift_array_exact(
    array: np.ndarray,
    shift_px: int,
    fill_value: object = 0,
) -> np.ndarray:
    output = np.full_like(array, fill_value)
    if shift_px == 0:
        output[...] = array
    elif 0 < shift_px < array.shape[1]:
        output[:, shift_px:] = array[:, : -shift_px]
    elif -array.shape[1] < shift_px < 0:
        output[:, :shift_px] = array[:, -shift_px:]
    return output


def _sample_map(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xi = np.rint(x).astype(np.int32)
    yi = np.rint(y).astype(np.int32)
    sampled = np.full(x.shape, np.nan, dtype=np.float32)
    valid = (
        (xi >= 0)
        & (xi < values.shape[1])
        & (yi >= 0)
        & (yi < values.shape[0])
    )
    sampled[valid] = values[yi[valid], xi[valid]]
    return sampled


def _grid_coverage(mask: np.ndarray, accepted: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return 0.0
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    covered = 0
    trusted = 0
    for row in range(8):
        for col in range(8):
            xa = x0 + (x1 - x0) * col // 8
            xb = x0 + (x1 - x0) * (col + 1) // 8
            ya = y0 + (y1 - y0) * row // 8
            yb = y0 + (y1 - y0) * (row + 1) // 8
            cell_mask = mask[ya:yb, xa:xb] > 0
            if int(cell_mask.sum()) < 30:
                continue
            trusted += 1
            cell_accepted = accepted[ya:yb, xa:xb] & cell_mask
            if int(cell_accepted.sum()) >= 20:
                covered += 1
    return covered / float(max(trusted, 1))


def _region_coverage(mask: np.ndarray, accepted: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return {}
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    height, width = mask.shape
    yy, xx = np.mgrid[:height, :width]
    nx = (xx - x0) / max(x1 - x0, 1.0)
    ny = (yy - y0) / max(y1 - y0, 1.0)
    regions = {
        "forehead": (ny >= 0.04) & (ny < 0.30) & (nx > 0.20) & (nx < 0.80),
        "cheek": (ny >= 0.36) & (ny < 0.70) & ((nx < 0.42) | (nx > 0.58)),
        "nose": (ny >= 0.30) & (ny < 0.66) & (nx >= 0.36) & (nx <= 0.64),
        "philtrum": (ny >= 0.62) & (ny < 0.76) & (nx >= 0.38) & (nx <= 0.62),
        "chin": (ny >= 0.74) & (ny < 0.96) & (nx >= 0.30) & (nx <= 0.70),
        "jaw": (ny >= 0.64) & (ny < 0.94) & ((nx < 0.34) | (nx > 0.66)),
    }
    output: Dict[str, float] = {}
    trusted_mask = mask > 0
    for name, region in regions.items():
        valid = trusted_mask & region
        total = int(valid.sum())
        output[name] = float((accepted & valid).sum() / max(total, 1))
    return output


def _draw_matches(
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    accepted: np.ndarray,
    max_matches: int = 220,
) -> np.ndarray:
    height = max(image_a.shape[0], image_b.shape[0])
    width_a = image_a.shape[1]
    canvas = np.zeros((height, width_a + image_b.shape[1], 3), dtype=np.uint8)
    canvas[: image_a.shape[0], :width_a] = image_a
    canvas[: image_b.shape[0], width_a:] = image_b
    indices = np.flatnonzero(accepted)
    if len(indices) > max_matches:
        indices = indices[np.linspace(0, len(indices) - 1, max_matches).astype(int)]
    for idx in indices:
        pa = tuple(np.rint(points_a[idx]).astype(int))
        pb_raw = np.rint(points_b[idx]).astype(int)
        pb = (int(pb_raw[0] + width_a), int(pb_raw[1]))
        cv2.line(canvas, pa, pb, (80, 220, 130), 1, cv2.LINE_AA)
        cv2.circle(canvas, pa, 2, (60, 230, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pb, 2, (60, 230, 255), -1, cv2.LINE_AA)
    return canvas


def _disparity_preview(disparity: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    preview = np.zeros((*disparity.shape, 3), dtype=np.uint8)
    values = disparity[accepted & np.isfinite(disparity)]
    if not len(values):
        return preview
    low, high = np.percentile(values, [2, 98])
    normalized = np.clip((disparity - low) / max(high - low, 1e-6), 0, 1)
    colors = cv2.applyColorMap(
        np.uint8(np.nan_to_num(normalized) * 255), cv2.COLORMAP_TURBO
    )
    preview[accepted] = colors[accepted]
    return preview


def _point_overlay(image: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    if not np.any(accepted):
        return overlay
    color = np.zeros_like(image)
    color[..., 1] = 220
    color[..., 2] = 90
    overlay[accepted] = cv2.addWeighted(
        image[accepted], 0.35, color[accepted], 0.65, 0
    )
    return overlay


def _rectified_guides(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    combined = np.hstack([image_a, image_b])
    for y in range(40, combined.shape[0], 64):
        cv2.line(combined, (0, y), (combined.shape[1] - 1, y), (70, 200, 255), 1)
    return combined


def write_ply(path: Path, xyz: np.ndarray, colors_bgr: np.ndarray) -> None:
    finite = np.isfinite(xyz).all(axis=1)
    xyz = np.asarray(xyz[finite], dtype=np.float32)
    colors_rgb = np.asarray(colors_bgr[finite, ::-1], dtype=np.uint8)
    if len(xyz) > 120000:
        selection = np.linspace(0, len(xyz) - 1, 120000).astype(np.int64)
        xyz = xyz[selection]
        colors_rgb = colors_rgb[selection]
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point, color in zip(xyz, colors_rgb):
            stream.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _passes_quality(metrics: Mapping[str, object]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    checks = (
        ("sparse_geometric_matches", int(metrics["sparse_geometric_matches"]) >= 100),
        (
            "median_rectified_vertical_error_px",
            float(metrics["median_rectified_vertical_error_px"]) <= 3.0,
        ),
        ("filtered_3d_points", int(metrics["filtered_3d_points"]) >= 3000),
        ("grid_coverage_ratio", float(metrics["grid_coverage_ratio"]) >= 0.30),
        ("covered_semantic_regions", int(metrics["covered_semantic_regions"]) >= 3),
        ("positive_depth_ratio", float(metrics["positive_depth_ratio"]) >= 0.95),
        (
            "median_lr_consistency_px",
            float(metrics["median_lr_consistency_px"]) <= 1.5,
        ),
        ("plausible_depth", bool(metrics["plausible_depth"])),
    )
    for name, passed in checks:
        if not passed:
            failures.append(name)
    return not failures, failures


def process_pair(
    pair_name: str,
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    camera_a: Camera,
    camera_b: Camera,
    output_dir: Path,
    work_size: Tuple[int, int] = WORK_SIZE,
) -> Dict[str, object]:
    K_a = scale_intrinsics(camera_a.K, camera_a.image_size, work_size)
    K_b = scale_intrinsics(camera_b.K, camera_b.image_size, work_size)
    R_ba, t_ba = relative_camera_transform(camera_a, camera_b)
    R1, R2, P1, P2, Q, roi1, _roi2 = cv2.stereoRectify(
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
    map_a = cv2.initUndistortRectifyMap(
        K_a, camera_a.dist, R1, P1, work_size, cv2.CV_32FC1
    )
    map_b = cv2.initUndistortRectifyMap(
        K_b, camera_b.dist, R2, P2, work_size, cv2.CV_32FC1
    )
    rect_a = cv2.remap(image_a, *map_a, cv2.INTER_LINEAR)
    rect_b = cv2.remap(image_b, *map_b, cv2.INTER_LINEAR)
    rect_mask_a = cv2.remap(mask_a, *map_a, cv2.INTER_NEAREST)
    rect_mask_b = cv2.remap(mask_b, *map_b, cv2.INTER_NEAREST)

    valid_roi = np.zeros((work_size[1], work_size[0]), dtype=np.uint8)
    x0 = max(int(roi1[0]), 0)
    y0 = max(int(roi1[1]), 0)
    x1 = min(int(roi1[0] + roi1[2]), work_size[0])
    y1 = min(int(roi1[1] + roi1[3]), work_size[1])
    if x1 > x0 and y1 > y0:
        valid_roi[y0:y1, x0:x1] = 255
    rect_mask_a = cv2.bitwise_and(rect_mask_a, valid_roi)

    points_a, points_b, vertical_error = reciprocal_sift_matches(
        rect_a, rect_b, rect_mask_a, rect_mask_b
    )
    geometric_sparse = vertical_error <= 3.0
    min_disp, max_disp = estimate_disparity_search(
        points_a,
        points_b,
        vertical_error,
        rect_mask_a,
        rect_mask_b,
    )

    disparity_center = int(round(0.5 * (min_disp + max_disp) / 16.0) * 16)
    _source_y, source_x = np.where(rect_mask_a > 0)
    common_shift = (
        int(round(work_size[0] * 0.5 - float(np.median(source_x))))
        if len(source_x)
        else 0
    )
    work_rect_a = _shift_array_exact(rect_a, common_shift)
    work_mask_a = _shift_array_exact(rect_mask_a, common_shift)
    work_rect_b = _shift_array_exact(
        rect_b, common_shift + disparity_center
    )
    work_mask_b = _shift_array_exact(
        rect_mask_b, common_shift + disparity_center
    )
    gray_a = _normalized_gray(work_rect_a)
    gray_b = _normalized_gray(work_rect_b)
    residual_min = min_disp - disparity_center
    residual_max = max_disp - disparity_center
    disparity_residual_ab = _compute_disparity(
        gray_a, gray_b, residual_min, residual_max
    )
    disparity_residual_ba = _compute_disparity(
        gray_b, gray_a, -residual_max, -residual_min
    )
    disparity_work = disparity_residual_ab + float(disparity_center)

    yy, xx = np.mgrid[:work_size[1], :work_size[0]]
    x_b = xx.astype(np.float32) - disparity_residual_ab
    sampled_ba = _sample_map(
        disparity_residual_ba, x_b, yy.astype(np.float32)
    )
    sampled_mask_b = _sample_map(
        work_mask_b.astype(np.float32), x_b, yy.astype(np.float32)
    )
    lr_error_work = np.abs(disparity_residual_ab + sampled_ba)

    min_valid_ab = float(residual_min - 1)
    min_valid_ba = float(-residual_max - 1)
    in_source_mask = work_mask_a > 0
    in_target_mask = sampled_mask_b > 127
    valid_forward = (
        np.isfinite(disparity_residual_ab)
        & (disparity_residual_ab > min_valid_ab)
    )
    valid_reverse = np.isfinite(sampled_ba) & (sampled_ba > min_valid_ba)
    lr_consistent = lr_error_work <= 1.5
    accepted_work = (
        in_source_mask
        & in_target_mask
        & valid_forward
        & valid_reverse
        & lr_consistent
    )
    disparity_ab = _shift_array_exact(
        disparity_work, -common_shift, np.nan
    )
    lr_error = _shift_array_exact(
        lr_error_work, -common_shift, np.nan
    )
    accepted = (
        _shift_array_exact(
            accepted_work.astype(np.uint8), -common_shift
        )
        > 0
    )

    xyz_rect = cv2.reprojectImageTo3D(disparity_ab, Q)
    xyz_front = xyz_rect @ R1
    xyz_side = xyz_front @ R_ba.T + t_ba
    finite = np.isfinite(xyz_front).all(axis=2)
    positive = (xyz_front[..., 2] > 0) & (xyz_side[..., 2] > 0)
    plausible = (
        (xyz_front[..., 2] >= MIN_FACE_DEPTH_M)
        & (xyz_front[..., 2] <= MAX_FACE_DEPTH_M)
    )
    pre_depth = accepted & finite
    candidate_depth = xyz_front[..., 2][pre_depth & positive]
    candidate_depth_percentiles = (
        np.percentile(candidate_depth, [1, 50, 99]).tolist()
        if len(candidate_depth)
        else [float("nan")] * 3
    )
    positive_depth_ratio = float(
        (positive & pre_depth).sum() / max(int(pre_depth.sum()), 1)
    )
    accepted &= finite & positive & plausible

    xyz = xyz_front[accepted]
    colors = rect_a[accepted]
    depth = xyz[:, 2] if len(xyz) else np.empty((0,), dtype=np.float32)
    depth_p01 = float(np.percentile(depth, 1)) if len(depth) else float("nan")
    depth_p99 = float(np.percentile(depth, 99)) if len(depth) else float("nan")
    depth_span = depth_p99 - depth_p01 if len(depth) else float("nan")
    plausible_depth = bool(
        len(depth)
        and MIN_FACE_DEPTH_M <= depth_p01 <= MAX_FACE_DEPTH_M
        and MIN_FACE_DEPTH_M <= depth_p99 <= MAX_FACE_DEPTH_M
        and 0.015 <= depth_span <= 0.50
    )

    region_coverage = _region_coverage(rect_mask_a, accepted)
    covered_regions = [
        name for name, ratio in region_coverage.items() if ratio >= 0.02
    ]
    sparse_count = int(geometric_sparse.sum())
    median_vertical = (
        float(np.median(vertical_error[geometric_sparse]))
        if sparse_count
        else float("inf")
    )
    median_lr = (
        float(np.median(lr_error[accepted]))
        if int(accepted.sum())
        else float("inf")
    )

    metrics: Dict[str, object] = {
        "pair": pair_name,
        "camera_a": camera_a.name,
        "camera_b": camera_b.name,
        "sparse_reciprocal_matches": int(len(points_a)),
        "sparse_geometric_matches": sparse_count,
        "median_rectified_vertical_error_px": median_vertical,
        "disparity_search": [int(min_disp), int(max_disp)],
        "disparity_center_shift_px": int(disparity_center),
        "common_image_shift_px": int(common_shift),
        "residual_disparity_search": [
            int(residual_min),
            int(residual_max),
        ],
        "filtered_3d_points": int(len(xyz)),
        "grid_coverage_ratio": float(_grid_coverage(rect_mask_a, accepted)),
        "region_coverage": region_coverage,
        "covered_regions": covered_regions,
        "covered_semantic_regions": int(len(covered_regions)),
        "positive_depth_ratio": positive_depth_ratio,
        "median_lr_consistency_px": median_lr,
        "depth_p01_m": depth_p01,
        "depth_p99_m": depth_p99,
        "depth_span_m": depth_span,
        "depth_bounds_m": [MIN_FACE_DEPTH_M, MAX_FACE_DEPTH_M],
        "candidate_depth_p01_m": float(candidate_depth_percentiles[0]),
        "candidate_depth_p50_m": float(candidate_depth_percentiles[1]),
        "candidate_depth_p99_m": float(candidate_depth_percentiles[2]),
        "plausible_depth": plausible_depth,
        "baseline_m": float(np.linalg.norm(t_ba)),
        "rectified_focal_px": float(P1[0, 0]),
        "filter_counts": {
            "source_mask": int(in_source_mask.sum()),
            "source_and_target_mask": int(
                (in_source_mask & in_target_mask).sum()
            ),
            "with_forward_disparity": int(
                (in_source_mask & in_target_mask & valid_forward).sum()
            ),
            "with_reverse_disparity": int(
                (
                    in_source_mask
                    & in_target_mask
                    & valid_forward
                    & valid_reverse
                ).sum()
            ),
            "with_lr_consistency": int(accepted_work.sum()),
            "with_positive_depth": int((pre_depth & positive).sum()),
            "with_plausible_depth": int(accepted.sum()),
        },
    }
    passed, failures = _passes_quality(metrics)
    metrics["passed"] = passed
    metrics["failed_gates"] = failures

    cv2.imwrite(
        str(output_dir / f"{pair_name}_rectified.jpg"),
        _rectified_guides(rect_a, rect_b),
    )
    cv2.imwrite(
        str(output_dir / f"{pair_name}_matches.jpg"),
        _draw_matches(
            rect_a, rect_b, points_a, points_b, geometric_sparse
        ),
    )
    cv2.imwrite(
        str(output_dir / f"{pair_name}_disparity.jpg"),
        _disparity_preview(disparity_ab, accepted),
    )
    cv2.imwrite(
        str(output_dir / f"{pair_name}_accepted.jpg"),
        _point_overlay(rect_a, accepted),
    )
    write_ply(output_dir / f"face_points_{pair_name}.ply", xyz, colors)
    metrics["_xyz"] = xyz
    metrics["_colors"] = colors
    return metrics


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


def write_report(
    output_dir: Path,
    pair_results: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    cards: List[str] = []
    for result in pair_results:
        pair = str(result["pair"])
        status = "通过" if result["passed"] else "未通过"
        status_class = "pass" if result["passed"] else "fail"
        failures = ", ".join(str(v) for v in result["failed_gates"]) or "无"
        regions = ", ".join(str(v) for v in result["covered_regions"]) or "无"
        cards.append(
            f"""
            <section class="pair">
              <div class="pair-head">
                <h2>{html.escape(pair)}</h2>
                <span class="status {status_class}">{status}</span>
              </div>
              <div class="metrics">
                <div><b>{int(result['sparse_geometric_matches'])}</b><span>可靠稀疏匹配</span></div>
                <div><b>{float(result['median_rectified_vertical_error_px']):.2f} px</b><span>极线中位误差</span></div>
                <div><b>{int(result['filtered_3d_points']):,}</b><span>过滤后三维点</span></div>
                <div><b>{float(result['grid_coverage_ratio']) * 100:.1f}%</b><span>脸部网格覆盖</span></div>
                <div><b>{float(result['median_lr_consistency_px']):.2f} px</b><span>左右一致性</span></div>
                <div><b>{float(result['depth_span_m']) * 1000:.1f} mm</b><span>深度范围</span></div>
              </div>
              <p>有覆盖的区域：{html.escape(regions)}</p>
              <p>未通过门槛：{html.escape(failures)}</p>
              <div class="images">
                <figure><img src="{pair}_matches.jpg"><figcaption>可靠特征匹配</figcaption></figure>
                <figure><img src="{pair}_accepted.jpg"><figcaption>接受的三维像素</figcaption></figure>
                <figure><img src="{pair}_disparity.jpg"><figcaption>过滤后视差</figcaption></figure>
                <figure><img src="{pair}_rectified.jpg"><figcaption>校正后极线检查</figcaption></figure>
              </div>
            </section>
            """
        )

    overall = "通过" if summary["passed"] else "未通过"
    overall_class = "pass" if summary["passed"] else "fail"
    document = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>跨视角三维证据验证</title>
<style>
:root{{--bg:#0c1117;--panel:#151c24;--line:#34404d;--text:#f2f5f7;--muted:#a9b4bf;--green:#55c982;--red:#e97878}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:"Microsoft YaHei","Segoe UI",sans-serif;letter-spacing:0}}
main{{width:min(1240px,calc(100% - 28px));margin:auto;padding:28px 0 48px}} h1,h2,p{{margin-top:0}} h1{{font-size:28px;margin-bottom:8px}}
.lead{{color:var(--muted);line-height:1.65}} .summary,.pair{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:20px;margin-top:18px}}
.pair-head{{display:flex;align-items:center;justify-content:space-between;gap:16px}} .status{{padding:5px 10px;border-radius:4px;font-weight:700}}
.status.pass{{background:var(--green);color:#07120b}} .status.fail{{background:var(--red);color:#190707}}
.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:14px 0}}
.metrics div{{background:#0f151c;border:1px solid var(--line);padding:12px;min-width:0}} .metrics b{{display:block;font-size:18px;overflow-wrap:anywhere}}
.metrics span{{display:block;color:var(--muted);font-size:12px;margin-top:4px}} .images{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}
figure{{margin:0;background:#0a0f14;border:1px solid var(--line)}} img{{display:block;width:100%;height:auto}} figcaption{{padding:9px 11px;color:var(--muted);font-size:13px}}
code{{color:#77c8df}} @media(max-width:850px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.images{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>跨视角三维证据验证</h1>
<p class="lead">该报告只判断三张 RGB 照片能否恢复足够可靠的三维点，不修改 FLAME、GLB 或相机标定。</p>
<section class="summary">
  <div class="pair-head"><h2>总结果</h2><span class="status {overall_class}">{overall}</span></div>
  <p>{html.escape(str(summary['conclusion']))}</p>
  <p>合并点云：<code>face_points_merged.ply</code>；原始指标：<code>quality.json</code></p>
</section>
{''.join(cards)}
</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def run_validation(
    root: Path,
    captures_dir: Path,
    calibration_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = load_calibration(calibration_path)
    camera_by_view = {camera.view: camera for camera in cameras.values()}
    image_by_view: Dict[str, np.ndarray] = {}
    mask_by_view: Dict[str, np.ndarray] = {}
    for view, camera in camera_by_view.items():
        candidates = sorted(captures_dir.glob(f"{camera.name}_*.jpg"))
        if not candidates:
            raise FileNotFoundError(f"No capture found for {camera.name}")
        image = cv2.imread(str(candidates[0]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {candidates[0]}")
        image_by_view[view] = cv2.resize(image, WORK_SIZE, interpolation=cv2.INTER_AREA)
        mask_path = root / "output" / "debug" / f"{view}_face_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Missing preprocessing mask: {mask_path}")
        mask_by_view[view] = restore_mask_to_work_frame(mask)

    pair_specs = (
        ("front_left", "front", "left"),
        ("front_right", "front", "right"),
    )
    pair_results: List[Dict[str, object]] = []
    merged_xyz: List[np.ndarray] = []
    merged_colors: List[np.ndarray] = []
    for pair_name, view_a, view_b in pair_specs:
        result = process_pair(
            pair_name=pair_name,
            image_a=image_by_view[view_a],
            image_b=image_by_view[view_b],
            mask_a=mask_by_view[view_a],
            mask_b=mask_by_view[view_b],
            camera_a=camera_by_view[view_a],
            camera_b=camera_by_view[view_b],
            output_dir=output_dir,
        )
        merged_xyz.append(result.pop("_xyz"))
        merged_colors.append(result.pop("_colors"))
        pair_results.append(result)

    all_xyz = np.concatenate(merged_xyz, axis=0) if merged_xyz else np.empty((0, 3))
    all_colors = (
        np.concatenate(merged_colors, axis=0)
        if merged_colors
        else np.empty((0, 3), dtype=np.uint8)
    )
    write_ply(output_dir / "face_points_merged.ply", all_xyz, all_colors)
    passed = all(bool(result["passed"]) for result in pair_results)
    if passed:
        conclusion = (
            "两组相邻视角均通过固定门槛，可以进入三维点约束的 FLAME "
            "整体形状拟合实验。"
        )
    else:
        failed_pairs = [
            str(result["pair"]) for result in pair_results if not result["passed"]
        ]
        sparse_total = sum(
            int(result["sparse_geometric_matches"]) for result in pair_results
        )
        if sparse_total == 0:
            conclusion = (
                "经典 SIFT/SGBM 双目验证未通过：少量视差点集中在强纹理区域，"
                "缺少可独立验证的稀疏几何对应，也没有覆盖双颊和下颌。当前点云"
                "不能用于拉动 FLAME；下一步应使用学习型跨视角匹配，并只允许"
                "受限外参微调。"
            )
        else:
            conclusion = (
                "当前三维证据尚未通过固定门槛；先处理匹配或标定问题，不应接入 "
                f"FLAME。未通过：{', '.join(failed_pairs)}。"
            )
    summary = {
        "passed": passed,
        "conclusion": conclusion,
        "input": str(captures_dir),
        "calibration": str(calibration_path),
        "work_size": list(WORK_SIZE),
        "pairs": pair_results,
    }
    (output_dir / "quality.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, pair_results, summary)
    return summary
