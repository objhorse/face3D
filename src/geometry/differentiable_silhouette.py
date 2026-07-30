from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


Resolution = Union[int, Tuple[int, int]]


@dataclass(frozen=True)
class SilhouetteTarget:
    target_np: np.ndarray
    reliability_np: np.ndarray
    trusted_curve_np: np.ndarray
    trusted_band_np: np.ndarray
    signed_distance_np: np.ndarray
    sdf_support_np: np.ndarray
    region_supports_np: Dict[str, np.ndarray]
    source_shape: Tuple[int, int]
    render_shape: Tuple[int, int]
    metadata: Dict[str, object]

    def tensors(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        target = torch.tensor(self.target_np, device=device, dtype=torch.float32)
        reliability = torch.tensor(self.reliability_np, device=device, dtype=torch.float32)
        return target, reliability

    def sdf_tensors(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        signed_distance = torch.tensor(
            self.signed_distance_np, device=device, dtype=torch.float32
        )
        support = torch.tensor(self.sdf_support_np, device=device, dtype=torch.float32)
        return signed_distance, support

    def regional_sdf_tensors(
        self, device: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        signed_distance = torch.tensor(
            self.signed_distance_np, device=device, dtype=torch.float32
        )
        supports = {
            name: torch.tensor(values, device=device, dtype=torch.float32)
            for name, values in self.region_supports_np.items()
        }
        return signed_distance, supports


_VERTICAL_REGIONS = (
    ("forehead", 0.10, 0.22),
    ("temple", 0.22, 0.36),
    ("cheekbone", 0.36, 0.50),
    ("cheek", 0.50, 0.66),
    ("jaw", 0.66, 0.86),
    ("chin", 0.86, 1.01),
)

IDENTITY_SILHOUETTE_REGIONS = ("forehead", "temple", "cheekbone", "cheek")


def _render_shape(image_shape: Tuple[int, int], resolution: Resolution) -> Tuple[int, int]:
    source_h, source_w = int(image_shape[0]), int(image_shape[1])
    if isinstance(resolution, tuple):
        return max(16, int(resolution[0])), max(16, int(resolution[1]))
    longest = max(source_h, source_w, 1)
    render_h = max(16, int(round(source_h * float(resolution) / longest)))
    render_w = max(16, int(round(source_w * float(resolution) / longest)))
    return render_h, render_w


def _largest_binary_component(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return np.asarray(labels == label, dtype=np.uint8)


def _signed_distance_field(binary: np.ndarray, face_width: int) -> np.ndarray:
    inside = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((1 - binary).astype(np.uint8), cv2.DIST_L2, 5)
    signed = (outside - inside) / float(max(int(face_width), 1))
    boundary = cv2.morphologyEx(
        binary.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    signed[boundary > 0] = 0.0
    return signed.astype(np.float32)


def _build_sdf_support(
    binary: np.ndarray,
    view_name: str,
    curve_y0: int,
    y1: int,
    profile_x_by_row: Dict[int, int],
) -> np.ndarray:
    height, width = binary.shape
    ys, xs = np.nonzero(binary)
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_width = max(1, x1 - x0 + 1)
    outer_margin = max(2, int(round(0.08 * bbox_width)))
    support = np.zeros((height, width), dtype=np.float32)
    normalized_name = str(view_name).lower()
    jaw_start = curve_y0 + int(round(0.64 * max(1, y1 - curve_y0 + 1)))

    for y in range(curve_y0, y1 + 1):
        row_x = np.flatnonzero(binary[y])
        if row_x.size < 2:
            continue
        left, right = int(row_x[0]), int(row_x[-1])
        row_width = max(1, right - left + 1)
        if normalized_name == "front":
            inward = max(3, int(round(0.28 * row_width)))
            support[y, max(0, left - outer_margin) : min(width, left + inward + 1)] = 1.0
            support[y, max(0, right - inward) : min(width, right + outer_margin + 1)] = 1.0
            if y >= jaw_start:
                support[y, max(0, left - outer_margin) : min(width, right + outer_margin + 1)] = 1.0
            continue

        profile_x = profile_x_by_row.get(y)
        if profile_x is None:
            continue
        inward = max(3, int(round(0.38 * row_width)))
        center_x = (width - 1) * 0.5
        if profile_x <= center_x:
            start = max(0, profile_x - outer_margin)
            end = min(width, profile_x + inward + 1)
        else:
            start = max(0, profile_x - inward)
            end = min(width, profile_x + outer_margin + 1)
        support[y, start:end] = 1.0

    # A small vertical dilation keeps gradients available between sampled mask rows.
    support = cv2.dilate(
        np.uint8(support > 0),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
    ).astype(np.float32)
    support[: max(0, curve_y0 - 1)] = 0.0
    return support


def _build_region_supports(
    sdf_support: np.ndarray,
    y0: int,
    y1: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, list[int]]]:
    height, _width = sdf_support.shape
    span = max(1, int(y1) - int(y0) + 1)
    supports: Dict[str, np.ndarray] = {}
    ranges: Dict[str, list[int]] = {}
    for name, start_fraction, end_fraction in _VERTICAL_REGIONS:
        start = max(0, int(round(y0 + start_fraction * span)))
        end = min(height, int(round(y0 + end_fraction * span)))
        region = np.zeros_like(sdf_support, dtype=np.float32)
        region[start:end] = sdf_support[start:end]
        supports[name] = region
        ranges[name] = [int(start), int(max(start, end - 1))]
    return supports, ranges


def build_silhouette_target(
    mask: np.ndarray,
    view_name: str,
    resolution: Resolution = 256,
    context_weight: float = 0.08,
) -> SilhouetteTarget:
    if mask is None or np.asarray(mask).ndim < 2:
        raise ValueError("A two-dimensional face mask is required")
    source_shape = tuple(int(v) for v in np.asarray(mask).shape[:2])
    render_shape = _render_shape(source_shape, resolution)
    resized = cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (render_shape[1], render_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    binary = _largest_binary_component(resized)
    ys, xs = np.nonzero(binary)
    if len(xs) < 64:
        raise ValueError("Face mask has too little foreground for silhouette refinement")

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_h = max(1, y1 - y0 + 1)
    center_x = (render_shape[1] - 1) * 0.5
    trusted_curve = np.zeros(render_shape, dtype=np.uint8)
    profile_x_by_row: Dict[int, int] = {}
    curve_y0 = min(y1, y0 + max(2, int(round(0.10 * bbox_h))))
    curve_rows = 0

    normalized_name = str(view_name).lower()
    for y in range(curve_y0, y1 + 1):
        row_x = np.flatnonzero(binary[y])
        if row_x.size < 2:
            continue
        left, right = int(row_x[0]), int(row_x[-1])
        if normalized_name == "front":
            trusted_curve[y, left] = 1
            trusted_curve[y, right] = 1
        else:
            profile_x = left if abs(left - center_x) <= abs(right - center_x) else right
            trusted_curve[y, profile_x] = 1
            profile_x_by_row[y] = profile_x
        curve_rows += 1

    boundary = cv2.morphologyEx(
        binary,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    jaw_start = y0 + int(round(0.68 * bbox_h))
    jaw = np.zeros_like(binary)
    if normalized_name == "front":
        jaw[jaw_start : y1 + 1] = boundary[jaw_start : y1 + 1]
    else:
        bbox_w = max(1, x1 - x0 + 1)
        jaw_reach = max(2, int(round(0.45 * bbox_w)))
        for y in range(jaw_start, y1 + 1):
            row_boundary = np.flatnonzero(boundary[y])
            profile_x = profile_x_by_row.get(y)
            if row_boundary.size == 0 or profile_x is None:
                continue
            if profile_x <= center_x:
                selected = row_boundary[
                    (row_boundary >= profile_x) & (row_boundary <= profile_x + jaw_reach)
                ]
            else:
                selected = row_boundary[
                    (row_boundary <= profile_x) & (row_boundary >= profile_x - jaw_reach)
                ]
            jaw[y, selected] = 1
    trusted = np.maximum(trusted_curve, jaw)

    band_radius = max(2, int(round(max(render_shape) / 128.0 * 4.0)))
    band_size = band_radius * 2 + 1
    trusted_band = cv2.dilate(
        trusted,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_size, band_size)),
    ).astype(np.float32)
    context_radius = max(2, int(round(max(render_shape) / 128.0 * 5.0)))
    context_size = context_radius * 2 + 1
    context = cv2.dilate(
        binary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (context_size, context_size)),
    ).astype(np.float32)
    reliability = np.maximum(context * float(context_weight), trusted_band)

    # The horizontal top edge is usually a hairline or crop boundary, not FLAME skin.
    top_guard_end = min(render_shape[0], curve_y0)
    reliability[y0:top_guard_end] = np.minimum(
        reliability[y0:top_guard_end], float(context_weight)
    )
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32)
    face_width_render_px = int(x1 - x0 + 1)
    signed_distance = _signed_distance_field(binary, face_width_render_px)
    sdf_support = _build_sdf_support(
        binary,
        normalized_name,
        curve_y0,
        y1,
        profile_x_by_row,
    )
    region_supports, region_ranges = _build_region_supports(sdf_support, y0, y1)

    metadata: Dict[str, object] = {
        "view": normalized_name,
        "bbox_xyxy": [x0, y0, x1, y1],
        "face_width_render_px": face_width_render_px,
        "face_width_source_px": float((x1 - x0 + 1) * source_shape[1] / render_shape[1]),
        "curve_rows": int(curve_rows),
        "reliable_fraction": float(np.mean(reliability >= 0.5)),
        "context_fraction": float(np.mean(reliability > 0.0)),
        "region_row_ranges": region_ranges,
    }
    return SilhouetteTarget(
        target_np=binary.astype(np.float32),
        reliability_np=reliability,
        trusted_curve_np=trusted.astype(np.float32),
        trusted_band_np=trusted_band.astype(np.float32),
        signed_distance_np=signed_distance,
        sdf_support_np=sdf_support,
        region_supports_np=region_supports,
        source_shape=source_shape,
        render_shape=render_shape,
        metadata=metadata,
    )


def make_interior_landmark_weights(
    point_count: int,
    base_weight: float,
    stable_indices: Iterable[int],
    stable_weight: float,
    device: str,
    contour_count: int = 17,
) -> torch.Tensor:
    weights = torch.full(
        (int(point_count),), float(base_weight), device=device, dtype=torch.float32
    )
    weights[: min(int(contour_count), int(point_count))] = 0.0
    stable = np.asarray(list(stable_indices), dtype=np.int64)
    stable = stable[(stable >= contour_count) & (stable < point_count)]
    if stable.size:
        index = torch.tensor(stable, device=device, dtype=torch.long)
        weights[index] = weights[index] + float(stable_weight)
    return weights


def camera_vertices_to_clip(
    vertices: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_shape: Tuple[int, int],
) -> torch.Tensor:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    if image_h < 2 or image_w < 2:
        raise ValueError("image dimensions must both be at least 2")

    camera = vertices @ R.transpose(0, 1) + t.reshape(1, 3)
    homogeneous = camera @ K.transpose(0, 1)
    depth = camera[:, 2]
    x_clip = 2.0 * homogeneous[:, 0] / float(image_w - 1) - depth
    y_clip = depth - 2.0 * homogeneous[:, 1] / float(image_h - 1)
    z_clip = depth * 0.5
    return torch.stack([x_clip, y_clip, z_clip, depth], dim=1)


def create_cuda_raster_context(device: str = "cuda"):
    if not torch.cuda.is_available():
        raise RuntimeError("Differentiable silhouette rendering requires CUDA")
    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise RuntimeError("nvdiffrast is required for silhouette refinement") from exc
    with torch.cuda.device(torch.device(device)):
        return dr.RasterizeCudaContext(device=device)


def render_soft_silhouette(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_shape: Tuple[int, int],
    render_shape: Tuple[int, int],
    context,
) -> torch.Tensor:
    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise RuntimeError("nvdiffrast is required for silhouette refinement") from exc

    clip = camera_vertices_to_clip(vertices, K, R, t, image_shape).unsqueeze(0).contiguous()
    triangles = faces.to(device=vertices.device, dtype=torch.int32).contiguous()
    rast, _rast_db = dr.rasterize(context, clip, triangles, resolution=list(render_shape))
    vertex_alpha = torch.ones(
        (1, vertices.shape[0], 1), device=vertices.device, dtype=vertices.dtype
    )
    alpha, _ = dr.interpolate(vertex_alpha, rast, triangles)
    alpha = dr.antialias(alpha, rast, clip, triangles)
    # nvdiffrast uses bottom-up image memory; OpenCV masks are top-down.
    return torch.flip(alpha[0, :, :, 0], dims=[0]).clamp(0.0, 1.0)


def weighted_silhouette_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reliability: torch.Tensor,
    dice_weight: float = 1.0,
    l1_weight: float = 0.5,
) -> torch.Tensor:
    weight_sum = reliability.sum().clamp_min(1e-6)
    intersection = (reliability * prediction * target).sum()
    denominator = (reliability * prediction).sum() + (reliability * target).sum()
    dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    l1 = (reliability * torch.abs(prediction - target)).sum() / weight_sum
    return float(dice_weight) * dice + float(l1_weight) * l1


def soft_silhouette_boundary(prediction: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 2:
        raise ValueError("prediction must have shape (H, W)")
    image = prediction[None, None]
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=prediction.device,
        dtype=prediction.dtype,
    )[None, None]
    sobel_y = sobel_x.transpose(-1, -2)
    grad_x = F.conv2d(padded, sobel_x)
    grad_y = F.conv2d(padded, sobel_y)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)[0, 0]
    scale = magnitude.detach().amax().clamp_min(1e-6)
    return (magnitude / scale).clamp(0.0, 1.0)


def signed_distance_boundary_loss(
    prediction: torch.Tensor,
    signed_distance: torch.Tensor,
    support: torch.Tensor,
    *,
    boundary: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if prediction.shape != signed_distance.shape or prediction.shape != support.shape:
        raise ValueError("prediction, signed_distance, and support must share a shape")
    boundary_weight = soft_silhouette_boundary(prediction) if boundary is None else boundary
    weights = boundary_weight * support
    return (weights * signed_distance.abs()).sum() / weights.sum().clamp_min(1e-6)


def regional_signed_distance_boundary_loss(
    prediction: torch.Tensor,
    signed_distance: torch.Tensor,
    region_supports: Mapping[str, torch.Tensor],
    *,
    boundary: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average SDF contour loss per semantic vertical region."""
    if prediction.shape != signed_distance.shape:
        raise ValueError("prediction and signed_distance must share a shape")
    boundary_weight = soft_silhouette_boundary(prediction) if boundary is None else boundary
    losses = []
    for support in region_supports.values():
        if support.shape != prediction.shape:
            raise ValueError("every region support must match prediction shape")
        if not bool(torch.count_nonzero(support).item()):
            continue
        weights = boundary_weight * support
        losses.append(
            (weights * signed_distance.abs()).sum() / weights.sum().clamp_min(1e-6)
        )
    if not losses:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return torch.stack(losses).mean()


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    return cv2.morphologyEx(
        binary,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)


def silhouette_metrics(
    prediction: np.ndarray,
    target: SilhouetteTarget,
) -> Dict[str, object]:
    pred = np.asarray(prediction >= 0.5, dtype=np.uint8)
    truth = np.asarray(target.target_np >= 0.5, dtype=np.uint8)
    reliability = np.asarray(target.reliability_np, dtype=np.float32)
    weighted_intersection = float((reliability * pred * truth).sum())
    weighted_union = float((reliability * np.maximum(pred, truth)).sum())
    weighted_sum = float((reliability * pred).sum() + (reliability * truth).sum())
    weighted_iou = weighted_intersection / max(weighted_union, 1e-6)
    weighted_dice = 2.0 * weighted_intersection / max(weighted_sum, 1e-6)

    pred_boundary = _binary_boundary(pred)
    target_points = np.asarray(target.trusted_curve_np > 0, dtype=bool)
    trusted_band = np.asarray(target.trusted_band_np > 0, dtype=bool)
    pred_points_in_band = pred_boundary & trusted_band
    if not np.any(pred_boundary) or not np.any(target_points):
        target_to_model_render_px = float("inf")
        model_to_target_render_px = float("inf")
        boundary_render_px = float("inf")
    else:
        distance_to_pred = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5)
        distance_to_target = cv2.distanceTransform((~target_points).astype(np.uint8), cv2.DIST_L2, 5)
        target_to_model_render_px = float(distance_to_pred[target_points].mean())
        if np.any(pred_points_in_band):
            model_to_target_render_px = float(distance_to_target[pred_points_in_band].mean())
            boundary_render_px = 0.5 * (
                target_to_model_render_px + model_to_target_render_px
            )
        else:
            model_to_target_render_px = float("inf")
            boundary_render_px = float("inf")

    scale_y = target.source_shape[0] / float(target.render_shape[0])
    scale_x = target.source_shape[1] / float(target.render_shape[1])
    boundary_source_px = boundary_render_px * 0.5 * (scale_x + scale_y)
    target_to_model_source_px = target_to_model_render_px * 0.5 * (scale_x + scale_y)
    model_to_target_source_px = model_to_target_render_px * 0.5 * (scale_x + scale_y)
    face_width_source_px = float(target.metadata["face_width_source_px"])
    boundary_face_width_pct = 100.0 * boundary_source_px / max(face_width_source_px, 1e-6)
    render_pixel_face_width_pct = 100.0 / max(
        float(target.metadata["face_width_render_px"]), 1.0
    )
    regions: Dict[str, Dict[str, float]] = {}
    finite_region_boundaries = []
    if np.any(pred_boundary) and np.any(target_points):
        distance_to_pred = cv2.distanceTransform(
            (~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5
        )
        distance_to_target = cv2.distanceTransform(
            (~target_points).astype(np.uint8), cv2.DIST_L2, 5
        )
        for name, support in target.region_supports_np.items():
            region_rows = np.any(np.asarray(support) > 0, axis=1)
            region_mask = np.repeat(region_rows[:, None], pred.shape[1], axis=1)
            region_target = target_points & region_mask
            region_prediction = pred_boundary & trusted_band & region_mask
            if np.any(region_target) and np.any(region_prediction):
                target_distance = float(distance_to_pred[region_target].mean())
                prediction_distance = float(
                    distance_to_target[region_prediction].mean()
                )
                region_boundary_render_px = 0.5 * (
                    target_distance + prediction_distance
                )
                region_boundary_source_px = region_boundary_render_px * 0.5 * (
                    scale_x + scale_y
                )
            else:
                region_boundary_source_px = float("inf")

            signed_width_values = []
            if str(target.metadata.get("view", "")) == "front":
                for row in np.flatnonzero(region_rows):
                    target_x = np.flatnonzero(truth[row])
                    prediction_x = np.flatnonzero(pred[row])
                    if target_x.size >= 2 and prediction_x.size >= 2:
                        target_width = int(target_x[-1] - target_x[0] + 1)
                        prediction_width = int(prediction_x[-1] - prediction_x[0] + 1)
                        signed_width_values.append(prediction_width - target_width)
            signed_width_error = (
                float(np.median(signed_width_values) * scale_x)
                if signed_width_values
                else float("nan")
            )
            regions[name] = {
                "boundary_mean_px": float(region_boundary_source_px),
                "signed_width_error_px": signed_width_error,
                "row_count": int(np.count_nonzero(region_rows)),
            }
            if np.isfinite(region_boundary_source_px):
                finite_region_boundaries.append(region_boundary_source_px)
    regional_balanced_boundary = (
        float(np.mean(finite_region_boundaries))
        if finite_region_boundaries
        else float("inf")
    )
    identity_boundaries = [
        regions[name]["boundary_mean_px"]
        for name in IDENTITY_SILHOUETTE_REGIONS
        if name in regions and np.isfinite(regions[name]["boundary_mean_px"])
    ]
    identity_balanced_boundary = (
        float(np.mean(identity_boundaries))
        if identity_boundaries
        else float("inf")
    )
    return {
        "weighted_iou": float(weighted_iou),
        "weighted_dice": float(weighted_dice),
        "trusted_region_iou": float(weighted_iou),
        "trusted_region_dice": float(weighted_dice),
        "boundary_mean_px": float(boundary_source_px),
        "trusted_boundary_mean_px": float(boundary_source_px),
        "trusted_boundary_face_width_pct": float(boundary_face_width_pct),
        "target_to_model_px": float(target_to_model_source_px),
        "model_to_target_px": float(model_to_target_source_px),
        "face_width_source_px": face_width_source_px,
        "render_pixel_face_width_pct": float(render_pixel_face_width_pct),
        "predicted_area_ratio": float(pred.mean()),
        "target_area_ratio": float(truth.mean()),
        "regional_balanced_boundary_mean_px": regional_balanced_boundary,
        "identity_balanced_boundary_mean_px": identity_balanced_boundary,
        "identity_region_names": list(IDENTITY_SILHOUETTE_REGIONS),
        "regions": regions,
    }


def evaluate_geometry_candidate(
    before_records: Iterable[Dict[str, object]],
    after_records: Iterable[Dict[str, object]],
    *,
    mesh_quality_gate: Optional[Dict[str, object]] = None,
    min_boundary_improve_pct: float = 0.10,
    min_improved_views: int = 2,
    max_view_worsen_pct: float = 0.15,
    max_overlap_drop: float = 0.005,
    max_interior_mean_worsen_pct: float = 0.15,
    max_interior_view_worsen_pct: float = 0.25,
    min_relative_boundary_improve: float = 0.0,
    min_front_relative_boundary_improve: float = 0.0,
) -> Dict[str, object]:
    """Apply independent geometry gates without constructing a mixed score."""
    before_by_view = {str(record["view"]): record for record in before_records}
    after_by_view = {str(record["view"]): record for record in after_records}
    per_view = []
    for view in sorted(set(before_by_view) & set(after_by_view)):
        before = before_by_view[view]
        after = after_by_view[view]
        before_silhouette = before.get("silhouette")
        after_silhouette = after.get("silhouette")
        if not isinstance(before_silhouette, dict) or not isinstance(after_silhouette, dict):
            continue
        before_boundary = float(before_silhouette.get("trusted_boundary_face_width_pct", np.inf))
        after_boundary = float(after_silhouette.get("trusted_boundary_face_width_pct", np.inf))
        face_width = float(after_silhouette.get("face_width_source_px", 0.0))
        if not np.isfinite(before_boundary) or not np.isfinite(after_boundary) or face_width <= 0:
            continue
        before_interior_pct = 100.0 * float(before.get("interior_mean_px", np.inf)) / face_width
        after_interior_pct = 100.0 * float(after.get("interior_mean_px", np.inf)) / face_width
        before_overlap = float(before_silhouette.get("trusted_region_dice", 0.0))
        after_overlap = float(after_silhouette.get("trusted_region_dice", 0.0))
        measurement_tolerance = max(
            float(max_view_worsen_pct),
            float(before_silhouette.get("render_pixel_face_width_pct", 0.0)),
            float(after_silhouette.get("render_pixel_face_width_pct", 0.0)),
        )
        boundary_improve = before_boundary - after_boundary
        relative_boundary_improve = boundary_improve / max(before_boundary, 1e-6)
        relative_measurement_tolerance = measurement_tolerance / max(
            before_boundary, 1e-6
        )
        per_view.append({
            "view": view,
            "before_boundary_pct": before_boundary,
            "after_boundary_pct": after_boundary,
            "boundary_improve_pct_points": boundary_improve,
            "relative_boundary_improve": relative_boundary_improve,
            "relative_measurement_tolerance": relative_measurement_tolerance,
            "measurement_tolerance_pct_points": measurement_tolerance,
            "measurably_improved": boundary_improve > measurement_tolerance,
            "preserved_within_resolution": boundary_improve >= -measurement_tolerance,
            "before_overlap_dice": before_overlap,
            "after_overlap_dice": after_overlap,
            "overlap_drop": before_overlap - after_overlap,
            "before_interior_pct": before_interior_pct,
            "after_interior_pct": after_interior_pct,
            "interior_worsen_pct_points": after_interior_pct - before_interior_pct,
        })

    valid_views = len(per_view)
    if per_view:
        boundary_improve = float(np.mean([
            item["boundary_improve_pct_points"] for item in per_view
        ]))
        relative_boundary_improve = float(np.mean([
            item["relative_boundary_improve"] for item in per_view
        ]))
        relative_measurement_tolerance = float(np.mean([
            item["relative_measurement_tolerance"] for item in per_view
        ]))
        improved_views = int(sum(
            item["measurably_improved"] for item in per_view
        ))
        preserved_views = int(sum(
            item["preserved_within_resolution"] for item in per_view
        ))
        max_view_worsen = float(max(
            0.0, max(-item["boundary_improve_pct_points"] for item in per_view)
        ))
        max_view_excess_worsen = float(max(
            0.0,
            max(
                -item["boundary_improve_pct_points"]
                - item["measurement_tolerance_pct_points"]
                for item in per_view
            ),
        ))
        overlap_drop = float(np.mean([
            item["before_overlap_dice"] for item in per_view
        ]) - np.mean([
            item["after_overlap_dice"] for item in per_view
        ]))
        interior_mean_worsen = float(np.mean([
            item["after_interior_pct"] for item in per_view
        ]) - np.mean([
            item["before_interior_pct"] for item in per_view
        ]))
        interior_max_view_worsen = float(max(
            item["interior_worsen_pct_points"] for item in per_view
        ))
    else:
        boundary_improve = float("-inf")
        relative_boundary_improve = float("-inf")
        relative_measurement_tolerance = 0.0
        improved_views = 0
        preserved_views = 0
        max_view_worsen = float("inf")
        max_view_excess_worsen = float("inf")
        overlap_drop = float("inf")
        interior_mean_worsen = float("inf")
        interior_max_view_worsen = float("inf")

    mesh_passed = bool((mesh_quality_gate or {"passed": True}).get("passed", False))
    front_relative_boundary_improve = next(
        (
            float(item["relative_boundary_improve"])
            for item in per_view
            if item["view"] == "front"
        ),
        float("-inf"),
    )
    front_relative_measurement_tolerance = next(
        (
            float(item["relative_measurement_tolerance"])
            for item in per_view
            if item["view"] == "front"
        ),
        0.0,
    )
    gates = {
        "enough_valid_views": valid_views >= 2,
        "trusted_boundary_improved": boundary_improve >= float(min_boundary_improve_pct),
        "relative_boundary_improved": (
            relative_boundary_improve >= float(min_relative_boundary_improve)
        ),
        "front_relative_boundary_improved": (
            front_relative_boundary_improve
            >= float(min_front_relative_boundary_improve)
        ),
        "view_consistency": (
            improved_views >= int(min_improved_views)
            and preserved_views == valid_views
            and max_view_excess_worsen <= 1e-9
        ),
        "trusted_overlap_preserved": overlap_drop <= float(max_overlap_drop),
        "interior_mean_preserved": (
            interior_mean_worsen <= float(max_interior_mean_worsen_pct)
        ),
        "interior_views_preserved": (
            interior_max_view_worsen <= float(max_interior_view_worsen_pct)
        ),
        "mesh_quality": mesh_passed,
    }
    failures = [name for name, passed in gates.items() if not passed]
    return {
        "accepted": not failures,
        "reason": "accepted by independent geometry gates" if not failures else "rejected: " + ", ".join(failures),
        "failed_gates": failures,
        "gates": gates,
        "metrics": {
            "valid_views": valid_views,
            "trusted_boundary_improve_pct_points": boundary_improve,
            "relative_boundary_improve": relative_boundary_improve,
            "relative_measurement_tolerance": relative_measurement_tolerance,
            "front_relative_boundary_improve": front_relative_boundary_improve,
            "front_relative_measurement_tolerance": (
                front_relative_measurement_tolerance
            ),
            "improved_views": improved_views,
            "preserved_views": preserved_views,
            "max_view_worsen_pct_points": max_view_worsen,
            "max_view_excess_worsen_pct_points": max_view_excess_worsen,
            "trusted_overlap_drop": overlap_drop,
            "interior_mean_worsen_pct_points": interior_mean_worsen,
            "interior_max_view_worsen_pct_points": interior_max_view_worsen,
        },
        "thresholds": {
            "min_boundary_improve_pct": float(min_boundary_improve_pct),
            "min_relative_boundary_improve": float(min_relative_boundary_improve),
            "min_front_relative_boundary_improve": float(
                min_front_relative_boundary_improve
            ),
            "min_improved_views": int(min_improved_views),
            "max_view_worsen_pct": float(max_view_worsen_pct),
            "view_consistency_rule": (
                "min_improved_views measurably improved and every valid view "
                "preserved within max(configured tolerance, one render pixel)"
            ),
            "max_overlap_drop": float(max_overlap_drop),
            "max_interior_mean_worsen_pct": float(max_interior_mean_worsen_pct),
            "max_interior_view_worsen_pct": float(max_interior_view_worsen_pct),
        },
        "per_view": per_view,
        "mesh_quality_gate": mesh_quality_gate or {"passed": True},
        "legacy_fixed_contour_excluded": True,
        "texture_metrics_excluded": True,
    }


def save_silhouette_debug(
    path: str,
    source_image: np.ndarray,
    prediction: np.ndarray,
    target: SilhouetteTarget,
) -> None:
    h, w = target.render_shape
    image = cv2.resize(source_image, (w, h), interpolation=cv2.INTER_AREA)
    pred = np.asarray(prediction >= 0.5, dtype=np.uint8)
    truth = np.asarray(target.target_np >= 0.5, dtype=np.uint8)
    pred_boundary = _binary_boundary(pred)
    target_boundary = _binary_boundary(truth)
    overlay = image.copy()
    overlay[target_boundary] = (255, 0, 255)
    overlay[pred_boundary] = (0, 255, 255)
    overlay[target.trusted_curve_np > 0] = (0, 255, 0)
    weight_vis = np.uint8(np.clip(target.reliability_np, 0.0, 1.0) * 255.0)
    weight_vis = cv2.applyColorMap(weight_vis, cv2.COLORMAP_VIRIDIS)
    target_vis = cv2.cvtColor(np.uint8(truth * 255), cv2.COLOR_GRAY2BGR)
    pred_vis = cv2.cvtColor(np.uint8(np.clip(prediction, 0.0, 1.0) * 255), cv2.COLOR_GRAY2BGR)
    canvas = np.hstack([image, target_vis, pred_vis, weight_vis, overlay])
    cv2.imwrite(str(path), canvas)
