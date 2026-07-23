"""Static truth report for strict, unwarped projective texture sampling."""

from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.appearance.projective_sampling import (
    DEFAULT_POSITIVE_DEPTH_EPSILON,
    compare_sampling_coordinates,
    project_points_strict,
)
from src.module3_texture import load_cameras, load_mesh_obj


SCHEMA_VERSION = 1
FINAL_SAMPLING_MODE = "unwarped_projective"
CPU_FALLBACK_MAX_FACES = 5000
_ALLOWED_VIEWS = frozenset({"left", "front", "right"})
_MAX_VISIBLE_POINTS = 120
_MAX_WARP_ARROWS = 48
MIN_DIAGNOSTIC_VISIBLE_VERTICES = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_sha256(image: np.ndarray) -> str:
    array = np.asarray(image)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validate_view_names(names, source: str) -> None:
    invalid = sorted(set(names) - _ALLOWED_VIEWS)
    if invalid:
        raise ValueError(
            f"{source} contains unsupported view name(s): {', '.join(invalid)}; "
            "only left/front/right are allowed"
        )


def _safe_output_path(output_root: Path, relative_name: str) -> Path:
    root = Path(output_root).resolve()
    candidate = (root / relative_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output path escapes report directory: {relative_name}") from exc
    if candidate == root:
        raise ValueError("output path must name a file inside the report directory")
    return candidate


def _validate_image(image: np.ndarray, view_name: str) -> np.ndarray:
    array = np.asarray(image)
    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[2] != 3
        or array.shape[0] <= 0
        or array.shape[1] <= 0
    ):
        raise ValueError(
            f"image '{view_name}' must be a non-empty uint8 RGB array with shape (H, W, 3)"
        )
    return array


def _normalize_image_size(value, name: str) -> tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer or (height, width)")
    if isinstance(value, (int, np.integer)):
        size = int(value)
        if size <= 0:
            raise ValueError(f"{name} must be positive")
        return size, size
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a positive integer or (height, width)"
        ) from exc
    if (
        len(items) != 2
        or any(isinstance(item, bool) for item in items)
        or any(not isinstance(item, (int, np.integer)) for item in items)
        or any(int(item) <= 0 for item in items)
    ):
        raise ValueError(f"{name} must be a positive integer or (height, width)")
    return int(items[0]), int(items[1])


def _resolve_expected_image_sizes(
    image_views,
    expected_image_size,
    expected_image_sizes,
) -> dict[str, tuple[int, int]]:
    if (expected_image_size is None) == (expected_image_sizes is None):
        raise ValueError(
            "provide exactly one of expected_image_size or expected_image_sizes"
        )
    if expected_image_sizes is not None:
        if not isinstance(expected_image_sizes, dict):
            raise ValueError("expected_image_sizes must be a view-to-size dictionary")
        _validate_view_names(expected_image_sizes, "expected_image_sizes")
        missing = sorted(set(image_views) - set(expected_image_sizes))
        extra = sorted(set(expected_image_sizes) - set(image_views))
        if missing or extra:
            raise ValueError(
                "expected_image_sizes must exactly match image views; "
                f"missing={missing}, extra={extra}"
            )
        return {
            view: _normalize_image_size(size, f"expected_image_sizes[{view!r}]")
            for view, size in expected_image_sizes.items()
        }
    normalized = _normalize_image_size(expected_image_size, "expected_image_size")
    return {view: normalized for view in image_views}


def _validate_camera_canvas(
    K: np.ndarray,
    image_shape: tuple[int, int],
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    intrinsics = np.asarray(K, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("camera intrinsics K must be a finite 3x3 matrix")
    checks = {
        "positive_focal_lengths": bool(
            intrinsics[0, 0] > 0.0 and intrinsics[1, 1] > 0.0
        ),
        "canonical_homogeneous_row": bool(
            np.allclose(
                intrinsics[2],
                np.array([0.0, 0.0, 1.0]),
                rtol=1e-6,
                atol=1e-7,
            )
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"camera intrinsics K failed validation: {failed}")
    actual = tuple(int(value) for value in image_shape)
    expected = tuple(int(value) for value in expected_shape)
    if actual != expected:
        raise ValueError(
            "camera canvas mismatch: image shape "
            f"{actual} does not equal required calibrated canvas {expected}"
        )
    return {
        "valid": True,
        "contract": "exact_expected_image_size",
        "actual_image_size": [actual[0], actual[1]],
        "expected_image_size": [expected[0], expected[1]],
        "intrinsics_checks": checks,
    }


def _load_semantic_regions(
    path: Optional[Path], vertex_count: int
) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    regions = payload.get("regions", {})
    if not isinstance(regions, dict):
        raise ValueError("semantic regions JSON must contain an object at 'regions'")

    result: dict[str, np.ndarray] = {}
    for name, raw_indices in regions.items():
        if not isinstance(raw_indices, list):
            continue
        indices = [
            int(value)
            for value in raw_indices
            if not isinstance(value, bool)
            and isinstance(value, (int, np.integer))
            and 0 <= int(value) < vertex_count
        ]
        if indices:
            result[str(name)] = np.unique(np.asarray(indices, dtype=np.int32))
    return result


def _in_image_mask(
    pixel_xy: np.ndarray, front: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    height, width = shape
    return (
        front
        & (pixel_xy[:, 0] >= 0.0)
        & (pixel_xy[:, 0] < float(width))
        & (pixel_xy[:, 1] >= 0.0)
        & (pixel_xy[:, 1] < float(height))
    )


def _camera_vertices_to_perspective_clip(
    vertices,
    K,
    R,
    t,
    image_shape: tuple[int, int],
):
    """Map OpenCV camera coordinates to nvdiffrast clip space with depth."""
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    if image_height < 2 or image_width < 2:
        raise ValueError("image dimensions must both be at least 2")

    camera = vertices @ R.transpose(0, 1) + t.reshape(1, 3)
    homogeneous = camera @ K.transpose(0, 1)
    depth = camera[:, 2]
    positive = depth > float(DEFAULT_POSITIVE_DEPTH_EPSILON)
    if not bool(positive.any().item()):
        raise ValueError("CUDA rasterization requires positive camera depth")

    positive_depth = depth[positive]
    minimum_depth = positive_depth.amin()
    maximum_depth = positive_depth.amax()
    tiny = vertices.new_tensor(np.finfo(np.float32).tiny)
    near = (minimum_depth * 0.5).clamp_min(tiny)
    far = maximum_depth * 1.5
    minimum_far = near + (near.abs() * 1e-3).clamp_min(vertices.new_tensor(1e-6))
    far = far.maximum(minimum_far)

    x_clip = 2.0 * homogeneous[:, 0] / float(image_width - 1) - depth
    y_clip = depth - 2.0 * homogeneous[:, 1] / float(image_height - 1)
    denominator = far - near
    coefficient_a = (far + near) / denominator
    coefficient_b = (-2.0 * far * near) / denominator
    z_clip = coefficient_a * depth + coefficient_b
    clip = vertices.new_empty((vertices.shape[0], 4))
    clip[:, 0] = x_clip
    clip[:, 1] = y_clip
    clip[:, 2] = z_clip
    clip[:, 3] = depth
    return clip, positive, near, far


class _CudaVisibilityBackend:
    name = "cuda_nvdiffrast"
    method = "nvdiffrast_triangle_id"

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        import nvdiffrast.torch as dr
        import torch

        from src.geometry.differentiable_silhouette import create_cuda_raster_context

        self._dr = dr
        self._torch = torch
        self._context = create_cuda_raster_context("cuda")
        self._vertices = torch.as_tensor(
            vertices, dtype=torch.float32, device="cuda"
        ).contiguous()
        self._faces = torch.as_tensor(
            faces, dtype=torch.int32, device="cuda"
        ).contiguous()
        self._faces_long = self._faces.to(dtype=torch.int64)
        self._face_count = int(len(faces))

    def render(
        self,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            K_tensor = torch.as_tensor(K, dtype=torch.float32, device="cuda")
            R_tensor = torch.as_tensor(R, dtype=torch.float32, device="cuda")
            t_tensor = torch.as_tensor(t, dtype=torch.float32, device="cuda")
            clip, positive_depth, _near, _far = _camera_vertices_to_perspective_clip(
                self._vertices, K_tensor, R_tensor, t_tensor, image_shape
            )
            valid_faces = torch.all(positive_depth[self._faces_long], dim=1)
            original_face_indices = torch.nonzero(
                valid_faces, as_tuple=False
            ).reshape(-1)
            if original_face_indices.numel() == 0:
                return np.zeros(image_shape, dtype=np.int32)
            raster_faces = self._faces[original_face_indices].contiguous()
            raster, _raster_derivative = self._dr.rasterize(
                self._context,
                clip.unsqueeze(0).contiguous(),
                raster_faces,
                resolution=[int(image_shape[0]), int(image_shape[1])],
            )
            # nvdiffrast is bottom-up; report images and projected pixels are top-down.
            local_triangle_ids = (
                torch.flip(raster[0, :, :, 3], dims=[0])
                .to(dtype=torch.int64)
                .cpu()
                .numpy()
            )
            original_face_indices_numpy = original_face_indices.cpu().numpy()

        triangle_ids = np.zeros(local_triangle_ids.shape, dtype=np.int32)
        foreground = local_triangle_ids > 0
        if np.any(foreground):
            local_zero_based = local_triangle_ids[foreground] - 1
            valid_local = (local_zero_based >= 0) & (
                local_zero_based < len(original_face_indices_numpy)
            )
            mapped = np.zeros(local_zero_based.shape, dtype=np.int32)
            mapped[valid_local] = (
                original_face_indices_numpy[local_zero_based[valid_local]] + 1
            )
            triangle_ids[foreground] = mapped
        return triangle_ids


def _create_cuda_visibility_backend(
    vertices: np.ndarray, faces: np.ndarray
) -> Optional[_CudaVisibilityBackend]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return _CudaVisibilityBackend(vertices, faces)
    except (ImportError, RuntimeError):
        return None


def _screen_barycentric(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> Optional[np.ndarray]:
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (
        a[1] - c[1]
    )
    if not np.isfinite(denominator) or abs(float(denominator)) <= 1e-12:
        return None
    weight_a = (
        (b[1] - c[1]) * (points[:, 0] - c[0])
        + (c[0] - b[0]) * (points[:, 1] - c[1])
    ) / denominator
    weight_b = (
        (c[1] - a[1]) * (points[:, 0] - c[0])
        + (a[0] - c[0]) * (points[:, 1] - c[1])
    ) / denominator
    return np.column_stack((weight_a, weight_b, 1.0 - weight_a - weight_b))


def _cpu_perspective_triangle_ids(
    pixel_xy: np.ndarray,
    vertex_depth: np.ndarray,
    front_facing: np.ndarray,
    faces: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize a small mesh with perspective-correct 1/z interpolation."""
    height, width = image_shape
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    triangle_ids = np.zeros((height, width), dtype=np.int32)
    projected = np.asarray(pixel_xy, dtype=np.float32)
    depth = np.asarray(vertex_depth, dtype=np.float32)
    front = np.asarray(front_facing, dtype=bool)
    triangles = np.asarray(faces, dtype=np.int32)

    for face_index, triangle in enumerate(triangles):
        if not np.all(front[triangle]):
            continue
        points = projected[triangle]
        triangle_depth = depth[triangle]
        if not np.isfinite(points).all() or np.any(triangle_depth <= 0.0):
            continue
        x_min = max(0, int(np.floor(np.min(points[:, 0]))))
        x_max = min(width - 1, int(np.ceil(np.max(points[:, 0]))))
        y_min = max(0, int(np.floor(np.min(points[:, 1]))))
        y_max = min(height - 1, int(np.ceil(np.max(points[:, 1]))))
        if x_min > x_max or y_min > y_max:
            continue

        xs = np.arange(x_min, x_max + 1, dtype=np.float32) + 0.5
        ys = np.arange(y_min, y_max + 1, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        samples = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        barycentric = _screen_barycentric(
            samples, points[0], points[1], points[2]
        )
        if barycentric is None:
            continue
        inside = np.all(barycentric >= -1e-6, axis=1)
        if not np.any(inside):
            continue
        selected_samples = samples[inside]
        inverse_depth = barycentric[inside] @ (1.0 / triangle_depth)
        valid_depth = np.isfinite(inverse_depth) & (inverse_depth > 0.0)
        if not np.any(valid_depth):
            continue
        selected_samples = selected_samples[valid_depth]
        perspective_depth = (1.0 / inverse_depth[valid_depth]).astype(np.float32)
        pixel_x = np.floor(selected_samples[:, 0]).astype(np.int32)
        pixel_y = np.floor(selected_samples[:, 1]).astype(np.int32)
        current = depth_buffer[pixel_y, pixel_x]
        winners = perspective_depth < current
        if np.any(winners):
            win_x = pixel_x[winners]
            win_y = pixel_y[winners]
            depth_buffer[win_y, win_x] = perspective_depth[winners]
            triangle_ids[win_y, win_x] = face_index + 1
    return triangle_ids


def _visible_vertices_from_triangle_ids(
    triangle_ids: np.ndarray,
    pixel_xy: np.ndarray,
    candidate_mask: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Confirm each vertex against winning adjacent faces near its pixel."""
    face_array = np.asarray(faces, dtype=np.int32)
    candidates = np.flatnonzero(candidate_mask)
    visible = np.zeros(len(pixel_xy), dtype=bool)
    if not candidates.size or not len(face_array):
        return visible

    height, width = triangle_ids.shape
    base = np.floor(np.asarray(pixel_xy)[candidates]).astype(np.int32)
    offsets = np.array(
        [
            [-1, -1],
            [0, -1],
            [1, -1],
            [-1, 0],
            [0, 0],
            [1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
        ],
        dtype=np.int32,
    )
    samples = base[:, None, :] + offsets[None, :, :]
    samples[:, :, 0] = np.clip(samples[:, :, 0], 0, width - 1)
    samples[:, :, 1] = np.clip(samples[:, :, 1], 0, height - 1)
    winner_ids = triangle_ids[samples[:, :, 1], samples[:, :, 0]]
    valid_winner = (winner_ids > 0) & (winner_ids <= len(face_array))
    safe_face_indices = np.clip(winner_ids - 1, 0, max(0, len(face_array) - 1))
    winner_vertices = face_array[safe_face_indices]
    adjacent = np.any(
        winner_vertices == candidates[:, None, None], axis=2
    ) & valid_winner
    visible[candidates] = np.any(adjacent, axis=1)
    return visible


def _draw_geometry_overlay(
    image: np.ndarray,
    geometry_mask: np.ndarray,
    pixel_xy: np.ndarray,
    visible: np.ndarray,
    semantic_regions: dict[str, np.ndarray],
) -> np.ndarray:
    overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask_u8 = np.where(geometry_mask, 255, 0).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1, cv2.LINE_AA)

    visible_indices = np.flatnonzero(visible)
    if visible_indices.size:
        stride = max(1, int(np.ceil(visible_indices.size / _MAX_VISIBLE_POINTS)))
        for index in visible_indices[::stride]:
            point = tuple(np.rint(pixel_xy[index]).astype(int))
            cv2.circle(overlay, point, 1, (120, 255, 120), -1, cv2.LINE_AA)

    for name, indices in semantic_regions.items():
        anchor_indices = indices[visible[indices]]
        if not anchor_indices.size:
            continue
        point = tuple(np.rint(np.mean(pixel_xy[anchor_indices], axis=0)).astype(int))
        cv2.circle(overlay, point, 3, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            name,
            (point[0] + 4, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _validate_warp_output(
    sampled: np.ndarray, expected_shape: tuple[int, int]
) -> np.ndarray:
    array = np.asarray(sampled)
    if array.shape != expected_shape:
        raise ValueError(
            f"warp output must have shape {expected_shape}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("warp output must contain only finite numeric coordinates")
    return np.asarray(array, dtype=np.float32)


def _draw_warp_arrows(
    overlay: np.ndarray,
    projected: np.ndarray,
    sampled: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    arrow_image = overlay.copy()
    indices = np.flatnonzero(valid)
    if indices.size:
        stride = max(1, int(np.ceil(indices.size / _MAX_WARP_ARROWS)))
        for index in indices[::stride][:_MAX_WARP_ARROWS]:
            start = tuple(np.rint(projected[index]).astype(int))
            end = tuple(np.rint(sampled[index]).astype(int))
            cv2.arrowedLine(
                arrow_image,
                start,
                end,
                (50, 80, 255),
                1,
                cv2.LINE_AA,
                tipLength=0.25,
            )
    return arrow_image


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"failed to write report image: {path}")


def _write_html(output_dir: Path, metrics: dict[str, Any]) -> None:
    cards = []
    for view_name, view in metrics["views"].items():
        legacy = view["legacy_sampling_displacement"]
        arrow_link = ""
        if view.get("legacy_warp_arrows_path"):
            arrow_path = html.escape(view["legacy_warp_arrows_path"])
            arrow_link = f'<p><a href="{arrow_path}">Legacy warp arrows</a></p>'
        cards.append(
            f"""
            <article>
              <h2>{html.escape(view_name)}</h2>
              <img src="{html.escape(view['overlay_path'])}" alt="{html.escape(view_name)} unwarped projection">
              <dl>
                <dt>Final sampling</dt><dd>{FINAL_SAMPLING_MODE}</dd>
                <dt>Visibility</dt><dd>{html.escape(view['visibility_backend'])} / {html.escape(view['visibility_method'])}</dd>
                <dt>Positive depth</dt><dd>{view['positive_depth_ratio']:.4f}</dd>
                <dt>In image</dt><dd>{view['in_image_ratio']:.4f}</dd>
                <dt>Raster visible</dt><dd>{view['depth_visible_ratio']:.4f}</dd>
                <dt>Legacy warp</dt><dd>{str(view['legacy_warp_present']).lower()}</dd>
                <dt>Legacy mean displacement</dt><dd>{html.escape(str(legacy['mean_displacement_px']))}</dd>
              </dl>
              {arrow_link}
            </article>
            """
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Projective Texture Truth Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f4f6f5; color: #18211f; }}
    header, main {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    header {{ border-bottom: 1px solid #c8d0cd; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    .mode {{ color: #17643b; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    article {{ background: #fff; border: 1px solid #ccd3d0; border-radius: 8px; overflow: hidden; }}
    article h2, article dl, article p {{ margin: 14px; }}
    img {{ display: block; width: 100%; height: auto; }}
    dl {{ display: grid; grid-template-columns: 1fr auto; gap: 6px 12px; font-size: 13px; }}
    dt, dd {{ margin: 0; }}
  </style>
</head>
<body>
  <header>
    <h1>Projective Texture Truth Report</h1>
    <div class="mode">Final sampling mode: {FINAL_SAMPLING_MODE}</div>
    <p>Legacy warps are measured for diagnosis only. They do not move final sampling coordinates.</p>
    <a href="truth_metrics.json">truth_metrics.json</a>
  </header>
  <main><div class="grid">{''.join(cards)}</div></main>
</body>
</html>
"""
    _safe_output_path(output_dir, "index.html").write_text(page, encoding="utf-8")


def _verify_provenance(
    *,
    obj_path: Path,
    cameras_path: Path,
    semantic_path: Optional[Path],
    images: dict[str, np.ndarray],
    mesh_hash: str,
    cameras_hash: str,
    semantic_hash: Optional[str],
    image_hashes: dict[str, str],
) -> None:
    changed = []
    if _sha256(obj_path) != mesh_hash:
        changed.append("face_mesh.obj")
    if _sha256(cameras_path) != cameras_hash:
        changed.append("cameras.json")
    if semantic_path is not None and _sha256(semantic_path) != semantic_hash:
        changed.append("semantic regions JSON")
    for view_name, image in images.items():
        if _image_sha256(image) != image_hashes[view_name]:
            changed.append(f"image:{view_name}")
    if changed:
        raise RuntimeError(
            "projective texture report provenance changed during generation: "
            + ", ".join(changed)
        )


def _unique_sibling(output_dir: Path, kind: str) -> Path:
    return output_dir.parent / f".{output_dir.name}.{kind}-{uuid.uuid4().hex}"


def _publish_report_directory(staging_dir: Path, output_dir: Path) -> None:
    """Atomically install a complete report directory, restoring on failure."""
    backup_dir = _unique_sibling(output_dir, "backup")
    rollback_dir = _unique_sibling(output_dir, "rollback")
    had_previous = output_dir.exists()
    previous_moved = False
    try:
        if had_previous:
            os.replace(str(output_dir), str(backup_dir))
            previous_moved = True
        os.replace(str(staging_dir), str(output_dir))
    except Exception:
        if previous_moved and backup_dir.exists():
            if output_dir.exists():
                os.replace(str(output_dir), str(rollback_dir))
            os.replace(str(backup_dir), str(output_dir))
            if rollback_dir.exists():
                shutil.rmtree(rollback_dir)
        raise

    if previous_moved and backup_dir.exists():
        try:
            shutil.rmtree(backup_dir)
        except Exception:
            # Backup cleanup is part of the transaction. Restore the previous
            # report rather than claiming success with ambiguous directories.
            os.replace(str(output_dir), str(rollback_dir))
            os.replace(str(backup_dir), str(output_dir))
            shutil.rmtree(rollback_dir)
            raise


def publish_report_directory_transactionally(
    staging_dir: Path, output_dir: Path
) -> None:
    """Publish a complete diagnostic directory while preserving any old report."""
    _publish_report_directory(Path(staging_dir), Path(output_dir))


def write_projective_texture_truth_report(
    *,
    mesh_dir: Path,
    images: dict[str, np.ndarray],
    output_dir: Path,
    expected_image_size: Optional[Any] = None,
    expected_image_sizes: Optional[dict[str, Any]] = None,
    sampling_warps: Optional[dict] = None,
    semantic_regions_path: Optional[Path] = None,
) -> dict:
    """Write strict projection overlays and read-only legacy-warp diagnostics."""
    if not images:
        raise ValueError("images must contain at least one left/front/right view")
    _validate_view_names(images, "images")
    expected_sizes = _resolve_expected_image_sizes(
        images, expected_image_size, expected_image_sizes
    )
    warps = sampling_warps or {}
    _validate_view_names(warps, "sampling_warps")
    orphan_warps = sorted(set(warps) - set(images))
    if orphan_warps:
        raise ValueError(f"sampling_warps has no matching image view: {', '.join(orphan_warps)}")

    validated_images = {
        view_name: _validate_image(image, view_name)
        for view_name, image in images.items()
    }
    mesh_dir = Path(mesh_dir)
    obj_path = mesh_dir / "face_mesh.obj"
    cameras_path = mesh_dir / "cameras.json"
    semantic_path = Path(semantic_regions_path) if semantic_regions_path else None
    mesh_hash_before = _sha256(obj_path)
    cameras_hash_before = _sha256(cameras_path)
    semantic_hash_before = _sha256(semantic_path) if semantic_path else None
    image_hashes_before = {
        view_name: _image_sha256(image)
        for view_name, image in validated_images.items()
    }

    vertices, faces, _uv_vertices, _uv_faces = load_mesh_obj(obj_path)
    cameras = load_cameras(cameras_path)
    _validate_view_names(cameras, "cameras.json")
    missing_cameras = sorted(set(validated_images) - set(cameras))
    if missing_cameras:
        raise ValueError(f"missing cameras for image views: {', '.join(missing_cameras)}")
    canvas_validation = {
        view_name: _validate_camera_canvas(
            cameras[view_name]["K"], image.shape[:2], expected_sizes[view_name]
        )
        for view_name, image in validated_images.items()
    }
    semantic_regions = _load_semantic_regions(semantic_path, len(vertices))

    cuda_backend = _create_cuda_visibility_backend(vertices, faces)
    if cuda_backend is None and len(faces) > CPU_FALLBACK_MAX_FACES:
        raise RuntimeError(
            "CUDA/nvdiffrast visibility is required for large mesh reports "
            f"({len(faces)} faces > CPU fallback limit {CPU_FALLBACK_MAX_FACES}); "
            "refusing slow Python z-buffer fallback"
        )
    visibility_backend = (
        cuda_backend.name if cuda_backend is not None else "cpu_fallback"
    )
    visibility_method = (
        cuda_backend.method
        if cuda_backend is not None
        else "perspective_face_id_vertex_adjacency"
    )

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_root.parent)
        )
    ).resolve()
    try:
        view_metrics: dict[str, dict[str, Any]] = {}
        for view_name in sorted(validated_images):
            image = validated_images[view_name]
            height, width = image.shape[:2]
            camera = cameras[view_name]
            projection = project_points_strict(
                vertices, camera["K"], camera["R"], camera["t"]
            )
            in_image = _in_image_mask(
                projection.pixel_xy, projection.front_facing, (height, width)
            )
            if cuda_backend is not None:
                triangle_ids = cuda_backend.render(
                    camera["K"], camera["R"], camera["t"], (height, width)
                )
            else:
                triangle_ids = _cpu_perspective_triangle_ids(
                    projection.pixel_xy,
                    projection.depth,
                    projection.front_facing,
                    faces,
                    (height, width),
                )
            if triangle_ids.shape != (height, width):
                raise ValueError(
                    "visibility backend returned triangle-id canvas with "
                    f"shape {triangle_ids.shape}, expected {(height, width)}"
                )
            if not np.issubdtype(triangle_ids.dtype, np.integer):
                raise ValueError("visibility backend triangle-id canvas must be integer")
            geometry_mask = triangle_ids > 0
            diagnostic_visible = _visible_vertices_from_triangle_ids(
                triangle_ids,
                projection.pixel_xy,
                in_image,
                faces,
            )
            visible_vertex_count = int(np.count_nonzero(diagnostic_visible))
            if visible_vertex_count < MIN_DIAGNOSTIC_VISIBLE_VERTICES:
                raise RuntimeError(
                    f"{view_name} projective diagnostic is not interpretable: "
                    f"only {visible_vertex_count} raster-visible vertices "
                    f"(minimum {MIN_DIAGNOSTIC_VISIBLE_VERTICES})"
                )
            overlay = _draw_geometry_overlay(
                image,
                geometry_mask,
                projection.pixel_xy,
                diagnostic_visible,
                semantic_regions,
            )
            overlay_name = f"{view_name}_unwarped_overlay.jpg"
            _write_image(_safe_output_path(staging_dir, overlay_name), overlay)

            warp = warps.get(view_name)
            if warp is None:
                legacy_displacement = compare_sampling_coordinates(
                    projection.pixel_xy,
                    projection.pixel_xy,
                    valid_mask=np.zeros(len(vertices), dtype=bool),
                )
                arrows_name = None
            else:
                visible_indices = np.flatnonzero(diagnostic_visible)
                warp_input = projection.pixel_xy[visible_indices].copy()
                warped_visible = _validate_warp_output(
                    warp.apply(warp_input, image.shape[:2]), warp_input.shape
                )
                sampled = projection.pixel_xy.copy()
                sampled[visible_indices] = warped_visible
                legacy_displacement = compare_sampling_coordinates(
                    projection.pixel_xy,
                    sampled,
                    valid_mask=diagnostic_visible,
                )
                arrows_name = f"{view_name}_legacy_warp_arrows.jpg"
                arrow_image = _draw_warp_arrows(
                    overlay,
                    projection.pixel_xy,
                    sampled,
                    diagnostic_visible,
                )
                _write_image(
                    _safe_output_path(staging_dir, arrows_name), arrow_image
                )

            vertex_count = int(len(vertices))
            denominator = float(vertex_count) if vertex_count else 1.0
            view_metrics[view_name] = {
                "image_size": [int(height), int(width)],
                "vertex_count": vertex_count,
                "visible_vertex_count": visible_vertex_count,
                "diagnostic_valid": True,
                "positive_depth_ratio": float(
                    np.count_nonzero(projection.front_facing) / denominator
                ),
                "in_image_ratio": float(np.count_nonzero(in_image) / denominator),
                "depth_visible_ratio": float(
                    np.count_nonzero(diagnostic_visible) / denominator
                ),
                "visibility_backend": visibility_backend,
                "visibility_method": visibility_method,
                "camera_canvas_validation": canvas_validation[view_name],
                "legacy_warp_present": warp is not None,
                "legacy_sampling_displacement": legacy_displacement,
                "overlay_path": overlay_name,
                "legacy_warp_arrows_path": arrows_name,
            }

        _verify_provenance(
            obj_path=obj_path,
            cameras_path=cameras_path,
            semantic_path=semantic_path,
            images=validated_images,
            mesh_hash=mesh_hash_before,
            cameras_hash=cameras_hash_before,
            semantic_hash=semantic_hash_before,
            image_hashes=image_hashes_before,
        )
        metrics: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "final_sampling_mode": FINAL_SAMPLING_MODE,
            "geometry_changed": False,
            "camera_changed": False,
            "face_mesh_sha256": mesh_hash_before,
            "cameras_sha256": cameras_hash_before,
            "face_mesh_sha256_before": mesh_hash_before,
            "cameras_sha256_before": cameras_hash_before,
            "semantic_regions_sha256": semantic_hash_before,
            "image_sha256": image_hashes_before,
            "visibility_backend": visibility_backend,
            "visibility_method": visibility_method,
            "views": view_metrics,
        }
        metrics_path = _safe_output_path(staging_dir, "truth_metrics.json")
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
        _write_html(staging_dir, metrics)
        _verify_provenance(
            obj_path=obj_path,
            cameras_path=cameras_path,
            semantic_path=semantic_path,
            images=validated_images,
            mesh_hash=mesh_hash_before,
            cameras_hash=cameras_hash_before,
            semantic_hash=semantic_hash_before,
            image_hashes=image_hashes_before,
        )
        publish_report_directory_transactionally(staging_dir, output_root)
        return metrics
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
