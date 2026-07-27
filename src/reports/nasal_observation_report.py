"""Offline artifacts for confidence-weighted nasal observations."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src.geometry.nasal_observations import (
    NasalObservationBundle,
    NasalViewObservation,
)
from src.geometry.observation_coordinates import (
    ObservationCoordinates,
    fundamental_matrix_work,
    normalize_image_to_work,
    normalize_mask_to_work,
    original_points_to_work,
)


REPORT_VIEWS = ("front", "subject-left", "subject-right")
CAMERA_VIEW_BY_SEMANTIC = {
    "front": "front",
    "subject-left": "left",
    "subject-right": "right",
}
ANCHOR_COLORS = {
    "subject-left": (64, 210, 255),
    "subject-right": (224, 120, 255),
}
OBSERVATION_COLOR = (255, 225, 60)
VARIANT_COLORS = {
    "eroded": (255, 175, 70),
    "base": (75, 230, 125),
    "dilated": (220, 105, 245),
}
ANCHOR_NUMBER = {
    "upper_tip": "1",
    "tip_apex": "2",
    "lower_tip": "3",
    "alar_transition": "4",
}


def read_image_file(
    path: str | Path,
    flags: int = cv2.IMREAD_COLOR,
    *,
    description: str = "image",
) -> np.ndarray:
    """Decode an image without relying on OpenCV's Windows path handling."""
    target = Path(path)
    try:
        encoded = np.fromfile(str(target), dtype=np.uint8)
    except OSError as exc:
        raise RuntimeError(
            f"failed to read {description} bytes from {target}: {exc}"
        ) from exc
    if encoded.size == 0:
        raise RuntimeError(f"{description} file is empty: {target}")
    try:
        image = cv2.imdecode(encoded, int(flags))
    except cv2.error as exc:
        raise RuntimeError(
            f"failed to decode {description} at {target}: {exc}"
        ) from exc
    if image is None:
        raise RuntimeError(f"failed to decode {description}: {target}")
    return image


def write_image_file(
    path: str | Path,
    image: np.ndarray,
    *,
    description: str = "image",
) -> None:
    """Encode an image before writing it to a Windows-compatible path."""
    target = Path(path)
    suffix = target.suffix.lower()
    if not suffix:
        raise ValueError(
            f"cannot encode {description} without a file extension: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        success, encoded = cv2.imencode(suffix, np.asarray(image))
    except cv2.error as exc:
        raise RuntimeError(
            f"failed to encode {description} for {target}: {exc}"
        ) from exc
    if not success:
        raise RuntimeError(f"failed to encode {description}: {target}")
    try:
        encoded.tofile(str(target))
    except OSError as exc:
        raise RuntimeError(
            f"failed to write {description} to {target}: {exc}"
        ) from exc


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _field_prefix(semantic_view: str) -> str:
    return semantic_view.replace("-", "_")


def _observation_summary(
    observation: NasalViewObservation,
) -> dict[str, Any]:
    confidence = np.asarray(observation.confidence, dtype=np.float64)
    distance = np.asarray(observation.distance_field, dtype=np.float64)
    boundary = np.asarray(observation.boundary, dtype=bool)
    boundary_confidence = confidence[boundary]
    finite_distance = distance[np.isfinite(distance)]
    return {
        "boundary_pixel_count": int(np.count_nonzero(boundary)),
        "boundary_points_by_name": {
            name: int(len(np.asarray(points).reshape(-1, 2)))
            for name, points in observation.boundaries_work.items()
        },
        "anchor_count": int(len(observation.anchors_work)),
        "confidence": {
            "minimum": float(np.min(confidence)),
            "maximum": float(np.max(confidence)),
            "mean": float(np.mean(confidence)),
            "nonzero_fraction": float(np.mean(confidence > 0.0)),
            "boundary_mean": (
                None
                if not len(boundary_confidence)
                else float(np.mean(boundary_confidence))
            ),
            "boundary_p10": (
                None
                if not len(boundary_confidence)
                else float(np.percentile(boundary_confidence, 10.0))
            ),
            "boundary_p90": (
                None
                if not len(boundary_confidence)
                else float(np.percentile(boundary_confidence, 90.0))
            ),
        },
        "distance_work_px": {
            "minimum": float(np.min(finite_distance)),
            "maximum": float(np.max(finite_distance)),
            "mean": float(np.mean(finite_distance)),
            "p90": float(np.percentile(finite_distance, 90.0)),
        },
    }


def _serialize_observation(
    observation: NasalViewObservation,
    fields: dict[str, np.ndarray],
) -> dict[str, Any]:
    prefix = _field_prefix(observation.semantic_view)

    def add_field(suffix: str, value: Any) -> str:
        key = f"{prefix}__{suffix}"
        fields[key] = np.asarray(value)
        return key

    named_distance_fields = {
        name: add_field(
            f"distance__{name.replace('-', '_')}",
            field,
        )
        for name, field in observation.distance_fields.items()
    }
    variant_fields = {
        name: add_field(
            f"variant_boundary__{name.replace('-', '_')}",
            field,
        )
        for name, field in observation.variant_boundaries.items()
    }
    camera = dict(_json_value(observation.camera_metadata))
    camera.setdefault("camera_name", observation.camera.name)
    camera.setdefault("camera_view", observation.camera.view)
    camera.setdefault("subject_relative_view", observation.semantic_view)
    return {
        "semantic_view": observation.semantic_view,
        "camera": camera,
        "original_size_wh": list(observation.original_size),
        "mask_canvas_shape_hw": list(observation.mask_canvas_shape),
        "work_size_wh": list(observation.work_size),
        "roi_work_xyxy": list(observation.roi_work_xyxy),
        "coordinate_metadata": _json_value(
            observation.coordinate_metadata
        ),
        "boundaries_work": _json_value(observation.boundaries_work),
        "anchors_work": _json_value(observation.anchors_work),
        "variant_boundaries_work": _json_value(
            observation.variant_boundaries_work
        ),
        "fields": {
            "boundary": add_field("boundary", observation.boundary),
            "distance": add_field(
                "distance_aggregate",
                observation.distance_field,
            ),
            "distance_by_boundary": named_distance_fields,
            "confidence": add_field(
                "confidence",
                observation.confidence,
            ),
            "variant_boundaries": variant_fields,
        },
        "summary": _observation_summary(observation),
    }


def write_nasal_observation_data(
    bundle: NasalObservationBundle,
    json_path: str | Path,
    fields_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write compact JSON metadata and compressed full-resolution work fields."""
    target_json = Path(json_path)
    requested_fields = Path(fields_path)
    target_fields = (
        requested_fields
        if requested_fields.suffix.lower() == ".npz"
        else Path(f"{requested_fields}.npz")
    )
    target_json.parent.mkdir(parents=True, exist_ok=True)
    target_fields.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, np.ndarray] = {}
    views = {
        semantic_view: _serialize_observation(observation, fields)
        for semantic_view, observation in bundle.by_view.items()
    }
    relative_fields = Path(
        os.path.relpath(target_fields, target_json.parent)
    ).as_posix()
    payload = {
        "schema_version": 1,
        "audit_only": True,
        "geometry_observations_only": True,
        "texture_scoring_included": False,
        "fields_npz": relative_fields,
        "camera_name_by_view": bundle.camera_name_by_view,
        "metadata": _json_value(metadata or {}),
        "views": views,
    }
    with target_fields.open("wb") as handle:
        np.savez_compressed(handle, **fields)
    target_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _work_bgr(
    image: np.ndarray,
    coordinates: ObservationCoordinates,
) -> np.ndarray:
    work = normalize_image_to_work(image, coordinates)
    if work.ndim == 2:
        return cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    if work.shape[2] == 4:
        return cv2.cvtColor(work, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(work, cv2.COLOR_RGB2BGR)


def _polyline(
    image: np.ndarray,
    points: Any,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(values):
        return
    cv2.polylines(
        image,
        [np.rint(values).astype(np.int32).reshape(-1, 1, 2)],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _point(
    image: np.ndarray,
    point: Any,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x, y = np.rint(np.asarray(point, dtype=np.float64)).astype(np.int32)
    cv2.circle(image, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
    if label:
        cv2.putText(
            image,
            label,
            (int(x) + 5, int(y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def _line_segment(
    line: np.ndarray,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    a, b, c = np.asarray(line, dtype=np.float64).reshape(3)
    candidates: list[tuple[float, float]] = []
    if abs(b) > 1e-12:
        for x in (0.0, float(width - 1)):
            y = -(a * x + c) / b
            if 0.0 <= y < height:
                candidates.append((x, y))
    if abs(a) > 1e-12:
        for y in (0.0, float(height - 1)):
            x = -(b * y + c) / a
            if 0.0 <= x < width:
                candidates.append((x, y))
    unique = []
    for candidate in candidates:
        if not any(np.linalg.norm(np.subtract(candidate, item)) < 0.5 for item in unique):
            unique.append(candidate)
    if len(unique) < 2:
        return None
    return (
        tuple(np.rint(unique[0]).astype(int)),
        tuple(np.rint(unique[1]).astype(int)),
    )


def _projected_priors_work(
    priors_by_subject_side: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ],
    camera_view: str,
    coordinates: ObservationCoordinates,
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for subject_side, priors_by_view in priors_by_subject_side.items():
        priors = priors_by_view.get(camera_view, {})
        names = list(priors)
        if not names:
            continue
        values = original_points_to_work(
            [priors[name] for name in names],
            coordinates,
        )
        result[subject_side] = {
            name: values[index]
            for index, name in enumerate(names)
        }
    return result


def _draw_priors(
    image: np.ndarray,
    priors_work: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    for subject_side, priors in priors_work.items():
        color = ANCHOR_COLORS.get(subject_side, (235, 200, 90))
        ordered = [
            priors[name]
            for name in (
                "upper_tip",
                "tip_apex",
                "lower_tip",
                "alar_transition",
            )
            if name in priors
        ]
        if ordered:
            _polyline(image, ordered, color, thickness=1)
        for point in priors.values():
            _point(image, point, color, "")


def _save_image(path: Path, image: np.ndarray) -> None:
    write_image_file(path, image, description="report image")


def _observation_by_camera_view(
    bundle: NasalObservationBundle,
) -> dict[str, NasalViewObservation]:
    return {
        observation.camera.view: observation
        for observation in bundle.by_view.values()
    }


def _write_overlay(
    path: Path,
    camera_view: str,
    observation: NasalViewObservation,
    image_rgb: np.ndarray,
    coordinates_by_view: Mapping[str, ObservationCoordinates],
    side_face_masks: Mapping[str, np.ndarray],
    priors_by_subject_side: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ],
    centerline_x_original: float,
    front_observation: NasalViewObservation,
) -> None:
    coordinates = coordinates_by_view[camera_view]
    overlay = _work_bgr(image_rgb, coordinates)
    priors_work = _projected_priors_work(
        priors_by_subject_side,
        camera_view,
        coordinates,
    )
    if camera_view == "front":
        _draw_priors(overlay, priors_work)
        for name, points in observation.boundaries_work.items():
            color = (
                ANCHOR_COLORS["subject-left"]
                if "subject-left" in name
                else ANCHOR_COLORS["subject-right"]
            )
            _polyline(overlay, points, color, thickness=2)
        centerline = original_points_to_work(
            [(centerline_x_original, 0.0)],
            coordinates,
        )[0, 0]
        cv2.line(
            overlay,
            (int(round(centerline)), 0),
            (int(round(centerline)), overlay.shape[0] - 1),
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    else:
        face_mask = normalize_mask_to_work(
            side_face_masks[camera_view],
            coordinates,
            name=f"{camera_view} face mask",
        )
        contours, _hierarchy = cv2.findContours(
            np.asarray(face_mask, dtype=np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if contours:
            cv2.drawContours(
                overlay,
                [max(contours, key=cv2.contourArea)],
                -1,
                (90, 220, 125),
                1,
                cv2.LINE_AA,
            )
        x0, y0, x1, y1 = observation.roi_work_xyxy
        cv2.rectangle(
            overlay,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            (75, 210, 245),
            1,
            cv2.LINE_AA,
        )
        front_coordinates = coordinates_by_view["front"]
        fundamental = fundamental_matrix_work(
            front_observation.camera,
            observation.camera,
            front_coordinates,
            coordinates,
        )
        front_priors = _projected_priors_work(
            priors_by_subject_side,
            "front",
            front_coordinates,
        )
        for subject_side, priors in front_priors.items():
            color = ANCHOR_COLORS.get(subject_side, (235, 200, 90))
            for point in priors.values():
                line = fundamental @ np.append(point, 1.0)
                segment = _line_segment(
                    line,
                    overlay.shape[1],
                    overlay.shape[0],
                )
                if segment is not None:
                    line_color = tuple(
                        int(round(channel * 0.68))
                        for channel in color
                    )
                    cv2.line(
                        overlay,
                        segment[0],
                        segment[1],
                        line_color,
                        1,
                        cv2.LINE_AA,
                    )
        _draw_priors(overlay, priors_work)
        for points in observation.boundaries_work.values():
            _polyline(
                overlay,
                points,
                OBSERVATION_COLOR,
                thickness=3,
            )
        for name, point in observation.anchors_work.items():
            _point(
                overlay,
                point,
                OBSERVATION_COLOR,
                ANCHOR_NUMBER.get(name, ""),
            )
    _save_image(path, overlay)


def _write_confidence(
    path: Path,
    observation: NasalViewObservation,
    image_rgb: np.ndarray,
    coordinates: ObservationCoordinates,
) -> None:
    background = _work_bgr(image_rgb, coordinates)
    values = np.clip(
        np.asarray(observation.confidence, dtype=np.float32),
        0.0,
        1.0,
    )
    heatmap = cv2.applyColorMap(
        np.rint(values * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    result = cv2.addWeighted(background, 0.35, heatmap, 0.65, 0.0)
    _save_image(path, result)


def _write_variants(
    path: Path,
    observation: NasalViewObservation,
    image_rgb: np.ndarray,
    coordinates: ObservationCoordinates,
) -> None:
    result = _work_bgr(image_rgb, coordinates)
    for variant_name in ("eroded", "base", "dilated"):
        curves = observation.variant_boundaries_work.get(
            variant_name,
            {},
        )
        color = VARIANT_COLORS[variant_name]
        for points in curves.values():
            _polyline(result, points, color, thickness=2)
    _save_image(path, result)


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _write_html(
    output_dir: Path,
    bundle: NasalObservationBundle,
) -> Path:
    sections = []
    rows = []
    for semantic_view in REPORT_VIEWS:
        observation = bundle.by_view[semantic_view]
        camera_view = CAMERA_VIEW_BY_SEMANTIC[semantic_view]
        summary = _observation_summary(observation)
        confidence = summary["confidence"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(semantic_view)}</td>"
            f"<td>{html.escape(observation.camera.name)}</td>"
            f"<td>{summary['boundary_pixel_count']}</td>"
            f"<td>{_format_float(confidence['boundary_mean'])}</td>"
            f"<td>{_format_float(confidence['boundary_p10'])}</td>"
            f"<td>{_format_float(confidence['boundary_p90'])}</td>"
            "</tr>"
        )
        sections.append(
            "<section>"
            f"<h2>{html.escape(semantic_view)}</h2>"
            "<div class=\"grid\">"
            f"<figure><img src=\"{camera_view}_overlay.png\" "
            f"alt=\"{semantic_view} geometry overlay\"><figcaption>"
            "Observation boundary, baseline priors, and fixed-rig ROI."
            "</figcaption></figure>"
            f"<figure><img src=\"{camera_view}_confidence.png\" "
            f"alt=\"{semantic_view} confidence heatmap\"><figcaption>"
            "Continuous observation confidence heatmap.</figcaption></figure>"
            f"<figure><img src=\"{camera_view}_variants.png\" "
            f"alt=\"{semantic_view} erosion base dilation comparison\">"
            "<figcaption>Erosion, base, and dilation boundary comparison."
            "</figcaption></figure>"
            "</div></section>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nasal Observation Audit</title>
<style>
body{{margin:0;background:#11161b;color:#eef2f5;
font-family:Segoe UI,Arial,sans-serif;letter-spacing:0}}
main{{width:min(1500px,calc(100% - 32px));margin:auto;padding:24px 0 40px}}
h1{{font-size:28px;margin:0 0 10px}} h2{{font-size:20px;margin:30px 0 12px}}
p{{color:#bac4cc;line-height:1.6;max-width:1050px}}
.notice{{color:#ffd166;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
figure{{margin:0}} img{{display:block;width:100%;height:auto;border:1px solid #39444d}}
figcaption{{color:#aeb9c2;font-size:13px;padding-top:7px;line-height:1.4}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin-top:18px}}
th,td{{padding:9px;border-bottom:1px solid #39444d;text-align:left}}
code{{color:#8bd3dd}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Nasal Observation Audit</h1>
<p class="notice">纯几何观测，不含纹理评分</p>
<p>
Front 图显示 subject-left/right 鼻翼 parser 边界、baseline 投影先验和中心线。
Side 图显示完整 face silhouette、选中的鼻头曲线、四个 observation anchors、
两条 baseline side prior 轨迹，以及由固定 rig 计算的 epipolar 线。
黄色矩形为有效 ROI；epipolar 与 ROI 只限定搜索区域，
真实目标来自 parser nose boundary 和 side face silhouette。
</p>
<p>
橙色为 subject-left baseline prior，紫色为 subject-right baseline prior；
青色曲线与数字为真实 observation；anchors 为：
<code>1 upper_tip</code>、<code>2 tip_apex</code>、
<code>3 lower_tip</code>、<code>4 alar_transition</code>。
</p>
<table><thead><tr><th>Semantic view</th><th>Camera</th><th>Boundary pixels</th>
<th>Boundary confidence mean</th><th>P10</th><th>P90</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
{''.join(sections)}
</main></body></html>"""
    index_path = output_dir / "index.html"
    index_path.write_text(document, encoding="utf-8")
    return index_path


def write_nasal_observation_report(
    output_dir: str | Path,
    bundle: NasalObservationBundle,
    *,
    images_by_view: Mapping[str, np.ndarray],
    side_face_masks: Mapping[str, np.ndarray],
    baseline_priors_by_subject_side: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ],
    coordinates_by_view: Mapping[str, ObservationCoordinates],
    centerline_x_original: float,
) -> Path:
    """Write a self-contained file-viewable three-view geometry report."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    observations = _observation_by_camera_view(bundle)
    missing_images = [
        view for view in ("left", "front", "right")
        if view not in images_by_view
    ]
    if missing_images:
        raise ValueError(
            "report images are missing views: " + ", ".join(missing_images)
        )
    missing_coordinates = [
        view for view in ("left", "front", "right")
        if view not in coordinates_by_view
    ]
    if missing_coordinates:
        raise ValueError(
            "report coordinates are missing views: "
            + ", ".join(missing_coordinates)
        )
    missing_masks = [
        view for view in ("left", "right")
        if view not in side_face_masks
    ]
    if missing_masks:
        raise ValueError(
            "report side masks are missing views: " + ", ".join(missing_masks)
        )
    front_observation = observations["front"]
    for camera_view in ("left", "front", "right"):
        observation = observations[camera_view]
        _write_overlay(
            target / f"{camera_view}_overlay.png",
            camera_view,
            observation,
            images_by_view[camera_view],
            coordinates_by_view,
            side_face_masks,
            baseline_priors_by_subject_side,
            centerline_x_original,
            front_observation,
        )
        _write_confidence(
            target / f"{camera_view}_confidence.png",
            observation,
            images_by_view[camera_view],
            coordinates_by_view[camera_view],
        )
        _write_variants(
            target / f"{camera_view}_variants.png",
            observation,
            images_by_view[camera_view],
            coordinates_by_view[camera_view],
        )
    return _write_html(target, bundle)
