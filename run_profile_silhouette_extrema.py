"""Audit nose-tip and chin depth from side-view facial silhouettes."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.geometry.profile_silhouette_extrema import (
    SUPPORTED_EXTREMA,
    build_profile_silhouette_extrema,
    original_points_to_canvas,
)
from src.geometry.profile_triangulation import load_profile_rig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--profile-report", type=Path, required=True)
    parser.add_argument("--preprocess-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_capture_images(
    captures: Path,
    rig,
) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for view, camera in rig.cameras_by_view.items():
        candidates = sorted(captures.glob(f"{camera.name}_*.jpg"))
        if len(candidates) != 1:
            raise ValueError(
                f"expected one {camera.name}_*.jpg, found {len(candidates)}"
            )
        image = cv2.imread(str(candidates[0]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read {candidates[0]}")
        if (image.shape[1], image.shape[0]) != tuple(camera.image_size):
            raise ValueError(f"image size mismatch for {camera.name}/{view}")
        images[view] = image
    return images


def _letterbox_image(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    height, width = image.shape[:2]
    scale = min(target_width / float(width), target_height / float(height))
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return canvas


def _draw_label(
    image: np.ndarray,
    point: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y = np.rint(point).astype(np.int32)
    cv2.circle(image, (int(x), int(y)), 8, color, 3, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (int(x) + 10, int(y) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _write_overlays(
    output_dir: Path,
    images: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    profile_report: dict[str, Any],
    report: dict[str, Any],
    rig,
) -> None:
    for side_view in ("left", "right"):
        mask = masks[side_view]
        overlay = _letterbox_image(images[side_view], mask.shape)
        contour, _hierarchy = cv2.findContours(
            np.asarray(mask > 0, dtype=np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if contour:
            cv2.drawContours(
                overlay,
                [max(contour, key=cv2.contourArea)],
                -1,
                (80, 220, 120),
                2,
                cv2.LINE_AA,
            )
        camera = rig.cameras_by_view[side_view]
        for semantic_name in SUPPORTED_EXTREMA:
            side_report = report["points"][semantic_name]["side_reports"][
                side_view
            ]
            prior_original = np.asarray(
                profile_report["detectors"][side_view][semantic_name][
                    "mediapipe_px"
                ],
                dtype=np.float64,
            )
            prior_canvas = original_points_to_canvas(
                prior_original,
                camera.image_size,
                mask.shape,
            ).reshape(2)
            _draw_label(
                overlay,
                prior_canvas,
                f"detector {semantic_name}",
                (220, 90, 220),
            )
            for variant in side_report["variants"]:
                selected = variant.get("selected_canvas_px")
                if selected is None:
                    continue
                offset = int(variant["offset_px"])
                color = (70, 220, 255) if offset == 0 else (255, 190, 70)
                _draw_label(
                    overlay,
                    np.asarray(selected, dtype=np.float64),
                    f"{semantic_name} mask {offset:+d}",
                    color,
                )
        target = output_dir / f"{side_view}_silhouette_extrema.png"
        if not cv2.imwrite(str(target), overlay):
            raise RuntimeError(f"failed to write {target}")

    front = _letterbox_image(
        images["front"],
        next(iter(masks.values())).shape,
    )
    front_camera = rig.cameras_by_view["front"]
    for semantic_name in SUPPORTED_EXTREMA:
        raw = np.asarray(
            profile_report["detectors"]["front"][semantic_name]["mediapipe_px"],
            dtype=np.float64,
        )
        point = original_points_to_canvas(
            raw,
            front_camera.image_size,
            front.shape[:2],
        ).reshape(2)
        _draw_label(front, point, f"front {semantic_name}", (70, 220, 255))
    target = output_dir / "front_semantic_rays.png"
    if not cv2.imwrite(str(target), front):
        raise RuntimeError(f"failed to write {target}")


def _format_optional(value: Any, scale: float = 1.0) -> str:
    if value is None:
        return "n/a"
    number = float(value) * scale
    return "n/a" if not np.isfinite(number) else f"{number:.2f}"


def _write_html(output_dir: Path, report: dict[str, Any]) -> None:
    rows = []
    for semantic_name in SUPPORTED_EXTREMA:
        point = report["points"][semantic_name]
        comparison = report["comparison_to_local_depth_surface"].get(
            semantic_name,
            {},
        )
        side_cells = []
        for side_view in ("left", "right"):
            side = point["side_reports"][side_view]
            center = side.get("selected_center") or {}
            selection = center.get("selection") or {}
            side_cells.append(
                f"{side_view}: {'PASS' if side['passed'] else 'REJECT'}; "
                f"epi {selection.get('epipolar_distance_work_px', float('nan')):.2f}px; "
                f"mask depth span {_format_optional(side['variant_depth_spread_m'], 1000.0)}mm"
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(semantic_name)}</td>"
            f"<td>{'PASS' if point['passed'] else 'REJECT'}</td>"
            f"<td>{point['valid_side_count']}</td>"
            f"<td>{_format_optional(point['cross_side_depth_delta_m'], 1000.0)}</td>"
            f"<td>{_format_optional(comparison.get('silhouette_depth_m'), 1000.0)}</td>"
            f"<td>{_format_optional(comparison.get('local_surface_depth_m'), 1000.0)}</td>"
            f"<td>{_format_optional(comparison.get('silhouette_minus_surface_mm'))}</td>"
            f"<td>{html.escape(' | '.join(side_cells))}</td>"
            f"<td>{html.escape(', '.join(point['issues']) or 'none')}</td>"
            "</tr>"
        )
    gate = report["quality_gate"]
    comparison = report["comparison_to_local_depth_surface"]
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>侧脸轮廓极值诊断</title><style>
body{{margin:0;background:#10151b;color:#edf2f7;font-family:Segoe UI,Arial,sans-serif;letter-spacing:0}}
main{{width:min(1500px,calc(100% - 32px));margin:auto;padding:24px 0 40px}}
h1{{font-size:28px}} h2{{font-size:20px;margin-top:28px}} p{{color:#b7c1cc;line-height:1.6}}
.pass{{color:#65d693}} .fail{{color:#ff8a8a}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
img{{display:block;width:100%;height:auto;border:1px solid #34404c}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:9px;border-bottom:1px solid #34404c;text-align:left;vertical-align:top}}
code{{color:#f2c879}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>侧脸轮廓极值诊断</h1>
<p class="{'pass' if gate['passed'] else 'fail'}">几何证据门禁：{'PASS' if gate['passed'] else 'REJECT'}；通过：{html.escape(', '.join(gate['accepted_points']) or 'none')}。</p>
<p>这不是贴图分数。正脸语义点只定义一条三维视线，左右侧脸的真实外轮廓与该视线的极线交点提供深度。绿色是原始 face mask 轮廓，紫色是检测器先验，青色是原 mask 结果，黄色是膨胀/腐蚀稳定性样本。</p>
<p>鼻尖相对上唇前突：<code>{_format_optional(comparison.get('nose_tip_minus_upper_lip_mm'))} mm</code>；口中心到下巴前后差：<code>{_format_optional(comparison.get('mouth_center_minus_chin_mm'))} mm</code>。局部 LoFTR 深度面只作对照，不再负责鼻尖和下巴。</p>
<table><thead><tr><th>部位</th><th>状态</th><th>有效侧</th><th>左右深度差 mm</th><th>轮廓深度 mm</th><th>局部面深度 mm</th><th>差值 mm</th><th>侧面证据</th><th>拒绝原因</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>正脸语义视线锚点</h2><img src="front_semantic_rays.png" alt="front semantic anchors">
<h2>侧脸轮廓证据</h2><div class="grid"><img src="left_silhouette_extrema.png" alt="left silhouette"><img src="right_silhouette_extrema.png" alt="right silhouette"></div>
</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rig = load_profile_rig(
        args.calibration.resolve(),
        max_stereo_rms_px=10.0,
    )
    profile_report = _read_json(args.profile_report.resolve())
    images = _load_capture_images(args.captures.resolve(), rig)
    masks = {
        side: cv2.imread(
            str(args.preprocess_dir.resolve() / f"{side}_face_mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        for side in ("left", "right")
    }
    missing = [side for side, mask in masks.items() if mask is None]
    if missing:
        raise FileNotFoundError(
            f"missing cached face masks for: {', '.join(missing)}"
        )
    report = build_profile_silhouette_extrema(profile_report, masks, rig)
    report["captures"] = str(args.captures.resolve())
    report["calibration"] = str(args.calibration.resolve())
    report["profile_report"] = str(args.profile_report.resolve())
    report["preprocess_dir"] = str(args.preprocess_dir.resolve())
    (output_dir / "silhouette_extrema_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_overlays(
        output_dir,
        images,
        masks,
        profile_report,
        report,
        rig,
    )
    _write_html(output_dir, report)
    print(f"Silhouette extrema audit: {output_dir / 'index.html'}")
    print(json.dumps(report["quality_gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
