"""Static audit report for cross-view nasal texture observations."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from src.cross_view_geometry import write_ply
from src.geometry.nasal_texture_observations import (
    NasalEpipolarMatchResult,
    NasalTextureObservationBundle,
)
from src.geometry.nasal_view_registration import FixedNasalViewRegistration


def _json_ready(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _point(point: np.ndarray) -> tuple[int, int]:
    return tuple(np.rint(point).astype(int))


def _draw_pair_overlay(
    image_rgb: np.ndarray,
    semantic_view: str,
    results_by_side: Mapping[str, NasalEpipolarMatchResult],
) -> np.ndarray:
    canvas = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    colors = {
        "subject-left": (90, 220, 110),
        "subject-right": (255, 180, 60),
    }
    for side, result in results_by_side.items():
        color = colors[side]
        for match in result.matches:
            pixel = match.front_pixel if semantic_view == "front" else match.side_pixel
            if semantic_view not in {"front", side}:
                continue
            cv2.circle(canvas, _point(pixel), 4, color, -1, lineType=cv2.LINE_AA)
        for rejection in result.rejected:
            if semantic_view == "front":
                pixel = rejection.seed.front_pixel
            elif semantic_view == side:
                pixel = rejection.seed.predicted_side_pixel
            else:
                continue
            x, y = _point(pixel)
            cv2.line(canvas, (x - 3, y - 3), (x + 3, y + 3), (70, 70, 235), 1)
            cv2.line(canvas, (x - 3, y + 3), (x + 3, y - 3), (70, 70, 235), 1)
    return canvas


def _draw_coverage(
    image_rgb: np.ndarray,
    bundle: NasalTextureObservationBundle,
) -> np.ndarray:
    canvas = np.asarray(image_rgb).copy()
    heat = np.zeros(canvas.shape[:2], dtype=np.float32)
    region_colors = {
        "soft_triangle": (250, 185, 70),
        "alar_dome": (80, 220, 120),
        "alar_groove": (110, 155, 255),
    }
    overlay = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for observation in bundle.trusted:
        pixel = observation.pair_match.front_pixel
        cv2.circle(heat, _point(pixel), 18, 1.0, -1, lineType=cv2.LINE_AA)
        color = region_colors.get(observation.pair_match.semantic_region, (230, 230, 230))
        cv2.circle(overlay, _point(pixel), 4, color, -1, lineType=cv2.LINE_AA)
    if float(heat.max()) > 0.0:
        heat /= float(heat.max())
        colored = cv2.applyColorMap(np.uint8(np.clip(heat, 0.0, 1.0) * 255), cv2.COLORMAP_TURBO)
        mask = heat > 0.0
        overlay[mask] = np.uint8(0.72 * overlay[mask] + 0.28 * colored[mask])
    return overlay


def _draw_distance_map(
    image_rgb: np.ndarray,
    bundle: NasalTextureObservationBundle,
    distances_m: Sequence[float],
) -> np.ndarray:
    canvas = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    values = np.asarray(distances_m, dtype=np.float64)
    scale = max(float(np.percentile(values, 90.0)), 1e-6) if len(values) else 1.0
    for observation, distance in zip(bundle.trusted, values):
        ratio = float(np.clip(distance / scale, 0.0, 1.0))
        color = (int(255 * ratio), int(255 * (1.0 - ratio)), 80)
        cv2.circle(
            canvas,
            _point(observation.pair_match.front_pixel),
            5,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def write_nasal_texture_observation_report(
    output_dir: str | Path,
    *,
    images_by_view: Mapping[str, np.ndarray],
    matches_by_side: Mapping[str, NasalEpipolarMatchResult],
    observations: NasalTextureObservationBundle,
    registration: FixedNasalViewRegistration,
    gate: Mapping[str, Any],
    model_distances_m: Sequence[float],
    metadata: Mapping[str, Any],
) -> Path:
    """Write an offline report without exporting or mutating model geometry."""
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    for view, image in images_by_view.items():
        overlay = _draw_pair_overlay(image, view, matches_by_side)
        cv2.imwrite(str(target / f"matches_{view}.png"), overlay)
    coverage = _draw_coverage(images_by_view["front"], observations)
    cv2.imwrite(str(target / "coverage_front.png"), coverage)
    distance_map = _draw_distance_map(
        images_by_view["front"],
        observations,
        model_distances_m,
    )
    cv2.imwrite(str(target / "model_distance_front.png"), distance_map)

    points = np.asarray(
        [value.point_reference_m for value in observations.trusted],
        dtype=np.float64,
    ).reshape(-1, 3)
    colors = np.tile(np.asarray([[80, 220, 110]], dtype=np.uint8), (len(points), 1))
    write_ply(target / "trusted_nasal_points.ply", points, colors)

    payload = {
        "schema": "nasal-texture-observation-audit-v1",
        "status": str(gate.get("status", "unknown")),
        "release_a_gate": dict(gate),
        "registration": registration.to_dict(),
        "matches": {
            side: result.to_dict() for side, result in matches_by_side.items()
        },
        "observations": observations.to_dict(),
        "model_distances_m": list(float(value) for value in model_distances_m),
        "metadata": dict(metadata),
    }
    (target / "metrics.json").write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    gate_rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{'pass' if passed else 'fail'}</td></tr>"
        for name, passed in dict(gate.get("gates", {})).items()
    )
    count_rows = "".join(
        f"<tr><td>{html.escape(str(side))}</td><td>{int(count)}</td></tr>"
        for side, count in dict(gate.get("trusted_by_side", {})).items()
    )
    cards = "".join(
        f"<figure><img src='matches_{html.escape(view)}.png'><figcaption>{html.escape(view)}</figcaption></figure>"
        for view in images_by_view
    )
    status = html.escape(str(gate.get("status", "unknown")))
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>鼻部跨视角纹理证据审计</title>
<style>body{{margin:0;background:#10161c;color:#eaf0f4;font-family:system-ui,sans-serif;letter-spacing:0}}main{{max-width:1280px;margin:auto;padding:22px}}h1{{font-size:25px}}.status{{font-size:20px;color:{'#75dda0' if gate.get('passed') else '#ff9e8e'}}}table{{border-collapse:collapse;width:100%;margin:12px 0 22px}}td{{padding:8px;border-bottom:1px solid #33424d}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}figure{{margin:0;border:1px solid #34434e}}img{{display:block;width:100%}}figcaption{{padding:7px}}.diag{{grid-template-columns:repeat(2,1fr)}}@media(max-width:850px){{.grid,.diag{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>鼻部跨视角弱纹理证据审计</h1><p class="status">Release A: {status}</p>
<p>绿色/橙色圆点为通过局部纹理、唯一性、反向匹配和固定外参三角化的点；红叉为被拒绝的搜索。此页只审计证据，不改变 v10 几何。</p>
<h2>门槛</h2><table>{gate_rows}</table><h2>可靠点数量</h2><table>{count_rows}</table>
<h2>跨视角匹配</h2><div class="grid">{cards}</div>
<h2>空间诊断</h2><div class="grid diag"><figure><img src="coverage_front.png"><figcaption>语义与空间覆盖</figcaption></figure><figure><img src="model_distance_front.png"><figcaption>三角化点到锁定 v10 鼻部三角面的距离</figcaption></figure></div>
<p>完整数值、拒绝原因和坐标来源见 <a href="metrics.json">metrics.json</a>。</p></main></body></html>"""
    report = target / "index.html"
    report.write_text(document, encoding="utf-8")
    return report


__all__ = ["write_nasal_texture_observation_report"]
