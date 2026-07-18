"""Static quality report for stable three-view reconstruction."""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def _rel(target: Optional[Path], base: Path) -> Optional[str]:
    if target is None:
        return None
    try:
        return target.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return target.resolve().as_posix()


def _copy_views(source_view_paths: Dict[str, Path], report_dir: Path) -> Dict[str, str]:
    copied = {}
    views_dir = report_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    for view, src in source_view_paths.items():
        if not src or not Path(src).exists():
            continue
        dst = views_dir / f"{view}{Path(src).suffix.lower() or '.jpg'}"
        try:
            shutil.copy2(src, dst)
            copied[view] = _rel(dst, report_dir) or str(dst)
        except Exception:
            copied[view] = str(src)
    return copied


def _copy_report_image(src: Path, report_dir: Path, prefix: str) -> Optional[str]:
    if not src.exists():
        return None
    dst_dir = report_dir / "debug_images"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{prefix}_{src.name}"
    try:
        shutil.copy2(src, dst)
        return _rel(dst, report_dir) or str(dst)
    except Exception:
        return None


def _metric_rows(metrics: Dict[str, Any]) -> str:
    keys = [
        "vertex_count",
        "face_count",
        "degenerate_faces",
        "tiny_faces",
        "extreme_aspect_faces",
        "boundary_edges",
        "nonmanifold_edges",
        "component_count",
        "watertight",
    ]
    rows = []
    for key in keys:
        rows.append(
            "<tr><th>%s</th><td>%s</td></tr>"
            % (html.escape(key), html.escape(str(metrics.get(key, ""))))
        )
    return "\n".join(rows)


def _shape_refinement_rows(report: Dict[str, Any]) -> str:
    decision = report.get("geometry_decision", {})
    metrics = decision.get("metrics", {})
    rows = [
        ("candidate", "accepted" if decision.get("accepted") else "rejected; baseline kept"),
        ("trusted contour improvement", f"{metrics.get('trusted_boundary_improve_pct_points', '')} % face width"),
        ("measurably improved views", metrics.get("improved_views", "")),
        ("views preserved within render resolution", metrics.get("preserved_views", "")),
        ("maximum view worsening", f"{metrics.get('max_view_worsen_pct_points', '')} % face width"),
        ("trusted overlap drop", metrics.get("trusted_overlap_drop", "")),
        ("interior landmark mean worsening", f"{metrics.get('interior_mean_worsen_pct_points', '')} % face width"),
        ("mesh quality", decision.get("gates", {}).get("mesh_quality", "")),
        ("legacy 0-16 contour", "diagnostic only; excluded from acceptance"),
        ("texture", "excluded from geometry metrics"),
    ]
    return "\n".join(
        "<tr><th>%s</th><td>%s</td></tr>"
        % (html.escape(str(key)), html.escape(str(value)))
        for key, value in rows
    )


def _identity_preservation_rows(report: Dict[str, Any]) -> str:
    joint = report.get("joint", {})
    final_gate = report.get("final", {})
    metrics = final_gate.get("metrics", {})
    thresholds = final_gate.get("thresholds", {})
    rows = [
        ("joint optimization", "pass" if joint.get("passed") else "failed"),
        ("selected anchor attempt", joint.get("selected_attempt", "")),
        ("coefficient L2 drift", metrics.get("coefficient_delta_l2", "")),
        ("coefficient L2 limit", thresholds.get("max_coefficient_l2", "")),
        ("mean neutral-mesh drift", f"{metrics.get('mean_displacement_pct', '')}% face width"),
        ("P95 neutral-mesh drift", f"{metrics.get('p95_displacement_pct', '')}% face width"),
        ("maximum neutral-mesh drift", f"{metrics.get('max_displacement_pct', '')}% face width"),
        ("final identity gate", "pass" if final_gate.get("passed") else "failed"),
        ("failed checks", ", ".join(final_gate.get("issues", []))),
        ("texture", "excluded from identity metrics"),
    ]
    return "\n".join(
        "<tr><th>%s</th><td>%s</td></tr>"
        % (html.escape(str(key)), html.escape(str(value)))
        for key, value in rows
    )


def write_stable_reconstruction_report(
    *,
    report_dir: Path,
    meta: Dict[str, Any],
    quality: Dict[str, Any],
    source_view_paths: Optional[Dict[str, Path]] = None,
    debug_root: Optional[Path] = None,
) -> Dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    copied_views = _copy_views(source_view_paths or {}, report_dir)

    quality_path = report_dir / "quality.json"
    meta_path = report_dir / "reconstruction_meta.json"
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    fit_quality = quality.get("fit", {}).get("geometry", {})
    final_quality = quality.get("texture", {}).get("final", {})
    fit_gate = quality.get("fit", {}).get("gate", {})
    texture_gate = quality.get("texture", {}).get("gate", {})
    shape_refinement = quality.get("fit", {}).get("shape_refinement", {})
    identity_preservation = quality.get("fit", {}).get("identity", {})
    confidence_map_path = (
        quality.get("texture", {})
        .get("confidence", {})
        .get("confidence_map", {})
        .get("path")
    )

    debug_root = debug_root or report_dir.parent
    projection = debug_root / "mesh_projection_front.jpg"
    optimized_reproj = sorted((debug_root / "optimized_pose_quality").glob("*optimized_reprojection.png")) if (debug_root / "optimized_pose_quality").exists() else []

    view_cards = ""
    for view in ("left", "front", "right"):
        src = copied_views.get(view)
        if src:
            view_cards += f'<figure><img src="{html.escape(src)}" alt="{view}"><figcaption>{html.escape(view)}</figcaption></figure>'

    reproj_cards = ""
    if projection.exists():
        projection_rel = _copy_report_image(projection, report_dir, "projection")
        reproj_cards += (
            '<figure><img src="%s" alt="front projection"><figcaption>front projection</figcaption></figure>'
            % html.escape(projection_rel or "")
        )
    for path in optimized_reproj[:3]:
        path_rel = _copy_report_image(path, report_dir, "landmark")
        reproj_cards += (
            '<figure><img src="%s" alt="%s"><figcaption>%s</figcaption></figure>'
            % (
                html.escape(path_rel or ""),
                html.escape(path.name),
                html.escape(path.stem),
            )
        )

    confidence_card = ""
    if confidence_map_path:
        confidence_rel = _copy_report_image(Path(confidence_map_path), report_dir, "texture_confidence")
        if confidence_rel:
            confidence_card = (
                '<figure><img src="%s" alt="texture confidence"><figcaption>texture confidence proxy</figcaption></figure>'
                % html.escape(confidence_rel)
            )

    silhouette_cards = ""
    silhouette_dir = debug_root / "shape_only_fine_tune"
    if silhouette_dir.exists():
        for view in ("left", "front", "right"):
            for stage in ("before", "after"):
                path = silhouette_dir / f"{view}_{stage}_silhouette.png"
                path_rel = _copy_report_image(path, report_dir, "silhouette")
                if path_rel:
                    caption = f"{view} {stage}: source / target / model / trust / overlay"
                    silhouette_cards += (
                        '<figure><img src="%s" alt="%s"><figcaption>%s</figcaption></figure>'
                        % (
                            html.escape(path_rel),
                            html.escape(caption),
                            html.escape(caption),
                        )
                    )

    gate_badge = "pass" if fit_gate.get("passed") and texture_gate.get("passed") else "risk"
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stable Three-View Reconstruction Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f7f4; color: #202124; }}
    header {{ padding: 28px 36px 18px; background: #163b3b; color: #fff; }}
    main {{ padding: 24px 36px 40px; max-width: 1180px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; background: #e7f2e7; color: #135a2a; font-weight: 700; }}
    .badge.risk {{ background: #fff0c2; color: #7a5200; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 8px 10px; font-size: 13px; color: #4f555a; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #ddd; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e5e5; text-align: left; font-size: 14px; }}
    th {{ width: 260px; background: #fafafa; }}
    .note {{ background: #fff; border-left: 4px solid #c9a227; padding: 12px 14px; margin-top: 12px; }}
    code {{ background: #ededed; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Stable Three-View Reconstruction</h1>
    <div class="badge {'' if gate_badge == 'pass' else 'risk'}">{html.escape(gate_badge)}</div>
  </header>
  <main>
    <section>
      <h2>Source Views</h2>
      <div class="grid">{view_cards or '<p>No source thumbnails available.</p>'}</div>
    </section>
    <section>
      <h2>Landmark And Projection Debug</h2>
      <div class="grid">{reproj_cards or '<p>No projection debug image available.</p>'}</div>
    </section>
    <section>
      <h2>Stable Geometry Quality</h2>
      <table>{_metric_rows(fit_quality)}</table>
    </section>
    <section>
      <h2>Identity Preservation</h2>
      <table>{_identity_preservation_rows(identity_preservation)}</table>
    </section>
    <section>
      <h2>Geometry Refinement Decision</h2>
      <table>{_shape_refinement_rows(shape_refinement)}</table>
      <div class="grid">{silhouette_cards or '<p>No silhouette diagnostics available.</p>'}</div>
    </section>
    <section>
      <h2>Textured Final Quality</h2>
      <table>{_metric_rows(final_quality)}</table>
    </section>
    <section>
      <h2>Texture Confidence</h2>
      <div class="note">
        低置信纹理区域会被标记或补色；稳定 v1 不通过删除 mesh 面片来处理不可见区域。
        三视角没有真实观测到的区域不作为医疗级测量依据。
      </div>
      <div class="grid">{confidence_card}</div>
      <p><a href="{html.escape(_rel(quality_path, report_dir) or quality_path.name)}">quality.json</a> ·
      <a href="{html.escape(_rel(meta_path, report_dir) or meta_path.name)}">reconstruction_meta.json</a></p>
    </section>
  </main>
</body>
</html>
"""
    index_path = report_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return {
        "index": str(index_path),
        "quality": str(quality_path),
        "meta": str(meta_path),
    }
