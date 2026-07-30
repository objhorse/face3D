"""Refine one fixed three-camera rig from multiple capture datasets."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.cross_view_geometry import (
    Camera,
    load_calibration,
    restore_mask_to_work_frame,
)
from src.geometry.joint_rig_refinement import (
    JointRigThresholds,
    MatchSet,
    reconcile_triplet_baseline_scales,
    refine_joint_pair,
)
from src.learned_cross_view_geometry import (
    WORK_SIZE,
    _draw_match_overlay,
    _load_loftr,
    _read_capture,
    _run_loftr,
    _sample_mask,
    write_refined_calibration_candidate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate one fixed camera2-to-side-camera pose from multiple "
            "capture datasets. The official calibration is never overwritten."
        )
    )
    parser.add_argument(
        "--captures",
        action="append",
        type=Path,
        required=True,
        help="Capture directory; repeat once per dataset.",
    )
    parser.add_argument(
        "--reconstruction",
        action="append",
        type=Path,
        required=True,
        help=(
            "Matching reconstruction containing debug/preprocess masks; "
            "repeat in the same order as --captures."
        ),
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-matches-per-dataset", type=int, default=320)
    return parser.parse_args()


def _read_face_mask(reconstruction: Path, camera: Camera) -> np.ndarray:
    path = (
        reconstruction
        / "debug"
        / "preprocess"
        / f"{camera.view}_face_mask.png"
    )
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"missing preprocessing face mask: {path}")
    return restore_mask_to_work_frame(mask, target_size=WORK_SIZE)


def _collect_pair_matches(
    *,
    dataset: str,
    camera_a: Camera,
    camera_b: Camera,
    images: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    matcher: object,
    device: object,
    output_dir: Path,
) -> MatchSet:
    points_a, points_b, confidence = _run_loftr(
        matcher,
        device,
        images[camera_a.name],
        images[camera_b.name],
    )
    confidence_floor = (
        max(0.10, float(np.percentile(confidence, 25.0)))
        if len(confidence)
        else 1.0
    )
    accepted = (
        _sample_mask(masks[camera_a.name], points_a)
        & _sample_mask(masks[camera_b.name], points_b)
        & (confidence >= confidence_floor)
    )
    pair_name = f"{camera_a.name}_{camera_b.name}"
    cv2.imwrite(
        str(output_dir / f"{dataset}_{pair_name}_matches.jpg"),
        _draw_match_overlay(
            images[camera_a.name],
            images[camera_b.name],
            points_a,
            points_b,
            accepted,
            accepted,
        ),
    )
    np.savez_compressed(
        output_dir / f"{dataset}_{pair_name}_accepted_matches.npz",
        points_front_work=points_a[accepted],
        points_side_work=points_b[accepted],
        confidence=confidence[accepted],
        work_size=np.asarray(WORK_SIZE, dtype=np.int32),
        camera_front=np.asarray(camera_a.name),
        camera_side=np.asarray(camera_b.name),
    )
    return MatchSet(
        dataset=dataset,
        points_a=points_a[accepted],
        points_b=points_b[accepted],
        confidence=confidence[accepted],
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_report(output_dir: Path, quality: dict[str, Any]) -> Path:
    pair_cards: list[str] = []
    for pair_name, metrics in quality["pairs"].items():
        rows: list[str] = []
        for dataset, result in metrics["validation"].items():
            error = result["absolute_y_error_px"]
            holdout = metrics["leave_one_out"][dataset]["absolute_y_error_px"]
            rows.append(
                "<tr>"
                f"<td>{html.escape(dataset)}</td>"
                f"<td>{error['p50']:.2f}</td>"
                f"<td>{error['p90']:.2f}</td>"
                f"<td>{holdout['p50']:.2f}</td>"
                f"<td>{holdout['p90']:.2f}</td>"
                "</tr>"
            )
        status = "pass" if metrics["accepted"] else "fail"
        issues = ", ".join(metrics["issues"]) or "none"
        warnings = ", ".join(metrics.get("warnings", [])) or "none"
        pair_cards.append(
            f"""
<section>
  <h2>{html.escape(pair_name)} <span class="{status}">{status}</span></h2>
  <p>Current-pose delta: rotation {metrics['rotation_delta_deg']:.2f} deg,
  translation direction {metrics['translation_direction_delta_deg']:.2f} deg.</p>
  <p>Leave-one-out spread: rotation
  {metrics['leave_one_out_pose_spread']['rotation_deg']:.2f} deg,
  translation direction
  {metrics['leave_one_out_pose_spread']['translation_direction_deg']:.2f} deg.</p>
  <p>Issues: <code>{html.escape(issues)}</code></p>
  <p>Warnings: <code>{html.escape(warnings)}</code></p>
  <table>
    <thead><tr><th>Dataset</th><th>Joint P50</th><th>Joint P90</th>
    <th>Holdout P50</th><th>Holdout P90</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>"""
        )
    candidate = quality.get("candidate_calibration")
    candidate_line = (
        f'<a href="{html.escape(Path(candidate).name)}">candidate calibration</a>'
        if candidate
        else "No calibration candidate was accepted."
    )
    reconciliation = quality.get("baseline_reconciliation", {})
    baseline_section = ""
    if reconciliation:
        current = reconciliation["current_baseline_m"]
        adjusted = reconciliation["adjusted_baseline_m"]
        ratio = reconciliation["pooled_left_over_right_depth_ratio"]
        baseline_section = f"""
<section>
  <h2>Three-view baseline ratio
  <span class="{'pass' if reconciliation['accepted'] else 'fail'}">
  {'pass' if reconciliation['accepted'] else 'fail'}</span></h2>
  <p>Common front-pixel triplets: {reconciliation['valid_triplets_total']};
  pooled left/right depth ratio P50: {ratio['p50']:.4f};
  cross-dataset median spread: {reconciliation['dataset_median_spread']:.4f}.</p>
  <p>Left baseline: {current['left'] * 1000.0:.2f} mm to
  {adjusted['left'] * 1000.0:.2f} mm. Right baseline:
  {current['right'] * 1000.0:.2f} mm to
  {adjusted['right'] * 1000.0:.2f} mm.</p>
</section>"""
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Joint fixed-rig refinement</title>
<style>
body {{ margin: 0; background: #10151b; color: #e8eef2;
font: 14px "Segoe UI", sans-serif; }}
main {{ width: min(1100px, calc(100% - 32px)); margin: 24px auto 48px; }}
section {{ padding: 16px 0; border-bottom: 1px solid #33414b; }}
h1, h2 {{ letter-spacing: 0; }} h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; }}
.pass {{ color: #83d6a2; }} .fail {{ color: #ff9d9d; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #27343e; }}
code {{ color: #b8d8e8; }} a {{ color: #8fd4f4; }}
</style></head><body><main>
<h1>Multi-dataset fixed-rig refinement</h1>
<p>Accepted: <strong class="{'pass' if quality['passed'] else 'fail'}">
{str(quality['passed']).lower()}</strong>. {candidate_line}</p>
<p>This diagnostic preserves the calibrated metric baseline and never changes
the official calibration automatically.</p>
{''.join(pair_cards)}
{baseline_section}
</main></body></html>"""
    path = output_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def run_joint_refinement(
    *,
    captures: list[Path],
    reconstructions: list[Path],
    calibration: Path,
    output_dir: Path,
    max_matches_per_dataset: int,
) -> dict[str, Any]:
    if len(captures) != len(reconstructions):
        raise ValueError("--captures and --reconstruction counts must match")
    if len(captures) < 2:
        raise ValueError("joint rig refinement requires at least two datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = load_calibration(calibration)
    by_view = {camera.view: camera for camera in cameras.values()}
    missing = {"left", "front", "right"} - set(by_view)
    if missing:
        raise ValueError(f"calibration is missing views: {sorted(missing)}")
    reference = by_view["front"]
    pair_specs = (
        (f"{reference.name}_{by_view['left'].name}", by_view["left"]),
        (f"{reference.name}_{by_view['right'].name}", by_view["right"]),
    )
    matcher, device = _load_loftr()
    matches_by_pair: dict[str, list[MatchSet]] = {
        name: [] for name, _camera in pair_specs
    }
    dataset_names: list[str] = []
    for capture_dir, reconstruction in zip(captures, reconstructions):
        capture_dir = capture_dir.resolve()
        reconstruction = reconstruction.resolve()
        dataset = capture_dir.name
        if dataset in dataset_names:
            raise ValueError(f"duplicate capture dataset: {dataset}")
        dataset_names.append(dataset)
        images = {
            name: _read_capture(capture_dir, name)
            for name in cameras
        }
        masks = {
            name: _read_face_mask(reconstruction, camera)
            for name, camera in cameras.items()
        }
        for pair_name, side_camera in pair_specs:
            matches_by_pair[pair_name].append(
                _collect_pair_matches(
                    dataset=dataset,
                    camera_a=reference,
                    camera_b=side_camera,
                    images=images,
                    masks=masks,
                    matcher=matcher,
                    device=device,
                    output_dir=output_dir,
                )
            )

    thresholds = JointRigThresholds(
        max_matches_per_dataset=int(max_matches_per_dataset)
    )
    pair_metrics: dict[str, Any] = {}
    candidate_results: list[dict[str, Any]] = []
    refined_by_pair: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pair_name, side_camera in pair_specs:
        metrics, rotation, translation = refine_joint_pair(
            matches_by_pair[pair_name],
            reference,
            side_camera,
            thresholds=thresholds,
            work_size=WORK_SIZE,
        )
        pair_metrics[pair_name] = metrics
        refined_by_pair[pair_name] = (rotation, translation)
        candidate_results.append(
            {
                "pair": pair_name,
                "camera_a": reference.name,
                "camera_b": side_camera.name,
                "refined_pose_candidate": metrics["refined_pose_candidate"],
                "_refined_rotation_ba": rotation,
                "_refined_translation_ba": translation,
            }
        )

    left_pair = f"{reference.name}_{by_view['left'].name}"
    right_pair = f"{reference.name}_{by_view['right'].name}"
    left_rotation, left_translation = refined_by_pair[left_pair]
    right_rotation, right_translation = refined_by_pair[right_pair]
    baseline_reconciliation, adjusted_left, adjusted_right = (
        reconcile_triplet_baseline_scales(
            matches_by_pair[left_pair],
            matches_by_pair[right_pair],
            reference,
            by_view["left"],
            by_view["right"],
            left_rotation,
            left_translation,
            right_rotation,
            right_translation,
            thresholds=thresholds,
            work_size=WORK_SIZE,
        )
    )
    for result in candidate_results:
        if result["camera_b"] == by_view["left"].name:
            result["_refined_translation_ba"] = adjusted_left
        elif result["camera_b"] == by_view["right"].name:
            result["_refined_translation_ba"] = adjusted_right

    passed = (
        all(metrics["accepted"] for metrics in pair_metrics.values())
        and baseline_reconciliation["accepted"]
    )
    candidate_path: Path | None = None
    updates: list[dict[str, Any]] = []
    if passed:
        candidate_path = output_dir / "camera_calibration_joint_candidate.json"
        updates = write_refined_calibration_candidate(
            calibration_path=calibration,
            output_path=candidate_path,
            cameras=cameras,
            pair_results=candidate_results,
        )
        candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_payload["loftr_refined_candidate"] = {
            "created_by": "run_joint_rig_refinement.py",
            "note": (
                "Multi-dataset diagnostic candidate. Overall metric scale is "
                "anchored by ChArUco; the left/right baseline ratio is refined "
                "from common front-pixel triplets under bounded priors. Do not "
                "replace the official calibration until profile triangulation "
                "also passes."
            ),
            "datasets": dataset_names,
            "work_size": list(WORK_SIZE),
            "updates": updates,
        }
        candidate_path.write_text(
            json.dumps(candidate_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    quality = {
        "passed": bool(passed),
        "audit_only": True,
        "source_calibration": str(calibration.resolve()),
        "candidate_calibration": (
            str(candidate_path.resolve()) if candidate_path else None
        ),
        "datasets": dataset_names,
        "work_size": list(WORK_SIZE),
        "device": str(device),
        "pairs": pair_metrics,
        "baseline_reconciliation": baseline_reconciliation,
        "candidate_updates": updates,
    }
    quality_path = output_dir / "quality.json"
    quality_path.write_text(
        json.dumps(_json_ready(quality), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path = _write_report(output_dir, quality)
    quality["quality_path"] = str(quality_path.resolve())
    quality["report_path"] = str(report_path.resolve())
    return quality


def main() -> int:
    args = _parse_args()
    result = run_joint_refinement(
        captures=[path.resolve() for path in args.captures],
        reconstructions=[path.resolve() for path in args.reconstruction],
        calibration=args.calibration.resolve(),
        output_dir=args.output.resolve(),
        max_matches_per_dataset=args.max_matches_per_dataset,
    )
    print(f"Joint fixed-rig accepted: {result['passed']}")
    for pair_name, metrics in result["pairs"].items():
        print(
            f"  {pair_name}: accepted={metrics['accepted']}, "
            f"rotation delta={metrics['rotation_delta_deg']:.2f} deg, "
            f"translation delta={metrics['translation_direction_delta_deg']:.2f} deg"
        )
        for dataset, validation in metrics["validation"].items():
            error = validation["absolute_y_error_px"]
            holdout = metrics["leave_one_out"][dataset]["absolute_y_error_px"]
            print(
                f"    {dataset}: joint P50/P90={error['p50']:.2f}/"
                f"{error['p90']:.2f}px, holdout={holdout['p50']:.2f}/"
                f"{holdout['p90']:.2f}px"
            )
        if metrics["issues"]:
            print(f"    issues: {', '.join(metrics['issues'])}")
        if metrics.get("warnings"):
            print(f"    warnings: {', '.join(metrics['warnings'])}")
    print(f"Report: {result['report_path']}")
    if result["candidate_calibration"]:
        print(f"Candidate: {result['candidate_calibration']}")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
