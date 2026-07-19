"""Audit profile depth across FLAME initialization and fitting stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src import config as cfg
from src.geometry.profile_depth_quality import (
    build_profile_depth_stage_report,
    compare_expression_depth,
    profile_depth_metrics,
)
from src.module2_geometry import FLAMEModel


ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reconstruction",
        type=Path,
        required=True,
        help="Stable reconstruction output containing meshes/stable_fit_meta.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: <reconstruction>/debug/profile_depth/profile_depth_baseline.json)",
    )
    parser.add_argument(
        "--max-expression-forward-mm",
        type=float,
        default=0.5,
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _parameter_array(
    parameters: dict[str, Any],
    key: str,
    expected_length: int,
) -> np.ndarray:
    value = np.asarray(parameters.get(key, []), dtype=np.float32).reshape(-1)
    if len(value) != expected_length:
        raise ValueError(
            f"{key} has {len(value)} values; expected {expected_length}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value


def _load_mesh_vertices(path: Path) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"invalid mesh vertices in {path}")
    return vertices


def audit_profile_depth(
    reconstruction: Path,
    *,
    max_expression_forward_mm: float = 0.5,
) -> dict[str, Any]:
    reconstruction = Path(reconstruction).resolve()
    mesh_dir = reconstruction / "meshes"
    fit_meta_path = mesh_dir / "stable_fit_meta.json"
    semantic_path = mesh_dir / "stable_semantic_regions.json"
    fit_meta = _read_json(fit_meta_path)
    semantic_payload = _read_json(semantic_path)
    regions = semantic_payload.get("regions")
    if not isinstance(regions, dict):
        raise ValueError(f"regions object is missing from {semantic_path}")

    optimized = (
        fit_meta.get("parameters", {})
        .get("optimized_parameters", {})
    )
    if not isinstance(optimized, dict):
        raise ValueError(f"optimized_parameters are missing from {fit_meta_path}")

    mica_shape = _parameter_array(
        optimized,
        "mica_identity_anchor",
        cfg.N_SHAPE_PARAMS,
    )
    optimized_shape = _parameter_array(
        optimized,
        "shape_params",
        cfg.N_SHAPE_PARAMS,
    )
    optimized_expression = _parameter_array(
        optimized,
        "expression_params",
        cfg.N_EXP_PARAMS,
    )

    flame = FLAMEModel(
        cfg.FLAME_MODEL_PATH,
        n_shape=cfg.N_SHAPE_PARAMS,
        n_exp=cfg.N_EXP_PARAMS,
    ).cpu()
    zero_shape = torch.zeros(cfg.N_SHAPE_PARAMS, dtype=torch.float32)
    zero_expression = torch.zeros(cfg.N_EXP_PARAMS, dtype=torch.float32)
    with torch.no_grad():
        stages = {
            "flame_mean": flame(zero_shape, zero_expression).cpu().numpy(),
            "mica_identity_anchor": flame(
                torch.from_numpy(mica_shape), zero_expression
            ).cpu().numpy(),
            "optimized_neutral": flame(
                torch.from_numpy(optimized_shape), zero_expression
            ).cpu().numpy(),
            "optimized_expression": flame(
                torch.from_numpy(optimized_shape),
                torch.from_numpy(optimized_expression),
            ).cpu().numpy(),
        }

    stage_report = build_profile_depth_stage_report(stages, regions)
    expression_gate = compare_expression_depth(
        stages["optimized_neutral"],
        stages["optimized_expression"],
        regions,
        max_mean_forward=max_expression_forward_mm,
    )

    exported = {}
    neutral_mesh_path = mesh_dir / "face_mesh_neutral.glb"
    expression_mesh_path = mesh_dir / "face_mesh.glb"
    if neutral_mesh_path.exists() and expression_mesh_path.exists():
        neutral_vertices = _load_mesh_vertices(neutral_mesh_path)
        expression_vertices = _load_mesh_vertices(expression_mesh_path)
        exported = {
            "neutral": profile_depth_metrics(neutral_vertices, regions),
            "expression": profile_depth_metrics(expression_vertices, regions),
            "expression_gate": compare_expression_depth(
                neutral_vertices,
                expression_vertices,
                regions,
                max_mean_forward=max_expression_forward_mm,
            ),
        }

    return {
        "audit_version": 1,
        "audit_only": True,
        "reconstruction": str(reconstruction),
        "source_paths": {
            "stable_fit_meta": str(fit_meta_path),
            "semantic_regions": str(semantic_path),
            "flame_model": str(cfg.FLAME_MODEL_PATH),
        },
        "semantic_source": semantic_payload.get("source"),
        "topology": semantic_payload.get("topology"),
        **stage_report,
        "expression_gate": expression_gate,
        "exported_mesh_verification": exported,
    }


def main() -> None:
    args = _parse_args()
    reconstruction = args.reconstruction.resolve()
    output_path = args.output
    if output_path is None:
        output_path = (
            reconstruction
            / "debug"
            / "profile_depth"
            / "profile_depth_baseline.json"
        )
    output_path = output_path.resolve()
    report = audit_profile_depth(
        reconstruction,
        max_expression_forward_mm=args.max_expression_forward_mm,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Profile depth audit: {output_path}")
    for name, metrics in report["stages"].items():
        print(
            f"  {name}: nose lead={metrics['nose_tip_minus_mouth']:.3f} mm, "
            f"mouth-chin={metrics['mouth_minus_chin']:.3f} mm"
        )
    gate = report["expression_gate"]
    print(
        "  expression mouth forward: "
        f"mean={gate['mouth_forward_mean']:.3f} mm, "
        f"p95={gate['mouth_forward_p95']:.3f} mm, "
        f"passed={gate['passed']}"
    )


if __name__ == "__main__":
    main()
