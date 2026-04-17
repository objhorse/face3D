"""
EMOCA per-view initializer (expression + pose).

EMOCA (Emotion-Aware DECA) is an extension of DECA with improved expression
and emotion disentanglement. It outputs the same FLAME parameter space.
Reference: https://github.com/radekd91/emoca

Expected repo layout:
  external/EMOCA/
    gdl/models/EMOCA.py  (or similar)

Expected checkpoint dir:
  models/EMOCA/EMOCA_v2_detail_EmotionMW_IDW_0.1/

If EMOCA is unavailable, returns None (caller falls back to face_alignment).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_emoca_per_view(
    images: Dict[str, np.ndarray],
    emoca_dir: Optional[Path],
    emoca_checkpoint,
    device: str,
    n_exp: int = 50,
) -> Optional[Dict[str, Optional[dict]]]:
    """
    Run EMOCA on each view.
    Returns {view: {"exp":(n_exp,), "cam":(3,)}} or None if unavailable.
    """
    if emoca_dir is None or not emoca_dir.exists():
        logger.warning(
            "EMOCA 目录不存在，跳过 EMOCA 初始化。\n"
            f"  期望路径: {emoca_dir}\n"
            "  请克隆 https://github.com/radekd91/emoca 到 external/EMOCA/"
        )
        return None

    emoca_str = str(emoca_dir)
    if emoca_str not in sys.path:
        sys.path.insert(0, emoca_str)

    try:
        model = _load_emoca(emoca_dir, emoca_checkpoint, device)
    except Exception as e:
        logger.warning(f"EMOCA 加载失败: {e}")
        return None

    results = {}
    for view_name, img in images.items():
        try:
            params = _run_emoca_single(model, img, device, n_exp)
            results[view_name] = params
            logger.info(f"  [{view_name}] EMOCA 成功")
        except Exception as e:
            logger.warning(f"  [{view_name}] EMOCA 推理失败: {e}")
            results[view_name] = None

    if not any(v is not None for v in results.values()):
        return None
    return results


def _load_emoca(emoca_dir: Path, checkpoint, device: str):
    """Load EMOCA model. Supports gdl-based EMOCA v2 API."""
    try:
        # EMOCA v2 uses gdl package
        from gdl.models.EMOCA import EMOCA
        import torch
        from omegaconf import OmegaConf

        cfg_path = Path(checkpoint) / "cfg.yaml" if checkpoint else None
        if cfg_path and cfg_path.exists():
            cfg = OmegaConf.load(str(cfg_path))
        else:
            raise FileNotFoundError(f"EMOCA cfg.yaml 不存在: {cfg_path}")

        model = EMOCA(cfg.model, device)
        ckpt_file = Path(checkpoint) / "model.ckpt"
        if ckpt_file.exists():
            ckpt = torch.load(str(ckpt_file), map_location=device)
            state = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state, strict=False)
        model.to(device).eval()
        logger.info(f"EMOCA 加载成功: {checkpoint}")
        return model

    except ImportError as e:
        raise ImportError(
            f"无法导入 EMOCA/gdl ({e})。请确认：\n"
            "  1. external/EMOCA/ 已克隆（https://github.com/radekd91/emoca）\n"
            "  2. EMOCA 依赖已安装（pip install -e external/EMOCA/）\n"
            "  3. models/EMOCA/ 权重目录存在"
        )


def _run_emoca_single(model, img: np.ndarray, device: str, n_exp: int) -> dict:
    import torch

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_224 = cv2.resize(img_bgr, (224, 224))
    img_t = torch.tensor(img_224).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_t = img_t.to(device)

    with torch.no_grad():
        batch = {"image": img_t}
        output = model.encode(batch)

    exp = output.get("expcode", output.get("exp", None))
    cam = output.get("cam", None)

    if exp is None:
        raise RuntimeError(f"EMOCA 输出未包含 expcode/exp，keys={list(output.keys())}")

    exp_np = exp.squeeze(0).cpu().numpy().flatten()[:n_exp]
    cam_np = cam.squeeze(0).cpu().numpy().flatten() if cam is not None else np.array([1.0, 0.0, 0.0])

    return {"exp": exp_np.astype(np.float32), "cam": cam_np.astype(np.float32)}
