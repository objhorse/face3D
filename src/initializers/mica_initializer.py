"""
MICA identity shape initializer.

Uses MICA (Zielon/MICA) official API:
  1. insightface RetinaFace → detect face + 5 keypoints
  2. insightface face_align.norm_crop → 112x112 ArcFace input + 224x224 face image
  3. mica.encode(images, arcface) → codedict
  4. mica.decode(codedict) → pred_shape_code (FLAME shape params)

Repo: external/MICA/
Checkpoint: external/MICA/data/pretrained/mica.tar
InsightFace models: ~/.insightface/models/antelopev2/
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# insightface model root (antelopev2 pack required)
_INSIGHTFACE_ROOT = str(Path.home() / ".insightface")


def get_mica_shape(
    images: Dict[str, np.ndarray],
    mica_dir: Optional[Path],
    mica_checkpoint: Optional[Path],
    device: str,
    n_shape: int,
) -> Tuple[np.ndarray, bool]:
    """
    Estimate shared FLAME identity shape from all views via MICA.

    Returns:
        shape_params: (n_shape,) float32
        success: bool
    """
    if mica_dir is None or not mica_dir.exists():
        logger.warning(
            f"MICA 目录不存在: {mica_dir}\n"
            "  请克隆 https://github.com/Zielon/MICA 到 external/MICA/"
        )
        return np.zeros(n_shape, dtype=np.float32), False

    mica_str = str(mica_dir)
    if mica_str not in sys.path:
        sys.path.insert(0, mica_str)

    try:
        model, app = _load_mica_and_detector(mica_dir, device, mica_checkpoint)
    except Exception as e:
        logger.warning(f"MICA 加载失败: {e}\n使用零初始化")
        return np.zeros(n_shape, dtype=np.float32), False

    shape_candidates = []
    for view_name, img in images.items():
        try:
            shape_v = _run_mica_single(model, app, img, device, n_shape)
            shape_candidates.append(shape_v)
            logger.info(f"  [{view_name}] MICA 身份形状估计成功")
        except Exception as e:
            logger.warning(f"  [{view_name}] MICA 推理失败: {e}")

    if not shape_candidates:
        logger.warning("所有视角 MICA 均失败，使用零初始化")
        return np.zeros(n_shape, dtype=np.float32), False

    shape_fused = np.mean(shape_candidates, axis=0).astype(np.float32)
    logger.info(
        f"MICA 共享形状融合: {len(shape_candidates)} 视角均值, "
        f"norm={float(np.linalg.norm(shape_fused)):.4f}"
    )
    return shape_fused, True


def _load_mica_and_detector(mica_dir: Path, device: str, mica_checkpoint=None):
    """Load MICA model + insightface detector."""
    import torch
    from insightface.app import FaceAnalysis

    # numpy 2.x compat patch (chumpy)
    import numpy as np
    for alias, target in [('bool', 'bool_'), ('int', 'int_'), ('float', 'float64'),
                           ('complex', 'complex128'), ('object', 'object_'),
                           ('str', 'str_'), ('unicode', 'str_')]:
        if not hasattr(np, alias):
            setattr(np, alias, getattr(np, target))

    from configs.config import get_cfg_defaults
    from utils import util as mica_util

    cfg = get_cfg_defaults()
    cfg.model.testing = True

    # Override checkpoint path with our custom location before model instantiation.
    # cfg.pretrained_model_path is read by load_model() inside MICA.__init__().
    if mica_checkpoint is not None:
        cfg.pretrained_model_path = str(mica_checkpoint)

    mica_device = "cuda:0" if device == "cuda" else "cpu"
    model = mica_util.find_model_using_name(
        model_dir="micalib.models", model_name=cfg.model.name
    )(cfg, mica_device)
    model.eval()

    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="antelopev2", root=_INSIGHTFACE_ROOT, providers=providers)
    app.prepare(ctx_id=0, det_size=(224, 224))

    return model, app


def _run_mica_single(model, app, img_rgb: np.ndarray, device: str, n_shape: int) -> np.ndarray:
    """Run MICA on a single RGB image → shape params (n_shape,)."""
    import torch
    from insightface.app.common import Face
    from insightface.utils import face_align
    from datasets.creation.util import get_arcface_input, get_center

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    bboxes, kpss = app.det_model.detect(img_bgr, max_num=0, metric="default")

    if bboxes.shape[0] == 0:
        raise RuntimeError("MICA: 未检测到人脸")

    i = get_center(bboxes, img_bgr)
    bbox = bboxes[i, 0:4]
    kps = kpss[i] if kpss is not None else None
    face = Face(bbox=bbox, kps=kps, det_score=bboxes[i, 4])

    # ArcFace input (112x112 blob)
    blob, _ = get_arcface_input(face, img_bgr)  # blob: (3, 112, 112) float32
    arcface_t = torch.tensor(blob).unsqueeze(0).to(device if device != "cuda" else "cuda:0")

    # Face image for encoder (224x224)
    aimg = face_align.norm_crop(img_bgr, landmark=face.kps, image_size=224)
    img_t = torch.tensor(aimg.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    img_t = img_t.to(device if device != "cuda" else "cuda:0")

    with torch.no_grad():
        codedict = model.encode(img_t, arcface_t)
        opdict = model.decode(codedict)

    shape_np = opdict["pred_shape_code"].squeeze(0).cpu().numpy().flatten()

    if len(shape_np) >= n_shape:
        return shape_np[:n_shape].astype(np.float32)
    out = np.zeros(n_shape, dtype=np.float32)
    out[:len(shape_np)] = shape_np
    return out
