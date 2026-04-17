from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_deca_per_view(
    images: Dict[str, np.ndarray],
    deca_dir: Optional[Path],
    device: str,
    n_exp: int = 50,
    deca_checkpoint: Optional[Path] = None,
) -> Optional[Dict[str, Optional[dict]]]:
    """
    Run DECA on each view.

    Preferred path:
    - instantiate official DECA class and call encode()

    Fallback path:
    - if DECA full init fails (commonly because pytorch3d is unavailable),
      load only the E_flame encoder from deca_model.tar and decode the packed
      parameter vector manually. This is sufficient for shape/exp/cam init.
    """
    if deca_dir is None or not deca_dir.exists():
        logger.info("DECA directory missing, skip DECA initializer")
        return None

    deca_str = str(deca_dir)
    if deca_str not in sys.path:
        sys.path.insert(0, deca_str)

    try:
        import torch
        from decalib.deca import DECA
        from decalib.utils.config import get_cfg_defaults
    except ImportError as e:
        logger.info(f"DECA unavailable ({e}), skip")
        return None

    # Use a fresh clone to avoid mutating the shared module-level singleton,
    # which would be a data race in concurrent server environments.
    deca_cfg = get_cfg_defaults()
    deca_cfg.model.use_tex = False

    # Override checkpoint path with our custom location before model instantiation.
    # deca_cfg.pretrained_modelpath is read by DECA.__init__() to load weights.
    if deca_checkpoint is not None:
        deca_cfg.pretrained_modelpath = str(deca_checkpoint)

    deca = None
    encoder_only = False

    try:
        deca = DECA(config=deca_cfg, device=device)
        logger.info("DECA full model initialized")
    except Exception as e:
        logger.warning(f"DECA full init failed, trying encoder-only fallback: {e}")
        encoder_only = True
        try:
            deca = _load_deca_encoder_only(device, deca_cfg)
            logger.info("DECA encoder-only fallback initialized")
        except Exception as fallback_e:
            logger.warning(f"DECA encoder-only fallback failed: {fallback_e}")
            return None

    results = {}
    for view_name, img in images.items():
        try:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_224 = cv2.resize(img_bgr, (224, 224))
            img_t = torch.tensor(img_224).permute(2, 0, 1).float() / 255.0
            img_t = img_t.unsqueeze(0).to(device)
            with torch.no_grad():
                codedict = _encode_deca_tensor(deca, deca_cfg, img_t, encoder_only)
            results[view_name] = {
                "shape": codedict["shape"].squeeze(0).cpu().numpy(),
                "exp": codedict["exp"].squeeze(0).cpu().numpy()[:n_exp],
                "cam": codedict["cam"].squeeze(0).cpu().numpy(),
            }
            logger.info(f"  [{view_name}] DECA success ({'encoder-only' if encoder_only else 'full'})")
        except Exception as e:
            logger.warning(f"  [{view_name}] DECA inference failed: {e}")
            results[view_name] = None

    if not any(v is not None for v in results.values()):
        return None
    return results


def _load_deca_encoder_only(device: str, deca_cfg):
    import torch
    from decalib.models.encoders import ResnetEncoder

    model_path = Path(deca_cfg.pretrained_modelpath)
    if not model_path.exists():
        raise FileNotFoundError(f"DECA checkpoint not found: {model_path}")

    n_param = (
        deca_cfg.model.n_shape
        + deca_cfg.model.n_tex
        + deca_cfg.model.n_exp
        + deca_cfg.model.n_pose
        + deca_cfg.model.n_cam
        + deca_cfg.model.n_light
    )
    encoder = ResnetEncoder(outsize=n_param).to(device)
    checkpoint = torch.load(str(model_path), map_location=device)
    state = checkpoint.get("E_flame", None)
    if state is None:
        raise KeyError("E_flame missing in DECA checkpoint")
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    return encoder


def _encode_deca_tensor(deca_model, deca_cfg, img_t, encoder_only: bool):
    if not encoder_only:
        return deca_model.encode(img_t)

    parameters = deca_model(img_t)
    codedict = {}
    start = 0
    for key in deca_cfg.model.param_list:
        n_key = int(deca_cfg.model.get("n_" + key))
        end = start + n_key
        chunk = parameters[:, start:end]
        if key == "light":
            chunk = chunk.reshape(chunk.shape[0], 9, 3)
        codedict[key] = chunk
        start = end
    return codedict
