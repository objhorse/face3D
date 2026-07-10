"""Run the stable three-view pipeline on the configured local capture."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from src import config as cfg
    from src.pipeline.stable_three_view import run_stable_three_view_pipeline

    cfg.ensure_dirs()
    image_paths = {
        view: cfg.DEFAULT_IMAGE_DIR / filename
        for view, filename in cfg.DEFAULT_VIEW_NAMES.items()
    }

    def progress(stage: str, pct: int, message: str) -> None:
        logger.info("[%s %s%%] %s", stage, pct, message)

    glb_path = run_stable_three_view_pipeline(
        session_id=None,
        patient_id="local_capture",
        image_paths=image_paths,
        session_output_dir=cfg.OUTPUT_DIR,
        manual_intrinsics=cfg.MANUAL_INTRINSICS,
        progress=progress,
    )
    logger.info("Stable model: %s", glb_path)
    logger.info("Report: %s", cfg.OUTPUT_DEBUG_DIR / "stable_pipeline" / "index.html")


if __name__ == "__main__":
    main()

