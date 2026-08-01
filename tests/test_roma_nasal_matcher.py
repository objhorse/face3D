from pathlib import Path

import numpy as np
import pytest

from src.geometry.roma_nasal_matcher import (
    NasalCropTransform,
    RoMaNasalSelectionConfig,
    RoMaNasalMatchBatch,
    build_square_nasal_crop,
    run_roma_nasal_pair,
)


def test_crop_transform_round_trip() -> None:
    transform = NasalCropTransform(
        work_size=(640, 480),
        bounds_xyxy=(200, 100, 400, 300),
    )
    crop = np.asarray([[0.5, 1.5], [199.0, 198.0]])
    assert np.allclose(transform.work_to_crop(transform.crop_to_work(crop)), crop)


def test_square_crop_contains_support() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    support = np.zeros((240, 320), dtype=bool)
    support[90:130, 140:180] = True
    crop, transform = build_square_nasal_crop(
        image,
        support,
        margin_px=20,
        minimum_size_px=96,
    )
    assert crop.shape == (96, 96, 3)
    x0, y0, x1, y1 = transform.bounds_xyxy
    assert x0 <= 140 and x1 >= 180
    assert y0 <= 90 and y1 >= 130


def test_match_batch_rejects_invalid_certainty() -> None:
    transform = NasalCropTransform((100, 100), (10, 10, 90, 90))
    with pytest.raises(ValueError, match="certainty"):
        RoMaNasalMatchBatch(
            front_pixels=np.asarray([[20.0, 20.0]]),
            side_pixels=np.asarray([[21.0, 20.0]]),
            certainty=np.asarray([1.1]),
            cycle_error_px=np.asarray([0.1]),
            front_crop=transform,
            side_crop=transform,
            metadata={},
        )


def test_selection_config_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="min_learned_certainty"):
        RoMaNasalSelectionConfig(min_learned_certainty=1.1)


def test_adapter_fails_before_subprocess_for_missing_worker() -> None:
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    support = np.zeros((160, 160), dtype=bool)
    support[50:110, 50:110] = True
    with pytest.raises(FileNotFoundError, match="worker"):
        run_roma_nasal_pair(
            image,
            image,
            support,
            support,
            python_executable=Path(__file__),
            worker_script=Path(__file__).with_name("missing_roma_worker.py"),
            torch_home=Path(__file__).parent,
        )
