from pathlib import Path

import numpy as np
import torch

from src.geometry.expression_fidelity import (
    feature_gap_diagnostics,
    mediapipe_expression_state,
    paired_vertical_gap_loss,
)
from src.geometry.template_fit import stable_neutral_source
from src.module3_texture import texture_alpha_masks, transparent_bottom_face_mask


def test_feature_gap_loss_penalizes_open_geometry_against_closed_observation():
    target = torch.zeros((68, 2), dtype=torch.float32)
    projected = target.clone()
    projected[61, 1] = -0.01
    projected[67, 1] = 0.01

    loss = paired_vertical_gap_loss(projected, target, ((61, 67),))

    assert float(loss) > 0.0


def test_feature_gap_diagnostics_report_eye_and_mouth_mismatch():
    target = np.zeros((68, 2), dtype=np.float32)
    projected = target.copy()
    projected[37, 1] = -4.0
    projected[41, 1] = 4.0
    projected[61, 1] = -6.0
    projected[67, 1] = 6.0

    report = feature_gap_diagnostics(projected, target)

    assert report["eye_gap_error_px"] > 0.0
    assert report["mouth_gap_error_px"] > 0.0


def test_mediapipe_expression_state_detects_closed_eye_and_mouth_gaps():
    points = np.zeros((478, 2), dtype=np.float32)
    points[33] = (0.0, 0.0)
    points[133] = (10.0, 0.0)
    points[362] = (20.0, 0.0)
    points[263] = (30.0, 0.0)
    points[160] = points[144] = (3.0, 0.0)
    points[158] = points[153] = (7.0, 0.0)
    points[385] = points[380] = (23.0, 0.0)
    points[387] = points[373] = (27.0, 0.0)
    points[78] = (0.0, 20.0)
    points[308] = (20.0, 20.0)
    points[13] = points[14] = (10.0, 20.0)

    state = mediapipe_expression_state(points)

    assert state["closed_eyes"] is True
    assert state["closed_mouth"] is True


def test_inpainted_valid_uv_pixels_are_opaque_but_remain_low_confidence():
    valid = np.ones((9, 9), dtype=bool)
    observed = np.ones((9, 9), dtype=bool)
    observed[4, 4] = False
    geometry_keep = np.ones((9, 9), dtype=bool)
    geometry_keep[8, :] = False

    alpha, confidence = texture_alpha_masks(valid, observed, geometry_keep)

    assert alpha[4, 4] == 255
    assert confidence[4, 4] == 0
    assert np.all(alpha[8, :] == 0)


def test_stable_neutral_source_requires_true_neutral_export():
    fixture_dir = Path(__file__).parent / "fixtures" / "reconstruction_recovery"
    neutral_mesh = fixture_dir / "face_mesh_neutral.glb"

    assert stable_neutral_source(fixture_dir) == neutral_mesh


def test_transparent_bottom_split_preserves_faces_and_selects_only_low_region():
    vertices = np.array(
        [[0.0, -2.0, 0.0], [1.0, -2.0, 0.0], [0.0, -2.0, 0.0],
         [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    hidden, y_floor = transparent_bottom_face_mask(vertices, faces, 0.25)

    assert y_floor < 0.0
    assert hidden.tolist() == [True, False]
    assert len(hidden) == len(faces)
