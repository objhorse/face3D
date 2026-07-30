import numpy as np

from src.appearance.stable_texture import _build_strict_feature_masks


def test_strict_feature_masks_preserve_central_face_parser_labels():
    labels = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
        ],
        dtype=np.uint8,
    )

    masks = _build_strict_feature_masks(
        {
            "front": {"parser_labels": labels},
            "left": {"parser_labels": labels},
        }
    )

    expected = np.isin(labels, [2, 3, 4, 5, 6, 10])
    np.testing.assert_array_equal(masks["front"] > 0, expected)
    np.testing.assert_array_equal(masks["left"] > 0, expected)


def test_strict_feature_masks_fall_back_to_mediapipe_landmarks():
    landmarks = np.zeros((468, 2), dtype=np.float32)
    landmark_indices = [
        71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
        168, 197, 5, 4, 75, 97, 2, 326, 305,
        33, 160, 158, 133, 153, 144,
        362, 385, 387, 263, 373, 380,
        61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
        78, 82, 13, 312, 308, 317, 14, 87,
    ]
    xs = np.linspace(24.0, 72.0, len(landmark_indices), dtype=np.float32)
    ys = 48.0 + 12.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, len(landmark_indices)))
    landmarks[landmark_indices] = np.column_stack((xs, ys))

    masks = _build_strict_feature_masks(
        {
            "front": {
                "parser_labels": None,
                "landmarks": landmarks,
                "image": np.zeros((96, 96, 3), dtype=np.uint8),
            },
        }
    )

    assert "front" in masks
    assert masks["front"].shape == (96, 96)
    assert int(np.count_nonzero(masks["front"])) > 0


def test_strict_feature_masks_skip_views_without_semantic_observations():
    masks = _build_strict_feature_masks({"front": {"parser_labels": None}})

    assert masks == {}


def test_landmark_fallback_leaves_mouth_unowned_without_mesh_semantics():
    landmarks = np.zeros((468, 2), dtype=np.float32)
    points_68 = np.full((68, 2), [48.0, 30.0], dtype=np.float32)
    points_68[17:27] = np.array(
        [[24, 28], [30, 26], [36, 26], [42, 28], [45, 30],
         [51, 30], [54, 28], [60, 26], [66, 26], [72, 28]],
        dtype=np.float32,
    )
    points_68[27:36] = np.array(
        [[48, 30], [48, 36], [48, 42], [48, 48], [38, 51],
         [43, 54], [48, 55], [53, 54], [58, 51]],
        dtype=np.float32,
    )
    points_68[36:48] = np.array(
        [[28, 38], [32, 36], [36, 36], [40, 38], [36, 40], [32, 40],
         [56, 38], [60, 36], [64, 36], [68, 38], [64, 40], [60, 40]],
        dtype=np.float32,
    )
    points_68[48:60] = np.array(
        [[30, 64], [35, 60], [42, 58], [48, 58], [54, 58], [61, 60],
         [66, 64], [61, 68], [54, 70], [48, 70], [42, 70], [35, 68]],
        dtype=np.float32,
    )
    points_68[60:68] = np.array(
        [[38, 64], [43, 62], [48, 62], [53, 62],
         [58, 64], [53, 66], [48, 66], [43, 66]],
        dtype=np.float32,
    )
    landmarks[[
        162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288,
        323, 454, 389, 71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
        168, 197, 5, 4, 75, 97, 2, 326, 305,
        33, 160, 158, 133, 153, 144,
        362, 385, 387, 263, 373, 380,
        61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
        78, 82, 13, 312, 308, 317, 14, 87,
    ]] = points_68

    masks = _build_strict_feature_masks(
        {
            "front": {
                "parser_labels": None,
                "landmarks": landmarks,
                "image": np.zeros((96, 96, 3), dtype=np.uint8),
            }
        }
    )

    assert masks["front"][64, 48] == 0
    assert masks["front"][59, 48] == 0
