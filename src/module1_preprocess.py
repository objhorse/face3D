import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PARSING_MODEL = None
_PARSING_DEVICE = None

try:
    from src import config as cfg
    TARGET_SIZE = int(getattr(cfg, "WORK_IMAGE_SIZE", 512))
except Exception:
    TARGET_SIZE = 512
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]


def _compute_resize_params(
    image_shape: Tuple[int, int],
    target: int,
) -> Tuple[float, int, int, int, int]:
    h, w = image_shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    x_off = (target - new_w) // 2
    y_off = (target - new_h) // 2
    return scale, new_w, new_h, x_off, y_off


def load_images(
    image_dir: Path,
    view_names: Dict[str, str],
) -> Dict[str, np.ndarray]:
    images = {}
    for view, filename in view_names.items():
        path = image_dir / filename
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        images[view] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logger.info(f"  [{view}] loaded {path.name}, size {img_bgr.shape[1]}x{img_bgr.shape[0]}")
    return images



def segment_face_black_bg(
    image: np.ndarray,
    threshold: int = 30,
    morph_kernel: int = 15,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = ((labels == largest) * 255).astype(np.uint8)
    return mask



def detect_landmarks_mediapipe(
    image: np.ndarray,
    view_name: str = "",
    min_detection_confidence: float = 0.3,
    min_tracking_confidence: float = 0.3,
    max_size: int = 960,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError("Please install mediapipe: pip install mediapipe")

    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        proc_img = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        proc_img = image
        scale = 1.0

    proc_h, proc_w = proc_img.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh:
        results = face_mesh.process(proc_img)

    if not results.multi_face_landmarks:
        logger.warning(f"  [{view_name}] MediaPipe found no face")
        return None, None

    face_lmks = results.multi_face_landmarks[0]
    landmarks_2d = np.array(
        [[lm.x * proc_w / scale, lm.y * proc_h / scale] for lm in face_lmks.landmark],
        dtype=np.float32,
    )
    visibility = np.array([lm.visibility for lm in face_lmks.landmark], dtype=np.float32)
    logger.info(f"  [{view_name}] detected {len(face_lmks.landmark)} landmarks (scale {scale:.2f})")
    return landmarks_2d, visibility


def _get_face_parsing_model():
    global _PARSING_MODEL, _PARSING_DEVICE
    if _PARSING_MODEL is not None:
        return _PARSING_MODEL

    try:
        import torch
        from facexlib.parsing import init_parsing_model
    except ImportError:
        logger.info("facexlib not available, skip model-based face parsing")
        return None

    _PARSING_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Initializing facexlib face parser on {_PARSING_DEVICE}...")
    _PARSING_MODEL = init_parsing_model(model_name='bisenet', device=_PARSING_DEVICE)
    return _PARSING_MODEL


def segment_face_labels_with_parser(image: np.ndarray) -> Optional[np.ndarray]:
    """Return the CelebAMask-HQ semantic label map at image resolution."""
    model = _get_face_parsing_model()
    if model is None:
        return None

    import torch
    from facexlib.utils import img2tensor
    from torchvision.transforms.functional import normalize

    face_input = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    face_input = img2tensor(face_input.astype('float32') / 255.0, bgr2rgb=False, float32=True)
    normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    face_input = torch.unsqueeze(face_input, 0).to(_PARSING_DEVICE)

    with torch.no_grad():
        out = model(face_input)[0]
    parsing = out.argmax(dim=1).squeeze().detach().cpu().numpy().astype(np.uint8)
    return cv2.resize(
        parsing,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def face_mask_from_parser_labels(parsing: np.ndarray) -> np.ndarray:
    """Convert CelebAMask-HQ labels into the existing face-only mask."""
    labels = np.asarray(parsing, dtype=np.uint8)

    # CelebAMask-HQ layout used by BiSeNet:
    # 0 bg, 1 skin, 2/3 brows, 4/5 eyes, 6 glasses, 7/8 ears, 9 ear_r,
    # 10 nose, 11 mouth, 12/13 lips, 14 neck, 15 neck_l, 16 cloth, 17 hair, 18 hat
    keep_labels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
    mask = np.isin(labels, list(keep_labels)).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    return mask


def segment_face_with_parser(image: np.ndarray) -> Optional[np.ndarray]:
    """Use facexlib BiSeNet to get a face-only mask."""
    parsing = segment_face_labels_with_parser(image)
    if parsing is None:
        return None
    return face_mask_from_parser_labels(parsing)



def create_face_mask_from_landmarks(
    landmarks_2d: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    h, w = image_shape[:2]
    hull_pts = landmarks_2d[FACE_OVAL].astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull_pts], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.dilate(mask, kernel)
    return mask


def create_face_shape_mask_from_landmarks(
    landmarks_2d: np.ndarray,
    image_shape: Tuple[int, int],
    dilate_px: int = 7,
) -> np.ndarray:
    h, w = image_shape[:2]
    hull_pts = landmarks_2d[FACE_OVAL].astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull_pts], 255)
    if dilate_px > 0:
        k = max(1, int(dilate_px))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)
    return mask



def create_landmark_hull_mask(
    landmarks_2d: np.ndarray,
    image_shape: Tuple[int, int],
    dilate_px: int = 21,
) -> np.ndarray:
    h, w = image_shape[:2]
    hull_pts = cv2.convexHull(landmarks_2d.astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull_pts, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    mask = cv2.dilate(mask, kernel)
    return mask



def refine_face_mask_grabcut(
    image: np.ndarray,
    landmarks_2d: np.ndarray,
    coarse_fg_mask: Optional[np.ndarray] = None,
    scale_divisor: int = 2,
) -> np.ndarray:
    h, w = image.shape[:2]
    face_oval = create_face_mask_from_landmarks(landmarks_2d, image.shape)
    hull_mask = create_landmark_hull_mask(landmarks_2d, image.shape)
    seed_mask = cv2.bitwise_or(face_oval, hull_mask)

    # Neck prior: remove anything clearly below the chin.
    chin_y = int(np.clip(landmarks_2d[152, 1], 0, h - 1))
    oval_pts = landmarks_2d[FACE_OVAL].astype(np.int32)
    oval_ymin = int(np.clip(oval_pts[:, 1].min(), 0, h - 1))
    oval_ymax = int(np.clip(oval_pts[:, 1].max(), 0, h - 1))
    face_h = max(oval_ymax - oval_ymin, 1)
    neck_cut_y = min(h, chin_y + max(4, int(0.04 * face_h)))

    if coarse_fg_mask is None:
        coarse_fg_mask = np.full((h, w), 255, dtype=np.uint8)
    coarse_fg_mask = coarse_fg_mask.copy()
    coarse_fg_mask[neck_cut_y:, :] = 0
    # Preserve parser-discovered side regions such as ears while still biasing
    # the optimization around the landmark-supported face core.
    coarse_support = cv2.morphologyEx(
        coarse_fg_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )

    sh = max(64, h // max(1, scale_divisor))
    sw = max(64, w // max(1, scale_divisor))
    img_small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (sw, sh))
    seed_small = cv2.resize(seed_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    coarse_small = cv2.resize(coarse_support, (sw, sh), interpolation=cv2.INTER_NEAREST)

    expanded = cv2.dilate(seed_small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    sure_fg = cv2.erode(seed_small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    gc_mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[coarse_small > 0] = cv2.GC_PR_FGD
    gc_mask[expanded > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(img_small, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        result_small = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except Exception as exc:
        logger.warning(f"GrabCut refinement failed ({exc}); falling back to seed mask")
        result_small = seed_small

    result = cv2.resize(result_small, (w, h), interpolation=cv2.INTER_LINEAR)
    result = (result > 127).astype(np.uint8) * 255
    # Keep the mask near the face region while still allowing the nose tip to protrude.
    guard_seed = cv2.dilate(
        seed_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    guard_parser = cv2.dilate(
        coarse_support,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    guard = cv2.bitwise_or(guard_seed, guard_parser)
    guard[neck_cut_y:, :] = 0
    result = cv2.bitwise_and(result, guard)
    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    result = cv2.morphologyEx(
        result,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    # Only keep the largest connected component to drop detached background blobs.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        result = ((labels == largest) * 255).astype(np.uint8)

    return result


def refine_face_mask_parser_primary(
    image: np.ndarray,
    landmarks_2d: np.ndarray,
    parser_mask: np.ndarray,
) -> np.ndarray:
    h, w = image.shape[:2]
    result = parser_mask.copy()

    chin_y = int(np.clip(landmarks_2d[152, 1], 0, h - 1))
    oval_pts = landmarks_2d[FACE_OVAL].astype(np.int32)
    oval_ymin = int(np.clip(oval_pts[:, 1].min(), 0, h - 1))
    oval_ymax = int(np.clip(oval_pts[:, 1].max(), 0, h - 1))
    face_h = max(oval_ymax - oval_ymin, 1)
    neck_cut_y = min(h, chin_y + max(3, int(0.03 * face_h)))
    result[neck_cut_y:, :] = 0

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    result = cv2.morphologyEx(
        result,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result)
    if num_labels > 1:
        face_center = landmarks_2d[[1, 4, 5, 195, 197]].mean(axis=0)
        best_label = 0
        best_score = None
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area <= 0:
                continue
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            ww = stats[label, cv2.CC_STAT_WIDTH]
            hh = stats[label, cv2.CC_STAT_HEIGHT]
            center = np.array([x + ww * 0.5, y + hh * 0.5], dtype=np.float32)
            dist = float(np.linalg.norm(center - face_center))
            score = dist - 0.0025 * area
            if best_score is None or score < best_score:
                best_score = score
                best_label = label
        if best_label > 0:
            result = ((labels == best_label) * 255).astype(np.uint8)
    return result



def _resize_to_target(image: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target and w == target:
        return image
    _, new_w, new_h, x_off, y_off = _compute_resize_params(image.shape, target)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas



def preprocess_all_views(
    images: Dict[str, np.ndarray],
    debug_dir: Optional[Path] = None,
    target_size: int = TARGET_SIZE,
) -> Dict[str, dict]:
    results = {}
    for view_name, image in images.items():
        logger.info(f"Preprocess {view_name}, original size {image.shape[1]}x{image.shape[0]}")
        original_image = image
        scale, _, _, x_off, y_off = _compute_resize_params(original_image.shape, target_size)

        # Detect on the original image, then map to the working canvas.
        lmks_orig, vis = detect_landmarks_mediapipe(original_image, view_name)

        image = _resize_to_target(original_image, target_size)
        bg_mask = segment_face_black_bg(image)
        parser_labels = segment_face_labels_with_parser(image)
        parser_mask = face_mask_from_parser_labels(parser_labels) if parser_labels is not None else None
        if parser_mask is not None:
            bg_mask = parser_mask

        lmks = None
        if lmks_orig is not None:
            lmks = lmks_orig.copy()
            lmks[:, 0] = lmks[:, 0] * scale + x_off
            lmks[:, 1] = lmks[:, 1] * scale + y_off

        if lmks is not None:
            shape_mask = create_face_shape_mask_from_landmarks(lmks, image.shape)
            if parser_mask is not None and view_name in {"left", "right"}:
                face_mask = refine_face_mask_parser_primary(image, lmks, parser_mask)
            else:
                face_mask = refine_face_mask_grabcut(image, lmks, coarse_fg_mask=bg_mask)
        else:
            face_mask = bg_mask.copy()
            shape_mask = face_mask.copy()

        results[view_name] = {
            'image': image,
            'landmarks': lmks,
            'visibility': vis,
            'face_mask': face_mask,
            'shape_mask': shape_mask,
            'bg_mask': bg_mask,
            'parser_mask': parser_mask,
            'parser_labels': parser_labels,
            'nose_mask': (
                (parser_labels == 10).astype(np.uint8) * 255
                if parser_labels is not None
                else np.zeros(image.shape[:2], dtype=np.uint8)
            ),
        }

        if debug_dir is not None:
            _save_debug_images(view_name, image, lmks, face_mask, shape_mask, bg_mask, parser_mask, debug_dir)

    return results



def _save_debug_images(
    view_name: str,
    image: np.ndarray,
    landmarks: Optional[np.ndarray],
    face_mask: np.ndarray,
    shape_mask: np.ndarray,
    bg_mask: np.ndarray,
    parser_mask: Optional[np.ndarray],
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(debug_dir / f'{view_name}_face_mask.png'), face_mask)
    cv2.imwrite(str(debug_dir / f'{view_name}_shape_mask.png'), shape_mask)
    cv2.imwrite(str(debug_dir / f'{view_name}_bg_mask.png'), bg_mask)
    if parser_mask is not None:
        cv2.imwrite(str(debug_dir / f'{view_name}_parser_mask.png'), parser_mask)

    if landmarks is not None:
        vis_img = img_bgr.copy()
        for pt in landmarks.astype(int):
            cv2.circle(vis_img, tuple(pt), 1, (0, 255, 0), -1)
        cv2.imwrite(str(debug_dir / f'{view_name}_landmarks.png'), vis_img)

    overlay = img_bgr.copy()
    overlay[face_mask == 0] = (overlay[face_mask == 0] * 0.3).astype(np.uint8)
    cv2.imwrite(str(debug_dir / f'{view_name}_masked.png'), overlay)
