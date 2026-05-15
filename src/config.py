"""
全局配置。所有路径、超参数集中管理。
修改内参时只需改 MANUAL_INTRINSICS 字段。
"""
import os
from pathlib import Path

# ── 根目录 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # D:/face3D

# ── 模型权重路径 ────────────────────────────────────────────────────────
MODELS_DIR          = ROOT / "models"
FLAME_MODEL_PATH    = MODELS_DIR / "FLAME" / "generic_model.pkl"
FLAME_LANDMARK_PATH = MODELS_DIR / "FLAME" / "landmark_embedding.npy"  # FLAME→68点映射
DECA_MODEL_PATH     = MODELS_DIR / "DECA" / "deca_model.tar"
MICA_MODEL_PATH     = MODELS_DIR / "MICA" / "mica.tar"
EMOCA_MODEL_PATH    = MODELS_DIR / "EMOCA" / "EMOCA_v2_detail_EmotionMW_IDW_0.1"
DEPTH_MODEL_DIR     = MODELS_DIR / "depth_anything"

# ── 外部仓库路径 ────────────────────────────────────────────────────────
EXTERNAL_DIR   = ROOT / "external"
DECA_REPO_DIR  = EXTERNAL_DIR / "DECA"
DUST3R_DIR     = EXTERNAL_DIR / "dust3r"
MICA_REPO_DIR  = EXTERNAL_DIR / "MICA"
EMOCA_REPO_DIR = EXTERNAL_DIR / "EMOCA"

# ── 输出目录 ────────────────────────────────────────────────────────────
OUTPUT_DIR          = ROOT / "output"
OUTPUT_MESH_DIR     = OUTPUT_DIR / "meshes"
OUTPUT_TEXTURE_DIR  = OUTPUT_DIR / "textures"
OUTPUT_DEBUG_DIR    = OUTPUT_DIR / "debug"
CAMERA_CALIBRATION_PATH = ROOT / "config" / "camera_calibration.json"
WORK_IMAGE_SIZE = 1024
USE_CAMERA_CALIBRATION = True
UNDISTORT_IMAGES = True
UNDISTORT_ALPHA = 0.0
ENABLE_UV_HOLE_FILL_FACES = True
ENABLE_VISIBLE_FACE_CROP = True
VISIBLE_FACE_CROP_DILATE_RINGS = 0
VISIBLE_FACE_CROP_Z_TOL = 0.006
FORCE_CLOSED_EYES = True
CLOSED_EYE_GEOMETRY_STRENGTH = 0.95
CLOSED_EYE_LOSS_WEIGHT = 0.0
TEST_DATA_DIR       = OUTPUT_DIR / "test_data"   # generate_test_data.py 输出

# ── 输入图像命名约定 ─────────────────────────────────────────────────────
# key: 视角名, value: 文件名（相对于图像目录）
DEFAULT_VIEW_NAMES = {
    "left":   "camera1_20260511_191357.jpg",
    "front":  "camera2_20260511_191357.jpg",
    "right":  "camera3_20260511_191357.jpg",
}
DEFAULT_IMAGE_DIR = ROOT / "new_captures" / "zyz_captures"

# ── 相机内参（手动覆盖接口） ──────────────────────────────────────────────
# 若为 None，则由 Dust3R 预测；若设置则跳过 Dust3R 直接使用
# 格式: {"fx": float, "fy": float, "cx": float, "cy": float}
# 或直接传 3×3 numpy array
MANUAL_INTRINSICS = None
# vivo X300 主摄 1x，FocalLength=6.25mm，FocalLengthIn35mmFilm=23mm，3072×4080
# 推导：crop=3.68，pixel_pitch=0.002302mm，fx_original=2715px，缩放512后fx=341

# ── 初始化后端 ────────────────────────────────────────────────────────
# "face_alignment" : 仅用 face_alignment 68点（原有方案）
# "deca"           : 仅用 DECA 估计初值（原有方案）
# "mica_deca"      : MICA 估计共享身份形状 + DECA 估计每视角表情/姿态（推荐）
# "mica_emoca"     : MICA 估计共享身份形状 + EMOCA 估计每视角表情/姿态
INIT_BACKEND = "mica_deca"

# ── MICA 配置 ─────────────────────────────────────────────────────────
# MICA 权重目录（需下载 https://github.com/Zielon/MICA）
MICA_DIR        = MICA_REPO_DIR
MICA_CHECKPOINT = MICA_MODEL_PATH

# ── EMOCA 配置 ────────────────────────────────────────────────────────
# EMOCA 仓库与权重（https://github.com/radekd91/emoca）
EMOCA_DIR        = EMOCA_REPO_DIR
EMOCA_CHECKPOINT = EMOCA_MODEL_PATH

# ── DECA 配置 ─────────────────────────────────────────────────────────
DECA_DIR = DECA_REPO_DIR

# ── 3DMM 超参数 ────────────────────────────────────────────────────────
N_SHAPE_PARAMS  = 300   # FLAME 形状参数维度（最大300）
N_EXP_PARAMS    = 100   # FLAME 表情参数维度（最大100）
LAMBDA_SHAPE    = 5e-4  # 形状正则化权重（降低=更贴合真实脸型）
LAMBDA_EXP      = 5e-4  # 表情正则化权重
LBFGS_MAX_ITER  = 100   # L-BFGS 最大迭代次数（提升拟合精度）
LBFGS_LR        = 0.05  # 小学习率防止 NaN

# ── 深度估计超参数 ──────────────────────────────────────────────────────
DEPTH_MODEL_SIZE = "large"       # "small"/"base"/"large"
DISPLACEMENT_SCALE = 0.002         # 顶点置换最大幅度（米）

# ── 硬件 ────────────────────────────────────────────────────────────────
DEVICE = "cuda"   # "cuda" or "cpu"

# ── 光照图层配置（多光照预留接口） ──────────────────────────────────────
LIGHTING_TYPES = {
    "white": {
        "display_name": "白光",
        "image_dir": str(DEFAULT_IMAGE_DIR),
        "views": DEFAULT_VIEW_NAMES,
    },
    # 预留接口示例（后续扩展）:
    # "uv": {
    #     "display_name": "UV光",
    #     "image_dir": str(ROOT / "image_uv"),
    #     "views": {...},
    # },
}
DEFAULT_LIGHTING = "white"

# ── 工具函数 ────────────────────────────────────────────────────────────
def ensure_dirs():
    """确保所有输出目录存在"""
    for d in [OUTPUT_MESH_DIR, OUTPUT_TEXTURE_DIR, OUTPUT_DEBUG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
