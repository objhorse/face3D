"""
FastAPI 服务层

端点：
  POST   /api/sessions              — 上传图片，创建重建会话
  GET    /api/sessions              — 列出所有会话（按时间倒序）
  GET    /api/sessions/{id}         — 查询单个会话详情
  GET    /api/sessions/{id}/model   — 下载 GLB 文件
  DELETE /api/sessions/{id}         — 删除会话及其文件

  POST   /api/calibration           — 提交手动相机内参（可选）
  POST   /api/compare               — ICP 对齐，返回变换矩阵

  WS     /ws/{session_id}           — 进度推送（JSON 消息流）
"""
import asyncio
import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Form, HTTPException,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_db, init_db, SessionLocal
from .models import Session as SessionModel
from .pipeline_runner import run_pipeline_async

logger = logging.getLogger(__name__)

# ── 输出目录 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
SESSIONS_DIR = ROOT / "output" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── 手动内参存储（进程内，重启后丢失）────────────────────────────────────────
_manual_intrinsics: Optional[dict] = None

# ── 进度队列：session_id → asyncio.Queue ─────────────────────────────────────
_progress_queues: Dict[int, asyncio.Queue] = {}

# ── WebSocket 连接管理 ────────────────────────────────────────────────────────
_ws_connections: Dict[int, List[WebSocket]] = {}


# ── 应用生命周期 ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("数据库初始化完成")
    yield


app = FastAPI(
    title="face3D API",
    description="3D 人脸重建服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务：前端
_frontend_dir = ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic 响应模型
# ══════════════════════════════════════════════════════════════════════════════

class SessionOut(BaseModel):
    id:          int
    patient_id:  str
    notes:       Optional[str]
    status:      str
    created_at:  datetime
    finished_at: Optional[datetime]
    has_model:   bool

    model_config = {"from_attributes": True}


class CompareRequest(BaseModel):
    session_id_a: int   # 参考（术前）
    session_id_b: int   # 对齐目标（术后）


class CalibrationRequest(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _session_dir(session_id: int) -> Path:
    return SESSIONS_DIR / str(session_id)


async def _get_session_or_404(session_id: int, db: AsyncSession) -> SessionModel:
    row = await db.get(SessionModel, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return row


async def _push_progress(session_id: int, queue: asyncio.Queue):
    """从队列读取进度并推送给所有已连接的 WebSocket 客户端"""
    while True:
        msg = await queue.get()
        connections = _ws_connections.get(session_id, [])
        dead = []
        for ws in connections:
            try:
                await ws.send_text(json.dumps(msg, ensure_ascii=False))
            except Exception:
                dead.append(ws)
        for d in dead:
            connections.remove(d)
        if msg.get("stage") in ("done", "error"):
            break


# ══════════════════════════════════════════════════════════════════════════════
# REST 端点：会话管理
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/sessions", response_model=SessionOut, status_code=201)
async def create_session(
    background_tasks: BackgroundTasks,
    patient_id:  str        = Form(...),
    notes:       str        = Form(""),
    left_image:  UploadFile = File(...),
    front_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    上传三张图片（左/正/右），创建重建会话。
    立即返回 session_id，重建在后台异步执行。
    通过 WebSocket /ws/{session_id} 接收进度。
    """
    # 创建数据库记录
    sess = SessionModel(
        patient_id=patient_id,
        notes=notes or None,
        status="pending",
    )
    db.add(sess)
    await db.commit()
    await db.refresh(sess)
    session_id = sess.id

    # 保存上传图片到会话目录
    img_dir = _session_dir(session_id) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    image_paths = {}
    for view_name, upload in [
        ("left", left_image),
        ("front", front_image),
        ("right", right_image),
    ]:
        suffix = Path(upload.filename).suffix or ".jpg"
        dest = img_dir / f"{view_name}{suffix}"
        content = await upload.read()
        with open(dest, "wb") as f:
            f.write(content)
        image_paths[view_name] = dest

    # 更新状态
    sess.status = "running"
    await db.commit()

    # 创建进度队列
    queue: asyncio.Queue = asyncio.Queue()
    _progress_queues[session_id] = queue

    # 用 asyncio.create_task 并发启动管道和进度推送，避免 BackgroundTasks 顺序执行死锁
    async def _run():
        try:
            glb_path = await run_pipeline_async(
                session_id=session_id,
                patient_id=patient_id,
                image_paths=image_paths,
                session_output_dir=_session_dir(session_id),
                progress_queue=queue,
                manual_intrinsics=_manual_intrinsics,
            )
            async with SessionLocal() as bg_db:
                async with bg_db.begin():
                    row = await bg_db.get(SessionModel, session_id)
                    if row:
                        row.status = "done"
                        row.glb_path = str(glb_path.relative_to(ROOT))
                        row.finished_at = datetime.utcnow()
        except Exception as exc:
            async with SessionLocal() as bg_db:
                async with bg_db.begin():
                    row = await bg_db.get(SessionModel, session_id)
                    if row:
                        row.status = "error"
                        row.error_msg = str(exc)
                        row.finished_at = datetime.utcnow()

    async def _launch():
        await asyncio.gather(
            _push_progress(session_id, queue),
            _run(),
        )

    background_tasks.add_task(_launch)

    return SessionOut(
        id=session_id,
        patient_id=patient_id,
        notes=notes or None,
        status="running",
        created_at=sess.created_at,
        finished_at=None,
        has_model=False,
    )


@app.get("/api/sessions", response_model=List[SessionOut])
async def list_sessions(
    patient_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """列出会话（可按 patient_id 过滤），按创建时间倒序"""
    stmt = select(SessionModel).order_by(SessionModel.created_at.desc())
    if patient_id:
        stmt = stmt.where(SessionModel.patient_id == patient_id)
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [
        SessionOut(
            **{k: getattr(r, k) for k in ("id", "patient_id", "notes", "status", "created_at", "finished_at")},
            has_model=bool(r.glb_path),
        )
        for r in rows
    ]


@app.get("/api/sessions/{session_id}", response_model=SessionOut)
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)):
    row = await _get_session_or_404(session_id, db)
    return SessionOut(
        **{k: getattr(row, k) for k in ("id", "patient_id", "notes", "status", "created_at", "finished_at")},
        has_model=bool(row.glb_path),
    )


@app.get("/api/sessions/{session_id}/model")
async def download_model(session_id: int, db: AsyncSession = Depends(get_db)):
    """下载该会话生成的 GLB 文件"""
    row = await _get_session_or_404(session_id, db)
    if not row.glb_path:
        raise HTTPException(status_code=404, detail="模型尚未生成")
    glb_abs = ROOT / row.glb_path
    if not glb_abs.exists():
        raise HTTPException(status_code=404, detail="GLB 文件不存在")
    return FileResponse(
        str(glb_abs),
        media_type="model/gltf-binary",
        filename=f"face_{session_id}.glb",
    )


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)):
    row = await _get_session_or_404(session_id, db)
    # 删除文件
    d = _session_dir(session_id)
    if d.exists():
        shutil.rmtree(d)
    await db.delete(row)
    await db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# REST 端点：手动相机内参
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/calibration", status_code=200)
async def get_calibration():
    path = ROOT / "config" / "camera_calibration.json"
    file_calibration = None
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            file_calibration = json.load(f)
    return {
        "manual_intrinsics": _manual_intrinsics,
        "file_calibration": file_calibration,
    }


@app.post("/api/calibration", status_code=200)
async def set_calibration(req: CalibrationRequest):
    """
    提交手动相机内参（全局生效，直到下一次提交或服务重启）。
    提交后创建的会话将使用此内参，跳过 Dust3R 预测。
    """
    global _manual_intrinsics
    _manual_intrinsics = {"fx": req.fx, "fy": req.fy, "cx": req.cx, "cy": req.cy}
    return {"message": "内参已更新", "intrinsics": _manual_intrinsics}


@app.delete("/api/calibration", status_code=200)
async def clear_calibration():
    """清除手动内参，恢复 Dust3R 自动预测"""
    global _manual_intrinsics
    _manual_intrinsics = None
    return {"message": "已清除手动内参，将使用 Dust3R 自动预测"}


# ══════════════════════════════════════════════════════════════════════════════
# REST 端点：术前术后对比 ICP
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/compare")
async def compare_sessions(req: CompareRequest, db: AsyncSession = Depends(get_db)):
    """
    后端 ICP 对齐：将 session_b（术后）对齐到 session_a（术前）坐标系。
    返回 4×4 变换矩阵（前端用 Three.js 实时应用，无需重新下载 GLB）。
    目标耗时 <10s。
    """
    import numpy as np

    sess_a = await _get_session_or_404(req.session_id_a, db)
    sess_b = await _get_session_or_404(req.session_id_b, db)

    for sess, label in [(sess_a, "session_a"), (sess_b, "session_b")]:
        if sess.status != "done" or not sess.glb_path:
            raise HTTPException(status_code=422, detail=f"{label} 模型尚未生成")

    glb_a = ROOT / sess_a.glb_path
    glb_b = ROOT / sess_b.glb_path

    loop = asyncio.get_event_loop()
    transform = await loop.run_in_executor(None, lambda: _icp_align(glb_a, glb_b))

    return {"transform_matrix": transform.tolist()}


def _icp_align(glb_source: Path, glb_target: Path):
    """
    用 Open3D ICP 将 source 对齐到 target。
    返回 (4,4) numpy float64 变换矩阵。
    """
    import numpy as np
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("请安装 open3d: pip install open3d")
    import trimesh

    def _load_pcd(glb_path: Path):
        scene = trimesh.load(str(glb_path))
        if isinstance(scene, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(scene.geometry.values()))
        else:
            mesh = scene
        pts = np.array(mesh.vertices, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals()
        return pcd

    src_pcd = _load_pcd(glb_source)
    tgt_pcd = _load_pcd(glb_target)

    # 粗对齐：质心对齐
    src_center = src_pcd.get_center()
    tgt_center = tgt_pcd.get_center()
    init_t = np.eye(4)
    init_t[:3, 3] = tgt_center - src_center

    # Point-to-Plane ICP
    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=0.005,   # 5mm
        init=init_t,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50,
        ),
    )
    return np.array(result.transformation)


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket：进度推送
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/{session_id}")
async def websocket_progress(websocket: WebSocket, session_id: int):
    """
    客户端通过 WS 接收重建进度。
    消息格式：{"stage": str, "pct": int, "message": str}
    当 stage=="done" 时，附加 "glb_url": "/api/sessions/{id}/model"
    """
    await websocket.accept()

    # 注册连接
    if session_id not in _ws_connections:
        _ws_connections[session_id] = []
    _ws_connections[session_id].append(websocket)

    try:
        # 如果已有进度队列（管道还在运行），只需保持连接等待推送
        # 如果没有队列，说明管道未启动或已完成，直接通知客户端
        if session_id not in _progress_queues:
            await websocket.send_text(json.dumps(
                {"stage": "info", "pct": 0, "message": "无正在运行的重建任务"},
                ensure_ascii=False,
            ))

        # 保持连接，直到客户端断开
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # 心跳：发送 ping
                await websocket.send_text('{"stage":"ping"}')
    except WebSocketDisconnect:
        pass
    finally:
        conns = _ws_connections.get(session_id, [])
        if websocket in conns:
            conns.remove(websocket)
