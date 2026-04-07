"""
SQLAlchemy ORM 模型
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Integer, String, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base


class Session(Base):
    """3D 重建会话（一次采集 = 一条记录）"""
    __tablename__ = "sessions"

    id:          Mapped[int]            = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id:  Mapped[str]            = mapped_column(String(64), index=True, nullable=False)
    notes:       Mapped[Optional[str]]  = mapped_column(Text, nullable=True)
    status:      Mapped[str]            = mapped_column(String(20), default="pending")
    # pending / running / done / error
    created_at:  Mapped[datetime]       = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    glb_path:    Mapped[Optional[str]]  = mapped_column(String(512), nullable=True)
    error_msg:   Mapped[Optional[str]]  = mapped_column(Text, nullable=True)
    meta:        Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
