"""
数据库配置（SQLAlchemy 2.x）

默认使用 SQLite（开发/单机部署）。
如需 PostgreSQL，设置环境变量：
  DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/face3d
"""
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./face3d.db",
)

# SQLite 需要 check_same_thread=False
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args=connect_args,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def init_db():
    """建表（首次启动时调用）"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI 依赖注入：获取数据库会话"""
    async with SessionLocal() as session:
        yield session
