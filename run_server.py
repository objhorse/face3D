"""
启动 FastAPI 服务

用法:
  D:/Anaconda/envs/gaussian/python.exe run_server.py

可选环境变量:
  DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/face3d
  PORT=8000
  HOST=0.0.0.0
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"启动 face3D 服务: http://{host}:{port}")
    print(f"API 文档: http://{host}:{port}/docs")
    print(f"按 Ctrl+C 停止")

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
