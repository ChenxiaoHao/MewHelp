import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.api.routes import router
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_settings()  # 启动即校验 .env 必填项，不带病启动
    except ValidationError as exc:
        raise SystemExit(
            f"[启动失败] .env 配置缺失或非法，请参考 .env.example 补全：\n{exc}"
        ) from exc
    yield


app = FastAPI(title="MewHelp 电商智能客服", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")
