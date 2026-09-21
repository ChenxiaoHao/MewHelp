import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.api.routes import router
from app.core.config import get_settings
from app.db.engine import check_db, dispose_engine, init_engine

logging.basicConfig(level=logging.INFO)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings = get_settings()  # 启动即校验 .env 必填项，不带病启动
    except ValidationError as exc:
        raise SystemExit(
            f"[启动失败] .env 配置缺失或非法，请参考 .env.example 补全：\n{exc}"
        ) from exc
    init_engine(settings)
    try:
        await check_db()  # 硬性核对点④的运行期形态：连不上快速失败
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "[启动失败] 无法连接 MySQL。请先启动 Docker Desktop，然后执行 "
            f"`docker compose up -d`，等容器 healthy 后重试。详情: {exc}"
        ) from exc
    logger.info(
        "MySQL 已连接 %s:%s/%s", settings.mysql_host, settings.mysql_port, settings.mysql_db
    )

    from app.rag import milvus_store  # 模块级 import 亦可;pymilvus 已是硬依赖

    try:
        _mv = milvus_store.get_client(settings.milvus_uri, timeout=3.0)
        if not milvus_store.health_ok(_mv):
            logger.warning("Milvus 不可达(%s):query_faq 将运行期报错,请 docker compose up -d", settings.milvus_uri)
    except Exception as exc:  # noqa: BLE001 —— 探活本身永不阻塞启动
        logger.warning("Milvus 探活异常(不阻塞启动): %s", exc)
    yield
    await dispose_engine()


app = FastAPI(title="MewHelp 电商智能客服", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")
