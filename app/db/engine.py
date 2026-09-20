"""异步引擎与会话工厂。main.py lifespan 负责 init/dispose，业务代码只取 session。"""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings) -> AsyncEngine:
    """幂等初始化（pool_pre_ping 防容器重启后的陈旧连接）。"""
    global _engine, _session_factory
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )
        _session_factory = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info(
            "async engine initialized: %s",
            _engine.url.render_as_string(hide_password=True),
        )
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("engine not initialized; call init_engine(settings) first")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("engine not initialized; call init_engine(settings) first")
    return _session_factory


async def check_db() -> None:
    """启动探活：SELECT 1。失败抛异常由 main.py 决定退出提示。"""
    async with get_session_factory()() as session:
        await session.execute(text("SELECT 1"))


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("async engine disposed")
    _engine = None
    _session_factory = None
