"""FastAPI application entry point with FastMCP mounted and APScheduler."""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.auth.hmac_auth import verify_request_signature
from app.config import get_settings
from app.services.scheduler_service import scheduler, setup_scheduler

logger = logging.getLogger(__name__)
settings = get_settings()

logging.basicConfig(
    level=logging.DEBUG if settings.APP_DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)


class HmacMiddleware(BaseHTTPMiddleware):
    """Verify HMAC-signed requests when HMAC_SECRET is configured.

    Only requests that carry the X-Telegram-Chat-Id header are checked.
    Requests without that header pass through (dependencies still guard access).
    When HMAC_SECRET is empty the middleware is a no-op (dev/test mode).
    """

    async def dispatch(self, request, call_next):
        if not settings.HMAC_SECRET:
            return await call_next(request)

        chat_id_header = request.headers.get("x-telegram-chat-id")
        if chat_id_header is None:
            return await call_next(request)

        ok = verify_request_signature(
            settings.HMAC_SECRET,
            chat_id_header,
            request.headers.get("x-request-timestamp"),
            request.headers.get("x-nonce"),
            request.headers.get("x-signature"),
        )
        if not ok:
            return JSONResponse({"detail": "Invalid request signature"}, status_code=401)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from app.services.wizard_checkpointer import _DEV_CHECKPOINTER_PATH
    from app.services.wizard_graph import build_wizard_graph, set_wizard_graph

    # Startup
    logger.info("Starting Goal Agent...")
    setup_scheduler()
    scheduler.start()
    logger.info("APScheduler started with %d jobs", len(scheduler.get_jobs()))

    async with AsyncSqliteSaver.from_conn_string(_DEV_CHECKPOINTER_PATH) as checkpointer:
        set_wizard_graph(build_wizard_graph(checkpointer))
        logger.info("Wizard graph: AsyncSqliteSaver at %s", _DEV_CHECKPOINTER_PATH)

        bot_task = None
        if settings.TELEGRAM_GO_GETTER_BOT_TOKEN:
            from app.bots.go_getter_bot import start_go_getter_bot

            bot_task = asyncio.create_task(start_go_getter_bot())
            logger.info("Telegram go getter bot task started")
        else:
            logger.info("TELEGRAM_GO_GETTER_BOT_TOKEN not set – go getter bot disabled")

        yield

        # Shutdown
        if bot_task:
            bot_task.cancel()
            with suppress(asyncio.CancelledError):
                await bot_task
            logger.info("Telegram go getter bot stopped")

        scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped")


app = FastAPI(
    title="Goal Agent",
    description="AI-powered goal and habit tracking agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HMAC signature verification (Issue #3) — must be added after CORS
app.add_middleware(HmacMiddleware)

# REST API router
from app.api.v1.router import router as api_router  # noqa: E402

app.include_router(api_router, prefix="/api/v1")

# FastMCP ASGI sub-app
from app.mcp.server import mcp  # noqa: E402

mcp_app = mcp.http_app(path="/mcp")
app.mount("/mcp", mcp_app)


@app.get("/health")
async def health():
    """Basic liveness probe."""
    return {"status": "ok", "service": "goal-agent"}


@app.get("/health/ready")
async def health_ready():
    """Readiness probe with database check."""
    from app.database import engine
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ready", "database": "ok"}
    except Exception as e:
        logger.error("Health ready check failed: %s", e)
        return JSONResponse({"status": "not_ready", "database": "error"}, status_code=503)
