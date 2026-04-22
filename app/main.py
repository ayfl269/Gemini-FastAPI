import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from .server.chat import router as chat_router
from .server.gemini_chat import router as gemini_router
from .server.health import router as health_router
from .server.images import router as images_router
from .server.middleware import (
    add_cors_middleware,
    add_exception_handler,
    cleanup_expired_images,
)
from .services import GeminiClientPool, LMDBConversationStore
from .utils import g_config

RETENTION_CLEANUP_INTERVAL_SECONDS = 6 * 60 * 60  # Check every 6 hours
COOKIE_REFRESH_INTERVAL_SECONDS = 10 * 60  # Check every 10 minutes


async def _run_retention_cleanup(stop_event: asyncio.Event) -> None:
    """
    Periodically enforce LMDB retention policy until the stop_event is set.
    """
    store = LMDBConversationStore()
    if store.retention_days <= 0:
        logger.info("LMDB retention cleanup disabled; skipping scheduler.")
        return

    logger.info(
        f"Starting LMDB retention cleanup task (retention={store.retention_days} day(s), interval={RETENTION_CLEANUP_INTERVAL_SECONDS} seconds)."
    )

    while not stop_event.is_set():
        try:
            store.cleanup_expired()
            cleanup_expired_images(store.retention_days)
        except Exception:
            logger.exception("LMDB retention cleanup task failed.")

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=RETENTION_CLEANUP_INTERVAL_SECONDS,
            )
        except TimeoutError:
            continue

    logger.info("LMDB retention cleanup task stopped.")


async def _run_cookie_refresh(stop_event: asyncio.Event) -> None:
    """
    Periodically check and refresh cookies for all clients.
    Proactively refreshes cookies before they expire to avoid service disruption.
    """
    if not g_config.gemini.auto_refresh:
        logger.info("Auto cookie refresh disabled.")
        return

    logger.info(
        f"Starting auto cookie refresh task (interval={COOKIE_REFRESH_INTERVAL_SECONDS} seconds)."
    )

    pool = GeminiClientPool()

    while not stop_event.is_set():
        try:
            await asyncio.sleep(COOKIE_REFRESH_INTERVAL_SECONDS)
            if stop_event.is_set():
                break

            logger.info(f"Running periodic cookie refresh check (clients: {len(pool.clients)})")

            for client in pool.clients:
                try:
                    if client.is_likely_expired():
                        logger.info(
                            f"[{client.id}] Cookie refresh triggered "
                            f"(failures={client.consecutive_failures}, "
                            f"last_success={client.last_success_time:.0f}s ago)."
                        )
                        success = await client.force_refresh()
                        if success:
                            logger.success(f"[{client.id}] Cookie refresh completed successfully")
                        else:
                            logger.error(f"[{client.id}] Cookie refresh failed - client not running after refresh")
                    else:
                        seconds_since = client.seconds_since_last_request()
                        logger.debug(
                            f"[{client.id}] Cookie OK - skipping refresh "
                            f"(failures={client.consecutive_failures}, "
                            f"last_success={time.time() - client.last_success_time:.0f}s ago, "
                            f"last_request={seconds_since:.0f}s ago)"
                        )
                except Exception:
                    logger.exception(f"[{client.id}] Cookie refresh check failed")
        except Exception:
            logger.exception("Cookie refresh task failed.")

    logger.info("Cookie refresh task stopped.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_stop_event = asyncio.Event()

    pool = GeminiClientPool()
    try:
        await pool.init()
    except Exception as e:
        logger.exception(f"Failed to initialize Gemini clients: {e}")
        raise

    cleanup_task = asyncio.create_task(_run_retention_cleanup(cleanup_stop_event))
    cookie_refresh_task = asyncio.create_task(_run_cookie_refresh(cleanup_stop_event))
    
    # Give the tasks a chance to start and surface immediate failures.
    await asyncio.sleep(0)
    if cleanup_task.done():
        try:
            cleanup_task.result()
        except Exception:
            logger.exception("LMDB retention cleanup task failed to start.")
            raise
    if cookie_refresh_task.done():
        try:
            cookie_refresh_task.result()
        except Exception:
            logger.exception("Cookie refresh task failed to start.")
            raise

    logger.info(f"Gemini clients initialized: {[c.id for c in pool.clients]}.")
    logger.info("Gemini API Server ready to serve requests.")

    try:
        yield
    finally:
        cleanup_stop_event.set()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            logger.debug("LMDB retention cleanup task cancelled during shutdown.")
        except Exception:
            logger.exception(
                "LMDB retention cleanup task terminated with an unexpected error during shutdown."
            )
        try:
            await cookie_refresh_task
        except asyncio.CancelledError:
            logger.debug("Cookie refresh task cancelled during shutdown.")
        except Exception:
            logger.exception(
                "Cookie refresh task terminated with an unexpected error during shutdown."
            )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gemini API Server",
        description="OpenAI-compatible API for Gemini Web",
        version="1.0.0",
        lifespan=lifespan,
    )

    add_cors_middleware(app)
    add_exception_handler(app)

    app.include_router(health_router, tags=["Health"])
    app.include_router(chat_router, tags=["Chat"])
    app.include_router(gemini_router, tags=["Gemini Native"])
    app.include_router(images_router, tags=["Images"])

    return app
