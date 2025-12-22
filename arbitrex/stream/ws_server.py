from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json
import logging
from collections import defaultdict
from typing import Optional

LOG = logging.getLogger("arbitrex.stream.ws")

app = FastAPI()

# Global reference to the event loop (will be set on startup)
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_tick_collector_pool = None  # Will be set by start script


def set_tick_collector_pool(pool):
    """Set the MT5 pool reference so we can start it after FastAPI starts"""
    global _tick_collector_pool
    _tick_collector_pool = pool


@app.on_event("startup")
async def startup_event():
    """Capture event loop on startup"""
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    LOG.info(f"✓ FastAPI startup: Captured event loop {_event_loop}")
    
    # Start tick collector now that event loop is ready
    if _tick_collector_pool is not None:
        LOG.info("✓ Starting tick collector now that event loop is ready...")
        # This will be called from the start script


class Broker:
    def __init__(self):
        self._subs = defaultdict(set)  # symbol -> set of WebSocket
        self._lock = asyncio.Lock()

    async def subscribe(self, ws: WebSocket, symbols):
        await self._lock.acquire()
        try:
            for s in symbols:
                self._subs[s].add(ws)
        finally:
            self._lock.release()

    async def unsubscribe_all(self, ws: WebSocket):
        await self._lock.acquire()
        try:
            for s, conns in list(self._subs.items()):
                if ws in conns:
                    conns.remove(ws)
                    if not conns:
                        del self._subs[s]
        finally:
            self._lock.release()

    async def publish(self, payload: dict):
        # payload must contain `symbol`
        sym = payload.get("symbol")
        if not sym:
            LOG.warning("Publish called with no symbol in payload")
            return
        
        await self._lock.acquire()
        try:
            conns = list(self._subs.get(sym, []))
        finally:
            self._lock.release()
        
        if not conns:
            LOG.debug("No subscribers for %s", sym)
            return
        
        msg = json.dumps(payload)
        LOG.debug("Publishing to %d subscriber(s) for %s: %s", len(conns), sym, msg[:100])
        
        for ws in conns:
            try:
                await ws.send_text(msg)
                LOG.debug("Successfully sent to subscriber for %s", sym)
            except Exception as e:
                LOG.warning("Failed send to subscriber for %s: %s", sym, e)


broker = Broker()


@app.get("/")
async def index():
    html = """
    <html>
      <head>
        <title>Arbitrex Tick Stream</title>
      </head>
      <body>
        <h3>Arbitrex Tick Stream</h3>
        <p>Connect via WebSocket at <code>/ws</code></p>
        <p>Status: <a href="/status">Check System Status</a></p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/status")
async def status():
    """System status endpoint"""
    global _event_loop
    
    total_subscribers = sum(len(subs) for subs in broker._subs.values())
    subscribed_symbols = list(broker._subs.keys())
    
    return {
        "status": "running",
        "event_loop_available": _event_loop is not None,
        "event_loop_running": _event_loop.is_running() if _event_loop else False,
        "active_subscribers": total_subscribers,
        "subscribed_symbols": subscribed_symbols,
        "symbol_count": len(subscribed_symbols)
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    LOG.info("WebSocket accepted from %s", websocket.client)
    try:
        while True:
            try:
                # Use a 5-second timeout to check for messages
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                LOG.info("Received from %s: %s", websocket.client, data)
                try:
                    obj = json.loads(data)
                except Exception as e:
                    LOG.warning("JSON parse failed: %s", e)
                    await websocket.send_text(json.dumps({"error": "invalid json"}))
                    continue
                # expect {"subscribe": ["EURUSD","XAUUSD"]}
                if "subscribe" in obj:
                    symbols = obj.get("subscribe") or []
                    await broker.subscribe(websocket, symbols)
                    LOG.info("Client %s subscribed to: %s", websocket.client, symbols)
                    await websocket.send_text(json.dumps({"subscribed": symbols}))
                elif "unsubscribe" in obj:
                    await broker.unsubscribe_all(websocket)
                    LOG.info("Client %s unsubscribed", websocket.client)
                    await websocket.send_text(json.dumps({"unsubscribed": True}))
                else:
                    await websocket.send_text(json.dumps({"error": "unknown command"}))
            except asyncio.TimeoutError:
                # No message for 5 seconds, that's OK - just continue listening
                LOG.debug("No message for 5s from %s", websocket.client)
                pass
    except WebSocketDisconnect:
        LOG.info("WebSocket disconnected from %s", websocket.client)
        await broker.unsubscribe_all(websocket)
    except Exception as e:
        LOG.exception("WebSocket error from %s: %s", websocket.client, e)
        await broker.unsubscribe_all(websocket)


def get_publisher():
    """Return an async-safe publisher compatible with MT5ConnectionPool's callback.

    The MT5 pool runs in threads; we expose a sync wrapper that schedules the async
    publish on the event loop. The loop is captured from uvicorn's context.
    """
    publish_count = [0]  # Use list to allow modification in nested function

    async def _publish_async(payload: dict):
        await broker.publish(payload)

    def _publish_sync(payload: dict):
        global _event_loop
        publish_count[0] += 1
        
        try:
            # Use the globally captured event loop
            if _event_loop is None:
                if publish_count[0] <= 5:
                    LOG.warning("Event loop not yet available (publish #%d) - is FastAPI started?", publish_count[0])
                return
            
            if not _event_loop.is_running():
                if publish_count[0] <= 5:
                    LOG.warning("Event loop exists but not running (publish #%d)", publish_count[0])
                return
            
            # Schedule the coroutine on the event loop
            future = asyncio.run_coroutine_threadsafe(_publish_async(payload), _event_loop)
            
            # Log first few publishes for debugging
            if publish_count[0] <= 5:
                LOG.info("✓ Published tick #%d for symbol %s", publish_count[0], payload.get('symbol'))
            elif publish_count[0] == 100:
                LOG.info("✓ Published 100 ticks successfully")
            elif publish_count[0] % 1000 == 0:
                LOG.info("✓ Published %d ticks total", publish_count[0])
            
        except Exception as e:
            if publish_count[0] <= 10:
                LOG.error("publish_sync failed (attempt %d): %s", publish_count[0], e, exc_info=True)
            else:
                LOG.debug("publish_sync failed: %s", e)

    return _publish_sync
