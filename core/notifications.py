import asyncio
import threading
import json
from typing import Optional

class NotificationManager:
    """Manages real-time event broadcasting via SSE."""
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def subscribe(self):
        """Register a new SSE client queue. Thread-safe via GIL (list.append is atomic)."""
        q = asyncio.Queue()
        self.clients.append(q)  # BUG-15 fix: no threading.Lock in async context
        return q

    def unsubscribe(self, q):
        """Remove a client queue. Safe without lock — list.remove is GIL-protected."""
        try:
            self.clients.remove(q)
        except ValueError:
            pass

    def broadcast(self, data: dict):
        msg = f"data: {json.dumps(data)}\n\n"
        with self.lock:
            loop = self._loop
            clients = list(self.clients)
        if loop is None or not loop.is_running():
            return
        for q in clients:
            try:
                loop.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:
                pass

notification_manager = NotificationManager()
