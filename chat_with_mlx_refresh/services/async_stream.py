from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, Optional


logger = logging.getLogger(__name__)

_STREAM_END = object()


@dataclass
class _StreamError:
    exception: BaseException


class ThreadedGeneratorBridge:
    def __init__(self, iterator: Iterator[Any], stop_event: Any) -> None:
        self.iterator = iterator
        self.stop_event = stop_event
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue[Any]] = None
        self._thread: Optional[threading.Thread] = None
        self._finished = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._iterator_closed = False
        self._iterator_close_error: Optional[BaseException] = None

    def __aiter__(self) -> AsyncIterator[Any]:
        return self.iterate()

    async def iterate(self) -> AsyncIterator[Any]:
        self._start()
        if self._queue is None:
            self.close(wait=False)
            return

        try:
            while True:
                item = await self._queue.get()
                if item is _STREAM_END:
                    break
                if isinstance(item, _StreamError):
                    raise item.exception
                yield item
        finally:
            self.close(wait=False)

    def close(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            started = self._started

        if not self._finished.is_set():
            self.stop_event.set()

        if wait and thread is not None and threading.current_thread() is not thread:
            thread.join()
        elif not started:
            self._close_iterator()
            self._finished.set()

    def wait_closed(self, timeout: Optional[float] = None) -> None:
        thread = None
        with self._lock:
            thread = self._thread
        if thread is not None and threading.current_thread() is not thread:
            thread.join(timeout=timeout)
        else:
            self._finished.wait(timeout=timeout)

    def _start(self) -> None:
        with self._lock:
            if self._started or self._closed:
                return
            self._started = True
            self._loop = asyncio.get_running_loop()
            self._queue = asyncio.Queue()
            self._thread = threading.Thread(target=self._run, name="mlx-stream-bridge", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            for item in self.iterator:
                if self.stop_event.is_set():
                    break
                self._enqueue(item)
        except BaseException as exc:
            self._enqueue(_StreamError(exc))
        finally:
            self._close_iterator()
            self._finished.set()
            self._enqueue(_STREAM_END)

    def _enqueue(self, item: Any) -> None:
        if self._loop is None or self._queue is None:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def _close_iterator(self) -> None:
        with self._lock:
            if self._iterator_closed:
                return
            self._iterator_closed = True
        close = getattr(self.iterator, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:
            with self._lock:
                if self._iterator_close_error is None:
                    self._iterator_close_error = exc
            logger.warning("Failed to close iterator %s: %s", self.iterator, exc)

    def get_close_error(self) -> Optional[BaseException]:
        with self._lock:
            return self._iterator_close_error
