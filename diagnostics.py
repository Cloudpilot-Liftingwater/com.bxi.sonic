"""Bounded best-effort logging off the motion thread; never owns control state."""
from collections import deque
import json
from threading import Condition, Thread


class DeferredDiagnostics:
    def __init__(self, logger):
        self.logger = logger
        self._condition = Condition()
        self._events = deque(maxlen=64)
        self._latest = {}
        self._closed = False
        self.dropped = 0
        self.errors = 0
        self._thread = Thread(target=self._run, name='sonic-diagnostics', daemon=True)
        self._thread.start()

    def submit(self, level, text, payload=None, key=None):
        with self._condition:
            if self._closed:
                return
            entry = (level, text, None if payload is None else dict(payload))
            if key is not None:
                self._latest[key] = entry
            else:
                self.dropped += len(self._events) == self._events.maxlen
                self._events.append(entry)
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._events or self._latest)
                if self._closed:
                    return
                if self._events:
                    entry = self._events.popleft()
                else:
                    key = next(iter(self._latest))
                    entry = self._latest.pop(key)
            # Never hold the producer's lock while formatting or writing logs.
            try:
                level, text, payload = entry
                if payload is not None:
                    text += json.dumps(payload, ensure_ascii=False)
                getattr(self.logger, level)(text)
            except Exception:
                self.errors += 1

    def close(self):
        with self._condition:
            self._closed = True
            self._events.clear()
            self._latest.clear()
            self._condition.notify()
        self._thread.join(timeout=.2)
