"""NVIDIA SONIC VideoWriter with lifecycle fixes; see LOCAL_CHANGES.patch."""
from pathlib import Path
import queue
import threading

import av
import numpy as np


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "h264",
        buffer_size: int = 50,
    ):
        self.output_path = output_path
        self.width, self.height = width, height
        self.fps, self.codec = fps, codec
        self.queue = queue.Queue(maxsize=buffer_size)
        self.container = self.stream = self._thread = None
        self._error = None
        self._stopped = False
        self.encoder_options = {}
        self.thread_count = 0

    def _assert_dimensions(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB {(self.height, self.width, 3)}, got "
                             f"{frame.dtype} {frame.shape}")

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError(f"Video encoder failed: {self.output_path}") from self._error

    def _start(self):
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(self.output_path), mode="w")
        try:
            self.stream = self.container.add_stream(self.codec, rate=self.fps)
            self.stream.width, self.stream.height = self.width, self.height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = dict(self.encoder_options)
            self.stream.thread_count = self.thread_count
            self._thread = threading.Thread(target=self._writer_worker, daemon=True)
            self._thread.start()
        except BaseException:
            self.container.close()
            raise

    def _put(self, value):
        while True:
            self._raise_error()
            try:
                self.queue.put(value, timeout=.1)
                self._raise_error()
                return
            except queue.Full:
                if not self._thread.is_alive():
                    self._raise_error()
                    raise RuntimeError("Video worker exited before consuming its queue")

    def add_frame(self, frame: np.ndarray) -> None:
        if self._stopped:
            raise RuntimeError("Video writer is closed")
        self._assert_dimensions(frame)
        self._raise_error()
        if self._thread is None:
            self._start()
        self._put(frame)

    def _writer_worker(self) -> None:
        try:
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                self._assert_dimensions(frame)
                for packet in self.stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    self.container.mux(packet)
            self._flush_stream()
        except BaseException as exc:
            self._error = exc
        finally:
            try:
                self.container.close()
            except BaseException as exc:
                self._error = self._error or exc

    def _flush_stream(self) -> None:
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def stop(self) -> str:
        """Join the encoder, including its final frame and delayed codec packets."""
        if not self._stopped:
            self._stopped = True
            if self._thread is not None:
                try:
                    self._put(None)
                finally:
                    self._thread.join()
        self._raise_error()
        return self.output_path

    def cancel(self) -> None:
        """Close the worker before deleting its incomplete output."""
        try:
            self.stop()
        finally:
            Path(self.output_path).unlink(missing_ok=True)
