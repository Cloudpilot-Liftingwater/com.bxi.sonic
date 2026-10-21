from __future__ import annotations


class RecordingState:
    ARMED = "armed"
    RECORDING = "recording"
    FINISHED = "finished"

    def __init__(self) -> None:
        self.phase = self.ARMED
        self.completed_episodes = 0

    def start(self) -> bool:
        if self.phase == self.FINISHED:
            self.phase = self.ARMED
        if self.phase != self.ARMED:
            return False
        self.phase = self.RECORDING
        return True

    def finish(self) -> bool:
        if self.phase != self.RECORDING:
            return False
        self.phase = self.FINISHED
        self.completed_episodes += 1
        return True

    def abort(self) -> bool:
        if self.phase != self.RECORDING:
            return False
        self.phase = self.ARMED
        return True

    def rearm(self) -> bool:
        if self.phase not in (self.ARMED, self.FINISHED):
            return False
        self.phase = self.ARMED
        return True

    def toggle(self) -> str | None:
        if self.phase in (self.ARMED, self.FINISHED):
            return "start"
        if self.phase == self.RECORDING:
            return "finish"
        return None
