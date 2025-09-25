import time
import threading
import numpy as np
from typing import List, Dict, Any, Callable, Iterable

class EGAgentSystem:
    def __init__(self):
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: Dict[str, List[Callable[[Any], None]]] = {}
        self._conversation: List[str] = ["系统: 测试模式已启动，等待指令…"]
        self._logs: List[str] = []
        self._entities = [
            {"name": "cup", "info": "state=on_table"},
            {"name": "chair", "info": "state=idle"},
            {"name": "door", "info": "state=closed"},
        ]
        self._tick = 0
        self._last_bt = None

    # --------------- control ---------------
    def start(self):
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._log("Test system started.")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._emit("status", self.status)

    def stop(self):
        if not self._running:
            return
        self._stop_event.set()
        self._running = False
        self._log("Test system stopped.")
        self._emit("status", self.status)

    @property
    def status(self) -> bool:
        return self._running

    # --------------- loop ---------------
    def _loop(self):
        while not self._stop_event.is_set():
            time.sleep(0.5)
            self._tick += 1
            # update entities occasionally
            if self._tick % 4 == 0:
                for e in self._entities:
                    if np.random.rand() < 0.3:
                        e["info"] = f"state={np.random.choice(['idle','active','done'])}"
                self._emit("entities", self.get_entity_rows())
            # add log
            self._log(f"Heartbeat tick={self._tick}")
            # conversation dribble
            if self._tick % 6 == 0:
                self._conversation.append(f"智能体: 模拟回复 {self._tick}")
                self._emit("conversation", self.get_conversation_text())

    # --------------- listeners ---------------
    def add_listener(self, kind: str, cb: Callable[[Any], None]):
        self._listeners.setdefault(kind, []).append(cb)

    def _emit(self, kind: str, data: Any):
        for cb in self._listeners.get(kind, []):
            try:
                cb(data)
            except Exception:
                pass

    # --------------- UI-facing API ---------------
    def feed_instruction(self, text: str):
        self._log(f"Instruction received: {text}")
        self._conversation.append(f"用户: {text}")
        self._conversation.append("智能体: (测试系统占位响应)")
        self._emit("conversation", self.get_conversation_text())

    def get_conversation_text(self) -> str:
        return "\n".join(self._conversation[-300:])

    def get_log_text_tail(self, n: int = 400) -> str:
        return "\n".join(self._logs[-n:])

    def get_entity_rows(self) -> Iterable[tuple]:
        for e in self._entities:
            yield e["name"], e["info"]

    # images (dummy gradients)
    def get_current_instance_seg_image(self) -> np.ndarray:
        return self._grad_image(220, 160, seed=1)

    def get_traversable_map_image(self) -> np.ndarray:
        return self._grad_image(220, 160, seed=2)

    def get_current_instance_3d_image(self) -> np.ndarray:
        return self._grad_image(220, 160, seed=3)

    def get_semantic_map_image(self) -> np.ndarray:
        return self._grad_image(480, 200, seed=4)

    def get_entity_bt_image(self) -> np.ndarray:
        if self._last_bt is None or self._tick % 10 == 0:
            self._last_bt = self._grad_image(300, 360, seed=self._tick)
        return self._last_bt

    # --------------- helpers ---------------
    def _log(self, line: str):
        ts = time.strftime("%H:%M:%S")
        self._logs.append(f"[{ts}] {line}")
        if len(self._logs) > 2000:
            self._logs = self._logs[-1000:]
        self._emit("log", self.get_log_text_tail())

    def _grad_image(self, w: int, h: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed + self._tick)
        base = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        gx = np.linspace(0, 255, w, dtype=np.uint8)
        base[..., 0] = (base[..., 0] // 2 + gx)
        return base
