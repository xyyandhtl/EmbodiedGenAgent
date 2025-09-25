import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import threading
import time
from typing import Callable, Dict, List, Any, Iterable

from EG_agent.prompts.object_sets import AllObject
from EG_agent.reasoning.logic_goal import LogicGoalGenerator
from EG_agent.planning.bt_planner import BTGenerator
from EG_agent.vlmap.vlmap_nav_ros2 import VLMapNavROS2
from EG_agent.system.agent.agent import Agent
from EG_agent.system.envs.isaacsim_env import IsaacsimEnv


class EGAgentSystem:
    current_goal = None
    def __init__(self):
        # 初始化 '逻辑Goal生成器'
        self.goal_generator = LogicGoalGenerator()

        # 构建 '行为树规划器'
        self.bt_generator = BTGenerator(env_name="embodied", 
                                        cur_cond_set=set(), 
                                        key_objects=list(AllObject))
        
        # 行为树执行的 'Agent载体’，通过bint_bt动态绑定行为树，被绑定到 '部署环境执行器' 和环境交互
        self.bt_agent = Agent()

        # '部署环境执行器'，定义如何和部署环境交互，和 'Agent载体’ 绑定
        self.env = IsaacsimEnv()
        self.env.place_agent(self.bt_agent)

        # 初始化 'VLM地图导航模块'
        self.vlmap_backend = VLMapNavROS2()

        # 运行控制
        self._running = False
        self._is_finished = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # 监听器: kind -> list[callback(data)]
        self._listeners: Dict[str, List[Callable[[Any], None]]] = {}
        # 缓存/占位数据
        self._conversation: List[str] = []
        self._logs: List[str] = []
        self._entity_info: List[dict] = []   # each: {"name":..., "info":...}
        self._last_bt_image: np.ndarray | None = None
        # TODO: 后续可注入真实数据生成器
        self._placeholder_counter = 0

    # ---------------------- 控制接口 ----------------------
    def start(self):
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._is_finished = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._stop_event.set()
        self._running = False
        # 允许环境安全停止
        try:
            if hasattr(self.env, "close"):
                self.env.close()
        except Exception:
            pass

    def _run_loop(self):
        # 主执行循环 (简化)
        self.env.reset()
        self._log("Environment reset.")
        is_finished = False
        while not self._stop_event.is_set() and not is_finished:
            # TODO: 替换为真实 step
            time.sleep(0.1)
            # 模拟 env.step
            try:
                is_finished = self.env.step()
            except Exception:
                is_finished = True
            # 条件完成判定 (占位)
            if self.current_goal and self.current_goal <= self.env.agents[0].condition_set:
                is_finished = True
            # 周期性占位数据刷新
            self._placeholder_counter += 1
            if self._placeholder_counter % 5 == 0:
                self._append_conversation(f"智能体: 占位回复 {self._placeholder_counter}")
                self._emit("conversation", self.get_conversation_text())
            if self._placeholder_counter % 7 == 0:
                self._update_entities_placeholder()
                self._emit("entities", self.get_entity_rows())
        self._is_finished = True
        self._running = False
        self._log("Agent loop stopped.")
        self._emit("status", self.status)

    # 保留原 run 名称以兼容外部调用
    def run(self):
        self.start()

    def set_env(self, env):
        self.env = env

    @property
    def finished(self) -> bool:
        return self._is_finished

    @property
    def status(self) -> bool:
        # True 表示正在运行
        return self._running

    # ---------------------- 监听/事件 ----------------------
    def add_listener(self, kind: str, cb: Callable[[Any], None]):
        self._listeners.setdefault(kind, []).append(cb)

    def _emit(self, kind: str, data: Any):
        for cb in self._listeners.get(kind, []):
            try:
                cb(data)
            except Exception:
                pass

    # ---------------------- 输入接口 ----------------------
    def feed_observation(self,
                         pose: np.ndarray,
                         intrinsics: np.ndarray,
                         image: np.ndarray,
                         depth: np.ndarray):
        # TODO: camera observation -> vlmap -> update condition / generate ll actions
        # 暂时仅记录一条日志
        self._log("Received observation (placeholder).")
        self._emit("observation", None)

    def feed_instruction(self, text: str):
        # TODO: goal_generator -> bt_generator -> bt_agent.bind_bt
        self._log(f"User instruction: {text}")
        self._append_conversation(f"用户: {text}")
        self._append_conversation("智能体: （占位解析中...）")
        self._emit("conversation", self.get_conversation_text())

    # ---------------------- 数据获取占位接口 ----------------------
    def get_conversation_text(self) -> str:
        return "\n".join(self._conversation[-400:])

    def get_log_text_tail(self, n: int = 400) -> str:
        return "\n".join(self._logs[-n:])

    def get_entity_rows(self) -> Iterable[tuple]:
        # returns iterable of (name, info)
        for e in self._entity_info:
            yield e["name"], e["info"]

    def get_semantic_map_image(self) -> np.ndarray:
        return self._gen_dummy_image(640, 240, "Semantic+Path")

    def get_entity_bt_image(self) -> np.ndarray:
        # 行为树图
        if self._last_bt_image is None or self._placeholder_counter % 10 == 0:
            self._last_bt_image = self._gen_dummy_image(300, 400, "Behavior Tree")
        return self._last_bt_image

    def get_traversable_map_image(self) -> np.ndarray:
        return self._gen_dummy_image(260, 180, "Traversable")

    def get_current_instance_seg_image(self) -> np.ndarray:
        return self._gen_dummy_image(260, 180, "Instance 2D")

    def get_current_instance_3d_image(self) -> np.ndarray:
        return self._gen_dummy_image(260, 180, "Instance 3D")

    # ---------------------- 占位内部工具 ----------------------
    def _append_conversation(self, line: str):
        self._conversation.append(line)
        if len(self._conversation) > 2000:
            self._conversation = self._conversation[-1000:]

    def _log(self, line: str):
        ts = time.strftime("%H:%M:%S")
        self._logs.append(f"[{ts}] {line}")
        if len(self._logs) > 5000:
            self._logs = self._logs[-3000:]
        self._emit("log", self.get_log_text_tail())

    def _update_entities_placeholder(self):
        # 模拟实体增量
        self._entity_info = [
            {"name": f"obj_{i}", "info": f"state={np.random.randint(0,3)}"}
            for i in range(1, 8)
        ]

    def _gen_dummy_image(self, w: int, h: int, text: str) -> np.ndarray:
        # 简单渐变 + 文本占位 (文本不真正绘制，UI 侧可忽略或自行绘制)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        gx = np.linspace(0, 255, w, dtype=np.uint8)
        gy = np.linspace(0, 255, h, dtype=np.uint8)
        img[..., 0] = gy[:, None]
        img[..., 1] = gx[None, :]
        img[..., 2] = (gy[:, None] // 2 + gx[None, :] // 2)
        return img


if __name__ == "__main__":
    agent_system = EGAgentSystem()
    agent_system.start()
    time.sleep(2)
    agent_system.feed_instruction("测试指令")
    time.sleep(2)
    agent_system.stop()
