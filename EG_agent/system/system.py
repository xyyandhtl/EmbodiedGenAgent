import numpy as np
import threading
import time
from typing import Callable, Dict, List, Any, Iterable
from pathlib import Path
from PIL import Image
from dynaconf import Dynaconf

from EG_agent.prompts.default_objects import AllObject
from EG_agent.reasoning.logic_goal import LogicGoalGenerator
from EG_agent.planning.bt_planner import BTGenerator
from EG_agent.planning.btpg import BehaviorTree
from EG_agent.vlmap.vlmap import VLMapNav
from EG_agent.system.envs.isaacsim_env import IsaacsimEnv
from EG_agent.system.module_path import AGENT_SYSTEM_PATH


class EGAgentSystem:
    """EGAgentSystem
    Orchestrates goal parsing, behavior tree planning, and environment execution.
    Emits lightweight UI events (conversation, log, entities, status) for the app.
    Note: This change only beautifies comments/docstrings; no logic changes.
    """
    goal: str
    goal_set: set
    bt: BehaviorTree
    bt_path: str = "behavior_tree.png"

    def __init__(self):
        """Initialize generators, environment runtime, and UI caches."""
        # 逻辑 Goal 生成器：用于将用户的自然语言指令（如“找到椅子”）解析为结构化的 逻辑目标
        self.goal_generator = LogicGoalGenerator()
        # 使用 goal_generator 的一个预定义 object set，同时给下面 bt_generator，供简单调试
        self.goal_generator.prepare_prompt(object_set=None)

        # 行为树规划器：用于根据逻辑目标，动态地生成一个 BT
        self.bt_generator = BTGenerator(env_name="embodied",
                                        cur_cond_set=set(),
                                        key_objects=list(AllObject))
        # TODO: 连续任务时需即时更新 cur_cond_set 和 key_objects

        # Agent 载体部署环境
        # 组成：通过 bint_bt 动态绑定行为树，定义 run_action 实现交互逻辑，通过 ROS2 与部署环境信息收发
        # 运行：行为树叶节点在被 tick 时，通过调用其绑定的 env 的 run_action 实现智能体到部署环境交互的 action
        self.env = IsaacsimEnv()
        cfg_path = f"{AGENT_SYSTEM_PATH}/agent_system.yaml"
        self.cfg = Dynaconf(settings_files=[cfg_path], lowercase_read=True, merge_enabled=False)

        # VLM 语义地图导航后端
        # 创建 VLMapNav 并挂到 IsaacsimEnv，避免 env 内部发布计时器过早触发
        self.vlmap_backend = VLMapNav()
        self.env.set_vlmap_backend(self.vlmap_backend)
        self.dm = self.vlmap_backend.dualmap    # a reference for fast access

        # 再配置 ROS（内部会创建订阅与可选的发布计时器）
        self.env.configure_ros(self.cfg)

        # 运行控制
        self._running = False
        self._is_finished = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # QT监听器: kind -> list[callback(data)]
        self._listeners: Dict[str, List[Callable[[Any], None]]] = {}

        # QT 缓存/占位数据
        self._conversation: List[str] = []
        self._logs: List[str] = []
        # self._entity_info: List[dict] = []   # each: {"name":..., "info":...}
        self._last_bt_image: np.ndarray = self._gen_dummy_image(300, 400, "Behavior Tree")
        self._placeholder_counter = 0

    # ---------------------- 控制接口 ----------------------
    def start(self):
        """Start the agent loop in a background thread."""
        if self._running:
            self._log("Agent system already running.")
            return
        self._log("Agent system started.")
        self._stop_event.clear()
        self._running = True
        self._is_finished = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Request stop and close environment safely."""
        if not self._running:
            self._log("Agent system not running.")
            return
        self._log("Agent system stop requested.")
        self._stop_event.set()
        self._running = False
        self.env.close()

    def save(self, map_path: str | None = None):
        """Save the current global map to the given path or default cfg.map_save_path."""
        path = map_path or getattr(self.vlmap_backend.cfg, "map_save_path", None)
        self._log(f"Saving current global map to: {path}")
        self.dm.save_map(map_path=path)
        self._log("Map saved.")

    def load(self, map_path: str | None = None):
        """Load the global map from the given path or default cfg.preload_path."""
        path = map_path or getattr(self.vlmap_backend.cfg, "preload_path", None)
        self._log(f"Loading global map from: {path}")
        self.dm.load_map(map_path=path)
        self._log(f"Map loaded with {len(self.dm.global_map_manager.global_map)} objects and "
                  f"{len(self.dm.global_map_manager.layout_map.point_cloud.points)} layout points"
                  f" and {len(self.dm.global_map_manager.layout_map.wall_pcd.points)} wall points")
        self.update_objects_from_map()
        
    def update_objects_from_map(self):
        entites = self.get_entity_rows()
        if entites:
            # 取所有 entity 的 name 并首字母大写组成 object_set
            self.bt_objects = {name.split('/', 1)[1].capitalize() for name, _ in entites if name.startswith("global/")}
            self._log(f"Prepared goal generator from loaded map with the following "
                      f"{len(self.bt_objects)} objects: \n{self.bt_objects} .")
            # 用这些地图 objects 更新 goal_generator 和 bt_generator 的 key_objects
            self.goal_generator.prepare_prompt(self.bt_objects)
            self.bt_generator.set_key_objects(list(self.bt_objects))
            self._log(f"已设置原子动作: \n{[action.name for action in self.bt_generator.planner.actions]}")
            chinese_objs = self.goal_generator.ask_question(
                f"请用中文列出以下英文目标集合：{self.bt_objects}", use_system_prompt=False)
            self._append_conversation(f"智能体: 可以参考的任务对象有 {chinese_objs}")
            self._emit("conversation", self.get_conversation_text())

    def _run_loop(self):
        """Main loop: step environment, propagate events, and check completion."""
        # 主执行循环 (简化)
        self.env.reset()
        self._log("Environment reset.")
        is_finished = False
        while not self._stop_event.is_set() and not is_finished:
            time.sleep(0.1)

            # 后端地图/导航处理一次
            # (1) 检查 synced_data_queue 中的最新帧是否为 关键帧
            # (2) dualmap.parallel_process 处理该关键帧（Detector 对图像生成物体观测结果；更新地图并计算导航路径（全局+局部））
            self.vlmap_backend.run_once(lambda: time.time())

            # 环境 step，行为树 agent 执行动作，和部署环境通信
            # is_finished = self.env.step()

        self.dm.end_process()
        self.vlmap_backend.shutdown_requested = True

        self._is_finished = True
        self._running = False
        self._log("Agent loop stopped.")
        self._emit("status", self.status)

    @property
    def finished(self) -> bool:
        """Whether the agent loop has finished."""
        return self._is_finished

    @property
    def status(self) -> bool:
        """Running state; True when the loop thread is active."""
        return self._running

    # ---------------------- 监听/事件 ----------------------
    def add_listener(self, kind: str, cb: Callable[[Any], None]):
        """Register an event listener for a given kind."""
        self._listeners.setdefault(kind, []).append(cb)

    def _emit(self, kind: str, data: Any):
        """Emit an event to all listeners of the given kind."""
        for cb in self._listeners.get(kind, []):
            try:
                cb(data)
            except Exception:
                pass

    # ---------------------- 输入接口 ----------------------
    def feed_instruction(self, text: str):
        """Plan a behavior tree from instruction, draw it, and update UI caches."""
        # text = "请前往拍摄车辆"   ＃ for test
        self.goal = self.goal_generator.generate_single(text)  # 1. 用户指令 ==> goal 逻辑指令（如：请前往控制室 转换为 RobotNear_ControlRoom）
        # self.goal = "RobotNear_Equipment"  # 调试测试
        self._log(f"[system] [feed_instruction] goal is: {self.goal}")

        if self.goal:
            self.bt = self.bt_generator.generate(self.goal)  # 2. goal 逻辑指令 ==> BehaviorTree 实例（如：['Walk(ControlRoom)']）
            self._log(f"[system] [feed_instruction] BT is created!")
            self.goal_set = self.bt_generator.goal_set

            self.update_cur_goal_set()  # 3. 将 BT 与 IsaacsimEnv环境交互层 绑定；调用 vlmap 查询每个目标的位置；将目标位置发送给 IsaacsimEnv，并刷新可视性

            # save behavior_tree.png to the current work dir to visualize in gui
            self.bt.draw(png_only=True)
            # load the saved behavior tree image into memory for GUI
            img_path = Path(self.bt_path)
            if img_path.exists():
                self._last_bt_image = np.array(Image.open(img_path).convert("RGB"))
                self._log(f"Behavior tree image updated: {img_path.resolve()}")
            else:
                self._log(f"Behavior tree image not found: {img_path.resolve()}")
        else:
            self._log(f"User instruction: {text}")
            self._append_conversation(f"用户: {text}")
            self._append_conversation(f"智能体: 无法理解指令")
            self._emit("conversation", self.get_conversation_text())
            return
        
        self._log(f"User instruction: {text}")
        self._append_conversation(f"用户: {text}")
        self._append_conversation(f"智能体: {self.goal}")
        self._emit("conversation", self.get_conversation_text())

    # ---------------------- 模块间数据交互 -----------------------
    def update_cur_goal_set(self):
        """After feed_instruction, bind bt to self.env, query the target object positions"""
        # (1) 将 BT 与 IsaacsimEnv环境交互层 绑定
        self.env.bind_bt(self.bt)

        # (2) 调用 vlmap，查询每个目标在地图中的具体坐标
        self._log(f"[system] [update_cur_goal_set] self.goal_set: {self.goal_set}")
        cur_goal_places = {}
        for obj in self.goal_set:
            self._append_conversation(f"智能体: 正在从地图索引目标 {obj}")
            self._emit("conversation", self.get_conversation_text())
            target_position = self.vlmap_backend.query_object(obj)
            self._log(f"[system] [update_cur_goal_set] query {obj} result: {target_position}")
            if target_position is not None:
                cur_goal_places[obj] = target_position

        # (3) 将 目标位置 传递给 IsaacsimEnv，并更新可视性（每个目标是否在当前相机的视锥内）
        if cur_goal_places:
            self.env.set_object_places(cur_goal_places)

    # ---------------------- 数据获取占位接口 ----------------------
    def get_conversation_text(self) -> str:
        """Return recent conversation text for UI."""
        return "\n".join(self._conversation[-400:])

    def get_log_text_tail(self, n: int = 400) -> str:
        """Return recent log tail for UI."""
        return "\n".join(self._logs[-n:])

    def get_entity_bt_image(self) -> np.ndarray:
        """Return last BT image (or a placeholder if not available)."""
        # 行为树图
        # if self._last_bt_image is None or self._placeholder_counter % 10 == 0:
        #     self._last_bt_image = self._gen_dummy_image(300, 400, "Behavior Tree")
        return self._last_bt_image

    def get_entity_rows(self) -> Iterable[tuple]:
        """Yield (name, info) tuples using dualmap content (local/global)."""
        rows = []
        classes = self.dm.visualizer.obj_classes.get_classes_arr()

        # Local objects summary
        # for local_obj in dm.local_map_manager.local_map[:20]:
        #     cid = local_obj.class_id
        #     name = classes[cid] if 0 <= cid < len(classes) else f"class_{cid}"
        #     pts = np.asarray(local_obj.pcd.points)
        #     npts = pts.shape[0]
        #     if npts > 0:
        #         mean = pts.mean(axis=0)
        #         mean_str = f"{mean[0]:.2f},{mean[1]:.2f},{mean[2]:.2f}"
        #     else:
        #         mean_str = "NA"
        #     rows.append((f"local/{name}", f"id={cid}, npts={npts}, pos=({mean_str})"))

        # Global objects summary
        for global_obj in self.dm.global_map_manager.global_map:
            cid: int = global_obj.class_id
            name = classes[cid] if 0 <= cid < len(classes) else f"class_{cid}"
            pts2 = np.asarray(global_obj.pcd_2d.points)
            npts2 = pts2.shape[0]
            if npts2 > 0:
                mean2 = pts2.mean(axis=0)
                mean2_str = f"{mean2[0]:.2f},{mean2[1]:.2f},{mean2[2]:.2f}"
            else:
                mean2_str = "NA"
            rows.append((f"global/{name}", f"id={cid}, npts={npts2}, pos=({mean2_str})"))

        return rows

    def get_semantic_map_image(self) -> np.ndarray:
        """Semantic/path map from dualmap; fallback to detector annotated image."""
        semantic_map = self.dm.get_semantic_map_image()
        # TODO: 语义实体地图 + 导航路径 (not at all)
        if semantic_map is not None:
            return semantic_map
        return self._gen_dummy_image(400, 300, "Semantic+Path")

    def get_traversable_map_image(self) -> np.ndarray:
        """Traversable map from dualmap."""
        traversable_map = self.dm.get_traversable_map_image()
        # TODO: 可通行地图 (not work)
        if traversable_map is not None:
            return traversable_map
        return self._gen_dummy_image(260, 180, "Traversable")

    def get_current_instance_seg_image(self) -> np.ndarray:
        """Current 2D instance segmentation from dualmap.detector FastSAM."""
        if self.dm.detector.annotated_image is not None:
            return self.dm.detector.annotated_image
        return self._gen_dummy_image(260, 180, "Instance 2D")

    def get_current_instance_3d_image(self) -> np.ndarray:
        """3D instance visualization image from dualmap."""
        # TODO: 3D实例分割可视化
        # if dm.visualizer.last_instance3d_image is not None:
        #     return dm.visualizer.last_instance3d_image
        return self._gen_dummy_image(260, 180, "Instance 3D")

    # ---------------------- 占位内部工具 ----------------------
    def _append_conversation(self, line: str):
        """Append a line to the conversation buffer with trimming."""
        self._conversation.append(line)
        if len(self._conversation) > 2000:
            self._conversation = self._conversation[-1000:]

    def _log(self, line: str):
        """Append a timestamped line to logs and emit 'log' event."""
        ts = time.strftime("%H:%M:%S")
        self._logs.append(f"[{ts}] {line}")
        if len(self._logs) > 5000:
            self._logs = self._logs[-3000:]
        self._emit("log", self.get_log_text_tail())

    def _gen_dummy_image(self, w: int, h: int, text: str) -> np.ndarray:
        """Generate a simple gradient placeholder image of shape (h, w, 3)."""
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
