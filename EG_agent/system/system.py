import numpy as np
import threading
import time
from typing import Callable, Dict, List, Any, Iterable
from pathlib import Path
from PIL import Image
from dynaconf import Dynaconf
import traceback

from EG_agent.reasoning.logic_goal import LogicGoalGenerator
from EG_agent.planning.bt_planner import BTGenerator
from EG_agent.planning.btpg import BehaviorTree
from EG_agent.vlmap.vlmap import VLMapNav
from EG_agent.vlmap.dualmap.core import Dualmap
from EG_agent.system.envs.isaacsim_env import IsaacsimEnv
from EG_agent.system.module_path import AGENT_SYSTEM_PATH


class EGAgentSystem:
    """EGAgentSystem
    Orchestrates goal parsing, behavior tree planning, and environment execution.
    Emits lightweight UI events (conversation, log, entities, status) for the app.
    Note: This change only beautifies comments/docstrings; no logic changes.
    """
    goal: str
    target_set: set = set()
    bt: BehaviorTree | None = None
    bt_name: str = "behavior_tree"

    def __init__(self):
        """Initialize generators, environment runtime, and UI caches."""
        # QT监听器与缓存
        self._listeners: Dict[str, List[Callable[[Any], None]]] = {}
        self._conversation: List[str] = ["智能体: 请先创建后台"]
        self._logs: List[str] = []
        self._last_bt_image: np.ndarray = self._gen_dummy_image(300, 400, "Behavior Tree")
        self._placeholder_counter = 0

        # 逻辑 Goal 生成器：用于将用户的自然语言指令（如“找到椅子”）解析为结构化的 逻辑目标
        self.goal_generator = LogicGoalGenerator()
        self.goal_generator.prepare_prompt(object_set=None)
        self._log(f"goal_generator prompt: \n{self.goal_generator.prompt_scene}")

        # 行为树规划器：用于根据逻辑目标，动态地生成一个 BT。连续任务时需即时更新 cur_cond_set 和 key_objects
        self.bt_generator = BTGenerator(env_name="embodied",
                                        cur_cond_set=set(),
                                        key_objects=[])

        # Agent 载体部署环境
        # 组成：通过 bint_bt 动态绑定行为树，定义 run_action 实现交互逻辑，通过 ROS2 与部署环境信息收发
        # 运行：行为树叶节点在被 tick 时，通过调用其绑定的 agent_env.run_action 实现智能体到部署环境交互的 action
        self.agent_env = IsaacsimEnv()
        cfg_path = f"{AGENT_SYSTEM_PATH}/agent_system.yaml"
        self.cfg = Dynaconf(settings_files=[cfg_path], lowercase_read=True, merge_enabled=False)

        # VLM 语义地图导航后端 -> 延迟创建，由 GUI 按钮触发
        self.vlmap_backend: VLMapNav = None
        self.dm: Dualmap = None

        # 运行控制
        self._running = False
        self._is_finished = False
        self._thread_bt: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._emit("conversation", self.get_conversation_text())

    # ---------------------- 状态查询 ----------------------
    @property
    def backend_ready(self) -> bool:
        return self.vlmap_backend is not None and self.dm is not None
    
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
        """Register an event listener for a given kind and immediately replay latest cached state."""
        self._listeners.setdefault(kind, []).append(cb)
        # 注册后立刻推送一次当前缓存数据，避免首次 emit 早于监听器注册而丢失
        try:
            if kind == "conversation":
                cb(self.get_conversation_text())
            elif kind == "log":
                cb(self.get_log_text_tail())
            elif kind == "entities" and self.backend_ready:
                cb(list(self.get_entity_rows()))
            elif kind == "status":
                cb(self.status)
        except Exception:
            pass

    def _emit(self, kind: str, data: Any):
        """Emit an event to all listeners of the given kind."""
        for cb in self._listeners.get(kind, []):
            try:
                cb(data)
            except Exception:
                pass

    # ---------------------- 控制接口 ----------------------
    def create_backend(self) -> bool:
        """Create the VLMapNav backend and configure ROS subscriptions."""
        if self.vlmap_backend is not None:
            self._log_warn("Backend already created.")
            return True
        self._conv("智能体: 正在创建后台，请稍候...")
        self._log_info("Creating backend...")
        self.vlmap_backend = VLMapNav()
        self.agent_env.set_vlmap_backend(self.vlmap_backend)
        # dualmap reference, set after backend is created
        self.dm = self.vlmap_backend.dualmap
        # 再配置 ROS（内部会创建订阅与可选的发布计时器）
        self._log_info("Configuring ROS...")
        self.agent_env.configure_ros(self.cfg)
        self._log_info("Backend created successfully.")
        self._conv_info("后台创建成功，ROS2通信已配置，请启动智能体。")
        # For test goal_inview
        # self.agent_env.set_object_places({"flag1": [5, 0, 0]})
        # self.agent_env.set_object_places({"flag2": [0, 5, 0]})
        # self.agent_env.run_action("mark", (5, 0, 0))
        # self.agent_env.run_action("mark", (0, 5, 0))
        return True
    
    def start(self):
        """Start the agent loop in a background thread."""
        if self._running:
            self._log_warn("Agent system already running.")
            return
        self._log_info("Agent system started.")
        if self.backend_ready:
            self.dm.start_threading()
            self._conv_info(f"检测和建图线程已启动。")

        self._stop_event.clear()
        self._running = True
        self._is_finished = False
        self._log_info("Environment reset.")
        self._thread_bt = threading.Thread(target=self._run_loop_bt, daemon=True)
        self._thread_bt.start()

        self._emit("status", self.status)

    def stop(self):
        """Request stop and close environment safely."""
        if not self._running:
            self._log_warn("Agent system not running.")
            return
        self._log_info("Agent system stop requested.")
        self._stop_event.set()
        if self._thread_bt.is_alive():
            self._thread_bt.join()
        if self.backend_ready:
            self.dm.end_process()
        self._is_finished = True
        self._running = False
        self._log_info("Agent loop stopped.")
        self._emit("status", self.status)

    def save(self, map_path: str | None = None):
        """Save the current global map to the given path or default cfg.map_save_path."""
        if not self.backend_ready:
            self._log_warn("Backend not created, cannot save map.")
            return
        path = map_path or getattr(self.vlmap_backend.cfg, "map_save_path", None)
        self._log_info(f"Saving current global map to: {path}")
        self.dm.save_map(map_path=path)
        self._log_info("Map saved.")

    def load(self, map_path: str | None = None):
        """Load the global map from the given path or default cfg.preload_path."""
        if not self.backend_ready:
            self._log_warn("Backend not created, cannot load map.")
            return
        path = map_path or getattr(self.vlmap_backend.cfg, "preload_path", None)
        self._log_info(f"Loading global map from: {path}")
        self.dm.load_map(map_path=path)
        self._log_info(f"Map loaded with {len(self.dm.global_map_manager.global_map)} objects and "
                       f"{len(self.dm.global_map_manager.layout_map.point_cloud.points)} layout points"
                       f" and {len(self.dm.global_map_manager.layout_map.wall_pcd.points)} wall points")
        self.update_objects_from_map()

        # For quick test, directly set a goal pose
        # self.vlmap_backend.get_global_path(goal_pose=np.array([3.5, 6.0, 0.0]))
        # self._log(f"Computed global_path: {self.dm.curr_global_path}")

        self._conv_info("地图加载完成。")

    def get_last_tick_output(self) -> str:
        return self.agent_env.last_tick_output

    def get_goal_inview(self) -> dict:
        return self.agent_env.goal_inview

    def get_cur_cmd_vel(self) -> tuple:
        """返回当前计算的速度命令 (vx, vy, wz)"""
        return self.agent_env.cur_cmd_vel

    def get_agent_pose(self) -> tuple:
        """返回当前机器人位姿 [x,y,z,qw,qx,qy,qz]"""
        return tuple(self.dm.realtime_pose[:3, 3]) if self.backend_ready else (0, 0, 0)

    def get_cur_target_pos(self) -> list:
        """返回当前所有目标的全局位置 {target_name: [x,y,z]}"""
        return self.agent_env.get_cur_target_pos()

    def _run_loop_bt(self):
        """BT Main loop: step environment, propagate events, and check completion."""
        try:
            self.agent_env.reset()
            bt_task_finshed = False
            while not self._stop_event.is_set() and not bt_task_finshed:
                bt_task_finshed = self.agent_env.step()
                if self.agent_env.tick_updated:
                    self._conv_info(f"行为树执行节点更新: {self.get_last_tick_output()}")
                    self.agent_env.tick_updated = False
        except Exception as e:
            tb = traceback.format_exc()
            self._log_error(f"Run loop_bt exception: {e}")
            self._log(tb)
            self._conv_err("[行为树]运行循环异常，已停止。请查看日志窗口。")

    def _run_loop_backend(self):
        """(已弃用,移至后台线程) Backend loop: backend run_once detector frontend"""
        try:
            while not self._stop_event.is_set():
                # time.sleep(0.01)
                # VLM 建图后台处理一帧数据
                self.vlmap_backend.run_once()
        except Exception as e:
            tb = traceback.format_exc()
            self._log_error(f"Run loop_backend exception: {e}")
            self._log(tb)
            self._conv_err("[地图后台]运行循环异常，已停止。请查看日志窗口。")

    # ---------------------- 输入接口 ----------------------
    def feed_instruction(self, text: str):
        """Plan a behavior tree from instruction, draw it, and update UI caches."""
        # 1. 用户指令 ==> goal 逻辑指令（如：请前往控制室 转换为 RobotNear_ControlRoom）
        self._conv(f"用户: {text}")
        self.goal = self.goal_generator.generate_single(text)
        # self.goal = "RobotNear_Equipment"  # 调试测试
        self._log_info(f"User instruction: {text}")
        self._log_info(f"[system] [feed_instruction] goal is: {self.goal}")
        self._conv(f"智能体: 已解析指令为逻辑目标: {self.goal}")
        if not self.goal:
            self._conv_err("无法理解指令，已中断指令下发")
            return

        # 2. goal 逻辑指令 ==> BehaviorTree 实例并可视化
        self.bt_generator.set_goal(self.goal)
        goal_candidates = self.bt_generator.goal_candidates
        self._log_info(f"[system] [feed_instruction] goal_candidates is: {goal_candidates}")
        self.agent_env.cur_goal_set = goal_candidates[0]
        # Extract target objects from goal candidates
        for goal_set in goal_candidates:
            self.target_set.update( { self.agent_env.extract_targets(goal) for goal in goal_set })
        if any(x is None for x in self.target_set):
            self._conv_err("无法理解指令中的目标对象，已中断指令下发")
            return
        self._log_info(f"[system] [feed_instruction] target_set is: {self.target_set}")
        self.bt_generator.set_key_objects(list(self.target_set))
        self._conv(f"智能体: 准备从逻辑目标生成以 {self.target_set} 为目标集的行为树")
        # Generate BehaviorTree
        self.bt = self.bt_generator.generate(btml_name="tree")
        if not self.bt:
            self._conv_err("无法解析指令生成行为树，已中断指令下发")
            return
        self._log_info(f"[system] [feed_instruction] Generated BehaviorTree: {self.bt}")
        self.bt.draw(png_only=True, file_name=self.bt_name)
        # load the saved behavior tree image into memory for GUI
        img_path = Path(self.bt_name + ".png")
        self._last_bt_image = np.array(Image.open(img_path).convert("RGB"))
        self._log(f"Behavior tree image updated: {img_path.resolve()}")
        self._conv_info("已生成行为树并显示在左下窗口")

        # 3-pre. 检查 VLMap 后台是否已创建, 若未创建, 则直接返回
        if not self.backend_ready:
            self._log_warn("[system] Backend not created, cannot query object positions.")
            self._conv_warn("VLMap后台未创建，无法执行行为树，请先点击右侧“创建后台”。")
            return

        # 3. 将 BT 与 IsaacsimEnv环境交互层 绑定
        self.agent_env.bind_bt(self.bt)
        self._log_info("[system] [feed_instruction] Binding BT to agent_env.")

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

    def update_objects_from_map(self):
        if not self.backend_ready:
            return
        entites = self.get_entity_rows()
        if entites:
            self.bt_objects = {name.split('/', 1)[1].capitalize() for name, _ in entites if name.startswith("global/")}
            self.goal_generator.prepare_prompt(self.bt_objects)
            # self._log(f"Updated goal_generator with scene_prompt: {self.goal_generator.prompt_scene}")
            # self.bt_generator.set_key_objects(list(self.bt_objects))
            # self._log(f"已设置原子动作: \n{[action.name for action in self.bt_generator.planner.actions]}")
            if self.cfg['caring_mode']:
                chinese_objs = self.goal_generator.ask_question(
                    f"请用中文列出以下英文目标集合：{self.bt_objects}", use_system_prompt=False)
                self._conv(f"智能体: 可以参考的任务对象有 {chinese_objs}")

    def get_semantic_map_image(self) -> np.ndarray:
        """Semantic/path map from dualmap; fallback to detector annotated image."""
        semantic_map = self.dm.global_map_manager.get_semantic_map_image()
        if semantic_map is not None:
            self.update_objects_from_map()
            return semantic_map
        return self._gen_dummy_image(400, 300, "Semantic+Path")

    def get_traversable_map_image(self) -> np.ndarray:
        """Traversable map from dualmap."""
        traversable_map = self.dm.global_map_manager.get_traversable_map_image()
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

    # 简洁的对话分级包装
    def _conv(self, line: str):
        """Append a line to the conversation buffer with trimming."""
        self._append_conversation(line)
        self._emit("conversation", self.get_conversation_text())

    def _conv_err(self, message: str):
        """Emit an agent error message with a tag for GUI styling."""
        self._conv(f"智能体: [!error] {message}")

    def _conv_warn(self, message: str):
        """Emit an agent warning message with a tag for GUI styling."""
        self._conv(f"智能体: [!warn] {message}")

    def _conv_info(self, message: str):
        """Emit an agent info message with a tag for GUI styling."""
        self._conv(f"智能体: [!info] {message}")

    def _log(self, line: str):
        """Append a timestamped line to logs and emit 'log' event."""
        ts = time.strftime("%H:%M:%S")
        self._logs.append(f"[{ts}] {line}")
        if len(self._logs) > 5000:
            self._logs = self._logs[-3000:]
        self._emit("log", self.get_log_text_tail())

    # 简洁的日志分级包装
    def _log_info(self, msg: str):
        self._log(f"[INFO] {msg}")

    def _log_warn(self, msg: str):
        self._log(f"[WARN] {msg}")

    def _log_error(self, msg: str):
        self._log(f"[ERROR] {msg}")

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


