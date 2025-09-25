# GUI Application
- the user interface to interact with the agent
- runtime visualization of map, instance, behavior_tree, ... 
- you can use [AgentSystem_to_test_gui](system_example.py) to only test the GUI layout, replace in [main.py](main.py)




### Build prompts (Use ChatGPT5)

Attempt 1:
> 我要写一个 生成式行为树具身智能体 的用户交互界面，请参考我给的界面设计图 gui_design.png，使用PyQt5写界面代码 window.ui，可以通过 ui2py.sh 转化为 window.py，并确保界面和设计图一致。

> 功能上，我的智能体是由这个 system.py 中的 EGAgentSystem 类来控制的，界面的启动和停止按钮控制智能体的运行。然后给我增加实现界面所需各种交互功能，在和 ui2py.sh 同一个目录下，新增功能实现的文件，太多的话可以分不同文件不同模块写。然后有需要的话完善我的 EGAgentSystem 类以满足要求。界面所需的各种图像、文字、列表等信息，可以都给我先在 EGAgentSystem 预留接口，等我后续自行补上。最后按你的理解，差异化各个小窗信息的更新频率，并美化界面。

Attempt 2:
我现在system.py还没开发完，能给我写个用于简单测试的system_test.py在main.py同目录下吗，能使main.py运行起界面来就行，随便 显示些什么信息在各框里

Attempt 3:
把界面给我改成默认全屏显示

Attempt 4:
当我点击 启动智能体 按钮时，报错
QObject: Cannot create children for a parent that is in a different thread.
(Parent is QTextDocument(0x6075f916ce20), parent's thread is QThread(0x6075f8d8caa0), current thread is QThread(0x7b5a80000d70)
Segmentation fault (core dumped)

Attempt 5:
给我写一下这个GUI所需agent_system的标准接口要求，我放进readme



### GUI 所需 agent_system 标准接口 (GPT生成，供参考)

本 GUI 通过一个名为 EGAgentSystem 的对象进行交互。任何实现均需满足以下接口与约定，以便可直接替换到 app/main.py 中使用。

#### 1) 生命周期与状态
- EGAgentSystem()
  - 构造函数不应做重负载初始化（如需可延迟到 start）。
- start() -> None
  - 启动内部工作线程/循环；可重入安全：重复调用应无副作用。
- stop() -> None
  - 停止内部线程/循环，尽快返回（建议 <1s）；可重入安全。
- status -> bool (property)
  - True 表示运行中；False 表示已停止。
- 可选: finished -> bool (property)
  - 任务完成态；GUI 当前未使用。

#### 2) 事件监听（线程间通信）
- add_listener(kind: str, cb: Callable[[Any], None]) -> None
  - 注册监听器；cb(data) 可在系统内部线程中被调用（跨线程）。
  - GUI 会在回调中将数据转为 Qt Signal 再在主线程更新 UI，故回调中请勿直接做 Qt UI 操作，也不要阻塞。
- 事件种类与数据约定：
  - "log": str                最新日志长文本（可整段发送）
  - "conversation": str       最新对话长文本（可整段发送）
  - "entities": Iterable[Tuple[Any, Any]]
      - 迭代器或列表，元素为 (name, info)，GUI 会转为字符串显示两列
  - "status": bool            运行状态（True/False）

建议但非强制的触发频率（仅供实现参考）：
- log/conversation: 有增量时触发即可
- entities: ~1s
- status: 状态变化时

#### 3) 命令输入
- feed_instruction(text: str) -> None
  - 接收用户输入指令；实现可以在内部进行解析并触发相关事件（如 conversation/log）。
- 可选: feed_observation(pose: np.ndarray, intrinsics: np.ndarray, image: np.ndarray, depth: np.ndarray) -> None
  - 供后续从外部注入观测；GUI 当前未调用。

#### 4) 文本数据获取（拉取式）
- get_conversation_text() -> str
  - 返回可显示的对话完整文本。
- get_log_text_tail(n: int = 400) -> str
  - 返回最近 n 行或合适长度的日志文本。

要求：
- 方法需快速返回（建议 <10ms），不可阻塞。
- 发生错误时返回上一次可用结果或空字符串，不抛出异常到 GUI。

#### 5) 实体列表获取（拉取式）
- get_entity_rows() -> Iterable[Tuple[Any, Any]]
  - 返回 (name, info) 的迭代器/列表；GUI 以两列表格显示，内部会对任意类型调用 str()。

#### 6) 图像数据获取（拉取式）
以下方法均返回 numpy.ndarray，dtype=uint8，形状 HxWx3，颜色空间为 RGB（GUI 内部会做 rgbSwapped 以适配 Qt）。
- get_semantic_map_image() -> np.ndarray         语义/路径可视化（低频 ~3s）
- get_traversable_map_image() -> np.ndarray      可通行地图（中频 ~1s）
- get_current_instance_seg_image() -> np.ndarray 当前视野实例分割（高频 ~200ms）
- get_current_instance_3d_image() -> np.ndarray  3D 实例可视化（低频 ~3s）
- get_entity_bt_image() -> np.ndarray            行为树图（低频 ~5s）

要求：
- 尺寸可变，GUI 会自动缩放。
- 如当前无新图，可返回上次缓存；如不可用可返回 None（GUI 会显示空图），但推荐始终返回有效图避免闪烁。
- 方法需快速返回（建议 <10ms）。

#### 7) 线程与性能约束
- start/stop 内部使用线程执行主循环或订阅回调，避免在 GUI 线程中阻塞。
- add_listener 回调可在工作线程触发；请勿在回调中做耗时操作或 UI 操作。
- 所有 get_* 拉取方法应为非阻塞、快速返回；可返回缓存的数据。
- stop() 应终止所有子线程/任务，确保进程可正常退出。

#### 8) 最小接口清单（签名）
```python
class EGAgentSystem:
    # lifecycle
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def status(self) -> bool: ...

    # listeners
    def add_listener(self, kind: str, cb) -> None: ...

    # commands
    def feed_instruction(self, text: str) -> None: ...
    # optional:
    def feed_observation(self, pose, intrinsics, image, depth) -> None: ...

    # pull text
    def get_conversation_text(self) -> str: ...
    def get_log_text_tail(self, n: int = 400) -> str: ...

    # pull entities
    def get_entity_rows(self):  # Iterable[Tuple[Any, Any]]
        ...

    # pull images (np.uint8, HxWx3, RGB)
    def get_semantic_map_image(self): ...
    def get_traversable_map_image(self): ...
    def get_current_instance_seg_image(self): ...
    def get_current_instance_3d_image(self): ...
    def get_entity_bt_image(self): ...
```

#### 9) 行为建议
- 出错时写入日志并尽量返回上次有效数据，而非抛异常到 GUI。
- 图片、文本的缓存与更新频率可以内部自定；上层 GUI 有定时器节流。
- 可按需扩展更多事件类型，但请保持以上最小接口稳定。