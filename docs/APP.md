# GUI Application

- 用户与具身智能体交互的桌面界面
- 运行时可视化：语义/路径地图、可通行地图、实例分割、行为树等
- 仅测试界面布局可使用示例后端：[AgentSystem_example](system_example.py)，在 [main.py](main.py) 中替换导入

---

## 使用说明

1) 启动
  - `python app/main.py`

2) 载入预建地图（可选）
- 右侧点击“载入地图”，选择已保存的地图目录

3) 发送指令
- 左侧“指令栏”输入并点击“发送”（或回车）

> TODO：支持用户指令自主探索，并在建图中维护行为树 Objects（后续补充）

---

## 后台 AgentSystem 最小接口清单

| 接口 | 形态 | 作用/说明 | 触发/频率 |
|---|---|---|---|
| EGAgentSystem() | ctor | 轻量初始化（重负载可延迟到 start） | — |
| start() / stop() | 方法 | 启动/停止内部工作线程/循环；需可重入 | — |
| status | 属性 bool | True 运行中 / False 已停止 | 状态变化时通知 |
| add_listener(kind, cb) | 方法 | 注册跨线程事件回调，不做 UI 操作与耗时任务 | kind: "log"、"conversation"、"entities"、"status" |
| feed_instruction(text) | 方法 | 接收用户自然语言指令，并驱动内部流程 | 有指令即调用 |
| feed_observation(...) | 方法（可选） | 外部注入观测（位姿/内参/RGB/Depth） | 视集成而定 |
| get_conversation_text() | 方法 | 返回完整对话文本 | 有增量时拉取 |
| get_log_text_tail(n=400) | 方法 | 返回最近日志文本 | 有增量时拉取 |
| get_entity_rows() | 方法 | 返回 (name, info) 可迭代，GUI 两列表显示 | ~1s |
| 地图与图像获取 | 方法组 | RGB uint8, HxWx3：<br>- get_semantic_map_image()<br>- get_traversable_map_image()<br>- get_current_instance_seg_image()<br>- get_current_instance_3d_image()<br>- get_entity_bt_image() | 频率建议：<br>语义/3D ~3s；<br>可通行 ~1s；<br>2D实例 ~200ms；<br>行为树 ~5s |

