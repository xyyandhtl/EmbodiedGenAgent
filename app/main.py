import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, uic
from zoom_widget import ZoomableImageWidget
import threading
import traceback  # 新增：格式化异常堆栈

# 正式运行
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from EG_agent.system.system import EGAgentSystem
# 仅测试界面
# from system_example import EGAgentSystem

APP_ROOT = os.path.abspath(os.path.dirname(__file__))
UI_PATH = os.path.join(APP_ROOT, "window.ui")

def np_to_qpix(img: np.ndarray) -> QtGui.QPixmap:
    if img is None:
        return QtGui.QPixmap()
    if img.ndim == 2:
        h, w = img.shape
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    else:
        h, w, c = img.shape
        if c == 3:
            qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        else:
            img = img[..., :3]
            qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.rgbSwapped())


class MainWindow(QtWidgets.QMainWindow):
    # ---- 新增: 跨线程通信信号 ----
    logSignal = QtCore.pyqtSignal(str)
    convSignal = QtCore.pyqtSignal(str)
    entitiesSignal = QtCore.pyqtSignal(list)   # list of (name, info)
    statusSignal = QtCore.pyqtSignal(bool)
    # 新增：通用任务完成信号（desc, ok, err）
    taskFinished = QtCore.pyqtSignal(str, bool, str)

    def __init__(self):
        super().__init__()

        uic.loadUi(UI_PATH, self)

        self.agent_system = EGAgentSystem()

        # 绑定“创建后台”按钮
        if hasattr(self, "createBackendBtn"):
            self.createBackendBtn.clicked.connect(self.on_create_backend)

        # 连接信号到槽 (主线程更新 UI)
        self.logSignal.connect(self._on_log_update)
        self.convSignal.connect(self._on_conv_update)
        self.entitiesSignal.connect(self._on_entities_update)
        self.statusSignal.connect(self._on_status_update)
        # 新增：任务完成通知
        self.taskFinished.connect(self._on_task_finished)

        # 按钮绑定
        self.startBtn.clicked.connect(self.on_start)
        self.stopBtn.clicked.connect(self.on_stop)
        self.sendInstructionBtn.clicked.connect(self.on_send_instruction)
        self.instructionEdit.returnPressed.connect(self.on_send_instruction)
        # 新增：保存/载入地图
        self.saveMapBtn.clicked.connect(self.on_save_map)
        self.loadMapBtn.clicked.connect(self.on_load_map)

        # 初始化时禁用控制（仅右侧四个按钮）
        self._set_controls_enabled(False)

        # Timers (差异化更新频率)
        self.timer_fast = QtCore.QTimer(self)      # 200ms
        self.timer_fast.timeout.connect(self.update_fast)
        self.timer_fast.start(200)

        self.timer_medium = QtCore.QTimer(self)    # 1s
        self.timer_medium.timeout.connect(self.update_medium)
        self.timer_medium.start(1000)

        self.timer_slow = QtCore.QTimer(self)      # 3s
        self.timer_slow.timeout.connect(self.update_slow)
        self.timer_slow.start(3000)

        self.timer_bt = QtCore.QTimer(self)        # 5s
        self.timer_bt.timeout.connect(self.update_bt)
        self.timer_bt.start(5000)

        # --- 新增：聊天视图初始化（替换对话文本框为聊天气泡） ---
        self._conv_msg_seen = 0  # 改为按消息计数，而不是按行
        self._init_chat_view()

        # 替换原直接更新 UI 的监听器 -> 仅发射信号
        self.agent_system.add_listener("log", lambda data: self.logSignal.emit(data))
        self.agent_system.add_listener("conversation", lambda data: self.convSignal.emit(data))
        self.agent_system.add_listener("entities", lambda data: self.entitiesSignal.emit(list(data)))
        self.agent_system.add_listener("status", lambda data: self.statusSignal.emit(bool(data)))

        self.entityTable.setColumnCount(2)
        self.entityTable.setHorizontalHeaderLabels(["实体", "信息"])
        self.entityTable.horizontalHeader().setStretchLastSection(True)
        self.entityTable.verticalHeader().setVisible(False)

        self.update_statusbar()

        # --- 新增：初始化图像与报告浏览器 ---
        self._setup_image_browser()
        self._setup_report_browser()

        # --- 新增：将 semanticMapLabel ==> semanticMapWidget，以实现用鼠标左键拖拽和滚轮平移 ---
        self.semanticMapWidget = ZoomableImageWidget()
        # Assuming the placeholder is in a layout within its parent
        if self.semanticMapLabel.parentWidget().layout() is not None:
            layout = self.semanticMapLabel.parentWidget().layout()
            index = layout.indexOf(self.semanticMapLabel)
            layout.removeWidget(self.semanticMapLabel)
            self.semanticMapLabel.deleteLater()
            layout.insertWidget(index, self.semanticMapWidget)
            # Set size policies to match what the QLabel might have had
            self.semanticMapWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        else:
            print("Warning: Could not find layout to replace semantic map placeholder.")

        # --- 新增：将 behaviorTreeLabel ==> behaviorTreeWidget，同样支持拖拽/缩放 ---
        self.behaviorTreeWidget = ZoomableImageWidget()
        if hasattr(self, "behaviorTreeLabel") and self.behaviorTreeLabel.parentWidget().layout() is not None:
            bt_layout = self.behaviorTreeLabel.parentWidget().layout()
            bt_index = bt_layout.indexOf(self.behaviorTreeLabel)
            bt_layout.removeWidget(self.behaviorTreeLabel)
            self.behaviorTreeLabel.deleteLater()
            bt_layout.insertWidget(bt_index, self.behaviorTreeWidget)
            self.behaviorTreeWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def on_create_backend(self):
        # 改为后台执行，避免阻塞 UI
        self._run_in_background("创建后台", self.agent_system.create_backend)

    def _set_controls_enabled(self, enabled: bool):
        # 仅控制右侧四个按钮变灰/启用
        for name in ("startBtn", "stopBtn", "saveMapBtn", "loadMapBtn"):
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(enabled)

    # ----------------- UI Events -----------------
    def on_start(self):
        self.agent_system.start()  # 启动 EGAgentSystem 的主线程 _run_loop
        self.update_statusbar()

    def on_stop(self):
        # 可能涉及资源关闭，异步更安全
        self._run_in_background("停止智能体", self.agent_system.stop)

    def on_send_instruction(self):
        # 允许未创建后台时发送指令；后端逻辑会提示无法查询目标位置
        text = self.instructionEdit.text().strip()
        if not text:
            return
        self.instructionEdit.clear()
        # 改为后台执行，避免 BT 生成/IO 阻塞 UI
        self._run_in_background("规划/执行指令", lambda: self.agent_system.feed_instruction(text))

    def on_save_map(self):
        # 默认目录：cfg.map_save_path
        default_dir = getattr(self.agent_system.vlmap_backend.cfg, "map_save_path", "") or ""
        # Dualmap 目前只能保存到预设目录，这里仍然调用后台保存
        self._run_in_background("保存地图", lambda: self.agent_system.save(map_path=default_dir))

    def on_load_map(self):
        # 默认目录：cfg.preload_path
        default_dir = getattr(self.agent_system.vlmap_backend.cfg, "preload_path", "") or ""
        if not default_dir:
            default_dir = os.path.expanduser("~")
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择地图载入文件夹", default_dir)
        if not path:
            return
        self._run_in_background("载入地图", lambda: self.agent_system.load(map_path=path))

    # ----------------- Periodic Updates -----------------
    def update_fast(self):
        if not self.agent_system.backend_ready:
            return
        """高频: 当前视野实例分割"""
        img = self.agent_system.get_current_instance_seg_image()
        self.instanceSegLabel.setPixmap(np_to_qpix(img))

    def update_medium(self):
        if not self.agent_system.backend_ready:
            return
        """中频: 可通行地图 + 实体表"""
        # 可通行地图
        self.traversableMapLabel.setPixmap(
            np_to_qpix(self.agent_system.get_traversable_map_image())
        )
        # 实体表
        self.refresh_entities()

    def update_slow(self):
        if not self.agent_system.backend_ready:
            return
        """低频: 3D实例 + 语义/路径地图"""
        # 3D实例
        self.instance3DLabel.setPixmap(
            np_to_qpix(self.agent_system.get_current_instance_3d_image())
        )
        # 语义地图
        self.semanticMapWidget.setPixmap(
            np_to_qpix(self.agent_system.get_semantic_map_image())
        )

    def update_bt(self):
        # if not self.agent_system.backend_ready:
        #     return
        # 行为树更新 -> ZoomableImageWidget
        if hasattr(self, "behaviorTreeWidget"):
            self.behaviorTreeWidget.setPixmap(
                np_to_qpix(self.agent_system.get_entity_bt_image())
            )

    # ----------------- Slots for signals (主线程执行) -----------------
    def _on_log_update(self, txt: str):
        self.logText.setPlainText(txt)
        self.logText.verticalScrollBar().setValue(self.logText.verticalScrollBar().maximum())

    def _on_conv_update(self, txt: str):
        # 解析为消息组（按“用户:”/“智能体:”开头分组），多行作为单条消息
        if not hasattr(self, "conversationList"):
            return
        groups = self._group_messages(txt)
        if len(groups) < self._conv_msg_seen:
            self._conv_msg_seen = 0
            self.conversationList.clear()
        for msg in groups[self._conv_msg_seen:]:
            self._append_chat_line(msg)  # 传入带角色前缀的整条消息
        self._conv_msg_seen = len(groups)
        self.conversationList.scrollToBottom()

    def _on_entities_update(self, rows):
        self.entityTable.setRowCount(len(rows))
        for r, (name, info) in enumerate(rows):
            self.entityTable.setItem(r, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.entityTable.setItem(r, 1, QtWidgets.QTableWidgetItem(str(info)))

    def _on_status_update(self, running: bool):
        self.update_statusbar()

    # ----------------- Chat view helpers -----------------
    def _init_chat_view(self):
        # 若 UI 未合并，安全跳过
        if not hasattr(self, "conversationList"):
            return
        lw: QtWidgets.QListWidget = self.conversationList
        lw.setSpacing(6)
        lw.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        lw.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        lw.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        lw.setWordWrap(True)
        # 让列表项在宽度变化时能触发我们调整行宽
        lw.installEventFilter(self)

        # 预生成圆形头像
        self._avatar_user = self._make_avatar(QtGui.QColor("#3d7cff"), "你")
        self._avatar_agent = self._make_avatar(QtGui.QColor("#6a7a8a"), "智")

        # 新增：可复用的聊天主题与严重级别标签
        self._severity_tags = {"[!error]": "error", "[!warn]": "warn"}
        self._chat_theme = {
            "user":        {"bg": "#3d7cff", "fg": "#ffffff", "border": "none"},
            "agent":       {"bg": "#3a3f4b", "fg": "#ffffff", "border": "none"},
            "agent_warn":  {"bg": "#5a4b2b", "fg": "#ffd666", "border": "1px solid #ffd666"},
            "agent_error": {"bg": "#5a2a2a", "fg": "#ffb3b3", "border": "2px solid #ff6b6b"},
        }

    def _group_messages(self, txt: str) -> list:
        """将全量对话文本按‘用户:’/‘智能体:’分组，保持单条消息内的换行。"""
        lines = txt.splitlines()
        groups = []
        role = None  # "用户" or "智能体"
        buf = []

        def flush():
            nonlocal role, buf
            if role is None and not buf:
                return
            content = "\n".join(buf).rstrip()
            if role is None:
                role = "智能体"
            if content:
                groups.append(f"{role}: {content}")
            role = None
            buf = []

        for ln in lines:
            if ln.startswith("用户:") or ln.startswith("智能体:"):
                flush()
                if ln.startswith("用户:"):
                    role = "用户"
                else:
                    role = "智能体"
                buf = [ln.split(":", 1)[1].lstrip()]
            else:
                if role is None and not groups:
                    role = "智能体"
                buf.append(ln)
        flush()
        return groups

    def _make_avatar(self, color: QtGui.QColor, text: str) -> QtGui.QPixmap:
        size = 36
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(color)
        p.drawEllipse(0, 0, size, size)
        # 文字
        f = QtGui.QFont()
        f.setPointSize(12)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QtGui.QPen(QtCore.Qt.white))
        p.drawText(pm.rect(), QtCore.Qt.AlignCenter, text[:1])
        p.end()
        return pm

    def _append_chat_line(self, line: str):
        if not hasattr(self, "conversationList"):
            return
        # 解析角色与内容
        is_user = False
        content = line
        if line.startswith("用户:"):
            is_user = True
            content = line.split(":", 1)[1].strip()
        elif line.startswith("智能体:"):
            is_user = False
            content = line.split(":", 1)[1].strip()

        # 新增：解析严重级别标签
        kind = None
        for tag, k in self._severity_tags.items():
            if content.startswith(tag):
                kind = k
                content = content[len(tag):].lstrip()
                break

        widget = self._build_chat_item(content or line, is_user, kind)
        item = QtWidgets.QListWidgetItem()
        # 使每一行占满列表视口宽度，便于将用户气泡推到最右侧边缘
        row_w = self.conversationList.viewport().width() or 600
        item.setSizeHint(QtCore.QSize(row_w, widget.sizeHint().height()))
        self.conversationList.addItem(item)
        self.conversationList.setItemWidget(item, widget)
        # 新增：确保添加后行宽正确（处理偶发的延迟布局）
        self._update_chat_item_widths()

    def _build_chat_item(self, text: str, is_user: bool, kind: str | None = None) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(6, 2, 6, 2)
        h.setSpacing(8)
        w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        avatar_lbl = QtWidgets.QLabel()
        avatar_lbl.setFixedSize(36, 36)
        avatar_lbl.setPixmap((self._avatar_user if is_user else self._avatar_agent))

        bubble = QtWidgets.QLabel(text)
        bubble.setTextFormat(QtCore.Qt.PlainText)
        bubble.setWordWrap(False)
        bubble.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # 新增：根据主题与严重级别设置样式
        if is_user:
            theme = self._chat_theme.get("user")
        else:
            key = "agent" if not kind else f"agent_{kind}"
            theme = self._chat_theme.get(key, self._chat_theme.get("agent"))
        bubble.setStyleSheet(
            "QLabel {"
            f" background: {theme['bg']};"
            f" color: {theme['fg']};"
            f" border: {theme['border']};"
            " border-radius: 8px;"
            " padding: 8px 10px;"
            "}"
        )

        bubble.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        if is_user:
            h.addStretch(1)
            h.addWidget(bubble, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
            h.addWidget(avatar_lbl, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        else:
            h.addWidget(avatar_lbl, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            h.addWidget(bubble, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            h.addStretch(1)
        return w

    # 新增：根据列表视口宽度，更新所有行宽以确保用户消息贴右边缘
    def _update_chat_item_widths(self):
        if not hasattr(self, "conversationList"):
            return
        lw = self.conversationList
        row_w = lw.viewport().width() or lw.width()
        for i in range(lw.count()):
            it = lw.item(i)
            if not it:
                continue
            w = lw.itemWidget(it)
            if not w:
                continue
            it.setSizeHint(QtCore.QSize(row_w, w.sizeHint().height()))

    # 新增：监听 conversationList 的尺寸变化
    def eventFilter(self, obj, event):
        if hasattr(self, "conversationList") and obj is self.conversationList:
            if event.type() == QtCore.QEvent.Resize:
                self._update_chat_item_widths()
        return super().eventFilter(obj, event)

    # ----------------- Legacy helper methods (仍被定时器调用) -----------------
    def refresh_conversation(self):
        # 聊天视图由信号驱动，这里保持兼容：从系统拉取文本并喂给 _on_conv_update
        if hasattr(self, "conversationList"):
            self._on_conv_update(self.agent_system.get_conversation_text())

    def refresh_log(self):
        self._on_log_update(self.agent_system.get_log_text_tail())

    def refresh_entities(self):
        rows = list(self.agent_system.get_entity_rows())
        self._on_entities_update(rows)

    def update_statusbar(self):
        if not hasattr(self, "statusbar"):
            return
        if not self.agent_system.backend_ready:
            self.statusbar.showMessage("智能体状态: 未创建后台")
            return
        state = "运行中" if self.agent_system.status else "已停止"
        self.statusbar.showMessage(f"智能体状态: {state}")

    # --- 新增：图像浏览器初始化与行为 ---
    def _setup_image_browser(self):
        # 允许 UI 里不存在时安全跳过（若未合并 .ui 修改）
        if not hasattr(self, "imageList"):
            return

        # 配置列表视图（在 .ui 已设定，这里确保一致）
        self.imageList.setViewMode(QtWidgets.QListView.IconMode)
        self.imageList.setIconSize(QtCore.QSize(96, 96))
        self.imageList.setResizeMode(QtWidgets.QListView.Adjust)
        self.imageList.setMovement(QtWidgets.QListView.Static)
        self.imageList.setWrapping(True)
        self.imageList.setSpacing(8)
        self.imageList.setDragEnabled(True)
        self.imageList.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        # 连接信号
        if hasattr(self, "browseImageDirBtn"):
            self.browseImageDirBtn.clicked.connect(self.on_browse_image_dir)
        if hasattr(self, "imageDirEdit"):
            self.imageDirEdit.returnPressed.connect(self.on_image_dir_entered)
        self.imageList.itemDoubleClicked.connect(self.on_image_double_clicked)

        # 监视文件夹变更
        self._imageWatcher = QtCore.QFileSystemWatcher(self)
        self._imageWatcher.directoryChanged.connect(self._on_image_dir_changed)

        # 选择默认目录并加载
        self._init_default_image_dir()

    def _init_default_image_dir(self):
        candidates = [
            os.path.join(APP_ROOT, "captured"),
            os.path.expanduser("~/Pictures"),
            os.path.expanduser("~/图片"),
        ]
        path = next((p for p in candidates if os.path.isdir(p)), "")
        if hasattr(self, "imageDirEdit"):
            self.imageDirEdit.setText(path)
        if path:
            self.load_image_folder(path)

    def on_browse_image_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择图像文件夹", self.imageDirEdit.text() if hasattr(self, "imageDirEdit") else ""
        )
        if path:
            if hasattr(self, "imageDirEdit"):
                self.imageDirEdit.setText(path)
            self.load_image_folder(path)

    def on_image_dir_entered(self):
        path = self.imageDirEdit.text().strip()
        self.load_image_folder(path)

    def _on_image_dir_changed(self, path):
        # 目录内容变化时刷新
        self.load_image_folder(path)

    def load_image_folder(self, path: str):
        if not hasattr(self, "imageList"):
            return
        self.imageList.clear()

        if not path or not os.path.isdir(path):
            return

        # 更新 watcher
        try:
            # 清理旧监听
            for d in self._imageWatcher.directories():
                self._imageWatcher.removePath(d)
            self._imageWatcher.addPath(path)
        except Exception:
            pass

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
        try:
            files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.splitext(f.lower())[1] in exts
            ]
        except Exception:
            files = []

        icon_sz = self.imageList.iconSize()
        for f in sorted(files):
            pix = QtGui.QPixmap(f)
            if pix.isNull():
                continue
            thumb = pix.scaled(icon_sz, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(thumb), os.path.basename(f))
            item.setToolTip(f)
            item.setData(QtCore.Qt.UserRole, f)
            self.imageList.addItem(item)

    def on_image_double_clicked(self, item: QtWidgets.QListWidgetItem):
        path = item.data(QtCore.Qt.UserRole)
        if not path or not os.path.isfile(path):
            return
        pix = QtGui.QPixmap(path)
        if pix.isNull():
            return

        # 简洁预览对话框
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(os.path.basename(path))
        dlg.setModal(True)
        v = QtWidgets.QVBoxLayout(dlg)
        lbl = QtWidgets.QLabel(dlg)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setScaledContents(True)
        lbl.setPixmap(pix)
        v.addWidget(lbl)
        # 合理的初始大小，支持用户调整
        dlg.resize(min(max(pix.width(), 640), 1200), min(max(pix.height(), 480), 900))
        dlg.exec_()

    # --- 新增：报告浏览器初始化与行为 ---
    def _setup_report_browser(self):
        if not hasattr(self, "reportList"):
            return

        self.reportList.setViewMode(QtWidgets.QListView.IconMode)
        self.reportList.setIconSize(QtCore.QSize(96, 96))
        self.reportList.setResizeMode(QtWidgets.QListView.Adjust)
        self.reportList.setMovement(QtWidgets.QListView.Static)
        self.reportList.setWrapping(True)
        self.reportList.setSpacing(8)
        self.reportList.setDragEnabled(True)
        self.reportList.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        if hasattr(self, "browseReportDirBtn"):
            self.browseReportDirBtn.clicked.connect(self.on_browse_report_dir)
        if hasattr(self, "reportDirEdit"):
            self.reportDirEdit.returnPressed.connect(self.on_report_dir_entered)
        self.reportList.itemDoubleClicked.connect(self.on_report_double_clicked)

        self._reportWatcher = QtCore.QFileSystemWatcher(self)
        self._reportWatcher.directoryChanged.connect(self._on_report_dir_changed)

        self._init_default_report_dir()

    def _init_default_report_dir(self):
        candidates = [
            os.path.join(APP_ROOT, "reports"),
            # os.path.join(APP_ROOT, "output", "reports"),
            os.path.expanduser("~/Documents"),
        ]
        path = next((p for p in candidates if os.path.isdir(p)), "")
        if hasattr(self, "reportDirEdit"):
            self.reportDirEdit.setText(path)
        if path:
            self.load_report_folder(path)

    def on_browse_report_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择报告文件夹", self.reportDirEdit.text() if hasattr(self, "reportDirEdit") else ""
        )
        if path:
            if hasattr(self, "reportDirEdit"):
                self.reportDirEdit.setText(path)
            self.load_report_folder(path)

    def on_report_dir_entered(self):
        path = self.reportDirEdit.text().strip()
        self.load_report_folder(path)

    def _on_report_dir_changed(self, path):
        self.load_report_folder(path)

    def load_report_folder(self, path: str):
        if not hasattr(self, "reportList"):
            return
        self.reportList.clear()

        if not path or not os.path.isdir(path):
            return

        try:
            for d in self._reportWatcher.directories():
                self._reportWatcher.removePath(d)
            self._reportWatcher.addPath(path)
        except Exception:
            pass

        # 支持常见报告类型
        report_exts = {
            ".pdf", ".html", ".htm", ".txt", ".md", ".log", ".json", ".csv"
        }
        try:
            files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.splitext(f.lower())[1] in report_exts
            ]
        except Exception:
            files = []

        icon_sz = self.reportList.iconSize()
        file_icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
        for f in sorted(files):
            # 简单生成缩略图：文本用标准图标，HTML/PDF不解析缩略图
            icon = file_icon
            pix = QtGui.QPixmap(icon_sz)
            pix.fill(QtCore.Qt.transparent)
            icon = icon

            item = QtWidgets.QListWidgetItem(icon, os.path.basename(f))
            item.setToolTip(f)
            item.setData(QtCore.Qt.UserRole, f)
            self.reportList.addItem(item)

    def on_report_double_clicked(self, item: QtWidgets.QListWidgetItem):
        path = item.data(QtCore.Qt.UserRole)
        if not path or not os.path.isfile(path):
            return
        self._open_report(path)

    def _open_report(self, path: str):
        ext = os.path.splitext(path.lower())[1]
        text_exts = {".txt", ".md", ".log", ".json", ".csv"}
        if ext in text_exts:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                content = f"无法读取文件：{e}"

            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle(os.path.basename(path))
            dlg.resize(800, 600)
            layout = QtWidgets.QVBoxLayout(dlg)
            text = QtWidgets.QTextEdit(dlg)
            text.setReadOnly(True)
            text.setPlainText(content)
            layout.addWidget(text)
            dlg.exec_()
        else:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    # --- 新增：后台任务执行与状态回调 ---
    def _set_busy(self, busy: bool, msg: str | None = None):
        if msg:
            # 让状态文字立刻可见
            self.statusbar.showMessage(msg, 0)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if busy:
            self._set_controls_enabled(False)
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._set_controls_enabled(self.agent_system.backend_ready)

    def _run_in_background(self, desc: str, func):
        # 显示即时状态并切换忙碌指针
        self._set_busy(True, f"{desc}中...")
        # 在后台线程里执行 func，并在结束时发射完成信号
        def worker():
            ok, err = True, ""
            try:
                func()
            except Exception:
                # 新增：捕获完整堆栈，便于定位问题
                ok, err = False, traceback.format_exc()
            finally:
                # 通知主线程结束
                self.taskFinished.emit(desc, ok, err)
        threading.Thread(target=worker, daemon=True).start()

    @QtCore.pyqtSlot(str, bool, str)
    def _on_task_finished(self, desc: str, ok: bool, err: str):
        self._set_busy(False)
        if ok:
            self.statusbar.showMessage(f"{desc}完成", 3000)
        else:
            self.statusbar.showMessage(f"{desc}失败: {err.splitlines()[-1] if err else ''}", 5000)
            # 新增：将完整异常堆栈追加到日志窗口
            if hasattr(self, "logText") and self.logText is not None:
                try:
                    cur = self.logText.toPlainText()
                    sep = "\n" if cur and not cur.endswith("\n") else ""
                    self.logText.setPlainText(cur + f"{sep}[{desc}] 异常:\n{err}\n")
                    self.logText.verticalScrollBar().setValue(self.logText.verticalScrollBar().maximum())
                except Exception:
                    pass
        # 状态栏状态同步
        self.update_statusbar()

    # ----------------- Close -----------------
    def closeEvent(self, event: QtGui.QCloseEvent):
        self.agent_system.stop()
        super().closeEvent(event)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.showMaximized()  # 最大化显示
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
