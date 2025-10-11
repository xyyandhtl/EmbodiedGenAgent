import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, uic

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

    def __init__(self):
        super().__init__()

        uic.loadUi(UI_PATH, self)

        self.agent = EGAgentSystem()

        # 连接信号到槽 (主线程更新 UI)
        self.logSignal.connect(self._on_log_update)
        self.convSignal.connect(self._on_conv_update)
        self.entitiesSignal.connect(self._on_entities_update)
        self.statusSignal.connect(self._on_status_update)

        # 按钮绑定
        self.startBtn.clicked.connect(self.on_start)
        self.stopBtn.clicked.connect(self.on_stop)
        self.sendInstructionBtn.clicked.connect(self.on_send_instruction)
        self.instructionEdit.returnPressed.connect(self.on_send_instruction)
        # 新增：保存/载入地图
        self.saveMapBtn.clicked.connect(self.on_save_map)
        self.loadMapBtn.clicked.connect(self.on_load_map)

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

        # 替换原直接更新 UI 的监听器 -> 仅发射信号
        self.agent.add_listener("log", lambda data: self.logSignal.emit(data))
        self.agent.add_listener("conversation", lambda data: self.convSignal.emit(data))
        self.agent.add_listener("entities", lambda data: self.entitiesSignal.emit(list(data)))
        self.agent.add_listener("status", lambda data: self.statusSignal.emit(bool(data)))

        self.entityTable.setColumnCount(2)
        self.entityTable.setHorizontalHeaderLabels(["实体", "信息"])
        self.entityTable.horizontalHeader().setStretchLastSection(True)
        self.entityTable.verticalHeader().setVisible(False)

        self.update_statusbar()

        # --- 新增：初始化图像与报告浏览器 ---
        self._setup_image_browser()
        self._setup_report_browser()

    # ----------------- UI Events -----------------
    def on_start(self):
        self.agent.start()  # 启动 EGAgentSystem 的主线程 _run_loop
        self.update_statusbar()

    def on_stop(self):
        self.agent.stop()
        self.update_statusbar()

    def on_send_instruction(self):
        text = self.instructionEdit.text().strip()
        if not text:
            return
        self.agent.feed_instruction(text)  # 将用户的自然语言指令 发送给 EGAgentSystem，并执行后续操作
        self.instructionEdit.clear()

    def on_save_map(self):
        # 默认目录：cfg.map_save_path
        default_dir = getattr(self.agent.vlmap_backend.cfg, "map_save_path", "") or ""
        if not default_dir:
            default_dir = os.path.expanduser("~")
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择地图保存文件夹", default_dir)
        if not path:
            return
        self.agent.save(map_path=path)
        self.statusbar.showMessage(f"地图已保存至: {path}", 3000)

    def on_load_map(self):
        # 默认目录：cfg.preload_path
        default_dir = getattr(self.agent.vlmap_backend.cfg, "preload_path", "") or ""
        if not default_dir:
            default_dir = os.path.expanduser("~")
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择地图载入文件夹", default_dir)
        if not path:
            return
        self.agent.load(map_path=path)
        self.statusbar.showMessage(f"已载入地图: {path}", 3000)

    # ----------------- Periodic Updates -----------------
    def update_fast(self):
        """高频: 当前视野实例分割"""
        img = self.agent.get_current_instance_seg_image()
        self.instanceSegLabel.setPixmap(np_to_qpix(img))

    def update_medium(self):
        """中频: 可通行地图 + 实体表"""
        self.traversableMapLabel.setPixmap(
            np_to_qpix(self.agent.get_traversable_map_image()))
        self.refresh_entities()

    def update_slow(self):
        """低频: 3D实例 + 语义/路径地图"""
        # 3D实例
        self.instance3DLabel.setPixmap(
            np_to_qpix(self.agent.get_current_instance_3d_image()))
        
        # 语义地图（长宽比修正）
        img = self.agent.get_semantic_map_image()
        pixmap = np_to_qpix(img)
        # 按比例缩放 QPixmap 以适应 QLabel 的尺寸
        scaled_pixmap = pixmap.scaled(
            self.semanticMapLabel.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.semanticMapLabel.setPixmap(scaled_pixmap)

    def update_bt(self):
        # 行为树更新
        self.behaviorTreeLabel.setPixmap(
            np_to_qpix(self.agent.get_entity_bt_image()))

    # ----------------- Slots for signals (主线程执行) -----------------
    def _on_log_update(self, txt: str):
        self.logText.setPlainText(txt)
        self.logText.verticalScrollBar().setValue(self.logText.verticalScrollBar().maximum())

    def _on_conv_update(self, txt: str):
        self.conversationText.setPlainText(txt)
        self.conversationText.verticalScrollBar().setValue(self.conversationText.verticalScrollBar().maximum())

    def _on_entities_update(self, rows):
        self.entityTable.setRowCount(len(rows))
        for r, (name, info) in enumerate(rows):
            self.entityTable.setItem(r, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.entityTable.setItem(r, 1, QtWidgets.QTableWidgetItem(str(info)))

    def _on_status_update(self, running: bool):
        self.update_statusbar()

    # ----------------- Legacy helper methods (仍被定时器调用) -----------------
    def refresh_conversation(self):
        # (保留以防手动调用) 线程安全: 仅从主线程调用
        self._on_conv_update(self.agent.get_conversation_text())

    def refresh_log(self):
        self._on_log_update(self.agent.get_log_text_tail())

    def refresh_entities(self):
        rows = list(self.agent.get_entity_rows())
        self._on_entities_update(rows)

    def update_statusbar(self):
        state = "运行中" if self.agent.status else "已停止"
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

    # ----------------- Close -----------------
    def closeEvent(self, event: QtGui.QCloseEvent):
        self.agent.stop()
        super().closeEvent(event)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.showMaximized()  # 最大化显示
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
