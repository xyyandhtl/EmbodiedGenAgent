import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, uic

# 正式运行
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from EG_agent.system.system import EGAgentSystem
# 仅测试界面
# from system_example import EGAgentSystem

UI_PATH = os.path.join(os.path.dirname(__file__), "window.ui")

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

    # ----------------- UI Events -----------------
    def on_start(self):
        self.agent.start()
        self.update_statusbar()

    def on_stop(self):
        self.agent.stop()
        self.update_statusbar()

    def on_send_instruction(self):
        text = self.instructionEdit.text().strip()
        if not text:
            return
        self.agent.feed_instruction(text)
        self.instructionEdit.clear()

    # ----------------- Periodic Updates -----------------
    def update_fast(self):
        # 高频: 当前视野实例分割
        img = self.agent.get_current_instance_seg_image()
        self.instanceSegLabel.setPixmap(np_to_qpix(img))

    def update_medium(self):
        # 中频: 可通行地图 + 实体表
        self.traversableMapLabel.setPixmap(
            np_to_qpix(self.agent.get_traversable_map_image()))
        self.refresh_entities()

    def update_slow(self):
        # 低频: 3D实例 + 语义/路径地图
        self.instance3DLabel.setPixmap(
            np_to_qpix(self.agent.get_current_instance_3d_image()))
        self.semanticMapLabel.setPixmap(
            np_to_qpix(self.agent.get_semantic_map_image()))

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
