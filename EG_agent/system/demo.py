# run_agent_camera.py

import cv2
import time
import numpy as np
from pathlib import Path
from PIL import Image

from agent_system import RobotAgentSystem, Observation
# from vlm_robot_agent.vlm_agent.conversation import ConversationManager

def main():
    # 视频源（文件）或摄像头索引
    source = "/home/lenovo/Documents/vlm_data/1.mp4"
    cap = cv2.VideoCapture(source)
    # 如果是摄像头可以设置参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    agent = RobotAgentSystem(goal_text="Go to open the window")

    print("┌ Initial Sub-Goals Plan ────────────────────────────")
    for idx, g in enumerate(agent.goal_manager.goal_stack, start=1):
        print(f"│ {idx}. {g.description}")
    print("└───────────────────────────────────────────────────")

    win_cam = "Robot View"
    cv2.namedWindow(win_cam, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_cam, 800, 600)

    # 控制读取频率
    target_fps        = 0.5         # 期望2s请求一次
    read_min_interval = 1.0 / target_fps
    frame_stride      = 60           # 若设为2表示每读取1帧跳过1帧（进一步降采样）

    plan_log = []
    last_action_t = time.time()
    action_interval = 0.5  # 行为决策周期
    last_read_t = 0.0
    last_status_print_t = 0.0
    status_print_interval = 0.5

    last_frame = None
    finished_video = False
    frame_index = 0

    while True:
        now = time.time()
        # 到时间再读取下一帧
        if now - last_read_t >= read_min_interval and not finished_video:
            # 抽帧：跳过若干帧（若 frame_stride > 1）
            if frame_stride > 1:
                # 用 grab() 更高效跳过
                for _ in range(frame_stride - 1):
                    cap.grab()
            ret, frame = cap.read()
            if not ret:
                finished_video = True
            else:
                last_frame = frame
                frame_index += 1
            last_read_t = now
        else:
            # 没到读取时间，降低 CPU
            time.sleep(0.005)

        if last_frame is None:
            if finished_video:
                print("📼 视频结束。")
                break
            continue

        # 显示（也可以选择仅在新帧时刷新）
        disp = last_frame.copy()
        mode = agent._current_mode()
        color = (0,255,0) if mode == 'navigation' else (0,0,255)
        cv2.rectangle(disp, (0,0), (300,30), color, -1)
        cv2.putText(disp, f"MODE: {mode.upper()} F#{frame_index}",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(win_cam, disp)

        # 交互模式
        if mode == "interaction" and agent.conversation:
            while agent.conversation.interactive_turn(listen_secs=5):
                pass
            agent.state_tracker.state = agent.state_tracker.NAVIGATING

        # 行为决策
        if mode == "navigation" and (time.time() - last_action_t > action_interval):
            last_action_t = time.time()
            img = Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
            try:
                action = agent.step(img)
                txt = f"{action.kind.name}: {action.params}"
            except Exception as e:
                txt = "Error: " + str(e)
            plan_log.append(txt)
            cv2.displayOverlay(win_cam, txt, 1000)
            print(f"🤖 {txt}")
            agent.goal_manager.pop_finished()

        # 终端状态打印
        if time.time() - last_status_print_t > status_print_interval:
            last_status_print_t = time.time()
            lines = []
            lines.append("\n===== Robot Status =====")
            lines.append(f"Frame Index: {frame_index}  (stride={frame_stride}, target_fps={target_fps})")
            lines.append("Sub-Goals:")
            for g in agent.goal_manager.goal_stack:
                lines.append(f"  - {g.description}")
            lines.append("Recent Actions:")
            for line in plan_log[-6:]:
                lines.append(f"  * {line}")
            if agent.conversation and agent.conversation.history:
                lines.append("Conversation (last 6 turns):")
                for turn in agent.conversation.history[-6:]:
                    speaker = getattr(turn, 'role', getattr(turn, 'speaker', 'robot'))
                    text = getattr(turn, 'text', getattr(turn, 'message',''))
                    prefix = "R" if speaker.lower().startswith("robot") else "H"
                    lines.append(f"  {prefix}: {text}")
            lines.append("========================")
            print("\n".join(lines), flush=True)

        # 退出
        if (cv2.waitKey(1) & 0xFF == ord('q')) or agent.finished:
            print("✅ Mission completed.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
