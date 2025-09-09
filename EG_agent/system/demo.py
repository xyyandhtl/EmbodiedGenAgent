# run_agent_camera.py

import cv2
import time
import numpy as np
from threading import Thread, Lock
from pathlib import Path
from PIL import Image

from agent_system import RobotAgentSystem, Observation
# from vlm_robot_agent.vlm_agent.conversation import ConversationManager

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS,          30)
        self.lock = Lock()
        self.frame = None
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, f = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = f

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()


def main():
    cam   = CameraStream("/home/lenovo/Documents/vlm_data/1.mp4")
    agent = RobotAgentSystem(goal_text="Go to open the window")

    # Show initial plan
    print("┌ Initial Sub-Goals Plan ────────────────────────────")
    for idx, g in enumerate(agent.goal_manager.goal_stack, start=1):
        print(f"│ {idx}. {g.description}")
    print("└───────────────────────────────────────────────────")

    win_cam  = "Robot View"
    win_info = "Robot Information"
    cv2.namedWindow(win_cam,  cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_cam, 800, 600)
    cv2.namedWindow(win_info, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_info, 800, 600)

    plan_log  = []
    last_time = time.time()
    interval  = 0.5  # ≈2 FPS

    while True:
        frame = cam.read()
        if frame is None:
            continue

        # --- Draw camera and mode header ---
        disp = frame.copy()
        mode = agent._current_mode()
        color = (0,255,0) if mode=='navigation' else (0,0,255)
        cv2.rectangle(disp, (0,0),(300,30), color, -1)
        cv2.putText(disp, f"MODE: {mode.upper()}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
        cv2.imshow(win_cam, disp)

        # --- If in interaction mode, run conversation loop ---
        if mode == "interaction" and agent.conversation:
            # Execute turns until the conversation ends
            while agent.conversation.interactive_turn(listen_secs=5):
                # Show each new turn in the console
                pass
            # After finishing, return to navigation
            agent.state_tracker.state = agent.state_tracker.NAVIGATING

        # --- Navigation/brief interaction tick to decide action ---
        if mode == "navigation":
            if time.time() - last_time > interval:
                last_time = time.time()
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                try:
                    action = agent.step(img)
                    txt    = f"{action.kind.name}: {action.params}"
                except Exception as e:
                    txt = "Error: " + str(e)

                # Overlay the action
                plan_log.append(txt)
                cv2.displayOverlay(win_cam, txt, 1000)
                print("🤖", txt)

                # Pop completed sub-goals
                agent.goal_manager.pop_finished()

        # --- Side info panel ---
        info = np.zeros((600,400,3), dtype=np.uint8)
        y = 20

        # Sub-Goals
        cv2.putText(info, "Sub-Goals:", (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
        y += 30
        for g in agent.goal_manager.goal_stack:
            cv2.putText(info, f"- {g.description}", (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180),1)
            y += 20
        y += 10

        # Recent Actions
        cv2.putText(info, "Recent Actions:", (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
        y += 30
        for line in plan_log[-6:]:
            cv2.putText(info, line, (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180),1)
            y += 20
        y += 10

        # Conversation (full history)
        if agent.conversation and agent.conversation.history:
            cv2.putText(info, "Conversation:", (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
            y += 30
            for turn in agent.conversation.history[-6:]:
                speaker = getattr(turn, 'role', getattr(turn, 'speaker', 'robot'))
                text    = getattr(turn, 'text', getattr(turn, 'message',''))
                prefix  = "R" if speaker.lower().startswith("robot") else "H"
                line    = f"{prefix}: {text}"
                cv2.putText(info, line, (10,y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180),1)
                y += 20

        cv2.imshow(win_info, info)

        # Exit if 'q' is pressed or mission is completed
        if (cv2.waitKey(1) & 0xFF == ord('q')) or agent.finished:
            print("✅ Mission completed.")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
