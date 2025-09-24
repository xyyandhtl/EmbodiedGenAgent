import time
import os
import numpy as np
import zmq
import pickle

class ZMQDataPublisher:
    """
    A simple ZeroMQ publisher to send Python objects from the simulation.
    """
    def __init__(self, port=5555):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        print(f"[ZMQ Publisher] Started on port {port}. Waiting for subscribers...")

    def publish_data(self, data):
        """
        Serializes and sends the data dictionary.
        """
        try:
            # Serialize the data using pickle
            message = pickle.dumps(data)
            self.socket.send(message)
        except Exception as e:
            print(f"[ZMQ Publisher] Error sending data: {e}")

    def close(self):
        self.socket.close()


# --- NEW: CommandHandler 封装 ZMQ 接收与命令处理 ---
class ZMQDataSubscriber:
    def __init__(self, address="tcp://localhost:5556"):
        self.context = zmq.Context()
        self.sub = self.context.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')
        self.sub.connect(address)
        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)
        # state
        self.latest_cmd_vel = None
        self.nav_pose = None
        self.enum_cmd = None

    def poll(self):
        # reset transient state
        self.latest_cmd_vel = None
        self.enum_cmd = None
        self.nav_pose = None
        events = dict(self.poller.poll(0))
        if self.sub in events:
            try:
                msg = self.sub.recv(flags=zmq.NOBLOCK)
                cmd_data = pickle.loads(msg)
                if "cmd_vel" in cmd_data:
                    arr = cmd_data["cmd_vel"]
                    self.latest_cmd_vel = (float(arr[0]), float(arr[5]))
                if "nav_pose" in cmd_data:
                    pos_np, quat_np = cmd_data["nav_pose"]
                    self.nav_pose = (pos_np, quat_np)
                if "enum" in cmd_data:
                    self.enum_cmd = int(cmd_data["enum"])
            except Exception:
                # ignore malformed/non-blocking errors
                pass

    def handle_enum_action(self, enum_cmd, rgb_tensor, depth_tensor, pose_tuple, sim_data_dir, marks, report_file):
        ts = int(time.time() * 1000)
        if enum_cmd == 0:
            # take photo
            if rgb_tensor is not None:
                rgb_arr = rgb_tensor[0, :, :, :3].cpu().numpy().astype(np.uint8)
            else:
                rgb_arr = None
            if depth_tensor is not None:
                depth_arr = (depth_tensor[0] * 1000).cpu().numpy().astype(np.uint16)
            else:
                depth_arr = None
            save_path = os.path.join(sim_data_dir, f"photo_{ts}.npz")
            np.savez_compressed(save_path, rgb=rgb_arr, depth=depth_arr)
            print(f"[ACTION] Saved photo to {save_path}")
        elif enum_cmd == 1:
            # mark pose
            if pose_tuple is not None:
                pos_np, quat_np = pose_tuple[0][0].cpu().numpy(), pose_tuple[1][0].cpu().numpy()
                marks.append((pos_np.tolist(), quat_np.tolist(), ts))
                with open(os.path.join(sim_data_dir, "marks.txt"), "a") as f:
                    f.write(f"{ts}\t{pos_np.tolist()}\t{quat_np.tolist()}\n")
                print(f"[ACTION] Marked pose at time {ts}")
            else:
                print("[ACTION] Mark requested but pose unavailable")
        elif enum_cmd == 2:
            # report
            report = {
                "timestamp": ts,
                "marks_count": len(marks)
            }
            with open(report_file, "a") as f:
                f.write(str(report) + "\n")
            print(f"[ACTION] Reported status: {report}")

    def close(self):
        try:
            self.sub.close()
        except Exception:
            pass
        try:
            self.context.term()
        except Exception:
            pass

