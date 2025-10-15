import zmq
import pickle

class ZMQBridge:
    """
    Combined ZeroMQ publisher/subscriber bridge.
    - Publishes data on PUB socket (bind on pub_port).
    - Subscribes to commands on SUB socket (connect to localhost:sub_port).

    Message schema:
    - Simulation -> ROS (publish_data payload dict):
        {
            "pose": (pos_np(shape=(3,)), quat_wxyz_np(shape=(4,))),  # optional
            "rgb": np.uint8[H,W,3],                                   # optional, RGB order
            "depth": np.uint16[H,W] or np.float32[H,W],               # optional
            # ... additional fields allowed
        }
    - ROS -> Simulation (received on SUB in poll):
        {
            "cmd_vel": np.float32[6],  # [lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]
            "nav_pose": (pos_np(shape=(3,)), quat_wxyz_np(shape=(4,))),
            "enum": int,
            "mark": None or np.ndarray(shape=(3,))  # None means default mark, array means [x,y,z]
        }
    """
    def __init__(self, pub_port: int = 5555, sub_port: int = 5556):
        self.context = zmq.Context()

        if pub_port == sub_port:
            raise ValueError("PUB and SUB must use different TCP ports.")

        # Publisher
        self.pub = self.context.socket(zmq.PUB)
        self.pub.bind(f"tcp://*:{pub_port}")
        print(f"[ZMQ Bridge] PUB started on port {pub_port}.")

        # Subscriber
        self.sub = self.context.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')  # subscribe to all
        # Keep only the most recent command to reduce backlog latency
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.connect(f"tcp://localhost:{sub_port}")
        print(f"[ZMQ Bridge] SUB connected to tcp://localhost:{sub_port}.")

        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)

        # State (updated by poll)
        self.latest_cmd_vel = None  # (lin_x, ang_z)
        self.nav_pose = None        # (pos_np, quat_np_wxyz)
        self.enum_cmd = None        # int
        # 新增：mark 请求（每 tick 重置；收到则为 {"pos": None|np.ndarray})
        self.mark_pos = None

    def publish_data(self, data):
        # try:
        message = pickle.dumps(data)
        self.pub.send(message)
        # except Exception as e:
        #     print(f"[ZMQ Bridge] Error sending data: {e}")

    def poll(self):
        # reset transient state
        self.latest_cmd_vel = None
        self.enum_cmd = None
        self.nav_pose = None
        self.mark_pos = None
        events = dict(self.poller.poll(0))
        if self.sub in events:
            try:
                # Drain all available messages; keep only the latest values this tick
                while True:
                    msg = self.sub.recv(flags=zmq.NOBLOCK)
                    cmd_data = pickle.loads(msg)
                    if "cmd_vel" in cmd_data:
                        arr = cmd_data["cmd_vel"]
                        self.latest_cmd_vel = (float(arr[0]), float(arr[5]))  # lin.x, ang.z
                    if "nav_pose" in cmd_data:
                        pos_np, quat_np_wxyz = cmd_data["nav_pose"]
                        self.nav_pose = (pos_np, quat_np_wxyz)
                    if "enum" in cmd_data:
                        self.enum_cmd = int(cmd_data["enum"])
                    # 新增：解析 mark
                    if "mark" in cmd_data:
                        # print(f"[ZMQ Bridge] Received mark: {cmd_data['mark']}")
                        val = cmd_data["mark"]
                        self.mark_pos = (float(val[0]), float(val[1]), float(val[2]))
                        # print(f"[ZMQ Bridge] Parsed mark position: {self.mark_pos}")
            except zmq.Again:
                pass
            except Exception:
                # EAGAIN or deserialization errors are ignored per tick
                pass

    def close(self):
        # try:
            self.pub.close()
        # except Exception:
        #     pass
        # try:
            self.sub.close()
        # except Exception:
        #     pass
        # try:
            self.context.term()
        # except Exception:
        #     pass

