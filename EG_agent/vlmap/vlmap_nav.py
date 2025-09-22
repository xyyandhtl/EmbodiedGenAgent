import time
import threading
import cv2
import numpy as np
import hydra
from collections import deque
from threading import Event
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R

from EG_agent.vlmap.dualmap.core import Dualmap
from EG_agent.vlmap.utils.types import DataInput
from EG_agent.vlmap.utils.time_utils import timing_context


class VLMapNav:
    def __init__(self, cfg: DictConfig = None):
        # If no cfg provided, load it here using Hydra (config_path matches previous main)
        if cfg is None:
            # from hydra.core.global_hydra import GlobalHydra
            # if GlobalHydra.instance().is_initialized():
            #     GlobalHydra.instance().clear()
            hydra.initialize(version_base=None, config_path="./config/")
            self.cfg = hydra.compose(config_name="runner_isaaclab")
        else:
            self.cfg = cfg

        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

        self.kf_idx = 0
        self.synced_data_queue = deque(maxlen=1)
        self.processing_thread = None

        self.sensor = None  # Deployment environment sensor handler

        self.stop_count = 0
        self.prev_count = -1

        self.orig_height = self.cfg.camera_params['image_height']
        self.orig_width = self.cfg.camera_params['image_width']
        self.fx = self.cfg.camera_params['fx']
        self.fy = self.cfg.camera_params['fy']
        self.cx = self.cfg.camera_params['cx']
        self.cy = self.cfg.camera_params['cy']

        self.intrinsics = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.dualmap = Dualmap(self.cfg)

    def on_new_frame(self):
        self.event.set()  # Notify main thread of new frame arrival

    def on_stream_stopped(self):
        print("Stream stopped")

    def connect_to_simulation(self, handler):
        """Connects to the Isaac Lab sensor handler."""
        self.sensor = handler
        self.event = handler.new_frame_event
        print("Successfully connected to Isaac Lab sensor handler.")

    def get_intrinsic_matrix(self, coeffs):
        return np.array([[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]])

    def start_processing_stream(self):
        while True:
            self.event.wait()  # Wait for a new frame signal from the simulation
            
            print("New frame received, processing...")
            # 1. Get data from the sensor handler
            depth = self.sensor.get_depth_frame()
            rgb = self.sensor.get_rgb_frame()
            intrinsics_matrix = self.sensor.get_intrinsics()
            pose = self.sensor.get_camera_pose()

            # Check if data is valid
            if rgb is None or depth is None or pose is None or intrinsics_matrix is None:
                self.event.clear() # Clear event and continue to next frame
                continue

            # 2. Convert torch tensors to numpy arrays
            rgb = rgb.cpu().numpy()
            depth = depth.cpu().numpy()
            intrinsics = intrinsics_matrix.cpu().numpy()
            cam_pos, cam_quat = pose[0].cpu().numpy(), pose[1].cpu().numpy()

            # 3. Process data (pose transformation, etc.)
            translation = cam_pos.flatten()
            quaternion = cam_quat.flatten() # [qw, qx, qy, qz], and follow ROS convention.
            rotation_matrix = R.from_quat(quaternion).as_matrix()

            # todo: isaaclab axis to dualmap axis
            T = np.eye(4)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = translation

            # Flip world YZ axes to ROS convention
            T[:, 1:3] *= -1

            # Rotate world frame: +90 deg around X
            T_fix = np.eye(4)
            T_fix[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()
            transformation_matrix = T_fix @ T

            depth_resized = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            timestamp = time.time()

            data_input = DataInput(
                idx=self.kf_idx,
                time_stamp=timestamp,
                color=rgb,
                depth=depth_resized,
                color_name=str(timestamp),
                intrinsics=intrinsics,
                pose=transformation_matrix
            )

            self.synced_data_queue.append(data_input)
            self.stop_count += 1
            self.event.clear()

    def start_processing_in_thread(self):
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self.start_processing_stream, daemon=True)
            self.processing_thread.start()
            print("Stream processing thread started")
        else:
            print("Stream processing already running")

    def get_synced_data_queue(self):
        return self.synced_data_queue
    
    def get_nav_path(self):
        # todo: wrap the path planning from dualmap navigation_helper
        pass

    def get_cmd_vel(self):
        # todo: impl a simple velocity command generator based on the planned path
        pass

    def query_object(self, query: str):
        # todo: query the object from bt, return the candidates infos 
        pass
    
    def run(self):
        # start the sensor processing thread
        self.start_processing_in_thread()

        end_count = 0

        while True:
            time.sleep(0.1)
            print("Main thread running...")

            if self.stop_count == self.prev_count:
                end_count += 1
            else:
                end_count = 0

            self.prev_count = self.stop_count

            use_end = getattr(self.cfg, "use_end_process", False) if self.cfg is not None else False
            # if no new frames for 50 x 0.1 = 5s, end_process, map would be saved
            if use_end and end_count > 50:
                print("No new frames detected. Terminating...")
                if self.dualmap is not None:
                    self.dualmap.end_process()
                break

            synced_queue = self.get_synced_data_queue()
            if not synced_queue:
                continue

            data_input = synced_queue[-1]
            if data_input is None:
                continue

            if self.dualmap is None:
                continue

            if not self.dualmap.check_keyframe(data_input.time_stamp, data_input.pose):
                continue

            kf_idx = self.dualmap.get_keyframe_idx()
            data_input.idx = kf_idx

            with timing_context("Time Per Frame", self.dualmap):
                use_parallel = getattr(self.cfg, "use_parallel", False) if self.cfg is not None else False
                if use_parallel:
                    self.dualmap.parallel_process(data_input)
                else:
                    self.dualmap.sequential_process(data_input)


if __name__ == "__main__":
    app = VLMapNav()
    # app.connect_to_simulation()
    app.run()
