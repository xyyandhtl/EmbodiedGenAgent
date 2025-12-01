#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
import message_filters

from cv_bridge import CvBridge
import cv2
import numpy as np

# ROS2 的 point cloud helper
from sensor_msgs_py import point_cloud2

class LidarImageOverlayNode(Node):
    def __init__(self):
        super().__init__('lidar_image_overlay')

        # ---------- 参数：把以下相机内参 / 畸变 / 外参 改成你的真实值 ----------
        # 相机内参矩阵 (fx, fy, cx, cy)
        self.camera_intrinsics = {
            "fx": 864.0,
            "fy": 864.0,
            "cx": 639.2,
            "cy": 373.3,
            # "fx": 320,
            # "fy": 320,
            # "cx": 320,
            # "cy": 240
        }

        # 畸变系数 [k1, k2, p1, p2, k3]
        # self.dist_coeffs = np.array([-0.354630, 0.102054, -0.001614, -0.001249, 0.000000], dtype=np.float64)  # <- 替换
        self.dist_coeffs = np.zeros((5,1))  # <- 如果没有畸变，使用这个

        # lidar -> cam 外参 4x4 矩阵 (你在问题中给出的)
        self.lidar2cam = np.array([
            [0.0, -1.0,  0.0,  0.0],
            [0.0,  0.0, -1.0, -0.0],
            [1.0,  0.0,  0.0, -0.2],
            [0.0,  0.0,  0.0,  1.0]
          # [0, -1, 0, 0],
          # [0, 0, -1, -0.2],
          # [1, 0, 0, -0.4],  
          # [0, 0, 0, 1]
        ], dtype=np.float64)

        # -----------------------------------------------------------------

        # cv_bridge
        self.bridge = CvBridge()

        # message_filters subscribers (使用 message_filters 的 Subscriber)
        image_sub = message_filters.Subscriber(self, Image, '/udp_cam/image')
        pc_sub = message_filters.Subscriber(self, PointCloud2, '/livox/lidar')
        # image_sub = message_filters.Subscriber(self, Image, '/camera/rgb/image_raw')
        # pc_sub = message_filters.Subscriber(self, PointCloud2, '/lidar/pointcloud')

        # ApproximateTimeSynchronizer 参数：队列长度、时间容差（秒）
        queue_size = 10
        slop = 0.05  # 50 ms 容差，可按需要调整
        ats = message_filters.ApproximateTimeSynchronizer([image_sub, pc_sub], queue_size, slop)
        ats.registerCallback(self.synced_callback)

        # 可视化窗口名字
        self.win_name = "Lidar->Image Overlay"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        self.get_logger().info('lidar_image_overlay node started.')

    def synced_callback(self, image_msg: Image, pc2_msg: PointCloud2):
        # Convert ROS Image -> OpenCV BGR
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge failure: {e}")
            return

        # Read points from PointCloud2
        pts = []
        for p in point_cloud2.read_points(pc2_msg, field_names=("x","y","z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        if len(pts) == 0:
            return
        pts = np.asarray(pts, dtype=np.float64)

        # Downsample for visualization
        pts = pts[::5]

        # Transform lidar->camera
        ones = np.ones((pts.shape[0], 1))
        pts_h = np.hstack([pts, ones])
        pts_cam = (self.lidar2cam @ pts_h.T).T[:, :3]

        # filter z>0
        mask = pts_cam[:, 2] > 0.05
        pts_cam = pts_cam[mask]
        if pts_cam.shape[0] == 0:
            return

        # camera intrinsics
        fx = self.camera_intrinsics["fx"]
        fy = self.camera_intrinsics["fy"]
        cx = self.camera_intrinsics["cx"]
        cy = self.camera_intrinsics["cy"]

        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float64)

        dist_coeffs = self.dist_coeffs

        rvec = np.zeros((3,1))
        tvec = np.zeros((3,1))

        object_points = pts_cam.astype(np.float32)
        imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts = imgpts.reshape(-1, 2)

        h, w = cv_image.shape[:2]
        vis = cv_image.copy()

        # ----------------------------------------------------------------------------
        # 新增：颜色模式
        # ----------------------------------------------------------------------------
        if not hasattr(self, "color_mode"):
            self.color_mode = 1   # 默认按深度着色

        depths_z = pts_cam[:, 2]
        dist_origin = np.linalg.norm(pts_cam, axis=1)
        xs = pts_cam[:, 0]
        ys = pts_cam[:, 1]

        # 归一化辅助函数
        def normalize(v):
            vmin, vmax = np.percentile(v, 2), np.percentile(v, 98)
            return np.clip((v - vmin) / (vmax - vmin + 1e-6), 0, 1)

        if self.color_mode == 1:
            color_param = normalize(depths_z)
            mode_text = "Color by Z-depth"
        elif self.color_mode == 2:
            color_param = normalize(dist_origin)
            mode_text = "Color by Distance to Origin"
        elif self.color_mode == 3:
            color_param = normalize(xs)
            mode_text = "Color by X"
        elif self.color_mode == 4:
            color_param = normalize(ys)
            mode_text = "Color by Y"
        elif self.color_mode == 5:
            color_param = np.zeros_like(depths_z)
            mode_text = "Color = Fixed (White)"
        else:
            color_param = normalize(depths_z)
            mode_text = "Color by Z-depth"

        # ----------------------------------------------------------------------------
        # 绘制点
        # ----------------------------------------------------------------------------
        uv_list = []

        for (u, v), cval in zip(imgpts, color_param):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < w and 0 <= vi < h:
                uv_list.append([ui, vi])
                if self.color_mode == 5:
                    color = (255,255,255)
                else:
                    color = tuple(int(c) for c in cv2.applyColorMap(
                        np.array([[int(cval*255)]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0])
                cv2.circle(vis, (ui, vi), 2, color, -1)

        uv_list = np.array(uv_list)
        if uv_list.shape[0] > 0:
            umin, vmin = uv_list.min(axis=0)
            umax, vmax = uv_list.max(axis=0)
            cv2.rectangle(vis, (umin, vmin), (umax, vmax), (0,255,0), 2)

        # text display
        cv2.putText(vis, f'pts:{len(uv_list)}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(vis, mode_text, (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # ----------------------------------------------------------------------------
        # 增强：键盘切换颜色模式
        # ----------------------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            self.color_mode = 1
        elif key == ord('2'):
            self.color_mode = 2
        elif key == ord('3'):
            self.color_mode = 3
        elif key == ord('4'):
            self.color_mode = 4
        elif key == ord('5'):
            self.color_mode = 5
        elif key == ord('c'):
            vis = cv_image.copy()
        elif key == ord('q'):
            rclpy.shutdown()

        cv2.imshow(self.win_name, vis)


def main(args=None):
    rclpy.init(args=args)
    node = LidarImageOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
