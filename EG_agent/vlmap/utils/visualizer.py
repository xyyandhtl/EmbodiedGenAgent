import logging
import dataclasses
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv

from supervision.draw.color import Color, ColorPalette

from EG_agent.vlmap.utils.types import ObjectClasses

# Set up the module-level logger
logger = logging.getLogger(__name__)

class ReRunVisualizer:
    _instance = None

    def __init__(self, cfg = None) -> None:

        if not self._initialized:
            super().__init__()
            self.cfg = cfg

            classes_path = cfg.yolo.classes_path
            if cfg.yolo.use_given_classes:
                classes_path = cfg.yolo.given_classes_path
                logger.info(f"[Visualizar] Using given classes, path:{classes_path}")

            # Object classes
            self.obj_classes = ObjectClasses(
                classes_file_path=classes_path,
                bg_classes=self.cfg.yolo.bg_classes,
                skip_bg=self.cfg.yolo.skip_bg)

            # Camera parameters
            self.intrinsic = None
            self.pose = None
            self.image = None
            self._intrinsic_initialized = False

            self._initialized = True

            self.overlapped_image = None

    def __new__(cls, cfg = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize the instance "once"
            cls._instance._use_rerun = None
            cls._instance._rerun = None
            cls._instance._initialized = False
        return cls._instance

    def set_use_rerun(self, use_rerun):
        self._use_rerun = use_rerun
        if self._use_rerun and self._rerun is None:
            try:
                import rerun as rr
                self._rerun = rr
                logger.info("[Visualizar] rerun is installed. Using rerun for logger.")
            except ImportError:
                logger.warning(
                    "[Visualizar] rerun is not installed. Not using rerun for logger.")
        else:
            logger.warning(
                "[Visualizar] rerun functionality is disabled in the config. Not using rerun for logger.")

    def __getattr__(self, name):
        def method(*args, **kwargs):
            if self._use_rerun and self._rerun:
                func = getattr(self._rerun, name, None)
                if func:
                    return func(*args, **kwargs)
                else:
                    logger.warning(f"[Visualizar] '{name}' is not a valid rerun method.")
            else:
                if not self._use_rerun:
                    logger.debug(f"[Visualizar] Skipping optional rerun call to '{name}' because rerun usage is disabled.")
                elif self._rerun is None:
                    logger.debug(f"[Visualizar] Skipping optional rerun call to '{name}' because rerun is not installed.")
        return method

    def update_intrinsic(self, intrinsic, force=False):
        if self._intrinsic_initialized and not force:
            return
        self.intrinsic = intrinsic
        self._intrinsic_initialized = True
        logger.info("[Visualizar] Intrinsic updated.")

    def update_pose(self, pose):
        self.pose = pose
        # logger.info("[Visualizar] Pose updated.")

    def set_camera_info(self, intrinsic, pose):
        self.update_intrinsic(intrinsic)
        self.update_pose(pose)

    def set_image(self, image):
        self.image = image

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a rotation matrix to a quaternion.
        
        Parameters:
        - R: A 3x3 rotation matrix.
        
        Returns:
        - A quaternion in the format [x, y, z, w].
        """
        # Make sure the matrix is a numpy array
        R = np.asarray(R)
        # Allocate space for the quaternion
        q = np.empty((4,), dtype=np.float32)
        # Compute the quaternion components
        q[3] = np.sqrt(np.maximum(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        q[0] = np.sqrt(np.maximum(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        q[1] = np.sqrt(np.maximum(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
        q[2] = np.sqrt(np.maximum(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
        q[0] *= np.sign(q[0] * (R[2, 1] - R[1, 2]))
        q[1] *= np.sign(q[1] * (R[0, 2] - R[2, 0]))
        q[2] *= np.sign(q[2] * (R[1, 0] - R[0, 1]))
        return q

    def quaternion_to_rotation_matrix(self, q):
        """
        Convert a quaternion into a rotation matrix.
        
        Parameters:
        - q: A quaternion in the format [x, y, z, w].
        
        Returns:
        - A 3x3 rotation matrix.
        """
        w, x, y, z = q[3], q[0], q[1], q[2]
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])

    def rotation_matrix_to_axis_angle(self, R):
        """
        Convert a rotation matrix to an axis-angle representation.
        
        Parameters:
        - R: A 3x3 rotation matrix.
        
        Returns:
        - A tuple containing the axis of rotation and the angle of rotation in radians.
        """
        # Compute the trace of the matrix
        trace = np.trace(R)
        # Compute the angle of rotation
        theta = np.arccos((trace - 1) / 2)
        # Compute the axis of rotation
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / np.linalg.norm(axis)
        return axis, theta

    def visualize_3d_bbox_overlapping(self, obj_names, obj_colors, obj_bboxes):

        proc_image = self.image.copy()
        K = self.intrinsic
        T_wc = self.pose
        T_cw = np.linalg.inv(T_wc)

        for name, color, bbox in zip(obj_names, obj_colors, obj_bboxes):
            proc_image = self._draw_projected_bbox(proc_image, bbox, name, color, K, T_cw)

        self.log(
            "world/camera/overlapped_image",
            self.Image(proc_image)
        )

    def _draw_projected_bbox(self, image, bbox, name, color, K, T_cw):
        # Get corners and reorder them explicitly
        corners = np.asarray(bbox.get_box_points())

        # Sort to consistent order: bottom face first, then top face
        center = np.mean(corners, axis=0)
        above = corners[:, 2] > center[2]
        below = ~above

        top = corners[above]
        bottom = corners[below]

        def sort_corners_xy(corners):
            center = np.mean(corners[:, :2], axis=0)
            angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
            return corners[np.argsort(angles)]

        bottom_sorted = sort_corners_xy(bottom)
        top_sorted = sort_corners_xy(top)

        ordered = np.vstack([bottom_sorted, top_sorted])

        # Transform to camera frame
        corners_homo = np.hstack([ordered, np.ones((8, 1))])  # (8, 4)
        corners_cam = (T_cw @ corners_homo.T).T[:, :3]

        if np.any(corners_cam[:, 2] <= 0):
            return image

        # Project
        corners_2d = (K @ corners_cam.T).T
        corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
        corners_2d = corners_2d.astype(int)

        # Draw edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7)   # sides
        ]

        # Convert color to valid BGR tuple
        color = tuple(int(c * 255) for c in np.clip(color, 0.0, 1.0))

        # Draw edges
        for i, j in edges:
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[j])
            cv2.line(image, pt1, pt2, color=color, thickness=2)

        # Draw label
        label_pos = tuple(corners_2d[0])
        cv2.putText(image, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image

    def get_semantic_map_image(self, global_map_manager) -> None | np.ndarray:
        """
        Generates a top-down 2D image of the global semantic map.
        """
        map_objects = global_map_manager.global_map
        if not map_objects:
            print(f"[visualizer] [get_semantic_map_image] Don't have objects in this map.")
            return None

        # Determine map boundaries
        all_points = np.vstack([np.asarray(obj.pcd_2d.points) for obj in map_objects if not obj.pcd_2d.is_empty()])
        all_points = all_points[np.isfinite(all_points).all(axis=1)]  # Filter out invalid values
        if all_points.size == 0:
            print(f"[visualizer] [get_semantic_map_image] All points of the objects is None.")
            return None
        print(f"[visualizer] [get_semantic_map_image] all_points shape: {all_points.shape}")

        min_coords = np.min(all_points[:, :2], axis=0)
        max_coords = np.max(all_points[:, :2], axis=0)

        # Create a blank image
        resolution = 0.05  # meters per pixel
        padding = 50 # pixels
        width = int((max_coords[0] - min_coords[0]) / resolution) + padding
        height = int((max_coords[1] - min_coords[1]) / resolution) + padding
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Get colors for each class
        colors = self.obj_classes.get_class_color_dict_by_index()

        # Draw each object
        for obj in map_objects:
            if obj.pcd_2d.is_empty():
                continue

            points = np.asarray(obj.pcd_2d.points)
            points = points[np.isfinite(points).all(axis=1)]
            if points.shape[0] == 0:
                continue

            class_id = obj.class_id
            class_name = self.obj_classes.get_classes_arr()[class_id]
            # print(f"[visualizer] [get_semantic_map_image] Object {obj.uid}, class_id = {class_id}, class_name = {class_name}, has 3D points, shape = {points.shape}")
            # Get float RGB color (0-1 range)
            color_rgb = colors.get(str(class_id), (1.0, 1.0, 1.0))
            # Convert to integer BGR color (0-255 range) for OpenCV
            color_bgr_int = tuple(int(c * 255) for c in (color_rgb[2], color_rgb[1], color_rgb[0]))
            # print(f"[visualizer] [get_semantic_map_image] its color_bgr = {color_bgr_int}")

            # Convert world coordinates to image coordinates
            img_points = ((points[:, :2] - min_coords) / resolution).astype(int)
            img_points += padding // 2
            img_points[:, 1] = height - img_points[:, 1] - 1  # Flip Y axis

            # Draw points as 1x1 squares
            for p in img_points:
                x, y = p
                if 0 <= x < width-1 and 0 <= y < height-1:
                    image[y:y+2, x:x+2] = color_bgr_int

            # Draw label
            if img_points.shape[0] > 0:
                label_pos = np.mean(img_points, axis=0).astype(int)
                if 0 <= label_pos[0] < width and 0 <= label_pos[1] < height:
                    cv2.putText(image, class_name, tuple(label_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

    def get_traversable_map_image(self, local_map_manager) -> None | np.ndarray:
        """
        Generates a top-down 2D image of the local traversable map.
        """
        grid = local_map_manager.get_traversability_grid()  # Assume this method exists
        if grid is None:
            return None

        # Create a color representation of the grid
        h, w = grid.shape
        image = np.zeros((h, w, 3), dtype=np.uint8)

        # Color based on grid value (0=occupied, 1=free)
        image[grid == 1] = [255, 255, 255]  # White for free space
        image[grid == 0] = [0, 0, 0]  # Black for occupied

        return cv2.resize(image, (260, 180), interpolation=cv2.INTER_NEAREST)
    
def visualize_result_rgb(
    image: np.ndarray,
    detections: sv.Detections,
    classes: list[str],
    color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
    instance_random_color: bool = False,
    draw_bboxes: bool = True
) -> np.ndarray:
    """
    Visualize the results of a single image. Annotate the image with the detections.

    Parameters:
    image (numpy.ndarray): The image to be visualized.
    detections (sv.Detection): The detections to be visualized.
    classes (list[str]): The classes to be visualized.
    color (Color | ColorPalette): The color palette to be used. Default is ColorPalette.default().
    instance_random_color (bool): Whether or not to use a random color for each instance. Default is False.
    draw_bboxes (bool): Whether or not to draw bounding boxes. Default is True.

    Returns:
    annotated_image (numpy.ndarray): The visualized image with detections.
    labels (list[str]): The labels for each detection.
    """
    image = image.copy()
    
    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        color = color
    )
    mask_annotator = sv.MaskAnnotator(
        color = color
    )
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_scale=0.5,
        text_thickness=1,
        text_padding=2
    )
    
    if hasattr(detections, "confidence") and hasattr(detections, "class_id"):
        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is not None and class_ids is not None:
            labels = [f"{classes[class_id]} {confidence:.2f}" for confidence, class_id in zip(confidences, class_ids)]
        else:
            labels = [f"{classes[class_id]}" for class_id in class_ids]
    else:
        print("No confidence or class_id detected! Or one of them is missing!")
    
    if instance_random_color:
        # generate random color for each instance
        # shallow copy
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))
    
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    
    if draw_bboxes:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    return annotated_image, labels


def show_obs_result(
    image: np.ndarray,
    obs: dict,
    classes: list[str],
    caption: Union[str, None] = None,
    draw_label_size: bool = False
):
    detections = sv.Detections(
        xyxy=obs['xyxy'],
        confidence=obs['confidence'],
        class_id=obs['class_id'],
        mask=obs['masks'],
    )
    
    annotated_image, _ = visualize_result_rgb(image, detections, classes)
    
    if draw_label_size:
        text = "label num: " + str(len(obs['xyxy']))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 100)  # Text start position
        font_scale = 1  # Font size
        color = (0, 255, 255)  # Text color (yellow)
        thickness = 2  # Text thickness
        cv2.putText(annotated_image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        
    cv2.imshow(caption, annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_det_result(
    image: np.ndarray,
    detections: sv.Detections,
    classes: list[str],
    caption: Union[str, None] = None,
    draw_label_size: bool = False
):
    annotated_image, _ = visualize_result_rgb(image, detections, classes)
    
    if draw_label_size:
        text = "label num: " + str(len(detections.xyxy))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 100)  # Text start position
        font_scale = 1  # Font size
        color = (0, 255, 255)  # Text color (yellow)
        thickness = 2  # Text thickness
        cv2.putText(annotated_image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    
    cv2.imshow(caption, annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_similarity_matrix(
    sim_mat: np.ndarray,
):
    plt.figure()
    plt.imshow(sim_mat, cmap='coolwarm', interpolation='nearest')

    # Add colorbar
    plt.colorbar(label='Correlation')

    # Add title and labels
    plt.title('Correlation Matrix')
    plt.xlabel('Object A')
    plt.ylabel('Object B')
    # Show image
    plt.show()

