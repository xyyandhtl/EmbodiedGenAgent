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

    def get_semantic_map_image(self, global_map_manager, current_pose=None, nav_path=None) -> None | np.ndarray:
        """
        Generates a top-down 2D image of the global semantic map, including the navigation path and current robot position.
        """
        from PIL import Image, ImageDraw, ImageFont

        if not global_map_manager.has_global_map():
            return None

        # Get all points from the global map
        all_points = np.vstack([np.asarray(obj.pcd_2d.points) for obj in global_map_manager.global_map if not obj.pcd_2d.is_empty()])
        all_points = all_points[np.isfinite(all_points).all(axis=1)]  # Filter out invalid values
        if all_points.size == 0:
            return None

        # Determine the bounding box of whole pcd
        min_coords = np.min(all_points[:, :2], axis=0)
        max_coords = np.max(all_points[:, :2], axis=0)
        map_size = max_coords - min_coords  # the size of the image

        resolution = 0.05  # meters per pixel
        scale_factor = 6.0
        padding = 100  # pixels
        width = int((map_size[0]) / resolution * scale_factor) + padding
        height = int((map_size[1]) / resolution * scale_factor) + padding
        
        pil_img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(pil_img)

        try:
            # Use a common, often pre-installed, anti-aliased font.
            # If not found, PIL will fall back to a default bitmap font.
            font = ImageFont.truetype("DejaVuSans.ttf", size=int(12 * (scale_factor / 3)))
            coord_font = ImageFont.truetype("DejaVuSans.ttf", size=int(10 * (scale_factor / 3)))
        except IOError:
            logger.warning("[Visualizer] DejaVuSans.ttf not found. Falling back to default font.")
            font = ImageFont.load_default()
            coord_font = ImageFont.load_default()

        # Transform points to image coordinates (with scaling)
        def world_to_img(point):
            point_img = ((point[:2] - min_coords) / resolution * scale_factor).astype(int)
            point_img[0] += padding // 2
            point_img[1] = height - point_img[1] - (padding // 2)  # flip Y axis
            return tuple(point_img)

        placed_label_boxes = [] # List to store bounding boxes of placed labels

        # 1. Draw each object
        for obj in global_map_manager.global_map:
            points = np.asarray(obj.pcd_2d.points)
            points = points[np.isfinite(points).all(axis=1)]
            if points.shape[0] == 0:
                continue
            # (1) Draw the points with class color on the image
            # Get the color for the object
            obj_name = self.obj_classes.get_classes_arr()[obj.class_id]
            color_rgb = self.obj_classes.get_class_color(obj_name)  # float RGB color (0-1 range)
            color_rgb_int = tuple(int(c * 255) for c in color_rgb)  # (0-255 range)

            # Transform points to image coordinates (with scaling)
            points_img = np.array([world_to_img(p) for p in points])

            radius = int(2 * (scale_factor / 3))
            for p_img in points_img:
                draw.ellipse([p_img[0]-radius, p_img[1]-radius, p_img[0]+radius, p_img[1]+radius], fill=color_rgb_int)

            # (2) Draw the class text on the image (with collision detection)
            # Get object bbox in image coords and text size
            obj_x_min, obj_y_min = np.min(points_img, axis=0)
            obj_x_max, obj_y_max = np.max(points_img, axis=0)
            centroid_img = np.mean(points_img, axis=0).astype(int)

            text_bbox = draw.textbbox((0, 0), obj_name, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

            # text_bbox candidate positions
            candidates = [
                (centroid_img[0] - text_w // 2, centroid_img[1] - text_h // 2 - 5),  # Center
                (obj_x_min + (obj_x_max - obj_x_min) // 2 - text_w // 2, obj_y_min - text_h),  # Top-center
                (obj_x_max + 1, obj_y_min + text_h // 2),  # Top-right
                (obj_x_min - text_w - 1, obj_y_min + text_h // 2),  # Top-left
                (obj_x_max + 1, obj_y_max),  # Bottom-right
                (obj_x_min - text_w - 1, obj_y_max),  # Bottom-left
            ]

            final_pos = None
            # Find a non-colliding position
            for pos in candidates:
                lx, ly = pos
                label_box = (lx, ly, lx + text_w, ly + text_h)

                # Check for collision with image boundaries
                if not (label_box[0] >= 0 and label_box[1] >= 0 and label_box[2] < width and label_box[3] < height):
                    continue

                # Check for collision with other labels
                is_colliding = False
                for placed_box in placed_label_boxes:
                    if not (label_box[2] < placed_box[0] or \
                            label_box[0] > placed_box[2] or \
                            label_box[3] < placed_box[1] or \
                            label_box[1] > placed_box[3]):
                        is_colliding = True
                        break
                if not is_colliding:
                    final_pos = pos
                    placed_label_boxes.append(label_box)
                    break
            # If no good position is found, fall back to the center
            if final_pos is None:
                final_pos = candidates[0]
            draw.text(final_pos, obj_name, font=font, fill=(0, 0, 0))

        # 2. Draw navigation path
        if nav_path and len(nav_path) > 1:
            print(f"nav_path: {nav_path}")
            path_points_img = []
            for point in nav_path:
                if isinstance(point, dict) and 'pose' in point and 'position' in point['pose']:
                    pos = point['pose']['position']
                    p = np.array([pos['x'], pos['y'], pos['z']])
                elif isinstance(point, (list, np.ndarray)) and len(point) >= 2:
                    p = np.array(point)
                else:
                    continue
                path_point_img = world_to_img(p)
                path_points_img.append(path_point_img)
            if len(path_points_img) > 1:
                draw.line(path_points_img, fill=(0, 255, 0), width=3) # Green line

        # Draw current pose as an arrow
        if current_pose is not None:
            pos = current_pose[:3, 3]
            rot_matrix = current_pose[:3, :3]
            
            # Forward vector in ROS is +Z
            fwd_vec_local = np.array([0, 0, 1])
            fwd_vec_world = rot_matrix @ fwd_vec_local
            
            pos_img = world_to_img(pos)

            # Arrow properties
            arrow_length = 16 * (scale_factor / 3)
            arrow_color = (255, 0, 0) # Red

            # Calculate arrow tip
            # We only care about the 2D direction on the map (X, Y)
            # The Y-axis is flipped in the image, so we subtract the y-component of the direction
            fwd_vec_2d = fwd_vec_world[:2]
            fwd_vec_2d_normalized = fwd_vec_2d / (np.linalg.norm(fwd_vec_2d) + 1e-6)

            tip_x = pos_img[0] + arrow_length * fwd_vec_2d_normalized[0]
            tip_y = pos_img[1] - arrow_length * fwd_vec_2d_normalized[1] # Subtract because Y is flipped

            # Define triangle points for the arrow
            # Perpendicular vector for arrow base
            perp_vec_2d = np.array([-fwd_vec_2d_normalized[1], fwd_vec_2d_normalized[0]])
            base_width = 8 * (scale_factor / 3)
            
            base_center_x = pos_img[0] - 0.5 * arrow_length * fwd_vec_2d_normalized[0]
            base_center_y = pos_img[1] + 0.5 * arrow_length * fwd_vec_2d_normalized[1]

            p1 = (tip_x, tip_y)
            p2 = (base_center_x + base_width * perp_vec_2d[0], base_center_y - base_width * perp_vec_2d[1])
            p3 = (base_center_x - base_width * perp_vec_2d[0], base_center_y + base_width * perp_vec_2d[1])

            draw.polygon([p1, p2, p3], fill=arrow_color)
            # Add coordinates text
            coord_text = f"({pos[0]:.2f}, {pos[1]:.2f})"
            text_pos = (pos_img[0] + 15, pos_img[1])
            draw.text(text_pos, coord_text, font=coord_font, fill=(0, 0, 0)) # Black text

        # Convert PIL (RGB) image to numpy array (BGR) for OpenCV
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
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

