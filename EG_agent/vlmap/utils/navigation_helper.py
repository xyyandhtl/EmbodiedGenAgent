import cv2
import random
import logging

import numpy as np
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, KDTree
from scipy.ndimage import binary_erosion
from dynaconf import Dynaconf
from scipy.ndimage import binary_dilation
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

logger = logging.getLogger(__name__)

class LayoutMap:
    def __init__(self, cfg, resolution=0.1, percentile=90, min_area=5, kernel_size=3):
        """
        Initialize the LayoutMap class.

        Args:
            resolution: Size of each grid cell (in meters).
            percentile: Percentile threshold for binarization.
            min_area: Minimum area for removing small connected components.
            kernel_size: Kernel size for morphological operations.
        """
        self.cfg = cfg
        self.resolution = resolution
        self.percentile = percentile
        self.min_area = min_area
        self.kernel_size = kernel_size

        self.point_cloud: o3d.geometry.PointCloud = None
        self.occ_map: np.ndarray = None
        self.x_edges: np.ndarray = None
        self.y_edges: np.ndarray = None

    def set_layout_pcd(self, layout_pcd, current_z):
        """
        Load point cloud and generate Occupancy Map.

        Args:
            layout_pcd: Point cloud data.
        """
        self.point_cloud = layout_pcd
        self.occ_map, self.x_edges, self.y_edges = self.create_occupancy_map(current_z)
        print("Occupancy Map created.")

    def create_occupancy_map(self, current_z):
        """
        Create Occupancy Map from point cloud data.
        """
        points = np.asarray(self.point_cloud.points)
        # 只保留 z 在 0.15 ~ 0.65 的点
        points = points[(points[:, 2] > current_z - 0.45) & (points[:, 2] < current_z + 0.05)]
        xy_points = points[:, :2]
        x_min, y_min = np.min(xy_points, axis=0)
        x_max, y_max = np.max(xy_points, axis=0)

        # 使用二维直方图创建占用地图
        occ_map, x_edges, y_edges = np.histogram2d(
            xy_points[:, 0],  # x坐标
            xy_points[:, 1],  # y坐标
            # x,y方向上的 grid_cell 个数
            bins=(int((x_max - x_min) / self.resolution), int((y_max - y_min) / self.resolution))
        )
        return occ_map, x_edges, y_edges

    def update_local_layout_occmap(self, partial_pcd, current_pose, update_radius):
        """
        Update the layout occ_map with a partial point cloud in a local region.
        Expands the occ_map size if necessary.
        """
        if self.occ_map is None:
            print("[LayoutMap] Occupancy map not initialized. Cannot perform local update.")
            return

        # 1) Calculate the bounds of the local update region in world coordinates
        center_world = current_pose[:3, 3]
        min_bound_world = center_world - update_radius
        max_bound_world = center_world + update_radius

        # 2) Check if the current occ_map bounds are sufficient
        x_min_world, x_max_world = self.x_edges[0], self.x_edges[-1]
        y_min_world, y_max_world = self.y_edges[0], self.y_edges[-1]

        expand_left = max(0, int(np.ceil((x_min_world - min_bound_world[0]) / self.resolution)))
        expand_right = max(0, int(np.ceil((max_bound_world[0] - x_max_world) / self.resolution)))
        expand_bottom = max(0, int(np.ceil((y_min_world - min_bound_world[1]) / self.resolution)))
        expand_top = max(0, int(np.ceil((max_bound_world[1] - y_max_world) / self.resolution)))

        if expand_left > 0 or expand_right > 0 or expand_bottom > 0 or expand_top > 0:
            # Expand occ_map and update x_edges and y_edges
            new_x_bins = self.occ_map.shape[0] + expand_left + expand_right
            new_y_bins = self.occ_map.shape[1] + expand_bottom + expand_top

            new_occ_map = np.zeros((new_x_bins, new_y_bins), dtype=self.occ_map.dtype)
            new_occ_map[expand_left:expand_left + self.occ_map.shape[0],
                        expand_bottom:expand_bottom + self.occ_map.shape[1]] = self.occ_map

            self.occ_map = new_occ_map
            self.x_edges = np.linspace(x_min_world - expand_left * self.resolution,
                                        x_max_world + expand_right * self.resolution,
                                        new_x_bins + 1)
            self.y_edges = np.linspace(y_min_world - expand_bottom * self.resolution,
                                        y_max_world + expand_top * self.resolution,
                                        new_y_bins + 1)

        # 3) Calculate the grid indices for the local update region
        min_x_grid = np.floor((min_bound_world[0] - self.x_edges[0]) / self.resolution).astype(int)
        min_y_grid = np.floor((min_bound_world[1] - self.y_edges[0]) / self.resolution).astype(int)
        max_x_grid = np.ceil((max_bound_world[0] - self.x_edges[0]) / self.resolution).astype(int)
        max_y_grid = np.ceil((max_bound_world[1] - self.y_edges[0]) / self.resolution).astype(int)

        # Clamp to the map boundaries
        min_x = max(0, min_x_grid)
        min_y = max(0, min_y_grid)
        max_x = min(self.occ_map.shape[0] - 1, max_x_grid)
        max_y = min(self.occ_map.shape[1] - 1, max_y_grid)

        # 4) Clear the local region and update with new data
        self.occ_map[min_x:max_x + 1, min_y:max_y + 1] = 0

        points = np.asarray(partial_pcd.points)
        current_z = center_world[2]
        points = points[(points[:, 2] > current_z - 0.45) & (points[:, 2] < current_z + 0.05)]
        if points.shape[0] > 0:
            xy_points = points[:, :2]
            new_occ, _, _ = np.histogram2d(
                xy_points[:, 0],
                xy_points[:, 1],
                bins=[self.x_edges, self.y_edges]
            )
            self.occ_map += new_occ

    def calculate_threshold(self, method="percentile"):
        """
        Calculate threshold based on Occupancy Map.

        Args:
            method: Threshold calculation method, options: "mean", "median", or "percentile".
        """
        non_zero_values = self.occ_map[self.occ_map > 0]  # 占用地图中 grid_cell 中的 非零值
        if method == "mean":
            return np.mean(non_zero_values)
        elif method == "median":
            return np.median(non_zero_values)
        elif method == "percentile":
            return np.percentile(non_zero_values, self.percentile)
        else:
            raise ValueError("Unsupported threshold calculation method.")

    def process_binary_map(self, remove_small_components=False):
        """
        Process binary map with connected component filtering and morphological operations.
        """
        if self.occ_map is None:
            logger.warning("[LayoutMap][process_binary_map] occ_map is None, return.")
            return None
        
        # Binarization
        threshold = 0   # TODO: a hyperparameter
        # threshold = self.calculate_threshold()
        # print(f"[LayoutMap] Binarization threshold: {threshold}")
        binary_map = (self.occ_map > threshold).astype(np.uint8)

        # Remove small connected components
        if remove_small_components:
            # cleaned_map = np.zeros_like(binary_map, dtype=np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
            binary_map = np.zeros_like(binary_map, dtype=np.uint8)

            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= self.min_area:
                    binary_map[labels == label] = 1

        # Morphological operation (closing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        processed_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)

        if self.cfg.edit_wall:
            processed_map = self.visualize_and_edit_map(processed_map)

        return processed_map

    def visualize_and_edit_map(self, processed_map):
        edited_map = processed_map.copy()
        cell_size = 3  # Adjust the size of each cell for better visualization
        drawing = False
        
        # Function to handle mouse events
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing
            grid_x = x // cell_size
            grid_y = y // cell_size
            if 0 <= grid_x < edited_map.shape[1] and 0 <= grid_y < edited_map.shape[0]:
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    edited_map[grid_y, grid_x] = 1 - edited_map[grid_y, grid_x]
                    update_display()
                elif event == cv2.EVENT_MOUSEMOVE and drawing:
                    if edited_map[grid_y, grid_x] == 0:
                        edited_map[grid_y, grid_x] = 1
                    update_display()
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False

        # Function to update display after modification
        def update_display():
            display_map = cv2.resize(edited_map * 255, (edited_map.shape[1] * cell_size, edited_map.shape[0] * cell_size), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Editable Map", display_map)
        
        # Create window and set mouse callback
        cv2.namedWindow("Editable Map")
        cv2.setMouseCallback("Editable Map", mouse_callback)
        update_display()

        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):  # Press 'q' to finish
                break

        cv2.destroyAllWindows()
        return edited_map

class PathPlanner:
    def __init__(self, binary_occ_map, map_origin, map_resolution, robot_radius, floor_height):
        """
        Initializes the PathPlanner with map information.

        Args:
            binary_occ_map (np.array): The occupancy grid (0=free, 1=obstacle).
            map_origin (np.array): The world coordinate of the grid's (0,0) cell.
            map_resolution (float): The size of one grid cell in meters.
            robot_radius (float): The robot's radius for inflating obstacles.
            floor_height (float): The z-coordinate for the path points.
        """
        self.map_origin = map_origin
        self.resolution = map_resolution
        self.floor_height = floor_height

        # Inflate map for collision avoidance
        self.inflated_map = self._inflate_obstacles(binary_occ_map, robot_radius, map_resolution)

        # Create the pathfinding grid object
        self.pathfinding_matrix = np.where(self.inflated_map == 0, 1, 0)
        self.grid = Grid(matrix=self.pathfinding_matrix.T.tolist())
        # cv2.imwrite("inflated_map.png", self.inflated_map * 255)
        logger.info(f"[PathPlanner] Grid {self.grid}, map origin: {map_origin}, resolution: {map_resolution}")

    def _inflate_obstacles(self, binary_occ, robot_radius, resolution):
        """
        Inflates obstacles in the occupancy grid to account for the robot's radius.
        """
        if robot_radius == 0 or resolution == 0:
            return binary_occ

        inflation_radius_cells = int(np.ceil(robot_radius / resolution))
        y, x = np.ogrid[-inflation_radius_cells:inflation_radius_cells + 1, -inflation_radius_cells:inflation_radius_cells + 1]
        structure = x ** 2 + y ** 2 <= inflation_radius_cells ** 2

        inflated_map = binary_dilation(binary_occ, structure=structure)
        return inflated_map.astype(int)

    def _simplify_path(self, path, min_distance=10):
        """
        Simplifies a grid-based path by removing collinear points.
        """
        if len(path) < 3:
            return path

        simplified_path = [path[0]]
        for i in range(1, len(path) - 1):
            vec1 = (path[i].x - simplified_path[-1].x, path[i].y - simplified_path[-1].y)
            vec2 = (path[i+1].x - path[i].x, path[i+1].y - path[i].y)
            if vec1 != vec2:
                dist = np.sqrt((path[i].x - simplified_path[-1].x)**2 + (path[i].y - simplified_path[-1].y)**2)
                if dist >= min_distance:
                    simplified_path.append(path[i])

        simplified_path.append(path[-1])
        return simplified_path

    def world_to_grid(self, world_pos):
        grid_pos = np.floor((np.array(world_pos[:2]) - self.map_origin) / self.resolution).astype(int)
        return tuple(grid_pos)

    def grid_to_world(self, grid_pos):
        world_pos = (np.array(grid_pos) * self.resolution) + self.map_origin
        return [world_pos[0], world_pos[1], self.floor_height]

    def sample_random_world_goal(self):
        """
        Samples a random traversable point from the grid with higher probability near edges,
        and returns its world coordinate.
        """
        traversable_points_yx = np.argwhere(self.pathfinding_matrix.T == 1)
        if traversable_points_yx.size == 0:
            logger.warning("[PathPlanner] No traversable points found to sample from.")
            return []

        H, W = self.pathfinding_matrix.shape

        # 计算每个点距离边界的最小距离（越靠边，值越小）
        distances_to_edge = np.min(
            np.stack([
                traversable_points_yx[:, 0],              # distance to top
                H - 1 - traversable_points_yx[:, 0],      # distance to bottom
                traversable_points_yx[:, 1],              # distance to left
                W - 1 - traversable_points_yx[:, 1]       # distance to right
            ], axis=1),
            axis=1
        )

        # 将距离转成“靠边权重”，距离越小，权重越大
        epsilon = 1e-6  # 防止除零
        weights = 1.0 / (distances_to_edge + epsilon)

        # 平滑化或增强权重效果（可调参数）
        weights = weights ** 2

        # 归一化权重
        weights = weights / np.sum(weights)

        # 按权重采样一个索引
        random_index = np.random.choice(len(traversable_points_yx), p=weights)
        random_grid_point_yx = traversable_points_yx[random_index]
        random_grid_point = (random_grid_point_yx[1], random_grid_point_yx[0])

        return self.grid_to_world(random_grid_point)

    def plan_path(self, start_world, goal_world, clear_radius=3):
        """
        clear_radius * resolution (0.1 m) is the true clear radius in meters
        """
        start_grid = self.world_to_grid(start_world)
        goal_grid = self.world_to_grid(goal_world)
        if start_grid[0] < 0 or start_grid[0] >= self.grid.width or start_grid[1] < 0 or start_grid[1] >= self.grid.height:
            logger.error(f"[PathPlanner] Start point {start_grid} is out of bounds.")
            return []
        if goal_grid[0] < 0 or goal_grid[0] >= self.grid.width or goal_grid[1] < 0 or goal_grid[1] >= self.grid.height:
            logger.error(f"[PathPlanner] Goal point {goal_grid} is out of bounds.")
            return []

        # 如果 start 不可行走，则临时清除周围障碍
        start_node = self.grid.node(start_grid[0], start_grid[1])
        if not start_node.walkable:
            logger.warning(f"[PathPlanner] Start point {start_grid} is on an obstacle or out of bounds. Clearing nearby obstacles.")
            for dx in range(-clear_radius, clear_radius + 1):
                for dy in range(-clear_radius, clear_radius + 1):
                    nx, ny = start_grid[0] + dx, start_grid[1] + dy
                    if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                        self.grid.node(nx, ny).walkable = True

        # 如果 goal 不可行走，则寻找最近 walkable 点
        goal_node = self.grid.node(goal_grid[0], goal_grid[1])
        if not goal_node.walkable:
            logger.warning(f"[PathPlanner] Original goal {goal_grid} is on an obstacle or out of bounds. Snapping to nearest walkable node.")
            free_nodes_yx = np.argwhere(self.pathfinding_matrix.T == 1)
            if free_nodes_yx.size == 0:
                logger.error("[PathPlanner] No walkable nodes found on the entire map.")
                return []
            goal_grid_yx = np.array([goal_grid[1], goal_grid[0]])
            distances = np.linalg.norm(free_nodes_yx - goal_grid_yx, axis=1)
            snapped_yx = free_nodes_yx[np.argmin(distances)]
            goal_grid = (snapped_yx[1], snapped_yx[0])
            logger.info(f"[PathPlanner] Snapped goal to {goal_grid}.")

        # A* 寻路
        start_node = self.grid.node(start_grid[0], start_grid[1])
        end_node = self.grid.node(goal_grid[0], goal_grid[1])
        finder = AStarFinder(diagonal_movement=True)
        path_grid_coords, _ = finder.find_path(start_node, end_node, self.grid)

        # 简化并转换回世界坐标
        if path_grid_coords:
            simplified_path_grid = self._simplify_path(path_grid_coords)
            logger.info(f"[PathPlanner] Path simplified from {len(path_grid_coords)} to {len(simplified_path_grid)} points.")
            path_world_coords = [self.grid_to_world((p.x, p.y)) for p in simplified_path_grid]
            return path_world_coords
        else:
            logger.warning("[PathPlanner] A* failed to find a path.")
            return []

# functions used in core for path refine
def remaining_path(path, current_pose):
    """
    Calculate the remaining path from the current pose along the global path.

    Parameters:
    - global_path: List of 3D points [(x1, y1, z1), (x2, y2, z2), ...].
    - current_pose: 4x4 numpy array representing the current transformation matrix.

    Returns:
    - remaining_path: List of remaining 3D points [(x, y, z), ...].
    """
    # Extract current position (translation part) from the transformation matrix
    current_position = current_pose[:3, 3]
    current_xy = current_position[:2]  # Ignore Z component

    # Find the closest point in the global path to the current position (based on XY distance only)
    distances = [np.linalg.norm(np.array(point[:2]) - current_xy) for point in path]
    closest_idx = np.argmin(distances)

    # Ensure we do not go backwards in the path
    remaining_path = path[closest_idx:]
    return remaining_path

def angle_between_points_3d(p1, p2, p3):
    """
    Calculate the angle between three 3D points on the XY plane (in degrees).

    Args:
        p1, p2, p3: 3D points (x, y, z).

    Returns:
        angle: The angle between the three points (in degrees, 0-180).
    """
    # Extract XY coordinates, ignoring Z
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Calculate the angle between vectors
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    # Avoid division by zero
    if norm_product == 0:
        return 0

    # Calculate the angle (range: 0-180 degrees)
    cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))

    return angle

def remove_sharp_turns_3d(path, angle_threshold=60):
    """
    Remove sharp turns in the 3D path on the XY plane, recursively processing until all angles meet the threshold.

    Args:
        path: List of 3D points [(x1, y1, z1), (x2, y2, z2), ...].
        angle_threshold: Angle threshold; turns greater than this are considered sharp turns.

    Returns:
        filtered_path: Path after removing sharp turns.
    """

    def filter_once(path):
        """Remove sharp turns in a single pass."""
        filtered_path = [path[0]]  # Keep the start point
        for i in range(1, len(path) - 1):
            p1 = path[i - 1]
            p2 = path[i]
            p3 = path[i + 1]

            angle = angle_between_points_3d(p1, p2, p3)

            # Keep the point if the angle is less than the threshold
            if angle < angle_threshold:
                filtered_path.append(p2)
        filtered_path.append(path[-1])  # Keep the end point
        return filtered_path

    # Recursively remove sharp turns until the path no longer changes
    previous_path = []
    filtered_path = path
    while filtered_path != previous_path:
        previous_path = filtered_path
        filtered_path = filter_once(filtered_path)

    return filtered_path


