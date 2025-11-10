import os
import json
import multiprocessing
import logging
import threading
import time
from pathlib import Path
from typing import List
from collections import deque
import cv2
import numpy as np
import open3d as o3d
from dynaconf import Dynaconf
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from EG_agent.vlmap.utils.object import GlobalObject
from EG_agent.vlmap.utils.types import Observation, GoalMode, ObjectClasses
from EG_agent.vlmap.utils.base_map_manager import BaseMapManager
from EG_agent.vlmap.utils.navigation_helper import LayoutMap, PathPlanner

# Set up the module-level logger
logger = logging.getLogger(__name__)


class GlobalMapManager(BaseMapManager):
    def __init__(
        self,
        cfg: Dynaconf,
    ) -> None:
        super().__init__(cfg)

        # global objects list
        self.global_map: List[GlobalObject] = []

        # set global flag in tracker
        self.tracker.set_global()

        GlobalObject.initialize_config(cfg)

        # For navigation --> Pathfinding
        self.path_planner: PathPlanner = None
        self.last_inflated_map = None
        self.inquiry = ''
        self.action_path = []
        self.has_action_path = False

        # layout information --> LayoutMap
        layout_resolution = float(self.cfg.layout_voxel_size)
        
        self.layout_map = LayoutMap(cfg, resolution=layout_resolution, percentile=90, min_area=5, kernel_size=3)
        self.binary_occ: np.ndarray | None = None

        # pass to the local map manager for inquiry
        self.global_candidate_bbox = None
        self.global_candidate_score = 0.0
        # for lost and found
        # save the previous queried global obj uids
        self.ignore_global_obj_list = []

        self.best_candidate_name = None
        
        self.preload_path_ok = False

        # Caching for the semantic map
        self.cached_semantic_map = None
        self.semantic_map_dirty = True
        self.semantic_map_metadata = {}

        # Caching for the traversable map
        self.cached_traversable_map = None
        self.traversable_map_dirty = True
        self.traversable_map_metadata = {}

        # Object classes
        # --- 加载指定的 要识别的 全部物体的 类别text ---
        classes_path = cfg.yolo.given_classes_path
        logger.info(f"[Detector][Init] Using given classes, path:{classes_path}")

        # Object classes
        self.obj_classes = ObjectClasses(
            classes_file_path=classes_path,
            bg_classes=self.cfg.yolo.bg_classes,
            skip_bg=self.cfg.yolo.skip_bg)

        # Store dynamic parameters for background updates
        self._curr_pose: np.ndarray = None  # Real-time pose
        self._nav_path: list = []
        self._traj_path = deque(maxlen=60)
        self._traj_path_lock = threading.Lock()  # Add a lock for thread-safe access to _traj_path
        self.goal_grid: tuple | None = None

        self.layout_initialized = False

        self.scale_factor = 2.0
        self.padding = 0
        self.resolution = float(self.cfg.resolution)
        self._cached_static_image = None  # Cache for static elements
        self._cached_static_metadata: dict = {
            'resolution': self.resolution, 
            'scale_factor': self.scale_factor,
            'padding': self.padding}  # Metadata for static elements
        self._dynamic_layer = None  # Cache for dynamic elements
        self._world_to_img_cache = {}  # Cache for world-to-image coordinate mapping
 
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", size=int(12 * (self.scale_factor / 2)))
        except IOError:
            self.font = ImageFont.load_default()

    def has_global_map(self) -> bool:
        return len(self.global_map) > 0

    def set_layout_info(self, layout_pcd, force_full_update=False, update_radius=None):
        if force_full_update or not self.layout_initialized:
            # First time or forced: process the whole layout
            self.layout_map.set_layout_pcd(layout_pcd, self._curr_pose[2, 3])
            self.layout_initialized = True
            logger.info("[GlobalMapManager] Initialized/updated layout map (occ_map only).")
        else:
            # Subsequent live updates: process only a local part
            if self._curr_pose is not None and update_radius:
                self.layout_map.update_local_layout_occmap(layout_pcd, self._curr_pose, update_radius)
                logger.debug("[GlobalMapManager] Updated layout occ_map with partial point cloud.")
            else:
                logger.warning("[GlobalMapManager] Skipping live layout update because current_pose is not provided or update_radius is 0.")

    def process_observations(
        self,
        curr_observations: List[Observation]
    ) -> None:

        # for debug, show the preload global map
        if len(self.global_map) > 0 and self.cfg.use_rerun:
            self.visualize_global_map()

        if len(curr_observations) == 0:
            logger.debug("[GlobalMap] No global observation update this time, return")
            return

        if self.is_initialized == False:
            # Init the global map
            logger.info("[GlobalMap] Init Global Map by first Local Map input")
            self.global_map = self.init_from_observation(curr_observations)
            self.is_initialized = True
            return

        # The test part, no matching just adding
        if self.cfg.no_update:
            logger.debug("[GlobalMap] No update mode, simply adding")
            for obs in curr_observations:
                self.global_map.append(GlobalObject(obs))

            if self.cfg.use_rerun:
                self.visualize_global_map()
            
            return

        # if not the first, then do the global matching
        logger.debug("[GlobalMap] Matching")
        self.tracker.set_current_frame(curr_observations)

        # Set tracker reference
        self.tracker.set_ref_map(self.global_map)
        self.tracker.matching_map()

        # After matching map, current frame information will be updated
        curr_observations = self.tracker.get_current_frame()

        # Update global map
        self.update_global_map(curr_observations)
        # visualize the global map


        if self.cfg.use_rerun:
            self.visualize_global_map()

    def init_from_observation(
        self,
        curr_observations: List[Observation]
    ) -> List[GlobalObject]:

        # global_map = []

        # for each local object, generate a global object and add to global_map
        for global_obs in curr_observations:

            global_obj = GlobalObject()
            global_obj.add_observation(global_obs)
            global_obj.update_info()

            self.global_map.append(global_obj)

        return self.global_map

    def update_global_map(
        self,
        curr_observations: List[Observation]
    ) -> None:
        # update the local map with the lateset observation
        for obs in curr_observations:
            if obs.matched_obj_idx == -1:
                # Add new global object
                global_obj = GlobalObject()
                global_obj.add_observation(obs)
                global_obj.update_info()
                self.global_map.append(global_obj)
            else:
                # Update existed global object
                matched_obj_idx = obs.matched_obj_idx
                matched_obj = self.global_map[matched_obj_idx]
                matched_obj.add_observation(obs)
                matched_obj.update_info()

        pass

    def save_map(
        self
    ) -> None:
        # get the directory
        save_dir = self.cfg.map_save_path

        # if os.path.exists(save_dir):
        #     shutil.rmtree(save_dir)
        #     logger.info(f"[GlobalMap] Cleared the directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        for i, obj in enumerate(self.global_map):
            if obj.save_path is not None:
                obj.save_path = obj._initialize_save_path()
                logger.debug(f"[GlobalMap] Saving No.{i} obj: {obj.save_path}")
                obj.save_to_disk()
            else:
                logger.info("[GlobalMap] No save path for local object")
                continue

    def load_map(self) -> None:
        """
        Load saved global map objects. If preload path exists, use it first;
        otherwise use default save path. If directory doesn't exist or is empty, do nothing.
        """
        # Use preload_global_map_path first, if not exists then use map_save_path
        if os.path.exists(self.cfg.preload_path):
            load_dir = self.cfg.preload_path
            logger.info(f"[GlobalMap] Using preload global map path: {load_dir}")
        else:
            load_dir = self.cfg.map_save_path
            logger.info(f"[GlobalMap] Preload path not found. Using default map save path: {load_dir}")

        # Check if directory exists
        if not os.path.exists(load_dir):
            logger.warning(f"[GlobalMap] Directory {load_dir} does not exist. Skipping map loading.")
            return

        # Get .pkl files in directory
        pkl_files = [file for file in os.listdir(load_dir) if file.endswith(".pkl")]

        # Skip loading if no .pkl files
        if not pkl_files:
            logger.warning(f"[GlobalMap] No .pkl files found in {load_dir}. Skipping map loading.")
            return

        # Load .pkl files into global map
        for file in pkl_files:
            obj_results_path = os.path.join(load_dir, file)
            loaded_obj = GlobalObject.load_from_disk(obj_results_path)

            if self.cfg.floor_height:
                # Convert the open3d Vector3dVector to a numpy array
                points = np.asarray(loaded_obj.pcd_2d.points)

                # Now you can modify the Z values (third column)
                points[:, 2] = self.cfg.floor_height

                # After modifying, convert it back to open3d Vector3dVector if needed
                loaded_obj.pcd_2d.points = o3d.utility.Vector3dVector(points)
                loaded_obj.bbox_2d = loaded_obj.pcd_2d.get_axis_aligned_bounding_box()

            self.global_map.append(loaded_obj)

        logger.info(f"[GlobalMap] Successfully preloaded {len(self.global_map)} objects")
        self.is_initialized = True
    
    def read_json_files(self, directory):
        data_records = {}

        # Get all JSON files in directory
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

        for json_file in json_files:
            file_path = os.path.join(directory, json_file)
            
            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Store data
            data_records[json_file] = data

        return data_records
    
    def update_path_planner(self) -> bool:
        if self.binary_occ is None or self.binary_occ.size == 0:
            logger.warning("[GlobalMapManager][update_path_planner] Binary occ_map not available.")
            return False

        # logger.info(f"binary_occ_map shape: {self.binary_occ.shape}")
        # logger.info(f"occ_map shape: {self.layout_map.occ_map.shape}")
        self.path_planner = PathPlanner(
            binary_occ_map=self.binary_occ,
            map_origin=np.array([self.layout_map.x_edges[0], self.layout_map.y_edges[0]]),
            map_resolution=self.layout_map.resolution,
            robot_radius=self.cfg.robot_radius,
            floor_height=self.cfg.floor_height
        )
        return True

    def process_binary_map(self):
        self.binary_occ = self.layout_map.process_binary_map()

    def calculate_global_path(
        self, goal_mode=GoalMode.POSE, goal_position=None
    ) -> List:
        """
        Calculates the global path by creating and using a PathPlanner instance.
        """
        if goal_position is None:
            logger.warning(f"[GlobalMapManager][calculate_global_path] No valid goal for {goal_mode}!")
            return []
        
        if not self.update_path_planner():
            return []

        # 2. Determine goal in world coordinates
        # goal_world = None
        # if goal_mode == GoalMode.POSE and goal_position is not None:
        #     goal_world = goal_position
        # elif goal_mode == GoalMode.RANDOM:
        #     logger.info("[GlobalMapManager] Goal mode: RANDOM. Sampling a random goal.")
        #     goal_world = self.path_planner.sample_random_world_goal()
        # elif goal_mode == GoalMode.CLICK:
        #     TODO

        # 3. Plan the path
        path = self.path_planner.plan_path(
            start_world=self._curr_pose[:3, 3],
            goal_world=goal_position
        )

        # 4. Update caches and return path
        self.last_inflated_map = self.path_planner.inflated_map
        self.mark_semantic_map_dirty()
        return path

    def find_best_candidate_with_inquiry(self):
        """
        This function finds the best candidate in the global map based on cosine similarity.
        It compares the input query with all objects in the global map and selects the object with the highest similarity.
        """
        text_query_ft = self.inquiry
        cos_sim = []
        obj_list = []

        # Loop through each object in the global map to calculate cosine similarity
        for obj in self.global_map:

            if obj.uid in self.ignore_global_obj_list:
                obj.nav_goal = False
                continue

            obj.nav_goal = False
            obj_feat = torch.from_numpy(obj.clip_ft).to("cuda")
            max_sim = F.cosine_similarity(text_query_ft.unsqueeze(0), obj_feat.unsqueeze(0), dim=-1).item()
            obj_name = self.obj_classes.get_classes_arr()[obj.class_id]
            logger.debug(f"[GlobalMap][Inquiry] =========={obj_name}==============")
            logger.debug(f"[GlobalMap][Inquiry] Itself: \t{max_sim:.3f}")

            # Check if there are related objects, if so calculate cosine similarity with related_objs
            if obj.related_objs:
                related_sims = []
                for related_obj_ft in obj.related_objs:
                    related_obj_ft_tensor = torch.from_numpy(related_obj_ft).to("cuda")
                    sim = F.cosine_similarity(text_query_ft.unsqueeze(0), related_obj_ft_tensor.unsqueeze(0),
                                              dim=-1).item()
                    related_sims.append(sim)
                    logger.debug(f"[GlobalMap][Inquiry] Related: \t{sim:.3f}")

                # Update max_sim with the largest similarity from related_objs
                max_sim = max(max_sim, max(related_sims))

            # Store the maximum similarity for this object
            cos_sim.append((obj, max_sim))

        # Now we have a list of tuples [(obj, max_sim), (obj, max_sim), ...]
        # Sort the objects by similarity, from highest to lowest
        sorted_candidates = sorted(cos_sim, key=lambda x: x[1], reverse=True)

        # --- FIX: Check if any candidates were found before accessing the list ---
        if not sorted_candidates:
            logger.debug("[GlobalMap][Inquiry] No matching object candidates found for the query.")
            return None, 0.0
        # --- END FIX ---

        # Get the best candidate (highest cosine similarity)
        best_candidate, best_similarity = sorted_candidates[0]

        if (
            self.global_candidate_score != 0.0
            and abs(best_similarity - self.global_candidate_score) > 0.1
        ):
            # now we find the same ancher object
            # find all the objects with the same name
            for obj in self.global_map:
                if obj.uid in self.ignore_global_obj_list:
                    continue
                obj_name = self.obj_classes.get_classes_arr()[obj.class_id]
                if obj_name == self.best_candidate_name:
                    obj_list.append(obj)
        
        if len(obj_list) != 0:
            best_candidate = obj_list[0]

        # Output the best candidate and its similarity
        best_candidate_name = self.obj_classes.get_classes_arr()[best_candidate.class_id]

        # logger.debug(f"[GlobalMap][Inquiry] We ignore {len(self.ignore_global_obj_list)} objects in this global query.")

        if self.best_candidate_name is None:
            self.best_candidate_name = best_candidate_name

        logger.debug(f"[GlobalMap][Inquiry] Memory Best Candidate '{self.best_candidate_name}'")

        # Set flag to the best candidate for visualization
        best_candidate.nav_goal = True

        # input("Press any key to continue...")

        return best_candidate, best_similarity

    def background_map_update(self):
        """Background thread worker that updates both semantic and traversable maps at low frequency."""
        if self.has_global_map():
            self._update_semantic_map_cache()
        if self.last_inflated_map is not None:
            self._update_traversable_map_cache()

    def _update_semantic_map_cache(self):
        """
        Optimized method to update the cached semantic map image with static and dynamic elements.
        - Static layer is only recomputed when semantic_map_dirty is True or geometry changes.
        - Dynamic layer (paths, trajectory, current pose) is always updated.
        """
        start = time.time()
        if not self.has_global_map():
            self.cached_semantic_map = None
            return

        # --- 1. Update static map if dirty ---
        if self.semantic_map_dirty:
            self.semantic_map_dirty = False

            if self.binary_occ is None:
                self.cached_semantic_map = None
                return

            x_edges, y_edges = self.layout_map.x_edges, self.layout_map.y_edges
            min_coords = np.array([x_edges[0], y_edges[0]])
            max_coords = np.array([x_edges[-1], y_edges[-1]])
            map_size = max_coords - min_coords

            width = int(map_size[0] / self.resolution * self.scale_factor) + self.padding
            height = int(map_size[1] / self.resolution * self.scale_factor) + self.padding

            # Store static metadata
            self._cached_static_metadata.update({
                "min_coords": min_coords,
                "width": width,
                "height": height,
            })

            pil_img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
            draw = ImageDraw.Draw(pil_img)

            # --- Draw walls more efficiently ---
            wall_color = (160, 160, 160, 255)
            occ_indices = np.argwhere(self.binary_occ == 1)
            if len(occ_indices) > 0:
                # Precompute coordinate bounds for all cells
                x0 = x_edges[occ_indices[:, 0]]
                y0 = y_edges[occ_indices[:, 1]]
                x1 = x_edges[occ_indices[:, 0] + 1]
                y1 = y_edges[occ_indices[:, 1] + 1]

                # Vectorized conversion to image coordinates
                p0s = np.stack([x0, y0, np.zeros_like(x0)], axis=-1)
                p1s = np.stack([x1, y1, np.zeros_like(x1)], axis=-1)
                p0s_img = self._world_to_img_batch(p0s)
                p1s_img = self._world_to_img_batch(p1s)

                for p0, p1 in zip(p0s_img, p1s_img):
                    left, right = min(p0[0], p1[0]), max(p0[0], p1[0])
                    top, bottom = min(p0[1], p1[1]), max(p0[1], p1[1])
                    draw.rectangle([left, top, right, bottom], fill=wall_color)

            # --- Draw objects ---
            placed_label_boxes = []
            for obj in self.global_map:
                if obj.pcd_2d.is_empty():
                    continue

                obj_name = self.obj_classes.get_classes_arr()[obj.class_id]
                color_rgb_int = tuple(int(c * 255) for c in self.obj_classes.get_class_color(obj_name)) + (255,)
                points = np.asarray(obj.pcd_2d.points)
                points_img = self._world_to_img_batch(points)

                # Draw point cloud efficiently
                radius = int(max(1, self.scale_factor * 0.5))
                for p_img in points_img:
                    draw.ellipse([p_img[0] - radius, p_img[1] - radius,
                                p_img[0] + radius, p_img[1] + radius],
                                fill=color_rgb_int)

                # Label placement with minimal overlap
                obj_x_min, obj_y_min = np.min(points_img, axis=0)
                obj_x_max, obj_y_max = np.max(points_img, axis=0)
                centroid_img = np.mean(points_img, axis=0).astype(int)
                text_bbox = draw.textbbox((0, 0), obj_name, font=self.font)
                text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

                candidates = [
                    (centroid_img[0] - text_w // 2, centroid_img[1] - text_h // 2 - 5),
                    (obj_x_min + (obj_x_max - obj_x_min) // 2 - text_w // 2, obj_y_min - text_h),
                    (obj_x_max + 1, obj_y_min + text_h // 2),
                    (obj_x_min - text_w - 1, obj_y_min + text_h // 2),
                    (obj_x_max + 1, obj_y_max),
                    (obj_x_min - text_w - 1, obj_y_max),
                ]

                final_pos = None
                for pos in candidates:
                    lx, ly = pos
                    label_box = (lx, ly, lx + text_w, ly + text_h)
                    if not (0 <= lx < width - text_w and 0 <= ly < height - text_h):
                        continue
                    if any(
                        not (label_box[2] < b[0] or label_box[0] > b[2] or
                            label_box[3] < b[1] or label_box[1] > b[3])
                        for b in placed_label_boxes
                    ):
                        continue
                    final_pos = pos
                    placed_label_boxes.append(label_box)
                    break
                if final_pos is None:
                    final_pos = candidates[0]

                draw.text(final_pos, obj_name, font=self.font, fill=(0, 0, 0, 255))

            self._cached_static_image = pil_img

        # --- 2. Dynamic layer ---
        static_img = self._cached_static_image
        if static_img is None:
            return

        dynamic_layer = Image.new("RGBA", static_img.size, (0, 0, 0, 0))
        draw_dynamic = ImageDraw.Draw(dynamic_layer)

        # --- Draw navigation path ---
        if self._nav_path and len(self._nav_path) > 1:
            path_points_img = self._world_to_img_batch(np.array(self._nav_path))
            path_points_tuples = [tuple(map(int, p)) for p in path_points_img]
            draw_dynamic.line(path_points_tuples, fill=(0, 255, 0, 255), width=6)
            for point in path_points_tuples:
                draw_dynamic.ellipse([point[0] - 3, point[1] - 3,
                                    point[0] + 3, point[1] + 3], fill=(255, 165, 0, 255))

        # --- Draw trajectory ---
        if self._traj_path and len(self._traj_path) > 1:
            with self._traj_path_lock:
                traj_points_img = self._world_to_img_batch(np.array(self._traj_path))
            traj_points_tuples = [tuple(map(int, p)) for p in traj_points_img]
            draw_dynamic.line(traj_points_tuples, fill=(0, 0, 255, 255), width=4)

        # --- Draw current pose ---
        if self._curr_pose is not None:
            pos_img = self._world_to_img(self._curr_pose[:3, 3])
            draw_dynamic.ellipse([pos_img[0] - 4, pos_img[1] - 4,
                                pos_img[0] + 4, pos_img[1] + 4],
                                fill=(255, 0, 0, 255))

        # --- Combine and finalize ---
        combined_img = Image.alpha_composite(static_img, dynamic_layer)
        self.cached_semantic_map = combined_img.convert("RGB")

        logger.info(f"[semantic] _update_semantic_map_cache: {time.time() - start:.4f} seconds")

    
    def _world_to_img(self, point):
        """
        Converts a world coordinate to an image coordinate, with caching.
        """
        key = tuple(point)
        if key in self._world_to_img_cache:
            return self._world_to_img_cache[key]

        metadata = self._cached_static_metadata
        point_img = ((point[:2] - metadata['min_coords']) / metadata['resolution'] * metadata['scale_factor']).astype(int)
        point_img[0] += metadata['padding'] // 2
        point_img[1] = metadata['height'] - point_img[1] - (metadata['padding'] // 2)
        self._world_to_img_cache[key] = tuple(point_img)
        return tuple(point_img)

    def _world_to_img_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized conversion from world coordinates to image coordinates.
        """
        metadata = self._cached_static_metadata
        min_coords = metadata["min_coords"]
        resolution = metadata["resolution"]
        scale_factor = metadata["scale_factor"]
        padding = metadata["padding"]
        height = metadata["height"]

        points_img = ((points[:, :2] - min_coords) / resolution * scale_factor).astype(int)
        points_img[:, 0] += padding // 2
        points_img[:, 1] = height - points_img[:, 1] - (padding // 2)
        return points_img

    def _update_traversable_map_cache(self):
        """
        Optimized method to update the cached traversable map image from the last inflated map.
        """
        if self.last_inflated_map is None or self.layout_map is None:
            self.cached_traversable_map = None
            return

        # Check if static elements need to be updated
        if self.traversable_map_dirty:
            self.traversable_map_dirty = False
            # Transpose because inflated_map is (x, y) but we want to treat it as (h, w) for image
            grid_to_display = self.last_inflated_map.T
            h, w = grid_to_display.shape

            # Create base image for static elements
            static_image = np.zeros((h, w, 3), dtype=np.uint8)
            static_image[grid_to_display == 0] = [255, 255, 255]  # White for free space
            static_image[grid_to_display == 1] = [100, 100, 100]  # Gray for occupied

            # Flip the image vertically to correct orientation
            static_image = cv2.flip(static_image, 0)

            # Convert to PIL image
            self._cached_static_traversable_image = Image.fromarray(static_image, 'RGB')
            self._cached_traversable_metadata = {
                'origin': np.array([self.layout_map.x_edges[0], self.layout_map.y_edges[0]]),
                'resolution': self.layout_map.resolution,
                'height': h,
                'width': w
            }

        # Create dynamic layer
        dynamic_layer = Image.new('RGBA', self._cached_static_traversable_image.size, (0, 0, 0, 0))
        draw_dynamic = ImageDraw.Draw(dynamic_layer)

        def world_to_grid_img(point):
            """
            Converts a world coordinate to grid image coordinates.
            """
            metadata = self._cached_traversable_metadata
            grid_x = int((point[0] - metadata['origin'][0]) / metadata['resolution'])
            grid_y = int((point[1] - metadata['origin'][1]) / metadata['resolution'])
            return grid_x, metadata['height'] - 1 - grid_y

        # Draw current pose
        if self._curr_pose is not None:
            pos = self._curr_pose[:3, 3]
            rot_matrix = self._curr_pose[:3, :3]
            fwd_vec_world = rot_matrix @ np.array([0, 0, 1])  # ROS forward is +Z
            pos_img = world_to_grid_img(pos)

            arrow_length = 16
            arrow_color = (255, 0, 0, 255)  # Red

            fwd_vec_2d_normalized = fwd_vec_world[:2] / (np.linalg.norm(fwd_vec_world[:2]) + 1e-6)

            # Tip in image coordinates (y direction is flipped)
            tip_x = pos_img[0] + arrow_length * fwd_vec_2d_normalized[0]
            tip_y = pos_img[1] - arrow_length * fwd_vec_2d_normalized[1]

            draw_dynamic.line([pos_img, (tip_x, tip_y)], fill=arrow_color, width=3)
            draw_dynamic.ellipse([pos_img[0]-4, pos_img[1]-4, pos_img[0]+4, pos_img[1]+4], fill=arrow_color)

        # Combine static and dynamic layers
        combined_img = Image.alpha_composite(self._cached_static_traversable_image.convert('RGBA'), dynamic_layer)
        self.cached_traversable_map = combined_img.convert('RGB')

    def mark_semantic_map_dirty(self):
        """Marks the semantic map as dirty, forcing a redraw on next get."""
        self.semantic_map_dirty = True

    def mark_traversable_map_dirty(self):
        """Marks the traversable map as dirty, forcing a redraw on next get."""
        self.traversable_map_dirty = True

    def get_semantic_map_image(self) -> None | np.ndarray:
        """
        Returns the latest cached semantic map image.
        The static map elements with dynamic elements are updated in the background thread.
        """
        if not self.has_global_map() or self.cached_semantic_map is None:
            return None

        start = time.time()
        # The image with all elements (static + dynamic) is already cached
        result = cv2.cvtColor(np.array(self.cached_semantic_map), cv2.COLOR_RGB2BGR)  # Convert PIL (RGB) image to numpy array (BGR) for OpenCV
        logger.debug(f"[semantic] get_semantic_map_image: {time.time() - start:.4f} seconds")
        return result

    def get_traversable_map_image(self) -> None | np.ndarray:
        """
        Returns the latest cached traversable map image.
        The static map elements with dynamic elements are updated in the background thread.
        """
        if self.cached_traversable_map is None:
            return None

        start = time.time()
        # The image with all elements (static + dynamic) is already cached
        result = cv2.cvtColor(np.array(self.cached_traversable_map), cv2.COLOR_RGB2BGR)  # Convert PIL (RGB) image to numpy array (BGR) for OpenCV
        logger.debug(f"[traversable] get_traversable_map_image: {time.time() - start:.4f} seconds")
        return result

    def update_pose_path(self, curr_pose=None, nav_path=None):
        """
        Updates the dynamic parameters that will be used by the background cache update functions.
        TODO: when to mark_traversable_map_dirty() and mark_semantic_map_dirty()
        """
        if curr_pose is not None:
            self._curr_pose = curr_pose
            self.mark_traversable_map_dirty()
            with self._traj_path_lock:  # Use the lock to ensure thread-safe access
                self._traj_path.append(curr_pose[:3, 3])
            self.mark_semantic_map_dirty()
        if nav_path is not None:
            self._nav_path = nav_path
            self.mark_semantic_map_dirty()

    # ===============================================
    # Rerun Visulization
    # ===============================================
    def visualize_global_map(
        self
    ) -> None:
        new_logged_entities = set()

        path_radii = self.cfg.path_radii

        if self.preload_path_ok is False and self.cfg.use_given_path:
            json_data = self.read_json_files(self.cfg.given_path_dir)
            # traverse all the json data
            for key, value in json_data.items():
                idx = key.split('.')[0]
                path_points = np.array(value)

                path_color = self.cfg.global_path_color

                # if idx == '3':
                #     path_color = (169, 220, 169)
                #     path_radii = 0.02

                # if idx == '5':
                #     path_color = self.cfg.action_path_color

                preload_path_entity = f"world/preload_path/{idx}"

                # Log the navigation path as a line strip (connecting consecutive points)
                self.visualizer.log(
                    preload_path_entity,
                    self.visualizer.LineStrips3D(
                        [path_points.tolist()],  # Convert the list of points to the required format
                        colors=[path_color],  # Green color for the path
                        radii=[path_radii]
                    )
                )
                # new_logged_entities.add(global_path_entity)
            
            self.preload_path_ok = True

        for global_obj in self.global_map:
            base_entity_path = "global/objects"

            obj_name = self.obj_classes.get_classes_arr()[global_obj.class_id]
            positions = np.asarray(global_obj.pcd_2d.points)
            colors = np.asarray(global_obj.pcd_2d.colors) * 255
            colors = colors.astype(np.uint8)
            curr_obj_color = self.obj_classes.get_class_color(obj_name)

            if global_obj.nav_goal:
                # set red
                curr_obj_color = (255, 0, 0)

            if self._nav_path:
                if global_obj.nav_goal:
                    # set red
                    curr_obj_color = (255, 0, 0)
                else:
                    curr_obj_color = (169, 169, 169)

            # Get current related num
            related_num = len(global_obj.related_objs)

            # log pcd data
            rgb_pcd_entity = base_entity_path + "/rgb_pcd" + f"/{global_obj.uid}"
            self.visualizer.log(
                rgb_pcd_entity,
                # entity_path + "/pcd",
                self.visualizer.Points3D(
                    positions,
                    colors=[curr_obj_color],
                    # labels=[obj_label],
                ),
                self.visualizer.AnyValues(
                    uuid=str(global_obj.uid),
                )
            )

            bbox_2d = global_obj.bbox_2d
            centers = [bbox_2d.get_center()]
            half_sizes = [bbox_2d.get_extent() / 2]

            bbox_entity = base_entity_path + "/bbox" + f"/{global_obj.uid}"

            self.visualizer.log(
                bbox_entity,
                # entity_path + "/bbox",
                self.visualizer.Boxes3D(
                    centers=centers,
                    half_sizes=half_sizes,
                    labels=[f"{obj_name}"],
                    colors=[curr_obj_color],
                ),
                self.visualizer.AnyValues(
                    uuid=str(global_obj.uid),
                    related_num=related_num,
                )
            )

            # log related bbox
            s = 0.04
            fixed_half_size = np.array([s, s, s])
            for i, related_bbox in enumerate(global_obj.related_bbox):
                center = related_bbox.get_center()

                center_changed = [center[0], center[1], self.cfg.related_height]

                # centers = [related_bbox.get_center()]
                centers = [center_changed]

                half_sizes = [fixed_half_size]

                # get color
                class_id = global_obj.related_color[i]
                obj_name = self.obj_classes.get_classes_arr()[class_id]
                obj_color = self.obj_classes.get_class_color(obj_name)

                related_bbox_entity = base_entity_path + "/related_bbox" + f"/{global_obj.uid}_{i}"
                self.visualizer.log(
                    related_bbox_entity,
                    # entity_path + "/bbox",
                    self.visualizer.Boxes3D(
                        centers=centers,
                        half_sizes=half_sizes,
                        colors=[obj_color],
                        fill_mode="solid",
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(global_obj.uid)
                    )
                )

                center_tilted = [center[0], center[1], self.cfg.related_height + 0.1]  # Increase Z axis value
                s = 0.01
                title_half_size = np.array([s, s, s])

                related_title_entity = base_entity_path + "/related_title" + f"/{global_obj.uid}_{i}"
                self.visualizer.log(
                    related_title_entity,
                    # entity_path + "/bbox",
                    self.visualizer.Boxes3D(
                        centers=[center_tilted],
                        half_sizes=title_half_size,
                        colors=[obj_color],
                        labels=[f"{obj_name}"],
                        fill_mode="solid",
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(global_obj.uid)
                    )
                )

                related_line_entity = base_entity_path + "/related_line" + f"/{global_obj.uid}_{i}"

                self.visualizer.log(
                    related_line_entity,
                    # entity_path + "/bbox",
                    self.visualizer.LineStrips3D(
                        [
                            [
                                [center[0], center[1], self.cfg.floor_height],
                                [center[0], center[1], self.cfg.related_height],
                            ],
                        ],
                        colors=[obj_color],
                        radii=[0.01],
                    )
                )

                new_logged_entities.add(related_bbox_entity)
                new_logged_entities.add(related_line_entity)
                new_logged_entities.add(related_title_entity)

            if self.cfg.show_global_map_3d_bbox:
                bbox_3d = global_obj.pcd.get_axis_aligned_bounding_box()
                centers = [bbox_3d.get_center()]
                half_sizes = [bbox_3d.get_extent() / 2]

                bbox_3d_entity = base_entity_path + "/bbox_3d" + f"/{global_obj.uid}"

                self.visualizer.log(
                    bbox_3d_entity,
                    # entity_path + "/bbox",
                    self.visualizer.Boxes3D(
                        centers=centers,
                        half_sizes=half_sizes,
                        colors=[curr_obj_color],
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(global_obj.uid)
                    )
                )
                new_logged_entities.add(bbox_3d_entity)

            new_logged_entities.add(rgb_pcd_entity)
            new_logged_entities.add(bbox_entity)

        # Draw the global path if available
        global_path_entity = "world/global_path"
        if self._nav_path:
            # Create a list of 3D points from the pos_path
            path_points = np.array(self._nav_path)

            path_color = self.cfg.global_path_color

            if self.has_action_path:
                path_color = (169, 220, 169)
                path_radii = 0.02

            # Log the navigation path as a line strip (connecting consecutive points)
            self.visualizer.log(
                global_path_entity,
                self.visualizer.LineStrips3D(
                    [path_points.tolist()],  # Convert the list of points to the required format
                    colors=[path_color],  # Green color for the path
                    radii=[path_radii]
                )
            )
            new_logged_entities.add(global_path_entity)

        action_path_entity = "world/action_path"
        if self.action_path is not None:
            # Create a list of 3D points from the pos_path
            path_points = np.array(self.action_path)

            # Log the navigation path as a line strip (connecting consecutive points)
            self.visualizer.log(
                action_path_entity,
                self.visualizer.LineStrips3D(
                    [path_points.tolist()],  # Convert the list of points to the required format
                    colors=[self.cfg.action_path_color],
                    radii=[path_radii]
                )
            )
            new_logged_entities.add(action_path_entity)

        if len(self.prev_entities) != 0:
            for entity_path in self.prev_entities:
                if entity_path not in new_logged_entities:
                    # logger.info(f"Clearing {entity_path}")
                    self.visualizer.log(
                        entity_path,
                        self.visualizer.Clear(recursive=True)
                    )
        self.prev_entities = new_logged_entities

