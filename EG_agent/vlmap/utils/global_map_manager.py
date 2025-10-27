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
from EG_agent.vlmap.utils.navigation_helper import NavigationGraph, LayoutMap

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

        # For navigation --> NavigationGraph
        self.nav_graph: NavigationGraph = None
        self.inquiry = ''
        self.action_path = []
        self.has_action_path = False

        # layout information --> LayoutMap
        layout_resolution = self.cfg.layout_voxel_size * 2
        self.layout_map = LayoutMap(cfg, resolution=layout_resolution, percentile=90, min_area=5, kernel_size=3)

        # pass to the local map manager for inquiry
        self.global_candidate_bbox = None
        self.global_candidate_score = 0.0
        # for lost and found
        # save the previous queried global obj uids
        self.ignore_global_obj_list = []

        self.best_candidate_name = None
        
        self.preload_path_ok = False
        self.vis_preload_wall_ok = False

        # Caching for the semantic map
        self.cached_semantic_map = None
        self.semantic_map_dirty = True
        self.semantic_map_metadata = {}

        # Caching for the traversable map
        self.cached_traversable_map = None
        self.traversable_map_dirty = True
        self.traversable_map_metadata = {}

        # Background thread for updating maps
        self._map_update_thread = None
        self._stop_map_update = threading.Event()
        # self._map_update_lock = threading.Lock()
        self._last_update_time = 0
        self._update_interval = 2.0  # Update every 2 seconds (low frequency)

        # Start background thread
        self._start_background_update_thread()

        # Object classes
        # --- 加载指定的 要识别的 全部物体的 类别text ---
        classes_path = cfg.yolo.classes_path
        if cfg.yolo.use_given_classes:
            classes_path = cfg.yolo.given_classes_path
            logger.info(f"[Detector][Init] Using given classes, path:{classes_path}")

        # Object classes
        self.obj_classes = ObjectClasses(
            classes_file_path=classes_path,
            bg_classes=self.cfg.yolo.bg_classes,
            skip_bg=self.cfg.yolo.skip_bg)

        # Store dynamic parameters for background updates
        self._curr_pose: np.ndarray = None
        self._nav_path: list = []
        self._traj_path = deque(maxlen=60)

        self.layout_initialized = False

    def append_traj(self, pose: np.ndarray) -> None:
        self._traj_path.append(pose)

    def has_global_map(self) -> bool:
        return len(self.global_map) > 0

    def set_layout_info(self, layout_pcd, force_full_update=False, current_pose=None, update_radius=0):
        if force_full_update or not self.layout_initialized:
            # First time or forced: process the whole layout
            self.layout_map.set_layout_pcd(layout_pcd)
            self.layout_map.extract_wall_pcd(num_samples_per_grid=10, z_value=self.cfg.floor_height)
            self.layout_initialized = True
            logger.info("[GlobalMapManager] Initialized/updated layout map with full point cloud.")
        else:
            # Subsequent live updates: process only a local part
            if current_pose is not None and update_radius > 0:
                self.layout_map.update_local_layout_occmap_wallpcd(layout_pcd, current_pose, update_radius)
                logger.info("[GlobalMapManager] Updated layout map with partial point cloud.")
            else:
                logger.warning("[GlobalMapManager] Skipping live layout update because current_pose is not provided or update_radius is 0.")

    def save_wall_pcd(self):
        self.layout_map.save_wall_pcd()

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
                logger.info(f"[GlobalMap] Saving No.{i} obj: {obj.save_path}")
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

    def load_wall(self) -> None:
        if os.path.exists(self.cfg.preload_path):
            load_dir = self.cfg.preload_path
            logger.info(f"[GlobalMap] Using preload wall path: {load_dir}")
        else:
            load_dir = self.cfg.map_save_path
            logger.info(f"[GlobalMap] Preload wall path not found. Using default map save path: {load_dir}")

        wall_pcd_path = os.path.join(load_dir, "wall.pcd")

        if not Path(wall_pcd_path).is_file():
            logger.warning(f"[GlobalMap] wall file not found at: {wall_pcd_path}")
            return None

        # load wall pcd
        wall_pcd = o3d.io.read_point_cloud(wall_pcd_path)
        # Convert to numpy array
        points = np.asarray(wall_pcd.points)

        # Set all z values to 22
        points[:, 2] = self.cfg.floor_height

        # Assign the modified points back to the point cloud
        wall_pcd.points = o3d.utility.Vector3dVector(points)
        logger.info(f"[GlobalMap] Wall loaded from: {wall_pcd_path}")

        # Save to class attribute
        self.layout_map.wall_pcd = wall_pcd
    
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

    def visualize_global_map(
        self
    ) -> None:
        new_logged_entities = set()

        path_radii = self.cfg.path_radii

        # show the preload wall pcd
        if self.vis_preload_wall_ok is False:
            if self.layout_map.wall_pcd is not None:
                self.visualizer.log(
                    "world/preload_wall",
                    self.visualizer.Points3D(
                        np.asarray(self.layout_map.wall_pcd.points),
                        colors=[(169, 169, 169)],
                        radii = [0.01]
                    )
                )
                self.vis_preload_wall_ok = True


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

            obj_name = self.visualizer.obj_classes.get_classes_arr()[global_obj.class_id]
            positions = np.asarray(global_obj.pcd_2d.points)
            colors = np.asarray(global_obj.pcd_2d.colors) * 255
            colors = colors.astype(np.uint8)
            curr_obj_color = self.visualizer.obj_classes.get_class_color(obj_name)

            if global_obj.nav_goal:
                # set red
                curr_obj_color = (255, 0, 0)

            if self.nav_graph is not None and self.nav_graph.pos_path is not None:
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
                obj_name = self.visualizer.obj_classes.get_classes_arr()[class_id]
                obj_color = self.visualizer.obj_classes.get_class_color(obj_name)

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
        if self.nav_graph is not None and self.nav_graph.pos_path is not None:
            # Create a list of 3D points from the pos_path
            path_points = np.array(self.nav_graph.pos_path)

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

        pass

    def create_nav_graph(self, curr_pose, resolution=0.03) -> None:
        """
            Generates the NavigationGraph based on the current global map.
            (A computationally intensive operation)
        """
        logger.info("[GlobalMapManager] [create_nav_graph] Creating navigation graph...")

        # # 1. Get all objects' pcd from global map
        # total_pcd = o3d.geometry.PointCloud()
        # for obj in self.global_map:
        #     total_pcd += obj.pcd_2d
        #
        # # 2. Add current pose into the total pcd
        # curr_point_coords = curr_pose[:3, 3]
        # curr_point = o3d.geometry.PointCloud()
        # curr_point.points = o3d.utility.Vector3dVector([curr_point_coords])
        # total_pcd += curr_point
        #
        # # 3. Add layout wall to the total pcd
        # total_pcd += self.layout_map.wall_pcd
        total_pcd = self.layout_map.wall_pcd

        if not total_pcd.has_points():
            logger.warning("[GlobalMapManager] [create_nav_graph] No points in map to create navigation graph.")
            self.nav_graph = None
            return

        # 4. Construct NavigationGraph and 2D occupancy map
        try:
            nav_graph = NavigationGraph(self.cfg, total_pcd, cell_size=resolution)
            self.nav_graph = nav_graph

            nav_graph.get_graph()
        except Exception as e:
            logger.error(f"[GlobalMapManager] [create_nav_graph] Failed to create navigation graph: {e}")
            self.nav_graph = None

    def calculate_global_path(
        self, curr_pose, goal_mode=GoalMode.POSE, resolution=0.03, goal_position=None
    ) -> List:
        # Step 1: Construct NavigationGraph and 2D occupancy map
        self.create_nav_graph(curr_pose, resolution)
        if self.nav_graph is None:
            logger.warning("[LocalMapManager] [calculate_local_path] Navigation graph not available. Skipping local_path calculation.")
            return []
        nav_graph = self.nav_graph

        # Step 2: Set Start point and goal point
        # transform curr pose to 2d coordinate
        curr_position = curr_pose[:3, 3]
        start_position_grid = nav_graph.calculate_pos_2d(curr_position)

        # Select and process goal based on mode
        goal_position_grid = self.get_goal_position(nav_graph, start_position_grid, goal_position, goal_mode)

        # Step 3: Find shortest path
        if goal_position_grid is not None:
            path = nav_graph.find_shortest_path(start_position_grid, goal_position_grid)
            if path:
                logger.info("[GlobalMap][Path] Path successfully generated.")
                return nav_graph.pos_path
            else:
                logger.info("[GlobalMap][Path] Failed to generate a valid path.")
                return []
        else:
            logger.info("[GlobalMap][Path] No valid goal position provided.")
            return []

    def get_goal_position(self, nav_graph, start_position_grid, goal_position_world, goal_mode):
        """
        Get the goal position based on the specified mode.

        Parameters:
        - nav_graph: An instance of NavigationGraph.
        - start_position: The starting position in the graph.
        - goal_mode: The mode to select the goal.

        Returns:
        - goal_position: The selected goal position.
        """
        if goal_mode == GoalMode.RANDOM:
            # logger.info("[GlobalMap][Path] Goal mode: RANDOM")
            return nav_graph.sample_random_point()

        if goal_mode == GoalMode.CLICK:
            # logger.info("[GlobalMap][Path] Goal mode: CLICK")
            return nav_graph.visualize_start_and_select_goal(start_position_grid)

        if goal_mode == GoalMode.POSE:
            # logger.info("[GlobalMap][Path] Goal mode: POSE")
            if not goal_position_world:
                return None
            return nav_graph.calculate_pos_2d(goal_position_world)

        if goal_mode == GoalMode.INQUIRY:
            # logger.info("[GlobalMap][Path] Goal mode: INQUIRY")
            # Step 1, from gloabl map find the best candidate
            global_goal_candidate, score = self.find_best_candidate_with_inquiry()

            # --- FIX: Check if a candidate was actually found ---
            if global_goal_candidate is None:
                logger.error("[GlobalMap][Path] Could not find a suitable goal candidate for the inquiry.")
                return None
            # --- END FIX ---

            # Get center of the best candidate
            goal_3d = global_goal_candidate.bbox_2d.get_center()

            # pass to the local map manager
            self.global_candidate_bbox = global_goal_candidate.bbox_2d
            if self.global_candidate_score == 0.0:
                self.global_candidate_score = score

            # save the queried obj into ignore list
            self.ignore_global_obj_list.append(global_goal_candidate.uid)

            # step 2, using navgraph to switch best candidate into specific goal point
            goal_2d = nav_graph.calculate_pos_2d(goal_3d)
            # check if is in free space, actually the goal is not
            if not nav_graph.free_space_check(goal_2d) is False:
                snapped_goal = nav_graph.snap_to_free_space_directional(goal_2d, start_position_grid, nav_graph.free_space)
                # nearest_node = nav_graph.find_nearest_node(goal_2d)

                if self.cfg.use_directional_path:
                    nearest_node = nav_graph.find_nearest_node(snapped_goal, start_position_grid)
                else:
                    nearest_node = nav_graph.find_nearest_node(goal_2d)

                goal_2d = np.array(nearest_node)

            logger.info(f"[GlobalMap][Path] Nearest node: {goal_2d}")

            return goal_2d

        logger.warning(f"[GlobalMap][Path] Invalid goal mode: {goal_mode}")
        return None

    def get_random_walkable_goal(self):
        """
        Samples a random walkable goal from the navigation graph's free space.

        Returns:
            np.ndarray: A 3D point representing a random goal, or None if not possible.
        """
        if self.nav_graph is None or self.nav_graph.graph is None:
            logger.warning("[GlobalMapManager] Navigation graph not created. Cannot sample a random goal.")
            return None

        random_grid_point = self.nav_graph.sample_random_point()
        if random_grid_point is None:
            logger.warning("[GlobalMapManager] Failed to sample a random point from the navigation graph.")
            return None

        # Convert grid coordinates back to world coordinates
        # TODO: Check if this is correct, 是否可以使用 self.nav_graph.calculate_pos_3d()
        world_pos_2d = random_grid_point * self.nav_graph.cell_size + self.nav_graph.pcd_min
        world_pos_3d = np.append(world_pos_2d, self.cfg.floor_height)

        logger.info(f"[GlobalMapManager] Sampled random walkable goal at {world_pos_3d}")
        return world_pos_3d

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
            obj_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]
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
                obj_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]
                if obj_name == self.best_candidate_name:
                    obj_list.append(obj)
        
        if len(obj_list) != 0:
            best_candidate = obj_list[0]

        # Output the best candidate and its similarity
        best_candidate_name = self.visualizer.obj_classes.get_classes_arr()[best_candidate.class_id]

        logger.debug(f"[GlobalMap][Inquiry] We ignore {len(self.ignore_global_obj_list)} objects in this global query.")
        logger.debug(f"[GlobalMap][Inquiry] Best Candidate: '{best_candidate_name}' with similarity: {best_similarity:.3f}")

        if self.best_candidate_name is None:
            self.best_candidate_name = best_candidate_name

        logger.debug(f"[GlobalMap][Inquiry] Memory Best Candidate '{self.best_candidate_name}'")

        # Set flag to the best candidate for visualization
        best_candidate.nav_goal = True

        # input("Press any key to continue...")

        return best_candidate, best_similarity

    def _start_background_update_thread(self):
        """Start the background thread for updating maps."""
        self._map_update_thread = threading.Thread(target=self._background_map_update_worker, daemon=True)
        self._map_update_thread.start()

    def _background_map_update_worker(self):
        """Background thread worker that updates both semantic and traversable maps at low frequency."""
        while not self._stop_map_update.is_set():
            try:
                # Check if an update is needed based on dirty flags or time interval
                current_time = time.time()
                time_ok = current_time - self._last_update_time >= self._update_interval
                sem_needs_update = self.semantic_map_dirty and time_ok
                tra_needs_update = self.traversable_map_dirty and time_ok

                if sem_needs_update:
                    # with self._map_update_lock:
                    self._last_update_time = current_time
                    # Update semantic map if dirty
                    if self.semantic_map_dirty and self.has_global_map():
                        self._update_semantic_map_cache(
                            resolution=0.03,  # default resolution
                        )
                if tra_needs_update:
                    self._last_update_time = current_time
                    # with self._map_update_lock:
                    # Update traversable map if dirty
                    if self.traversable_map_dirty and self.nav_graph:
                        self._update_traversable_map_cache()
                
                # Sleep for a short time to prevent busy waiting
                self._stop_map_update.wait(timeout=0.5)

            except Exception as e:
                logger.error(f"[GlobalMapManager] Error in background map update thread: {e}")
                # Sleep before retrying to avoid rapid error loops
                self._stop_map_update.wait(timeout=1)

    def _update_semantic_map_cache(self, resolution=0.03):
        """
        Updates the cached semantic map image with static elements.
        """
        if not self.has_global_map():
            self.cached_semantic_map = None
            return

        # 1. Aggregate all points to determine map boundaries
        all_points_lists = []
        for obj in self.global_map:
            if not obj.pcd_2d.is_empty():
                all_points_lists.append(np.asarray(obj.pcd_2d.points))

        wall_pcd = self.layout_map.wall_pcd if self.layout_map else None
        if wall_pcd and not wall_pcd.is_empty():
            all_points_lists.append(np.asarray(wall_pcd.points))

        if not all_points_lists:
            self.cached_semantic_map = None
            return

        all_points = np.vstack(all_points_lists)
        all_points = all_points[np.isfinite(all_points).all(axis=1)]
        if all_points.size == 0:
            self.cached_semantic_map = None
            return

        # 2. Determine map dimensions and create metadata
        min_coords = np.min(all_points[:, :2], axis=0)
        max_coords = np.max(all_points[:, :2], axis=0)
        map_size = max_coords - min_coords
        scale_factor = 2.0
        padding = 100
        width = int((map_size[0]) / resolution * scale_factor) + padding
        height = int((map_size[1]) / resolution * scale_factor) + padding

        metadata = {
            'min_coords': min_coords, 'resolution': resolution, 'scale_factor': scale_factor,
            'padding': padding, 'width': width, 'height': height
        }

        # 3. Create base image and draw static elements
        pil_img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=int(12 * (scale_factor / 2)))
        except IOError:
            font = ImageFont.load_default()

        def world_to_img(point):
            point_img = ((point[:2] - metadata['min_coords']) / metadata['resolution'] * metadata['scale_factor']).astype(int)
            point_img[0] += metadata['padding'] // 2
            point_img[1] = metadata['height'] - point_img[1] - (metadata['padding'] // 2)
            return tuple(point_img)

        placed_label_boxes = []

        # Draw walls if available
        if wall_pcd and not wall_pcd.is_empty():
            wall_points_data = np.asarray(wall_pcd.points)
            wall_points_img = np.array([world_to_img(p) for p in wall_points_data if np.isfinite(p).all()])
            radius = int(1 * (scale_factor / 2))
            for p_img in wall_points_img:
                draw.ellipse([p_img[0]-radius, p_img[1]-radius, p_img[0]+radius, p_img[1]+radius], fill=(160, 160, 160))

        # Draw objects
        for obj in self.global_map:
            if obj.pcd_2d.is_empty():
                continue

            points = np.asarray(obj.pcd_2d.points)
            if points.shape[0] == 0:
                continue

            obj_name = self.obj_classes.get_classes_arr()[obj.class_id]
            color_rgb_int = tuple(int(c * 255) for c in self.obj_classes.get_class_color(obj_name))

            points_img = np.array([world_to_img(p) for p in points])
            radius = int(1 * (scale_factor / 2))
            for p_img in points_img:
                draw.ellipse([p_img[0]-radius, p_img[1]-radius, p_img[0]+radius, p_img[1]+radius], fill=color_rgb_int)
            # Draw class text (with collision detection)
            obj_x_min, obj_y_min = np.min(points_img, axis=0)
            obj_x_max, obj_y_max = np.max(points_img, axis=0)
            centroid_img = np.mean(points_img, axis=0).astype(int)

            text_bbox = draw.textbbox((0, 0), obj_name, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

            candidates = [
                (centroid_img[0] - text_w // 2, centroid_img[1] - text_h // 2 - 5),
                (obj_x_min + (obj_x_max - obj_x_min) // 2 - text_w // 2, obj_y_min - text_h),
                (obj_x_max + 1, obj_y_min + text_h // 2),
                (obj_x_min - text_w - 1, obj_y_min + text_h // 2),
                (obj_x_max + 1, obj_y_max),
                (obj_x_min - text_w - 1, obj_y_max),
            ]
            # Find a non-colliding position
            final_pos = None
            for pos in candidates:
                lx, ly = pos
                label_box = (lx, ly, lx + text_w, ly + text_h)
                if not (label_box[0] >= 0 and label_box[1] >= 0 and label_box[2] < width and label_box[3] < height):
                    continue
                is_colliding = False
                for placed_box in placed_label_boxes:
                    if not (label_box[2] < placed_box[0] or label_box[0] > placed_box[2] or label_box[3] < placed_box[1] or label_box[1] > placed_box[3]):
                        is_colliding = True
                        break
                if not is_colliding:
                    final_pos = pos
                    placed_label_boxes.append(label_box)
                    break
            if final_pos is None:
                final_pos = candidates[0]
            draw.text(final_pos, obj_name, font=font, fill=(0, 0, 0))

        # 4. Draw dynamic elements: nav path, trajectory, and current pose
        def world_to_img_cached(point):
            point_img = ((point[:2] - metadata['min_coords']) / metadata['resolution'] * metadata['scale_factor']).astype(int)
            point_img[0] += metadata['padding'] // 2
            point_img[1] = metadata['height'] - point_img[1] - (metadata['padding'] // 2)
            return tuple(point_img)

        try:
            coord_font = ImageFont.truetype("DejaVuSans.ttf", size=int(10 * (scale_factor / 3)))
        except IOError:
            coord_font = ImageFont.load_default()

        # (1) Draw navigation path
        if self._nav_path and len(self._nav_path) > 1:
            path_points_img = [world_to_img_cached(np.array(p)) for p in self._nav_path if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2]
            if len(path_points_img) > 1:
                draw.line(path_points_img, fill=(0, 255, 0), width=6)  # Green line
                for point in path_points_img:
                    draw.ellipse([point[0]-3, point[1]-3, point[0]+3, point[1]+3], fill=(255, 0, 0))

        # (2) Draw trajectory
        if self._traj_path and len(self._traj_path) > 1:
            traj_points_img = [world_to_img_cached(np.array(p)) for p in self._traj_path if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2]
            if len(traj_points_img) > 1:
                draw.line(traj_points_img, fill=(0, 0, 255), width=4)  # Blue line

        # (3) Draw current pose
        if self._curr_pose is not None:
            pos = self._curr_pose[:3, 3]
            rot_matrix = self._curr_pose[:3, :3]
            fwd_vec_world = rot_matrix @ np.array([0, 0, 1]) # ROS forward is +Z
            pos_img = world_to_img_cached(pos)

            arrow_length = 16 * (scale_factor / 3)
            arrow_color = (255, 0, 0)  # Red
            fwd_vec_2d_normalized = fwd_vec_world[:2] / (np.linalg.norm(fwd_vec_world[:2]) + 1e-6)
            tip_x = pos_img[0] + arrow_length * fwd_vec_2d_normalized[0]
            tip_y = pos_img[1] - arrow_length * fwd_vec_2d_normalized[1]  # Subtract because Y is flipped

            # Define triangle points for the arrow
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
            draw.text((pos_img[0] + 15, pos_img[1]), coord_text, font=coord_font, fill=(0, 0, 0))

        # 5. Update the cached image and metadata
        self.cached_semantic_map = pil_img
        self.semantic_map_metadata = metadata
        self.semantic_map_dirty = False

    def _update_traversable_map_cache(self):
        """
        Updates the cached traversable map image.
        """
        if not self.nav_graph:
            return

        free_grid = self.nav_graph.free_space
        if free_grid is None:
            return

        h, w = free_grid.shape

        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[free_grid == 1] = [255, 255, 255]  # White for free space
        image[free_grid == 0] = [100, 100, 100]  # Gray for occupied

        # Flip the image vertically to correct orientation
        image = cv2.flip(image, 0)

        # PIL expects RGB
        pil_img = Image.fromarray(image, 'RGB')
        draw = ImageDraw.Draw(pil_img)

        metadata = {'origin': self.nav_graph.pcd_min,
                    'resolution': self.nav_graph.cell_size,
                    'height': h, 'width': w}

        def world_to_grid_img(point):
            # Transform world point to grid indices
            grid_x = int((point[0] - metadata['origin'][0]) / metadata['resolution'])
            grid_y = int((point[1] - metadata['origin'][1]) / metadata['resolution'])
            # Transform grid indices to image coordinates (y is flipped)
            return grid_x, metadata['height'] - 1 - grid_y

        if self._curr_pose is not None:
            pos = self._curr_pose[:3, 3]
            rot_matrix = self._curr_pose[:3, :3]
            fwd_vec_world = rot_matrix @ np.array([0, 0, 1]) # ROS forward is +Z
            pos_img = world_to_grid_img(pos)

            arrow_length = 16
            arrow_color = (255, 0, 0) # Red

            fwd_vec_2d_normalized = fwd_vec_world[:2] / (np.linalg.norm(fwd_vec_world[:2]) + 1e-6)

            # Tip in image coordinates (y direction is flipped)
            tip_x = pos_img[0] + arrow_length * fwd_vec_2d_normalized[0]
            tip_y = pos_img[1] - arrow_length * fwd_vec_2d_normalized[1]

            draw.line([pos_img, (tip_x, tip_y)], fill=arrow_color, width=3)
            draw.ellipse([pos_img[0]-4, pos_img[1]-4, pos_img[0]+4, pos_img[1]+4], fill=arrow_color)

        self.cached_traversable_map = pil_img
        self.traversable_map_metadata = metadata
        self.traversable_map_dirty = False

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
        logger.info(f"[semantic] get_semantic_map_image: {time.time() - start:.4f} seconds")
        return result

    def get_traversable_map_image(self) -> None | np.ndarray:
        """
        Returns the latest cached traversable map image.
        The static map elements with dynamic elements are updated in the background thread.
        """
        if not self.nav_graph or self.cached_traversable_map is None:
            return None

        start = time.time()
        # The image with all elements (static + dynamic) is already cached
        result = cv2.cvtColor(np.array(self.cached_traversable_map), cv2.COLOR_RGB2BGR)  # Convert PIL (RGB) image to numpy array (BGR) for OpenCV
        logger.info(f"[traversable] get_traversable_map_image: {time.time() - start:.4f} seconds")
        return result

    def update_pose_path(self, curr_pose=None, nav_path=None):
        """
        Updates the dynamic parameters that will be used by the background cache update functions.
        TODO: when to mark_traversable_map_dirty() and mark_semantic_map_dirty()
        """
        if curr_pose is not None:
            self._curr_pose = curr_pose
            self.mark_traversable_map_dirty()
            self._traj_path.append(curr_pose[:3, 3])
            # self.mark_semantic_map_dirty()
        if nav_path is not None:
            self._nav_path = nav_path
            self.mark_semantic_map_dirty()

    def shutdown_semantic(self):
        """Safely shuts down the visualizer and its background thread."""
        logger.info("[Visualizer] Shutting down semantic map background thread.")

        # Stop the background thread
        if self._map_update_thread:
            self._stop_map_update.set()
            self._map_update_thread.join(timeout=2)  # Wait up to 2 seconds for thread to finish
            if self._map_update_thread.is_alive():
                logger.warning("[Visualizer] Background map update thread did not terminate gracefully.")
            else:
                logger.info("[Visualizer] Background map update thread terminated successfully.")
