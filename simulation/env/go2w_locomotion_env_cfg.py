import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp
import simulation.mdp as user_mdp


##
# Pre-defined configs
##
from simulation.assets.robots.unitree import UNITREE_GO2W_CFG
from simulation.assets.terrains.terrain_cfg import FLAT_TERRAINS_CFG
from simulation.utils import compute_cam_cfg


GO2W_LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

GO2W_WHEEL_JOINT_NAMES = [
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
]

GO2W_JOINT_NAMES = GO2W_LEG_JOINT_NAMES + GO2W_WHEEL_JOINT_NAMES


@configclass
class VelocitySceneCfg(InteractiveSceneCfg):
    # Terrain
    terrain = None  # to be set outside

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=10000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=10000.0),
    )
    cylinder_light = AssetBaseCfg(
        prim_path="/World/cylinderLight",
        spawn=sim_utils.CylinderLightCfg(
            length=100, radius=0.3, treat_as_line=False, intensity=10000.0
        ),
    )
    cylinder_light.init_state.pos = (0, 0, 2.0)

    # Robots
    robot: ArticulationCfg = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Sensors
    rgbd_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/rgbd_camera",
        update_period=0.06, # ~15Hz, close to realsense, > self.sim.render_interval
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=compute_cam_cfg(W=640, H=480, fov_deg_x=90.0),
        # spawn=sim_utils.PinholeCameraCfg(
        #     focal_length=24.0, focus_distance=400.0, horizontal_aperture=54.0, clipping_range=(0.1, 1.0e5)
        # ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.2), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
    )
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=[0.0, 0.0, 20.0]),
    #     ray_alignment='yaw',
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    #     max_distance=100.0,
    # )
    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*",
    #     update_period=0.0,
    #     history_length=3,
    #     debug_vis=True,
    #     track_air_time=True
    # )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=GO2W_LEG_JOINT_NAMES,
        scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=GO2W_WHEEL_JOINT_NAMES,
        scale=5.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
            # noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
       )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
            # noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=(2.0, 2.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=user_mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg(name="robot", joint_names=GO2W_WHEEL_JOINT_NAMES),
            },
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True)},
            # noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


@configclass
class LocomotionVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment on Go2W."""

    # Scene settings
    scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # dummy settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        # Viewer settings
        self.viewer.eye = (-2, 0.0, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        # self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        # self.sim.disable_contact_processing = True

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.sim.dt * self.decimation

        self.is_finite_horizon = False