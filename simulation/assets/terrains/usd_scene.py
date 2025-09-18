from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import UsdFileCfg

from ...assets import SIMULATION_DATA_DIR


ASSET_CARLA = AssetBaseCfg(
    prim_path=f"/World/Terrain",
    spawn=UsdFileCfg(
        visible=True,
        usd_path=f"{SIMULATION_DATA_DIR}/usd/carla/carla.usd",
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(-200.0, -125.0, 0.0)),
)

ASSET_WAREHOUSE = AssetBaseCfg(
    prim_path=f"/World/Terrain",
    spawn=UsdFileCfg(
        visible=True,
        usd_path=f"{SIMULATION_DATA_DIR}/usd/warehouse/warehouse_new.usd",
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)