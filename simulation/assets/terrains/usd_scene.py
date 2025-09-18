from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import UsdFileCfg

from ...assets import SIMULATION_DATA_DIR


ASSET_DICT = {
    "carla": AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=UsdFileCfg(
            visible=True,
            usd_path=f"{SIMULATION_DATA_DIR}/terrains/carla/carla.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-200.0, -125.0, 0.0)),
    ),
    "warehouse": AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=UsdFileCfg(
            visible=True,
            usd_path=f"{SIMULATION_DATA_DIR}/terrains/warehouse/warehouse_new.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    ),
    "lobby": AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=UsdFileCfg(
            visible=True,
            usd_path=f"{SIMULATION_DATA_DIR}/terrains/lobby/World_Lobby.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-20, -18, -1.0)),
    ),  # 
    "white_home_01": AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=UsdFileCfg(
            visible=True,
            usd_path=f"{SIMULATION_DATA_DIR}/terrains/Storage/WhiteHome01.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
    ),
}