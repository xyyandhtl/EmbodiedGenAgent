from pxr import Usd, UsdGeom, UsdPhysics
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import UsdFileCfg

from ...assets import SIMULATION_DATA_DIR


def add_collision_and_material(prim, static_friction=0.7, dynamic_friction=0.5, restitution=0.05):
    """递归给 prim 及子 Mesh 添加碰撞和摩擦"""
    if not prim or not prim.IsValid():
        return

    # 如果 prim 是可变换类型，说明可以加 CollisionAPI
    if UsdGeom.Xformable(prim).GetPrim().IsValid():
        UsdPhysics.CollisionAPI.Apply(prim)

    # 如果是 Mesh，额外加 MeshCollisionAPI 和 Material
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim)
        mat_api = UsdPhysics.MaterialAPI.Apply(prim)
        mat_api.CreateStaticFrictionAttr().Set(static_friction)
        mat_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
        mat_api.CreateRestitutionAttr().Set(restitution)

    # 遍历子节点
    for child in prim.GetChildren():
        add_collision_and_material(child, static_friction, dynamic_friction, restitution)


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
            usd_path=f"{SIMULATION_DATA_DIR}/terrains/warehouse/IsaacWarehouse.usd",
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
    "test": AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=UsdFileCfg(
            visible=True,
            usd_path=f"{SIMULATION_DATA_DIR}/terrains/IsaacWarehouse/IsaacWarehouse.usd",
            # usd_path=f"{SIMULATION_DATA_DIR}/terrains/Ragnarok/Koenigsegg_Ragnarok.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.5)),
    ),
}