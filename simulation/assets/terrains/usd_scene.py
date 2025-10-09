from pxr import Usd, UsdGeom, UsdPhysics
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import UsdFileCfg

from ...assets import SIMULATION_DATA_DIR


def add_collision_and_material(prim, static_friction=0.7, dynamic_friction=0.5, restitution=0.05, recursive=False):
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
    if recursive:
        for child in prim.GetChildren():
            add_collision_and_material(child, static_friction, dynamic_friction, restitution)


ASSET_DICT = {
    # some assets links: 
    #   https://github.com/usd-wg/assets
    #   https://developer.nvidia.com/usd?utm_source=chatgpt.com
    #   https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html
    #   https://sketchfab.com/c1527823531/collections/7ba5ce3cc61543ddb5649166ebe50bc1-02254f3590f74e50aeee85a6bd19168c
    "carla": AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=UsdFileCfg(
            visible=True,
            usd_path=f"{SIMULATION_DATA_DIR}/terrains/carla/carla.usd",
            # https://drive.google.com/drive/folders/1SiGH3LGDxikIS0cmn6WFqujwO8liKcSE?usp=drive_link
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-200.0, -75.0, 0.0)),
    ),
    "warehouse": AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=UsdFileCfg(
            visible=True,
            usd_path=f"{SIMULATION_DATA_DIR}/terrains/warehouse/IsaacWarehouse.usd",
            # https://d4i3qtqj3r0z5.cloudfront.net/Warehouse_NVD%4010013.zip
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    ),
    "lobby": AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=UsdFileCfg(
            visible=True,
            usd_path=f"{SIMULATION_DATA_DIR}/terrains/lobby/World_Lobby.usd",
            # https://developer.nvidia.com/downloads/usd/siggraph/dataset
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-20, -18, -1.0)),
    ),
    "davinci": AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=UsdFileCfg(
            visible=True,
            # usd_path=f"{SIMULATION_DATA_DIR}/terrains/davinci/xx.usd",
            usd_path=f"/media/lenovo/1/USD/distributable_2023_davinci_v2/Distributable_2023_Davinci/asset/assembly/Interior/Interior.usd",
            # usd_path=f"/media/lenovo/1/USD/distributable_2023_davinci_v2/Distributable_2023_Davinci/shot/rt/rt_010/pub/assembly/rt_010_assembly_create.usd",
            # https://developer.nvidia.com/downloads/usd/dataset/davinci_workshop/distributable_2023_davinci_v2.zip
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 1.0, 0.0), rot=(0.7071, 0.7071, 0.0, 0.0)),
    ),
    "test": AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=UsdFileCfg(
            visible=True,
            # usd_path=f"{SIMULATION_DATA_DIR}/terrains/IsaacWarehouse/IsaacWarehouse.usd",
            usd_path=f"/media/lenovo/1/USD/island-usd-v2.1/island/usd/islandPrman.usda",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0.0)),
    ),
    "sketchfab": AssetBaseCfg(
        prim_path=f"/World/Terrain",
        spawn=UsdFileCfg(
            visible=True,
            # usd_path=f"{SIMULATION_DATA_DIR}/terrains/IsaacWarehouse/IsaacWarehouse.usd",
            usd_path=f"/home/lenovo/Projects/GenAgent/simulation/data/terrains/Road/scene.usdc",
            scale=(0.05, 0.05, 0.05),
            # scale=(0.01, 0.01, 0.01),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0.0), rot=(0.7071, 0.7071, 0.0, 0.0)),
    ),
}