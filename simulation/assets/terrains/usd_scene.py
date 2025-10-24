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
            add_collision_and_material(child, static_friction, dynamic_friction, restitution, recursive)


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
            usd_path=f"{SIMULATION_DATA_DIR}/terrains/distributable_2023_davinci_v2/Distributable_2023_Davinci/asset/assembly/Interior/Interior.usd",
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
            usd_path=f"{SIMULATION_DATA_DIR}/terrains/Road/scene.usdc",
            scale=(0.05, 0.05, 0.05),
            # usd_path=f"{SIMULATION_DATA_DIR}/terrains/factory_optymize_v2/scene.usdc",
            # scale=(0.1, 0.1, 0.1),
            # usd_path=f"{SIMULATION_DATA_DIR}/terrains/CCity_Building_Set_1/scene.usdc",
            # scale=(0.1, 0.1, 0.1),
            # Road: https://sketchfab-prod-media.s3.amazonaws.com/archives/6a0e3b97a8a54f2a909d351322916293/usdz/a8a92840a69f45dfba44436472d88013/Neighbourhood_City_Modular_lowpoly.usdz?AWSAccessKeyId=ASIAZ4EAQ242LKVSATTF&Signature=bZl522vBvCLV3VHF4UT0y0SBPmg%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDcaCWV1LXdlc3QtMSJHMEUCIQCVAsf2qK8AuOq7pl1tFv4tpHuLrxpBTiT2Vt4iR4S%2BXwIgE4PXLsjCjy11rwYshbmOYrfh67VlgLmdpwAgLl6MofgquwUI0P%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw2Nzg4NzQzNzE4OTIiDEpHSnbZFX%2BdreB%2FPyqPBbHKGm9o%2Fp%2B8VFx0fQM%2Fne9Cq4OAkAdU5y7mrvipJvP%2B4l9N98dV14vsoNu%2FNX7MtDesrST1SbqS7LTtYWYRSSzvHYev1P4WRkNuM0XBSAkmo2uB1hPhDQZl2WMODm2woU6MPtFVf4S96VQq5oi3Q3t3RxivykTLVRmRI0h4OtPOoPXa%2FhJZFvQUO%2BKnTFg5Xxn5nIPvuRpt4orGv0N60A5WZYSXDybPQQykimPbdhHJREDAix824xbje%2BDnNH55EdB13Eq8grHaHOIc%2FoldVnhpBdzimzYh4GbYJ5irjFF%2BAS%2FGjJjj%2FVJ1ncKA6n3vgup%2FdVeGl0cAgKRxzpAAH6P%2BlH9BzJc5fvUcBIR9K2MWP420PDpbZhZCnL2rrkCHpUbZ0CWQ79UCV%2FR6L7lBuoaeyxdN0%2F1fumO17fpqqH3PudF1SNJ%2F3NXXBFP3veU1bl2QvrCbGWw8hYGW4QnN0SaJkLJpIPMpRTqIIyu9XAsdL49WULxbY68axIH%2FxeScJijGiDRT%2BCaKbqIe8gDu%2BXkeu3LRnY%2FB0LCoQpFAJJduF0g98Uy3xdA%2BZkQ5X4A1yR7MecW%2FA%2FQWrOoCB8iD%2FuDF2qcqe8%2FBcLh14DmInHNG3t8kT8egdNrBgLMVjIwEYUCOywZPKhOYbBV6YPjTHoHPKvUlhyj6i2hohEg9jFHq%2FX%2FxMeeuCztUdGdWzOD3kBereaEzVUBke3A6Ql1Hg8z%2BLg4raEuY547I6mD3m3NEMhx7J4wY63aOPWgdY3RphJPO9eqGmPUoPJqYPeoPA4%2BCxQFZI9VLo06VxoUrylXJ6wINXTM6b07dCKu4NITnvv0KiwyQm7Ar9Zbd4L7CH803AgnLCPFN3pbw3NwSb8Qw6KqdxwY6sQEdyodHEG6eI33QPasQ2v890QKvNnXeMnfAgN%2B3ysLnH%2FYTExh7kcwUUHuvPccmHplQ7MMbbSiZf48GBBjb2JnH%2BFTRumBvS4C9u913wtuEVQfQsk%2BvzSIjLSh8c2kY7siY2RK0tpmJXPCk5b5maSyGZeWU%2Bli9XnUmrT5AK0C3APrpYLPzRwUNZYt5oW%2FF8FnUMNGqinRr2qp8HyO10X3SEHLG6BMyFr5a9x%2FQE%2FJp6vg%3D&Expires=1759994322
            # factory_optymize_v2: https://sketchfab-prod-media.s3.amazonaws.com/archives/0c7e077eeef94c59867ce4ee3865890f/usdz/b81fb2a0374d487181a87f6fe2762296/factory_optymize_v2.usdz?AWSAccessKeyId=ASIAZ4EAQ242HFNFVV57&Signature=Cm5jaSUIlF1yrE%2BfoYctx%2BYGoHE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDgaCWV1LXdlc3QtMSJGMEQCIGtyvqSFk6KQJuhWOgIaaQS1ECJoKF0LmExsfN%2FxYTDMAiBn%2BZr3p%2BJYQK0zUsRZy8O6ES6Ss0BXcRvj4BIyxg3M5iq7BQjR%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDY3ODg3NDM3MTg5MiIMgDdNic7KHonqIwbnKo8FUUdHwnLdg%2BKFuNp6l5FoLvoQmRBx4TLD%2FCySocGnWz1vDWAHzsKUdKEyXNulnKrqYkv9StF3eOvwa9VRCj19n5F5sCFiLQbedgYq72U6Z2NEniZgZpeE0n%2F4TnuqLGGn07KJZjPBsCr7Zup%2FMC4lv5LdGTHEQDG96RIXK8ldfHvIPx0azfawkl%2Fxc1JHKgwfOSppYUUusw9sL2eilXaw6q1bzaBev5J0TX9jVQOuJ9Tl770Wgx9pcnEfSaqbFeKtxxBnbodcicWyvUVqaUSGvEV3o0ls%2F93rN707boAbgqBscNhvqdAZfIjv9%2F7i%2FDu%2F5PdZJ%2FRsadrG%2FnQtc44o2k3N6ZPsMHFQ3tgknCSPa0KGinvHJtFNWG61xDbwI%2BjIopvQkTWbZQDnqLir8sBap%2FSxdmKvvEfxSX3UpobXeTefR%2Ff%2FUZT8ygZPk6w7098BV%2Bmj%2Fn%2BHREs%2B67jvJv3H7tHaRlt6nyswOk3Kfsu00TmBxKC8emzHUS2qrxOHR1wAY8WJdek%2BLy92SMZRdAaQKk0TJEh8%2Bq85V6LQWwkp9RIG9D4Fs%2FqHN7lC9BHt4hlelICerCyixFmvgDgMlmpiR3A2feTB7S4sDwvB8zcacYZ5j%2FhI9nKZcJ8iARJKvxQ%2B5LsfURY2B6GvzMBKQS4O05OlBqp3QPBtpvCgVi%2BGNLv2AekLFeuUj3p%2FshE0pu45Elma0G6yBmrppiIxgLvTlJ%2BSuLCRdhTU9tXFjLG%2FvFGGhHwtk5dxqc32exoA1Zd4w59vkFwvi3tnU04rsYbyz7bNwdpz2EX0pWSavCFT3nw5dbcwV1qb%2FIBymiDW9QDgUzjKx%2BlGJBnMWQlNbhRuxQ5FC3JCs1uUwNqoQYEDwDCk2J3HBjqyAa8DpG%2F26xaWGXMaGpV9ujIf04%2F5Wm%2Fe7NXoe5oS7CuMTLwzeqzMkIDy1x%2BLEVoMvYseoGa7Wu4p2mFrfoUg%2FZls%2B%2Fms2fxdf46lPe2m3GvDJTbeUodRXB4e0AP7xvSfeWsc40b2RhpdSWRDzohiTplCb7XnOCa9AIblWMqSKJhtsQ720oEV3ZSbH9Bgowon18jpDn5BgP29IuddZ477orfpsfgnqPR6fVJyIuEC%2F9H3byk%3D&Expires=1759998644
            # CCity_Building_Set_1: https://sketchfab-prod-media.s3.amazonaws.com/archives/a2d5c7bfcc2148fb8994864c43dfcc97/usdz/934ab788aab04314bf1239898302d3dd/CCity_Building_Set_1.usdz?AWSAccessKeyId=ASIAZ4EAQ242HFU2EKX7&Signature=v9T%2BVMvDxwKbfzS2oeaCcp4yCAc%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDgaCWV1LXdlc3QtMSJGMEQCIDbQM42TpWSqviyG79tgfW4TOnk%2FsYUMAfOtMlDolJUlAiAVrZ%2B5aGQ1ooraVzrvBCfEp6IbG9QvJMZtF1mtTcvFfCq6BQjR%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDY3ODg3NDM3MTg5MiIM4gYSSf%2FTGZlqof6PKo4F%2B1xdsg5%2FsYDVszPAyY4nNmnb%2FZoh1Ra%2BBFuaeAqAQtwvIqnoz9RfXb79G%2BOFPPbjjb3cqNtxnJxLNHNFXb0QiVHucWeEX3X%2Ft%2B0%2FN54PFA%2Ft8hl%2B%2B2uEZK%2FbNbOgO3I9LC%2Fpq2L8UL4IHuB13092Go3nibTfJBcA95XxaFEgdW7ufOn%2FPmSQ2t5oMGHUbukhSLfo3BjbZzenA0z4lOa30Bpnfm2MicN4rFg%2FkiQtk1CKTqC9q0FrYNw7Ef80Ql25Q4Qe7Qa6YL1kl9HAfMT9FqgGdUM889YLrcU9ocfXxndShuCv9wp2VmSMFMutUKUq6h8gr%2BN9Jw0OHm433lYK0rZ8hR9u3gE%2FtFeXA5GYSYurkDT5pIoEErxdtSYXYpmFP%2FR%2B0jY1xn0fdUFe0mRD3umBJ5v23aHjfB8rnhUnJKNqRSG3ZZay2IK9QcI18PZiKK0RkGK28M%2FM3jaCQ7AwiHgXjFfHRXkDUEutKaAbva1gs%2FLsJQRhfstkRu77FzCGOsC9vlHyjI9x6G93HeCYHtKN7RG7aWAaskmQo52rLNy9ShTUMMNzdN9iJhOh1tPQ3wgBRY6CeBnbIcipVa%2B5up38Yr5Ef0wgdYoNF9PlsPqxvsV82V4jT93ftocdD6wK2qVlvD0LVwhidhEfXHirO%2BBFkw9Ql11eQu4voqTGvfuOtbtbjZJIdxy%2B2hILJ8dHu63xPJHig9kOhBz3vpLg33kyHrOztZxsdHKYy37kxOMlLPjzyBRhUg4z8Jjn8hoiytA%2Fypqbp%2BMBIXzvBP9uyXZ%2FuXSy4082UIVgcVkIeFv8Q0k7Yq2mPqV9%2Fim8j6UhNzfY4Mls0cj8I4SuROqiUdXixETBXXz22Zr%2BgmBtMPDHnccGOrIBJNler1zP09Im4fjPYRsScV4QANj3xetg22d0XiwCfhaVOHjwPgUfBzm5RlIJH%2BMN%2FxbRAJ15XrmQudoas6TV6GTpuMMItcC42OL4UhzBLEdHLlByAPGlqiffmWs3gv6gW3WliUbYDqvXyB%2FKFEsuR4hn5qxapJqQGgdlXpIweGWOhJBGzsyYjzhtUVYVqdJy9j81igjlW4lrbktwWBT7AQ1K2xFomJLXBx0vpgVfiFls0w%3D%3D&Expires=1759998219
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0.0), rot=(0.7071, 0.7071, 0.0, 0.0)),
    ),
}