import os
import time
import numpy as np
from typing import Optional, Tuple, List

from PIL import Image
from pxr import Usd, UsdGeom, Gf

from ..assets import SIMULATION_DATA_DIR

def _save_rgb_jpg(rgb_np_uint8: np.ndarray, out_path: str) -> None:
    img = Image.fromarray(rgb_np_uint8)
    img.save(out_path, format="JPEG")

def _yaw_from_quat_xyzw(q_xyzw: np.ndarray) -> float:
    # q = [x, y, z, w]
    x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(t3, t4))

def _spawn_flag_at(stage, world_pos_xyz: np.ndarray, name_suffix: str) -> None:
    """
    Spawn a flag USD asset at the given world position using mark_usd_path.
    world_pos_xyz: np.array([x,y,z]) for base position (ground contact).
    """
    parent_path = f"/World/Flag_{name_suffix}"
    xform = UsdGeom.Xform.Define(stage, parent_path)
    prim = xform.GetPrim()

    mark_usd_path = f"{SIMULATION_DATA_DIR}/robots/marks/Chinese_Flag/scene.usdc"

    # Reference the flag asset and place it at the specified world position
    prim.GetReferences().ClearReferences()
    prim.GetReferences().AddReference(mark_usd_path)

    # 使用 XformCommonAPI 来复用/更新 T/R/S，避免重复单轴旋转 xformOp
    xform_api = UsdGeom.XformCommonAPI(prim)
    scale = 0.005
    xform_api.SetTranslate(Gf.Vec3d(float(world_pos_xyz[0]),
                                    float(world_pos_xyz[1]),
                                    float(world_pos_xyz[2])))
    xform_api.SetRotate(Gf.Vec3f(90.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xform_api.SetScale(Gf.Vec3f(scale, scale, scale))

def handle_enum_action(
    enum_cmd: int,
    rgb_tensor,
    pose_tuple: Optional[Tuple],
    stage,
    captured_dir: str,
    reports_dir: str,
    marks: List,
) -> None:
    """
    Handle enum_cmd:
    - 0: Capture RGB to JPG in captured_dir with timestamp
    - 1: Mark (spawn a simple flag ahead of robot in stage)
    - 2: Report (write a text file with marks_count in reports_dir)
    """
    ts_fmt = time.strftime("%Y%m%d_%H%M%S")
    ts_ms = int(time.time() * 1000)
    print(f"[ACTION] Handling enum_cmd={enum_cmd} at {ts_fmt}")

    if enum_cmd == 0:
        if rgb_tensor is not None:
            try:
                rgb_arr = rgb_tensor[0, :, :, :3].cpu().numpy().astype(np.uint8)
                out_path = os.path.join(captured_dir, f"{ts_fmt}.jpg")
                _save_rgb_jpg(rgb_arr, out_path)
                print(f"[ACTION] Saved photo to {out_path}")
            except Exception as e:
                print(f"[ACTION] Failed to save photo: {e}")
        else:
            print("[ACTION] Capture requested but RGB unavailable")

    elif enum_cmd == 1:
        if pose_tuple is not None:
            pos_np = pose_tuple[0][0].cpu().numpy()
            quat_wxyz_np = pose_tuple[1][0].cpu().numpy()
            quat_xyzw_np = np.roll(quat_wxyz_np, -1)
            yaw = _yaw_from_quat_xyzw(quat_xyzw_np)  # assumes [x,y,z,w]
            # 机器人本地前向为 +Y，将 +Y 轴按 yaw 旋转：[-sin(yaw), cos(yaw), 0]
            forward = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32)
            offset = 1.0
            flag_pos = pos_np + forward * offset
            flag_pos[2] -= 0.2
            # 加入毫秒保证 prim 路径唯一，避免对同一 prim 反复追加 xformOp
            _spawn_flag_at(stage, flag_pos, f"{ts_fmt}_{ts_ms}")
            marks.append((pos_np.tolist(), quat_xyzw_np.tolist(), ts_ms))
            print(f"[ACTION] Placed flag at {flag_pos.tolist()} (mark count={len(marks)})")
        else:
            print("[ACTION] Mark requested but pose unavailable")

    elif enum_cmd == 2:
        report = {"timestamp": ts_ms, "marks_count": len(marks)}
        out_path = os.path.join(reports_dir, f"report_{ts_fmt}.txt")
        try:
            with open(out_path, "w") as f:
                f.write(str(report) + "\n")
            print(f"[ACTION] Reported status to {out_path}: {report}")
        except Exception as e:
            print(f"[ACTION] Failed to write report: {e}")
