import os
import time
import numpy as np
from typing import Optional, Tuple, List

from PIL import Image
from pxr import UsdGeom, Gf

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
    Spawn a simple 'flag' composed of two cubes: a thin pole and a small banner.
    world_pos_xyz: np.array([x,y,z]) for base position (ground contact).
    """
    parent_path = f"/World/Flag_{name_suffix}"
    UsdGeom.Xform.Define(stage, parent_path)

    # Pole
    pole_path = f"{parent_path}/Pole"
    pole = UsdGeom.Cube.Define(stage, pole_path)
    pole_xf = UsdGeom.Xformable(pole.GetPrim())
    pole_xf.AddScaleOp().Set(Gf.Vec3f(0.02, 0.02, 1.0))  # ~1m height
    pole_xf.AddTranslateOp().Set(Gf.Vec3f(world_pos_xyz[0], world_pos_xyz[1], world_pos_xyz[2] + 0.5))
    UsdGeom.Gprim(pole.GetPrim()).GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.3, 0.3)])

    # Banner
    banner_path = f"{parent_path}/Banner"
    banner = UsdGeom.Cube.Define(stage, banner_path)
    banner_xf = UsdGeom.Xformable(banner.GetPrim())
    banner_xf.AddScaleOp().Set(Gf.Vec3f(0.4, 0.02, 0.25))
    banner_xf.AddTranslateOp().Set(Gf.Vec3f(world_pos_xyz[0] + 0.2, world_pos_xyz[1], world_pos_xyz[2] + 0.9))
    UsdGeom.Gprim(banner.GetPrim()).GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

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
            try:
                pos_np = pose_tuple[0][0].cpu().numpy()
                quat_wxyz_np = pose_tuple[1][0].cpu().numpy()
                quat_xyzw_np = np.roll(quat_wxyz_np, -1)
                yaw = _yaw_from_quat_xyzw(quat_xyzw_np)  # assumes [x,y,z,w]
                forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
                offset = 0.8
                flag_pos = pos_np + forward * offset
                _spawn_flag_at(stage, flag_pos, ts_fmt)
                marks.append((pos_np.tolist(), quat_xyzw_np.tolist(), ts_ms))
                print(f"[ACTION] Placed flag at {flag_pos.tolist()} (mark count={len(marks)})")
            except Exception as e:
                print(f"[ACTION] Failed to place flag (TODO): {e}")
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
