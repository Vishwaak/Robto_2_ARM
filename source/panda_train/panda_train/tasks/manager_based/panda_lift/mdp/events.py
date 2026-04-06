import torch
import numpy as np
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def randomize_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
    pos_range: dict = {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)},
    rot_range: dict = {"roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05)},
) -> None:
    """
    Randomize wrist camera position and orientation on reset.
    Simulates real-world mounting uncertainty.
    """
    import omni.usd
    from pxr import UsdGeom, Gf
    import math

    stage = omni.usd.get_context().get_stage()

    for env_id in env_ids.tolist():
        # Build prim path for this env's camera
        cam_path = f"/World/envs/env_{env_id}/Robot/panda_hand/wrist_camera"
        prim = stage.GetPrimAtPath(cam_path)

        if not prim.IsValid():
            continue

        xform = UsdGeom.Xformable(prim)

        # Base position (from your mount drawing)
        base_pos = [0.05, 0.0, -0.065]  # 50mm forward, 65mm down

        # Random position offset
        rand_pos = [
            base_pos[0] + np.random.uniform(*pos_range.get("x", (0, 0))),
            base_pos[1] + np.random.uniform(*pos_range.get("y", (0, 0))),
            base_pos[2] + np.random.uniform(*pos_range.get("z", (0, 0))),
        ]

        # Random rotation offset (in radians)
        roll  = np.random.uniform(*rot_range.get("roll",  (0, 0)))
        pitch = np.random.uniform(*rot_range.get("pitch", (0, 0)))
        yaw   = np.random.uniform(*rot_range.get("yaw",   (0, 0)))

        # Convert euler to quaternion
        cr, sr = math.cos(roll/2),  math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2),   math.sin(yaw/2)

        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

        # Set transform on prim
        xform_ops = xform.GetOrderedXformOps()
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*rand_pos))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(Gf.Quatd(qw, qx, qy, qz))
                
import torch
import numpy as np
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg



def randomize_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
    pos_range: dict = {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)},
    rot_range: dict = {"roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05)},
) -> None:
    """
    Randomize wrist camera position and orientation on reset.
    Simulates real-world mounting uncertainty.
    """
    import omni.usd
    from pxr import UsdGeom, Gf
    import math

    stage = omni.usd.get_context().get_stage()

    for env_id in env_ids.tolist():
        # Build prim path for this env's camera
        cam_path = f"/World/envs/env_{env_id}/Robot/panda_hand/wrist_camera"
        prim = stage.GetPrimAtPath(cam_path)

        if not prim.IsValid():
            continue

        xform = UsdGeom.Xformable(prim)

        # Base position (from your mount drawing)
        base_pos = [0.05, 0.0, -0.065]  # 50mm forward, 65mm down

        # Random position offset
        rand_pos = [
            base_pos[0] + np.random.uniform(*pos_range.get("x", (0, 0))),
            base_pos[1] + np.random.uniform(*pos_range.get("y", (0, 0))),
            base_pos[2] + np.random.uniform(*pos_range.get("z", (0, 0))),
        ]

        # Random rotation offset (in radians)
        roll  = np.random.uniform(*rot_range.get("roll",  (0, 0)))
        pitch = np.random.uniform(*rot_range.get("pitch", (0, 0)))
        yaw   = np.random.uniform(*rot_range.get("yaw",   (0, 0)))

        # Convert euler to quaternion
        cr, sr = math.cos(roll/2),  math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2),   math.sin(yaw/2)

        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

        # Set transform on prim
        xform_ops = xform.GetOrderedXformOps()
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*rand_pos))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(Gf.Quatd(qw, qx, qy, qz))