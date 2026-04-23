"""
Read joint properties from a robot USD file.
Usage: python read_usd_robot.py /path/to/robot.usd
"""

import sys
import math
from pxr import Usd, UsdPhysics

def read_robot_usd(usd_path: str):
    stage = Usd.Stage.Open(usd_path)

    # ── 1. All joints ────────────────────────────────────────────────
    print("=" * 60)
    print("JOINTS")
    print("=" * 60)
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if "Joint" in t:
            print(f"  {prim.GetName():<40} {t}")

    # ── 2. Drive API (stiffness / damping / maxForce) ────────────────
    print("\n" + "=" * 60)
    print("DRIVE API  (stiffness | damping | maxForce)")
    print("=" * 60)
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.DriveAPI):
            continue
        for drive_type in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI(prim, drive_type)
            s = drive.GetStiffnessAttr()
            d = drive.GetDampingAttr()
            f = drive.GetMaxForceAttr()
            if s and s.Get() is not None:
                print(f"  {prim.GetName():<30} [{drive_type}]"
                      f"  stiffness={s.Get():<10g}"
                      f"  damping={d.Get():<10g}"
                      f"  maxForce={f.Get()}")

    # ── 3. Joint limits ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("JOINT LIMITS  (degrees for revolute, metres for prismatic)")
    print("=" * 60)
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if t == "PhysicsRevoluteJoint":
            j = UsdPhysics.RevoluteJoint(prim)
            lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
            lo_r = math.radians(lo) if lo is not None else None
            hi_r = math.radians(hi) if hi is not None else None
            print(f"  {prim.GetName():<30}  [{lo:.4f}, {hi:.4f}] deg"
                  f"  →  [{lo_r:.4f}, {hi_r:.4f}] rad")
        elif t == "PhysicsPrismaticJoint":
            j = UsdPhysics.PrismaticJoint(prim)
            lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
            print(f"  {prim.GetName():<30}  [{lo}, {hi}] m")

    # ── 4. Velocity limits ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VELOCITY LIMITS  (physxJoint:maxJointVelocity)")
    print("=" * 60)
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if "Joint" not in t or "Fixed" in t:
            continue
        vel = prim.GetAttribute("physxJoint:maxJointVelocity")
        if vel and vel.Get() is not None:
            v = vel.Get()
            unit = "m/s" if t == "PhysicsPrismaticJoint" else "deg/s"
            rad = f"  →  {math.radians(v):.4f} rad/s" if unit == "deg/s" else ""
            print(f"  {prim.GetName():<30}  {v:.4f} {unit}{rad}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/xerous/Desktop/project/panda_train/source/panda_train/panda_train/tasks/manager_based/panda_lift/usd/franka_fr3.py"
    read_robot_usd(path)