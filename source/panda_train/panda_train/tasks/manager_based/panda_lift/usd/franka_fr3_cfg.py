import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

FRANKA_FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
       usd_path="/home/xerous/Desktop/project/panda_train/source/panda_train/panda_train/tasks/manager_based/panda_lift/usd/fr3.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            # angular_damping=0.0,
            # linear_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            # fix_root_link=True,
        ),

        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
        drive_type="force",  # switches to force control, clears baked stiffness/damping
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.11245,
            "fr3_joint2": -0.6490,
            "fr3_joint3": 0.0569,
            "fr3_joint4": -2.2476,
            "fr3_joint5": 0.041,
            "fr3_joint6": 1.6819,
            "fr3_joint7": 0.8643,
            "fr3_finger_joint.*": 0.04,
        },
    ),

    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=60000.0,
            damping=6000.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=60000.0,
            damping=6000.0,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit_sim=100.0,
            stiffness=60000,
            damping=6000,
        ),
    },
    
    soft_joint_pos_limit_factor=1.0,
)



#  init_state=ArticulationCfg.InitialStateCfg(
#         joint_pos={
#             "fr3_joint1": 0.0,
#             "fr3_joint2": -0.569,
#             "fr3_joint3": 0.0,
#             "fr3_joint4": -2.510,
#             "fr3_joint5": 0.0,
#             "fr3_joint6": 3.037,
#             "fr3_joint7": 0.741,
#             "fr3_finger_joint1": 0.04,
#             "fr3_finger_joint2": 0.04,
#         },
#     ),