import airsim


# Path for the record file, to be used with standalone simulations
record_file = "./record1.csv"

# Array with the UAV's IDs that are
# in the Airsim settings file (for the online simulations)
drone_ids = ["uav1"]
cam_types = [
    airsim.ImageType.Scene,
    airsim.ImageType.DepthVis,
    airsim.ImageType.Segmentation,
]
panoramic = True
cam_poses = [
    airsim.Pose(airsim.Vector3r(-205, -162, 40), airsim.to_quaternion(-30, 0, 45)),
    airsim.Pose(airsim.Vector3r(-205, -162, 40), airsim.to_quaternion(-30, 0, 180)),
]

initial_pose_offset = [14, -28, 8.4]

# Available pedestrian models
pedestrians = [
    "person10_41",
    "person07",
    "person01_20",
    "person05_23",
    "person04_22",
    "person06_24",
    "person02_21",
    "person08_30",
    "person09_36",
]

################################################################################
# execute_run.py configuration
save_rt_paths_as_txt = False
save_sionna_3d_scenes_as_png = False
save_all_data_as_npz = False
plot_beam = False
plot_realtime_throughput = False
scene_file_name = "central_park"
rx_3D_object_name = "mesh-Cube"
rx_starting_x = 23.69
rx_starting_y = -3.351
rx_starting_z = 139
rx_alpha = -0.523599
rx_beta = 0
rx_gamma = 0
rx_antenna_pattern = "tr38901"
rx_antenna_polarization = "V"
tx_x = -154
tx_y = 64
tx_z = 120
tx_alpha = -2.0944
tx_beta = 0.785398
tx_gamma = 0
tx_antenna_pattern = "tr38901"
tx_antenna_polarization = "V"
step_size = 15
number_of_steps = 1
nTx = 64
nRx = 4
random_seed = 1
carrier_frequency = 40e9
cam_z = 700
rx_number = 1
################################################################################
# caviar_integration.py configuration
is_sync = True
is_rescue_mission = True
simulation_time_step = 0.5
save_multimodal = False

# RL trajectory planner (Stable-Baselines3)
# RL planning is mandatory: mission waypoints are generated and optimized without path*.csv.
rl_algorithm = "PPO"  # PPO | A2C | DQN
rl_total_timesteps = 5000
rl_learning_rate = 3e-4
rl_gamma = 0.99
rl_force_retrain = True
rl_model_dir = "./trained_models"
rl_model_name = "trajectory_planner"
rl_save_final_model = True

# Save strategy: evaluate periodically and save best only when performance improves.
rl_eval_freq = 1000
rl_eval_episodes = 5

# Candidate waypoint generation (no CSV input required)
rl_initial_position = (-320.34, -206.58, 128.0)
rl_waypoint_spacing = 40.0
rl_waypoint_rings = 2
rl_max_candidate_waypoints = 12

# Reward shaping
rl_planner_max_steps = 7
rl_planner_pedestrian_probability = 0.35
rl_planner_pedestrian_reward = 1.0
rl_planner_distance_penalty = 0.002
rl_planner_revisit_penalty = 0.25

# AirSim connection and startup configuration
#
# Default setup for mixed OS execution:
# - AirSim (Unreal environment) runs manually on Windows host
# - CAVIAR Python orchestration runs in WSL2 (Ubuntu)
#
# "auto" tries to resolve the Windows host IP from /etc/resolv.conf (WSL2 default).
airsim_host = "172.21.32.1"
airsim_port = 41451

# Keep this as False when AirSim is started manually on Windows.
# Set to True only for all-Linux setups that should auto-launch Unreal.
start_airsim_from_simulate = False

# Linux AirSim executable path (used only when start_airsim_from_simulate=True)
airsim_linux_executable = (
    "3d/central_park/LinuxNoEditor/central_park/Binaries/Linux/central_park-Linux-DebugGame"
)
