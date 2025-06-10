# train_dreamerv3_offline.py (Corrected with configurable ROS path)

import os
import sys
import subprocess
import argparse
from pathlib import Path

DREAMERV3_REPO_PATH = os.path.expanduser('~/dreamerv3')

def main(args):
    logdir = Path(args.logdir)
    mcap_path = Path(args.mcap_path)
    offline_data_dir = logdir / 'offline_data'
    ros2_setup_path = Path(args.ros2_setup_path).expanduser()
    workspace_setup_path = Path(args.workspace_setup_path).expanduser()

    # --- 1. Run Data Preparation ---
    print("-" * 50)
    print("STEP 1: Preparing offline data from MCAP files...")
    print("-" * 50)
    
    # Check if the setup files exist before trying to source them
    if not ros2_setup_path.exists():
        print(f"Error: ROS 2 setup file not found at '{ros2_setup_path}'")
        print("Please provide the correct path using the --ros2_setup_path argument.")
        return
    if not workspace_setup_path.exists():
        print(f"Error: Workspace setup file not found at '{workspace_setup_path}'")
        print("Please provide the correct path using the --workspace_setup_path argument.")
        return

    processing_script = 'process_mcap_to_dreamerv3.py'
    cmd_process_list = [
        sys.executable, processing_script,
        '--mcap_path', str(mcap_path),
        '--output_dir', str(offline_data_dir),
        '--config', args.config
    ]
    
    try:
        print("Running data processing with ROS2 environment...")
        # Construct the command to be run in a shell that sources the ROS environments
        ros_sourced_cmd = (
            f"source {ros2_setup_path} && "
            f"source {workspace_setup_path} && "
            f"{' '.join(cmd_process_list)}"
        )
        
        # We need to run this command from a clean environment to avoid conflicts.
        # The python subprocess will inherit the environment of this script.
        # It's better to ensure the current env is clean.
        clean_env = os.environ.copy()
        for key in list(clean_env.keys()):
            if 'ROS' in key or 'AMENT' in key:
                del clean_env[key]

        # Execute the command. The shell=True is necessary for 'source' to work.
        subprocess.run(ros_sourced_cmd, shell=True, check=True, executable='/bin/bash', env=clean_env)
        
        print("Data preparation successful.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Data preparation failed: {e}")
        return

    # --- 2. Run DreamerV3 Training ---
    print("\n" + "-" * 50)
    print("STEP 2: Launching DreamerV3 training...")
    print("-" * 50)

    main_script = Path(DREAMERV3_REPO_PATH) / 'dreamerv3' / 'main.py'
    if not main_script.exists():
        raise FileNotFoundError(f"DreamerV3 main.py not found at {main_script}")

    cmd_train = [
        sys.executable, str(main_script),
        f'--logdir={str(logdir)}',
        '--configs=defaults', 
        '--task=auv_control',
        f'--replay.dir={str(offline_data_dir)}',
        '--replay.online=False',
        '--run.train_ratio=512',
        '--run.steps=100000',
        '--batch_size=16',
        '--batch_length=64',
        '--agent.opt.lr=1e-4',
    ]
    
    print("Running DreamerV3 with command:")
    print(" \\\n    ".join(cmd_train))

    # Set PYTHONPATH for the environment wrapper, ensure no ROS paths
    # The clean_env from before is perfect for this.
    project_dir = os.path.dirname(os.path.abspath(__file__))
    clean_env['PYTHONPATH'] = f"{project_dir}:{clean_env.get('PYTHONPATH', '')}"
    
    try:
        subprocess.run(cmd_train, check=True, env=clean_env)
        print("DreamerV3 training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"DreamerV3 training failed with return code: {e.returncode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DreamerV3 from MCAP files.')
    parser.add_argument('--mcap_path', type=str, required=True, help='Path to the rosbag directory.')
    parser.add_argument('--logdir', type=str, default='logs/dreamerv3_auv_offline', help='Directory to save logs and offline data.')
    parser.add_argument('--config', type=str, default='config/config_sac.yaml', help='Path to AUV environment config file.')
    
    # --- NEW ARGUMENTS ---
    parser.add_argument(
        '--ros2_setup_path', 
        type=str, 
        default='/opt/ros/humble/setup.bash', # Keep default, but now it's easy to override
        help='Path to the main ROS 2 setup.bash file.'
    )
    parser.add_argument(
        '--workspace_setup_path', 
        type=str, 
        default='~/race2_ws/install/setup.bash', # Default for your workspace
        help='Path to your Colcon workspace setup.bash file.'
    )
    
    args = parser.parse_args()
    main(args)