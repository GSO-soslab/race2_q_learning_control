# Underwater Vehicle Control using Deep Reinforcement Learning

This repository contains a SAC (Soft Actor-Critic) based system to control an underwater vehicle using reinforcement learning. The agent learns optimal control policies for thrusters and servos to minimize position, velocity, and orientation errors.

## Overview

The system supports two training modes:
- **CSV Training**: Offline training using recorded AUV data from rosbags
- **Online Training**: Real-time training with ROS2 simulation/hardware

## Dependencies

- Python 3.x
- PyTorch
- Stable-Baselines3
- ROS2 (for online training and rosbag conversion)
- NumPy, Pandas, Matplotlib
- YAML for configuration

## Setup

### 1. Install Dependencies
```bash
pip install torch stable-baselines3 pandas matplotlib pyyaml
```

### 2. Configure Training Parameters
Edit `config/config_sac.yaml` to adjust network architecture, learning parameters, and reward weights.

## Usage

### CSV Training (Offline)

1. **Convert rosbag/MCAP files to CSV:**
```bash
# Convert MCAP file to CSV
python rosbag_to_csv.py /path/to/file.mcap -o /path/to/csv/output

# List topics in MCAP file first (optional)
python rosbag_to_csv.py /path/to/file.mcap --list-topics
```

2. **Organize CSV data:**
```bash
python csv_organizer.py /path/to/csv/output data/organized/
```

3. **Validate data (optional):**
```bash
python csv_debug_script.py data/organized/
```

4. **Train the agent:**
```bash
# Basic training
python sac_sb3.py --csv_directory data/organized/ --timesteps 10000

# Resume from checkpoint
python sac_sb3.py --csv_directory data/organized/ --resume_from_checkpoint logs/checkpoints/model.zip
```

### Online Training (Real-time)

1. **Start ROS2 simulation/hardware**

2. **Train the agent:**
```bash
# Basic training
python sac_sb3.py --timesteps 10000

# Resume from checkpoint
python sac_sb3.py --resume_from_checkpoint logs/checkpoints/model.zip
```

### Testing Trained Models

```bash
# Test with CSV data
python sac_sb3.py --csv_directory data/organized/ --mode test --model logs/final_model.zip

# Test with online simulation
python sac_sb3.py --mode test --model logs/final_model.zip
```

## Key Features

- **Coupling-Aware Rewards**: Advanced reward functions considering AUV dynamics
- **Warmup Monitoring**: Tracks training phases and gradient updates
- **Episode Diversity**: Random setpoints (online) or data segments (CSV)
- **Enhanced Logging**: Tensorboard integration with detailed metrics

## File Structure

- `sac_sb3.py` - Main training script
- `AUVEnv.py` - Custom environment for AUV control
- `csv_data_manager.py` - Handles CSV data loading and synchronization
- `rosbag_to_csv.py` - Converts MCAP/rosbag files to CSV
- `csv_organizer.py` - Utility to organize CSV files from rosbags
- `csv_debug_script.py` - Validates CSV data for training
- `config/config_sac.yaml` - Training configuration parameters