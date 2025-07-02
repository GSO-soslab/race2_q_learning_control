# SAC Offline Training - Quick Start Guide

**ROS2 Bag → CSV → Train SAC → Deploy**

---

## 📋 Prerequisites
```bash
# Install dependencies
pip install torch pandas numpy pyyaml tqdm matplotlib tensorboard
pip install rosbag2_py rclpy 

# Source ROS2 (if using bag conversion)
source /opt/ros/jazzy/setup.bash
```

---

## ⚡ Quick Commands

### 1️⃣ **Convert ROS2 Bag to Training CSV**
```bash
python rosbag_to_training.py your_data.mcap -o training_data.csv -c config/config_sac.yaml
```

### 2️⃣ **Train SAC Model**
```bash
python sac_training.py training_data.csv -c config/config_sac.yaml
```

### 3️⃣ **Run Inference (Continuous Control)**
```bash
python sac_inference.py runs/checkpoints/sac_training_YYYYMMDD_HHMMSS/best_model.pth --config config/config_sac.yaml
```

---

## 📁 Expected File Structure
```
SAC_offline/
├── config/
│   └── config_sac.yaml      # Main SAC config
├── rosbag_to_training.py     # Step 1: Data conversion
├── sac_training.py           # Step 2: Model training  
├── sac_inference.py          # Step 3: Deployment
├── coupling_rewards.py       # Custom reward functions
└── your_data.mcap            # Input ROS2 bag file
```

---

## 🔧 Common Options

### **Data Conversion Options**
```bash
# List available topics (without converting)
python rosbag_to_training.py your_data.mcap --list-topics

# Specify output file
python rosbag_to_training.py your_data.mcap -o custom_name.csv

# Use different config
python rosbag_to_training.py your_data.mcap -c config/custom_config.yaml
```

### **Training Options**
```bash
# Override epochs and batch size
python sac_training.py data.csv --epochs 1000 --batch-size 128

# Use specific GPU
python sac_training.py data.csv --device cuda:0

# Enable debug mode
python sac_training.py data.csv --debug

# Disable tensorboard, enable wandb
python sac_training.py data.csv --no-tensorboard --wandb
```

### **Inference Options**
```bash
# Run specific number of episodes (instead of continuous)
python sac_inference.py model.pth --episodes 10 --max-steps 500

# Use stochastic policy
python sac_inference.py model.pth --stochastic

# Custom control frequency and setpoint interval
python sac_inference.py model.pth --frequency 5.0 --setpoint-interval 60.0

# Disable automatic setpoint changes
python sac_inference.py model.pth --no-setpoint-changes
```

---

## 📊 Monitor Training
```bash
# View training progress
tensorboard --logdir runs/

# Check saved models
ls runs/checkpoints/sac_training_*/
```

---

## 🛟 Troubleshooting

### **Common Issues**
```bash
# ROS2 import errors
source /opt/ros/humble/setup.bash

# CUDA out of memory
python sac_training.py data.csv --batch-size 64 --device cpu

# No data in bag file
python rosbag_to_training.py data.mcap --list-topics
```

### **Check Results**
```bash
# Verify CSV conversion
head -5 training_data.csv
wc -l training_data.csv

# Check model loading
python -c "import torch; print(torch.load('model.pth', map_location='cpu').keys())"

# Test inference without ROS2
python sac_inference.py model.pth --episodes 1 --max-steps 10
```

---