import gym
import numpy as np
import os
import yaml
import time
import argparse
from datetime import datetime
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
from AUVEnv import AUVEnv


def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")
    
parser = argparse.ArgumentParser(description='Train DDPG agent for AUV control')
parser.add_argument('--config', type=str, default='config/config_ddpg.yaml', help='Path to config file')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Training or testing mode')
parser.add_argument('--model', type=str, default=None, help='Path to model file for testing')
parser.add_argument('--timesteps', type=int, default=None, help='Total timesteps for training')
args = parser.parse_args()
# Load configuration
config_path = os.path.join(os.path.dirname(__file__), args.config)
config = load_config(config_path)

# Set random seed
random_seed = config['others']['random_seed']
np.random.seed(random_seed)
th.manual_seed(random_seed)


# Set up logging directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs", f"ddpg_{timestamp}")
os.makedirs(log_dir, exist_ok=True)

env = AUVEnv()

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None

noise_sigma = config['agent']['epsilon_initial'] 
action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions),
    sigma=noise_sigma * np.ones(n_actions),
    theta=0.15  # Default OU process parameter
)


# Configure DDPG hyperparameters from config
learning_rate = config['agent']['learning_rate']
buffer_size = config['agent']['buffer_size']
batch_size = config['agent']['batch_size']
tau = config['agent']['tau']
gamma = config['agent']['gamma']

# Policy network architecture
policy_kwargs = {
    "activation_fn": th.nn.ReLU,
    "net_arch": {
        "pi": config['qnetwork']['actor_hidden_layers'], 
        "qf": config['qnetwork']['critic_hidden_layers']
    }
}

# Initialize the DDPG model
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    learning_rate=learning_rate,
    buffer_size=buffer_size,
    batch_size=batch_size,
    tau=tau,
    gamma=gamma,
    seed = random_seed,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=log_dir
)

print(model.policy) 

# Training the model with the callback
model.learn(total_timesteps=50000)

# model.learn(total_timesteps=4000)
model.save("ddpg_race2_auv")

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_race2_auv")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()