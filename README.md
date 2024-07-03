
# [WIP] Underwater Vehicle Control using Deep Q-Learning

This repository contains a Deep Q-Learning-based system to control an underwater vehicle using reinforcement learning (RL). The agent interacts with the environment through ROS (Robot Operating System) topics and learns an optimal policy to control the vehicle's thrusters and servos in order to minimize position, velocity, and orientation errors.

## Overview

This project uses a Deep Q-Network (DQN) to train an agent to control an underwater vehicle. The agent learns from the environment by interacting with it through ROS topics, receiving feedback in the form of errors (position, velocity, orientation), and adjusting the thrusters and servos to minimize these errors.

The code includes:
- A custom `GridWorldEnv` environment for interaction with the underwater vehicle.
- A Deep Q-Network implemented using PyTorch (`QNetwork` class).
- A replay buffer for experience replay (`ReplayBuffer` class).
- A training loop that uses the DQN to learn optimal control policies (`Agent` class).

## Dependencies

- Python 3.x
- PyTorch
- ROS (Robot Operating System)
- NumPy
- Matplotlib
