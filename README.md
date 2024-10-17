
# Underwater Vehicle Control using Deep Q-Learning

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

Make sure you have ROS set up and running before using this project. The code relies on ROS topics for real-time communication with the underwater vehicle.

## Setup

1. **Install the dependencies:**
   ```bash
   pip install torch numpy matplotlib
   ```
   
   Ensure that ROS is installed and properly set up in your system. You can follow [ROS installation instructions](http://wiki.ros.org/ROS/Installation) to install the necessary ROS packages.

2. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/underwater-vehicle-dqn.git
   cd underwater-vehicle-dqn
   ```

In this mode, the agent will continue to learn and improve its policy while interacting with the environment.

## Key Components

- **`GridWorldEnv`**: The custom environment simulating the underwater vehicle's control system. It interacts with the ROS topics to receive error feedback and send control commands to the vehicle's thrusters and servos.
  
- **`QNetwork`**: A PyTorch neural network that estimates the Q-values for each action given a state. This is the core of the DQN algorithm.

- **`ReplayBuffer`**: A memory buffer that stores past experiences (state, action, reward, next state) to be replayed for training. It helps stabilize the learning process by breaking correlations in the data.

- **`Agent`**: The agent that uses the QNetwork to interact with the environment. It selects actions using an epsilon-greedy policy and updates its Q-values through experience replay.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
