#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import yaml
from collections import deque
from std_msgs.msg import Header, Float64
from geometry_msgs.msg import Vector3
from mvp_msgs.msg import ControlProcess


from geometry_msgs.msg import Pose, Twist


# Set the path to the config file in the parent config directory
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_ddpg.yaml')

# Load the configuration file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class OUActionNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""
    def __init__(self, mean, std_deviation, theta=0.2, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        
    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, buffer_capacity=10000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)

class Actor(nn.Module):
    """Actor Network for DDPG using PyTorch"""
    def __init__(self, state_dim, action_dim, action_bound, hidden_dims=[600, 400, 300]):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        
        # Define network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
        
    def forward(self, state):
        x = self.hidden_layers(state)
        x = torch.tanh(self.output_layer(x))  # tanh activation for [-1, 1] output
        return x * self.action_bound

class Critic(nn.Module):
    """Critic Network for DDPG using PyTorch"""
    def __init__(self, state_dim, action_dim, hidden_dims=[600, 400, 300]):
        super(Critic, self).__init__()
        
        # State input processing
        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        # Action input processing
        self.action_layer = nn.Sequential(
            nn.Linear(action_dim, hidden_dims[1]),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined_layers = nn.Sequential(
            nn.Linear(hidden_dims[1] * 2, hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        
    def forward(self, state, action):
        state_features = self.state_layers(state)
        action_features = self.action_layer(action)
        combined = torch.cat([state_features, action_features], dim=1)
        q_value = self.combined_layers(combined)
        return q_value

class DDPG:
    """DDPG Agent for AUV control using PyTorch"""
    def __init__(self, state_dim, action_dim, action_bound, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device
        
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_bound).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=0.001)
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer()
        
        # Initialize noise process
        self.noise = OUActionNoise(
            mean=np.zeros(action_dim),
            std_deviation=0.2 * np.ones(action_dim)
        )
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001   # Target network update rate
        
    def get_action(self, state, add_noise=True):
        """Return action for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            noise = self.noise()
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.buffer.add(state, action, reward, next_state, done)
    
    def learn(self):
        """Update actor and critic networks from replay buffer"""
        if self.buffer.size() < self.buffer.batch_size:
            return None, None
        
        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor using deterministic policy gradient
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_targets()
        
        return critic_loss.item(), actor_loss.item()
    
    def update_targets(self):
        """Soft update target networks"""
        # Update actor target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_weights(self, path):
        """Save model weights"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)
    
    def load_weights(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

class DDPG_ROS2(Node):
    """DDPG Agent for AUV control integrated with ROS2"""
    def __init__(self, config):
        super().__init__('ddpg_auv_control')
        
        self.config = config

        # Define dimensions
        self.state_dim = 18  # 3D position + 3D orientation + 3D velocity + 3D angular rate
        self.action_dim = 6  # 4 thrusters + 2 servo angles
        self.action_bound = 0.6  # All commands between -1 and 1
        
        # Create DDPG agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        self.agent = DDPG(self.state_dim, self.action_dim, self.action_bound, self.device)
        
        # ROS2 publishers for each actuator
        self.thruster_pubs = {
            'heave_bow': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_bow', 1),
            'heave_stern': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_stern', 1),
            'surge_port': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_port', 1),
            'surge_starboard': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_starboard', 1),
            'port_servo': self.create_publisher(Float64, '/race2_auv/control/surge_port_servo', 1),
            'starboard_servo': self.create_publisher(Float64, '/race2_auv/control/surge_starboard_servo', 1)
        }
        
        self.create_subscription(ControlProcess,  
                                '/race2_auv/controller/process/value',
                                self.state_callback,
                                10)

        self.create_subscription(ControlProcess, 
                                '/race2_auv/controller/process/setpoint', 
                                self.setpoint_callback,
                                10)
        
        self.create_subscription(ControlProcess, 
                                '/race2_auv/controller/process/error', 
                                self.error_callback,
                                10)

        self.create_subscription(Float64, 
                                 '/race2_auv/control/surge_port_servo', 
                                 self.update_joint_port, 
                                 10)
        
        self.create_subscription(Float64, 
                                 '/race2_auv/control/surge_starboard_servo', 
                                 self.update_joint_starboard, 
                                 10)
                                 
        self.create_subscription(Float64, 
                                 '/race2_auv/control/thruster/heave_bow', 
                                 self.update_thrust_heave_bow, 
                                 10)
        
        self.create_subscription(Float64, 
                                 '/race2_auv/control/thruster/surge_port', 
                                 self.update_thrust_surge_port, 
                                 10)
        
        self.create_subscription(Float64, 
                                 '/race2_auv/control/thruster/surge_starboard', 
                                 self.update_thrust_surge_starboard, 
                                 10)
        
        self.create_subscription(Float64, 
                                 '/race2_auv/control/thruster/sway_stern', 
                                 self.update_thrust_sway_stern, 
                                 10)
        
        # Training parameters
        self.declare_parameter('training_mode', True)
        self.declare_parameter('max_steps', 700)
        self.declare_parameter('model_path', '')
        
        self.declare_parameter('max_episodes', 2000)  # Default 1000 episodes
        self.max_episodes = self.get_parameter('max_episodes').value

        self.training_mode = self.get_parameter('training_mode').value
        self.max_steps = self.get_parameter('max_steps').value
        model_path = self.get_parameter('model_path').value
        
        # State tracking
        self.current_state = None
        self.prev_state = None
        self.prev_action = None
        
        # Initialize state variables
        self.position_state = np.zeros(3)
        self.orientation_state = np.zeros(3)
        self.v_state = np.zeros(3)
        self.omega_ref_state = np.zeros(3)
        
        # Initialize setpoint variables
        self.position_setpoint = np.zeros(3)
        self.orientation_setpoint = np.zeros(3)
        self.v_setpoint = np.zeros(3)
        self.omega_ref_setpoint = np.zeros(3)
        
        # Initialize error variables
        self.position_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)
        
        # Initialize state error for RL
        self.state_err = np.zeros(6)  # Same size as state_dim
        self.prev_state_err = None

        # Initialize actuator variables
        self.thruster_action = np.zeros(4)  # For the 4 thrusters
        self.joint_angles = np.zeros(2)     # For the 2 servo angles
        
        self.joint_angles_port = 0.0
        self.joint_angles_starboard = 0.0

        self.thrust_heave_bow = 0.0
        self.thrust_surge_port = 0.0
        self.thrust_surge_starboard = 0.0
        self.thrust_sway_stern = 0.0

        # Initialize history arrays for smoothness calculations
        self.joint_positions_history = np.zeros((10, 2))  # Store last 10 servo positions
        self.u_prev = np.zeros((10, 4))  # Store last 10 thruster commands
        self.thruster_command_action_prev = np.zeros((10, 4))  # Store last 10 thruster actions
        
        # Load model if available
        if model_path:
            try:
                self.agent.load_weights(model_path)
                self.get_logger().info(f"Loaded model from: {model_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load model from: {model_path}. Error: {e}")
        
        # Episode tracking
        self.episode_step = 0
        self.episode_count = 0
        self.episode_reward = 0
        
        # Create timer for control loop
        self.timer = self.create_timer(0.01, self.control_loop)  # 5 Hz control loop
        
    def state_callback(self, data):
        """Process state updates from sensors"""
        # Extract state values
        self.position_state = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_state = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_state = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_state = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
        
        # Update current state for RL agent
        self.current_state = np.concatenate([
                self.position_state[2:3],
                self.v_state[:2],
                self.orientation_state[:3],
        ])
        
        # Calculate errors
        # self.update_errors()

    # def update_errors(self):
    #     """Calculate errors between current state and setpoint"""
    #     if hasattr(self, 'current_state') and hasattr(self, 'current_setpoint'):
    #         # Calculate vector error for RL state representation
    #         self.state_err = self.current_setpoint - self.current_state
            
    #         # Normalize orientation error components (last 3 elements) to [-pi, pi]
    #         for i in range(3, 6):  # Indices for orientation components
    #             self.state_err[i] = ((self.state_err[i] + np.pi) % (2 * np.pi)) - np.pi
            
    #         # Also calculate individual component errors for debugging/logging
    #         self.position_err = self.position_setpoint - self.position_state
    #         self.v_err = self.v_setpoint - self.v_state
            
    #         # Handle orientation errors with angle wrapping
    #         self.orientation_err = self.orientation_setpoint - self.orientation_state
    #         self.orientation_err = np.array([
    #             ((angle + np.pi) % (2 * np.pi)) - np.pi 
    #             for angle in self.orientation_err
    #         ])
            
    #         self.omega_ref_err = self.omega_ref_setpoint - self.omega_ref_state

    #                 # Update current state for RL agent
    #     self.state_err = np.concatenate([
    #             self.position_err[2:3],
    #             self.v_err[:2],
    #             self.orientation_err[:3],
    #     ])

    def setpoint_callback(self, data):
        """Process setpoint updates"""
        self.position_setpoint = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_setpoint = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_setpoint = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_setpoint = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
        
        self.current_setpoint = np.concatenate([
        self.position_setpoint[2:3],
        self.v_setpoint[:2],
        self.orientation_setpoint[:3],
        ])

    def error_callback(self, data):
        """Process error updates"""
        self.position_err = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_err = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
        
        self.state_err = np.concatenate([
        self.position_err[2:3],
        self.v_err[:2],
        self.orientation_err[:3],
        ])

        # Update errors after setpoint changes
        # self.update_errors()

    def update_joint_port(self, data):
        self.joint_angles_port = data.data
        self.update_joints()

    def update_joint_starboard(self, data):
        self.joint_angles_starboard = data.data
        self.update_joints()

    def update_joints(self):
        self.joint_angles = [self.joint_angles_port, self.joint_angles_starboard]

    def update_thrust_surge_port(self, data):
        self.thrust_surge_port = data.data

    def update_thrust_surge_starboard(self, data):
        self.thrust_surge_starboard = data.data

    def update_thrust_heave_bow(self, data):
        self.thrust_heave_bow = data.data

    def update_thrust_sway_stern(self, data):
        self.thrust_sway_stern = data.data


    def publish_action(self, action):
        """Publish actions to ROS2 topics"""
        # Split action into thruster commands and servo angles
        thruster_cmds = action[:4]
        servo_angles = action[4:]
        
        # Store for reward calculation
        self.thruster_action = thruster_cmds
        self.joint_angles = servo_angles
        
        # Map to appropriate publishers
        thruster_mapping = [
            ('heave_bow', thruster_cmds[0]),
            ('surge_port', thruster_cmds[1]),
            ('surge_starboard', thruster_cmds[2]),
            ('heave_stern', thruster_cmds[3]),
            ('port_servo', servo_angles[0]),
            ('starboard_servo', servo_angles[1])
        ]
        
        # Publish commands
        for name, value in thruster_mapping:
            msg = Float64()
            msg.data = float(value)
            self.thruster_pubs[name].publish(msg)

    def calculate_reward(self, prev_state, current_state):
        """Calculate reward based on specified error components"""
        w = self.config['reward_function']
        w1, w2, w3, w4, w5, w6 ,w7, w8 = w['w1'], w['w2'], w['w3'], w['w4'], w['w5'], w['w6'], w['w7'],w['w8']
        state_error_weights = np.array(w['state_error_weights'])
        # Extract specific error components as requested
        error = np.concatenate(
            [self.position_err[2:3], # Depth
             self.v_err[:2], # Surge and sway
             self.orientation_err[:3], # roll, pitch, yaw
            ]).astype(np.float32)
        print(error)
        # Compute performance error (quadratic penalty)
        # weighted_errors =  * state_error_weights * error
        # performance_error = np.sum(weighted_errors ** 2)
        error_column = error.reshape(-1, 1)
        performance_error = np.dot(error , np.diag(state_error_weights))
        performance_error = np.dot(performance_error,error_column)
        performance_error = np.exp(-performance_error)
        
        # Servo smoothness penalty using sine and cosine components
        servo_smoothness_penalty = 0
        delta_theta = np.zeros(2)

        for i in range(2):
            # Compute average sine and cosine of the historical angles
            avg_sin = np.average(np.sin(self.joint_positions_history[:, i]))
            avg_cos = np.average(np.cos(self.joint_positions_history[:, i]))
            historical_avg_angle = np.arctan2(avg_sin, avg_cos)

            # Get the current angle from self.joint_angles
            current_angle = self.joint_angles[i]

            # Calculate the angular difference
            delta_theta[i] = np.abs(current_angle - historical_avg_angle)

        # Accumulate the smoothness penalty
        servo_smoothness_penalty = np.linalg.norm(delta_theta)
        # Update joint_positions_history
        self.joint_positions_history = np.vstack((self.joint_positions_history[1:], self.joint_angles))

        # Thruster usage penalty
        u_t = np.array([
            self.thrust_heave_bow,
            self.thrust_surge_port,
            self.thrust_surge_starboard,
            self.thrust_sway_stern
        ])
        thruster_usage_penalty = np.sum(np.abs(u_t))

        # # Thruster smoothness penalty
        # if len(self.u_prev) > 0:
        #     thruster_smoothness_penalty = np.linalg.norm(u_t - np.average(self.u_prev, axis=0))
        #     self.u_prev = np.vstack((self.u_prev[1:], u_t))

        # Thruster smoothness penalty
        thruster_smoothness_penalty = np.linalg.norm(u_t - np.average(self.u_prev, axis=0))

        # Update self.u_prev to store the history
        self.u_prev = np.vstack((self.u_prev[1:], u_t))

        # Thruster delta reward
        thruster_delta_reward = np.linalg.norm(u_t - self.u_prev[-2])

        # Servo angle penalty
        servo_angle_penalty = np.linalg.norm(self.joint_angles)
        # print("Smoothness penalty:", thruster_smoothness_penalty)


        # Update thruster_command_action_prev with clear logic
        self.thruster_command_action_prev = np.vstack((
            self.thruster_command_action_prev[1:],  # Keep all except the first
            self.thruster_action  # Add the newest action
        ))
        thruster_action_penalty = np.sum(self.thruster_action - np.average(self.thruster_command_action_prev, axis=0))
        # Debugging output (optional)
        # print("Thruster Action Penalty:", self.thruster_command_action_prev)

        # Thruster Direction Change Penalty
        direction_change_penalty = w8 * thruster_action_penalty ** 2  # Quadratic penalty

        # Total reward
        reward =   -(
            - w1 * performance_error +
            w2 * servo_smoothness_penalty +
            w3 * thruster_usage_penalty +
            w4 * thruster_smoothness_penalty +
            w5 * servo_angle_penalty +
            w6 * thruster_delta_reward +
            w7 * thruster_action_penalty +
            w8 * direction_change_penalty
        )
        
        # print(f"Performance Error Contribution: {-w1 * performance_error}")
        # print(f"Servo Smoothness Penalty Contribution: {-w2 * servo_smoothness_penalty}")
        # print(f"Thruster Usage Penalty Contribution: {-w3 * thruster_usage_penalty}")
        # print(f"Thruster Smoothness Penalty Contribution: {-w4 * thruster_smoothness_penalty}")
        # print(f"Servo Angle Penalty Contribution: {-w5 * servo_angle_penalty}")
        # print(f"Thruster Delta Reward Contribution: {-w6 * thruster_delta_reward}")
        # print(f"Thruster action penalty: {w7 * thruster_action_penalty}")
        # print(reward)
        return reward

    def control_loop(self):
        """Main control loop using state errors as network input"""
        # # Only proceed if we have received both state and setpoint
        # if not (self.current_state and self.current_setpoint):
        #     self.get_logger().info("Waiting for state and setpoint data...")
        #     return
        
        state = np.concatenate(
            [self.position_err[2:3],     # Depth
            self.v_err[:2],             # Surge and sway
            self.orientation_err[:3],   # roll, pitch, yaw
            # self.omega_ref_err[2:3],
            self.position_state[2:3],
            self.v_state[:2],
            self.orientation_state[:3],
            # self.omega_ref_state[2:3],
            # self.position_setpoint[2:3],
            # self.v_setpoint[:2],
            # self.orientation_setpoint[:3],
            self.joint_angles,
            np.array([self.thrust_heave_bow,      # Thrust components
                    self.thrust_surge_port,
                    self.thrust_surge_starboard,
                    self.thrust_sway_stern])
            ])
        
        # Get action from agent based on state errors
        action = self.agent.get_action(state, add_noise=self.training_mode)
        
        # Publish thruster and servo commands
        self.publish_action(action)
        
        # If in training mode, generate reward and train
        if self.training_mode and self.prev_state is not None and self.prev_action is not None:
            reward = self.calculate_reward(self.prev_state, state)
            
            # Ensure reward is a scalar value
            if isinstance(reward, np.ndarray):
                reward = float(reward.item())
                
            self.episode_reward += reward
            done = self.is_done(state)  # Changed from self.state_err to state
            
            # Store in replay buffer
            self.agent.remember(self.prev_state, self.prev_action, reward, state, done)
            
            # Train agent
            critic_loss, actor_loss = self.agent.learn()
            if critic_loss is not None:
                self.get_logger().debug(f"Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")
                
            # Update episode step counter
            self.episode_step += 1
            
            # Check for episode end
            if done or self.episode_step >= self.max_steps:
                self.get_logger().info(f"Episode {self.episode_count} completed: Steps={self.episode_step}, Reward={self.episode_reward:.2f}")
                self.episode_step = 0
                self.episode_reward = 0
                self.episode_count += 1
                
                # Save model periodically
                if self.episode_count % 500 == 0:
                    model_path = f"ddpg_auv_model_ep{self.episode_count}.pt"
                    self.agent.save_weights(model_path)
                    self.get_logger().info(f"Model saved to {model_path}")
                
                # Check if we've reached the maximum number of episodes
                if self.episode_count >= self.max_episodes:
                    self.get_logger().info(f"Reached maximum number of episodes ({self.max_episodes}). Training complete.")
                    self.training_mode = False  # Stop training mode
                    final_model_path = "ddpg_auv_model_final.pt"
                    self.agent.save_weights(final_model_path)
                    self.get_logger().info(f"Final model saved to {final_model_path}")
        
        # Store state and action for next training step
        self.prev_state = state.copy()  # Changed from self.prev_state_err = self.state_err.copy()
        self.prev_action = action

    def is_done(self, state):
        """Check if episode should terminate based on performance and safety constraints"""
        # Extract the error components that are used in the reward calculation
        error = np.concatenate([
            self.position_err[2:3],     # Depth
            self.v_err[:2],             # Surge and sway
            self.orientation_err[:3],   # roll, pitch, yaw
        ]).astype(np.float32)
        
        # Calculate performance error similar to reward function
        state_error_weights = np.array(self.config['reward_function']['state_error_weights'])
        error_column = error.reshape(-1, 1)
        performance_error = np.dot(error, np.diag(state_error_weights))
        performance_error = np.dot(performance_error, error_column)
        
        # Task completion criteria
        # Convert performance_error to a more intuitive measure (lower is better)
        performance_metric = np.exp(-performance_error)
        task_completed = performance_metric > 1.0  # 95% of perfect performance
        
        # Safety constraints
        # Max depth limit (assuming negative z is deeper)
        max_depth = 10.0  # meters, adjust as needed
        depth_exceeded = self.position_state[2] < -max_depth
        
        # Max orientation error limits
        max_roll_error = 0.5    # radians (~30 degrees)
        max_pitch_error = 0.5   # radians (~30 degrees)
        max_yaw_error = 0.3     # radians (~40 degrees)
        print(self.orientation_err[2])
        orientation_error_exceeded = (abs(self.orientation_err[0]) > max_roll_error or 
                                    abs(self.orientation_err[1]) > max_pitch_error or
                                    abs(self.orientation_err[2]) > max_yaw_error)
        
        # Max velocity limits
        max_velocity = 0.3  # m/s, adjust as needed
        velocity_exceeded = np.linalg.norm(self.v_state[:1]) > max_velocity
        
        # Episode terminates if task is completed or safety constraints are violated
        done = task_completed or depth_exceeded or orientation_error_exceeded or velocity_exceeded
        
        # Optionally, you could log the reason for termination
        if done:
            if task_completed:
                self.get_logger().info("Episode completed: Task objectives achieved")
            if depth_exceeded:
                self.get_logger().warn("Episode terminated: Maximum depth exceeded")
            if orientation_error_exceeded:
                self.get_logger().warn("Episode terminated: Maximum orientation error exceeded")
            if velocity_exceeded:
                self.get_logger().warn("Episode terminated: Maximum velocity exceeded")
        
        return done
    
    # def calculate_reward(self, prev_state, current_state):
    #     """Calculate reward based on state transition"""
    #     # Extract position and orientation from states
    #     prev_pos = prev_state[:3]
    #     prev_ori = prev_state[3:6]
    #     curr_pos = current_state[:3]
    #     curr_ori = current_state[3:6]
        
    #     # Position error (negative distance to goal)
    #     pos_error_prev = -np.linalg.norm(prev_pos - self.goal_pos)
    #     pos_error_curr = -np.linalg.norm(curr_pos - self.goal_pos)
        
    #     # Orientation error
    #     ori_error_prev = -np.linalg.norm(prev_ori - self.goal_ori)
    #     ori_error_curr = -np.linalg.norm(curr_ori - self.goal_ori)
        
    #     # Reward is improvement in position and orientation
    #     pos_reward = (pos_error_curr - pos_error_prev) * 10  # Scale factor
    #     ori_reward = (ori_error_curr - ori_error_prev) * 5   # Scale factor
        
    #     # Penalize excessive velocity and angular rates
    #     vel_penalty = -0.1 * np.linalg.norm(current_state[6:9])
    #     ang_rate_penalty = -0.1 * np.linalg.norm(current_state[9:12])
        
    #     # Penalize excessive control effort
    #     control_penalty = -0.05 * np.linalg.norm(self.prev_action)
        
    #     # Sum all reward components
    #     reward = pos_reward + ori_reward + vel_penalty + ang_rate_penalty + control_penalty
        
    #     # Bonus for reaching goal
    #     if np.linalg.norm(curr_pos - self.goal_pos) < 0.5 and np.linalg.norm(curr_ori - self.goal_ori) < 0.2:
    #         reward += 100
            
    #     return reward
        
    # def is_done(self, state):
    #     """Check if episode is complete"""
    #     # Extract position and orientation
    #     position = state[:3]
    #     orientation = state[3:6]
        
    #     # Check if goal reached
    #     pos_error = np.linalg.norm(position - self.goal_pos)
    #     ori_error = np.linalg.norm(orientation - self.goal_ori)
        
    #     # Episode complete if goal reached
    #     return pos_error < 0.5 and ori_error < 0.2
    
    # def publish_action(self, action):
    #     """Publish actions to thruster and servo topics"""
    #     # Distribute the action vector to appropriate actuators
    #     # action[0-3] are thrusters, action[4-5] are servo angles
        
    #     # Create message objects
    #     heave_bow_msg = Float64()
    #     heave_bow_msg.data = float(action[0])
        
    #     heave_stern_msg = Float64()
    #     heave_stern_msg.data = float(action[1])
        
    #     surge_port_msg = Float64()
    #     surge_port_msg.data = float(action[2])
        
    #     surge_starboard_msg = Float64()
    #     surge_starboard_msg.data = float(action[3])
        
    #     port_servo_msg = Float64()
    #     port_servo_msg.data = float(action[4])
        
    #     starboard_servo_msg = Float64()
    #     starboard_servo_msg.data = float(action[5])
        
    #     # Publish thruster commands
    #     self.thruster_pubs['heave_bow'].publish(heave_bow_msg)
    #     self.thruster_pubs['heave_stern'].publish(heave_stern_msg)
    #     self.thruster_pubs['surge_port'].publish(surge_port_msg)
    #     self.thruster_pubs['surge_starboard'].publish(surge_starboard_msg)
        
    #     # Publish servo angle commands
    #     self.thruster_pubs['port_servo'].publish(port_servo_msg)
    #     self.thruster_pubs['starboard_servo'].publish(starboard_servo_msg)
        
    #     # Log action if debugging
    #     self.get_logger().debug(f"Action: Thrusters={action[:4]}, Servos={action[4:]}")

def main(args=None):
    rclpy.init(args=args)
    ddpg_ros = DDPG_ROS2(config)
    
    try:
        rclpy.spin(ddpg_ros)
    except KeyboardInterrupt:
        pass
    finally:
        # Save final model
        ddpg_ros.agent.save_weights("ddpg_auv_model_final.pt")
        ddpg_ros.get_logger().info("Final model saved to ddpg_auv_model_final.pt")
        
        # Destroy the node
        ddpg_ros.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()