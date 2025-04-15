#!/usr/bin/env python3
import os, time
import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from std_msgs.msg import Header, Float64
from mvp_msgs.msg import ControlProcess
from collections import deque

# Set the path to the config file in the parent config directory
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_ddpg.yaml')

# Load the configuration file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class OUActionNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""
    def __init__(self, mean, std_deviation, theta=0.30, dt=1e-2, x_initial=None):
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
    """Experience replay buffer with separate states for actor and critic"""
    def __init__(self, actor_state_dim, critic_state_dim, buffer_capacity=100000, batch_size=60):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_capacity)
        self.actor_state_dim = actor_state_dim
        self.critic_state_dim = critic_state_dim
    
    def add(self, actor_state, critic_state, action, reward, next_actor_state, next_critic_state, done):
        """Add experience to buffer"""
        self.buffer.append((actor_state, critic_state, action, reward, next_actor_state, next_critic_state, done))
    
    def sample(self):
        """Sample a batch of experiences with separate state arrays"""
        batch = random.sample(self.buffer, self.batch_size)
        
        # Separate the experiences
        actor_states = np.array([experience[0] for experience in batch])
        critic_states = np.array([experience[1] for experience in batch])
        actions = np.array([experience[2] for experience in batch])
        rewards = np.array([experience[3] for experience in batch])
        next_actor_states = np.array([experience[4] for experience in batch])
        next_critic_states = np.array([experience[5] for experience in batch])
        dones = np.array([experience[6] for experience in batch])
        
        # Convert to PyTorch tensors
        actor_states = torch.FloatTensor(actor_states)
        critic_states = torch.FloatTensor(critic_states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_actor_states = torch.FloatTensor(next_actor_states)
        next_critic_states = torch.FloatTensor(next_critic_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        return actor_states, critic_states, actions, rewards, next_actor_states, next_critic_states, dones
    
    def size(self):
        """Return the current size of the buffer"""
        return len(self.buffer)

class Actor(nn.Module):
    """Actor Network for DDPG using PyTorch, configured from config file"""
    def __init__(self, state_dim, action_dim, action_bound, config):
        # Validate config is provided and contains required keys
        if config is None:
            raise ValueError("Configuration must be provided for Actor initialization")
        
        if 'qnetwork' not in config:
            raise ValueError("Configuration must contain 'qnetwork' key")
        
        if 'actor_hidden_layers' not in config.get('qnetwork', {}):
            raise ValueError("Configuration must specify 'actor_hidden_layers' in 'qnetwork'")
        
        super(Actor, self).__init__()
        self.action_bound = action_bound
        
        hidden_dims = config.get('qnetwork', {}).get('actor_hidden_layers')
        
        # Dynamically create layers
        layers = []
        current_dim = state_dim
        
        # Create hidden layers dynamically
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # Create network
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, action_dim)
    
    def forward(self, state):
        x = self.layers(state)
        x = torch.tanh(self.output_layer(x))
        return x * self.action_bound

class Critic(nn.Module):
    """Critic Network for DDPG using PyTorch, configured from config file"""
    def __init__(self, state_dim, action_dim, config):
        # Validate config is provided and contains required keys
        if config is None:
            raise ValueError("Configuration must be provided for Critic initialization")
        if 'qnetwork' not in config:
            raise ValueError("Configuration must contain 'qnetwork' key")
        if 'critic_hidden_layers' not in config.get('qnetwork', {}):
            raise ValueError("Configuration must specify 'critic_hidden_layers' in 'qnetwork'")
        
        super(Critic, self).__init__()
        hidden_dims = config.get('qnetwork', {}).get('critic_hidden_layers')
        
        # Ensure there's at least one hidden layer for state and action
        if len(hidden_dims) < 1:
            raise ValueError("Critic must have at least one hidden layer")
        
        # Separate processing for state and action
        # Dynamically create state layers
        state_layers = []
        current_state_dim = state_dim
        for hidden_dim in hidden_dims[:2]:  # Up to first two layers for state
            state_layers.append(nn.Linear(current_state_dim, hidden_dim))
            state_layers.append(nn.ReLU())
            current_state_dim = hidden_dim
        self.state_layers = nn.Sequential(*state_layers)
        
        # Action processing layer
        self.action_layer = nn.Sequential(
            nn.Linear(action_dim, hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0]),
            nn.ReLU()
        )
        
        # Combined processing layers
        combined_dim = hidden_dims[1] * 2 if len(hidden_dims) > 1 else sum(hidden_dims)
        combined_layers = []
        for hidden_dim in hidden_dims[2:] if len(hidden_dims) > 2 else []:
            combined_layers.append(nn.Linear(combined_dim, hidden_dim))
            combined_layers.append(nn.ReLU())
            combined_dim = hidden_dim
        
        # Final output layer with Sigmoid activation
        combined_layers.append(nn.Linear(combined_dim, 1))
        combined_layers.append(nn.ReLU()) 
        
        self.combined_layers = nn.Sequential(*combined_layers)
    
    def forward(self, state, action):
        state_features = self.state_layers(state)
        action_features = self.action_layer(action)
        combined = torch.cat([state_features, action_features], dim=1)
        return self.combined_layers(combined)

class DDPG:
    """DDPG Agent for AUV control using PyTorch"""
    def __init__(self, actor_state_dim, critic_state_dim, action_dim, action_bound, config , device="cuda" if torch.cuda.is_available() else "cpu"):
        self.actor_state_dim = actor_state_dim
        self.critic_state_dim = critic_state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device
        
        # Initialize actor and critic networks with separate state dimensions
        self.actor = Actor(actor_state_dim, action_dim, action_bound, config).to(device)
        self.actor_target = Actor(actor_state_dim, action_dim, action_bound, config).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(critic_state_dim, action_dim, config).to(device)
        self.critic_target = Critic(critic_state_dim, action_dim, config).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        # Initialize replay buffer (modified to store both actor and critic states)
        self.buffer = ReplayBuffer(actor_state_dim, critic_state_dim)
        
        # Initialize noise process
        self.noise = OUActionNoise(
            mean=np.zeros(action_dim),
            std_deviation=0.2 * np.ones(action_dim)
        )
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001  # Target network update rate (0.01 for depth only)
    
    def get_action(self, actor_state, add_noise=True):
        """Return action for given actor state"""
        state_tensor = torch.FloatTensor(actor_state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            noise = self.noise()
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
        
        return action
    
    def remember(self, actor_state, critic_state, action, reward, next_actor_state, next_critic_state, done):
        """Store experience in replay buffer with separate states for actor and critic"""
        self.buffer.add(actor_state, critic_state, action, reward, next_actor_state, next_critic_state, done)
    
    def learn(self):
        """Update actor and critic networks from replay buffer"""
        if self.buffer.size() < self.buffer.batch_size:
            return None, None
        
        # Sample a batch from replay buffer
        actor_states, critic_states, actions, rewards, next_actor_states, next_critic_states, dones = self.buffer.sample()
        
        # Move tensors to device
        actor_states = actor_states.to(self.device)
        critic_states = critic_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_actor_states = next_actor_states.to(self.device)
        next_critic_states = next_critic_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_actor_states)
            next_q_values = self.critic_target(next_critic_states, next_actions)
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
            # print("Target Q Values!!!", target_q)
        
        current_q = self.critic(critic_states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor using deterministic policy gradient
        actions_pred = self.actor(actor_states)
        actor_loss = -self.critic(critic_states, actions_pred).mean()
        
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
    """DDPG Agent for AUV control integrated with ROS2 (inference only)"""
    def __init__(self, config):
        super().__init__('ddpg_auv_control')

        self.config = config

        self.actor_state_dim = 8
        self.critic_state_dim = 10
        self.action_dim = 6
        self.action_bound = 1.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        self.agent = DDPG(self.actor_state_dim, self.critic_state_dim, self.action_dim, self.action_bound, config, device=self.device)

        # ROS2 publishers
        self.thruster_pubs = {
            'heave_bow': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_bow', 1),
            'heave_stern': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_stern', 1),
            'surge_port': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_port', 1),
            'surge_starboard': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_starboard', 1),
            'port_servo': self.create_publisher(Float64, '/race2_auv/control/surge_port_servo', 1),
            'starboard_servo': self.create_publisher(Float64, '/race2_auv/control/surge_starboard_servo', 1)
        }

        self.create_subscription(ControlProcess, '/race2_auv/controller/process/value', self.state_callback, 10)
        self.create_subscription(ControlProcess, '/race2_auv/controller/process/setpoint', self.setpoint_callback, 10)
        self.create_subscription(ControlProcess, '/race2_auv/controller/process/error', self.error_callback, 10)
        self.create_subscription(Float64, '/race2_auv/control/surge_port_servo', self.update_joint_port, 10)
        self.create_subscription(Float64, '/race2_auv/control/surge_starboard_servo', self.update_joint_starboard, 10)
        self.create_subscription(Float64, '/race2_auv/control/thruster/heave_bow', self.update_thrust_heave_bow, 10)
        self.create_subscription(Float64, '/race2_auv/control/thruster/surge_port', self.update_thrust_surge_port, 10)
        self.create_subscription(Float64, '/race2_auv/control/thruster/surge_starboard', self.update_thrust_surge_starboard, 10)
        self.create_subscription(Float64, '/race2_auv/control/thruster/heave_stern', self.update_thrust_heave_stern, 10)

        self.declare_parameter('training_mode', False)  # Set to False for inference
        self.declare_parameter('model_path', 'ddpg_auv_model_ep1600.pt')

        self.training_mode = self.get_parameter('training_mode').value
        model_path = self.get_parameter('model_path').value

        # State tracking
        self.position_state = np.zeros(3)
        self.orientation_state = np.zeros(3)
        self.v_state = np.zeros(3)
        self.omega_ref_state = np.zeros(3)

        self.position_setpoint = np.zeros(3)
        self.orientation_setpoint = np.zeros(3)
        self.v_setpoint = np.zeros(3)
        self.omega_ref_setpoint = np.zeros(3)

        self.position_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)

        self.joint_angles_port = 0.0
        self.joint_angles_starboard = 0.0

        self.thrust_heave_bow = 0.0
        self.thrust_surge_port = 0.0
        self.thrust_surge_starboard = 0.0
        self.thrust_heave_stern = 0.0

        # Load model weights
        if model_path:
            try:
                self.agent.load_weights(model_path)
                self.get_logger().info(f"Loaded model from: {model_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load model from: {model_path}. Error: {e}")

        self.timer = self.create_timer(1/20, self.control_loop)

    def state_callback(self, data):
        """Process state updates from sensors"""
        # Extract state values
        self.position_state = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_state = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_state = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_state = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])

    def setpoint_callback(self, data):
        """Process setpoint updates"""
        self.position_setpoint = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_setpoint = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_setpoint = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_setpoint = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])

    def error_callback(self, data):
        """Process error updates"""
        self.position_err = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_err = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])

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

    def update_thrust_heave_stern(self, data):
        self.thrust_heave_stern = data.data

    def convert_servo_command_to_radians(self, normalized_command, min_angle_rad=-0.5, max_angle_rad=0.5):
        """
        Convert normalized servo command [-1, 1] to radians within specified min/max range
        """
        normalized_command = np.clip(normalized_command, -1.0, 1.0)
        min_angle_rad = self.config['min_servo_angle_rad']
        max_angle_rad = self.config['max_servo_angle_rad']
        angle_rad = min_angle_rad + (normalized_command + 1.0) * (max_angle_rad - min_angle_rad) / 2.0
        return angle_rad
    
    def publish_action(self, action):
        """Publish actions to ROS2 topics"""
        thruster_cmds = action[:4]
        servo_angles_normalized = action[4:]    

        servo_angles_rad = [
            self.convert_servo_command_to_radians(servo_angles_normalized[0]),
            self.convert_servo_command_to_radians(servo_angles_normalized[1])
        ]

        thruster_mapping = [
            ('heave_bow', thruster_cmds[2]),
            ('heave_stern', thruster_cmds[3]),
            ('surge_port', thruster_cmds[0]),
            ('surge_starboard', thruster_cmds[1]),
            ('port_servo', servo_angles_rad[0]),
            ('starboard_servo', servo_angles_rad[1])
        ]

        for name, value in thruster_mapping:
            msg = Float64()
            msg.data = float(value)
            self.thruster_pubs[name].publish(msg)

    def control_loop(self):
        """Control loop for inference only"""
        actor_state = np.concatenate([
            self.position_state[2:3],
            self.v_state[2:3],
            self.orientation_state[:3],
            self.omega_ref_state[:3]
        ])

        action = self.agent.get_action(actor_state, add_noise=False)  # No noise during inference
        self.publish_action(action)


def main(args=None):
    rclpy.init(args=args)

    # Set the path to the config file in the parent config directory
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_ddpg.yaml')

    # Load the configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ddpg_ros2_node = DDPG_ROS2(config)

    rclpy.spin(ddpg_ros2_node)

    ddpg_ros2_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
