#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from std_msgs.msg import Header, Float64
from geometry_msgs.msg import Vector3
from mvp_msgs.msg import ControlProcess


from geometry_msgs.msg import Pose, Twist


class OUActionNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
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
    def __init__(self, buffer_capacity=100000, batch_size=64):
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
    def __init__(self, state_dim, action_dim, action_bound, hidden_dims=[512, 256, 128]):
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
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 256, 128]):
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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer()
        
        # Initialize noise process
        self.noise = OUActionNoise(
            mean=np.zeros(action_dim),
            std_deviation=0.2 * np.ones(action_dim)
        )
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Target network update rate
        
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
    def __init__(self):
        super().__init__('ddpg_auv_control')
        
        # Define dimensions
        self.state_dim = 12  # 3D position + 3D orientation + 3D velocity + 3D angular rate
        self.action_dim = 6  # 4 thrusters + 2 servo angles
        self.action_bound = 1.0  # All commands between -1 and 1
        
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
        
        # ROS2 subscriber for AUV state
        self.state_sub = self.create_subscription(
            Pose,  # Replace with your actual AUVState message type
            '/auv/state',
            self.state_callback,
            10
        )
        
        # Training parameters
        self.declare_parameter('training_mode', True)
        self.declare_parameter('max_steps', 1000)
        self.declare_parameter('goal_position', [0.0, 0.0, 0.0])
        self.declare_parameter('goal_orientation', [0.0, 0.0, 0.0])
        self.declare_parameter('model_path', '')
        
        self.training_mode = self.get_parameter('training_mode').value
        self.max_steps = self.get_parameter('max_steps').value
        self.goal_pos = np.array(self.get_parameter('goal_position').value)
        self.goal_ori = np.array(self.get_parameter('goal_orientation').value)
        model_path = self.get_parameter('model_path').value
        
        # State tracking
        self.current_state = None
        self.prev_state = None
        self.prev_action = None
        
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
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz control loop
        
    def state_callback(self, msg):
        """Process incoming AUV state message"""
        # Extract data from ROS2 message
        # Note: Adjust this based on your actual message type structure
        position = np.array([msg.position.x, msg.position.y, msg.position.z])
        # Extract quaternion for orientation (you might need to convert to Euler angles)
        orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ])
        
        # If you have velocity and angular rate in a different message,
        # you'll need to subscribe to that as well
        # For now, we'll assume they're part of the state message or available elsewhere
        velocity = np.array([0.0, 0.0, 0.0])  # Replace with actual velocity data
        angular_rate = np.array([0.0, 0.0, 0.0])  # Replace with actual angular rate data
        
        # Combine into state vector
        self.current_state = np.concatenate([position, orientation, velocity, angular_rate])
        
    def control_loop(self):
        """Main control loop"""
        if self.current_state is not None:
            # Get action from agent
            action = self.agent.get_action(self.current_state, add_noise=self.training_mode)
            
            # Publish thruster and servo commands
            self.publish_action(action)
            
            # If in training mode, generate reward and train
            if self.training_mode and self.prev_state is not None and self.prev_action is not None:
                reward = self.calculate_reward(self.prev_state, self.current_state)
                self.episode_reward += reward
                done = self.is_done(self.current_state)
                
                # Store in replay buffer
                self.agent.remember(self.prev_state, self.prev_action, reward, self.current_state, done)
                
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
                    if self.episode_count % 10 == 0:
                        model_path = f"ddpg_auv_model_ep{self.episode_count}.pt"
                        self.agent.save_weights(model_path)
                        self.get_logger().info(f"Model saved to {model_path}")
            
            # Store state and action for training
            self.prev_state = self.current_state
            self.prev_action = action
        
    def calculate_reward(self, prev_state, current_state):
        """Calculate reward based on state transition"""
        # Extract position and orientation from states
        prev_pos = prev_state[:3]
        prev_ori = prev_state[3:6]
        curr_pos = current_state[:3]
        curr_ori = current_state[3:6]
        
        # Position error (negative distance to goal)
        pos_error_prev = -np.linalg.norm(prev_pos - self.goal_pos)
        pos_error_curr = -np.linalg.norm(curr_pos - self.goal_pos)
        
        # Orientation error
        ori_error_prev = -np.linalg.norm(prev_ori - self.goal_ori)
        ori_error_curr = -np.linalg.norm(curr_ori - self.goal_ori)
        
        # Reward is improvement in position and orientation
        pos_reward = (pos_error_curr - pos_error_prev) * 10  # Scale factor
        ori_reward = (ori_error_curr - ori_error_prev) * 5   # Scale factor
        
        # Penalize excessive velocity and angular rates
        vel_penalty = -0.1 * np.linalg.norm(current_state[6:9])
        ang_rate_penalty = -0.1 * np.linalg.norm(current_state[9:12])
        
        # Penalize excessive control effort
        control_penalty = -0.05 * np.linalg.norm(self.prev_action)
        
        # Sum all reward components
        reward = pos_reward + ori_reward + vel_penalty + ang_rate_penalty + control_penalty
        
        # Bonus for reaching goal
        if np.linalg.norm(curr_pos - self.goal_pos) < 0.5 and np.linalg.norm(curr_ori - self.goal_ori) < 0.2:
            reward += 100
            
        return reward
        
    def is_done(self, state):
        """Check if episode is complete"""
        # Extract position and orientation
        position = state[:3]
        orientation = state[3:6]
        
        # Check if goal reached
        pos_error = np.linalg.norm(position - self.goal_pos)
        ori_error = np.linalg.norm(orientation - self.goal_ori)
        
        # Episode complete if goal reached
        return pos_error < 0.5 and ori_error < 0.2
    
    def publish_action(self, action):
        """Publish actions to thruster and servo topics"""
        # Distribute the action vector to appropriate actuators
        # action[0-3] are thrusters, action[4-5] are servo angles
        
        # Create message objects
        heave_bow_msg = Float64()
        heave_bow_msg.data = float(action[0])
        
        heave_stern_msg = Float64()
        heave_stern_msg.data = float(action[1])
        
        surge_port_msg = Float64()
        surge_port_msg.data = float(action[2])
        
        surge_starboard_msg = Float64()
        surge_starboard_msg.data = float(action[3])
        
        port_servo_msg = Float64()
        port_servo_msg.data = float(action[4])
        
        starboard_servo_msg = Float64()
        starboard_servo_msg.data = float(action[5])
        
        # Publish thruster commands
        self.thruster_pubs['heave_bow'].publish(heave_bow_msg)
        self.thruster_pubs['heave_stern'].publish(heave_stern_msg)
        self.thruster_pubs['surge_port'].publish(surge_port_msg)
        self.thruster_pubs['surge_starboard'].publish(surge_starboard_msg)
        
        # Publish servo angle commands
        self.thruster_pubs['port_servo'].publish(port_servo_msg)
        self.thruster_pubs['starboard_servo'].publish(starboard_servo_msg)
        
        # Log action if debugging
        self.get_logger().debug(f"Action: Thrusters={action[:4]}, Servos={action[4:]}")

def main(args=None):
    rclpy.init(args=args)
    ddpg_ros = DDPG_ROS2()
    
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