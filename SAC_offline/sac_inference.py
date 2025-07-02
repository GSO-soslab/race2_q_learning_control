#!/usr/bin/env python3

import torch
import numpy as np
import yaml
import argparse
import os
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32, Header
from geometry_msgs.msg import Vector3
from mvp_msgs.msg import ControlProcess
from sensor_msgs.msg import Imu
import random


class ActorNetwork(torch.nn.Module):
    """SAC Actor Network - embedded to avoid import issues"""
    
    def __init__(self, state_dim, action_dim, hidden_layers=[256, 256], activation='relu', dropout=0.0):
        super(ActorNetwork, self).__init__()
        
        # Get activation function
        activation_fn = self._get_activation(activation)
        
        # Build hidden layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.network = torch.nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean_layer = torch.nn.Linear(input_dim, action_dim)
        self.log_std_layer = torch.nn.Linear(input_dim, action_dim)
        
        # Action bounds for AUV thrusters/servos
        self.action_scale = 1.0
        self.action_bias = 0.0
    
    def _get_activation(self, activation):
        """Get activation function by name"""
        activations = {
            'relu': torch.nn.ReLU,
            'tanh': torch.nn.Tanh,
            'elu': torch.nn.ELU,
            'leaky_relu': torch.nn.LeakyReLU,
            'swish': torch.nn.SiLU
        }
        return activations.get(activation.lower(), torch.nn.ReLU)
    
    def forward(self, state):
        x = self.network(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        
        # Enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


class SimpleAUVROS2Node(Node):
    """Simplified ROS2 node for SAC inference - no external dependencies"""
    
    def __init__(self, config):
        super().__init__('sac_inference_node')
        
        self.config = config
        
        # Initialize state variables
        self.position_state = np.zeros(3)
        self.orientation_state = np.zeros(3)
        self.v_state = np.zeros(3)
        self.omega_ref_state = np.zeros(3)
        
        # Initialize error variables
        self.position_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)
        self.linear_acceleration = np.zeros(3)
        
        # Create publishers for thrusters
        self.thruster_pubs = {
            'heave_bow': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_bow', 1),
            'heave_stern': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_stern', 1),
            'surge_port': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_port', 1),
            'surge_starboard': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_starboard', 1),
            'port_servo': self.create_publisher(Float64, '/race2_auv/control/surge_port_servo', 1),
            'starboard_servo': self.create_publisher(Float64, '/race2_auv/control/surge_starboard_servo', 1)
        }
        
        # Create setpoint publisher
        self.setpoint_publisher = self.create_publisher(
            ControlProcess,
            '/race2_auv/controller/process/set_point',
            10
        )
        
        # Create setpoint timer
        self.setpoint_timer = self.create_timer(0.2, self.publish_current_setpoint_callback)
        
        # Create subscribers
        self.create_subscription(ControlProcess,  
                                '/race2_auv/controller/process/value',
                                self.state_callback, 2)
        
        self.create_subscription(ControlProcess, 
                                '/race2_auv/controller/process/error', 
                                self.error_callback, 1)
        
        self.create_subscription(Imu, '/race2_auv/imu/data', self.imu_callback, 1)
        
        # Tracking variables
        self.new_state_available = False
        self.new_error_available = False
        self.last_action_timestamp = 0
        self.last_state_timestamp = 0
        self.last_error_timestamp = 0
        
        # Current setpoint and tracking
        self.current_setpoint = None
        self.setpoint_count = 0
        self.last_setpoint_time = 0
        
        # Setpoint ranges from config
        setpoint_config = self.config.get('setpoint', {})
        self.pos_z_range = setpoint_config.get('pos_z_range', [1.0, 8.0])
        self.ori_z_range = setpoint_config.get('ori_z_range', [-2.14, 2.14])
        self.ori_y_range = setpoint_config.get('ori_y_range', [-0.0, 0.0])
        self.vel_x_range = setpoint_config.get('vel_x_range', [-0.2, 0.2])
        
        # Dynamic setpoint settings
        inference_config = self.config.get('inference', {})
        self.setpoint_change_interval = inference_config.get('setpoint_change_interval_sec', 30.0)
        self.auto_change_setpoints = inference_config.get('auto_change_setpoints', True)
    
    def publish_current_setpoint_callback(self):
        """Continuously publish the current setpoint"""
        if self.current_setpoint is not None:
            self.current_setpoint.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_publisher.publish(self.current_setpoint)
    
    # Check if setpoint should change
    def should_change_setpoint(self):
        """Check if it's time to change setpoint"""
        if not self.auto_change_setpoints:
            return False
        
        current_time = time.time()
        if self.last_setpoint_time == 0:
            return True  # First setpoint
        
        return (current_time - self.last_setpoint_time) >= self.setpoint_change_interval
    
    def publish_new_setpoint(self):
        """Generate and publish a new setpoint"""
        msg = ControlProcess()
        msg.header = Header()
        msg.header.frame_id = "race2_auv/world_ned"
        msg.child_frame_id = "race2_auv/cg_link"
        msg.control_mode = "4dof"
        
        # Generate random setpoint
        msg.position = Vector3(
            x=0.0, 
            y=0.0, 
            z=random.uniform(self.pos_z_range[0], self.pos_z_range[1])
        )
        msg.orientation = Vector3(
            x=3.14, 
            y=random.uniform(self.ori_y_range[0], self.ori_y_range[1]), 
            z=random.uniform(self.ori_z_range[0], self.ori_z_range[1])
        )
        msg.velocity = Vector3(
            x=random.uniform(self.vel_x_range[0], self.vel_x_range[1]), 
            y=0.0, 
            z=0.0
        )
        msg.angular_rate = Vector3(x=0.0, y=0.0, z=0.0)
        
        self.current_setpoint = msg
        # Track setpoint changes
        self.last_setpoint_time = time.time()
        self.setpoint_count += 1
        
        msg.header.stamp = self.get_clock().now().to_msg()
        self.setpoint_publisher.publish(msg)
        
        return msg
    
    def get_current_setpoint_info(self):
        """Get human-readable info about current setpoint"""
        if self.current_setpoint is None:
            return "No setpoint generated yet"
        
        # Include setpoint count
        return (f"Setpoint #{self.setpoint_count} - pos_z: {self.current_setpoint.position.z:.2f}, "
                f"ori_z: {self.current_setpoint.orientation.z:.2f}, "
                f"ori_y: {self.current_setpoint.orientation.y:.2f}, "
                f"vel_x: {self.current_setpoint.velocity.x:.2f}")
    
    def state_callback(self, data):
        """Process state updates"""
        self.last_state_timestamp = time.time()
        self.position_state = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_state = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_state = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_state = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
        
        if hasattr(self, 'last_action_timestamp') and self.last_state_timestamp > self.last_action_timestamp:
            self.new_state_available = True
    
    def error_callback(self, data):
        """Process error updates"""
        self.last_error_timestamp = time.time()
        self.position_err = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_err = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
        
        if hasattr(self, 'last_action_timestamp') and self.last_error_timestamp > self.last_action_timestamp:
            self.new_error_available = True
    
    def imu_callback(self, data):
        """Process IMU data"""
        self.linear_acceleration[0] = data.linear_acceleration.x
        self.linear_acceleration[1] = data.linear_acceleration.y
        self.linear_acceleration[2] = data.linear_acceleration.z
    
    def publish_action(self, action):
        """Publish actions to ROS2 topics"""
        heave_bow = action[0]
        heave_stern = action[1]
        surge_port = action[2]
        surge_starboard = action[3]
        
        # Publish thruster commands
        self.thruster_pubs['heave_bow'].publish(Float64(data=float(heave_bow)))
        self.thruster_pubs['heave_stern'].publish(Float64(data=float(heave_stern)))
        self.thruster_pubs['surge_port'].publish(Float64(data=float(surge_port)))
        self.thruster_pubs['surge_starboard'].publish(Float64(data=float(surge_starboard)))
        
        # Publish servo commands if available
        if len(action) > 4:
            self.thruster_pubs['port_servo'].publish(Float64(data=float(action[4])))
        if len(action) > 5:
            self.thruster_pubs['starboard_servo'].publish(Float64(data=float(action[5])))
        
        self.last_action_timestamp = time.time()
        self.new_state_available = False
        self.new_error_available = False


class OfflineSACInference:
    """Inference class for trained SAC model with direct ROS2/Stonefish interaction"""
    
    def __init__(self, model_path, config_path=None, device='auto', control_frequency=10.0):
        self.device = self._setup_device(device)
        self.logger = self._setup_logging()
        
        # Control frequency (should match training data frequency)
        # Priority: command line arg > config file > default (10 Hz)
        config = self._load_config(config_path)
        config_frequency = config.get('inference', {}).get('control_frequency', 10.0)
        self.control_frequency = control_frequency if control_frequency != 10.0 else config_frequency
        self.control_period = 1.0 / self.control_frequency
        self.logger.info(f"Control frequency set to: {self.control_frequency} Hz ({self.control_period*1000:.0f}ms period)")
        
        # Log source of frequency setting
        if control_frequency != 10.0:
            self.logger.info(f"  Source: Command line argument (--frequency {control_frequency})")
        elif 'inference' in config and 'control_frequency' in config['inference']:
            self.logger.info(f"  Source: Config file ({config_frequency} Hz)")
        else:
            self.logger.info(f"  Source: Default (10 Hz to match typical PID training data)")
        
        # Load the trained model checkpoint
        self.logger.info(f"Loading model from: {model_path}")
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration and parameters
        self.config = self.checkpoint.get('config', config)
        self.state_dim = self.checkpoint['state_dim']
        self.action_dim = self.checkpoint['action_dim']
        self.normalization_params = self.checkpoint.get('normalization_params', {})
        
        # Initialize and load the trained actor network
        self._load_actor()
        
        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init(args=None)
        
        # Initialize the simplified AUV ROS2 node (no import dependencies)
        self.logger.info("Initializing simplified AUV ROS2 node for inference...")
        self.node = SimpleAUVROS2Node(self.config)
        
        # Inference tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_history = []
        self.state_history = []
        
        self.logger.info(f"✅ SAC Inference initialized")
        self.logger.info(f"   Model: {os.path.basename(model_path)}")
        self.logger.info(f"   State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"   Device: {self.device}")
        
        # Log setpoint settings
        inference_config = self.config.get('inference', {})
        if inference_config.get('auto_change_setpoints', True):
            self.logger.info(f"   Setpoint changes: Every {inference_config.get('setpoint_change_interval_sec', 30.0)}s")
        else:
            self.logger.info(f"   Setpoint changes: Disabled")
    
    def _setup_device(self, device):
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('SAC_Inference')
    
    def _load_config(self, config_path):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return minimal default config with NEW setpoint settings
            return {
                'setpoint': {
                    'pos_z_range': [1.0, 8.0],
                    'ori_z_range': [-2.14, 2.14],
                    'ori_y_range': [-0.0, 0.0],
                    'vel_x_range': [-0.25, 0.25]
                },
                'inference': {
                    'control_frequency': 10.0,
                    'max_episode_steps': 500,
                    'default_deterministic': True,
                    'timeout_multiplier': 1.5,
                    'setpoint_change_interval_sec': 30.0,
                    'auto_change_setpoints': True
                }
            }
    
    def _load_actor(self):
        """Load the trained actor network"""
        model_config = self.config.get('model', {})
        actor_config = model_config.get('actor', {})
        
        self.actor = ActorNetwork(
            self.state_dim,
            self.action_dim,
            hidden_layers=actor_config.get('hidden_layers', [256, 256]),
            activation=actor_config.get('activation', 'relu'),
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        # Load the trained weights
        self.actor.load_state_dict(self.checkpoint['actor_state_dict'])
        self.actor.eval()  # Set to evaluation mode
        
        self.logger.info("✅ Actor network loaded and ready for inference")
    
    def predict_action(self, state, deterministic=True):
        """Predict action for a given state using the trained actor"""
        with torch.no_grad():
            # Convert to tensor if needed
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state)
            else:
                state_tensor = torch.FloatTensor(np.array(state))
            
            # Add batch dimension if needed
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            state_tensor = state_tensor.to(self.device)
            
            if deterministic:
                # Use mean action (deterministic policy)
                mean, _ = self.actor.forward(state_tensor)
                action = torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
            else:
                # Sample from policy distribution (stochastic policy)
                action, _, _ = self.actor.sample(state_tensor)
            
            # Convert back to numpy and remove batch dimension
            return action.cpu().numpy()[0]
    
    def _get_current_observation(self):
        """Get current observation from ROS2 node state"""
        # Same observation construction as in AUVEnv._step_online_mode
        depth = self.node.position_state[2:3]
        depth_error = self.node.position_err[2:3]
        surge_velocity_error = self.node.v_err[0:1]
        sway_velocity_error = self.node.v_err[1:2]
        
        roll_sin_err = np.array([np.sin(self.node.orientation_err[0])])
        roll_cos_err = np.array([np.cos(self.node.orientation_err[0])])
        pitch_sin_err = np.array([np.sin(self.node.orientation_err[1])])
        pitch_cos_err = np.array([np.cos(self.node.orientation_err[1])])
        yaw_sin_err = np.array([np.sin(self.node.orientation_err[2])])
        yaw_cos_err = np.array([np.cos(self.node.orientation_err[2])])
        
        roll_sin_current = np.array([np.sin(self.node.orientation_state[0])])
        roll_cos_current = np.array([np.cos(self.node.orientation_state[0])])
        pitch_sin_current = np.array([np.sin(self.node.orientation_state[1])])
        pitch_cos_current = np.array([np.cos(self.node.orientation_state[1])])
        yaw_sin_current = np.array([np.sin(self.node.orientation_state[2])])
        yaw_cos_current = np.array([np.cos(self.node.orientation_state[2])])
        
        surge_velocity = self.node.v_state[0:1]
        sway_velocity = self.node.v_state[1:2]
        heave_velocity = self.node.v_state[2:3]
        
        roll_rate = self.node.omega_ref_state[0:1]
        pitch_rate = self.node.omega_ref_state[1:2]
        yaw_rate = self.node.omega_ref_state[2:3]
        
        x_acceleration = self.node.linear_acceleration[0:1]
        y_acceleration = self.node.linear_acceleration[1:2]
        
        observation = np.concatenate([
            depth,
            depth_error,
            surge_velocity_error,
            sway_velocity_error,
            roll_sin_err,
            roll_cos_err,
            pitch_sin_err,
            pitch_cos_err,
            yaw_sin_err,
            yaw_cos_err,
            roll_sin_current,
            roll_cos_current,
            pitch_sin_current,
            pitch_cos_current,
            yaw_sin_current,
            yaw_cos_current,
            surge_velocity,
            sway_velocity,
            heave_velocity,
            roll_rate,
            pitch_rate,
            yaw_rate,
            x_acceleration,
            y_acceleration,
        ])
        
        return observation
    
    def _spin_node(self, timeout_sec=0.1):
        """Process ROS callbacks for a limited time"""
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.01)
    
    def _wait_for_initial_data(self, timeout_sec=5.0):
        """Wait for initial state and error data from ROS2"""
        self.logger.info("Waiting for initial ROS2 data...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            self._spin_node(timeout_sec=0.1)
            
            # Check if we have both state and error data
            if (hasattr(self.node, 'position_state') and 
                hasattr(self.node, 'position_err') and
                self.node.last_state_timestamp > 0 and 
                self.node.last_error_timestamp > 0):
                self.logger.info("✅ Initial ROS2 data received")
                return True
        
        self.logger.warning("⚠️  Timeout waiting for initial ROS2 data")
        return False
    
    # NEW ADDITION: Continuous operation mode
    def run_continuous(self, deterministic=None, verbose=True):
        """Run continuous control with dynamic setpoints (runs until interrupted)"""
        
        # Use config defaults if not specified
        inference_config = self.config.get('inference', {})
        if deterministic is None:
            deterministic = inference_config.get('default_deterministic', True)
        
        policy_type = "deterministic" if deterministic else "stochastic"
        self.logger.info(f"Starting continuous SAC control...")
        self.logger.info(f"  Policy: {policy_type}")
        if self.node.auto_change_setpoints:
            self.logger.info(f"  Setpoint changes: Every {self.node.setpoint_change_interval}s")
        else:
            self.logger.info(f"  Setpoint changes: Disabled")
        self.logger.info(f"  Press Ctrl+C to stop")
        
        # Wait for initial data
        if not self._wait_for_initial_data():
            self.logger.error("Failed to get initial ROS2 data")
            return None
        
        # Generate initial setpoint
        setpoint = self.node.publish_new_setpoint()
        self.logger.info(f"Initial setpoint: {self.node.get_current_setpoint_info()}")
        
        # Initialize tracking
        total_steps = 0
        step_rewards = []
        start_time = time.time()
        last_log_time = start_time
        log_interval = 10.0  # Log every 10 seconds
        
        try:
            # Main control loop - runs until interrupted
            while True:
                # NEW: Check if we should change setpoint
                if self.node.should_change_setpoint():
                    old_info = self.node.get_current_setpoint_info()
                    setpoint = self.node.publish_new_setpoint()
                    new_info = self.node.get_current_setpoint_info()
                    self.logger.info(f"🎯 Setpoint changed: {new_info}")
                
                # Get current observation from ROS2 node
                try:
                    obs = self._get_current_observation()
                except Exception as e:
                    self.logger.warning(f"Failed to get observation: {e}")
                    self._spin_node(timeout_sec=0.1)
                    continue
                
                # Predict action using trained actor
                action = self.predict_action(obs, deterministic=deterministic)
                
                # Publish action directly to ROS2 topics
                self.node.publish_action(action)
                
                # Wait for new state/error data after action
                timeout_multiplier = self.config.get('inference', {}).get('timeout_multiplier', 1.5)
                timeout_sec = self.control_period * timeout_multiplier
                start_wait_time = time.time()
                
                while not (self.node.new_state_available and self.node.new_error_available):
                    self._spin_node(timeout_sec=0.01)
                    if time.time() - start_wait_time > timeout_sec:
                        self.logger.warning("Timeout waiting for state/error updates")
                        break
                
                # Calculate reward (simplified version)
                if self.node.new_state_available and self.node.new_error_available:
                    # Get state error for reward calculation
                    state_error_array = np.concatenate([
                        self.node.position_err[2:3],  # depth error
                        self.node.v_err[0:2],  # surge, sway velocity errors
                        self.node.v_err[2:3],  # heave velocity error
                        self.node.orientation_err[:3]  # roll, pitch, yaw errors
                    ])
                    
                    # Simple quadratic penalty reward
                    reward = -np.sum(state_error_array ** 2)
                    step_rewards.append(reward)
                    total_steps += 1
                    
                    # Periodic logging
                    current_time = time.time()
                    if current_time - last_log_time >= log_interval:
                        elapsed_time = current_time - start_time
                        avg_reward = np.mean(step_rewards[-100:])  # Last 100 steps
                        
                        self.logger.info(f"📊 Step {total_steps} | "
                                       f"Time: {elapsed_time:.1f}s | "
                                       f"Avg Reward: {avg_reward:.4f} | "
                                       f"Setpoints: {self.node.setpoint_count}")
                        last_log_time = current_time
                else:
                    self.logger.warning("No new state/error data available")
                
                # Match training frequency
                time.sleep(self.control_period)
        
        except KeyboardInterrupt:
            self.logger.info("🛑 Control interrupted by user")
        
        # Final summary
        total_time = time.time() - start_time
        self.logger.info(f"📈 Continuous control completed:")
        self.logger.info(f"   Total time: {total_time:.1f}s")
        self.logger.info(f"   Total steps: {total_steps}")
        self.logger.info(f"   Setpoint changes: {self.node.setpoint_count}")
        if step_rewards:
            self.logger.info(f"   Average reward: {np.mean(step_rewards):.4f}")
        
        return {
            'total_time': total_time,
            'total_steps': total_steps,
            'setpoint_changes': self.node.setpoint_count,
            'rewards': np.array(step_rewards) if step_rewards else np.array([]),
        }
    
    def run_episode(self, max_steps=None, deterministic=None, verbose=True):
        """Run a single episode using direct ROS2 interaction"""
        
        # Use config defaults if not specified
        inference_config = self.config.get('inference', {})
        if max_steps is None:
            max_steps = inference_config.get('max_episode_steps', 500)
        if deterministic is None:
            deterministic = inference_config.get('default_deterministic', True)
        
        self.logger.info(f"Starting episode with trained SAC policy...")
        self.logger.info(f"  Max steps: {max_steps}, Deterministic: {deterministic}")
        
        # Generate and publish new setpoint for this episode
        setpoint = self.node.publish_new_setpoint()
        setpoint_info = self.node.get_current_setpoint_info()
        self.logger.info(f"Episode setpoint: {setpoint_info}")
        
        # Wait for initial data
        if not self._wait_for_initial_data():
            self.logger.error("Failed to get initial ROS2 data")
            return None
        
        # Initialize episode tracking
        episode_reward = 0.0
        episode_steps = 0
        episode_actions = []
        episode_states = []
        episode_rewards = []
        
        # Episode loop
        while episode_steps < max_steps:
            # Get current observation from ROS2 node
            try:
                obs = self._get_current_observation()
                episode_states.append(obs.copy())
            except Exception as e:
                self.logger.warning(f"Failed to get observation: {e}")
                self._spin_node(timeout_sec=0.1)
                continue
            
            # Predict action using trained actor
            action = self.predict_action(obs, deterministic=deterministic)
            episode_actions.append(action.copy())
            
            # Publish action directly to ROS2 topics
            self.node.publish_action(action)
            
            # Wait for new state/error data after action (adaptive timeout based on frequency)
            timeout_multiplier = self.config.get('inference', {}).get('timeout_multiplier', 1.5)
            timeout_sec = self.control_period * timeout_multiplier  # e.g., 0.1s * 1.5 = 0.15s
            start_time = time.time()
            
            while not (self.node.new_state_available and self.node.new_error_available):
                self._spin_node(timeout_sec=0.01)  # Reduced from 0.1 to 0.01 for faster spinning
                if time.time() - start_time > timeout_sec:
                    self.logger.warning("Timeout waiting for state/error updates")
                    break
            
            # Calculate reward (simplified version, you can use your full reward function)
            if self.node.new_state_available and self.node.new_error_available:
                # Get state error for reward calculation
                state_error_array = np.concatenate([
                    self.node.position_err[2:3],  # depth error
                    self.node.v_err[0:2],  # surge, sway velocity errors
                    self.node.v_err[2:3],  # heave velocity error
                    self.node.orientation_err[:3]  # roll, pitch, yaw errors
                ])
                
                # Simple quadratic penalty reward (you can replace with your full reward function)
                reward = -np.sum(state_error_array ** 2)
                
                episode_reward += reward
                episode_rewards.append(reward)
                episode_steps += 1
                
                # Logging
                if verbose and episode_steps % 50 == 0:
                    self.logger.info(f"  Step {episode_steps}: Action={action}, Reward={reward:.4f}")
            else:
                self.logger.warning("No new state/error data available")
                break
            
            # Match training frequency: use configured control period
            time.sleep(self.control_period)  # Default: 0.1s for 10 Hz to match PID training data
        
        # Episode summary
        self.logger.info(f"Episode completed:")
        self.logger.info(f"  Steps: {episode_steps}")
        self.logger.info(f"  Total reward: {episode_reward:.4f}")
        self.logger.info(f"  Average reward: {episode_reward/episode_steps:.4f}")
        self.logger.info(f"  Setpoint: {setpoint_info}")
        
        # Store episode data
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_steps)
        self.action_history.append(np.array(episode_actions))
        self.state_history.append(np.array(episode_states))
        
        return {
            'total_reward': episode_reward,
            'steps': episode_steps,
            'actions': np.array(episode_actions),
            'states': np.array(episode_states),
            'rewards': np.array(episode_rewards),
            'setpoint_info': setpoint_info,
            'terminated': False,
            'truncated': episode_steps >= max_steps
        }
    
    def run_multiple_episodes(self, num_episodes=5, max_steps=500, deterministic=True):
        """Run multiple episodes for evaluation"""
        
        self.logger.info(f"Running {num_episodes} episodes for evaluation...")
        
        episode_results = []
        
        for episode in range(num_episodes):
            self.logger.info(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            result = self.run_episode(
                max_steps=max_steps,
                deterministic=deterministic,
                verbose=True
            )
            
            if result is not None:
                episode_results.append(result)
            else:
                self.logger.error(f"Episode {episode + 1} failed")
            
            # Pause between episodes
            time.sleep(2.0)
        
        if not episode_results:
            self.logger.error("No successful episodes completed")
            return None, None
        
        # Calculate summary statistics
        rewards = [r['total_reward'] for r in episode_results]
        lengths = [r['steps'] for r in episode_results]
        
        summary = {
            'num_episodes': len(episode_results),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'success_rate': sum(1 for r in episode_results if not r['truncated']) / len(episode_results)
        }
        
        # Print summary
        self.logger.info(f"\n🎯 Evaluation Summary ({len(episode_results)} episodes):")
        self.logger.info(f"  Average reward: {summary['avg_reward']:.3f} ± {summary['std_reward']:.3f}")
        self.logger.info(f"  Reward range: [{summary['min_reward']:.3f}, {summary['max_reward']:.3f}]")
        self.logger.info(f"  Average length: {summary['avg_length']:.1f} ± {summary['std_length']:.1f}")
        self.logger.info(f"  Success rate: {summary['success_rate']:.1%}")
        
        return episode_results, summary
    
    def save_results(self, episode_results, summary, output_dir=None):
        """Save evaluation results and plots"""
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"inference_results_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary statistics
        summary_file = os.path.join(output_dir, 'evaluation_summary.yaml')
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        rewards = [r['total_reward'] for r in episode_results]
        axes[0, 0].bar(range(1, len(rewards) + 1), rewards)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards (Trained SAC Policy)')
        axes[0, 0].grid(True, axis='y')
        
        # Episode lengths
        lengths = [r['steps'] for r in episode_results]
        axes[0, 1].bar(range(1, len(lengths) + 1), lengths, color='orange')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Length (steps)')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True, axis='y')
        
        # Action history for last episode
        if episode_results:
            last_actions = episode_results[-1]['actions']
            for i in range(min(4, last_actions.shape[1])):  # Plot first 4 action dimensions
                axes[1, 0].plot(last_actions[:, i], label=f'Action {i}')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Action Value')
            axes[1, 0].set_title('Action History (Last Episode)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Summary statistics text
        axes[1, 1].axis('off')
        summary_text = f"""Evaluation Results:
Episodes: {summary['num_episodes']}
Avg Reward: {summary['avg_reward']:.3f} ± {summary['std_reward']:.3f}
Avg Length: {summary['avg_length']:.1f} ± {summary['std_length']:.1f}
Success Rate: {summary['success_rate']:.1%}

Model Info:
State Dim: {self.state_dim}
Action Dim: {self.action_dim}
Device: {self.device}

Policy: Trained SAC (Offline → Online)
Environment: Direct ROS2 + Stonefish
Interface: Simplified ROS2 (no dependencies)"""
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'evaluation_plots.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Results saved to: {output_dir}")
        self.logger.info(f"   Summary: {summary_file}")
        self.logger.info(f"   Plots: {plot_file}")
        
        return output_dir
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'node'):
            self.node.destroy_node()


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='SAC Inference with Dynamic Setpoints')
    parser.add_argument('model_path', help='Path to trained SAC model (.pth file)')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    # Operation modes
    parser.add_argument('--continuous', action='store_true', default=True,
                       help='Run continuous control (default mode)')
    parser.add_argument('--episodes', type=int, default=0, 
                       help='Number of episodes to run (0 = continuous mode)')
    parser.add_argument('--max-steps', type=int, default=500, 
                       help='Maximum steps per episode (only for episode mode)')
    
    # Control settings
    parser.add_argument('--deterministic', action='store_true', default=True, 
                       help='Use deterministic policy (default: True)')
    parser.add_argument('--stochastic', action='store_true', 
                       help='Use stochastic policy (overrides --deterministic)')
    parser.add_argument('--frequency', type=float, default=10.0, 
                       help='Control frequency in Hz (default: 10 Hz)')
    
    # Setpoint settings
    parser.add_argument('--setpoint-interval', type=float, default=30.0,
                       help='Setpoint change interval in seconds (default: 30s)')
    parser.add_argument('--no-setpoint-changes', action='store_true',
                       help='Disable automatic setpoint changes')
    
    # Other options
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return 1
    
    # Determine operation mode
    continuous_mode = args.episodes == 0
    deterministic = args.deterministic and not args.stochastic
    policy_type = "deterministic" if deterministic else "stochastic"
    
    try:
        # Load base config and apply overrides
        base_config = {}
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                base_config = yaml.safe_load(f)
        
        # Override config with command line args
        if 'inference' not in base_config:
            base_config['inference'] = {}
        
        base_config['inference'].update({
            'control_frequency': args.frequency,
            'default_deterministic': deterministic,
            'setpoint_change_interval_sec': args.setpoint_interval,
            'auto_change_setpoints': not args.no_setpoint_changes,
        })
        
        print(f"🚀 Initializing SAC inference...")
        print(f"   Model: {args.model_path}")
        print(f"   Mode: {'Continuous' if continuous_mode else f'{args.episodes} episodes'}")
        print(f"   Policy: {policy_type}")
        if not args.no_setpoint_changes:
            print(f"   Setpoint changes: Every {args.setpoint_interval}s")
        else:
            print(f"   Setpoint changes: Disabled")
        
        # Save config temporarily
        temp_config_path = 'temp_inference_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(base_config, f)
        
        inference = OfflineSACInference(
            model_path=args.model_path,
            config_path=temp_config_path,
            device=args.device,
            control_frequency=args.frequency
        )
        
        if continuous_mode:
            print(f"\n🎮 Starting continuous control (Press Ctrl+C to stop)...")
            results = inference.run_continuous(deterministic=deterministic)
        else:
            print(f"\n🎮 Running {args.episodes} episodes...")
            results = inference.run_multiple_episodes(
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                deterministic=deterministic
            )
            
            if results:
                output_dir = inference.save_results(results[0], results[1], args.output_dir)
                print(f"\n✅ Results saved to: {output_dir}")
        
        # Cleanup
        os.remove(temp_config_path)
        inference.close()
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Inference interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())