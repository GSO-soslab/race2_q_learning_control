import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os, time, datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32
from geometry_msgs.msg import TwistStamped
from mvp_msgs.msg import ControlProcess
import yaml
from rclpy.clock import Clock

class AUVEnvNode(Node):
    """Node to handle ROS2 communications for the AUV environment"""
    
    def __init__(self, config):
        super().__init__('auv_env_node')
        
        self.config = config
        
        thruster_size = self.config['environment']['thruster_size']
        servo_size = self.config['environment']['servo_joints_size']
        
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
        
        # Initialize actuator variables
        self.joint_angles_port = 0.0
        self.joint_angles_starboard = 0.0
        self.joint_angles = np.zeros(servo_size)
        
        self.thrust_heave_bow = 0.0
        self.thrust_surge_port = 0.0
        self.thrust_surge_starboard = 0.0
        self.thrust_heave_stern = 0.0
        
        # Create publishers
        self.thruster_pubs = {
            'heave_bow': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_bow', 1),
            'heave_stern': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_stern', 1),
            'surge_port': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_port', 1),
            'surge_starboard': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_starboard', 1),
            'port_servo': self.create_publisher(Float64, '/race2_auv/control/surge_port_servo', 1),
            'starboard_servo': self.create_publisher(Float64, '/race2_auv/control/surge_starboard_servo', 1)
        }
        
        # Create subscribers
        self.create_subscription(ControlProcess,  
                                '/race2_auv/controller/process/value',
                                self.state_callback,
                                2)
        
        self.create_subscription(ControlProcess, 
                                '/race2_auv/controller/process/error', 
                                self.error_callback,
                                1)
        
        self.create_subscription(Float64, 
                                '/race2_auv/control/surge_port_servo', 
                                self.update_joint_port, 
                                1)
        
        self.create_subscription(Float64, 
                                '/race2_auv/control/surge_starboard_servo', 
                                self.update_joint_starboard, 
                                1)
        
        self.create_subscription(Float64, 
                                '/race2_auv/control/thruster/heave_bow', 
                                self.update_thrust_heave_bow, 
                                1)
        
        self.create_subscription(Float64, 
                                '/race2_auv/control/thruster/surge_port', 
                                self.update_thrust_surge_port, 
                                1)
        
        self.create_subscription(Float64, 
                                '/race2_auv/control/thruster/surge_starboard', 
                                self.update_thrust_surge_starboard, 
                                1)
        
        self.create_subscription(Float64, 
                                '/race2_auv/control/thruster/heave_stern', 
                                self.update_thrust_heave_stern, 
                                1)
        
        # For communication between callbacks and the environment
        self.new_state_available = False
        self.new_error_available = False
        self.last_action_timestamp = 0
        self.last_state_timestamp = 0
        self.last_error_timestamp = 0
        
        # Declare parameters
        self.declare_parameter('max_steps', 500)
        self.max_steps = self.get_parameter('max_steps').value
    
    def error_callback(self, data):
        """Process error updates"""
        self.get_logger().debug("Error callback triggered!")
        self.last_error_timestamp = time.time()
        self.position_err = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_err = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
        
        self.state_err = np.concatenate([
            self.position_err[2:3],
            self.v_err[:2],
            self.orientation_err[:3],
            self.omega_ref_err[:3],
        ])
        
        if hasattr(self, 'last_action_timestamp') and self.last_error_timestamp > self.last_action_timestamp:
            self.new_error_available = True
    
    def state_callback(self, data):
        """Process state updates from sensors"""
        self.get_logger().debug("State callback triggered!")
        # Add timestamp to the state observation
        self.last_state_timestamp = time.time()
        
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
        
        # Flag indicating we have a new state after the last action
        if hasattr(self, 'last_action_timestamp') and self.last_state_timestamp > self.last_action_timestamp:
            self.new_state_available = True
    
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
    
    def publish_action(self, action, num_thrusters, num_servos):
        """Publish actions to ROS2 topics"""
        # Split action into thruster commands and servo angles
        thruster_cmds = action[:num_thrusters]
        servo_angles_normalized = action[num_thrusters:] if len(action) > num_thrusters else []
        
        # Convert servo commands to radians if needed
        servo_angles_rad = []
        for angle in servo_angles_normalized:
            servo_angles_rad.append(self.convert_servo_command_to_radians(angle))
        
        # Map to appropriate publishers
        # All DOFs - modify as needed for your specific configuration
        thruster_mapping = [
            ('heave_bow', 0.8 * thruster_cmds[0]),
            ('heave_stern', 0.8 * thruster_cmds[1])
            # ('surge_port',  0.0 * thruster_cmds[0]),
            # ('surge_starboard', 0.0 * thruster_cmds[1]),
            # ('port_servo', 0.0 *servo_angles_rad[0]),
            # ('starboard_servo',0.0 * servo_angles_rad[1])
        ]
        
        # Add servo mappings if needed
        if len(servo_angles_rad) >= 2:
            thruster_mapping.extend([
                ('port_servo', servo_angles_rad[0]),
                ('starboard_servo', servo_angles_rad[1])
            ])
        
        # Publish commands
        for name, value in thruster_mapping:
            msg = Float64()
            msg.data = float(value)
            self.thruster_pubs[name].publish(msg)
            self.get_logger().debug(f"Published {value} to {name}")
        
        # Record action timestamp
        self.last_action_timestamp = time.time()
        
        # Reset the new state flag since we're waiting for a new state after this action
        self.new_state_available = False
        self.new_error_available = False
        
        return thruster_cmds, servo_angles_rad
    
    def convert_servo_command_to_radians(self, normalized_command):
        """
        Convert normalized servo command [-1, 1] to radians within specified min/max range
        """
        # Ensure normalized command is within [-1, 1]
        normalized_command = np.clip(normalized_command, -1.0, 1.0)
        
        min_angle_rad = self.config['min_servo_angle_rad']
        max_angle_rad = self.config['max_servo_angle_rad']
        # Map from [-1, 1] to [min_angle_rad, max_angle_rad]
        angle_rad = min_angle_rad + (normalized_command + 1.0) * (max_angle_rad - min_angle_rad) / 2.0
        
        return angle_rad


class AUVEnv(gym.Env):
    """Custom AUV Environment that follows gym interface"""
    
    def __init__(self):
        super(AUVEnv, self).__init__()
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config_ddpg.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init(args=None)
        
        # Create ROS2 node
        self.node = AUVEnvNode(self.config)
        
        # Set up action and observation spaces
        thruster_size = self.config['environment']['thruster_size']
        servo_size = self.config['environment']['servo_joints_size']
        
        self.num_thrusters = thruster_size
        self.num_servos = servo_size
        
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(thruster_size + servo_size,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(14,),  # state dimension
            dtype=np.float32
        )
        
        # Initialize episode-related variables
        self.episode_step = 0
        self.episode_reward = 0.0
        
        # Initialize history arrays for smoothness calculations
        self.joint_positions_history = np.zeros((10, self.num_servos))  # Store last 10 servo positions
        self.u_prev = np.zeros((10, self.num_thrusters))  # Store last 10 thruster commands
        self.thruster_command_action_prev = np.zeros((10, self.num_thrusters))  # Store last 10 thruster actions
        
        # Initialize tracking variables
        self.thruster_action = np.zeros(self.num_thrusters)
        self.joint_angles = np.zeros(self.num_servos)
        self.last_action = np.zeros(thruster_size + servo_size)
    
    def reset(self, seed=None):
        """Reset the environment to initial state and return the initial observation"""
        # Process ROS events to ensure we have the latest state
        self._spin_node(timeout_sec=0.1)

        if seed is not None:
            np.random.seed(seed)
        
        self.episode_step = 0
        self.episode_reward = 0
        
        # Get current state from the node
        depth = self.node.position_state[2:3]
        surge = self.node.v_state[0:1]     
        sway = self.node.v_state[1:2]      
        heave = self.node.v_state[2:3]        
        roll = self.node.orientation_state[0:1]         
        pitch = self.node.orientation_state[1:2]         
        yaw = self.node.orientation_state[2:3]           

        # Extract error variables
        depth_error = self.node.position_err[2:3]        
        surge_error = self.node.v_err[0:1]   
        sway_error = self.node.v_err[1:2]       
        heave_error = self.node.v_err[2:3]      
        roll_error = self.node.orientation_err[0:1]     
        pitch_error = self.node.orientation_err[1:2]     
        yaw_error = self.node.orientation_err[2:3]    

        # Create initial observation
        initial_observation = np.concatenate([
            depth_error, 
            surge_error, 
            sway_error,
            heave_error, 
            roll_error, 
            pitch_error, 
            yaw_error,  
            depth, 
            surge, 
            sway,
            heave, 
            roll, 
            pitch, 
            yaw           
        ])

        info = {}
        return initial_observation, info
    
    def step(self, action):
        """Execute action in the environment and return next state, reward, termination flag, etc."""
        # Publish action to ROS
        thruster_cmds, servo_angles_rad = self.node.publish_action(action, self.num_thrusters, self.num_servos)
        
        # Store for reward calculation
        self.thruster_action = thruster_cmds
        self.joint_angles = servo_angles_rad
        self.last_action = action.copy()
        
        # Wait for callbacks to be processed
        timeout_sec = 1.0
        start_time = time.time()
        time.sleep(1.0)
        # Process ROS events to handle callbacks
        while not (self.node.new_state_available and self.node.new_error_available):
            self._spin_node(timeout_sec=0.1)
            
            if time.time() - start_time > timeout_sec:
                print("Warning: Timeout waiting for state/error updates")
                break
        
        # Get updated state from the node
        if self.node.new_state_available and self.node.new_error_available:
            updated_depth = self.node.position_state[2:3]
            updated_surge = self.node.v_state[0:1]     
            updated_sway = self.node.v_state[1:2]      
            updated_heave = self.node.v_state[2:3]        
            updated_roll = self.node.orientation_state[0:1]         
            updated_pitch = self.node.orientation_state[1:2]         
            updated_yaw = self.node.orientation_state[2:3]           

            updated_depth_error = self.node.position_err[2:3]        
            updated_surge_error = self.node.v_err[0:1]   
            updated_sway_error = self.node.v_err[1:2]       
            updated_heave_error = self.node.v_err[2:3]      
            updated_roll_error = self.node.orientation_err[0:1]     
            updated_pitch_error = self.node.orientation_err[1:2]     
            updated_yaw_error = self.node.orientation_err[2:3] 

            observation = np.concatenate([
                updated_depth_error, 
                updated_surge_error, 
                updated_sway_error,
                updated_heave_error,
                updated_roll_error, 
                updated_pitch_error, 
                updated_yaw_error,
                updated_depth, 
                updated_surge, 
                updated_sway,
                updated_heave,
                updated_roll, 
                updated_pitch, 
                updated_yaw
            ])
            state_error_array = np.concatenate([
                updated_depth_error, 
                updated_surge_error, 
                updated_sway_error,
                updated_heave_error,
                updated_roll_error, 
                updated_pitch_error, 
                updated_yaw_error
            ]
            )
            terminated = False
        else: 
            print("Warning: No new state/error available, returning dummy observation")
            observation = np.zeros(14)  # Dummy observation
            terminated = True 
        print(state_error_array)
        # Calculate reward
        reward = self.calculate_reward(state_error_array)
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
        self.episode_reward += reward
        
        # self.episode_reward.append

        # rewards = np.array(self.episode_reward)
        # mean = rewards.mean()
        # std = rewards.std() if rewards.std() > 1e-8 else 1.0
        # self.normalized_rewards = (rewards - mean) / std

        # Check if episode should end
        truncated = False
        if self.episode_step >= self.node.max_steps:
            truncated = True
        
        # Increment step counter
        self.episode_step += 1
        
        return observation, reward, terminated, truncated, {}
    
    def _spin_node(self, timeout_sec=0.1):
        """Process ROS callbacks for a limited time"""
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.01)
    
    def calculate_reward(self,state_error_array):
        """Calculate reward based on specified error components"""
        w = self.config['reward_function']
        w1, w2, w3, w4, w5, w6, w7, w8 = w['w1'], w['w2'], w['w3'], w['w4'], w['w5'], w['w6'], w['w7'], w['w8']
        state_error_weights = np.array(w['state_error_weights'])
        
        # Extract specific error components
        # error = np.concatenate([
        #     self.node.position_err[2:3],  # Depth
        #     self.node.v_err[:2],          # Surge and sway
        #     self.node.orientation_err[:3], # roll, pitch, yaw
        # ]).astype(np.float32)
        error = state_error_array
        # # Compute performance error (quadratic penalty)
        # error_column = error.reshape(-1, 1)
        # # performance_error = np.dot(error, np.diag(state_error_weights))
        # # performance_error = np.dot(performance_error, error_column)
        # # performance_error = np.exp(-performance_error)
        # performance_error = error_column * np.diag(state_error_weights) * error
        # print("PE",performance_error)
        # performance_error = np.exp(-performance_error)
        # print("Reward Error",performance_error)
        error_column = error.reshape(-1, 1)  # Shape: (N,1)
        error_row = error.reshape(1, -1)     # Shape: (1,N)
        weights_diag = np.diag(state_error_weights)  # Shape: (N,N)
        # print(error_column.shape)
        # print(error_row.shape)
        # print(weights_diag.shape)
        
        performance_error = error_row @ weights_diag @ error_column
        performance_error = -performance_error
        # print(performance_error)

        # Servo smoothness penalty using sine and cosine components
        servo_smoothness_penalty = 0
        delta_theta = np.zeros(self.num_servos)

        for i in range(self.num_servos):
            if len(self.joint_positions_history) > 0:
                # Compute average sine and cosine of the historical angles
                avg_sin = np.average(np.sin(self.joint_positions_history[:, i]))
                avg_cos = np.average(np.cos(self.joint_positions_history[:, i]))
                historical_avg_angle = np.arctan2(avg_sin, avg_cos)

                # Get the current angle from self.joint_angles
                current_angle = self.joint_angles[i] if i < len(self.joint_angles) else 0
                
                # Calculate the angular difference
                delta_theta[i] = np.abs(current_angle - historical_avg_angle)

        # Accumulate the smoothness penalty
        servo_smoothness_penalty = np.linalg.norm(delta_theta)
        
        # Update joint_positions_history
        self.joint_positions_history = np.vstack((self.joint_positions_history[1:], self.joint_angles))

        # Thruster usage penalty
        u_t = np.array([
            self.node.thrust_heave_bow,
            self.node.thrust_heave_stern
        ])
        thruster_usage_penalty = np.sum(np.abs(u_t))

        # Thruster smoothness penalty
        thruster_smoothness_penalty = np.linalg.norm(u_t - np.average(self.u_prev, axis=0))

        # Update self.u_prev to store the history
        self.u_prev = np.vstack((self.u_prev[1:], u_t))

        # Thruster delta reward
        thruster_delta_reward = np.linalg.norm(u_t - self.u_prev[-2]) if len(self.u_prev) >= 2 else 0

        # Servo angle penalty
        servo_angle_penalty = np.linalg.norm(self.joint_angles)

        # Update thruster_command_action_prev with clear logic
        self.thruster_command_action_prev = np.vstack((
            self.thruster_command_action_prev[1:],  # Keep all except the first
            self.thruster_action  # Add the newest action
        ))

        # Thruster action penalty
        thruster_action_penalty = np.sum(np.abs(self.thruster_action - np.average(self.thruster_command_action_prev, axis=0)))

        # Thruster Direction Change Penalty
        direction_change_penalty = w8 * thruster_action_penalty ** 2  # Quadratic penalty

        # Total reward
        reward = -(
            -w1 * performance_error +  # positive without exponential
            w2 * servo_smoothness_penalty +
            w3 * thruster_usage_penalty +
            w4 * thruster_smoothness_penalty +
            w5 * servo_angle_penalty +
            w6 * thruster_delta_reward +
            w7 * thruster_action_penalty +
            w8 * direction_change_penalty
        )
        
        return reward


    # def calculate_reward(self, state_error_array):
    #     """
    #     Improved reward function that encourages exploration and provides
    #     better learning signals for deep reinforcement learning
    #     """
    #     w = self.config['reward_function']
    #     w1, w2, w3, w4, w5, w6, w7, w8 = w['w1'], w['w2'], w['w3'], w['w4'], w['w5'], w['w6'], w['w7'], w['w8']
    #     state_error_weights = np.array(w['state_error_weights'])
        
    #     # Extract errors
    #     error = state_error_array
        
    #     # Base performance reward - using a more gradual error to reward mapping
    #     # The key is to make smaller errors give distinctive reward signals
    #     # rather than having reward go to near-zero for large errors
    #     performance_reward = 0
    #     for i, err in enumerate(error):
    #         if state_error_weights[i] > 0:  # Only process weighted errors
    #             # Quadratic error term, but scaled to avoid very small values
    #             error_contribution = -state_error_weights[i] * (err ** 2)
                
    #             # Using a softer scaling function that maps -inf to 0 and 0 to 1
    #             # This ensures better reward signal even when far from target
    #             scaled_contribution = 2.0 / (1.0 + np.exp(error_contribution * 0.5)) - 1.0
                
    #             performance_reward += scaled_contribution
        
    #     # Normalize by number of active weights to keep reward in reasonable range
    #     active_weights = np.sum(state_error_weights > 0)
    #     if active_weights > 0:
    #         performance_reward /= active_weights
        
    #     # Calculate penalties (if enabled in config)
    #     penalties = 0.0
        
    #     # Servo smoothness penalty
    #     if w2 > 0:
    #         servo_smoothness_penalty = 0
    #         delta_theta = np.zeros(self.num_servos)
            
    #         for i in range(self.num_servos):
    #             if len(self.joint_positions_history) > 0:
    #                 avg_sin = np.average(np.sin(self.joint_positions_history[:, i]))
    #                 avg_cos = np.average(np.cos(self.joint_positions_history[:, i]))
    #                 historical_avg_angle = np.arctan2(avg_sin, avg_cos)
    #                 current_angle = self.joint_angles[i] if i < len(self.joint_angles) else 0
    #                 delta_theta[i] = np.abs(current_angle - historical_avg_angle)
                    
    #         servo_smoothness_penalty = np.linalg.norm(delta_theta)
    #         penalties += w2 * servo_smoothness_penalty
        
    #     # Thruster usage penalty - be careful with this! Can discourage learning
    #     if w3 > 0:
    #         u_t = np.array([
    #             self.node.thrust_heave_bow,
    #             self.node.thrust_heave_stern
    #         ])
    #         thruster_usage_penalty = np.sum(np.abs(u_t))
    #         penalties += w3 * thruster_usage_penalty * 0.1  # Scale down to avoid dominating reward
        
    #     # Other penalties - similarly scaled down to avoid dominating
    #     if w4 > 0:
    #         u_t = np.array([
    #             self.node.thrust_heave_bow,
    #             self.node.thrust_heave_stern
    #         ])
    #         thruster_smoothness_penalty = np.linalg.norm(u_t - np.average(self.u_prev, axis=0))
    #         penalties += w4 * thruster_smoothness_penalty * 0.1
        
    #     if w5 > 0:
    #         servo_angle_penalty = np.linalg.norm(self.joint_angles)
    #         penalties += w5 * servo_angle_penalty * 0.1
        
    #     if w6 > 0:
    #         u_t = np.array([
    #             self.node.thrust_heave_bow,
    #             self.node.thrust_heave_stern
    #         ])
    #         thruster_delta_reward = np.linalg.norm(u_t - self.u_prev[-2]) if len(self.u_prev) >= 2 else 0
    #         penalties += w6 * thruster_delta_reward * 0.1
        
    #     if w7 > 0:
    #         thruster_action_penalty = np.sum(np.abs(self.thruster_action - np.average(self.thruster_command_action_prev, axis=0)))
    #         penalties += w7 * thruster_action_penalty * 0.1
        
    #     if w8 > 0:
    #         thruster_action_penalty = np.sum(np.abs(self.thruster_action - np.average(self.thruster_command_action_prev, axis=0)))
    #         direction_change_penalty = thruster_action_penalty ** 2  
    #         penalties += w8 * direction_change_penalty * 0.1
        
    #     # Update history for next iteration
    #     self.joint_positions_history = np.vstack((self.joint_positions_history[1:], self.joint_angles))
        
    #     u_t = np.array([
    #         self.node.thrust_heave_bow,
    #         self.node.thrust_heave_stern
    #     ])
    #     self.u_prev = np.vstack((self.u_prev[1:], u_t))
        
    #     self.thruster_command_action_prev = np.vstack((
    #         self.thruster_command_action_prev[1:],
    #         self.thruster_action
    #     ))
        
    #     # Exploration bonus (decreases over time)
    #     episode_fraction = min(1.0, self.episode_step / 500)
    #     exploration_bonus = 0.2 * (1.0 - episode_fraction)
        
    #     # Calculate final reward - performance reward is positive, penalties are negative
    #     reward = w1 * performance_reward - penalties + exploration_bonus
        
    #     # Add logging every N steps
    #     if self.episode_step % 50 == 0:
    #         self.node.get_logger().info(
    #             f"Step {self.episode_step}: performance={performance_reward:.3f}, "
    #             f"penalties={penalties:.3f}, exploration={exploration_bonus:.3f}, "
    #             f"reward={reward:.3f}, depth_err={error[0]:.3f}, pitch_err={error[4]:.3f}"
    #         )
        
    #     return reward
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'node') and self.node is not None:
            self.node.destroy_node()


def main(args=None):
    """
    Main function to initialize ROS2 and make the environment available for external use.
    This doesn't run any training itself - it's meant to be imported by a training script.
    """

    if not rclpy.ok():
        rclpy.init(args=args)
    

    return AUVEnv()


if __name__ == "__main__":
    main()