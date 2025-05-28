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
        
        self.thruster_size = self.config['environment']['thruster_size']
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
        
        # Convert orientation errors to sin/cos representation to avoid angle wrapping issues
        roll_sin_err = np.sin(self.orientation_err[0])
        roll_cos_err = np.cos(self.orientation_err[0])
        pitch_sin_err = np.sin(self.orientation_err[1])
        pitch_cos_err = np.cos(self.orientation_err[1])
        yaw_sin_err = np.sin(self.orientation_err[2])
        yaw_cos_err = np.cos(self.orientation_err[2])
        
        # Update the state error array with the new representation
        self.state_err = np.concatenate([
            self.position_err[2:3],
            self.v_err[:2],
            np.array([roll_sin_err, roll_cos_err, pitch_sin_err, pitch_cos_err, yaw_sin_err, yaw_cos_err]),
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
        
        # Convert orientation to sin/cos representation to avoid angle wrapping issues
        roll_sin = np.sin(self.orientation_state[0])
        roll_cos = np.cos(self.orientation_state[0])
        pitch_sin = np.sin(self.orientation_state[1])
        pitch_cos = np.cos(self.orientation_state[1])
        yaw_sin = np.sin(self.orientation_state[2])
        yaw_cos = np.cos(self.orientation_state[2])
        
        # Update current state for RL agent with sin/cos representation
        self.current_state = np.concatenate([
            self.position_state[2:3],
            self.v_state[:2],
            np.array([roll_sin, roll_cos, pitch_sin, pitch_cos, yaw_sin, yaw_cos]),
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
            ('heave_bow', thruster_cmds[0]),
            ('heave_stern',  thruster_cmds[1]),
            ('surge_port',  thruster_cmds[2]),
            ('surge_starboard', thruster_cmds[3])
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
        
        min_angle_rad = self.config['environment']['min_servo_angle_rad']
        max_angle_rad = self.config['environment']['max_servo_angle_rad']
        # Map from [-1, 1] to [min_angle_rad, max_angle_rad]
        angle_rad = min_angle_rad + (normalized_command + 1.0) * (max_angle_rad - min_angle_rad) / 2.0
        
        return angle_rad


class AUVEnv(gym.Env):
    """Custom AUV Environment that follows gym interface"""
    
    def __init__(self):
        super(AUVEnv, self).__init__()
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config_sac.yaml')
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
        
        # Update observation space to account for sin/cos representation of angles
        # Original state had 14 dimensions, with sin/cos for 3 angles we add 3 more dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(20,),  
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
        self._spin_node(timeout_sec=0.1)

        if seed is not None:
            np.random.seed(seed)
        
        self.episode_step = 0
        self.episode_reward = 0
        
        zero_thruster_cmds = np.zeros(self.num_thrusters)  
        zero_servo_angles_rad = np.zeros(self.num_servos)  
        zero_action = np.zeros(self.num_thrusters + self.num_servos)  
        
        # Extract state variables with sin/cos representation for angles
        depth = self.node.position_state[2:3]
        surge = self.node.v_state[0:1]     
        sway = self.node.v_state[1:2]      
        heave = self.node.v_state[2:3]
        
        # Convert Euler angles to sin/cos representation
        roll = self.node.orientation_state[0]
        pitch = self.node.orientation_state[1]
        yaw = self.node.orientation_state[2]
        
        roll_sin = np.sin(roll)
        roll_cos = np.cos(roll)
        pitch_sin = np.sin(pitch)
        pitch_cos = np.cos(pitch)
        yaw_sin = np.sin(yaw)
        yaw_cos = np.cos(yaw)

        # Extract error variables with sin/cos representation for angle errors
        depth_error = self.node.position_err[2:3]        
        surge_error = self.node.v_err[0:1]   
        sway_error = self.node.v_err[1:2]       
        heave_error = self.node.v_err[2:3]
        
        roll_error = self.node.orientation_err[0]
        pitch_error = self.node.orientation_err[1]
        yaw_error = self.node.orientation_err[2]
        
        roll_sin_error = np.sin(roll_error)
        roll_cos_error = np.cos(roll_error)
        pitch_sin_error = np.sin(pitch_error)
        pitch_cos_error = np.cos(pitch_error)
        yaw_sin_error = np.sin(yaw_error)
        yaw_cos_error = np.cos(yaw_error)

        # Create initial observation with sin/cos representation
        initial_observation = np.concatenate([
            depth_error,
            20 * surge_error,
            sway_error,
            heave_error,
            np.array([roll_sin_error]),
            np.array([roll_cos_error]),
            np.array([pitch_sin_error]),
            np.array([pitch_cos_error]),
            np.array([yaw_sin_error]),
            np.array([yaw_cos_error]),
            depth,
            20 * surge,
            sway,
            heave,
            np.array([roll_sin]),
            np.array([roll_cos]),
            np.array([pitch_sin]),
            np.array([pitch_cos]),
            np.array([yaw_sin]),
            np.array([yaw_cos])
        ])


        info = {}
        return initial_observation, info
    
    def step(self, action):
        """Execute action in the environment and return next state, reward, termination flag, etc."""

        state_error_array = None

        # Get the current state BEFORE taking the action
        current_depth_error = self.node.position_err[2:3].copy()
        current_surge_error = self.node.v_err[0:1].copy()
        current_sway_error = self.node.v_err[1:2].copy()
        current_heave_error = self.node.v_err[2:3].copy()
        
        # Convert orientation errors to sin/cos
        current_roll_error = self.node.orientation_err[0]
        current_pitch_error = self.node.orientation_err[1]
        current_yaw_error = self.node.orientation_err[2]
        
        current_roll_sin_error = np.sin(current_roll_error)
        current_roll_cos_error = np.cos(current_roll_error)
        current_pitch_sin_error = np.sin(current_pitch_error)
        current_pitch_cos_error = np.cos(current_pitch_error)
        current_yaw_sin_error = np.sin(current_yaw_error)
        current_yaw_cos_error = np.cos(current_yaw_error)

        current_depth = self.node.position_state[2:3].copy()
        current_surge = self.node.v_state[0:1].copy()
        current_sway = self.node.v_state[1:2].copy()
        current_heave = self.node.v_state[2:3].copy()
        
        # Convert orientation to sin/cos
        current_roll = self.node.orientation_state[0]
        current_pitch = self.node.orientation_state[1]
        current_yaw = self.node.orientation_state[2]
        
        current_roll_sin = np.sin(current_roll)
        current_roll_cos = np.cos(current_roll)
        current_pitch_sin = np.sin(current_pitch)
        current_pitch_cos = np.cos(current_pitch)
        current_yaw_sin = np.sin(current_yaw)
        current_yaw_cos = np.cos(current_yaw)
        
        current_state = np.concatenate([
            current_depth_error,
            current_surge_error,
            current_sway_error,
            current_heave_error,
            np.array([current_roll_sin_error, current_roll_cos_error]),
            np.array([current_pitch_sin_error, current_pitch_cos_error]),
            np.array([current_yaw_sin_error, current_yaw_cos_error]),
            current_depth,
            current_surge,
            current_sway,
            current_heave,
            np.array([current_roll_sin, current_roll_cos]),
            np.array([current_pitch_sin, current_pitch_cos]),
            np.array([current_yaw_sin, current_yaw_cos])
        ])

        # Publish action to ROS
        thruster_cmds, servo_angles_rad = self.node.publish_action(action, self.num_thrusters, self.num_servos)
        # Store for reward calculation
        self.thruster_action = thruster_cmds
        self.joint_angles = servo_angles_rad
        self.last_action = action.copy()
        
        # Wait for callbacks to be processed
        timeout_sec = 0.5
        start_time = time.time()
        
        # Process ROS events to handle callbacks
        while not (self.node.new_state_available and self.node.new_error_available):
            self._spin_node(timeout_sec=0.48)
            if time.time() - start_time > timeout_sec:
                print("Warning: Timeout waiting for state/error updates")
                break
                
        # Get updated state from the node
        if self.node.new_state_available and self.node.new_error_available:
            updated_depth = self.node.position_state[2:3]
            updated_surge = self.node.v_state[0:1]
            updated_sway = self.node.v_state[1:2]
            updated_heave = self.node.v_state[2:3]
            
            # Convert updated orientation to sin/cos
            updated_roll = self.node.orientation_state[0]
            updated_pitch = self.node.orientation_state[1]
            updated_yaw = self.node.orientation_state[2]
            
            updated_roll_sin = np.sin(updated_roll)
            updated_roll_cos = np.cos(updated_roll)
            updated_pitch_sin = np.sin(updated_pitch)
            updated_pitch_cos = np.cos(updated_pitch)
            updated_yaw_sin = np.sin(updated_yaw)
            updated_yaw_cos = np.cos(updated_yaw)
            
            updated_depth_error = self.node.position_err[2:3]
            updated_surge_error = self.node.v_err[0:1]
            updated_sway_error = self.node.v_err[1:2]
            updated_heave_error = self.node.v_err[2:3]
            
            # Convert updated orientation errors to sin/cos
            updated_roll_error = self.node.orientation_err[0]
            updated_pitch_error = self.node.orientation_err[1]
            updated_yaw_error = self.node.orientation_err[2]
            
            updated_roll_sin_error = np.sin(updated_roll_error)
            updated_roll_cos_error = np.cos(updated_roll_error)
            updated_pitch_sin_error = np.sin(updated_pitch_error)
            updated_pitch_cos_error = np.cos(updated_pitch_error)
            updated_yaw_sin_error = np.sin(updated_yaw_error)
            updated_yaw_cos_error = np.cos(updated_yaw_error)
            
            observation = np.concatenate([
                updated_depth_error,
                20 * updated_surge_error,
                updated_sway_error,
                updated_heave_error,
                np.array([updated_roll_sin_error, updated_roll_cos_error]),
                np.array([updated_pitch_sin_error, updated_pitch_cos_error]),
                np.array([updated_yaw_sin_error, updated_yaw_cos_error]),
                updated_depth,
                20 * updated_surge,
                updated_sway,
                updated_heave,
                np.array([updated_roll_sin, updated_roll_cos]),
                np.array([updated_pitch_sin, updated_pitch_cos]),
                np.array([updated_yaw_sin, updated_yaw_cos])
            ])
            
            # Create error array for reward calculation
            # Include both sin and cos components for angle errors
            state_error_array = np.concatenate([
                updated_depth_error,
                updated_surge_error,
                updated_sway_error,
                updated_heave_error,
                np.array([updated_roll_sin_error, updated_roll_cos_error]),
                np.array([updated_pitch_sin_error, updated_pitch_cos_error]),
                np.array([updated_yaw_sin_error, updated_yaw_cos_error])
            ])

            if state_error_array is None:
                state_error_array = np.zeros(len(state_error_array))

            terminated = False

        else:
            print("Warning: No new state/error available, returning dummy observation")
            observation = np.zeros(20)  # Dummy observation with updated dimensions
            terminated = True

        # Calculate reward
        reward = self.calculate_reward(state_error_array)
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())

        # Store raw reward in episode rewards list
        if not hasattr(self, 'episode_rewards'):
            self.episode_rewards = []
        self.episode_rewards.append(reward)

        # Track cumulative episode reward (using raw reward)
        self.episode_reward += reward

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
    
    def calculate_reward(self, state_error_array):
        """Calculate reward with sin/cos representation for angles"""
        w = self.config['reward_function']
        w1, w2, w3, w4, w5, w6, w7, w8 = w['w1'], w['w2'], w['w3'], w['w4'], w['w5'], w['w6'], w['w7'], w['w8']
        
        # Update state_error_weights to account for sin/cos components
        # Original weights were for 7 components, now we have 10 (3 angles -> 6 sin/cos components)
        original_weights = np.array(w['state_error_weights'])
        
        # Create new weights array accounting for sin/cos representation
        # For each angle, distribute its weight across both sin and cos components
        state_error_weights = np.zeros(10)
        
        # Position and velocity errors (unchanged)
        state_error_weights[0] = original_weights[0]  # depth
        state_error_weights[1] = original_weights[1]  # surge
        state_error_weights[2] = original_weights[2]  # sway
        state_error_weights[3] = original_weights[3]  # heave
        
        # Orientation errors (distribute weights between sin/cos pairs)
        state_error_weights[4] = original_weights[4] / 2  # roll sin
        state_error_weights[5] = original_weights[4] / 2  # roll cos
        state_error_weights[6] = original_weights[5] / 2  # pitch sin
        state_error_weights[7] = original_weights[5] / 2  # pitch cos
        state_error_weights[8] = original_weights[6] / 2  # yaw sin
        state_error_weights[9] = original_weights[6] / 2  # yaw cos

        # Extract error
        error = state_error_array
        if error is None:
            error = np.zeros(len(state_error_weights))
            
        # Performance error calculation
        error_column = error.reshape(-1, 1)  # Shape: (N,1)
        error_row = error.reshape(1, -1)     # Shape: (1,N)
        weights_diag = np.diag(state_error_weights)  # Shape: (N,N)
        
        performance_error = error_row @ weights_diag @ error_column
        performance_error = -performance_error
        
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
            self.node.thrust_heave_stern,
            self.node.thrust_surge_port,
            self.node.thrust_surge_starboard
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

        # Final reward calculation
        reward = -(
            -w1 * performance_error +
            w2 * servo_smoothness_penalty +
            w3 * thruster_usage_penalty +
            w4 * thruster_smoothness_penalty +
            w5 * servo_angle_penalty +
            w6 * thruster_delta_reward +
            w7 * thruster_action_penalty +
            w8 * direction_change_penalty
        )
        
        return reward

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