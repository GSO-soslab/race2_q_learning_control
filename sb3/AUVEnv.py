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
        self.joint_angles = np.zeros(servo_size) # This is AUVEnvNode.joint_angles
        
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
        
        # self.state_err is not directly used for observation but might be for reward or internal logic.
        # Its composition doesn't directly match the observation structure.
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
        
        # Update current state for RL agent (this specific concatenation is not used for observation directly)
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
        self.joint_angles = np.array([self.joint_angles_port, self.joint_angles_starboard], dtype=np.float32) # Ensure it's a numpy array

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
            msg.data = float(value) # Ensure value is a standard float
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
        
        # Observation space: 14 dimensions
        # [depth_err, surge_err*20, sway_err, heave_err, roll_err, pitch_err, yaw_err,
        #  depth, surge*20, sway, heave, roll, pitch, yaw]
        # TODO: If observations are reliably scaled to [-1, 1], you might change low/high here.
        # For now, keeping -np.inf, np.inf as per "don't touch" rule for this part.
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(14,),  # state dimension
            dtype=np.float32
        )

        # <<< START OF MINMAX SCALER ADDITION: Attributes >>>
        # Define the approximate expected min/max bounds for each raw observation component.
        # These MUST be tuned based on your AUV's operational limits and typical error ranges.
        # Order: [depth_err, surge_err*20, sway_err, heave_err, roll_err, pitch_err, yaw_err,
        #         depth, surge*20, sway, heave, roll, pitch, yaw]
        self.obs_min_bounds = np.array([
            -10.0,  # depth_error (m)
            -0.8, # surge_error * 20 (m/s * 20)
            -0.4,  # sway_error (m/s)
            -1.0,  # heave_error (m/s)
            -np.pi, # roll_error (rad)
            -np.pi, # pitch_error (rad)
            -np.pi, # yaw_error (rad)
            0.0,   # depth (m) - assuming min depth is 0
            -0.8, # surge * 20 (m/s * 20) - assuming max reverse surge 2 m/s
            -0.4,  # sway (m/s)
            -1.0,  # heave (m/s) - assuming can go up/down
            -np.pi, # roll (rad)
            -np.pi/2, # pitch (rad)
            -np.pi  # yaw (rad)
        ], dtype=np.float32)

        self.obs_max_bounds = np.array([
            10.0,   # depth_error (m)
            0.8,  # surge_error * 20 (m/s * 20)
            0.4,   # sway_error (m/s)
            0.8,   # heave_error (m/s)
            np.pi, # roll_error (rad)
            np.pi, # pitch_error (rad)
            np.pi, # yaw_error (rad)
            10.0,  # depth (m) - assuming max depth 10m
            0.8,  # surge * 20 (m/s * 20) - assuming max forward surge 2 m/s
            0.6,   # sway (m/s)
            0.8,   # heave (m/s)
            np.pi, # roll (rad)
            np.pi/2, # pitch (rad)
            np.pi  # yaw (rad)
        ], dtype=np.float32)

        self.observation_scaled_low = -1.0
        self.observation_scaled_high = 1.0
        # <<< END OF MINMAX SCALER ADDITION: Attributes >>>
        
        # Initialize episode-related variables
        self.episode_step = 0
        self.episode_reward = 0.0
        
        # Initialize history arrays for smoothness calculations
        # Ensure joint_angles_history is initialized correctly based on num_servos
        self.joint_positions_history = np.zeros((10, self.num_servos), dtype=np.float32)  # Store last 10 servo positions
        self.u_prev = np.zeros((10, self.num_thrusters), dtype=np.float32)  # Store last 10 thruster commands
        self.thruster_command_action_prev = np.zeros((10, self.num_thrusters), dtype=np.float32)  # Store last 10 thruster actions
        
        # Initialize tracking variables
        self.thruster_action = np.zeros(self.num_thrusters, dtype=np.float32)
        self.joint_angles = np.zeros(self.num_servos, dtype=np.float32) # This is AUVEnv.joint_angles, distinct from node's
        self.last_action = np.zeros(thruster_size + servo_size, dtype=np.float32)

    # <<< START OF MINMAX SCALER ADDITION: Scaling Method >>>
    def _scale_observation(self, observation):
        """Scales the observation to the range [self.observation_scaled_low, self.observation_scaled_high]."""
        # Clip observation to be within defined bounds to avoid issues with values outside expected range
        # and to prevent division by zero if obs_min_bounds == obs_max_bounds for some component.
        # However, if a component of obs_min_bounds is equal to obs_max_bounds, scaling that component is problematic.
        # We'll handle this by ensuring the range is not zero.
        range_ = self.obs_max_bounds - self.obs_min_bounds
        # Replace zeros in range_ with a small epsilon to avoid division by zero,
        # or handle components where min=max by setting scaled value to 0 or mid-point.
        # For simplicity, if min == max, the scaled value will be 0 if input is also that value,
        # or it will map to low/high if input is outside (due to clipping).
        # A more robust approach might be to not scale components where min==max, or scale them to a fixed mid-value.
        
        # Ensure range_ has no zero values to prevent division by zero
        # If min_bound == max_bound for an element, its range is 0.
        # Scaled value for such elements will be `self.observation_scaled_low` if obs <= min_bound,
        # `self.observation_scaled_high` if obs >= max_bound,
        # or result in NaN if obs is between (due to 0/0).
        # Let's explicitly handle the case where range is zero.
        
        scaled_observation = np.zeros_like(observation, dtype=np.float32)
        for i in range(len(observation)):
            if range_[i] == 0:
                # If min and max are the same, and observation is also that value, scale to mid-point of target.
                # Or, if observation is different (should be clipped), it will be scaled to low/high.
                # For safety, let's map it to the lower bound of the scaled range if min=max.
                # A common choice is to map to 0 in the scaled range.
                if observation[i] == self.obs_min_bounds[i]: # or self.obs_max_bounds[i]
                     # map to the middle of the scaled range, e.g. 0 if scaled range is [-1, 1]
                    scaled_observation[i] = (self.observation_scaled_low + self.observation_scaled_high) / 2.0
                elif observation[i] < self.obs_min_bounds[i]:
                    scaled_observation[i] = self.observation_scaled_low
                else: # observation[i] > self.obs_max_bounds[i]
                    scaled_observation[i] = self.observation_scaled_high
            else:
                # Standard Min-Max scaling formula
                scaled_observation[i] = (observation[i] - self.obs_min_bounds[i]) / range_[i] \
                                      * (self.observation_scaled_high - self.observation_scaled_low) \
                                      + self.observation_scaled_low
        
        # Clip to ensure the scaled observation is strictly within the target range,
        # especially due to potential floating point inaccuracies or if original obs was outside bounds.
        return np.clip(scaled_observation, self.observation_scaled_low, self.observation_scaled_high).astype(np.float32)
    # <<< END OF MINMAX SCALER ADDITION: Scaling Method >>>
    
    def reset(self, seed=None):
        """Reset the environment to initial state and return the initial observation"""
        #so the states come in
        self._spin_node(timeout_sec=0.1) # Allow callbacks to update node's internal state

        if seed is not None:
            # super().reset(seed=seed) # For newer gymnasium versions, if you need to seed internal RNGs
            np.random.seed(seed)
        
        self.episode_step = 0
        self.episode_reward = 0
        
        # Resetting actions and history arrays
        self.joint_positions_history = np.zeros((10, self.num_servos), dtype=np.float32)
        self.u_prev = np.zeros((10, self.num_thrusters), dtype=np.float32)
        self.thruster_command_action_prev = np.zeros((10, self.num_thrusters), dtype=np.float32)
        self.thruster_action = np.zeros(self.num_thrusters, dtype=np.float32)
        self.joint_angles = np.zeros(self.num_servos, dtype=np.float32) # AUVEnv's copy
        self.last_action = np.zeros(self.num_thrusters + self.num_servos, dtype=np.float32)

        # It's good practice to reset PID integral and prev_error terms here if they exist from reward function
        if hasattr(self, 'error_integral'):
            self.error_integral = np.zeros_like(self.error_integral)
        if hasattr(self, 'prev_error'):
            self.prev_error = np.zeros_like(self.prev_error)

        # zero_thruster_cmds = np.zeros(self.num_thrusters)  
        # zero_servo_angles_rad = np.zeros(self.num_servos)  
        # zero_action = np.zeros(self.num_thrusters + self.num_servos)  
        # self.node.publish_action(zero_action, self.num_thrusters, self.num_servos) # Optionally send a zero action

        # Wait for fresh state after potential zero action or just to ensure latest state
        self._spin_node(timeout_sec=0.2) # Increased timeout slightly for reset

        # Extract state variables from the node
        depth = self.node.position_state[2:3].astype(np.float32)
        surge = self.node.v_state[0:1].astype(np.float32)     
        sway = self.node.v_state[1:2].astype(np.float32)      
        heave = self.node.v_state[2:3].astype(np.float32)        
        roll = self.node.orientation_state[0:1].astype(np.float32)         
        pitch = self.node.orientation_state[1:2].astype(np.float32)         
        yaw = self.node.orientation_state[2:3].astype(np.float32)           

        # Extract error variables from the node
        depth_error = self.node.position_err[2:3].astype(np.float32)        
        surge_error = self.node.v_err[0:1].astype(np.float32)   
        sway_error = self.node.v_err[1:2].astype(np.float32)       
        heave_error = self.node.v_err[2:3].astype(np.float32)      
        roll_error = self.node.orientation_err[0:1].astype(np.float32)     
        pitch_error = self.node.orientation_err[1:2].astype(np.float32)     
        yaw_error = self.node.orientation_err[2:3].astype(np.float32)    
        print(depth_error,depth)
        # Create initial observation (raw, before scaling)
        # Make sure the order matches self.obs_min_bounds and self.obs_max_bounds
        raw_initial_observation = np.concatenate([
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
        ]).astype(np.float32)

        # <<< START OF MINMAX SCALER ADDITION: Scale observation >>>
        scaled_initial_observation = self._scale_observation(raw_initial_observation)
        # <<< END OF MINMAX SCALER ADDITION: Scale observation >>>

        info = {} # Standard practice for Gymnasium's reset
        return scaled_initial_observation, info
    
    def step(self, action):
        """Execute action in the environment and return next state, reward, termination flag, etc."""

        # Ensure action is float32, as action_space dictates
        action = np.array(action, dtype=np.float32)

        # state_error_array will be populated after new state/error is received
        state_error_array = None 

        # Get the current state BEFORE taking the action (not used directly, but good for potential debugging)
        # current_depth_error = self.node.position_err[2:3].copy()
        # ... (rest of current state variables)
        # current_state = np.concatenate([...]) # This was for logging, can be omitted for performance

        # Publish action to ROS
        # IMPORTANT: The following line was commented out in your original code.
        # If it remains commented, the agent's actions are not actually sent to the AUV.
        # This means thruster_cmds and servo_angles_rad below will be based on the *previous* AUV state,
        # not the *agent's* commanded action for this step.
        thruster_cmds_from_agent, servo_angles_rad_from_agent = self.node.publish_action(action, self.num_thrusters, self.num_servos)
        
        # Store the agent's commanded thruster actions and servo angles for reward calculation
        self.thruster_action = np.array(thruster_cmds_from_agent, dtype=np.float32) # This should be what the agent commanded
        self.joint_angles = np.array(servo_angles_rad_from_agent, dtype=np.float32) # AUVEnv's copy of commanded servo angles
        self.last_action = action.copy() # Store the raw action from the agent

        # Wait for callbacks to be processed and new state/error to be available
        timeout_sec = 1.0 # Original timeout
        start_time = time.time()
        # Reset flags before waiting
        self.node.new_state_available = False
        self.node.new_error_available = False

        while not (self.node.new_state_available and self.node.new_error_available):
            self._spin_node(timeout_sec=0.05) # Spin more frequently with shorter timeout per spin
            if time.time() - start_time > timeout_sec:
                self.node.get_logger().warn("Warning: Timeout waiting for state/error updates in step")
                break
        
        # Get updated state from the node
        raw_observation = np.zeros(14, dtype=np.float32) # Dummy observation in case of timeout
        terminated = False # Default to not terminated

        if self.node.new_state_available and self.node.new_error_available:
            updated_depth = self.node.position_state[2:3].astype(np.float32)
            updated_surge = self.node.v_state[0:1].astype(np.float32)
            updated_sway = self.node.v_state[1:2].astype(np.float32)
            updated_heave = self.node.v_state[2:3].astype(np.float32)
            updated_roll = self.node.orientation_state[0:1].astype(np.float32)
            updated_pitch = self.node.orientation_state[1:2].astype(np.float32)
            updated_yaw = self.node.orientation_state[2:3].astype(np.float32)

            updated_depth_error = self.node.position_err[2:3].astype(np.float32)
            updated_surge_error = self.node.v_err[0:1].astype(np.float32)
            updated_sway_error = self.node.v_err[1:2].astype(np.float32)
            updated_heave_error = self.node.v_err[2:3].astype(np.float32)
            updated_roll_error = self.node.orientation_err[0:1].astype(np.float32)
            updated_pitch_error = self.node.orientation_err[1:2].astype(np.float32)
            updated_yaw_error = self.node.orientation_err[2:3].astype(np.float32)
            
            # Construct the raw observation
            # Order must match self.obs_min_bounds and self.obs_max_bounds
            raw_observation = np.concatenate([
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
            ]).astype(np.float32)
            
            # State error array for reward calculation (unscaled, without the *20 multipliers for surge)
            state_error_array = np.concatenate([
                updated_depth_error,
                updated_surge_error, # Note: surge_error (not *20) for reward calc consistency
                updated_sway_error,
                updated_heave_error,
                updated_roll_error,
                updated_pitch_error,
                updated_yaw_error
            ]).astype(np.float32)

        else:
            self.node.get_logger().warn("Warning: No new state/error available after action, returning zero observation.")
            # raw_observation remains zeros
            state_error_array = np.zeros(7, dtype=np.float32) # Dummy error array for reward
            terminated = True # Consider terminating if state is not updating

        # <<< START OF MINMAX SCALER ADDITION: Scale observation >>>
        scaled_observation = self._scale_observation(raw_observation)
        # <<< END OF MINMAX SCALER ADDITION: Scale observation >>>

        # Calculate reward using the unscaled state_error_array
        reward = self.calculate_reward(state_error_array) 
        if isinstance(reward, np.ndarray): # Ensure reward is a scalar float
            reward = float(reward.item())

        # Store raw reward in episode rewards list (optional, for analysis)
        if not hasattr(self, 'episode_rewards_history'): # Renamed to avoid confusion with episode_reward sum
            self.episode_rewards_history = []
        self.episode_rewards_history.append(reward)

        # # Normalize reward using the history of rewards (Commented out as per original)
        # if len(self.episode_rewards_history) > 1:
        #     rewards_hist = np.array(self.episode_rewards_history)
        #     mean = rewards_hist.mean()
        #     std = rewards_hist.std() if rewards_hist.std() > 1e-8 else 1.0
        #     normalized_reward = 10 * (reward - mean) / std
        # else:
        #     normalized_reward = 10 * reward
        # print(normalized_reward) # This was commented out, so keeping it that way.

        # Track cumulative episode reward (using raw reward)
        self.episode_reward += reward

        # Check if episode should end due to max steps
        truncated = False
        if self.episode_step >= self.node.max_steps -1 : # -1 because step counter increments after this
            truncated = True
        # print ("Immediate reward: ",reward) # Commented out as per original
        
        # Increment step counter
        self.episode_step += 1
        
        # info dictionary
        info = {}

        return scaled_observation, reward, terminated, truncated, info
    
    def _spin_node(self, timeout_sec=0.1):
        """Process ROS callbacks for a limited time"""
        # Original implementation was a loop, which is fine.
        # An alternative: rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        # if we had a future to wait on. For general callbacks, spin_once in a loop is typical.
        if rclpy.ok():
            start_time = self.node.get_clock().now()
            end_time = start_time + rclpy.duration.Duration(seconds=timeout_sec)
            while rclpy.ok() and self.node.get_clock().now() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.01) # Short timeout for spin_once
    
    def calculate_reward(self, state_error_array):
        """Calculate reward with individual PID components for each error term"""
        w = self.config['reward_function']
        w1, w2, w3, w4, w5, w6, w7, w8 = w['w1'], w['w2'], w['w3'], w['w4'], w['w5'], w['w6'], w['w7'], w['w8']
        state_error_weights = np.array(w['state_error_weights'], dtype=np.float32)

        # Ensure state_error_array is correctly shaped and typed
        error = np.array(state_error_array, dtype=np.float32).flatten() # Ensure it's a flat 1D array
        
        if error is None or len(error) != len(state_error_weights): # Should match 7 error components
            self.node.get_logger().error(f"Error array mismatch in reward. Expected 7, got {len(error) if error is not None else 'None'}")
            error = np.zeros(len(state_error_weights), dtype=np.float32) # Fallback

        # # Compute performance error (quadratic penalty) - this was the old approach
        # error_column = error.reshape(-1, 1)
        # performance_error_old = error_column * np.diag(state_error_weights) * error # This is element-wise, not dot product for quadratic form
        # performance_error_old = np.exp(-performance_error_old) # This was also element-wise
        
        # Correct quadratic form for performance error: e^T * W * e
        # performance_error_quadratic_form = error.reshape(1, -1) @ np.diag(state_error_weights) @ error.reshape(-1, 1)
        # performance_error_exp = np.exp(-performance_error_quadratic_form) # This yields a single scalar
        # The provided code seems to intend a PID-style reward rather than this direct exponential form.

        # Initialize PID tracking structures if they don’t exist
        if not hasattr(self, 'error_integral'):
            self.error_integral = np.zeros_like(error, dtype=np.float32)
            self.prev_error = np.zeros_like(error, dtype=np.float32)

        # Read PID gain multipliers from config
        pid_config = w['pid_gains']
        p_mult = float(pid_config.get('P_multiplier', 1.0)) # Use .get for safety
        i_mult = float(pid_config.get('I_multiplier', 0.1))
        d_mult = float(pid_config.get('D_multiplier', 0.01))

        # Set base PID gains using config multipliers
        # These gains are applied to the squared errors.
        self.pid_gains = {
            'P': state_error_weights * p_mult,
            'I': state_error_weights * i_mult,
            'D': state_error_weights * d_mult
        }
        
        # === CALCULATE PID COMPONENTS FOR EACH ERROR TERM ===
        # Initialize arrays to store individual PID terms
        p_terms = np.zeros_like(error, dtype=np.float32)
        i_terms = np.zeros_like(error, dtype=np.float32)
        d_terms = np.zeros_like(error, dtype=np.float32)
        
        # Calculate terms for each error component
        for i in range(len(error)):
            # Proportional term - current error squared
            p_terms[i] = -(error[i]**2) * self.pid_gains['P'][i]
            
            # Integral term - accumulated error with decay (integral of error, then squared for penalty)
            integral_decay = 0.95  # Prevent integral windup
            self.error_integral[i] = self.error_integral[i] * integral_decay + error[i] # Accumulate error
            i_terms[i] = -(self.error_integral[i]**2) * self.pid_gains['I'][i] # Penalty on squared integral
            
            # Differential term - rate of change of error squared
            error_diff = error[i] - self.prev_error[i]
            d_terms[i] = -(error_diff**2) * self.pid_gains['D'][i]
        
        # Store current error for next iteration's differential calculation
        self.prev_error = error.copy()
        
        # === COMBINE PID TERMS ===
        # Sum individual PID components
        pid_reward_component = np.sum(p_terms) + np.sum(i_terms) + np.sum(d_terms) # This will be negative or zero
        
        # === EXISTING PENALTY TERMS (unchanged in structure, ensure types) ===
        # Servo smoothness penalty using sine and cosine components
        servo_smoothness_penalty_val = 0.0 # Changed variable name to avoid conflict
        # self.joint_angles is from AUVEnv (commanded by agent), self.node.joint_angles is from subscribers (actual)
        # For reward, usually, we penalize commanded actions/angles.
        current_servo_angles_for_reward = self.joint_angles # Use agent's commanded servo angles

        if self.num_servos > 0: # Proceed only if there are servos
            delta_theta = np.zeros(self.num_servos, dtype=np.float32)
            for i in range(self.num_servos):
                if len(self.joint_positions_history) > 0 and self.joint_positions_history.shape[1] == self.num_servos:
                    avg_sin = np.average(np.sin(self.joint_positions_history[:, i]))
                    avg_cos = np.average(np.cos(self.joint_positions_history[:, i]))
                    historical_avg_angle = np.arctan2(avg_sin, avg_cos)
                    current_angle = current_servo_angles_for_reward[i] if i < len(current_servo_angles_for_reward) else 0.0
                    delta_theta[i] = np.abs(current_angle - historical_avg_angle) # This difference can be > pi, consider angular difference
                    # A better angular difference:
                    # diff = current_angle - historical_avg_angle
                    # delta_theta[i] = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))

            servo_smoothness_penalty_val = np.linalg.norm(delta_theta)
        
        # Update joint_positions_history with agent's commanded servo angles
        if self.num_servos > 0:
             self.joint_positions_history = np.vstack((self.joint_positions_history[1:], current_servo_angles_for_reward.reshape(1, -1))).astype(np.float32)


        # Thruster usage penalty (based on actual thruster values reported by node, or agent's commanded thrusters?)
        # Original used self.node.thrust_... which are subscribed values (actual effort if controller is perfect)
        # Let's assume it refers to actual thruster outputs for "usage"
        u_t_actual = np.array([
            self.node.thrust_heave_bow,
            self.node.thrust_heave_stern,
            self.node.thrust_surge_port,
            self.node.thrust_surge_starboard
        ], dtype=np.float32)
        thruster_usage_penalty_val = np.sum(np.abs(u_t_actual))

        # Thruster smoothness penalty (based on actual thruster outputs)
        thruster_smoothness_penalty_val = 0.0
        if len(self.u_prev) > 0 and self.u_prev.shape[0] > 0 : # Ensure u_prev is not empty
            thruster_smoothness_penalty_val = np.linalg.norm(u_t_actual - np.average(self.u_prev, axis=0))

        # Update self.u_prev to store the history of actual thruster outputs
        self.u_prev = np.vstack((self.u_prev[1:], u_t_actual.reshape(1, -1))).astype(np.float32)

        # Thruster delta reward/penalty (change from previous actual thruster output)
        thruster_delta_penalty_val = 0.0 # Renamed, as it's usually a penalty
        if len(self.u_prev) >= 2: # Need at least current and one previous
             thruster_delta_penalty_val = np.linalg.norm(u_t_actual - self.u_prev[-2]) # u_prev[-2] is previous actual u_t

        # Servo angle penalty (based on agent's commanded servo angles)
        servo_angle_penalty_val = np.linalg.norm(current_servo_angles_for_reward) if self.num_servos > 0 else 0.0

        # Thruster action penalty (based on agent's commanded thruster actions stored in self.thruster_action)
        thruster_action_penalty_val = 0.0
        # self.thruster_action should hold the thruster commands from the current agent action
        if len(self.thruster_command_action_prev) > 0 and self.thruster_command_action_prev.shape[0] > 0:
            thruster_action_penalty_val = np.sum(np.abs(self.thruster_action - np.average(self.thruster_command_action_prev, axis=0)))
        
        # Update thruster_command_action_prev with current agent's commanded thruster actions
        self.thruster_command_action_prev = np.vstack((
            self.thruster_command_action_prev[1:],
            self.thruster_action.reshape(1, -1) 
        )).astype(np.float32)


        # Thruster Direction Change Penalty (based on thruster_action_penalty_val)
        direction_change_penalty_val = thruster_action_penalty_val ** 2

        # === FINAL REWARD CALCULATION ===
        # The PID reward component is already negative.
        # Penalties should subtract from the reward (i.e., be positive values multiplied by negative weights, or add to a negative sum)
        # Original formula: reward = -( -w1 * pid_reward + w2*p2 + w3*p3 + ...)
        # This means pid_reward (which is negative) gets multiplied by -w1 (positive result if w1>0), then negated.
        # So, a less negative (better) pid_reward results in a more negative final reward. This seems inverted.
        # Let's assume: pid_reward is good (less negative is better). Penalties are bad (larger is worse).
        # So, reward = w1 * pid_reward - w2*p2 - w3*p3 ...
        # If pid_reward is already calculated as a sum of negative terms (penalties), then:
        # reward = w1 * pid_reward_component (where w1 is positive, pid_reward_component is negative)
        #          - w2 * servo_smoothness_penalty_val
        #          - w3 * thruster_usage_penalty_val
        #          ...
        
        # Using the original structure:
        # The pid_reward_component is already negative. If w1 is positive, -w1 * pid_reward_component becomes positive.
        # Then the whole thing is negated. This means a more negative pid_reward_component (worse performance)
        # leads to a more positive value inside the parenthesis, and thus a more negative final reward. This is correct.
        reward = -(
            -w1 * pid_reward_component +  # pid_reward_component is typically <=0. If w1>0, -w1*pid_reward is >=0.
            w2 * servo_smoothness_penalty_val +
            w3 * thruster_usage_penalty_val +
            w4 * thruster_smoothness_penalty_val +
            w5 * servo_angle_penalty_val +
            w6 * thruster_delta_penalty_val + # Renamed for clarity
            w7 * thruster_action_penalty_val +
            w8 * direction_change_penalty_val # Original had w8 * thruster_action_penalty ** 2
        )
        
        # Optional: add diagnostic logging
        # if hasattr(self, 'episode_step') and self.episode_step % 100 == 0: # Changed self.step_count to self.episode_step
            # self.node.get_logger().info(f"Step {self.episode_step} | PID Reward: {pid_reward_component:.4f} | P: {np.sum(p_terms):.4f} | I: {np.sum(i_terms):.4f} | D: {np.sum(d_terms):.4f}")
            # self.node.get_logger().info(f"Penalties: ServoSmooth={servo_smoothness_penalty_val:.2f}, ThrusterUse={thruster_usage_penalty_val:.2f}, ThrusterSmooth={thruster_smoothness_penalty_val:.2f}")
            # self.node.get_logger().info(f"Final Reward: {reward:.4f}")
        
        return float(reward) # Ensure it's a scalar float

    # def calculate_reward(self, state_error_array): # The second reward function was commented out, keeping it so.
    #     """
    #     Improved reward function that encourages exploration and provides
    #     better learning signals for deep reinforcement learning
    #     """
    # ... (rest of the commented out reward function)
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'node') and self.node is not None:
            self.node.destroy_node()
        # rclpy.shutdown() # Typically, shutdown is handled by the script that initialized rclpy.


def main(args=None):
    """
    Main function to initialize ROS2 and make the environment available for external use.
    This doesn't run any training itself - it's meant to be imported by a training script.
    """

    if not rclpy.ok(): # Check if rclpy is already initialized
        rclpy.init(args=args)
    
    # Example of creating and testing the environment (optional)
    # try:
    #     env = AUVEnv()
    #     print("AUVEnv created successfully.")
    #     obs, info = env.reset()
    #     print(f"Initial observation (scaled): {obs.shape}, {obs}")
    #     for _ in range(5):
    #         action = env.action_space.sample()
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         print(f"Step: Obs={obs.shape}, Reward={reward:.4f}, Term={terminated}, Trunc={truncated}")
    #         if terminated or truncated:
    #             print("Episode finished.")
    #             obs, info = env.reset()
    #             print(f"Reset. New obs: {obs.shape}")
    # finally:
    #     if 'env' in locals() and hasattr(env, 'close'):
    #         env.close()
    #     if rclpy.ok(): # Only shutdown if main initialized it and no other nodes are expected.
    #         # Be cautious with global shutdown if other parts of an application use rclpy.
    #         # rclpy.shutdown() 
    #         pass

    return AUVEnv() # Returns an instance of the environment


if __name__ == "__main__":
    # This allows the file to be run to potentially test the environment setup,
    # but the primary use is for `main()` to be called by an external training script.
    print("AUV Environment script. To use, import AUVEnv or call main().")
    # Example:
    # env_instance = main()
    # print("AUVEnv instance created via main(). Call env_instance.reset(), env_instance.step(), etc.")
    # # ... do something with env_instance ...
    # if env_instance:
    #    env_instance.close()
    # if rclpy.ok():
    #    rclpy.shutdown()
    pass # Placeholder if no direct execution test is desired here.