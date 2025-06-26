import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os, time, datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32, Header
from geometry_msgs.msg import Vector3
from mvp_msgs.msg import ControlProcess
from sensor_msgs.msg import Imu
import yaml
from rclpy.clock import Clock
import random

# Import the coupling reward calculator
from coupling_rewards import CouplingAwareRewardCalculator
from csv_data_manager import CSVDataManager
class SetpointManager:
    """Manages setpoint generation for episode-based training"""
    
    def __init__(self, config):
        # Load setpoint configuration from config or use defaults
        setpoint_config = config.get('setpoint', {})
        
        self.pos_z_range = setpoint_config.get('pos_z_range', [1.0, 8.0])
        self.ori_z_range = setpoint_config.get('ori_z_range', [-2.14, 2.14])
        self.ori_y_range = setpoint_config.get('ori_y_range', [-0.1, 0.1])
        self.vel_x_range = setpoint_config.get('vel_x_range', [-0.6, 0.6])
        
        # Fixed values
        self.frame_id = "race2_auv/world_ned"
        self.child_frame_id = "race2_auv/cg_link"
        self.control_mode = "4dof"
        
        self.current_setpoint = None
        print(f"SetpointManager initialized with ranges:")
        print(f"  pos_z: {self.pos_z_range}")
        print(f"  ori_z: {self.ori_z_range}")
        print(f"  ori_y: {self.ori_y_range}")
        print(f"  vel_x: {self.vel_x_range}")
    
    def generate_new_setpoint(self):
        """Generate a new random setpoint for a new episode"""
        msg = ControlProcess()
        msg.header = Header()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        msg.control_mode = self.control_mode
        
        # Generate random values within specified ranges
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
        return msg
    
    def get_current_setpoint(self):
        """Get the current setpoint"""
        return self.current_setpoint
    
    def get_setpoint_info(self):
        """Get human-readable info about current setpoint"""
        if self.current_setpoint is None:
            return "No setpoint generated yet"
        
        return (f"Setpoint - pos_z: {self.current_setpoint.position.z:.2f}, "
                f"ori_z: {self.current_setpoint.orientation.z:.2f}, "
                f"ori_y: {self.current_setpoint.orientation.y:.2f}, "
                f"vel_x: {self.current_setpoint.velocity.x:.2f}")

class AUVEnvNode(Node):
    """Node to handle ROS2 communications for the AUV environment"""
    
    def __init__(self, config):
        super().__init__('auv_env_node')
        
        self.config = config
        
        self.thruster_size = self.config['environment']['thruster_size']
        servo_size = self.config['environment']['servo_joints_size']
        
        # Initialize setpoint manager
        self.setpoint_manager = SetpointManager(config)
        
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
        
        # Create setpoint publisher
        self.setpoint_publisher = self.create_publisher(
            ControlProcess,
            '/race2_auv/controller/process/set_point',
            10
        )
        
        self.smoothed_surge_pub = self.create_publisher(Float64, '/race2/smoothed_surge_velocity', 1)

        # Create timer for continuous setpoint publishing
        self.setpoint_publish_rate = 5.0  # 5 Hz
        self.setpoint_timer = self.create_timer(
            1.0 / self.setpoint_publish_rate, 
            self.publish_current_setpoint_callback
        )
        
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
        
        self.create_subscription(
            Imu,
            '/race2_auv/imu/data',
            self.imu_callback,
            1
        )
        
        # For communication between callbacks and the environment
        self.new_state_available = False
        self.new_error_available = False
        self.last_action_timestamp = 0
        self.last_state_timestamp = 0
        self.last_error_timestamp = 0
        
        # Declare parameters
        self.declare_parameter('max_steps', 500)
        self.max_steps = self.get_parameter('max_steps').value
    
    def publish_current_setpoint_callback(self):
        """Continuously publish the current setpoint """
        if self.setpoint_manager.current_setpoint is not None:
            # Update timestamp and publish
            self.setpoint_manager.current_setpoint.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_publisher.publish(self.setpoint_manager.current_setpoint)
            self.get_logger().debug("Published current setpoint")
    
    def publish_new_setpoint(self):
        """Generate and publish a new setpoint for the episode"""
        setpoint = self.setpoint_manager.generate_new_setpoint()
        setpoint.header.stamp = self.get_clock().now().to_msg()
        
        # Publish the setpoint immediately
        self.setpoint_publisher.publish(setpoint)
        
        # Add some debugging
        self.get_logger().info(f"Published new setpoint: {self.setpoint_manager.get_setpoint_info()}")
        self.get_logger().info(f"Setpoint publisher topic: {self.setpoint_publisher.topic_name}")
        self.get_logger().info(f"Setpoint publisher subscriber count: {self.setpoint_publisher.get_subscription_count()}")
        
        return setpoint
    
    def get_current_setpoint_info(self):
        """Get information about the current setpoint"""
        return self.setpoint_manager.get_setpoint_info()
    
    def error_callback(self, data):
        """Process error updates"""
        self.get_logger().debug("Error callback triggered!")
        self.last_error_timestamp = time.time()
        self.position_err = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_err = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
        
        # Smooth surge velocity error (v_err[0])
        if not hasattr(self, 'surge_error_filter'):
            self.surge_error_filter = self.v_err[0]  # Initialize on first call
        
        alpha = 0.65  # Smoothing factor
        self.surge_error_filter = alpha * self.surge_error_filter + (1 - alpha) * self.v_err[0]
        
        # Use smoothed surge error
        smoothed_v_err = self.v_err.copy()
        smoothed_v_err[0] = self.surge_error_filter
        
        # Convert orientation errors to sin/cos representation to avoid angle wrapping issues
        roll_sin_err = np.sin(self.orientation_err[0])
        roll_cos_err = np.cos(self.orientation_err[0])
        pitch_sin_err = np.sin(self.orientation_err[1])
        pitch_cos_err = np.cos(self.orientation_err[1])
        yaw_sin_err = np.sin(self.orientation_err[2])
        yaw_cos_err = np.cos(self.orientation_err[2])
        
        # Update the state error array with the smoothed surge error
        self.state_err = np.concatenate([
            self.position_err[2:3],
            smoothed_v_err[:2],  # Using smoothed surge error here
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
        
        # Smooth surge velocity state (v_state[0])
        if not hasattr(self, 'surge_state_filter'):
            self.surge_state_filter = self.v_state[0]  # Initialize on first call
        
        alpha = 0.65  # Same smoothing factor as error callback
        self.surge_state_filter = alpha * self.surge_state_filter + (1 - alpha) * self.v_state[0]
        
        # Use smoothed surge state
        smoothed_v_state = self.v_state.copy()
        smoothed_v_state[0] = self.surge_state_filter
        
        # Publish smoothed surge velocity
        from std_msgs.msg import Float64
        surge_msg = Float64()
        surge_msg.data = float(self.surge_state_filter)
        self.smoothed_surge_pub.publish(surge_msg)
        
        # Convert orientation to sin/cos representation to avoid angle wrapping issues
        roll_sin = np.sin(self.orientation_state[0])
        roll_cos = np.cos(self.orientation_state[0])
        pitch_sin = np.sin(self.orientation_state[1])
        pitch_cos = np.cos(self.orientation_state[1])
        yaw_sin = np.sin(self.orientation_state[2])
        yaw_cos = np.cos(self.orientation_state[2])
        
        # Update current state for RL agent with smoothed surge velocity
        self.current_state = np.concatenate([
            self.position_state[2:3],
            smoothed_v_state[:2],  # Using smoothed surge velocity here
            np.array([roll_sin, roll_cos, pitch_sin, pitch_cos, yaw_sin, yaw_cos]),
        ])
        
        # Flag indicating we have a new state after the last action
        if hasattr(self, 'last_action_timestamp') and self.last_state_timestamp > self.last_action_timestamp:
            self.new_state_available = True
    
    def imu_callback(self, data):
        """Process IMU data including linear accelerations"""
        self.get_logger().debug("IMU callback triggered!")
        
        # Store linear accelerations (NEW)
        self.linear_acceleration[0] = data.linear_acceleration.x
        self.linear_acceleration[1] = data.linear_acceleration.y
        self.linear_acceleration[2] = data.linear_acceleration.z

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
        
        heave_bow = action[0]
        heave_stern = action[1]
        # surge_command = action[2]     # Pure surge desire
        # yaw_command = action[3]       # Pure yaw desire (usually 0 for straight)
        surge_port = action[2]
        surge_starboard = action[3]
        # # Convert to physical thrusters
        # surge_port = max(-1.0, min(1.0, 0.8 * surge_command + 0.2 * yaw_command))
        # surge_starboard = max(-1.0, min(1.0, 0.8 * surge_command - 0.2 * yaw_command))
    
        thruster_cmds = [heave_bow, heave_stern, surge_port, surge_starboard]
        
        # Handle servo commands if present
        servo_angles_normalized = action[4:] if len(action) > 4 else []
        
        # Convert servo commands to radians if needed
        servo_angles_rad = []
        for angle in servo_angles_normalized:
            servo_angles_rad.append(self.convert_servo_command_to_radians(angle))
        
        # Map to appropriate publishers
        thruster_mapping = [
            ('heave_bow', thruster_cmds[0]),
            ('heave_stern', thruster_cmds[1]),
            ('surge_port', thruster_cmds[2]),
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
    
    # def __init__(self):
    def __init__(self, csv_directory=None):
        super(AUVEnv, self).__init__()
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config_sac.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init(args=None)
        
        self.csv_mode = csv_directory is not None
        
        if self.csv_mode:
            print("Initializing AUVEnv in CSV mode...")
            self.csv_manager = CSVDataManager(csv_directory, self.config)
            self.node = None  # No ROS2 node needed
        else:
            print("Initializing AUVEnv in online ROS2 mode...")
            # Initialize ROS2 if not already done
            if not rclpy.ok():
                rclpy.init(args=None)
            
            # Create ROS2 node
            self.node = AUVEnvNode(self.config)
            self.csv_manager = None
        
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
        
        # Update observation space to include angular rates and all velocities
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(24,),  # Updated size: 10 errors + 3 velocities + 3 angular rates
            dtype=np.float32
        )
        
        # Initialize episode-related variables
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_count = 0  # Track total episodes
        
        # Initialize history arrays for smoothness calculations
        self.joint_positions_history = np.zeros((10, self.num_servos))  # Store last 10 servo positions
        self.u_prev = np.zeros((100, self.num_thrusters))  # Store last 10 thruster commands
        self.thruster_command_action_prev = np.zeros((100, self.num_thrusters))  # Store last 10 thruster actions
        
        # Initialize tracking variables
        self.thruster_action = np.zeros(self.num_thrusters)
        self.joint_angles = np.zeros(self.num_servos)
        self.last_action = np.zeros(thruster_size + servo_size)
        
        # Initialize coupling-aware reward calculator
        self.coupling_calculator = None
        
        # Store episode setpoint info for logging
        self.current_episode_setpoint_info = "No setpoint set yet"
    
    def reset(self, seed=None):
        """Reset the environment to initial state and return the initial observation"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_count += 1
        
        if self.csv_mode:
            episode_info = self.csv_manager.reset_episode(episode_length=500)
            self.current_episode_setpoint_info = f"CSV Episode {self.episode_count} (start: {episode_info['start_timestamp']:.1f}s)"
            print(f"Episode {self.episode_count} started - CSV data from {episode_info['start_timestamp']:.1f}s")
            
            initial_observation = self.csv_manager._get_current_state_observation()
            
        else:
            setpoint = self.node.publish_new_setpoint()
            self.current_episode_setpoint_info = self.node.get_current_setpoint_info()
            
            print(f"Episode {self.episode_count} started - {self.current_episode_setpoint_info}")
            
            # time.sleep(0.05)
            # self._spin_node(timeout_sec=0.3)
            
            # setpoint = self.node.publish_new_setpoint()
            # time.sleep(0.01)
            # self._spin_node(timeout_sec=0.3)
            
            time.sleep(0.1)           # Was: 0.05
            self._spin_node(timeout_sec=0.1)    # Was: 0.3

            setpoint = self.node.publish_new_setpoint()
            time.sleep(0.05)          # Was: 0.01  
            self._spin_node(timeout_sec=0.1)    # Was: 0.3

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
            
            initial_observation = np.concatenate([
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

        if self.coupling_calculator is None:
            from coupling_rewards import CouplingAwareRewardCalculator
            self.coupling_calculator = CouplingAwareRewardCalculator(self.config)

        info = {
            'episode_count': self.episode_count,
            'setpoint_info': self.current_episode_setpoint_info,
            'mode': 'csv' if self.csv_mode else 'online'
        }
        
        if self.csv_mode:
            info['csv_episode_info'] = episode_info
            stats = self.csv_manager.get_dataset_stats()
            info['dataset_stats'] = stats
        else:
            info['setpoint_values'] = {
                'pos_z': setpoint.position.z,
                'ori_z': setpoint.orientation.z,
                'ori_y': setpoint.orientation.y,
                'vel_x': setpoint.velocity.x
            }
        
        return initial_observation, info
    
    def step(self, action):
        """Execute action in the environment and return next state, reward, termination flag, etc."""
        
        if self.csv_mode:
            return self._step_csv_mode(action)
        else:
            return self._step_online_mode(action)

    def _step_csv_mode(self, action):
        """Handle step in CSV mode using recorded PID actions"""
        
        # Advance to next step in synchronized data
        episode_done, step_info = self.csv_manager.step_episode()
        
        # Get current observation from CSV data
        observation = self.csv_manager._get_current_state_observation()
        
        # Get state error array for reward calculation
        state_error_array = self.csv_manager.get_state_error_array()
        
        #Use recorded PID actions from CSV
        recorded_pid_action = self.csv_manager.get_recorded_action_at_current_step()
        
        # Store recorded actions for reward calculation
        self.thruster_action = recorded_pid_action[:4]  # PID thruster commands
        self.joint_angles = recorded_pid_action[4:6] if len(recorded_pid_action) > 4 else np.zeros(2)  # PID servo commands
        self.last_action = recorded_pid_action.copy()  # PID action, not agent action
        
        # Create mock node with recorded PID actions for reward calculation
        if not hasattr(self, '_mock_node'):
            self._mock_node = type('MockNode', (), {})()
        
        self._mock_node.thrust_heave_bow = recorded_pid_action[0]      # What PID actually did
        self._mock_node.thrust_heave_stern = recorded_pid_action[1]    # What PID actually did
        self._mock_node.thrust_surge_port = recorded_pid_action[2]     # What PID actually did
        self._mock_node.thrust_surge_starboard = recorded_pid_action[3] # What PID actually did
        
        # Use mock node for reward calculation
        original_node = self.node
        self.node = self._mock_node
        
        # Calculate reward based on PID's performance
        reward = self.calculate_reward(state_error_array)
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
        
        # Restore original node reference
        self.node = original_node
        
        # Store raw reward
        if not hasattr(self, 'episode_rewards'):
            self.episode_rewards = []
        self.episode_rewards.append(reward)
        
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = episode_done
        truncated = False
        
        max_steps = getattr(self, 'max_steps', 500)
        if self.episode_step >= max_steps:
            truncated = True
            
        if step_info.get('reason') == 'data_complete':
            terminated = True
        
        self.episode_step += 1
        
        info = {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward, 
            'setpoint_info': self.current_episode_setpoint_info,
            'csv_step_info': step_info,
            'simulation_time': step_info.get('timestamp', self.episode_step * 0.1),
            'data_frequency': '10Hz (0.1s per step)',
            'recorded_action': recorded_pid_action,  # The PID action that was actually used
            'agent_action': action  # The action the agent wanted to take (for comparison/analysis)
        }
        
        return observation, reward, terminated, truncated, info

    
    def _step_online_mode(self, action):
        """Execute action in the environment and return next state, reward, termination flag, etc."""

        state_error_array = None

        thruster_cmds, servo_angles_rad = self.node.publish_action(action, self.num_thrusters, self.num_servos)
        self.thruster_action = thruster_cmds
        self.joint_angles = servo_angles_rad
        self.last_action = action.copy()
        
        timeout_sec = 0.3
        start_time = time.time()
        
        while not (self.node.new_state_available and self.node.new_error_available):
            self._spin_node(timeout_sec=0.28)
            if time.time() - start_time > timeout_sec:
                print("Warning: Timeout waiting for state/error updates")
                break
                
        if self.node.new_state_available and self.node.new_error_available:
            depth = self.node.position_state[2:3]
            depth_error = self.node.position_err[2:3]
            surge_velocity_error = self.node.v_err[0:1]
            sway_velocity_error = self.node.v_err[1:2]

            roll_error = self.node.orientation_err[0:1]
            pitch_error = self.node.orientation_err[1:2]
            yaw_error = self.node.orientation_err[2:3]

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
            
            state_error_array = np.concatenate([
                depth_error,
                surge_velocity_error,
                sway_velocity_error,
                self.node.v_err[2:3], #heave error
                # np.array([np.sin(self.node.orientation_err[0]), np.cos(self.node.orientation_err[0])]),
                # np.array([np.sin(self.node.orientation_err[1]), np.cos(self.node.orientation_err[1])]),
                # np.array([np.sin(self.node.orientation_err[2]), np.cos(self.node.orientation_err[2])])
                roll_error,
                pitch_error,
                yaw_error
            ])

            terminated = False

        else:
            print("Warning: No new state/error available, returning dummy observation")
            observation = np.zeros(24)
            terminated = True
            state_error_array = np.zeros(7)

        reward = self.calculate_reward(state_error_array)

        if isinstance(reward, np.ndarray):
            reward = float(reward.item())

        if not hasattr(self, 'episode_rewards'):
            self.episode_rewards = []
        self.episode_rewards.append(reward)

        self.episode_reward += reward

        truncated = False
        if self.episode_step >= self.node.max_steps:
            truncated = True
            
        self.episode_step += 1

        info = {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'setpoint_info': self.current_episode_setpoint_info
        }

        return observation, reward, terminated, truncated, info
    
    def _spin_node(self, timeout_sec=0.1):
        """Process ROS callbacks for a limited time"""
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.01)
    
    def calculate_reward(self, state_error_array):
        """Calculate coupling-aware reward with physics-based coupling"""
        
        # Get coupling method from config
        coupling_method = self.config.get('coupling', {}).get('method', 'v1')
        
        # Calculate coupling-aware performance error
        if coupling_method == 'v1':
            performance_error = self.coupling_calculator.calculate_coupling_aware_reward_v1(
                state_error_array, self.episode_step)
        elif coupling_method == 'v2':
            performance_error = self.coupling_calculator.calculate_coupling_aware_reward_v2(
                state_error_array, self.episode_step)
        elif coupling_method == 'v3':
            performance_error = self.coupling_calculator.calculate_coupling_aware_reward_v3(
                state_error_array, self.episode_step)
        elif coupling_method == 'v4':
            performance_error = self.coupling_calculator.calculate_coupling_aware_reward_v4_enhanced(
                state_error_array, self.episode_step)
        elif coupling_method == 'standard':
            # Fallback to original reward calculation
            performance_error = self._calculate_standard_performance_error(state_error_array)
        else:
            # Default to v1 if method not recognized
            print(f"Warning: Unknown coupling method '{coupling_method}', defaulting to v1")
            performance_error = self.coupling_calculator.calculate_coupling_aware_reward_v1(
                state_error_array, self.episode_step)
        
        # Apply performance weight
        w = self.config['reward_function']
        coupling_performance_reward = w['w1'] * performance_error
        
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
        if self.num_servos > 0:
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
        servo_angle_penalty = np.linalg.norm(self.joint_angles) if len(self.joint_angles) > 0 else 0

        # Update thruster_command_action_prev with clear logic
        self.thruster_command_action_prev = np.vstack((
            self.thruster_command_action_prev[1:],  # Keep all except the first
            self.thruster_action  # Add the newest action
        ))

        # Thruster action penalty
        thruster_action_penalty = np.sum(np.abs(self.thruster_action - np.average(self.thruster_command_action_prev, axis=0)))

        # Thruster Direction Change Penalty
        direction_change_penalty = w['w8'] * thruster_action_penalty ** 2  # Quadratic penalty

        # Final reward calculation with coupling-aware performance term
        reward = (
            coupling_performance_reward -  # This now includes coupling awareness
            w['w2'] * servo_smoothness_penalty -
            w['w3'] * thruster_usage_penalty -
            w['w4'] * thruster_smoothness_penalty -
            w['w5'] * servo_angle_penalty -
            w['w6'] * thruster_delta_reward -
            w['w7'] * thruster_action_penalty -
            w['w8'] * direction_change_penalty
        )
        
        # Optional: Add coupling-specific logging for debugging
        if hasattr(self.coupling_calculator, 'error_history') and len(self.coupling_calculator.error_history) > 0:
            current_errors = self.coupling_calculator.error_history[-1]
            
            # Log coupling metrics every 50 steps for debugging
            if self.episode_step % 50 == 0:
                diagnostics = self.coupling_calculator.get_diagnostics()
                if diagnostics:
                    print(f"Step {self.episode_step}: Coupling Metrics")
                    print(f"  Learning Phase: {diagnostics.get('learning_phase', 'unknown')}")
                    print(f"  Surge-Yaw: surge={current_errors['surge']:.4f}, yaw={current_errors['yaw']:.4f}")
                    print(f"  Pitch-Depth: pitch={current_errors['pitch']:.4f}, depth={current_errors['depth']:.4f}")
                    print(f"  Surge-Yaw Magnitude: {diagnostics['surge_yaw_magnitude']:.4f}")
                    print(f"  Pitch-Depth Magnitude: {diagnostics['pitch_depth_magnitude']:.4f}")
                    print(f"  Coupling Performance: {coupling_performance_reward:.4f}")
                    print(f"  Total Reward: {reward:.4f}")
        
        return reward
    
    def _calculate_standard_performance_error(self, state_error_array):
        """Standard performance error calculation for fallback"""
        original_weights = np.array(self.config['reward_function']['state_error_weights'])
        # state_error_weights = self._expand_weights_for_sincos(original_weights, state_error_array)

        state_error_weights = original_weights

        error_column = state_error_array.reshape(-1, 1)
        error_row = state_error_array.reshape(1, -1)
        weights_diag = np.diag(state_error_weights)
        
        performance_error = error_row @ weights_diag @ error_column
        return float((-performance_error).item())
    
    def _expand_weights_for_sincos(self, original_weights, state_error_array):
        """Expand weights array to account for sin/cos representation"""
        # Dynamically size based on actual state_error_array
        target_size = len(state_error_array)
        state_error_weights = np.zeros(target_size)
        
        # Fill as much as we can from original_weights
        original_weights = np.array(original_weights)
        
        if len(original_weights) == 7 and target_size == 10:
            # Standard case: 7 original -> 10 expanded (sin/cos for 3 angles)
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
            
        elif len(original_weights) == target_size:
            # Already the right size, use as-is
            state_error_weights = original_weights.copy()
            
        else:
            # General case: fill what we can, pad/truncate as needed
            min_size = min(len(original_weights), target_size)
            state_error_weights[:min_size] = original_weights[:min_size]
            
            # If we need more weights than we have, use the last weight value
            if target_size > len(original_weights):
                last_weight = original_weights[-1] if len(original_weights) > 0 else 0.001
                state_error_weights[len(original_weights):] = last_weight
        
        return state_error_weights
    
    def get_coupling_diagnostics(self):
        """Get detailed coupling diagnostics for analysis"""
        if self.coupling_calculator is None:
            return None
        return self.coupling_calculator.get_diagnostics()

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'node') and self.node is not None:
            self.node.destroy_node()


def main(args=None, csv_directory=None):
    """
    Main function to initialize and return the environment.
    
    Args:
        args: ROS2 args (only used in online mode)
        csv_directory: Path to CSV directory for offline training (optional)
    """
    
    if csv_directory is None and not rclpy.ok():
        rclpy.init(args=args)
    
    return AUVEnv(csv_directory=csv_directory)


if __name__ == "__main__":
    main()