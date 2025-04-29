import gym
from gym import spaces
import numpy as np
import os,time, datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32
from geometry_msgs.msg import TwistStamped
from mvp_msgs.msg import ControlProcess
import yaml
from rclpy.clock import Clock

###############################################
##### Gymnasiyum required functions   #########
###############################################

class AUVEnv(gym.Env):
    """Custom AUV Environment that follows gym interface"""
    
    def __init__(self):
        super(AUVEnv, self).__init__()
        
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config_ddpg.yaml')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        rclpy.init(args=None)
        self.node = rclpy.create_node('auv_env_node')

        # self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        

        thruster_size = self.config['environment']['thruster_size']
        servo_size = self.config['environment']['servo_joints_size']
        

        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(thruster_size + servo_size,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(12,),  #state dimension
            dtype=np.float32
        )
        
        self.thruster_pubs = {
            'heave_bow': self.node.create_publisher(Float64, '/race2_auv/control/thruster/heave_bow', 1),
            'heave_stern': self.node.create_publisher(Float64, '/race2_auv/control/thruster/heave_stern', 1),
            'surge_port': self.node.create_publisher(Float64, '/race2_auv/control/thruster/surge_port', 1),
            'surge_starboard': self.node.create_publisher(Float64, '/race2_auv/control/thruster/surge_starboard', 1),
            'port_servo': self.node.create_publisher(Float64, '/race2_auv/control/surge_port_servo', 1),
            'starboard_servo': self.node.create_publisher(Float64, '/race2_auv/control/surge_starboard_servo', 1)
        }

        
        self.node.create_subscription(ControlProcess,  
                                    '/race2_auv/controller/process/value',
                                    self.state_callback,
                                    10)

        # self.node.create_subscription(ControlProcess, 
        #                             '/race2_auv/controller/process/setpoint', 
        #                             self.setpoint_callback,
        #                             10)

        self.node.create_subscription(ControlProcess, 
                                    '/race2_auv/controller/process/error', 
                                    self.error_callback,
                                    10)

        self.node.create_subscription(Float64, 
                                    '/race2_auv/control/surge_port_servo', 
                                    self.update_joint_port, 
                                    10)

        self.node.create_subscription(Float64, 
                                    '/race2_auv/control/surge_starboard_servo', 
                                    self.update_joint_starboard, 
                                    10)

        self.node.create_subscription(Float64, 
                                    '/race2_auv/control/thruster/heave_bow', 
                                    self.update_thrust_heave_bow, 
                                    10)

        self.node.create_subscription(Float64, 
                                    '/race2_auv/control/thruster/surge_port', 
                                    self.update_thrust_surge_port, 
                                    10)

        self.node.create_subscription(Float64, 
                                    '/race2_auv/control/thruster/surge_starboard', 
                                    self.update_thrust_surge_starboard, 
                                    10)

        self.node.create_subscription(Float64, 
                                    '/race2_auv/control/thruster/heave_stern', 
                                    self.update_thrust_heave_stern, 
                                    10)

        
        # State tracking
        self.current_actor_state = None
        self.current_critic_state = None
        self.prev_actor_state = None
        self.prev_critic_state = None
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
        self.num_thrusters = 2
        self.num_servos = 0
        self.thruster_action = np.zeros(self.num_thrusters)  # For the 4 thrusters
        self.joint_angles = np.zeros(self.num_servos)     # For the 2 servo angles
        
        self.joint_angles_port = 0.0
        self.joint_angles_starboard = 0.0

        self.thrust_heave_bow = 0.0
        self.thrust_surge_port = 0.0
        self.thrust_surge_starboard = 0.0
        self.thrust_heave_stern = 0.0

        # Initialize history arrays for smoothness calculations
        self.joint_positions_history = np.zeros((10, self.num_servos))  # Store last 10 servo positions
        self.u_prev = np.zeros((10, self.num_thrusters))  # Store last 10 thruster commands
        self.thruster_command_action_prev = np.zeros((10,self.num_thrusters))  # Store last 10 thruster actions



        self.node.declare_parameter('max_steps', 500)
        self.node.max_steps = self.node.get_parameter('max_steps').value
        

    def reset(self):
        # Reset the environment to initial state
        # This might involve sending reset commands to your ROS system
        # Return the initial observation

        self.episode_step = 0
        self.episode_reward = 0.0
        # self.publish_action(action)

        depth = self.position_state[2:3]
        surge = self.v_state[0:1]     
        sway = self.v_state[1:2]      
        heave = self.v_state[2:3]        
        roll = self.orientation_state[0:1]         
        pitch = self.orientation_state[1:2]         
        yaw = self.orientation_state[2:3]           

        # Extract error variables
        depth_error = self.position_err[2:3]        
        surge_error = self.v_err[0:1]   
        sway_error = self.v_err[1:2]       
        heave_error = self.v_err[2:3]      
        roll_error = self.orientation_err[0:1]     
        pitch_error = self.orientation_err[1:2]     
        yaw_error = self.orientation_err[2:3]    

        # Extract rate variables
        roll_rate = self.omega_ref_state[0:1]
        pitch_rate = self.omega_ref_state[1:2]
        yaw_rate = self.omega_ref_state[2:3]
        roll_rate_error = self.omega_ref_err[0:1]
        pitch_rate_error = self.omega_ref_err[1:2] 
        yaw_rate_error = self.omega_ref_err[2:3]

        # Create actor state
        initial_observation = np.concatenate([
            depth_error, 
            surge_error, 
            sway_error, 
            roll_error, 
            pitch_error, 
            yaw_error,  
            depth, 
            surge, 
            sway, 
            roll, 
            pitch, 
            yaw           
        ])

        return initial_observation
        
    def step(self, action):
        # Execute action in the environment (send to ROS)
        # Wait for new state after action
        # Calculate reward
        # Determine if episode is done
        # Return (observation, reward, done, info)

        # action = self.agent.get_action(self.actor_state, add_noise=self.training_mode)
        # action = np.reshape(action, -1) 
        
        self.publish_action(action)

        time.sleep(0.1)

        updated_depth = self.position_state[2:3]
        updated_surge = self.v_state[0:1]     
        updated_sway = self.v_state[1:2]      
        updated_heave = self.v_state[2:3]        
        updated_roll = self.orientation_state[0:1]         
        updated_pitch = self.orientation_state[1:2]         
        updated_yaw = self.orientation_state[2:3]           

        updated_depth_error = self.position_err[2:3]        
        updated_surge_error = self.v_err[0:1]   
        updated_sway_error = self.v_err[1:2]       
        updated_heave_error = self.v_err[2:3]      
        updated_roll_error = self.orientation_err[0:1]     
        updated_pitch_error = self.orientation_err[1:2]     
        updated_yaw_error = self.orientation_err[2:3] 

        observation = np.concatenate([
            updated_depth_error, 
            updated_surge_error, 
            updated_sway_error,
            updated_roll_error, 
            updated_pitch_error, 
            updated_yaw_error,
            updated_depth, 
            updated_surge, 
            updated_sway,
            updated_roll, 
            updated_pitch, 
            updated_yaw
        ])

        reward = self.calculate_reward()
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())

        self.episode_reward += reward

        # done = False
        # if self.episode_step >= self.node.max_steps:
        #     done = True
        terminated = False  # episode ended because the task succeeded or failed.
        truncated = False

        if self.episode_step >= self.node.max_steps:
            truncated = True

        # Increment step counter
        self.episode_step += 1

        return observation, reward, terminated, truncated, {}
        


    def render(self, mode='human'):
        # Optional: visualization
        pass

###############################################
########## ROS specific functions  ############
###############################################

    def error_callback(self, data):
        """Process error updates"""

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

    def publish_action(self, action):
        """Publish actions to ROS2 topics"""
        # Split action into thruster commands and servo angles

        thruster_cmds = action[:self.num_thrusters]
        # servo_angles_normalized = action[4:]    

        # servo_angles_rad = [
        #     self.convert_servo_command_to_radians(servo_angles_normalized[0]),
        #     self.convert_servo_command_to_radians(servo_angles_normalized[1])
        # ]

        # Store for reward calculation
        self.thruster_action = thruster_cmds
        # self.joint_angles = servo_angles_rad
        
        # Map to appropriate publishers
        #All DOFs
        thruster_mapping = [
            ('heave_bow',  0.8 * thruster_cmds[0]),
            ('heave_stern', 0.8 * thruster_cmds[1])
            # ('surge_port',  0.0 * thruster_cmds[0]),
            # ('surge_starboard', 0.0 * thruster_cmds[1]),
            # ('port_servo', 0.0 *servo_angles_rad[0]),
            # ('starboard_servo',0.0 * servo_angles_rad[1])
        ]

        # Publish commands
        for name, value in thruster_mapping:
            msg = Float64()
            msg.data = float(value)
            self.thruster_pubs[name].publish(msg)
    
        # Record action timestamp
        self.last_action_timestamp = time.time()
        self.last_action = action.copy()
        
        # Reset the new state flag since we're waiting for a new state after this action
        self.new_state_available = False
        self.new_error_available = False

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
        
        Args:
            normalized_command (float): Normalized command between -1 and 1
            min_angle_rad (float): Minimum angle in radians (default -0.5 rad, ~-28.6°)
            max_angle_rad (float): Maximum angle in radians (default 0.5 rad, ~28.6°)
        
        Returns:
            float: Angle in radians clamped to the specified range
        """
        # Ensure normalized command is within [-1, 1]
        normalized_command = np.clip(normalized_command, -1.0, 1.0)
        
        min_angle_rad = self.config['min_servo_angle_rad']
        max_angle_rad = self.config['max_servo_angle_rad']
        # Map from [-1, 1] to [min_angle_rad, max_angle_rad]
        angle_rad = min_angle_rad + (normalized_command + 1.0) * (max_angle_rad - min_angle_rad) / 2.0
        
        return angle_rad
    
    def calculate_reward(self):
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
        # Compute performance error (quadratic penalty)
        # weighted_errors =  state_error_weights * error
        # performance_error = np.sum(weighted_errors ** 2)
        error_column = error.reshape(-1, 1)
        performance_error = np.dot(error , np.diag(state_error_weights))
        performance_error = np.dot(performance_error,error_column)
        performance_error = np.exp(-performance_error)

        # Servo smoothness penalty using sine and cosine components
        servo_smoothness_penalty = 0
        delta_theta = np.zeros(self.num_servos)

        for i in range(self.num_servos):
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
            # self.thrust_surge_port,
            # self.thrust_surge_starboard,
            self.thrust_heave_stern
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
            -w1 * performance_error + #positive without exponential
            w2 * servo_smoothness_penalty +
            w3 * thruster_usage_penalty +
            w4 * thruster_smoothness_penalty +
            w5 * servo_angle_penalty +
            w6 * thruster_delta_reward +
            w7 * thruster_action_penalty +
            w8 * direction_change_penalty
        )
        
        # self.get_logger().info(f"Performance Error Contribution: {-w1 * performance_error}")
        # self.get_logger().info(f"Servo Smoothness Penalty Contribution: {-w2 * servo_smoothness_penalty}")
        # self.get_logger().info(f"Thruster Usage Penalty Contribution: {-w3 * thruster_usage_penalty}")
        # self.get_logger().info(f"Thruster Smoothness Penalty Contribution: {-w4 * thruster_smoothness_penalty}")
        # self.get_logger().info(f"Servo Angle Penalty Contribution: {-w5 * servo_angle_penalty}")
        # self.get_logger().info(f"Thruster Delta Reward Contribution: {-w6 * thruster_delta_reward}")
        # self.get_logger().info(f"Thruster action penalty: {w7 * thruster_action_penalty}")
        # self.get_logger().info(f"Reward: {reward}")
        return reward
    
    def load_config():
        """Load configuration from YAML file"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_ddpg.yaml')
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")