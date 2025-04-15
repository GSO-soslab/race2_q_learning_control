import os,time, datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import TwistStamped
from mvp_msgs.msg import ControlProcess
from sensor_msgs.msg import Imu
import DDPG
import torch
import numpy as np

class DDPG_ROS(Node):
    """DDPG Agent for AUV control integrated with ROS2"""
    def __init__(self, config):
        super().__init__('ddpg_auv_control')
        
        self.config = config

        # Define separate dimensions for actor and critic
        self.actor_state_dim = 13   # Example: position_err, v_err, orientation_err
        self.critic_state_dim = 19  # error states + commands
        self.action_dim = 6  # 4 thrusters + 2 servo angles
        self.action_bound = 1.0  # All commands between -1 and 1
        
        # Create DDPG agent with separate state dimensions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        self.agent = DDPG(self.actor_state_dim, self.critic_state_dim, self.action_dim, self.action_bound, config, device=self.device)

        # Rest of the initialization code remains the same
        # ROS2 publishers for each actuator
        self.thruster_pubs = {
            'heave_bow': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_bow', 1),
            'heave_stern': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_stern', 1),
            'surge_port': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_port', 1),
            'surge_starboard': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_starboard', 1),
            'port_servo': self.create_publisher(Float64, '/race2_auv/control/surge_port_servo', 1),
            'starboard_servo': self.create_publisher(Float64, '/race2_auv/control/surge_starboard_servo', 1)
        }
        
        self.imu_subscription = self.create_subscription(
                                Imu,
                                '/race2_auv/imu/data',
                                self.imu_callback,
                                10)

        self.dvl_subscription = self.create_subscription(
                                TwistStamped,  # Or stonefish_ros2/DVL 
                                '/race2_auv/dvl/twist',  # Or the stonefish raw topic
                                self.dvl_callback,
                                10)
        
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
                                '/race2_auv/control/thruster/heave_stern', 
                                self.update_thrust_heave_stern, 
                                10)
        
        # Training parameters
        self.declare_parameter('training_mode', True)
        self.declare_parameter('max_steps', 500)
        self.declare_parameter('model_path', '')
        
        self.declare_parameter('max_episodes', 10000)  # Default 1000 episodes
        self.max_episodes = self.get_parameter('max_episodes').value

        self.training_mode = self.get_parameter('training_mode').value
        self.max_steps = self.get_parameter('max_steps').value
        model_path = self.get_parameter('model_path').value
        
        # State tracking
        self.current_actor_state = None
        self.current_critic_state = None
        self.prev_actor_state = None
        self.prev_critic_state = None
        self.prev_action = None
        
        # IMU data init
        self.linear_vel = np.zeros(3)  # vt
        self.angular_vel = np.zeros(3)  # ωt
        self.linear_accel = np.zeros(3)  # v̇t
        self.angular_accel = np.zeros(3)  # ω̇t
        self.prev_angular_vel = np.zeros(3)  # For calculating angular acceleration
        self.prev_time_imu = self.get_clock().now()
        self.previous_commands = np.zeros(6)  # ut-1 (example size, adjust as needed)
        self.velocity_error = np.zeros(3)
        self.velocity_setpoint = [0.2,0.0,0.0]
        
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
        self.thrust_heave_stern = 0.0

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
        self.timer = self.create_timer(1/10, self.control_loop)  # 100 Hz control loop
        

    def imu_callback(self, msg):
        # Extract angular velocity
        self.angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        # Extract linear acceleration
        self.linear_accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        
        # Calculate angular acceleration
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time_imu).nanoseconds / 1e9
        if dt > 0:
            self.angular_accel = (self.angular_vel - self.prev_angular_vel) / dt
        
        # Update previous values
        self.prev_angular_vel = self.angular_vel.copy()
        self.prev_time_imu = current_time
        
        # Construct state vector whenever we get new data
        self.update_state()
    
    def dvl_callback(self, msg):
        # Extract linear velocity from DVL
        self.linear_vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])
        
        # Calculate velocity error
        self.velocity_error = self.linear_vel - self.velocity_setpoint
        
        # Construct state vector whenever we get new data
        self.update_state()
    
    def update_state(self):
        # Construct the complete state vector according to the paper
        state = np.concatenate([
            self.linear_vel,      # vt
            self.angular_vel,     # ωt
            self.linear_accel,    # v̇t
            self.angular_accel,   # ω̇t
            self.previous_commands, # ut-1
            self.velocity_error   # et
        ])
        
        # self.get_logger().info(f"State updated: {state}")
        return state

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
        self.omega_ref_err[:3],
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
    
    def publish_action(self, action):
        """Publish actions to ROS2 topics"""
        # Split action into thruster commands and servo angles

        thruster_cmds = action[:4]
        servo_angles_normalized = action[4:]    

        servo_angles_rad = [
            self.convert_servo_command_to_radians(servo_angles_normalized[0]),
            self.convert_servo_command_to_radians(servo_angles_normalized[1])
        ]

        # Store for reward calculation
        self.thruster_action = thruster_cmds
        self.joint_angles = servo_angles_rad
        
        # Map to appropriate publishers
        #All DOFs
        thruster_mapping = [
            ('heave_bow', thruster_cmds[2]),
            ('heave_stern', thruster_cmds[3]),
            ('surge_port',  0.6 * thruster_cmds[0]),
            ('surge_starboard', 0.6 * thruster_cmds[1]),
            ('port_servo', servo_angles_rad[0]),
            ('starboard_servo', servo_angles_rad[1])
        ]

        ## Only depth
        # thruster_mapping = [
        #     ('heave_bow', thruster_cmds[0]),
        #     ('surge_port', 0),
        #     ('surge_starboard', 0),
        #     ('heave_stern', thruster_cmds[3]),
        #     ('port_servo', 0),
        #     ('starboard_servo', 0)
        # ]

        # Depth and heading
        # thruster_mapping = [
        #     ('heave_bow', thruster_cmds[0]),
        #     ('surge_port', thruster_cmds[1]),
        #     ('surge_starboard', thruster_cmds[2]),
        #     ('heave_stern', thruster_cmds[3]),
        #     ('port_servo', 0),
        #     ('starboard_servo', 0)
        # ]

        # Publish commands
        for name, value in thruster_mapping:
            msg = Float64()
            msg.data = float(value)
            self.thruster_pubs[name].publish(msg)

    # def calculate_reward(self, prev_state, current_state):
    #     # Main objective - track velocity setpoint
    #     reward_velocity = -np.linalg.norm(self.velocity_error)
        
    #     # Smoothness reward - penalize jerky thruster changes
    #     thruster_diff = self.thruster_action - self.thruster_command_action_prev[0]
    #     reward_smoothness = -0.2 * np.sum(np.abs(thruster_diff))
        
    #     # Stability reward - penalize excessive angular motion
    #     reward_stability = -0.3 * np.linalg.norm(self.angular_vel)
        
    #     # Energy efficiency - penalize high thruster usage
    #     reward_energy = -0.1 * np.sum(np.square(self.thruster_action))
        
    #     # Combined reward
    #     reward = reward_velocity + reward_smoothness + reward_stability + reward_energy
        
    #     # For debugging
    #     self.get_logger().debug(f"Rewards: vel={reward_velocity:.2f}, smooth={reward_smoothness:.2f}, " 
    #                         f"stab={reward_stability:.2f}, energy={reward_energy:.2f}, total={reward:.2f}")
        
    #     return reward
    
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
        # Compute performance error (quadratic penalty)
        # weighted_errors =  state_error_weights * error
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
        """Main control loop using separate state inputs for actor and critic"""

        # # vt: Linear velocities from DVL
        # linear_vel = self.v_state 
        # # print(linear_vel)
        # # ωt: Angular velocities from IMU
        # angular_vel = self.omega_ref_state  
        # # print(angular_vel)
        # # v̇t: Linear accelerations from IMU
        # linear_accel = self.linear_accel  # Ensure this is populated from IMU data
        # # print(linear_accel)
        # # ω̇t: Angular accelerations (derived from IMU angular velocities)
        # angular_accel = self.angular_accel  
        # # print(angular_accel)
        # # ut-1: Previous commands
        # prev_commands = self.prev_action if hasattr(self, 'prev_action') and self.prev_action is not None else np.zeros(self.action_dim)
        # # print(prev_commands)
        # # et: Velocity error
        # velocity_error = self.v_err  # Difference between current and desired velocities
        # # print(velocity_error)
        # # breakpoint()
        # # Create the complete state according to the paper definition
        # complete_state = np.concatenate([
        #     linear_vel,      # vt
        #     angular_vel,     # ωt
        #     linear_accel,    # v̇t
        #     angular_accel,   # ω̇t
        #     prev_commands,   # ut-1
        #     velocity_error   # et
        # ])
        
        # actor_state = complete_state  # Use the complete state for actor
        
        # critic_state = np.concatenate([
        #     complete_state,     # Paper-defined state
        #     self.joint_angles,  # Current servo angles
        #     np.array([          # Current thrust values
        #         self.thrust_heave_bow,
        #         self.thrust_surge_port,
        #         self.thrust_surge_starboard,
        #         self.thrust_heave_stern
        #     ])
        # ])
        
        depth = self.position_state[2:3]
        surge = self.v_state[0:1]     
        sway = self.v_state[1:2]      
        heave = self.v_state[2:3]        
        roll = self.orientation_state[0:1]         
        pitch = self.orientation_state[1:2]         
        yaw = self.orientation_state[2:3]           

        depth_error = self.position_err[2:3]        
        surge_error = self.v_err[0:1]   
        sway_error = self.v_err[1:2]       
        heave_error = self.v_err[2:3]      
        roll_error = self.orientation_err[0:1]     
        pitch_error = self.orientation_err[1:2]     
        yaw_error = self.orientation_err[2:3]    

        roll_rate = self.omega_ref_state[0:1]
        pitch_rate = self.omega_ref_state[1:2]
        yaw_rate = self.omega_ref_state[2:3]
        roll_rate_error = self.omega_ref_err[0:1]
        pitch_rate_error = self.omega_ref_err[1:2] 
        yaw_rate_error = self.omega_ref_err[2:3]

        actor_state = np.concatenate([
            depth_error,               
            surge_error,    
            sway_error,       
            # heave_error,     
            roll_error, 
            pitch_error, 
            yaw_error,  
            depth,                      
            surge, 
            sway, 
            heave,  
            roll, 
            pitch, 
            yaw / np.pi,           
            # roll_rate,
            # pitch_rate,
            # yaw_rate,
        ])

        # Create critic state by concatenating the components you want
        critic_state = np.concatenate([
            depth_error,                 
            surge_error,
            sway_error, 
            #heave_error,  
            roll_error, 
            pitch_error, 
            yaw_error, 
            # roll_rate_error,
            # pitch_rate_error,
            # yaw_rate_error,
            depth,                      
            surge, 
            sway, 
            heave,  
            roll,
            pitch,
            yaw / np.pi,           
            # roll_rate,
            # pitch_rate,
            # yaw_rate,
            self.joint_angles,           
            np.array([                  
                self.thrust_heave_bow,
                self.thrust_surge_port,
                self.thrust_surge_starboard,
                self.thrust_heave_stern
            ])
        ])

        # Get action from agent based on actor state only
        action = self.agent.get_action(actor_state, add_noise=self.training_mode)
        action = np.reshape(action, -1) 
        # Publish thruster and servo commands
        self.publish_action(action)
        
        # Calculate current time if episode_start_time is not set
        if not hasattr(self, 'episode_start_time') or self.episode_start_time is None:
            self.episode_start_time = time.time()

        done = False

        # If in training mode, generate reward and train
        if self.training_mode and self.prev_actor_state is not None and self.prev_critic_state is not None and self.prev_action is not None:
            reward = self.calculate_reward(self.prev_critic_state, critic_state)
            
            # Ensure reward is a scalar value
            if isinstance(reward, np.ndarray):
                reward = float(reward.item())
            # Add reward to episode total
            self.episode_reward += reward
            
            # Check if episode is done
            done = self.is_done(critic_state)
            
            # Store in replay buffer with separate states
            self.agent.remember(
                self.prev_actor_state, 
                self.prev_critic_state, 
                self.prev_action, 
                reward, 
                actor_state, 
                critic_state, 
                done
            )
            
            # Train agent
            critic_loss, actor_loss = self.agent.learn()
            if critic_loss is not None:
                self.get_logger().debug(f"Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")
                
        elif not self.training_mode:
            # When in inference mode, still check if episode is done
            done = self.is_done(critic_state)
            
        # Update episode step counter
        self.episode_step += 1

        # Check for episode end
        if done or self.episode_step >= self.max_steps:
            self.get_logger().info(f"Episode {self.episode_count} completed: Steps={self.episode_step}, Reward={self.episode_reward:.2f}")
            self.episode_step = 0  # Reset step counter (you had self.episode_step = self.episode_step)
            self.episode_count += 1
            
            #Learning rate update feature
            # Track recent rewards for learning rate adjustment
            if not hasattr(self, 'recent_rewards'):
                self.recent_rewards = []
            
            self.recent_rewards.append(self.episode_reward)
            
            # Keep only last 10 rewards for moving average
            if len(self.recent_rewards) > 10:
                self.recent_rewards.pop(0)
            
            # Calculate average reward
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
            
            # Update learning rates based on performance
            lr_updated = self.agent.update_learning_rates(self.episode_count, avg_reward)
            
            if lr_updated:
                self.get_logger().info(f"Episode {self.episode_count}: Learning rate decreased due to performance plateau")
        
            # Save model periodically
            if self.episode_count % 1 == 0:
                #     model_path = f"ddpg_auv_model_ep{self.episode_count}.pt"
                #     self.agent.save_weights(model_path)
                #     self.get_logger().info(f"Model saved to {model_path}")
                
                    # Create a session ID only once when the program starts
                    if not hasattr(self, 'session_id'):
                        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Create the session directory
                        self.checkpoint_dir = os.path.join("checkpoints", f"session_{self.session_id}")
                        os.makedirs(self.checkpoint_dir, exist_ok=True)
                        self.get_logger().info(f"Created checkpoint directory: {self.checkpoint_dir}")
                    
                    # Save the model in the session directory with incrementing episode numbers
                    model_path = os.path.join(self.checkpoint_dir, f"ddpg_auv_model_ep{self.episode_count}.pt")
                    self.agent.save_weights(model_path)
                    self.get_logger().info(f"Model saved to {model_path}")
                
            if self.episode_count >= self.max_episodes:
                self.get_logger().info(f"Reached maximum number of episodes ({self.max_episodes}). Training complete.")
                final_model_path = "ddpg_auv_model_final.pt"
                self.agent.save_weights(final_model_path)
                self.get_logger().info(f"Final model saved to {final_model_path}")
                # Load the saved model back for inference
                self.agent.load_weights(final_model_path)
                self.training_mode = False  # Stop training mode
                self.get_logger().info("Switching to inference mode - controller will continue sending actions")

            # Reset episode reward AFTER logging it
            self.episode_reward = 0
            # Reset episode start time for the next episode
            self.episode_start_time = time.time()
        
        # Store state and action for next training step
        self.prev_actor_state = actor_state.copy()
        self.prev_critic_state = critic_state.copy()
        self.prev_action = action

    def is_done(self, state):
        """Check if episode should terminate based on time/step limits only"""
        
        # Time-based termination
        max_time = self.config['training']['max_t']  # Max allowed episode duration (seconds)
        elapsed_time = time.time() - self.episode_start_time  # Calculate elapsed time
        time_limit_exceeded = elapsed_time >= max_time

        # Step-based termination 
        max_steps = self.config['training']['max_t']
        step_limit_exceeded = self.episode_step >= max_steps

        # Yaw error termination - terminate if yaw error exceeds 10 degrees
        yaw_error = self.orientation_err[2:3]
        yaw_error_exceeded = abs(float(yaw_error)) > (10 * np.pi / 180)  # Convert 10 degrees to radians
        
        # Episode terminates if time limit, step limit, or yaw error is exceeded
        done = time_limit_exceeded #or yaw_error_exceeded #or step_limit_exceeded


        # Log the reason for termination
        # if done:
        #     if time_limit_exceeded:
        #         # self.get_logger().info(f"Episode terminated: Time limit exceeded ({elapsed_time:.2f}/{max_time:.2f} seconds)")
        #     if step_limit_exceeded:
        #         # self.get_logger().info(f"Episode terminated: Step count limit exceeded ({self.episode_step}/{max_steps})")
                
        return done
def main(args=None):
    rclpy.init(args=args)
    ddpg_ros = DDPG_ROS(DDPG)
    
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