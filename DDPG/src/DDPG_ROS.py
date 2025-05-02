import os,time, datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32
from geometry_msgs.msg import TwistStamped
from mvp_msgs.msg import ControlProcess
from sensor_msgs.msg import Imu
from DDPG import DDPG 
import torch
import numpy as np
from config_utils import load_config 


class DDPG_ROS(Node):
    """DDPG Agent for AUV control integrated with ROS2"""
    def __init__(self, config):
        super().__init__('ddpg_auv_control')
        
        self.config = config

        # Define separate dimensions for actor and critic
        self.actor_state_dim = 13   # Example: position_err, v_err, orientation_err
        self.critic_state_dim = 13  # error states + commands
        self.action_dim = 2  # 4 thrusters + 2 servo angles
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
        

        # publishers for training metrics
        self.q_value_pub = self.create_publisher(Float32, 'ddpg/q_value', 10)
        self.reward_pub = self.create_publisher(Float32, 'ddpg/reward', 10)
        self.actor_loss_pub = self.create_publisher(Float32, 'ddpg/actor_loss', 10)
        self.critic_loss_pub = self.create_publisher(Float32, 'ddpg/critic_loss', 10)
        self.episode_pub = self.create_publisher(Float32, 'ddpg/episode', 10)

        
        self.create_subscription(ControlProcess,  
                                '/race2_auv/controller/process/value',
                                self.state_callback,
                                1)

        self.create_subscription(ControlProcess, 
                                '/race2_auv/controller/process/setpoint', 
                                self.setpoint_callback,
                                1)
        
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
        
        # Training parameters
        self.declare_parameter('checkpoints_save_period', 100)
        self.checkpoints_save_period = self.get_parameter('checkpoints_save_period').value
        
        self.declare_parameter('training_mode', True)
        self.declare_parameter('max_steps', 100)
        self.max_steps = self.get_parameter('max_steps').value
        
        self.declare_parameter('reward_history_window_size', self.max_steps)
        self.reward_history_window_size = self.get_parameter('reward_history_window_size').value
        # self.declare_parameter('model_path', '/home/soslab/race2_ws/src/race2_q_learning_control/DDPG/src/checkpoints/session_20250421_130624/ddpg_auv_model_ep770.pt')
        # self.declare_parameter('model_path', '/home/farhang/race2_ws/src/race2_q_learning_control/DDPG/src/checkpoints/session_20250422_191702/ddpg_auv_model_ep1300.pt')
        # self.declare_parameter('model_path', '/home/farhang/race2_ws/src/race2_q_learning_control/DDPG/src/checkpoints/session_20250424_172104/ddpg_auv_model_ep20.pt')

        self.declare_parameter('model_path', '')
        self.declare_parameter('max_episodes', 700)  # Default 1000 episodes
        self.max_episodes = self.get_parameter('max_episodes').value

        self.training_mode = self.get_parameter('training_mode').value
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
        
        # Episode counter
        self.current_episode = 0
        self.step_counter = 0

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
        self.timer = self.create_timer(0.1, self.control_loop)  # 100 Hz control loop
        
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
    
    def publish_metrics(self, reward, q_value=None, actor_loss=None, critic_loss=None):
        # Publish current episode
        episode_msg = Float32()
        episode_msg.data = float(self.current_episode)
        self.episode_pub.publish(episode_msg)
        
        # Publish step within episode
        step_msg = Float32()
        step_msg.data = float(self.step_counter)
        self.create_publisher(Float32, 'ddpg/step', 10).publish(step_msg)
        
        # Publish reward
        reward_msg = Float32()
        reward_msg.data = float(reward)
        self.reward_pub.publish(reward_msg)
        
        # Publish Q-value if available
        if q_value is not None:
            q_msg = Float32()
            q_msg.data = float(q_value)
            self.q_value_pub.publish(q_msg)
        
        # Publish losses if available
        if actor_loss is not None:
            actor_msg = Float32()
            actor_msg.data = float(actor_loss)
            self.actor_loss_pub.publish(actor_msg)
        
        if critic_loss is not None:
            critic_msg = Float32()
            critic_msg.data = float(critic_loss)
            self.critic_loss_pub.publish(critic_msg)

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
            ('heave_bow',   thruster_cmds[0]),
            ('heave_stern',  thruster_cmds[1])
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
            self.thrust_heave_bow,
            self.thrust_heave_stern
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
    
    def check_if_stuck(model, name="Network"): print(f"{name} weight change: {sum(p.grad.abs().mean().item() if p.grad is not None else 0 for p in model.parameters()):.8f}")
    
    def control_loop(self):
        """Main control loop using separate state inputs for actor and critic"""
        
        # Initialize attributes if needed
        if not hasattr(self, 'new_state_available'):
            self.new_state_available = False
        if not hasattr(self, 'new_error_available'):
            self.new_error_available = False 
        if not hasattr(self, 'last_action_timestamp'):
            self.last_action_timestamp = 0
        if not hasattr(self, 'episode_start_time') or self.episode_start_time is None:
            self.episode_start_time = time.time()
        if not hasattr(self, 'recent_rewards'):
            self.recent_rewards = []
        if not hasattr(self, 'session_id') and self.training_mode:
            self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_dir = os.path.join("checkpoints", f"session_{self.session_id}")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.get_logger().info(f"Created checkpoint directory: {self.checkpoint_dir}")

        # done = False
        # Extract state variables
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
        self.actor_state = np.concatenate([
            depth_error, 
            surge_error, 
            sway_error, 
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

        # Use same state for critic
        self.critic_state = self.actor_state
        
        # Get action from agent
        action = self.agent.get_action(self.actor_state, add_noise=self.training_mode)
        action = np.reshape(action, -1) 
        # print(action)
        # action = [self.thrust_heave_bow, self.thrust_heave_stern]
        # Publish action
        self.publish_action(action)
        self.publish_time = time.time()
        
        # Check if episode is done
        done = self.is_done(self.critic_state)

        # Training mode logic
        time.sleep(0.5)

        # Get updated state after action
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

        # Create next state
        next_actor_state = np.concatenate([
            updated_depth_error, 
            updated_surge_error, 
            updated_sway_error,
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
        next_critic_state = next_actor_state
        state_error_array = np.concatenate(
            [
            updated_depth_error, 
            updated_surge_error, 
            updated_sway_error,
            updated_roll_error, 
            updated_pitch_error, 
            updated_yaw_error,
            ])
        # Calculate reward
        reward = self.calculate_reward(state_error_array)
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
        
        # Add to episode reward
        self.episode_reward += reward

        # Store experience in replay buffer
        self.agent.remember(
            self.actor_state, 
            self.critic_state, 
            action, 
            100 * reward, 
            next_actor_state, 
            next_critic_state, 
            done
        )
        
        # Train agent
        result = self.agent.learn()
        if all(v is not None for v in result):
            critic_loss, actor_loss, reward_value, current_q_value , target_q_value = result
            self.publish_metrics(reward_value, current_q_value, actor_loss, critic_loss)
            self.step_counter += 1        
            self.get_logger().info(f"Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}, Reward: {reward_value:.4f}  ,Current Q_value: {current_q_value:.4f} , Target Q_value: {target_q_value:.4f}")
        else:
            self.get_logger().debug("Learn returned None — skipping training this step.")
        
        # Update episode step counter
        self.episode_step += 1

        # Check for episode end
        if done or self.episode_step >= self.max_steps:
            self.get_logger().info(f"Episode {self.episode_count} completed: Steps={self.episode_step}, Reward={self.episode_reward:.2f}")
            self.episode_step = 0
            self.episode_count += 1
            
            # Learning rate update feature
            self.recent_rewards.append(self.episode_reward)
            
            #keep rewards for moving average
            if len(self.recent_rewards) > self.reward_history_window_size:
                self.recent_rewards.pop(0)
            
            # Calculate average reward
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
            
            # Update learning rates based on performance
            if self.training_mode:
                lr_updated = self.agent.update_learning_rates(self.episode_count, avg_reward)
                if lr_updated:
                    self.get_logger().info(f"Episode {self.episode_count}: Learning rate decreased due to performance plateau")
            
                # Save model periodically (only in training mode)
                if self.episode_count % self.checkpoints_save_period == 0:
                    model_path = os.path.join(self.checkpoint_dir, f"ddpg_auv_model_ep{self.episode_count}.pt")
                    self.agent.save_weights(model_path)
                    self.get_logger().info(f"Model saved to {model_path}")
                
                # Check if we've reached max episodes
                if self.episode_count >= self.max_episodes:
                    self.get_logger().info(f"Reached maximum number of episodes ({self.max_episodes}). Training complete.")
                    final_model_path = "ddpg_auv_model_final.pt"
                    self.agent.save_weights(final_model_path)
                    self.get_logger().info(f"Final model saved to {final_model_path}")
                    # Load the saved model back for inference
                    self.agent.load_weights(final_model_path)
                    self.training_mode = False
                    self.get_logger().info("Switching to inference mode - controller will continue sending actions")

            # Reset episode reward and start time
            self.episode_reward = 0
            self.episode_start_time = time.time()

    def is_done(self, state):
        """Check if episode should terminate based on time/step limits only"""
        
        # Time-based termination
        max_time = self.config['training']['max_t']  # Max allowed episode duration
        elapsed_time = time.time() - self.episode_start_time  # Calculate elapsed time
        time_limit_exceeded = elapsed_time >= max_time

        # Step-based termination 
        max_steps = self.config['training']['max_t']
        step_limit_exceeded = self.episode_step >= max_steps

        # Yaw error termination - terminate if yaw error exceeds 10 degrees
        yaw_error = self.orientation_err[2:3]
        # yaw_error_exceeded = abs(float(yaw_error)) > (10 * np.pi / 180)  # Convert 10 degrees to radians
        
        # Episode terminates if time limit, step limit, or yaw error is exceeded
        done = step_limit_exceeded # or time_limit_exceeded #or yaw_error_exceeded #or 

        self.current_episode += 1
        self.step_counter = 0

        # Log the reason for termination
        # if done:
        #     if time_limit_exceeded:
        #         # self.get_logger().info(f"Episode terminated: Time limit exceeded ({elapsed_time:.2f}/{max_time:.2f} seconds)")
        #     if step_limit_exceeded:
        #         # self.get_logger().info(f"Episode terminated: Step count limit exceeded ({self.episode_step}/{max_steps})")
                
        return done

def main(args=None):
    # Initialize ROS
    rclpy.init(args=args)
    
    try:

        config = load_config()
        ddpg_ros = DDPG_ROS(config)
        
        try:
            rclpy.spin(ddpg_ros)
        except KeyboardInterrupt:
            pass
        finally:
            # Save final model
            ddpg_ros.agent.save_weights("ddpg_auv_model_final.pt")
            ddpg_ros.get_logger().info("Final model saved to ddpg_auv_model_final.pt")
            ddpg_ros.destroy_node()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure ROS is properly shut down even if exceptions occur
        rclpy.shutdown()

if __name__ == "__main__":
    main()