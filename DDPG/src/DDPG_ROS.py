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
        

        # publishers for training metrics
        self.q_value_pub = self.create_publisher(Float32, 'ddpg/q_value', 10)
        self.reward_pub = self.create_publisher(Float32, 'ddpg/reward', 10)
        self.actor_loss_pub = self.create_publisher(Float32, 'ddpg/actor_loss', 10)
        self.critic_loss_pub = self.create_publisher(Float32, 'ddpg/critic_loss', 10)
        self.episode_pub = self.create_publisher(Float32, 'ddpg/episode', 10)

        
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
        self.timer = self.create_timer(1/10, self.control_loop)  # 100 Hz control loop
        
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

        if not hasattr(self, 'new_state_available'):
            self.new_state_available = False
        if not hasattr(self, 'last_action_timestamp'):
            self.last_action_timestamp = 0
        
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

        
        # Calculate current time if episode_start_time is not set
        if not hasattr(self, 'episode_start_time') or self.episode_start_time is None:
            self.episode_start_time = time.time()

        done = False

        # If in training mode, generate reward and train
        if (self.training_mode and self.prev_actor_state is not None and 
                self.prev_critic_state is not None and self.prev_action is not None and 
                self.new_state_available):
            print("New State received!!!")
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
            # critic_loss, actor_loss, reward_value, current_q_value = self.agent.learn()    #old version
            result = self.agent.learn()
            if all(v is not None for v in result):
                critic_loss, actor_loss, reward_value, current_q_value = result
                self.publish_metrics(reward_value, current_q_value, actor_loss, critic_loss)
                self.step_counter += 1        
                self.get_logger().debug(f"Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")
            else:
                print("Learn returned None — skipping training this step.")
            
            # if critic_loss is not None:

                
        elif not self.training_mode:
            # When in inference mode, still check if episode is done
            done = self.is_done(critic_state)
            
        # Publish thruster and servo commands
        self.publish_action(action)

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