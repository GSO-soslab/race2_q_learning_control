import os
import random
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rospy
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from mvp_msgs.msg import ControlProcess
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Int32MultiArray
from std_srvs.srv import SetBool, SetBoolResponse
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set the path to the config file in the parent config directory
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

# Load the configuration file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Set random seeds for reproducibility
random_seed = config['others']['random_seed']
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

 
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
        super(QNetwork, self).__init__()
        
        layers = []
        input_size = int(state_size)  

        # Create the hidden layers
        for hidden_layer in hidden_layers:
            hidden_layer = int(hidden_layer)  
            print("input_size:", input_size, "hidden_layer:", hidden_layer)  
            layers.append(nn.Linear(input_size, hidden_layer))
            layers.append(nn.Tanh())
            input_size = hidden_layer

        # Output layer for action_size actions
        layers.append(nn.Linear(input_size, int(action_size))) 

        # Define the network
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.memory.append(experience)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, state_size, action_size, config):
        self.state_size = config['environment']['error_size'] + config['environment']['state_size'] + config['environment']['servo_joints_size']
        print("State size:", self.state_size)
        self.action_size = action_size
        self.gamma = config['agent']['gamma']
        self.batch_size = config['agent']['batch_size']
        self.tau = config['agent']['tau']

        # Initialize the Q-network and the target network
        hidden_layers = tuple(config['qnetwork']['hidden_layers'])
        print("Hidden layers:", hidden_layers) 
        # self.qnetwork = QNetwork(state_size, action_size, hidden_layers) #policy network
        # self.target_network = QNetwork(state_size, action_size, hidden_layers)

        self.qnetwork = QNetwork(state_size, action_size, hidden_layers).to(device)  # Move Q-network to GPU
        self.target_network = QNetwork(state_size, action_size, hidden_layers).to(device)  # Move target network to GPU

        # # Apply weights initialization to both networks
        # self.qnetwork.apply(self.weights_init)
        # self.target_network.apply(self.weights_init)

        # Initialize the optimizer for the Q-network
        lr = config['agent']['learning_rate']
        self.optimizer = optim.AdamW(self.qnetwork.parameters(), lr=lr)

        # Replay buffer
        buffer_size = config['agent']['buffer_size']
        self.memory = ReplayBuffer(buffer_size, self.batch_size)

        # Initially set the target network to have the same weights as the Q-network
        self.update_target_network()

    def weights_init(self, m):
        """Initialize the weights of the network."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def update_target_network(self):
        # Soft update of the target network
        for target_param, local_param in zip(self.target_network.parameters(), self.qnetwork.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def act(self, state, epsilon=0.1):
        if random.random() > epsilon:
            # state = torch.FloatTensor(state).unsqueeze(0)
            state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Move state tensor to GPU
            with torch.no_grad():
                action_values = self.qnetwork(state)
            action_index = torch.argmax(action_values).item()
            return action_index
        else:
            return random.choice(range(self.action_size))

    def step(self, state, action, reward, next_state, done):
        # Store the experience in the replay buffer
        self.memory.add((state, action, reward, next_state, done))

        # Initialize loss to None
        loss = None

        # If there are enough samples in memory, learn from them
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            loss = self.learn(experiences)  # Capture the returned loss

        return loss  # Return the loss value (or None if no learning occurred)

    def learn(self, experiences):
        # Unpack experiences
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to tensors
        batch_size = len(states)
        # states = torch.FloatTensor(np.array(states))
        # actions = torch.LongTensor(np.array(actions)).view(-1, 1)
        # rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1)
        # next_states = torch.FloatTensor(np.array(next_states))
        # dones = torch.FloatTensor(np.array(dones)).view(-1, 1)

        states = torch.FloatTensor(np.array(states)).to(device) 
        actions = torch.LongTensor(np.array(actions)).view(-1, 1).to(device) 
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1).to(device) 
        next_states = torch.FloatTensor(np.array(next_states)).to(device)  
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1).to(device) 

        # Compute current Q-values
        q_values = self.qnetwork(states)

        # Double DQN logic
        with torch.no_grad():
            next_q_values = self.qnetwork(next_states)
            max_actions = next_q_values.argmax(1).unsqueeze(1)
            next_q_values_target = self.target_network(next_states).gather(1, max_actions)

        # Compute target Q-values
        q_targets = rewards + (self.gamma * next_q_values_target * (1 - dones))

        # Gather the Q-values for the actions taken
        q_values_for_actions = q_values.gather(1, actions)

        # Compute loss using Huber loss
        loss = nn.SmoothL1Loss()(q_values_for_actions, q_targets)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.update_target_network()

        return loss.item()


class GridWorldEnv:
    def __init__(self, config):
        self.config = config
        #temporarily dropped passing joint angle changes to the network
        self.state_size = config['environment']['error_size'] + config['environment']['state_size'] + config['environment']['servo_joints_size']
        self.action_size = config['environment']['action_size']
        self.position_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)
        self.position_state = np.zeros(3)
        self.v_state = np.zeros(3)
        self.orientation_state = np.zeros(3)
        self.omega_ref_state = np.zeros(3)
        self._episode_ended = False
        self.max_episode_duration = config['environment']['max_episode_duration']
        self.thruster_history_length = config['environment']['thruster_history_length']
        self.servo_history_length = config['environment']['servo_history_length']
        self.sampling_time = config['environment']['sampling_time']
        self.action_mapping = {int(k): v for k, v in config['environment']['action_mapping'].items()}
        
        # Initialize scaler for all inputs
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        # Fit scaler with dummy data initially
        self.scaler.fit(np.zeros((1, self.state_size)))

        # Retrieve servo and thruster size from config
        self.servo_joints_size = config['environment']['servo_joints_size']
        thruster_size = config['environment']['thruster_size']

        # Initialize joint positions for servos
        self.joint_angles = np.zeros(self.servo_joints_size)
        self.joint_positions_history = np.zeros((self.servo_history_length, self.servo_joints_size))
        self.u_prev = np.zeros((self.thruster_history_length, thruster_size))


        # ROS node initialization
        rospy.init_node('underwater_vehicle_env', anonymous=True)

        self.thruster_action_pub = rospy.Publisher('/thruster_action', Int32MultiArray, queue_size=10)

        # Initialize the policy control state
        self.use_policy = True
        self.stop_training = False

        # Service for enabling/disabling the policy
        rospy.Service('/toggle_policy', SetBool, self.toggle_policy_service)

        # Service for saving the policy manually
        rospy.Service('/save_policy', SetBool, self.save_policy_service)

        rospy.Subscriber('/race2/controller/process/error', ControlProcess, self.update_current_error)
        rospy.Subscriber('/race2/controller/process/state', ControlProcess, self.update_current_state)
        rospy.Subscriber('/race2/control/thruster/heave_bow', Float64, self.update_thrust_heave_bow)
        rospy.Subscriber('/race2/control/thruster/surge_port', Float64, self.update_thrust_surge_port)
        rospy.Subscriber('/race2/control/thruster/sway_stern', Float64, self.update_thrust_sway_stern)
        rospy.Subscriber('/race2/control/thruster/surge_starboard', Float64, self.update_thrust_surge_starboard)
        rospy.Subscriber('/race2/control/servos/joint_states', JointState, self.update_joint_states)

        self.reset()

    # Service callback to save the policy manually
    def save_policy_service(self, request):
        """Service callback to save the policy manually."""
        self.stop_training = True  # Set training to stop
        return SetBoolResponse(success=True, message="Training stopped, policy will be saved.")

    def toggle_policy_service(self, request):
        """Service callback to enable/disable policy use."""
        self.use_policy = request.data

        if not self.use_policy:
            # If the policy is disabled, set stop_training flag to True to stop the training loop
            self.stop_training = True
            # Also send the thruster command [1, 1, 1, 1, 1, 1]
            thruster_command = Int32MultiArray(data=[1, 1, 1, 1, 1, 1])
            self.thruster_action_pub.publish(thruster_command)
            rospy.loginfo("Policy disabled. Thruster set to [1, 1, 1, 1, 1, 1] and training stopped.")

        return SetBoolResponse(success=True, message="Policy control and training updated")

    # def update_current_error(self, data):
    #     self.position_err = np.array([data.position.z ])
    #     self.orientation_err = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
    #     self.v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
    #     self.omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])

    def update_current_error(self, data):
        # Extract errors
        raw_position_err = np.array([data.position.x,data.position.y,data.position.z])
        raw_orientation_err = np.array([ data.orientation.x,data.orientation.y, data.orientation.z])
        raw_v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        raw_omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])

        # Combine errors into a single numpy array
        raw_state = np.concatenate([raw_position_err, 
                                    raw_orientation_err, 
                                    raw_v_err, 
                                    raw_omega_ref_err
                                    ])
        raw_state_np = raw_state.reshape(1, -1)  # Reshape for scaler

        # Dynamically fit and transform the state
        self.scaler.partial_fit(raw_state_np)
        scaled_state_np = self.scaler.transform(raw_state_np)

        # Convert back to PyTorch tensor
        scaled_state_tensor = torch.from_numpy(scaled_state_np).float()

        # Split the scaled tensor into components
        self.position_err = scaled_state_tensor[0, :3]  # Depth (first element)
        self.orientation_err = scaled_state_tensor[0, 3:6]  # Orientation angles (next three elements except roll)
        self.v_err = scaled_state_tensor[0, 6:9]  # Surge and sway (next two elements)
        self.omega_ref_err = scaled_state_tensor[0, 9:12] #dummy since doesnt have to be used

    def update_current_state(self, data):
        # Extract errors
        raw_position_state = np.array([data.position.x,data.position.y,data.position.z])
        raw_orientation_state = np.array([ data.orientation.x,data.orientation.y, data.orientation.z])
        raw_v_state = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        raw_omega_ref_state = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])

        # Combine errors into a single numpy array
        raw_state = np.concatenate([raw_position_state, 
                                    raw_orientation_state, 
                                    raw_v_state, 
                                    raw_omega_ref_state
                                    ])
        raw_state_np = raw_state.reshape(1, -1)

        # Dynamically fit and transform the state
        self.scaler.partial_fit(raw_state_np)
        scaled_state_np = self.scaler.transform(raw_state_np)

        # Convert back to PyTorch tensor
        scaled_state_tensor = torch.from_numpy(scaled_state_np).float()

        # Split the scaled tensor into components
        self.position_state = scaled_state_tensor[0, :3]  # Depth (first element)
        self.orientation_state = scaled_state_tensor[0, 3:6]  # Orientation angles (next three elements except roll)
        self.v_state = scaled_state_tensor[0, 6:9]  # Surge and sway (next two elements)
        self.omega_ref_state = scaled_state_tensor[0, 9:12] #dummy since doesnt have to be used

    def update_joint_states(self, data):
        self.joint_angles = np.array(data.position[:2])

    def update_thrust_surge_port(self, data):
        self.thrust_surge_port = data.data
    def update_thrust_surge_starboard(self, data):
        self.thrust_surge_starboard = data.data

    def update_thrust_heave_bow(self, data):
        self.thrust_heave_bow = data.data

    def update_thrust_sway_stern(self, data):
        self.thrust_sway_stern = data.data

    def step(self, action_index):
        if not self.use_policy:
            rospy.loginfo("Policy is disabled, using static thruster command.")
            state = np.concatenate(
                [self.position_err[2:3], # Depth
                 self.v_err[:2], # Surge and sway
                 self.orientation_err[1:3], # roll, pitch, yaw
                #  self.omega_ref_err
                 self.position_state[ 2:3],
                 self.v_state[:2],
                 self.orientation_state[1:3],
                    # self.omgea_ref_state
                 self.joint_angles
                ])
            
            return state, 0, True, {}

        # Map the action index to actual action values
        action = self.action_mapping[action_index]
        action1, action2 = action  # Unpack the action values

        # Create the array: [action1, 1, action2, 1, 1, 1]
        thruster_command = Int32MultiArray(data=[action1, 1, action2, 1, 1, 1])

        # Publish the action array
        self.thruster_action_pub.publish(thruster_command)

        rospy.sleep(self.sampling_time) 
        current_time = rospy.get_time()
        elapsed_time_total = current_time - self.start_time

        # Determine if the episode has ended
        done = elapsed_time_total > self.max_episode_duration
        # done = True  # Assume episode ends after one action
        if done:
            self._episode_ended = True

        # Calculate reward
        reward = self.calculate_reward()

        # Prepare the next state
        next_state = np.concatenate(
                [self.position_err[2:3], # Depth
                 self.v_err[:2], # Surge and sway
                 self.orientation_err[1:3], # roll, pitch, yaw
                #  self.omega_ref_err
                 self.position_state[2:3],
                 self.v_state[:2],
                 self.orientation_state[1:3],
                    # self.omgea_ref_state
                 self.joint_angles
                ])

        # print("Next State:", next_state)
        return next_state, reward, done, {}

    def reset(self):
        self._episode_ended = False
        self.start_time = rospy.get_time()

        # Initialize joint angles randomly or to a specific value
        # initial_joint_angles = np.random.uniform(low=-np.pi, high=np.pi, size=2)
        # self.joint_angles = initial_joint_angles
        # self.joint_angles = np.zeros(self.servo_joints_size)  
              
        # Initialize joint_positions_history with the initial joint angles
        self.joint_positions_history = np.full(self.joint_positions_history.shape, 0)

        # Return the initial state
        return np.concatenate(
                [self.position_err[2:3], # Depth
                 self.v_err[ :2], # Surge and sway
                 self.orientation_err[1:3], # roll, pitch, yaw
                #  self.omega_ref_err
                 self.position_state[2:3],
                 self.v_state[ :2],
                 self.orientation_state[1:3],
                    # self.omgea_ref_state
                 self.joint_angles
                ])

    def calculate_reward(self):
        # Reward function parameters
        w = self.config['reward_function']
        w1, w2, w3, w4, w5, w6 = w['w1'], w['w2'], w['w3'], w['w4'], w['w5'], w['w6']
        state_error_weights = np.array(w['state_error_weights'])

        # Compute the error vector
        error = np.concatenate(
            [self.position_err[2:3], # Depth
                 self.v_err[:2], # Surge and sway
                 self.orientation_err[1:3], # pitch, yaw
            ]).astype(np.float32)
        # Compute performance error (quadratic penalty)
        weighted_errors = state_error_weights * error

        performance_error = np.sum(weighted_errors ** 2)

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
            self.thrust_sway_stern
        ])
        thruster_usage_penalty = np.sum(np.abs(u_t))

        # # Thruster smoothness penalty
        # if len(self.u_prev) > 0:
        #     thruster_smoothness_penalty = np.linalg.norm(u_t - np.average(self.u_prev, axis=0))
        #     self.u_prev = np.vstack((self.u_prev[1:], u_t))

        # Thruster smoothness penalty
        if len(self.u_prev) > 1:  # Check if there’s enough history to calculate smoothness
            thruster_smoothness_penalty = np.linalg.norm(u_t - np.average(self.u_prev, axis=0))
        else:
            thruster_smoothness_penalty = np.linalg.norm(u_t)  # Initial penalty based on u_t

        # Update self.u_prev to store the history
        self.u_prev = np.vstack((self.u_prev[1:], u_t)) if len(self.u_prev) > 1 else np.vstack((self.u_prev, u_t))

        # Thruster delta reward
        thruster_delta_reward = np.linalg.norm(u_t - self.u_prev[-2])

        # Servo angle penalty
        servo_angle_penalty = np.linalg.norm(self.joint_angles)
        # print("Smoothness penalty:", thruster_smoothness_penalty)
        # Total reward
        reward =   - (
            w1 * performance_error +
            w2 * servo_smoothness_penalty +
            w3 * thruster_usage_penalty +
            w4 * thruster_smoothness_penalty +
            w5 * servo_angle_penalty +
            w6 * thruster_delta_reward
        )
        
        # print(f"Performance Error Contribution: {-w1 * performance_error}")
        # print(f"Servo Smoothness Penalty Contribution: {-w2 * servo_smoothness_penalty}")
        # print(f"Thruster Usage Penalty Contribution: {-w3 * thruster_usage_penalty}")
        # print(f"Thruster Smoothness Penalty Contribution: {-w4 * thruster_smoothness_penalty}")
        # print(f"Servo Angle Penalty Contribution: {-w5 * servo_angle_penalty}")
        # print(f"Thruster Delta Reward Contribution: {-w6 * thruster_delta_reward}")

        return reward

# episode based learning 1st version
# def continuous_learning(env, agent, config):
#     max_episodes = config['training']['max_episodes']
#     max_t = config['training']['max_t']
#     target_avg_reward = config['training']['target_avg_reward']
#     epsilon = config['agent']['epsilon_initial']
#     epsilon_decay = config['agent']['epsilon_decay']
#     epsilon_min = config['agent']['epsilon_min']

#     episode_count = 0
#     rate = rospy.Rate(50)  # Set a rate (e.g., 10 Hz)

#     # Initialize plotting
#     plt.ion()
#     fig, ax = plt.subplots(3, 1, figsize=(10, 12))

#     reward_history = []
#     q_value_history = []
#     loss_history = []
#     episodes = []

#     # Set up the plots
#     ax[0].set_title('Total Reward per Episode')
#     ax[0].set_xlabel('Episode')
#     ax[0].set_ylabel('Total Reward')
#     reward_line, = ax[0].plot([], [], label='Reward')
#     ax[0].legend()

#     ax[1].set_title('Max Q-value per Episode')
#     ax[1].set_xlabel('Episode')
#     ax[1].set_ylabel('Max Q-value')
#     q_value_line, = ax[1].plot([], [], label='Max Q-value', color='orange')
#     ax[1].legend()

#     ax[2].set_title('Average Loss per Episode')
#     ax[2].set_xlabel('Episode')
#     ax[2].set_ylabel('Average Loss')
#     loss_line, = ax[2].plot([], [], label='Loss', color='green')
#     ax[2].legend()

#     # Training Loop
#     while episode_count < max_episodes and not rospy.is_shutdown():
#         if env.stop_training:
#             # rospy.loginfo("Training has been stopped.")
#             rospy.loginfo("Service called: Saving policy and stopping training.")
#             save_model(agent)  # Save the trained model
#             break  # Exit the training loop if training is stopped

#         episode_count += 1
#         state = env.reset()  # Reset environment to get initial state
#         score = 0
#         max_q_value = -float('inf')  # Initialize max Q-value for this episode
#         episode_loss = 0.0  # Initialize episode loss
#         loss_steps = 0  # Number of steps where learning occurred

#         for t in range(max_t):  # Limit each episode to max_t steps
#             action_index = agent.act(state, epsilon)
#             next_state, reward, done, _ = env.step(action_index)
#             loss = agent.step(state, action_index, reward, next_state, done)
#             state = next_state

#             score += reward

#             # Get Q-values for the current state
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             with torch.no_grad():
#                 q_values = agent.qnetwork(state_tensor)
#             current_max_q = q_values.max().item()
#             if current_max_q > max_q_value:
#                 max_q_value = current_max_q  # Update max Q-value for this episode

#             # Accumulate loss if learning occurred
#             if loss is not None:
#                 episode_loss += loss
#                 loss_steps += 1

#             if done:
#                 break

#             # Sleep to maintain the loop rate
#             rate.sleep()

#         # Decay epsilon after each episode
#         epsilon = max(epsilon_min, epsilon_decay * epsilon)

#         # Calculate average loss for the episode
#         average_loss = episode_loss / loss_steps if loss_steps > 0 else 0.0

#         # Append data for plotting
#         episodes.append(episode_count)
#         reward_history.append(score)
#         q_value_history.append(max_q_value)
#         loss_history.append(average_loss)

#         # Update the plots
#         update_plots(ax, episodes, reward_history, q_value_history, loss_history,
#                      reward_line, q_value_line, loss_line)

#         # Optionally, print the episode score and loss for monitoring
#         print(f"Episode {episode_count}: Score: {score:.2f}, Max Q-value: {max_q_value:.2f}, Average Loss: {average_loss:.4f}, Epsilon: {epsilon:.3f}")

#         # Check if average reward over the last N episodes meets the target
#         if target_avg_reward is not None and len(reward_history) >= 10:
#             avg_reward_recent = np.mean(reward_history[-10:])
#             if avg_reward_recent >= target_avg_reward:
#                 print(f"Stopping training as average reward over last 10 episodes is {avg_reward_recent:.2f} (>= {target_avg_reward})")
#                 break

#     # Save the model if the loop was exited due to max episodes
#     if not env.stop_training:
#         save_model(agent)
    
#     # After training, set epsilon to 0 to use the greedy policy
#     epsilon = 0.0

#     # Save the trained model
#     save_model(agent)

#     # Now, enter an infinite loop where the agent continues to act using the learned policy
#     print("Training complete. Continuing to run with the learned policy.")
#     while not rospy.is_shutdown():
#         state = env.reset()
#         done = False
#         total_reward = 0
#         while not done and not rospy.is_shutdown():
#             action_index = agent.act(state, epsilon=0.0)  # Use greedy policy
#             next_state, reward, done, _ = env.step(action_index)
#             state = next_state
#             total_reward += reward

#             # Sleep to maintain the loop rate
#             rate.sleep()

#         print(f"Episode completed with total reward: {total_reward}")

# Batch based learning 2nd version
def continuous_learning(env, agent, config):
    max_episodes = config['training']['max_episodes']
    max_t = config['training']['max_t']
    target_avg_reward = config['training']['target_avg_reward']
    epsilon = config['agent']['epsilon_initial']
    epsilon_decay = config['agent']['epsilon_decay']
    epsilon_min = config['agent']['epsilon_min']

    episode_count = 0
    rate = rospy.Rate(50) 

    # Initialize plotting
    plt.ion()
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))

    reward_history = []
    q_value_history = []
    loss_history = []
    episodes = []

    # Set up the plots
    ax[0].set_title('Total Reward per Episode')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Total Reward')
    reward_line, = ax[0].plot([], [], label='Reward')
    ax[0].legend()

    ax[1].set_title('Max Q-value per Episode')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Max Q-value')
    q_value_line, = ax[1].plot([], [], label='Max Q-value', color='orange')
    ax[1].legend()

    ax[2].set_title('Average Loss per Episode')
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('Average Loss')
    loss_line, = ax[2].plot([], [], label='Loss', color='green')
    ax[2].legend()

    # Training Loop
    while episode_count < max_episodes and not rospy.is_shutdown():
        if env.stop_training:
            rospy.loginfo("Service called: Saving policy and stopping training.")
            save_model(agent)  # Save the trained model
            break  # Exit the training loop if training is stopped

        episode_count += 1
        state = env.reset()  # Reset environment to get initial state

        # Track episode metrics (not directly used in training)
        score = 0
        total_reward = 0 
        step_count = 0    
        max_q_value = float('-inf')
        episode_loss = 0.0
        loss_steps = 0

        for episode_t in range(max_t):
            # 1. Select an action according to current policy (with exploration)
            action_index = agent.act(state, epsilon)

            # 2. Execute the action in the environment
            next_state, reward, done, _ = env.step(action_index)

            # 3. Store transition and (optionally) do a batch update
            #    agent.step(...) internally:
            #    - pushes (state, action, reward, next_state, done) into replay buffer
            #    - if memory is large enough and at the right interval, samples a batch
            #      and performs a training step
            loss = agent.step(state, action_index, reward, next_state, done)

            # 4. Move to the next state
            state = next_state

            # 5. Accumulate reward for logging (episode-level, not used for training)
            # score += reward

            total_reward += reward
            step_count += 1
            
            average_reward = total_reward / step_count if step_count > 0 else 0.0

            # 6. Log the max Q-value for debugging/analysis
            # state_tensor = torch.FloatTensor(state).unsqueeze(0)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.qnetwork(state_tensor)
            current_max_q = q_values.max().item()
            max_q_value = max(max_q_value, current_max_q)

            # 7. Track loss if a batch update occurred in agent.step(...)
            if loss is not None:
                episode_loss += loss
                loss_steps += 1

            if done:
                break

            rate.sleep()  # Sleep to maintain loop rate

        # 8. Decay epsilon after each episode
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # 9. Calculate average loss for the episode
        average_loss = episode_loss / loss_steps if loss_steps > 0 else 0.0

        # 10. Store metrics for plotting
        episodes.append(episode_count)
        # reward_history.append(score)
        reward_history.append(average_reward)
        q_value_history.append(max_q_value)
        loss_history.append(average_loss)

        # 11. Update the plots
        update_plots(ax, episodes, reward_history, q_value_history, loss_history,
                     reward_line, q_value_line, loss_line)

        # 12. Print status for monitoring
        print(f"Episode {episode_count}: Score: {score:.2f}, Max Q-value: {max_q_value:.2f}, "
              f"Average Loss: {average_loss:.4f}, Epsilon: {epsilon:.3f}")

        # 13. Check if average reward meets the target
        if target_avg_reward is not None and len(reward_history) >= 10:
            avg_reward_recent = np.mean(reward_history[-10:])
            if avg_reward_recent >= target_avg_reward:
                print(f"Stopping training as average reward over last 10 episodes is "
                      f"{avg_reward_recent:.2f} (>= {target_avg_reward})")
                break

    # Save the model after training loop is done
    if not env.stop_training:
        save_model(agent)

    # After training, set epsilon to 0 to use the greedy policy
    epsilon = 0.0
    save_model(agent)
    print("Training complete. Continuing to run with the learned policy.")

    # Run indefinitely using the learned (greedy) policy
    while not rospy.is_shutdown():
        state = env.reset()
        done = False
        total_reward = 0
        while not done and not rospy.is_shutdown():
            # Greedy action (no exploration)
            action_index = agent.act(state, epsilon=0.0)
            next_state, reward, done, _ = env.step(action_index)
            state = next_state
            total_reward += reward
            rate.sleep()

        print(f"Episode completed with total reward: {total_reward}")


def update_plots(ax, episodes, reward_history, q_value_history, loss_history,
                 reward_line, q_value_line, loss_line):
    # Update reward plot
    reward_line.set_xdata(episodes)
    reward_line.set_ydata(reward_history)
    ax[0].relim()
    ax[0].autoscale_view()

    # Update Q-value plot
    q_value_line.set_xdata(episodes)
    q_value_line.set_ydata(q_value_history)
    ax[1].relim()
    ax[1].autoscale_view()

    # Update loss plot
    loss_line.set_xdata(episodes)
    loss_line.set_ydata(loss_history)
    ax[2].relim()
    ax[2].autoscale_view()

    plt.draw()
    plt.pause(0.2)  # Pause to update the plots


def save_model(agent, filename_prefix='dqn_model'):
    """
    Save the trained model to a file with the current date and time appended to the filename.
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{filename_prefix}_{current_time}.pth"
    torch.save(agent.qnetwork.state_dict(), filename)
    print(f"Model saved to {filename}")


# def evaluate_agent(env, agent, config):
#     num_episodes = config['evaluation']['num_episodes']
#     total_scores = []
#     for episode in range(num_episodes):
#         state = env.reset()
#         score = 0
#         done = False
#         while not done and not rospy.is_shutdown():
#             action_index = agent.act(state, epsilon=0.0)  # Greedy policy
#             next_state, reward, done, _ = env.step(action_index)
#             state = next_state
#             score += reward
#         total_scores.append(score)
#         print(f"Evaluation Episode {episode + 1}: Score: {score:.2f}")
#     avg_score = np.mean(total_scores)
#     print(f"Average Evaluation Score over {num_episodes} episodes: {avg_score:.2f}")


if __name__ == '__main__':
    env = GridWorldEnv(config)
    state_size = env.state_size
    action_size = env.action_size
    agent = Agent(state_size, action_size, config)

    # Train the agent
    continuous_learning(env, agent, config)