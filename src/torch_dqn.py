import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rospy
from mvp_msgs.msg import ControlProcess
from std_srvs.srv import SetBool, SetBoolResponse
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Int32MultiArray
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime

action_mapping = {
    0: [1, 1],
    1: [-1, 1],
    2: [1, -1],
    3: [-1, -1]
}


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=(400,300,200,120,80,30)):
    # def __init__(self, state_size, action_size, hidden_layers=(120,80,30)):
        super(QNetwork, self).__init__()
        layers = []
        input_size = state_size

        # Create the hidden layers
        for hidden_layer in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_layer))
            layers.append(nn.ReLU())
            input_size = hidden_layer

        # Output layer for action_size actions
        layers.append(nn.Linear(input_size, action_size))  # action_size is 4

        # Define the network
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x  # Output Q-values for each action

        
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
    def __init__(self, state_size, action_size, buffer_size=20000, batch_size=32, gamma=0.95, lr=1e-4, tau=0.05):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau  # For soft update

        # Initialize the Q-network and the target network
        self.qnetwork = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        
        # Apply weights initialization to both networks
        self.qnetwork.apply(self.weights_init)
        self.target_network.apply(self.weights_init)

        # Initialize the optimizer for the Q-network
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # Initially set the target network to have the same weights as the Q-network
        self.update_target_network()

    def weights_init(self, m):
        """Initialize the weights of the network."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier uniform initialization for the weights
            nn.init.constant_(m.bias, 0)       # Set biases to zero

    def update_target_network(self):
        # Soft update of the target network
        for target_param, local_param in zip(self.target_network.parameters(), self.qnetwork.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def act(self, state, epsilon=0.1):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.qnetwork(state)
            action_index = torch.argmax(action_values).item()
            return action_index
        else:
            return random.choice([0, 1, 2, 3])

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
        states = torch.FloatTensor(np.array(states))  # [batch_size, state_size]
        actions = torch.LongTensor(np.array(actions)).view(-1, 1)  # [batch_size, 1]
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1)  # [batch_size, 1]
        next_states = torch.FloatTensor(np.array(next_states))  # [batch_size, state_size]
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1)  # [batch_size, 1]

        # Compute current Q-values
        q_values = self.qnetwork(states)  # [batch_size, action_size]

        # Double DQN logic:
        # Use the Q-network to select actions (greedy policy)
        with torch.no_grad():
            next_q_values = self.qnetwork(next_states)
            max_actions = next_q_values.argmax(1).unsqueeze(1)  # [batch_size, 1]

            # Use the target network to evaluate the Q-values of those actions
            next_q_values_target = self.target_network(next_states).gather(1, max_actions)  # [batch_size, 1]


        # Compute target Q-values
        q_targets = rewards + (self.gamma * next_q_values_target * (1 - dones))  # [batch_size, 1]

        # # Standard DQN logic: compute the maximum Q-value for next states directly from the target network
        # with torch.no_grad():
        #     next_q_values_target = self.target_network(next_states).max(1).values  # Maximum Q-value for each next state

        # # Compute target Q-values
        # q_targets = rewards + (self.gamma * next_q_values_target * (1 - dones))  # [batch_size, 1]

        # Gather the Q-values for the actions taken
        q_values_for_actions = q_values.gather(1, actions)  # [batch_size, 1]

        # Compute loss using Huber loss
        loss = nn.SmoothL1Loss()(q_values_for_actions, q_targets)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.update_target_network()

        return loss.item()  # Return the loss value

class GridWorldEnv:
    def __init__(self):
        self.state_size = 12 + 2  # 12 original state values + 4 sine/cosine components for 2 angles
        self.action_size = 4
        self.position_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)
        self._episode_ended = False
        self.max_episode_duration = 10.0  # Set the maximum episode duration in seconds
        self.thruter_history_numbers = 100
        self.sevo_history_numbers = 100
        # Initialize joint positions for servos
        # self.joint_positions = np.zeros(2)  # Assuming 2 joints (port and starboard)
        # self.prev_joint_positions = np.zeros(2)  # For servo smoothness penalty
        self.joint_positions = np.zeros(2)  # Two servos: [sin1, sin2, cos1, cos2]
        self.prev_joint_positions = np.zeros(2)  # Same structure as joint_positions

        self.joint_positions_history = np.zeros((self.sevo_history_numbers, 2))  # Initialize with zeros
        self.u_prev = np.zeros((self.thruter_history_numbers, 4))  # to store previous thruster inputs

        # comment this for now if inference is running
        rospy.init_node('underwater_vehicle_env', anonymous=True)
        
        self.thruster_action_pub = rospy.Publisher('/thruster_action', Int32MultiArray, queue_size=10)
        
        
        
        # Initialize the policy control state
        self.use_policy = True  # Default is to use the policy
        self.stop_training = False  # Default is to continue training
        
        # service for enabling/disabling the policy
        rospy.Service('/toggle_policy', SetBool, self.toggle_policy_service)

        rospy.Subscriber('/race2/controller/process/error', ControlProcess, self.update_current_error)
        rospy.Subscriber('/race2/control/thruster/heave_bow', Float64, self.update_thrust_heave_bow)
        rospy.Subscriber('/race2/control/thruster/surge_port', Float64, self.update_thrust_surge_port)
        rospy.Subscriber('/race2/control/thruster/sway_stern', Float64, self.update_thrust_sway_stern)
        rospy.Subscriber('/race2/control/thruster/surge_starboard', Float64, self.update_thrust_surge_starboard)
        rospy.Subscriber('/race2/control/servos/joint_states', JointState, self.update_joint_states)
        
        self.reset()

    def toggle_policy_service(self, request):
        """Service callback to enable/disable policy use."""
        self.use_policy = request.data  # Enable policy if True, disable if False

        if not self.use_policy:
            # If the policy is disabled, set stop_training flag to True to stop the training loop
            self.stop_training = True
            # Also send the thruster command [1, 1, 1, 1, 1, 1]
            thruster_command = Int32MultiArray(data=[1, 1, 1, 1, 1, 1])
            self.thruster_action_pub.publish(thruster_command)
            rospy.loginfo("Policy disabled. Thruster set to [1, 1, 1, 1, 1, 1] and training stopped.")

        return SetBoolResponse(success=True, message="Policy control and training updated")


    def update_current_error(self, data):
        self.position_err = np.array([data.position.x, data.position.y, data.position.z])
        self.v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.orientation_err = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])

    def update_joint_states(self, data):
        
        angles = np.array(data.position[:2]) 
        self.joint_angles = angles  

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
            return np.concatenate([self.position_err, self.v_err, self.orientation_err, self.omega_ref_err]), 0, True, {}
        
        # Map the action index to actual action values
        action = action_mapping[action_index]
        action1, action2 = action  # Unpack the action values

        # Create the array: [action1, 1, action2, 1, 1, 1]
        thruster_command = Int32MultiArray(data=[action1, 1, action2, 1, 1, 1])

        # Publish the action array
        self.thruster_action_pub.publish(thruster_command)

        # Dynamic waiting based on error stabilization
        max_wait_time = 3.0 # Maximum time to wait
        wait_interval = 0.2  # Time between checks
        elapsed_wait_time = 0.0
        previous_error = None
        error_stability_threshold = 0.01  # Define a suitable threshold

        # while elapsed_wait_time < max_wait_time:
        #     rospy.sleep(wait_interval)
        #     elapsed_wait_time += wait_interval

        #     # Ensure state variables are updated via callbacks

        #     # Calculate the current total error
        #     current_error = np.concatenate([
        #         self.position_err,
        #         self.v_err,
        #         self.orientation_err,
        #         self.omega_ref_err
        #     ])
        #     # print(f"Current error: {current_error}")
            
        #     if previous_error is not None:
        #         error_diff = np.abs(current_error - previous_error)
        #         # print(f"Error difference: {error_diff}")

        #         # Check if all individual error differences are below the threshold
        #         if np.all(error_diff < error_stability_threshold):
        #             # All error components have stabilized sufficiently
        #             # print(f"All errors stabilized after {elapsed_wait_time:.2f} seconds")
        #             break  # Exit the loop if all error changes are below the threshold
        #         else:
        #             # Optionally, you can log which errors are still fluctuating
        #             unstable_indices = np.where(error_diff >= error_stability_threshold)[0]
        #             print(f"Errors at indices {unstable_indices} are still fluctuating.")

        #     previous_error = current_error

        rospy.sleep(0.208)
        # Get the current time
        current_time = rospy.get_time()

        # Calculate elapsed time since the episode started
        elapsed_time_total = current_time - self.start_time
        # print("Elapsed time: ", elapsed_time_total)  # Debugging

        # Determine if the episode has ended
        done = elapsed_time_total > self.max_episode_duration
        done = True  # Assume episode ends after one action
        if done:
            self._episode_ended = True

        # Calculate reward
        reward = self.calculate_reward()


        # Prepare the next state
        next_state = np.concatenate([
            self.position_err,
            self.v_err,
            self.orientation_err,
            self.omega_ref_err,
            self.joint_positions
        ])

        return next_state, reward, done, {}


    def reset(self):
        self._episode_ended = False
        self.start_time = rospy.get_time()  # Get the current time

        # Initialize joint angles randomly or to a specific value
        initial_joint_angles = np.random.uniform(low=-np.pi, high=np.pi, size=2)
        self.joint_angles = initial_joint_angles

        # Initialize joint_positions_history with the initial joint angles
        # self.joint_positions_history = np.tile(self.joint_angles, (32, 2))  # Shape: (32, 2)
        self.joint_positions_history = np.full(self.joint_positions_history.shape, 0)
        # Return the initial state
        return np.concatenate([
            self.position_err,
            self.v_err,
            self.orientation_err,
            self.omega_ref_err,
            self.joint_positions 
        ])

    def calculate_reward(self):
        # Reward function constants
        w1 = 1.0  # Performance error weight
        w2 = 3.0  # Servo smoothness penalty weight
        w3 = 0.0  # Thruster usage penalty weight
        w4 = 2.0  # Action smoothness penalty weight
        w5 = 0.0  # Servo angle penalty weight
        w6 = 1.0

        # Weights for state errors (adjust as needed)
        state_error_weights = np.array([
            0.0,  # Position error x (not influenced by agent)
            0.0,  # Position error y
            0.5,  # Position error z
            0.5,  # Velocity error vx (surge)
            0.5,  # Velocity error vy (sway)
            0.0,  # Velocity error vz (heave)
            0.5,  # Orientation error roll
            0.5,  # Orientation error pitch
            1.0,  # Orientation error yaw
            0.0,  # Angular velocity error p
            0.0,  # Angular velocity error q
            0.0   # Angular velocity error r
        ])

        # Compute the error vector
        error = np.concatenate([
            self.position_err,       # Position error (x, y, z)
            self.v_err,              # Velocity error (vx, vy, vz)
            self.orientation_err,    # Orientation error (roll, pitch, yaw)
            self.omega_ref_err       # Angular velocity error (p, q, r)
        ]).astype(np.float32)

        # Compute performance error (quadratic penalty)
        weighted_errors = state_error_weights * error
        performance_error = np.sum(weighted_errors ** 2)

        # # Servo movement penalty
        # angle_diffs = angle_difference(self.joint_positions, self.prev_joint_positions)
        # servo_movement_penalty = np.linalg.norm(angle_diffs)
        # self.prev_joint_positions = self.joint_positions.copy()

        # Servo smoothness penalty using sine and cosine components for smoothness
        servo_smoothness_penalty = 0
        delta_theta = np.zeros(2)
        print(self.joint_positions_history.shape) 

        for i in range(2):  # Assuming two servos
            
            # Compute average sine and cosine of the historical angles
            avg_sin = np.average(np.sin(self.joint_positions_history[:, i]))
            avg_cos = np.average(np.cos(self.joint_positions_history[:, i]))
            print(avg_cos.size)
            # Calculate the average angle in radians from the sine and cosine averages
            historical_avg_angle = np.arctan2(avg_sin, avg_cos)
            
            # Get the current angle from self.joint_angles
            current_angle = self.joint_angles[i]

            # Calculate the angular difference
            delta_theta[i] = np.abs(current_angle - historical_avg_angle)

        # Accumulate the smoothness penalty
        servo_smoothness_penalty = np.linalg.norm(delta_theta)
        print(self.joint_positions_history.shape)  # Use the history directly
        print(self.joint_angles.shape)
        self.joint_positions_history = np.vstack((self.joint_positions_history[1:], self.joint_angles))

        # self.joint_positions_history[:-1] = self.joint_positions_history[1:]

        # self.joint_positions_history[-1] = self.joint_angles


        # Thruster usage penalty
        u_t = np.array([
            self.thrust_heave_bow,
            self.thrust_surge_port,
            self.thrust_surge_starboard,
            self.thrust_sway_stern
        ])
        thruster_usage_penalty = np.sum(abs(u_t))

        # thruster smoothness penalty
        if(len(self.u_prev) > 0):
            print("size")
            print(u_t.size)
            print(self.u_prev.size)
            # temp_var =np.average(self.u_prev, axis=0) 
            # print(temp_var.size)
            thruster_smoothness_penalty = np.linalg.norm(u_t - np.average(self.u_prev, axis=0) )
            # self.u_prev = np.roll(u_t, shift=-1, axis=0)
            self.u_prev = np.vstack((self.u_prev[1:], u_t))


        # thruster delta reward
        thruster_delta_reward = np.linalg.norm(u_t - self.u_prev[-2])

        #include servo angle penalties or rewards
        servo_angle_penalty = np.linalg.norm(self.joint_positions)

        # Total reward
        reward = - (
            w1 * performance_error +
            w2 * servo_smoothness_penalty +
            w3 * thruster_usage_penalty +
            w4 * thruster_smoothness_penalty +
            w5 * servo_angle_penalty +  
            w6 * thruster_delta_reward
        )
        return reward


def angle_difference(angle1, angle2):
    """Compute the minimal difference between two angles, handling wrapping."""
    diff = angle1 - angle2
    return (diff + np.pi) % (2 * np.pi) - np.pi

    # def calculate_reward(self):
    #     # Reward function constants
    #     w1 = 1.0  # Performance error weight
    #     w2 = 0.8  # Servo movement smoothness penalty weight
    #     w3 = 0.1  # Thruster usage penalty weight
    #     w4 = 0.1  # Action smoothness penalty weight

    #     # Weights for state errors
    #     state_error_weights = np.array([
    #         0.0,  # Position error x (not influenced by agent)
    #         0.0,  # Position error y
    #         0.5,  # Position error z
    #         0.5,  # Velocity error vx (surge)
    #         0.5,  # Velocity error vy (sway)
    #         0.0,  # Velocity error vz (heave)
    #         0.5,  # Orientation error roll
    #         0.5,  # Orientation error pitch
    #         0.5,  # Orientation error yaw
    #         0.0,  # Angular velocity error p
    #         0.0,  # Angular velocity error q
    #         0.0   # Angular velocity error r
    #     ])

    #     # Compute the error vector
    #     error = np.concatenate([
    #         self.position_err,       # Position error (x, y, z)
    #         self.v_err,              # Velocity error (vx, vy, vz)
    #         self.orientation_err,    # Orientation error (roll, pitch, yaw)
    #         self.omega_ref_err       # Angular velocity error (p, q, r)
    #     ]).astype(np.float32)

    #     # Compute performance error (quadratic penalty)
    #     weighted_errors = state_error_weights * error
    #     performance_error = np.sum(weighted_errors ** 2)

    #     # Servo movement smoothness penalty (history-based)
    #     servo_smoothness_penalty = np.mean(np.linalg.norm(np.diff(self.servo_pose_history, axis=0), axis=1))

    #     # Update servo pose history
    #     self.servo_pose_history = np.roll(self.servo_pose_history, shift=-1, axis=0)
    #     self.servo_pose_history[-1] = self.joint_positions

    #     # Thruster usage penalty
    #     u_t = np.array([
    #         self.thrust_heave_bow,
    #         self.thrust_surge_port,
    #         self.thrust_surge_starboard,
    #         self.thrust_sway_stern
    #     ])
    #     thruster_usage_penalty = np.sum(np.abs(u_t))

    #     # Action smoothness penalty (history-based)
    #     # Calculate average of recent control inputs
    #     control_input_history_avg = np.mean(self.control_input_history, axis=0)
    #     action_smoothness_penalty = np.linalg.norm(control_input_history_avg - u_t)

    #     # Update control input history
    #     self.control_input_history = np.roll(self.control_input_history, shift=-1, axis=0)
    #     self.control_input_history[-1] = u_t
 
    #     # Total reward
    #     reward = - (
    #         w1 * performance_error +
    #         w2 * servo_smoothness_penalty +
    #         w3 * thruster_usage_penalty +
    #         w4 * action_smoothness_penalty
    #     )

    #     return reward

def continuous_learning(env, agent, max_episodes=600, max_t=1000, target_avg_reward=None):
    episode_count = 0
    rate = rospy.Rate(10)  # Set a rate (e.g., 10 Hz)

    epsilon = 1.0  # Initialize epsilon
    epsilon_decay = 0.991  # Decay rate
    epsilon_min = 0.2  # Minimum epsilon value

    # Initialize plotting
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))  # Added a third subplot for loss

    reward_history = []
    q_value_history = []
    loss_history = []
    episodes = []

    # Set up the reward plot
    ax[0].set_title('Total Reward per Episode')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Total Reward')
    reward_line, = ax[0].plot([], [], label='Reward')
    ax[0].legend()

    # Set up the Q-value plot
    ax[1].set_title('Max Q-value per Episode')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Max Q-value')
    q_value_line, = ax[1].plot([], [], label='Max Q-value', color='orange')
    ax[1].legend()

    # Set up the loss plot
    ax[2].set_title('Average Loss per Episode')
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('Average Loss')
    loss_line, = ax[2].plot([], [], label='Loss', color='green')
    ax[2].legend()

    # Training Loop
    while episode_count < max_episodes and not rospy.is_shutdown():
        
        # Check if training is disabled
        if env.stop_training:
            rospy.loginfo("Training has been stopped.")
            break  # Exit the training loop if training is stopped

        episode_count += 1
        state = env.reset()  # Reset environment to get initial state
        score = 0
        max_q_value = -float('inf')  # Initialize max Q-value for this episode
        episode_loss = 0.0  # Initialize episode loss
        loss_steps = 0  # Number of steps where learning occurred

        for t in range(max_t):  # Limit each episode to max_t steps
            action_index = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action_index)
            loss = agent.step(state, action_index, reward, next_state, done)
            state = next_state

            score += reward

            # Get Q-values for the current state
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.qnetwork(state_tensor)
            current_max_q = q_values.max().item()
            if current_max_q > max_q_value:
                max_q_value = current_max_q  # Update max Q-value for this episode

            # Accumulate loss if learning occurred
            if loss is not None:
                episode_loss += loss
                loss_steps += 1

            if done:
                break

            # Sleep to maintain the loop rate
            rate.sleep()

        # Decay epsilon after each episode
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # Calculate average loss for the episode
        average_loss = episode_loss / loss_steps if loss_steps > 0 else 0.0

        # Append data for plotting
        episodes.append(episode_count)
        reward_history.append(score)
        q_value_history.append(max_q_value)
        loss_history.append(average_loss)

        # Update the plots
        update_plots(ax, episodes, reward_history, q_value_history, loss_history,
                     reward_line, q_value_line, loss_line)

        # Optionally, print the episode score and loss for monitoring
        print(f"Episode {episode_count}: Score: {score:.2f}, Max Q-value: {max_q_value:.2f}, Average Loss: {average_loss:.4f}, Epsilon: {epsilon:.3f}")

        # Check if average reward over the last N episodes meets the target
        if target_avg_reward is not None and len(reward_history) >= 10:
            avg_reward_recent = np.mean(reward_history[-10:])
            if avg_reward_recent >= target_avg_reward:
                print(f"Stopping training as average reward over last 10 episodes is {avg_reward_recent:.2f} (>= {target_avg_reward})")
                break

    # After training, set epsilon to 0 to use the greedy policy
    epsilon = 0.0

    # Save the trained model
    save_model(agent)

    # Now, enter an infinite loop where the agent continues to act using the learned policy
    print("Training complete. Continuing to run with the learned policy.")
    while not rospy.is_shutdown():
        state = env.reset()
        done = False
        total_reward = 0
        while not done and not rospy.is_shutdown():
            action_index = agent.act(state, epsilon=0.0)  # Use greedy policy
            next_state, reward, done, _ = env.step(action_index)
            state = next_state
            total_reward += reward

            # Sleep to maintain the loop rate
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
    plt.pause(0.01)  # Pause to update the plots

def save_model(agent, filename_prefix='dqn_model'):
    """
    Save the trained model to a file with the current date and time appended to the filename.

    Args:
        agent: The agent containing the Q-network to be saved.
        filename_prefix (str): The prefix for the filename. Defaults to 'dqn_model'.
    """
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create the filename with date and time
    filename = f"{filename_prefix}_{current_time}.pth"
    
    # Save the state dictionary of the Q-network
    torch.save(agent.qnetwork.state_dict(), filename)
    
    print(f"Model saved to {filename}")

def evaluate_agent(env, agent, num_episodes=10):
    total_scores = []
    for episode in range(num_episodes):
        state = env.reset()
        score = 0
        done = False
        while not done and not rospy.is_shutdown():
            action_index = agent.act(state, epsilon=0.0)  # Greedy policy
            next_state, reward, done, _ = env.step(action_index)
            state = next_state
            score += reward
        total_scores.append(score)
        print(f"Evaluation Episode {episode + 1}: Score: {score:.2f}")
    avg_score = np.mean(total_scores)
    print(f"Average Evaluation Score over {num_episodes} episodes: {avg_score:.2f}")

if __name__ == '__main__':
    env = GridWorldEnv()
    state_size = env.state_size
    action_size = env.action_size
    agent = Agent(state_size, action_size)
    
    # Train the agent
    continuous_learning(env, agent, max_episodes=1200, target_avg_reward=20)