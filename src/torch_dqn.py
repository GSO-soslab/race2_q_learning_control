import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rospy
from mvp_msgs.msg import ControlProcess
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float32,Int32MultiArray
from collections import deque
import random
import matplotlib.pyplot as plt


# action_mapping = {
#     0: [-1, -1],
#     1: [-1, 1],
#     2: [1, -1],
#     3: [1, 1]
# }

action_mapping = {
    0: [-1, -1],
    1: [1, 1]
}

def angle_difference(angle1, angle2):
    diff = angle1 - angle2
    return (diff + np.pi) % (2 * np.pi) - np.pi

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=(500, 400, 300)):
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

class GridWorldEnv:
    def __init__(self):
        self.state_size = 12
        self.action_size = 2
        self.position_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)
        self.joint_positions = np.zeros(2)
        self._episode_ended = False

        # Initialize setpoint variables
        self.current_setpoint = np.zeros(12)  # Store the current setpoint (position, velocity, orientation, angular velocity)
        self.previous_setpoint = np.zeros(12)  # Store the previous setpoint

        # Initialize joint positions for servos
        self.joint_positions = np.zeros(2)  # Assuming 2 joints (port and starboard)
        self.prev_joint_positions = np.zeros(2)  # For servo smoothness penalty
        self.joint_positions_history = deque(maxlen=1000)  # Keeps the last 100 positions


        self.u_prev = np.zeros((100, 4))  # to store 10 previous control inputs, each having 2 values (surge_port and surge_starboard)

        rospy.init_node('underwater_vehicle_env', anonymous=True)
        self.thruster_action_pub = rospy.Publisher('/thruster_action', Int32MultiArray, queue_size=10)
        rospy.Subscriber('/race2/controller/process/error', ControlProcess, self.update_current_error)
        rospy.Subscriber('/race2/controller/process/set_point', ControlProcess, self.update_current_setpoint)
        rospy.Subscriber('/race2/control/thruster/heave_bow', Float64, self.update_thrust_heave_bow)
        rospy.Subscriber('/race2/control/thruster/surge_port', Float64, self.update_thrust_surge_port)
        rospy.Subscriber('/race2/control/thruster/sway_stern', Float64, self.update_thrust_sway_stern)
        rospy.Subscriber('/race2/control/thruster/surge_starboard', Float64, self.update_thrust_surge_starboard)
        rospy.Subscriber('/race2/control/servos/joint_states', JointState, self.update_joint_states)
        self.reset()

    def update_current_setpoint(self, data):
        # Update setpoint values based on data received from ROS
        self.previous_setpoint = self.current_setpoint.copy()  # Store the previous setpoint
        self.current_setpoint = np.array([
            data.position.x, data.position.y, data.position.z,  # Position setpoint
            data.velocity.x, data.velocity.y, data.velocity.z,  # Velocity setpoint
            data.orientation.x, data.orientation.y, data.orientation.z,  # Orientation setpoint
            data.angular_rate.x, data.angular_rate.y, data.angular_rate.z  # Angular rate setpoint
        ])

    def calculate_lambda(self):
        # Calculate Lambda dynamically based on setpoint changes
        Lambda = np.zeros(12)  # Initialize a zero Lambda array

        # Compare current setpoint with the previous one and set corresponding Lambda values to 1 if there is a change
        for i in range(len(self.current_setpoint)):
            if self.current_setpoint[i] != self.previous_setpoint[i]:
                Lambda[i] = 1.0

        # Convert Lambda array to a diagonal matrix
        Lambda_matrix = np.diag(Lambda)
        return Lambda_matrix

    def update_current_error(self, data):
        self.position_err = np.array([data.position.x, data.position.y, data.position.z])
        self.v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.orientation_err = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])

    def update_joint_states(self, data):
        self.joint_positions = np.array(data.position[:2])

    def update_thrust_surge_port(self, data):
        self.thrust_surge_port = data.data

    def update_thrust_surge_starboard(self, data):
        self.thrust_surge_starboard = data.data

    def update_thrust_heave_bow(self, data):
        self.thrust_heave_bow = data.data

    def update_thrust_sway_stern(self, data):
        self.thrust_sway_stern = data.data


    # def step(self, action_index):
    #     # Ensure action_mapping is accessible
    #     global action_mapping
    #     # Map the action index to actual action values
    #     action = action_mapping[action_index]
    #     action1, action2 = action  # Unpack the action values

    #     # Create the array: [action1, 1, action2, 1, 1, 1]
    #     thruster_command = Int32MultiArray(data=[action1, 1, action2, 1, 1, 1])

    #     # Publish the action array
    #     self.thruster_action_pub.publish(thruster_command)

    #     # **Add a wait time to allow the AUV to adapt**
    #     rospy.sleep(0.1)  # Sleep for 1 second

    #     # Update the rest of the step logic
    #     current_time = rospy.get_time()
    #     reward = self.calculate_reward()

    #     # Prepare the next state
    #     next_state = np.concatenate([
    #         self.position_err,
    #         self.v_err,
    #         self.orientation_err,
    #         self.omega_ref_err
    #     ])

    #     # Determine if the episode has ended
    #     if current_time > 100000:
    #         self._episode_ended = True
    #         done = True
    #     else:
    #         done = False

    #     # Return the next_state, reward, done flag, and info dict
    #     return next_state, reward, done, {}

    def step(self, action_index):
        # Ensure action_mapping is accessible
        global action_mapping
        # Map the action index to actual action values
        action = action_mapping[action_index]
        action1, action2 = action  # Unpack the action values

        # Create the array: [action1, 1, action2, 1, 1, 1]
        thruster_command = Int32MultiArray(data=[action1, 1, action2, 1, 1, 1])

        # Publish the action array
        self.thruster_action_pub.publish(thruster_command)

        # **Wait for AUV to stabilize**
        rospy.sleep(2.0)  # Stabilization time of 1 seconds

        # Update the rest of the step logic after stabilization
        current_time = rospy.get_time()
        reward = self.calculate_reward()

        # Prepare the next state
        next_state = np.concatenate([
            self.position_err,
            self.v_err,
            self.orientation_err,
            self.omega_ref_err
        ])

        # Determine if the episode has ended
        if current_time > 100000:
            self._episode_ended = True
            done = True
        else:
            done = False

        # Return the next_state, reward, done flag, and info dict
        return next_state, reward, done, {}


    def reset(self):
        self._episode_ended = False
        self.position_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)
        return np.concatenate([self.position_err, self.v_err, self.orientation_err, self.omega_ref_err])

    # def angle_difference(self, angle1, angle2):
    #     diff = angle1 - angle2
    #     return (diff + np.pi) % (2 * np.pi) - np.pi

    def calculate_reward(self):
        # Reward function constants
        Lambda = np.diag([0.2, 0.0, 0.0, 1.0, 0.5, 0.1, 0.1, 0.5, 0.4, 0.2, 0.5, 1.0])

        # Dynamically update Lambda based on setpoint changes
        #Lambda = self.calculate_lambda()
        #print(Lambda)
        a = 0.7
        xi = 0.02
        theta = 0.05
        zeta = 0.1
        alpha = 0.1  # For thruster magnitude penalty
        small_lambda = 0.2
        # Normalize angular errors
        # orientation_err_normalized = angle_difference(self.orientation_err, np.zeros_like(self.orientation_err))
        # omega_ref_err_normalized = angle_difference(self.omega_ref_err, np.zeros_like(self.omega_ref_err))

        # Compute the error vector
        # error = np.concatenate([
        #     self.position_err,        # Position error (x, y, z)
        #     self.v_err,               # Velocity error (vz, vy, vx)
        #     omega_ref_err_normalized, # Normalized angular velocity error
        #     orientation_err_normalized # Normalized orientation error
        # ]).astype(np.float32)

        error = np.concatenate([
            self.position_err,        # Position error (x, y, z)
            self.v_err,               # Velocity error (vz, vy, vx)
            self.orientation_err, # Normalized angular velocity error
            self.omega_ref_err # Normalized orientation error
        ]).astype(np.float32)

        print("Error:", error)
        squared_error = np.dot(error.T, np.dot(Lambda, error))
        # print (squared_error)
        # term1 = small_lambda * np.exp((-1.0 / a**2 )* squared_error)
        # print(-1/a**2)

        print("Squared error:", squared_error)
        exp_term = np.exp((-1.0 / a**2) * squared_error * 0.0001)
        print("Exponential term:", exp_term)
        term1 = small_lambda * exp_term
        print("Term1:", term1)

        # Gather the current control inputs (u_t)
        u_t = np.array([
            self.thrust_heave_bow,
            self.thrust_surge_port,
            self.thrust_surge_starboard,
            self.thrust_sway_stern
        ])

        # Update previous control inputs and compute deviation from average
        self.u_prev = np.roll(self.u_prev, shift=-1, axis=0)
        self.u_prev[-1] = u_t
        u_avg = np.mean(self.u_prev, axis=0)
        term3 = -xi * np.linalg.norm(u_avg - u_t)

        # Compute thruster magnitude penalty
        thruster_magnitude_penalty = alpha * np.linalg.norm(u_t)

        # Update joint positions history
        self.joint_positions_history.append(self.joint_positions.copy())

        # Compute average of previous servo positions
        if len(self.joint_positions_history) > 0:
            joint_positions_array = np.array(self.joint_positions_history)
            servo_positions_avg = np.mean(joint_positions_array, axis=0)
        else:
            servo_positions_avg = self.joint_positions

        # Compute servo smoothness penalty (change from previous position)
        angle_differences = angle_difference(self.joint_positions, self.prev_joint_positions)
        servo_smoothness_penalty = np.linalg.norm(angle_differences)
        self.prev_joint_positions = self.joint_positions.copy()

        # Compute servo control penalty (difference from average position)
        servo_control_penalty = np.linalg.norm(self.joint_positions - servo_positions_avg)

        # Combine terms into the reward
        reward = term1 + term3 - thruster_magnitude_penalty - theta * servo_smoothness_penalty - zeta * servo_control_penalty
        print (term1)
        return reward

class Agent:
    def __init__(self, state_size, action_size, buffer_size=50000, batch_size=62, gamma=0.99, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size  # Now 4
        self.gamma = gamma
        self.batch_size = batch_size
        self.qnetwork = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def act(self, state, epsilon=0.1):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.qnetwork(state)
            action_index = torch.argmax(action_values).item()
            return action_index
        else:
            # return random.choice([0, 1, 2, 3])
            return random.choice([0, 1])

    def step(self, state, action, reward, next_state, done):
        # Store the experience in the replay buffer
        self.memory.add((state, action, reward, next_state, done))

        # If there are enough samples in memory, learn from them
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

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

        # Compute next Q-values
        next_q_values = self.qnetwork(next_states).detach()  # [batch_size, action_size]

        # Compute target Q-values
        max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)  # [batch_size, 1]
        q_targets = rewards + (self.gamma * max_next_q_values * (1 - dones))  # [batch_size, 1]

        # Gather the Q-values for the actions taken
        q_values_for_actions = q_values.gather(1, actions)  # [batch_size, 1]

        # Compute loss
        loss = nn.MSELoss()(q_values_for_actions, q_targets)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_agent(env, agent, n_episodes=5000, max_t=1000):
    scores = []
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    q_values_history = []  # List to store Q-values for plotting

    for episode in range(n_episodes):
        state = env.reset()
        score = 0

        for t in range(max_t):
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            # Store the Q-values for the current state
            current_q_values = agent.qnetwork(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
            #q_values_history.append(current_q_values.flatten())  # Flatten if multi-dimensional
            q_values_history.append(np.max(current_q_values))

            if done:
                break

        scores.append(score)
        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        #print(f"Episode {episode}: Score: {score}, Replay Buffer Size: {len(agent.memory)}, Q-values: {current_q_values}")

        # Convert q_values_history to a NumPy array for easier manipulation
        q_values_history = np.array(q_values_history)

        # Plot the scores
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(scores)
        plt.title('Scores over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Score')

        # Plot the Q-values
        plt.subplot(1, 2, 2)
        plt.plot(q_values_history)
        plt.title('Q-values over Episodes')
        plt.xlabel('Steps')
        plt.ylabel('Q-values')
        plt.legend([f'Action {i}' for i in range(agent.action_size)], loc='upper right')

        plt.tight_layout()
        plt.show()

    return scores


def run_policy(env, agent, max_t=100):
    state = env.reset()
    while not rospy.is_shutdown():  # Ensure ROS can handle shutdown signals
        action = agent.act(state, epsilon=0.0)  # Use a greedy policy
        next_state, reward, done, _ = env.step(action)
        state = next_state

        # Continually learn even while running the policy
        agent.step(state, action, reward, next_state, done)  # Continue learning

        if done:
            state = env.reset()

def continuous_learning(env, agent, eval_interval=2000, max_t=1000):
    episode_count = 0
    rate = rospy.Rate(10)  # Set a rate (e.g., 10 Hz)

    epsilon = 1.0  # Initialize epsilon
    epsilon_decay = 0.995  # Decay rate
    epsilon_min = 0.01  # Minimum epsilon value


    # Initialize plotting
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    reward_history = []
    q_value_history = []
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

    while not rospy.is_shutdown():
        episode_count += 1
        state = env.reset()  # Reset environment to get initial state
        score = 0
        max_q_value = -float('inf')  # Initialize max Q-value for this episode

        for t in range(max_t):  # Limit each episode to max_t steps
            action_index = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action_index)
            agent.step(state, action_index, reward, next_state, done)
            state = next_state

            score += reward

            # Get Q-values for the current state
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.qnetwork(state_tensor)
            current_max_q = q_values.max().item()
            if current_max_q > max_q_value:
                max_q_value = current_max_q  # Update max Q-value for this episode

            if done:
                break

            # Sleep to maintain the loop rate
            rate.sleep()


        # Decay epsilon after each episode
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # Append data for plotting
        reward_history.append(score)
        q_value_history.append(max_q_value)
        episodes.append(episode_count)

        # Update the plots
        reward_line.set_xdata(episodes)
        reward_line.set_ydata(reward_history)
        ax[0].relim()
        ax[0].autoscale_view()

        q_value_line.set_xdata(episodes)
        q_value_line.set_ydata(q_value_history)
        ax[1].relim()
        ax[1].autoscale_view()

        plt.draw()
        plt.pause(0.001)  # Pause to update the plots

        # Optionally, print the episode score for monitoring
       # print(f"Episode {episode_count}: Score: {score:.2f}, Max Q-value: {max_q_value:.2f}, Epsilon: {epsilon:.3f}")

        if episode_count % eval_interval == 0:
            # Optionally evaluate performance after a certain number of episodes
            evaluate_agent(env, agent)



def evaluate_agent(env, agent):
    eval_scores = []
    for _ in range(10):  # Evaluate over 10 episodes
        state = env.reset()
        score = 0
        for t in range(100):
            action = agent.act(state, epsilon=0.0)  # Use the greedy policy
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward

            if done:
                break
        eval_scores.append(score)
    # print(f"Evaluation Scores: {eval_scores}, Average Score: {np.mean(eval_scores)}")


if __name__ == "__main__":
    env = GridWorldEnv()
    # agent = Agent(state_size=12, action_size=4)
    agent = Agent(state_size=12, action_size=2)

    continuous_learning(env, agent)
    # Train the agent
    # scores = train_agent(env, agent)

    # Optionally plot results
    # plt.plot(scores)
    # plt.show()

    # Run the trained policy
    run_policy(env, agent)