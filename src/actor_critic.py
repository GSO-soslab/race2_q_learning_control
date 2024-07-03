import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rospy
from mvp_msgs.msg import ControlProcess
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Int32MultiArray
from collections import deque
import random
import matplotlib.pyplot as plt

action_mapping = {
    0: [-1, -1],
    1: [-1, 1],
    2: [1, -1],
    3: [1, 1]
}


def angle_difference(angle1, angle2):
    diff = angle1 - angle2
    return (diff + np.pi) % (2 * np.pi) - np.pi


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=(300, 100, 20)):
        super(Actor, self).__init__()
        layers = []
        input_size = state_size

        # Create the hidden layers
        for hidden_layer in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_layer))
            layers.append(nn.ReLU())
            input_size = hidden_layer

        # Output layer for action probabilities
        layers.append(nn.Linear(input_size, action_size))
        layers.append(nn.Softmax(dim=-1))

        # Define the network
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, state_size, hidden_layers=(300, 100, 20)):
        super(Critic, self).__init__()
        layers = []
        input_size = state_size

        # Create the hidden layers
        for hidden_layer in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_layer))
            layers.append(nn.ReLU())
            input_size = hidden_layer

        # Output layer for state value
        layers.append(nn.Linear(input_size, 1))

        # Define the network
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GridWorldEnv:
    def __init__(self):
        self.state_size = 12
        self.action_size = 4
        self.position_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)
        self._episode_ended = False

        # Initialize joint positions for servos
        self.joint_positions = np.zeros(2)  # Assuming you have 2 joints (port and starboard)
        self.prev_joint_positions = np.zeros(2)  # For servo smoothness penalty
        self.joint_positions_history = deque(maxlen=2000)  # Keeps the last 2000 positions

        self.u_prev = np.zeros((100, 4))  # to store previous control inputs

        rospy.init_node('underwater_vehicle_env', anonymous=True)
        self.thruster_action_pub = rospy.Publisher('/thruster_action', Int32MultiArray, queue_size=10)
        rospy.Subscriber('/race2/controller/process/error', ControlProcess, self.update_current_error)
        rospy.Subscriber('/race2/control/thruster/heave_bow', Float64, self.update_thrust_heave_bow)
        rospy.Subscriber('/race2/control/thruster/surge_port', Float64, self.update_thrust_surge_port)
        rospy.Subscriber('/race2/control/thruster/sway_stern', Float64, self.update_thrust_sway_stern)
        rospy.Subscriber('/race2/control/thruster/surge_starboard', Float64, self.update_thrust_surge_starboard)
        rospy.Subscriber('/race2/control/servos/joint_states', JointState, self.update_joint_states)
        self.reset()

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

    def step(self, action_index):
        # Map the action index to actual action values
        action = action_mapping[action_index]
        action1, action2 = action  # Unpack the action values

        # Create the array: [action1, 1, action2, 1, 1, 1]
        thruster_command = Int32MultiArray(data=[action1, 1, action2, 1, 1, 1])

        # Publish the action array
        self.thruster_action_pub.publish(thruster_command)

        # Wait for AUV to stabilize
        rospy.sleep(0.18)  # Stabilization time

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

    def calculate_reward(self):
        # Reward function constants
        Lambda = np.diag([0.0, 0.0, 0.7, 1.0, 0.7, 0.0, 0.5, 1.0, 0.7, 0.0, 0.0, 0.0])
        a = 1.0
        xi = 0.2
        theta = 0.4
        zeta = 0.4
        alpha = 0.3  # For thruster magnitude penalty
        small_lambda = 0.9

        # Compute the error vector
        error = np.concatenate([
            self.position_err,        # Position error (x, y, z)
            self.v_err,               # Velocity error (vz, vy, vx)
            self.orientation_err,     # Orientation error
            self.omega_ref_err        # Angular velocity error
        ]).astype(np.float32)

        squared_error = np.dot(error.T, np.dot(Lambda, error))
        exp_term = np.exp((-1.0 / a**2) * squared_error)
        term1 = small_lambda * exp_term

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
        return reward

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
    def __init__(self, state_size, action_size, buffer_size=20000, batch_size=32, gamma=0.99, lr=1e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        # Actor and Critic networks
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        return action.item()

    def step(self, state, action, reward, next_state, done):
        # Store the experience in the replay buffer
        self.memory.add((state, action, reward, next_state, done))

        # If there are enough samples in memory, learn from them
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            value_loss = self.learn(experiences)
            return value_loss
        return None

    def learn(self, experiences):
        # Unpack experiences
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).view(-1)
        rewards = torch.FloatTensor(np.array(rewards)).view(-1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).view(-1)

        # Compute state values and advantages
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        advantages = rewards + self.gamma * next_values * (1 - dones) - values

        # Update Critic
        critic_loss = advantages.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        action_probs = self.actor(states)
        action_distribution = torch.distributions.Categorical(action_probs)
        log_probs = action_distribution.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Return critic loss for value loss plotting
        return critic_loss.item()

def continuous_learning(env, agent, max_t=1000):
    episode_count = 0
    rate = rospy.Rate(10)  # Set a rate (e.g., 10 Hz)

    # Initialize plotting
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    reward_history = []
    value_loss_history = []
    episodes = []

    # Set up the reward plot
    ax[0].set_title('Total Reward per Episode')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Total Reward')
    reward_line, = ax[0].plot([], [], label='Reward')
    ax[0].legend()

    # Set up the Value loss plot
    ax[1].set_title('Value Loss per Episode')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Value Loss')
    value_loss_line, = ax[1].plot([], [], label='Value Loss', color='orange')
    ax[1].legend()

    while not rospy.is_shutdown():
        episode_count += 1
        state = env.reset()  # Reset environment to get initial state
        score = 0
        value_losses = []

        for t in range(max_t):  # Limit each episode to max_t steps
            action_index = agent.act(state)
            next_state, reward, done, _ = env.step(action_index)
            value_loss = agent.step(state, action_index, reward, next_state, done)
            state = next_state

            score += reward
            if value_loss is not None:
                value_losses.append(value_loss)

            if done:
                break

            # Sleep to maintain the loop rate
            rate.sleep()

        # Calculate the average value loss for the episode
        avg_value_loss = np.mean(value_losses) if value_losses else 0

        # Append data for plotting
        reward_history.append(score)
        value_loss_history.append(avg_value_loss)
        episodes.append(episode_count)

        # Update the plots
        reward_line.set_xdata(episodes)
        reward_line.set_ydata(reward_history)
        ax[0].relim()
        ax[0].autoscale_view()

        value_loss_line.set_xdata(episodes)
        value_loss_line.set_ydata(value_loss_history)
        ax[1].relim()
        ax[1].autoscale_view()

        plt.draw()
        plt.pause(0.1)  # Pause to update the plots

        # Optionally, print the episode score for monitoring
        print(f"Episode {episode_count}: Score: {score:.2f}, Avg Value Loss: {avg_value_loss:.4f}")

        # Evaluate the agent periodically
        if episode_count % 50 == 0:
            evaluate_agent(env, agent)

def evaluate_agent(env, agent):
    eval_scores = []
    for _ in range(10):  # Evaluate over 10 episodes
        state = env.reset()
        score = 0
        for t in range(100):
            action = agent.act(state)  # Use the policy without exploration
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward

            if done:
                break
        eval_scores.append(score)
    print(f"Evaluation Scores: {eval_scores}, Average Score: {np.mean(eval_scores)}")

if __name__ == "__main__":
    env = GridWorldEnv()
    agent = Agent(state_size=12, action_size=4)

    continuous_learning(env, agent)
    
