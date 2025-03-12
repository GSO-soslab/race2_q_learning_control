import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
import time
from mvp_msgs.msg import ControlProcess

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNThrusterDirectionController:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self, update_target=False):
        if len(self.memory) < self.batch_size:
            return 0
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        if update_target:
            self.update_target_model()
        
        return loss.item()

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(torch.load(filepath))


class DQNThrusterDirectionNode(Node):
    def __init__(self):
        super().__init__('dqn_thruster_direction_controller')
        
        # Define state and action dimensions
        self.state_size = 12  # [position(3), orientation(3), velocity(3), angular_rate(3)]
        self.action_size = 4  # 2^2 = 4 combinations for two thrusters, each with 2 directions
        
        # Initialize DQN controller
        self.dqn = DQNThrusterDirectionController(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=10000,
            batch_size=32
        )
        
        # Initialize state variables
        self.position_state = np.zeros(3)
        self.orientation_state = np.zeros(3)
        self.v_state = np.zeros(3)
        self.omega_ref_state = np.zeros(3)
        
        # Keep track of previous state and action for learning
        self.prev_state = None
        self.prev_action = None
        self.prev_time = time.time()
        
        # Thresholds for determining when episode is done
        self.error_threshold = 0.05  # Consider position/orientation error below this as success
        
        # Create subscription to control process values
        self.create_subscription(
            ControlProcess,  # Replace with your actual message type
            '/race2_auv/controller/process/value',
            self.update_current_state,
            10
        )
        
        # Create publisher for thruster direction commands
        self.thruster_action_pub = self.create_publisher(
            Int16MultiArray,
            '/race2_auv/vector_thruster_direction',
            10
        )
        
        # Create publisher for debug info
        self.debug_publisher = self.create_publisher(
            Int16MultiArray,
            '/race2_auv/dqn/debug',
            10
        )
        
        # Training timer - runs at 5 Hz to match PID controller
        self.training_timer = self.create_timer(0.2, self.training_callback)
        
        self.get_logger().info('DQN Thruster Direction Controller initialized')
        
    def update_current_state(self, data):
        """Callback for state updates from PID controller"""
        # Extract state data from the message
        raw_position_state = np.array([data.position.x, data.position.y, data.position.z])
        raw_orientation_state = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        raw_v_state = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        raw_omega_ref_state = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
        
        # Store the raw state components
        self.position_state = raw_position_state
        self.orientation_state = raw_orientation_state
        self.v_state = raw_v_state
        self.omega_ref_state = raw_omega_ref_state
        
        # Create the combined state vector
        current_state = np.concatenate([
            self.position_state,
            self.orientation_state,
            self.v_state,
            self.omega_ref_state
        ])
        
        # Get action from DQN (thruster direction flags)
        action_idx = self.dqn.act(current_state)
        direction_flags = self.action_to_direction_flags(action_idx)
        
        # Publish direction flags as Int16MultiArray
        thruster_command = Int16MultiArray()
        thruster_command.data = [int(direction_flags[0]), int(direction_flags[1])]
        self.thruster_action_pub.publish(thruster_command)
        
        # Calculate reward based on current state
        if self.prev_state is not None:
            # Calculate time elapsed
            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time
            
            # Calculate reward
            reward = self.calculate_reward(self.prev_state, current_state, dt)
            
            # Determine if episode is done
            done = np.all(np.abs(current_state[:6]) < self.error_threshold)  # Check position and orientation errors
            
            # Store experience in replay memory
            self.dqn.remember(self.prev_state, self.prev_action, reward, current_state, done)
            
            # Publish debug info
            debug_msg = Int16MultiArray()
            debug_msg.data = [int(reward), int(done), int(self.dqn.epsilon * 100)]
            self.debug_publisher.publish(debug_msg)
            
            if done:
                self.get_logger().info(f"Goal reached! Reward: {reward:.2f}, Epsilon: {self.dqn.epsilon:.2f}")
                
                # Save model periodically when done
                self.dqn.save_model('dqn_thruster_model.h5')
        
        # Update previous state and action
        self.prev_state = current_state
        self.prev_action = action_idx
        
    def action_to_direction_flags(self, action_idx):
        """
        Convert action index to thruster direction flags
        
        Action mapping:
        0: [-1, -1] (Both thrusters reverse)
        1: [-1,  1] (Thruster 1 reverse, Thruster 2 forward)
        2: [ 1, -1] (Thruster 1 forward, Thruster 2 reverse)
        3: [ 1,  1] (Both thrusters forward)
        """
        if action_idx == 0:
            return [-1, -1]
        elif action_idx == 1:
            return [-1, 1]
        elif action_idx == 2:
            return [1, -1]
        elif action_idx == 3:
            return [1, 1]
        else:
            return [0, 0]  # Default case (shouldn't happen)
    
    def calculate_reward(self, prev_state, current_state, dt):
        """Calculate reward based on state improvement"""
        # Extract position and orientation errors from states
        prev_pos_error = np.linalg.norm(prev_state[:3])
        current_pos_error = np.linalg.norm(current_state[:3])
        
        prev_orient_error = np.linalg.norm(prev_state[3:6])
        current_orient_error = np.linalg.norm(current_state[3:6])
        
        # Reward for position error reduction
        pos_improvement = prev_pos_error - current_pos_error
        pos_reward = 10.0 * pos_improvement / dt if dt > 0 else 0
        
        # Reward for orientation error reduction
        orient_improvement = prev_orient_error - current_orient_error
        orient_reward = 5.0 * orient_improvement / dt if dt > 0 else 0
        
        # Penalty for high velocities (to promote smooth motion)
        velocity_magnitude = np.linalg.norm(current_state[6:9])
        velocity_penalty = -0.1 * velocity_magnitude if velocity_magnitude > 0.5 else 0
        
        # Penalty for oscillations
        oscillation_penalty = 0
        if np.any(np.sign(current_state[6:9]) != np.sign(prev_state[6:9])):
            oscillation_penalty -= 0.5
        
        # Combine rewards
        total_reward = pos_reward + orient_reward + velocity_penalty + oscillation_penalty
        
        # Bonus for reaching very low error
        if current_pos_error < self.error_threshold and current_orient_error < self.error_threshold:
            total_reward += 10.0
            
        return total_reward
    
    def training_callback(self):
        """Periodically train the DQN model at 5 Hz"""
        loss = self.dqn.replay(update_target=True)
        if loss > 0:
            self.get_logger().debug(f"Training loss: {loss:.4f}")


def main(args=None):
    rclpy.init(args=args)
    node = DQNThrusterDirectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()