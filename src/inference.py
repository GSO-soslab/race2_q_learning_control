import os
import yaml
import torch
import rclpy
import time
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int16MultiArray, Float64
from std_srvs.srv import SetBool
from mvp_msgs.msg import ControlProcess 
from torch_dqn import GridWorldEnv, AdaptiveScaler
from torch_dqn import QNetwork
import numpy as np


# Load configuration from config.yaml
config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


class InferenceAgent:
    def __init__(self, model_path, state_size, action_size, logger):
        self.logger = logger
        
        # Load the trained Q-network
        self.qnetwork = QNetwork(state_size, action_size, config['qnetwork']['hidden_layers'])
        try:
            self.qnetwork.load_state_dict(torch.load(model_path))
            self.qnetwork.eval()  # Set the network to evaluation mode (no training)
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    # def act(self, state):
    #     # Convert state to a tensor and get action values from the Q-network
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     with torch.no_grad():
    #         action_values = self.qnetwork(state)
    #     action_index = torch.argmax(action_values).item()
    #     return action_index

    def act(self, state, epsilon=0.1):
        """
        Select an action using an epsilon-greedy strategy.
        
        Args:
            state (numpy.ndarray): Current state
            epsilon (float): Probability of taking a random action
        
        Returns:
            int: Selected action index
        """
        # Convert state to a tensor and get action values from the Q-network
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork(state)
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Random action
            action_index = np.random.randint(0, action_values.shape[1])
        else:
            # Greedy action selection
            action_index = torch.argmax(action_values).item()
        
        return action_index

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.scaler = AdaptiveScaler()
        # Create callback group for services
        self.callback_group = ReentrantCallbackGroup()
        
        # Initialize inference_enabled as False by default
        self.inference_enabled = False
        self.current_state = None
        self.done = False
        
                # Initialize state variables
        self.position_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)
        self.position_state = np.zeros(3)
        self.v_state = np.zeros(3)
        self.orientation_state = np.zeros(3)
        self.omega_ref_state = np.zeros(3)
        self.joint_angles_port = 0.0
        self.joint_angles_starboard = 0.0
        self.joint_angles = [0.0, 0.0]
        self.thrust_heave_bow = 0.0
        self.thrust_surge_port = 0.0
        self.thrust_surge_starboard = 0.0
        self.thrust_sway_stern = 0.0
        self.action_mapping = {int(k): v for k, v in config['environment']['action_mapping'].items()}



        # Default thruster command when inference is disabled
        self.default_thruster_command = [1, 1]
        
        # Declare and get parameters
        self.declare_parameter('model_path', 'dqn_model_2025-03-07_16-07-20.pth')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        # Create QoS profile for better reliability
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10
        )

        # Initialize environment and agent
        self.env = GridWorldEnv(config)
        self.agent = InferenceAgent(
            model_path, 
            self.env.state_size, 
            self.env.action_size,
            self.get_logger()
        )
        
                # Create all subscribers
        self.create_subscription(
            ControlProcess, 
            '/race2_auv/controller/process/error', 
            self.update_current_error, 
            10)
        
        self.create_subscription(
            ControlProcess, 
            '/race2_auv/controller/process/value', 
            self.update_current_state, 
            10)
            
        self.create_subscription(
            Float64, 
            '/race2_auv/control/surge_port_servo', 
            self.update_joint_port, 
            10)
        
        self.create_subscription(
            Float64, 
            '/race2_auv/control/surge_starboard_servo', 
            self.update_joint_starboard, 
            10)
                             
        self.create_subscription(
            Float64, 
            '/race2_auv/control/thruster/heave_bow', 
            self.update_thrust_heave_bow, 
            10)
        
        self.create_subscription(
            Float64, 
            '/race2_auv/control/thruster/surge_port', 
            self.update_thrust_surge_port, 
            10)
        
        self.create_subscription(
            Float64, 
            '/race2_auv/control/thruster/surge_starboard', 
            self.update_thrust_surge_starboard, 
            10)
        
        self.create_subscription(
            Float64, 
            '/race2_auv/control/thruster/sway_stern', 
            self.update_thrust_sway_stern, 
            10)
        # Create publisher for thruster action
        self.thruster_action_pub = self.create_publisher(
            Int16MultiArray,
            '/race2_auv/vector_thruster_direction',
            qos_profile
        )

        # Create service for toggling inference with callback group
        self.toggle_service = self.create_service(
            SetBool,
            '/enable_inference',
            self.toggle_inference_callback,
            callback_group=self.callback_group
        )

        # Create client for toggle policy service
        self.toggle_policy_client = self.create_client(
            SetBool,
            '/toggle_policy',
            callback_group=self.callback_group
        )

        # Create timer for main loop with callback group
        self.timer = self.create_timer(
            0.19996,  # 5Hz rate
            self.inference_loop,
            callback_group=self.callback_group
        )
        
        self.episode = 0
        self.total_reward = 0
        self.get_logger().info('Inference node initialized with inference disabled')

    def update_current_error(self, data):
        self.position_err = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_err = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
        
    def update_current_state(self, data):
        self.position_state = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_state = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_state = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_state = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])

    def update_joint_port(self, data):
        self.joint_angles_port = data.data
        self.joint_angles[0] = data.data

    def update_joint_starboard(self, data):
        self.joint_angles_starboard = data.data
        self.joint_angles[1] = data.data

    def update_thrust_heave_bow(self, data):
        self.thrust_heave_bow = data.data

    def update_thrust_surge_port(self, data):
        self.thrust_surge_port = data.data

    def update_thrust_surge_starboard(self, data):
        self.thrust_surge_starboard = data.data

    def update_thrust_sway_stern(self, data):
        self.thrust_sway_stern = data.data

    def get_current_state(self):
        """Returns the current state vector."""
        state = np.concatenate([
            self.position_err[2:3],  # Depth
            self.v_err[:2],          # Surge and sway
            self.orientation_err[:3], # roll, pitch, yaw
            self.position_state[2:3],
            self.v_state[:2],
            self.orientation_state[:3],
            self.omega_ref_state[2:3],
            self.joint_angles,
            np.array([self.thrust_heave_bow,
                     self.thrust_surge_port,
                     self.thrust_surge_starboard,
                     self.thrust_sway_stern])
        ])

        normalized_state = self.scaler.update_and_normalize(state)
        return normalized_state

    def publish_default_thruster_command(self):
        """Publish the default thruster command."""
        msg = Int16MultiArray()
        msg.data = self.default_thruster_command
        self.thruster_action_pub.publish(msg)
        self.get_logger().info(f"Published default thruster command: {self.default_thruster_command}")
        
    def toggle_inference_callback(self, request, response):
        """Service callback to enable/disable inference."""
        prev_state = self.inference_enabled
        self.inference_enabled = request.data
        
        if self.inference_enabled:
            if not prev_state:  # Only reset if we're transitioning from disabled to enabled
                self.episode += 1
                self.current_state = self.env.reset()
                self.done = False
                self.total_reward = 0
                self.get_logger().info(f"Starting Episode {self.episode}")
                self.toggle_policy(True)
            self.get_logger().info("Inference enabled.")
        else:
            if prev_state:  # Only cleanup if we're transitioning from enabled to disabled
                self.toggle_policy(False)
                self.current_state = None
                self.done = True
            self.get_logger().info("Inference disabled.")
            # Publish default thruster command immediately when disabling
            self.publish_default_thruster_command()
        
        response.success = True
        response.message = "Inference state updated."
        return response

    def toggle_policy(self, enable_policy):
        """Calls the /toggle_policy service to enable or disable the policy."""
        if not self.toggle_policy_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/toggle_policy service not available')
            return False

        request = SetBool.Request()
        request.data = enable_policy

        try:
            future = self.toggle_policy_client.call_async(request)
            return True
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return False

    # def inference_loop(self):
    #     """Main inference loop."""
    #     if not self.inference_enabled:
    #         # Publish default thruster command
    #         self.publish_default_thruster_command()
    #         return

    #     if self.current_state is not None and not self.done:
    #         # Get action from the policy
    #         self.current_state = self.get_current_state()
    #         # print("Current State:", self.current_state)
    #         action_index = self.agent.act(self.current_state)
    #         print("Action Index", action_index)
    #         # Take the action in the environment
    #         next_state, reward, done, _ = self.env.step(action_index)
    #         # print(next_state)
    #         self.current_state = next_state
    #         self.total_reward += reward
    #         # print(self.total_reward)
    #         self.done = done

    #         if self.done:
    #             self.get_logger().info(f"Episode {self.episode} completed with total reward: {self.total_reward}")
    #             if self.inference_enabled:  # Only start new episode if still enabled
    #                 self.episode += 1
    #                 self.current_state = self.env.reset()
    #                 self.done = False
    #                 self.total_reward = 0
    #                 self.get_logger().info(f"Starting Episode {self.episode}")

    # def inference_loop(self):
    #     if not self.inference_enabled:
    #         self.publish_default_thruster_command()
    #         return

    #     # Get current state
    #     current_state = self.get_current_state()
        
    #     # Add some logging to debug
    #     # self.get_logger().info(f"Current state: {current_state}")
        
    #     # Get action from the policy
    #     action_index = self.agent.act(current_state)
        
    #     # Create and publish thruster command
    #     action = self.action_mapping[action_index]
    #     action1, action2 = action  # Unpack the action values
    #     self.thruster_action = action
    #     # Create the array: [action1, action2]
    #     thruster_command = Int16MultiArray(data=[action1, action2])
    #     print("Thruster1: {} , Thruster2: {}".format(action1, action2))
    #     # Publish the action array
    #     self.thruster_action_pub.publish(thruster_command)
    #     time.sleep(0.205)

    def inference_loop(self):
        if not self.inference_enabled:
            self.publish_default_thruster_command()
            return

        # Get current state
        current_state = self.get_current_state()
        
        # Optional: Log or print the current state for debugging
        # self.get_logger().info(f"Current state: {current_state}")
        
        # Get action from the policy with some exploration
        action_index = self.agent.act(current_state, epsilon=0.1)  # 2% random exploration
        
        # Map action index to actual thruster commands
        action = self.action_mapping[action_index]
        action1, action2 = action  # Unpack the action values
        
        # Optional: Log the selected action and its corresponding Q-values
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.agent.qnetwork(state_tensor)
            self.get_logger().info(f"Q-values: {q_values.numpy()[0]}")
            self.get_logger().info(f"Selected action: {action_index}, Thruster command: [{action1}, {action2}]")
        
        # Create and publish thruster command
        thruster_command = Int16MultiArray(data=[action1, action2])
        self.thruster_action_pub.publish(thruster_command)

def main(args=None):
    rclpy.init(args=args)
    
    inference_node = InferenceNode()
    
    try:
        rclpy.spin(inference_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure policy is disabled when shutting down
        inference_node.toggle_policy(False)
        inference_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()