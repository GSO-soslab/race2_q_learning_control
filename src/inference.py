import os
import yaml
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from mvp_msgs.msg import ControlProcess
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Int32MultiArray
from collections import deque
from torch_dqn import GridWorldEnv
from torch_dqn import QNetwork
from std_srvs.srv import SetBool, SetBoolResponse  

# Load configuration from config.yaml
config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class InferenceAgent:
    def __init__(self, model_path, state_size, action_size, node):
        # Load the trained Q-network
        self.node = node
        self.qnetwork = QNetwork(state_size, action_size, config['qnetwork']['hidden_layers'])
        try:
            self.qnetwork.load_state_dict(torch.load(model_path))
            self.qnetwork.eval()  # Set the network to evaluation mode (no training)
        except Exception as e:
            self.node.get_logger().error(f"Failed to load model from {model_path}: {e}")
            raise

    def act(self, state):
        # Convert state to a tensor and get action values from the Q-network
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork(state)
        action_index = torch.argmax(action_values).item()
        self.node.get_logger().info(f"Predicted action values: {action_values}, Chosen action: {action_index}")
        return action_index

class InferenceNode(Node):
    def __init__(self, model_path):
        super().__init__('inference_node')
        
        # Create the environment with the config
        self.env = GridWorldEnv(config)
        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.inference_enabled = True

        # Create the inference agent
        self.agent = InferenceAgent(model_path, self.state_size, self.action_size, self)

        # Publisher for the thruster action
        self.thruster_action_pub = self.create_publisher(Int32MultiArray, '/thruster_action', 10)

        # Service to enable/disable inference
        self.create_service(SetBool, '/enable_inference', self.toggle_inference_service)

        # Timer for the inference loop
        self.timer = self.create_timer(0.2, self.run_inference)  # 5 Hz loop rate
        self.episode = 0

    def toggle_inference_service(self, request, response):
        """Service callback to enable/disable inference."""
        self.inference_enabled = request.data  # Enable inference if True, disable if False

        if self.inference_enabled:
            self.get_logger().info("Inference enabled.")
        else:
            self.get_logger().info("Inference disabled.")

        response.success = True
        response.message = "Inference state updated."
        return response

    def toggle_policy(self, enable_policy):
        """Calls the /toggle_policy service to enable or disable the policy."""
        client = self.create_client(SetBool, '/toggle_policy')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for /toggle_policy service...")
        
        request = SetBool.Request()
        request.data = enable_policy
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            if future.result().success:
                self.get_logger().info(f"Service call succeeded: {future.result().message}")
            else:
                self.get_logger().warn(f"Service call failed: {future.result().message}")
        else:
            self.get_logger().error("Service call failed.")

    def run_inference(self):
        if self.inference_enabled:  # Check if inference is enabled
            self.episode += 1
            state = self.env.reset()
            done = False
            total_reward = 0

            self.get_logger().info(f"Starting Episode {self.episode}")
            
            # Enable the policy at the start of the episode
            self.toggle_policy(True)

            while not done and rclpy.ok() and self.inference_enabled:
                # Get action from the policy
                action_index = self.agent.act(state)
                # Take the action in the environment
                next_state, reward, done, _ = self.env.step(action_index)
                state = next_state
                total_reward += reward

            # Disable the policy at the end of every episode
            self.toggle_policy(False)
            self.get_logger().info(f"Episode {self.episode} completed with total reward: {total_reward}")
        else:
            self.get_logger().info("Inference disabled, publishing thruster command [1, 1, 1, 1, 1, 1]")

            # Create the thruster command message
            thruster_command = Int32MultiArray()
            thruster_command.data = [1, 1, 1, 1, 1, 1]

            # Publish the thruster command
            self.thruster_action_pub.publish(thruster_command)


if __name__ == '__main__':
    rclpy.init()
    
    try:
        # Get model path parameter
        model_path = 'dqn_model_2025-01-17_11-29-50.pth'
        
        # Create and spin the inference node
        node = InferenceNode(model_path)
        rclpy.spin(node)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        node.destroy_node()
        rclpy.shutdown()
