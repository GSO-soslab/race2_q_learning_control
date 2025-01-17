import os
import yaml
import torch
import numpy as np
import rospy
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
    def __init__(self, model_path, state_size, action_size):
        # Load the trained Q-network
        self.qnetwork = QNetwork(state_size, action_size, config['qnetwork']['hidden_layers'])
        try:
            self.qnetwork.load_state_dict(torch.load(model_path))
            self.qnetwork.eval()  # Set the network to evaluation mode (no training)
        except Exception as e:
            rospy.logerr(f"Failed to load model from {model_path}: {e}")
            raise

    def act(self, state):
        # Convert state to a tensor and get action values from the Q-network
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork(state)
        action_index = torch.argmax(action_values).item()
        rospy.loginfo(f"Predicted action values: {action_values}, Chosen action: {action_index}")
        return action_index

# Global variable to track if inference is enabled
inference_enabled = True

def toggle_inference_service(request):
    """Service callback to enable/disable inference."""
    global inference_enabled
    inference_enabled = request.data  # Enable inference if True, disable if False

    if inference_enabled:
        rospy.loginfo("Inference enabled.")
    else:
        rospy.loginfo("Inference disabled.")

    return SetBoolResponse(success=True, message="Inference state updated.")

def toggle_policy(enable_policy):
    """Calls the /toggle_policy service to enable or disable the policy."""
    rospy.wait_for_service('/toggle_policy')  # Wait for the service to be available
    try:
        # Create a proxy to the service '/toggle_policy' which uses SetBool
        toggle_policy_service = rospy.ServiceProxy('/toggle_policy', SetBool)

        # Call the service and pass the desired state (True to enable, False to disable)
        response = toggle_policy_service(enable_policy)

        # Print the response from the service
        if response.success:
            rospy.loginfo(f"Service call succeeded: {response.message}")
        else:
            rospy.logwarn(f"Service call failed: {response.message}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def run_inference(model_path):
    rospy.init_node('inference_node', anonymous=True)

    # Create the environment with the config
    env = GridWorldEnv(config)  # Pass config to GridWorldEnv
    state_size = env.state_size
    action_size = env.action_size

    # Create the inference agent
    agent = InferenceAgent(model_path, state_size, action_size)

    # Publisher for the thruster action
    thruster_action_pub = rospy.Publisher('/thruster_action', Int32MultiArray, queue_size=10)

    rate = rospy.Rate(5)  # Define a loop rate (e.g., 10 Hz)

    global inference_enabled
    episode = 0  # To keep track of the number of episodes
    while not rospy.is_shutdown():
        if inference_enabled:  # Check if inference is enabled
            episode += 1
            state = env.reset()
            done = False
            total_reward = 0

            rospy.loginfo(f"Starting Episode {episode}")
            
            # Enable the policy at the start of the episode
            toggle_policy(True)

            while not done and not rospy.is_shutdown() and inference_enabled:
                # Get action from the policy
                action_index = agent.act(state)
                # Take the action in the environment
                next_state, reward, done, _ = env.step(action_index)
                state = next_state

                total_reward += reward

                # Sleep to maintain the loop rate
                rate.sleep()

            # Disable the policy at the end of every episode
            toggle_policy(False)

            rospy.loginfo(f"Episode {episode} completed with total reward: {total_reward}")
        else:
            rospy.loginfo("Inference disabled, publishing thruster command [1, 1, 1, 1, 1, 1]")

            # Create the thruster command message
            thruster_command = Int32MultiArray(data=[1, 1, 1, 1, 1, 1])

            # Publish the thruster command
            thruster_action_pub.publish(thruster_command)

            # Sleep and wait for inference to be enabled
            rate.sleep()

if __name__ == '__main__':
    try:
        # Register the enable_inference service
        rospy.Service('/enable_inference', SetBool, toggle_inference_service)
        
        # Path to the saved model
        model_path = rospy.get_param('~model_path', 'dqn_model_2025-01-17_11-29-50.pth')

        # Run the inference indefinitely
        run_inference(model_path)
        
    except rospy.ROSInterruptException:
        pass