# process_mcap_to_dreamerv3.py (v4 - using rosbag2_py)
import yaml
import numpy as np
import os
from pathlib import Path
import argparse

# --- The key import ---
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# Import the necessary components from your environment file
try:
    from AUVEnv import AUVEnv, CouplingAwareRewardCalculator
except ImportError:
    print("Error: Could not import AUVEnv. Make sure AUVEnv.py is in the same directory or in your PYTHONPATH.")
    exit(1)

def get_rosbag_options(path, storage_id='mcap'):
    """Helper function to create storage and converter options."""
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    return storage_options, converter_options

def process_mcap_files(mcap_path, output_dir, config_path):
    """
    Reads ROS2 MCAP bag files using the 'rosbag2_py' library, reconstructs episodes,
    and saves them in DreamerV3's .npz format.
    """
    print(f"Starting MCAP processing from: {mcap_path}")
    print(f"Outputting .npz files to: {output_dir}")

    # Load config for reward calculation
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize a dummy environment to reuse its reward logic.
    dummy_env = AUVEnv()
    dummy_env.coupling_calculator = CouplingAwareRewardCalculator(config)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- 1. Read all messages using rosbag2_py ---
    storage_options, converter_options = get_rosbag_options(mcap_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get the message type for each topic
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    messages = []
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        messages.append((topic, t, msg))
    
    del reader # Close the bag file

    print(f"Read {len(messages)} messages from the bag.")

    # --- 2. Reconstruct Episodes (Logic is the same) ---
    episodes = []
    current_episode = []
    for topic, timestamp, msg in messages:
        if topic == '/race2_auv/controller/process/set_point':
            if current_episode:
                episodes.append(current_episode)
            current_episode = []
        current_episode.append((topic, timestamp, msg))
    
    if current_episode:
        episodes.append(current_episode)
    
    if not episodes:
        print("  - Warning: No episodes found. Treating entire bag as one episode.")
        episodes.append(messages)

    # --- 3. Process each episode into NPZ format (Logic is the same) ---
    episode_count = 0
    for ep_messages in episodes:
        if len(ep_messages) < 10:
            continue

        ep_data = {'observation': [], 'action': [], 'reward': [], 'is_first': [], 'is_last': [], 'is_terminal': [], 'discount': []}
        state_updates = [(ts, msg) for topic, ts, msg in ep_messages if topic == '/race2_auv/controller/process/value']

        for i, (ts_state, msg_state) in enumerate(state_updates):
            def get_latest_before(topic, timestamp):
                candidates = [m for t, ts, m in ep_messages if t == topic and ts < timestamp]
                return candidates[-1] if candidates else None
            
            msg_error = get_latest_before('/race2_auv/controller/process/error', ts_state)
            msg_imu = get_latest_before('/race2_auv/imu/data', ts_state)
            
            actions = {
                'heave_bow': get_latest_before('/race2_auv/control/thruster/heave_bow', ts_state),
                'heave_stern': get_latest_before('/race2_auv/control/thruster/heave_stern', ts_state),
                'surge_port': get_latest_before('/race2_auv/control/thruster/surge_port', ts_state),
                'surge_starboard': get_latest_before('/race2_auv/control/thruster/surge_starboard', ts_state),
                'port_servo': get_latest_before('/race2_auv/control/surge_port_servo', ts_state),
                'starboard_servo': get_latest_before('/race2_auv/control/surge_starboard_servo', ts_state),
            }

            if not all([msg_error, msg_imu] + list(actions.values())):
                continue

            # Construct Observation
            obs_parts = [
                np.array([msg_error.position.z, msg_error.velocity.x, msg_error.velocity.y]),
                np.array([np.sin(msg_error.orientation.x), np.cos(msg_error.orientation.x)]),
                np.array([np.sin(msg_error.orientation.y), np.cos(msg_error.orientation.y)]),
                np.array([np.sin(msg_error.orientation.z), np.cos(msg_error.orientation.z)]),
                np.array([msg_state.velocity.x, msg_state.velocity.y, msg_state.velocity.z]),
                np.array([msg_state.angular_rate.x, msg_state.angular_rate.y, msg_state.angular_rate.z]),
                np.array([msg_imu.linear_acceleration.x, msg_imu.linear_acceleration.y])
            ]
            observation = np.concatenate(obs_parts).astype(np.float32)

            # Construct Action
            action = np.array([
                actions['heave_bow'].data, actions['heave_stern'].data,
                actions['surge_port'].data, actions['surge_starboard'].data,
                actions['port_servo'].data, actions['starboard_servo'].data
            ]).astype(np.float32)

            # Recalculate Reward
            state_error_array = np.concatenate([
                np.array([msg_error.position.z, msg_error.velocity.x, msg_error.velocity.y, msg_error.velocity.z]),
                np.array([np.sin(msg_error.orientation.x), np.cos(msg_error.orientation.x)]),
                np.array([np.sin(msg_error.orientation.y), np.cos(msg_error.orientation.y)]),
                np.array([np.sin(msg_error.orientation.z), np.cos(msg_error.orientation.z)]),
            ])
            
            dummy_env.thruster_action = np.array([action[0], action[1], action[2], action[3]])
            dummy_env.joint_angles = np.array([action[4], action[5]])
            dummy_env.node.thrust_heave_bow = action[0]
            dummy_env.node.thrust_heave_stern = action[1]
            dummy_env.node.thrust_surge_port = action[2]
            dummy_env.node.thrust_surge_starboard = action[3]
            reward = dummy_env.calculate_reward(state_error_array)
            
            ep_data['observation'].append(observation)
            ep_data['action'].append(action)
            ep_data['reward'].append(reward)

        if len(ep_data['observation']) > 1:
            num_steps = len(ep_data['observation'])
            ep_data['is_first'] = [True] + [False] * (num_steps - 1)
            ep_data['is_last'] = [False] * (num_steps - 1) + [True]
            ep_data['is_terminal'] = [False] * (num_steps - 1) + [True]
            ep_data['discount'] = [1.0] * num_steps

            for key, val in ep_data.items():
                dtype = np.float32 if key not in ['is_first', 'is_last', 'is_terminal'] else bool
                ep_data[key] = np.array(val, dtype=dtype)

            filename = Path(output_dir) / f'episode_{episode_count:06d}.npz'
            np.savez_compressed(filename, **ep_data)
            print(f"  -> Saved episode {episode_count} with {num_steps} steps to {filename}")
            episode_count += 1
    
    print(f"\nFinished processing. Total episodes created: {episode_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # The URI for rosbag2_py should be the path to the bag file *without the extension*
    parser.add_argument('--mcap_path', type=str, required=True, help='Path to the rosbag directory (e.g., my_bag/ not my_bag/my_bag_0.mcap)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the .npz dataset')
    parser.add_argument('--config', type=str, default='config/config_sac.yaml', help='Path to environment config for reward calculation')
    args = parser.parse_args()
    
    process_mcap_files(args.mcap_path, args.output_dir, args.config)