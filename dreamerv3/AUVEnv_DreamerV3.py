#!/usr/bin/env python3

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os, time, datetime
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64, Float32, Header
from geometry_msgs.msg import TwistStamped, Vector3
from mvp_msgs.msg import ControlProcess
from sensor_msgs.msg import Imu
import yaml
from rclpy.clock import Clock
import threading
import random
from collections import deque
import h5py
import pickle
from rclpy.serialization import deserialize_message
import struct  # For manual parsing fallback

# Import the coupling reward calculator
from coupling_rewards_dreamerv3 import CouplingAwareRewardCalculator, DreamerV3RewardAdapter

class SetpointManager:
    """Manages dynamic setpoint generation and updates for DreamerV3 training"""
    
    def __init__(self, config):
        self.config = config
        self.setpoint_config = config.get('setpoint', {})
        
        # Setpoint parameters
        self.rate_hz = self.setpoint_config.get('rate_hz', 5.0)
        self.random_duration = self.setpoint_config.get('random_duration', 200.0)
        self.pos_z_range = self.setpoint_config.get('pos_z_range', [1.0, 8.0])
        self.ori_z_range = self.setpoint_config.get('ori_z_range', [-2.14, 2.14])
        self.ori_y_range = self.setpoint_config.get('ori_y_range', [-0.1, 0.1])
        self.vel_x_range = self.setpoint_config.get('vel_x_range', [-0.6, 0.6])
        
        # Current setpoint state
        self.current_setpoint = self.generate_random_setpoint()
        self.last_update_time = time.time()
        self.lock = threading.Lock()
        
    def generate_random_setpoint(self):
        """Generate a random setpoint within configured ranges"""
        setpoint = {
            'position': {
                'x': 0.0,
                'y': 0.0, 
                'z': random.uniform(self.pos_z_range[0], self.pos_z_range[1])
            },
            'orientation': {
                'x': 3.14,  # Fixed roll
                'y': random.uniform(self.ori_y_range[0], self.ori_y_range[1]),
                'z': random.uniform(self.ori_z_range[0], self.ori_z_range[1])
            },
            'velocity': {
                'x': random.uniform(self.vel_x_range[0], self.vel_x_range[1]),
                'y': 0.0,
                'z': 0.0
            },
            'angular_rate': {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0
            }
        }
        return setpoint
    
    def update_if_needed(self):
        """Update setpoint if duration has elapsed"""
        with self.lock:
            current_time = time.time()
            if current_time - self.last_update_time >= self.random_duration:
                self.current_setpoint = self.generate_random_setpoint()
                self.last_update_time = current_time
                return True
        return False
    
    def get_current_setpoint(self):
        """Get current setpoint thread-safely"""
        with self.lock:
            return self.current_setpoint.copy()
    
    def set_setpoint(self, setpoint):
        """Manually set a specific setpoint"""
        with self.lock:
            self.current_setpoint = setpoint.copy()
            self.last_update_time = time.time()

class ROSBagDataLoader:
    """Enhanced data loader with MCAP support for DreamerV3 training"""
    
    def __init__(self, config):
        self.config = config
        self.offline_config = config.get('offline_training', {})
        self.data_paths = self.offline_config.get('rosbag_paths', [])
        self.batch_size = self.offline_config.get('batch_size', 32)
        self.sequence_length = self.offline_config.get('sequence_length', 50)
        
        # Data storage
        self.episodes = []
        self.current_episode_idx = 0
        self.current_step_idx = 0
        
        # MCAP specific settings
        self.target_topics = [
            '/race2_auv/controller/process/error',
            '/race2_auv/controller/process/value', 
            '/race2_auv/control/thruster/heave_bow',
            '/race2_auv/control/thruster/heave_stern',
            '/race2_auv/control/thruster/surge_port',
            '/race2_auv/control/thruster/surge_starboard'
        ]
        
        if self.data_paths:
            self.load_data()
    
    def load_data(self):
        """Load data from various formats including MCAP"""
        print(f"Loading offline data from {len(self.data_paths)} files...")
        
        for data_path in self.data_paths:
            print(f"Processing: {data_path}")
            
            if not os.path.exists(data_path):
                print(f"File not found: {data_path}")
                continue
                
            if data_path.endswith('.mcap'):
                self.load_mcap_data(data_path)
            elif data_path.endswith('.h5'):
                self.load_hdf5_data(data_path)
            elif data_path.endswith('.pkl'):
                self.load_pickle_data(data_path)
            elif data_path.endswith('.bag') or data_path.endswith('.db3'):
                self.load_rosbag2_data(data_path)
            else:
                print(f"Unsupported file format: {data_path}")
        
        print(f"Loaded {len(self.episodes)} episodes for offline training")
    
    def load_mcap_data(self, mcap_path):
        """Load data from MCAP files"""
        try:
            # Method 1: Try with rosbag2_py
            episodes = self._read_mcap_with_rosbag2(mcap_path)
            if episodes:
                self.episodes.extend(episodes)
                print(f"Loaded {len(episodes)} episodes from {mcap_path} using rosbag2_py")
                return
            
            # Method 2: Try with mcap library
            episodes = self._read_mcap_with_mcap_lib(mcap_path)
            if episodes:
                self.episodes.extend(episodes)
                print(f"Loaded {len(episodes)} episodes from {mcap_path} using mcap library")
                return
            
            print(f"No episodes extracted from {mcap_path}")
            
        except Exception as e:
            print(f"Error loading MCAP data from {mcap_path}: {e}")
    
    def _read_mcap_with_rosbag2(self, mcap_path):
        """Read MCAP using rosbag2_py"""
        try:
            from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
            
            # Configure storage options for MCAP
            storage_options = StorageOptions(uri=str(mcap_path), storage_id='mcap')
            converter_options = ConverterOptions('', '')
            
            reader = SequentialReader()
            reader.open(storage_options, converter_options)
            
            # Get topic metadata
            topic_types = reader.get_all_topics_and_types()
            print("Available topics in MCAP:")
            for topic_info in topic_types:
                print(f"  {topic_info.name}: {topic_info.type}")
            
            # Collect messages by topic
            messages_by_topic = {topic: [] for topic in self.target_topics}
            message_count = 0
            successful_parses = 0
            
            while reader.has_next():
                topic_name, data, timestamp = reader.read_next()
                message_count += 1
                
                if topic_name in self.target_topics:
                    # Deserialize message data
                    msg = self._deserialize_message(topic_name, data)
                    if msg:
                        messages_by_topic[topic_name].append({
                            'timestamp': timestamp,
                            'data': msg
                        })
                        successful_parses += 1
            
            reader.close()
            
            print(f"Read {message_count} total messages from MCAP")
            print(f"Successfully parsed {successful_parses} messages")
            for topic, msgs in messages_by_topic.items():
                print(f"  {topic}: {len(msgs)} messages")
            
            # Convert messages to episodes
            episodes = self._messages_to_episodes(messages_by_topic)
            return episodes
            
        except ImportError:
            print("rosbag2_py not available")
            return []
        except Exception as e:
            print(f"Error reading MCAP with rosbag2_py: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _read_mcap_with_mcap_lib(self, mcap_path):
        """Read MCAP using mcap library"""
        try:
            from mcap.reader import make_reader
            from mcap_ros2.reader import read_ros2_messages
            
            messages_by_topic = {topic: [] for topic in self.target_topics}
            message_count = 0
            
            with open(mcap_path, "rb") as f:
                reader = make_reader(f)
                
                for schema, channel, message, ros_msg in read_ros2_messages(reader):
                    message_count += 1
                    topic_name = channel.topic
                    
                    if topic_name in self.target_topics:
                        messages_by_topic[topic_name].append({
                            'timestamp': message.log_time,
                            'data': self._convert_ros_msg_to_dict(ros_msg)
                        })
            
            print(f"Read {message_count} total messages from MCAP using mcap library")
            for topic, msgs in messages_by_topic.items():
                print(f"  {topic}: {len(msgs)} messages")
            
            # Convert messages to episodes
            episodes = self._messages_to_episodes(messages_by_topic)
            return episodes
            
        except ImportError:
            print("mcap library not available. Install with: pip install mcap mcap-ros2-support")
            return []
        except Exception as e:
            print(f"Error reading MCAP with mcap library: {e}")
            return []
    
    def _deserialize_message(self, topic_name, data):
        """Deserialize ROS2 message data properly"""
        try:
            if '/controller/process/' in topic_name:
                # For ControlProcess messages
                from mvp_msgs.msg import ControlProcess
                from rclpy.serialization import deserialize_message
                
                msg = deserialize_message(data, ControlProcess)
                return {
                    'header': {
                        'stamp': {
                            'sec': msg.header.stamp.sec,
                            'nanosec': msg.header.stamp.nanosec
                        },
                        'frame_id': msg.header.frame_id
                    },
                    'control_mode': msg.control_mode,
                    'position': {
                        'x': msg.position.x,
                        'y': msg.position.y,
                        'z': msg.position.z
                    },
                    'orientation': {
                        'x': msg.orientation.x,
                        'y': msg.orientation.y,
                        'z': msg.orientation.z
                    },
                    'child_frame_id': msg.child_frame_id,
                    'velocity': {
                        'x': msg.velocity.x,
                        'y': msg.velocity.y,
                        'z': msg.velocity.z
                    },
                    'angular_rate': {
                        'x': msg.angular_rate.x,
                        'y': msg.angular_rate.y,
                        'z': msg.angular_rate.z
                    }
                }
                
            elif '/control/thruster/' in topic_name or '/control/surge_' in topic_name:
                # For Float64 thruster commands
                from std_msgs.msg import Float64
                from rclpy.serialization import deserialize_message
                
                msg = deserialize_message(data, Float64)
                return {
                    'data': msg.data
                }
                
        except Exception as e:
            print(f"Error deserializing message for {topic_name}: {e}")
            # Fallback to manual parsing if deserialization fails
            return self._manual_parse_message(topic_name, data)
    
    def _convert_ros_msg_to_dict(self, ros_msg):
        """Convert ROS message to dictionary (for mcap library method)"""
        try:
            # Check message type and extract data
            msg_type = type(ros_msg).__name__
            
            if msg_type == 'ControlProcess':
                return {
                    'position': {
                        'x': ros_msg.position.x,
                        'y': ros_msg.position.y,
                        'z': ros_msg.position.z
                    },
                    'orientation': {
                        'x': ros_msg.orientation.x,
                        'y': ros_msg.orientation.y,
                        'z': ros_msg.orientation.z
                    },
                    'velocity': {
                        'x': ros_msg.velocity.x,
                        'y': ros_msg.velocity.y,
                        'z': ros_msg.velocity.z
                    },
                    'angular_rate': {
                        'x': ros_msg.angular_rate.x,
                        'y': ros_msg.angular_rate.y,
                        'z': ros_msg.angular_rate.z
                    }
                }
            elif msg_type == 'Float64':
                return {
                    'data': ros_msg.data
                }
            else:
                return {}
                
        except Exception as e:
            print(f"Error converting ROS message to dict: {e}")
            return {}
    
    def _manual_parse_message(self, topic_name, data):
        """Manual parsing fallback for when deserialization fails"""
        try:
            import struct
            
            if '/controller/process/' in topic_name:
                # Manual parsing for ControlProcess - simplified fallback
                return {
                    'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'angular_rate': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                }
            elif '/control/thruster/' in topic_name:
                # Manual parsing for Float64
                if len(data) >= 8:
                    value = struct.unpack('<d', data[:8])[0]  # Little-endian double
                    return {'data': value}
                else:
                    return {'data': 0.0}
                    
        except Exception as e:
            print(f"Manual parsing also failed for {topic_name}: {e}")
            return None
    
    def _messages_to_episodes(self, messages_by_topic):
        """Optimized episode creation for large datasets"""
        episodes = []
        
        # Get error and value messages for episode structure
        error_messages = messages_by_topic.get('/race2_auv/controller/process/error', [])
        value_messages = messages_by_topic.get('/race2_auv/controller/process/value', [])
        
        print(f"Found {len(error_messages)} error messages, {len(value_messages)} value messages")
        
        # Use error messages as primary reference, fallback to value messages
        reference_messages = error_messages if error_messages else value_messages
        
        if not reference_messages:
            print("No controller messages found to create episodes")
            return episodes
        
        # OPTIMIZATION 1: Limit processing for very large datasets
        max_messages = 50000  # Process at most 50k messages to avoid memory issues
        if len(reference_messages) > max_messages:
            print(f"Large dataset detected ({len(reference_messages)} messages)")
            print(f"Processing first {max_messages} messages for efficiency")
            reference_messages = reference_messages[:max_messages]
        
        # OPTIMIZATION 2: Sort only the subset we're using
        print("Sorting messages by timestamp...")
        reference_messages.sort(key=lambda x: x['timestamp'])
        
        # OPTIMIZATION 3: Much more aggressive episode splitting
        min_episode_length = 50        # Increased minimum length
        max_episode_length = 200       # NEW: Maximum episode length
        episode_split_threshold = 1.0e9  # Reduced to 1 second (was 5 seconds)
        
        current_episode = []
        last_timestamp = None
        episode_count = 0
        max_episodes = 100  # NEW: Limit number of episodes
        
        print("Creating episodes...")
        
        for i, msg in enumerate(reference_messages):
            if episode_count >= max_episodes:
                print(f"Reached maximum episodes limit ({max_episodes})")
                break
                
            current_timestamp = msg['timestamp']
            
            # Check conditions for starting new episode
            should_split = False
            
            # Condition 1: Time gap too large
            if (last_timestamp is not None and 
                current_timestamp - last_timestamp > episode_split_threshold):
                should_split = True
            
            # Condition 2: Episode too long
            if len(current_episode) >= max_episode_length:
                should_split = True
            
            if should_split and len(current_episode) >= min_episode_length:
                # Create episode (simplified - no thruster matching for speed)
                episode_data = self._create_simple_episode(current_episode)
                if episode_data:
                    episodes.append(episode_data)
                    episode_count += 1
                    print(f"Created episode {episode_count} with {len(episode_data['observations'])} steps")
                
                # Start new episode
                current_episode = []
            
            current_episode.append(msg)
            last_timestamp = current_timestamp
            
            # Progress update
            if i % 10000 == 0:
                print(f"Processed {i}/{len(reference_messages)} messages, {episode_count} episodes created")
        
        # Process final episode
        if len(current_episode) >= min_episode_length and episode_count < max_episodes:
            episode_data = self._create_simple_episode(current_episode)
            if episode_data:
                episodes.append(episode_data)
                episode_count += 1
                print(f"Created final episode {episode_count} with {len(episode_data['observations'])} steps")
        
        print(f"Total episodes created: {len(episodes)}")
        return episodes

    def _create_simple_episode(self, messages):
        """Create episode with simplified action extraction (faster)"""
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': []
        }
        
        for i, msg in enumerate(messages):
            # Create observation from message data
            obs = self._create_observation_from_message(msg['data'])
            episode_data['observations'].append(obs)
            
            # SIMPLIFIED: Use dummy actions for speed (you can enhance later)
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            episode_data['actions'].append(action)
            
            # SIMPLIFIED: Use zero rewards for speed  
            reward = 0.0
            episode_data['rewards'].append(reward)
            
            # Mark last step as terminal
            terminal = (i == len(messages) - 1)
            episode_data['terminals'].append(terminal)
        
        # Convert lists to numpy arrays
        try:
            episode_data['observations'] = np.array(episode_data['observations'], dtype=np.float32)
            episode_data['actions'] = np.array(episode_data['actions'], dtype=np.float32)
            episode_data['rewards'] = np.array(episode_data['rewards'], dtype=np.float32)
            episode_data['terminals'] = np.array(episode_data['terminals'], dtype=bool)
            
            return episode_data
        except Exception as e:
            print(f"Error creating episode arrays: {e}")
            return None
    
    def _create_episode_from_messages(self, messages, thruster_messages):
        """Create episode data from message sequence"""
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': []
        }
        
        for i, msg in enumerate(messages):
            # Create observation from message data
            obs = self._create_observation_from_message(msg['data'])
            episode_data['observations'].append(obs)
            
            # Create action from thruster commands (time-matched)
            action = self._create_action_from_thrusters(msg['timestamp'], thruster_messages)
            episode_data['actions'].append(action)
            
            # Calculate reward (simplified - you can enhance this)
            reward = self._calculate_simple_reward(msg['data'])
            episode_data['rewards'].append(reward)
            
            # Mark last step as terminal
            terminal = (i == len(messages) - 1)
            episode_data['terminals'].append(terminal)
        
        # Convert lists to numpy arrays
        episode_data['observations'] = np.array(episode_data['observations'], dtype=np.float32)
        episode_data['actions'] = np.array(episode_data['actions'], dtype=np.float32)
        episode_data['rewards'] = np.array(episode_data['rewards'], dtype=np.float32)
        episode_data['terminals'] = np.array(episode_data['terminals'], dtype=bool)
        
        return episode_data
    
    def _create_action_from_thrusters(self, timestamp, thruster_messages):
        """Create action vector from thruster command messages"""
        # Initialize action with zeros
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Map thruster topics to action indices
        thruster_mapping = {
            '/race2_auv/control/thruster/heave_bow': 0,
            '/race2_auv/control/thruster/heave_stern': 1,
            '/race2_auv/control/thruster/surge_port': 2,
            '/race2_auv/control/thruster/surge_starboard': 3
        }
        
        # Find closest thruster commands in time
        time_tolerance = 1.0e8  # 100ms in nanoseconds
        
        for topic, index in thruster_mapping.items():
            if topic in thruster_messages:
                # Find closest message in time
                closest_msg = None
                min_time_diff = float('inf')
                
                for thruster_msg in thruster_messages[topic]:
                    time_diff = abs(thruster_msg['timestamp'] - timestamp)
                    if time_diff < min_time_diff and time_diff < time_tolerance:
                        min_time_diff = time_diff
                        closest_msg = thruster_msg
                
                if closest_msg and 'data' in closest_msg['data']:
                    action[index] = closest_msg['data']['data']
        
        return action
    
    def _calculate_simple_reward(self, msg_data):
        """Calculate simple reward from message data"""
        try:
            # Simple reward based on position and orientation errors
            pos_error = np.sqrt(
                msg_data['position']['x']**2 + 
                msg_data['position']['y']**2 + 
                msg_data['position']['z']**2
            )
            
            ori_error = np.sqrt(
                msg_data['orientation']['x']**2 + 
                msg_data['orientation']['y']**2 + 
                msg_data['orientation']['z']**2
            )
            
            # Simple negative error reward
            reward = -(pos_error + ori_error)
            return float(reward)
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0
    
    def _create_observation_from_message(self, msg_data):
        """Enhanced observation creation from message data"""
        try:
            # Extract values from parsed message
            pos_z = msg_data.get('position', {}).get('z', 0.0)
            vel_x = msg_data.get('velocity', {}).get('x', 0.0)
            vel_y = msg_data.get('velocity', {}).get('y', 0.0)
            vel_z = msg_data.get('velocity', {}).get('z', 0.0)
            
            ori_x = msg_data.get('orientation', {}).get('x', 0.0)
            ori_y = msg_data.get('orientation', {}).get('y', 0.0)
            ori_z = msg_data.get('orientation', {}).get('z', 0.0)
            
            ang_x = msg_data.get('angular_rate', {}).get('x', 0.0)
            ang_y = msg_data.get('angular_rate', {}).get('y', 0.0)
            ang_z = msg_data.get('angular_rate', {}).get('z', 0.0)
            
            # Create observation vector matching your environment (17 elements)
            obs = np.array([
                pos_z,                    # [0] depth error/position
                vel_x,                    # [1] surge error/velocity
                vel_y,                    # [2] sway error/velocity
                np.sin(ori_x),           # [3] roll sin
                np.cos(ori_x),           # [4] roll cos
                np.sin(ori_y),           # [5] pitch sin
                np.cos(ori_y),           # [6] pitch cos
                np.sin(ori_z),           # [7] yaw sin
                np.cos(ori_z),           # [8] yaw cos
                vel_x,                    # [9] surge velocity
                vel_y,                    # [10] sway velocity
                vel_z,                    # [11] heave velocity
                ang_x,                    # [12] roll rate
                ang_y,                    # [13] pitch rate
                ang_z,                    # [14] yaw rate
                0.0,                      # [15] x acceleration (placeholder)
                0.0                       # [16] y acceleration (placeholder)
            ], dtype=np.float32)
            
            return obs
            
        except Exception as e:
            print(f"Error creating observation: {e}")
            # Fallback: create zero observation
            return np.zeros(17, dtype=np.float32)
    
    def load_rosbag2_data(self, bag_path):
        """Load data from ROS2 bag format"""
        # Similar to MCAP loading but for .bag/.db3 files
        print(f"ROS2 bag format not fully implemented for {bag_path}")
    
    def load_hdf5_data(self, file_path):
        """Load data from HDF5 format"""
        try:
            with h5py.File(file_path, 'r') as f:
                for episode_key in f.keys():
                    episode_data = {
                        'observations': f[episode_key]['observations'][:],
                        'actions': f[episode_key]['actions'][:],
                        'rewards': f[episode_key]['rewards'][:],
                        'terminals': f[episode_key]['terminals'][:]
                    }
                    self.episodes.append(episode_data)
        except Exception as e:
            print(f"Error loading HDF5 data from {file_path}: {e}")
    
    def load_pickle_data(self, file_path):
        """Load data from pickle format"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    self.episodes.extend(data)
                else:
                    self.episodes.append(data)
        except Exception as e:
            print(f"Error loading pickle data from {file_path}: {e}")
    
    def get_batch(self):
        """Get a batch of sequential data for DreamerV3"""
        if not self.episodes:
            return None
        
        batch = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': []
        }
        
        for _ in range(self.batch_size):
            episode = self.episodes[self.current_episode_idx]
            
            # Get sequence starting from current step
            start_idx = self.current_step_idx
            end_idx = min(start_idx + self.sequence_length, len(episode['observations']))
            
            # Pad sequence if needed
            seq_len = end_idx - start_idx
            if seq_len < self.sequence_length:
                # Pad with zeros
                obs_seq = np.pad(episode['observations'][start_idx:end_idx], 
                               ((0, self.sequence_length - seq_len), (0, 0)), mode='constant')
                act_seq = np.pad(episode['actions'][start_idx:end_idx],
                               ((0, self.sequence_length - seq_len), (0, 0)), mode='constant')
                rew_seq = np.pad(episode['rewards'][start_idx:end_idx],
                               ((0, self.sequence_length - seq_len),), mode='constant')
                term_seq = np.pad(episode['terminals'][start_idx:end_idx],
                                ((0, self.sequence_length - seq_len),), mode='constant')
            else:
                obs_seq = episode['observations'][start_idx:end_idx]
                act_seq = episode['actions'][start_idx:end_idx]
                rew_seq = episode['rewards'][start_idx:end_idx]
                term_seq = episode['terminals'][start_idx:end_idx]
            
            batch['observations'].append(obs_seq)
            batch['actions'].append(act_seq)
            batch['rewards'].append(rew_seq)
            batch['terminals'].append(term_seq)
            
            # Update indices
            self.current_step_idx += self.sequence_length
            if self.current_step_idx >= len(episode['observations']):
                self.current_episode_idx = (self.current_episode_idx + 1) % len(self.episodes)
                self.current_step_idx = 0
        
        # Convert to numpy arrays
        for key in batch:
            batch[key] = np.array(batch[key])
        
        return batch
        
class AUVEnvNode(Node):
    """Enhanced ROS2 node with lifecycle management and setpoint integration"""
    
    def __init__(self, config):
        super().__init__('auv_env_dreamerv3_node')
        
        self.config = config
        
        # Threading and callback groups
        self.sensor_callback_group = ReentrantCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Environment parameters
        self.thruster_size = self.config['environment']['thruster_size']
        servo_size = self.config['environment']['servo_joints_size']
        
        # QoS profiles for different message types
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        self.control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Initialize state variables with thread safety
        self._state_lock = threading.Lock()
        self._initialize_state_variables(servo_size)
        
        # Setpoint manager
        self.setpoint_manager = SetpointManager(config)
        
        # Create publishers and subscribers
        self._setup_ros_interfaces()
        
        # Communication flags
        self.new_state_available = False
        self.new_error_available = False
        self.last_action_timestamp = 0
        self.last_state_timestamp = 0
        self.last_error_timestamp = 0
        
        # Setpoint publishing
        self.setpoint_timer = self.create_timer(
            1.0 / self.setpoint_manager.rate_hz,
            self.publish_setpoint_callback,
            callback_group=self.control_callback_group
        )
        
        # Parameters
        self.declare_parameter('max_steps', 500)
        self.max_steps = self.get_parameter('max_steps').value
        
        self.get_logger().info("AUV DreamerV3 Environment Node initialized")
    
    def _initialize_state_variables(self, servo_size):
        """Initialize all state variables with proper dimensions"""
        # State variables
        self.position_state = np.zeros(3)
        self.orientation_state = np.zeros(3)
        self.v_state = np.zeros(3)
        self.omega_ref_state = np.zeros(3)
        
        # Error variables
        self.position_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)
        self.linear_acceleration = np.zeros(3)

        # Actuator variables
        self.joint_angles_port = 0.0
        self.joint_angles_starboard = 0.0
        self.joint_angles = np.zeros(servo_size)
        
        self.thrust_heave_bow = 0.0
        self.thrust_surge_port = 0.0
        self.thrust_surge_starboard = 0.0
        self.thrust_heave_stern = 0.0
    
    def _setup_ros_interfaces(self):
        """Setup all ROS2 publishers and subscribers"""
        # Publishers for control commands
        self.thruster_pubs = {
            'heave_bow': self.create_publisher(
                Float64, '/race2_auv/control/thruster/heave_bow', 
                self.control_qos, callback_group=self.control_callback_group),
            'heave_stern': self.create_publisher(
                Float64, '/race2_auv/control/thruster/heave_stern', 
                self.control_qos, callback_group=self.control_callback_group),
            'surge_port': self.create_publisher(
                Float64, '/race2_auv/control/thruster/surge_port', 
                self.control_qos, callback_group=self.control_callback_group),
            'surge_starboard': self.create_publisher(
                Float64, '/race2_auv/control/thruster/surge_starboard', 
                self.control_qos, callback_group=self.control_callback_group),
            'port_servo': self.create_publisher(
                Float64, '/race2_auv/control/surge_port_servo', 
                self.control_qos, callback_group=self.control_callback_group),
            'starboard_servo': self.create_publisher(
                Float64, '/race2_auv/control/surge_starboard_servo', 
                self.control_qos, callback_group=self.control_callback_group)
        }
        
        # Publisher for setpoints
        self.setpoint_pub = self.create_publisher(
            ControlProcess,
            '/race2_auv/controller/process/set_point',
            self.control_qos,
            callback_group=self.control_callback_group
        )
        
        # Subscribers for sensor data
        self.create_subscription(
            ControlProcess,  
            '/race2_auv/controller/process/value',
            self.state_callback,
            self.sensor_qos,
            callback_group=self.sensor_callback_group
        )
        
        self.create_subscription(
            ControlProcess, 
            '/race2_auv/controller/process/error', 
            self.error_callback,
            self.sensor_qos,
            callback_group=self.sensor_callback_group
        )
        
        self.create_subscription(
            Imu,
            '/race2_auv/imu/data',
            self.imu_callback,
            self.sensor_qos,
            callback_group=self.sensor_callback_group
        )
        
        # Subscribers for actuator feedback
        actuator_subs = [
            ('/race2_auv/control/surge_port_servo', self.update_joint_port),
            ('/race2_auv/control/surge_starboard_servo', self.update_joint_starboard),
            ('/race2_auv/control/thruster/heave_bow', self.update_thrust_heave_bow),
            ('/race2_auv/control/thruster/surge_port', self.update_thrust_surge_port),
            ('/race2_auv/control/thruster/surge_starboard', self.update_thrust_surge_starboard),
            ('/race2_auv/control/thruster/heave_stern', self.update_thrust_heave_stern)
        ]
        
        for topic, callback in actuator_subs:
            self.create_subscription(
                Float64, topic, callback, 
                self.sensor_qos, callback_group=self.sensor_callback_group
            )
    
    def publish_setpoint_callback(self):
        """Publish current setpoint and update if needed"""
        # Check if setpoint needs updating
        self.setpoint_manager.update_if_needed()
        
        # Get current setpoint
        setpoint_data = self.setpoint_manager.get_current_setpoint()
        
        # Create ROS message
        msg = ControlProcess()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "race2_auv/world_ned"
        msg.child_frame_id = "race2_auv/cg_link"
        msg.control_mode = "4dof"
        
        # Fill position, orientation, velocity, angular_rate
        msg.position = Vector3(
            x=setpoint_data['position']['x'],
            y=setpoint_data['position']['y'],
            z=setpoint_data['position']['z']
        )
        
        msg.orientation = Vector3(
            x=setpoint_data['orientation']['x'],
            y=setpoint_data['orientation']['y'],
            z=setpoint_data['orientation']['z']
        )
        
        msg.velocity = Vector3(
            x=setpoint_data['velocity']['x'],
            y=setpoint_data['velocity']['y'],
            z=setpoint_data['velocity']['z']
        )
        
        msg.angular_rate = Vector3(
            x=setpoint_data['angular_rate']['x'],
            y=setpoint_data['angular_rate']['y'],
            z=setpoint_data['angular_rate']['z']
        )
        
        # Publish setpoint
        self.setpoint_pub.publish(msg)
        self.get_logger().debug("Published setpoint")
    
    # Keep all the existing callback methods (state_callback, error_callback, etc.)
    # but add thread safety
    
    def error_callback(self, data):
        """Process error updates with thread safety"""
        with self._state_lock:
            self.get_logger().debug("Error callback triggered!")
            self.last_error_timestamp = time.time()
            self.position_err = np.array([data.position.x, data.position.y, data.position.z])
            self.orientation_err = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
            self.v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
            self.omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
            
            # Convert orientation errors to sin/cos representation
            roll_sin_err = np.sin(self.orientation_err[0])
            roll_cos_err = np.cos(self.orientation_err[0])
            pitch_sin_err = np.sin(self.orientation_err[1])
            pitch_cos_err = np.cos(self.orientation_err[1])
            yaw_sin_err = np.sin(self.orientation_err[2])
            yaw_cos_err = np.cos(self.orientation_err[2])
            
            # Update the state error array
            self.state_err = np.concatenate([
                self.position_err[2:3],
                self.v_err[:2],
                np.array([roll_sin_err, roll_cos_err, pitch_sin_err, pitch_cos_err, yaw_sin_err, yaw_cos_err]),
                self.omega_ref_err[:3],
            ])
            
            if hasattr(self, 'last_action_timestamp') and self.last_error_timestamp > self.last_action_timestamp:
                self.new_error_available = True
    
    def state_callback(self, data):
        """Process state updates with thread safety"""
        with self._state_lock:
            self.get_logger().debug("State callback triggered!")
            self.last_state_timestamp = time.time()
            
            # Extract state values
            self.position_state = np.array([data.position.x, data.position.y, data.position.z])
            self.orientation_state = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
            self.v_state = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
            self.omega_ref_state = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
            
            # Convert orientation to sin/cos representation
            roll_sin = np.sin(self.orientation_state[0])
            roll_cos = np.cos(self.orientation_state[0])
            pitch_sin = np.sin(self.orientation_state[1])
            pitch_cos = np.cos(self.orientation_state[1])
            yaw_sin = np.sin(self.orientation_state[2])
            yaw_cos = np.cos(self.orientation_state[2])
            
            # Update current state for RL agent
            self.current_state = np.concatenate([
                self.position_state[2:3],
                self.v_state[:2],
                np.array([roll_sin, roll_cos, pitch_sin, pitch_cos, yaw_sin, yaw_cos]),
            ])
            
            if hasattr(self, 'last_action_timestamp') and self.last_state_timestamp > self.last_action_timestamp:
                self.new_state_available = True
    
    def imu_callback(self, data):
        """Process IMU data with thread safety"""
        with self._state_lock:
            self.get_logger().debug("IMU callback triggered!")
            self.linear_acceleration[0] = data.linear_acceleration.x
            self.linear_acceleration[1] = data.linear_acceleration.y
            self.linear_acceleration[2] = data.linear_acceleration.z

    # Keep all the actuator update methods with thread safety
    def update_joint_port(self, data):
        with self._state_lock:
            self.joint_angles_port = data.data
            self.update_joints()

    def update_joint_starboard(self, data):
        with self._state_lock:
            self.joint_angles_starboard = data.data
            self.update_joints()

    def update_joints(self):
        self.joint_angles = [self.joint_angles_port, self.joint_angles_starboard]

    def update_thrust_surge_port(self, data):
        with self._state_lock:
            self.thrust_surge_port = data.data

    def update_thrust_surge_starboard(self, data):
        with self._state_lock:
            self.thrust_surge_starboard = data.data

    def update_thrust_heave_bow(self, data):
        with self._state_lock:
            self.thrust_heave_bow = data.data

    def update_thrust_heave_stern(self, data):
        with self._state_lock:
            self.thrust_heave_stern = data.data
    
    def publish_action(self, action, num_thrusters, num_servos):
        """Publish actions to ROS2 topics with thread safety"""
        with self._state_lock:
            # Action mapping (keep existing logic)
            heave_bow = action[0]
            heave_stern = action[1]
            surge_command = action[2]
            yaw_command = action[3]
            
            # Convert to physical thrusters
            surge_port = max(-1.0, min(1.0, 0.8 * surge_command + 0.2 * yaw_command))
            surge_starboard = max(-1.0, min(1.0, 0.8 * surge_command - 0.2 * yaw_command))
        
            thruster_cmds = [heave_bow, heave_stern, surge_port, surge_starboard]
            
            # Handle servo commands
            servo_angles_normalized = action[4:] if len(action) > 4 else []
            servo_angles_rad = []
            for angle in servo_angles_normalized:
                servo_angles_rad.append(self.convert_servo_command_to_radians(angle))
            
            # Publish commands
            thruster_mapping = [
                ('heave_bow', thruster_cmds[0]),
                ('heave_stern', thruster_cmds[1]),
                ('surge_port', thruster_cmds[2]),
                ('surge_starboard', thruster_cmds[3])
            ]
            
            if len(servo_angles_rad) >= 2:
                thruster_mapping.extend([
                    ('port_servo', servo_angles_rad[0]),
                    ('starboard_servo', servo_angles_rad[1])
                ])
            
            # Publish commands
            for name, value in thruster_mapping:
                msg = Float64()
                msg.data = float(value)
                self.thruster_pubs[name].publish(msg)
                self.get_logger().debug(f"Published {value} to {name}")
            
            # Record action timestamp
            self.last_action_timestamp = time.time()
            
            # Reset flags
            self.new_state_available = False
            self.new_error_available = False
            
            return thruster_cmds, servo_angles_rad
    
    def convert_servo_command_to_radians(self, normalized_command):
        """Convert normalized servo command to radians"""
        normalized_command = np.clip(normalized_command, -1.0, 1.0)
        min_angle_rad = self.config['environment']['min_servo_angle_rad']
        max_angle_rad = self.config['environment']['max_servo_angle_rad']
        angle_rad = min_angle_rad + (normalized_command + 1.0) * (max_angle_rad - min_angle_rad) / 2.0
        return angle_rad

class AUVEnvDreamerV3(gym.Env):
    """
    DreamerV3-compatible AUV Environment with integrated setpoint publisher
    Supports both online and offline training modes
    """
    
    def __init__(self, config_path=None, offline_mode=False):
        super(AUVEnvDreamerV3, self).__init__()
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'config_dreamerv3.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.offline_mode = offline_mode
        
        # Initialize ROS2 if in online mode
        if not self.offline_mode:
            if not rclpy.ok():
                rclpy.init(args=None)
            
            # Create ROS2 node
            self.node = AUVEnvNode(self.config)
            
            # Setup executor for multi-threaded operation
            self.executor = MultiThreadedExecutor(num_threads=4)
            self.executor.add_node(self.node)
            
            # Start executor in separate thread
            self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.executor_thread.start()
        else:
            # Initialize offline data loader
            self.data_loader = ROSBagDataLoader(self.config)
            self.node = None
        
        # Set up action and observation spaces for DreamerV3
        self._setup_spaces()
        
        # Initialize episode-related variables
        self.episode_step = 0
        self.episode_reward = 0.0
        self.max_episode_steps = self.config['training']['max_t']
        
        # Initialize coupling-aware reward calculator with DreamerV3 adapter
        self.coupling_calculator = CouplingAwareRewardCalculator(self.config)
        self.reward_adapter = DreamerV3RewardAdapter(self.coupling_calculator, self.config)
        
        # Initialize history for smoothness calculations
        thruster_size = self.config['environment']['thruster_size']
        servo_size = self.config['environment']['servo_joints_size']
        
        self.joint_positions_history = np.zeros((10, servo_size))
        self.u_prev = np.zeros((100, thruster_size))
        self.thruster_command_action_prev = np.zeros((100, thruster_size))
        
        # Current action/state tracking
        self.thruster_action = np.zeros(thruster_size)
        self.joint_angles = np.zeros(servo_size)
        self.last_action = np.zeros(thruster_size + servo_size)
        
        # Offline mode variables
        if self.offline_mode:
            self.current_episode_data = None
            self.episode_step_offline = 0
            
        print(f"AUV DreamerV3 Environment initialized in {'offline' if offline_mode else 'online'} mode")
    
    def _setup_spaces(self):
        """Setup observation and action spaces for DreamerV3"""
        thruster_size = self.config['environment']['thruster_size']
        servo_size = self.config['environment']['servo_joints_size']
        
        # Action space: continuous control for thrusters and servos
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(thruster_size + servo_size,), 
            dtype=np.float32
        )
        
        # DreamerV3 expects Dict observation space for multi-modal inputs
        # Separate proprioceptive and image-like observations
        self.observation_space = spaces.Dict({
            # Proprioceptive observations (MLP encoder)
            'vector': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(17,),  # Updated size: 10 errors + 3 velocities + 3 angular rates + 1 acceleration
                dtype=np.float32
            ),
            # Placeholder for potential visual observations (CNN encoder)
            # Uncomment if you have sonar/camera data
            # 'image': spaces.Box(
            #     low=0, 
            #     high=255, 
            #     shape=(64, 64, 1),  # Sonar image
            #     dtype=np.uint8
            # )
        })
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        if seed is not None:
            np.random.seed(seed)
        
        self.episode_step = 0
        self.episode_reward = 0
        
        if self.offline_mode:
            return self._reset_offline()
        else:
            return self._reset_online()
    
    def _reset_offline(self):
        """Reset for offline training mode"""
        # Get new episode data from data loader
        if self.data_loader.episodes:
            episode_idx = np.random.randint(len(self.data_loader.episodes))
            self.current_episode_data = self.data_loader.episodes[episode_idx]
            self.episode_step_offline = 0
            
            # Return first observation
            obs_vector = self.current_episode_data['observations'][0]
            observation = {'vector': obs_vector.astype(np.float32)}
            
            return observation, {}
        else:
            # Fallback to dummy observation if no data available
            observation = {'vector': np.zeros(17, dtype=np.float32)}
            return observation, {}
    
    def _reset_online(self):
        """Reset for online training mode"""
        # Spin node to get fresh data
        self._spin_node(timeout_sec=0.51)
        
        # Initialize coupling calculator if needed
        if self.coupling_calculator is None:
            self.coupling_calculator = CouplingAwareRewardCalculator(self.config)
            self.reward_adapter = DreamerV3RewardAdapter(self.coupling_calculator, self.config)
        
        # Get initial observation from ROS
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.offline_mode:
            return self._step_offline(action)
        else:
            return self._step_online(action)
    
    def _step_offline(self, action):
        """Step for offline training mode"""
        if self.current_episode_data is None:
            # Return dummy data if no episode loaded
            observation = {'vector': np.zeros(17, dtype=np.float32)}
            return observation, 0.0, True, False, {}
        
        # Get data for current step
        self.episode_step_offline += 1
        
        if self.episode_step_offline >= len(self.current_episode_data['observations']):
            # Episode finished
            observation = {'vector': self.current_episode_data['observations'][-1].astype(np.float32)}
            reward = float(self.current_episode_data['rewards'][-1])
            terminated = bool(self.current_episode_data['terminals'][-1])
            truncated = True
            return observation, reward, terminated, truncated, {}
        
        # Get current step data
        obs_vector = self.current_episode_data['observations'][self.episode_step_offline]
        reward = float(self.current_episode_data['rewards'][self.episode_step_offline])
        terminated = bool(self.current_episode_data['terminals'][self.episode_step_offline])
        
        observation = {'vector': obs_vector.astype(np.float32)}
        truncated = self.episode_step_offline >= self.max_episode_steps
        
        self.episode_step += 1
        
        return observation, reward, terminated, truncated, {}
    
    def _step_online(self, action):
        """Step for online training mode"""
        # Store action for reward calculation
        self.last_action = action.copy()
        
        # Publish action to ROS
        thruster_cmds, servo_angles_rad = self.node.publish_action(
            action, 
            self.config['environment']['thruster_size'],
            self.config['environment']['servo_joints_size']
        )
        
        # Store for reward calculation
        self.thruster_action = thruster_cmds
        self.joint_angles = servo_angles_rad
        
        # Wait for new sensor data
        timeout_sec = 0.5
        start_time = time.time()
        
        while not (self.node.new_state_available and self.node.new_error_available):
            self._spin_node(timeout_sec=0.01)
            if time.time() - start_time > timeout_sec:
                print("Warning: Timeout waiting for state/error updates")
                break
        
        # Get updated observation
        observation = self._get_observation()
        
        # Calculate reward using coupling-aware methods
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = False  # Add your termination logic here
        truncated = self.episode_step >= self.max_episode_steps
        
        self.episode_step += 1
        self.episode_reward += reward
        
        return observation, reward, terminated, truncated, {}
    
    def _get_observation(self):
        """Get current observation from ROS node"""
        if self.node is None:
            return {'vector': np.zeros(17, dtype=np.float32)}
        
        with self.node._state_lock:
            # Extract error components
            depth_error = self.node.position_err[2:3]
            surge_error = self.node.v_err[0:1]
            sway_error = self.node.v_err[1:2]
            
            # Convert orientation errors to sin/cos
            roll_error = self.node.orientation_err[0]
            pitch_error = self.node.orientation_err[1]
            yaw_error = self.node.orientation_err[2]
            
            roll_sin_error = np.sin(roll_error)
            roll_cos_error = np.cos(roll_error)
            pitch_sin_error = np.sin(pitch_error)
            pitch_cos_error = np.cos(pitch_error)
            yaw_sin_error = np.sin(yaw_error)
            yaw_cos_error = np.cos(yaw_error)
            
            # Extract velocity and angular rate components
            surge = self.node.v_state[0:1]
            sway = self.node.v_state[1:2]
            heave = self.node.v_state[2:3]
            
            # Create observation vector
            obs_vector = np.concatenate([
                # Error components (9 elements)
                depth_error,                    # [0]
                surge_error,                    # [1] 
                sway_error,                     # [2]
                np.array([roll_sin_error]),     # [3]
                np.array([roll_cos_error]),     # [4]
                np.array([pitch_sin_error]),    # [5]
                np.array([pitch_cos_error]),    # [6]
                np.array([yaw_sin_error]),      # [7]
                np.array([yaw_cos_error]),      # [8]
                
                # Velocity components (3 elements)
                surge,                          # [9] - surge velocity
                sway,                           # [10] - sway velocity  
                heave,                          # [11] - heave velocity
                
                # Angular rate components (3 elements)
                self.node.omega_ref_state[0:1], # [12] - roll rate
                self.node.omega_ref_state[1:2], # [13] - pitch rate
                self.node.omega_ref_state[2:3], # [14] - yaw rate

                # Acceleration components (2 elements)
                self.node.linear_acceleration[0:1], # [15] - x acceleration
                self.node.linear_acceleration[1:2]  # [16] - y acceleration
            ])
        
        # DreamerV3 expects Dict observations
        observation = {
            'vector': obs_vector.astype(np.float32)
        }
        
        return observation
    
    def _calculate_reward(self):
        """Calculate coupling-aware reward using adapter"""
        if self.node is None:
            return 0.0
        
        with self.node._state_lock:
            # Create state error array
            depth_error = self.node.position_err[2]
            surge_error = self.node.v_err[0]
            sway_error = self.node.v_err[1]
            heave_error = self.node.v_err[2]
            
            # Convert orientation errors to sin/cos
            roll_error = self.node.orientation_err[0]
            pitch_error = self.node.orientation_err[1]
            yaw_error = self.node.orientation_err[2]
            
            roll_sin_error = np.sin(roll_error)
            roll_cos_error = np.cos(roll_error)
            pitch_sin_error = np.sin(pitch_error)
            pitch_cos_error = np.cos(pitch_error)
            yaw_sin_error = np.sin(yaw_error)
            yaw_cos_error = np.cos(yaw_error)
            
            state_error_array = np.array([
                depth_error,
                surge_error,
                sway_error,
                heave_error,
                roll_sin_error, roll_cos_error,
                pitch_sin_error, pitch_cos_error,
                yaw_sin_error, yaw_cos_error
            ])
        
        # Use reward adapter for DreamerV3 compatibility
        reward = self.reward_adapter.calculate_reward(
            state_error_array, 
            self.last_action,
            self.episode_step
        )
        
        return reward
    
    def _spin_node(self, timeout_sec=0.1):
        """Process ROS callbacks for a limited time"""
        if self.node is None:
            return
        
        # Node spinning is handled by the executor thread
        time.sleep(timeout_sec)
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'executor') and self.executor is not None:
            self.executor.shutdown()
        
        if hasattr(self, 'node') and self.node is not None:
            self.node.destroy_node()
        
        if not self.offline_mode and rclpy.ok():
            rclpy.shutdown()

def main(args=None):
    """Main function for testing the environment"""
    # Test online mode
    env_online = AUVEnvDreamerV3(offline_mode=False)
    
    # Test environment
    obs, info = env_online.reset()
    print(f"Initial observation shape: {obs['vector'].shape}")
    
    for i in range(10):
        action = env_online.action_space.sample()
        obs, reward, terminated, truncated, info = env_online.step(action)
        print(f"Step {i}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            obs, info = env_online.reset()
    
    env_online.close()

if __name__ == "__main__":
    main()