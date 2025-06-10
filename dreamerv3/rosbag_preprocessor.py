#!/usr/bin/env python3
"""
Fast PID data processor for DreamerV3 offline training
Optimized for continuous control data (not episodic)
"""

import os
import sys
import numpy as np
import h5py
import yaml
from pathlib import Path
import time
from collections import defaultdict
import argparse

# ROS2 imports
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    ROSBAG2_AVAILABLE = True
except ImportError:
    print("Warning: rosbag2_py not available")
    ROSBAG2_AVAILABLE = False

class FastPIDProcessor:
    """
    Fast processor for continuous PID control data
    Creates artificial episodes from continuous time series
    """
    
    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        # Processing parameters
        self.target_hz = 10.0  # Downsample to 10 Hz
        self.episode_length = 200  # 200 steps = 20 seconds per episode
        self.min_episode_length = 50  # Minimum viable episode
        
        # Topics we need (only the essential ones)
        self.required_topics = {
            'state': '/race2_auv/controller/process/value',
            'error': '/race2_auv/controller/process/error',
            'thrusters': [
                '/race2_auv/control/thruster/heave_bow',
                '/race2_auv/control/thruster/heave_stern', 
                '/race2_auv/control/thruster/surge_port',
                '/race2_auv/control/thruster/surge_starboard'
            ]
        }
        
        print("Fast PID Processor initialized")
        print(f"Target frequency: {self.target_hz} Hz")
        print(f"Episode length: {self.episode_length} steps ({self.episode_length/self.target_hz:.1f} seconds)")
    
    def process_bag_fast(self, bag_path, output_dir, max_episodes=None):
        """
        Fast processing of PID control bag
        
        Args:
            bag_path: Path to ROS bag directory
            output_dir: Output directory for processed data
            max_episodes: Maximum episodes to extract (None = all)
        """
        if not ROSBAG2_AVAILABLE:
            raise ImportError("rosbag2_py not available")
        
        start_time = time.time()
        
        # Setup paths
        bag_path = Path(bag_path).expanduser()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing bag: {bag_path}")
        print(f"Output directory: {output_path}")
        
        # Open bag
        storage_options = StorageOptions(uri=str(bag_path), storage_id='mcap')
        converter_options = ConverterOptions('', '')
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Get topic info
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        # Check required topics
        available_topics = set(type_map.keys())
        required_flat = [self.required_topics['state'], self.required_topics['error']] + self.required_topics['thrusters']
        missing_topics = set(required_flat) - available_topics
        
        if missing_topics:
            print(f"⚠️  Missing topics: {missing_topics}")
            print(f"✅ Available topics: {len(available_topics)}")
            print("Continuing with available data...")
        
        # Collect synchronized data
        print("📊 Collecting synchronized data...")
        sync_data = self._collect_synchronized_data(reader, type_map)
        
        reader.close()
        
        collect_time = time.time() - start_time
        print(f"✅ Data collection completed in {collect_time:.1f} seconds")
        print(f"📈 Collected {len(sync_data['timestamps'])} synchronized samples")
        
        if len(sync_data['timestamps']) < self.min_episode_length:
            print(f"❌ Insufficient data: {len(sync_data['timestamps'])} < {self.min_episode_length}")
            return
        
        # Convert to episodes
        print("🔄 Converting to episodes...")
        episodes = self._create_episodes_from_continuous_data(sync_data, max_episodes)
        
        # Save processed data
        print("💾 Saving processed data...")
        self._save_episodes(episodes, output_path)
        
        total_time = time.time() - start_time
        print(f"🎉 Processing completed in {total_time:.1f} seconds")
        print(f"📊 Created {len(episodes)} episodes")
        print(f"⚡ Processing rate: {len(sync_data['timestamps'])/total_time:.0f} messages/second")
    
    def _collect_synchronized_data(self, reader, type_map):
        """Collect and synchronize data from all required topics"""
        
        # Data buffers - USE TOPIC NAMES AS KEYS
        data_buffers = defaultdict(list)
        timestamps = defaultdict(list)
        
        message_count = 0
        last_progress = 0
        
        # Read all messages efficiently
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            # Only process required topics
            if topic not in [self.required_topics['state'], self.required_topics['error']] + self.required_topics['thrusters']:
                continue
            
            try:
                # Fast deserialization
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                
                # Store timestamp and data - USE TOPIC NAME AS KEY
                ts = timestamp / 1e9  # Convert to seconds
                timestamps[topic].append(ts)
                
                if topic == self.required_topics['state']:
                    # State data
                    data_buffers[topic].append([
                        msg.position.z, msg.velocity.x, msg.velocity.y, msg.velocity.z,
                        msg.orientation.x, msg.orientation.y, msg.orientation.z,
                        msg.angular_rate.x, msg.angular_rate.y, msg.angular_rate.z
                    ])
                
                elif topic == self.required_topics['error']:
                    # Error data
                    data_buffers[topic].append([
                        msg.position.z, msg.velocity.x, msg.velocity.y,
                        msg.orientation.x, msg.orientation.y, msg.orientation.z
                    ])
                
                elif topic in self.required_topics['thrusters']:
                    # Thruster data - USE TOPIC NAME AS KEY
                    data_buffers[topic].append(msg.data)
                
                message_count += 1
                
                # Progress feedback (every 100k messages)
                if message_count % 100000 == 0:
                    progress = message_count // 100000
                    if progress > last_progress:
                        print(f"    Processed {message_count} messages...")
                        last_progress = progress
                
            except Exception as e:
                # Skip problematic messages
                continue
        
        print(f"    Total messages processed: {message_count}")
        
        # Synchronize to common timeline
        return self._synchronize_data(data_buffers, timestamps)

    def _synchronize_data(self, data_buffers, timestamps):
        """Synchronize data to common timeline at target frequency"""
        
        # DEBUG: Check what data was actually collected
        print(f"\n=== DEBUG SYNCHRONIZATION ===")
        print(f"Data buffers keys: {list(data_buffers.keys())}")
        print(f"Timestamps keys: {list(timestamps.keys())}")
        
        for key in data_buffers.keys():
            if key in timestamps:
                print(f"{key}: {len(data_buffers[key])} data points, {len(timestamps[key])} timestamps")
                if len(data_buffers[key]) > 0:
                    print(f"  First data sample: {data_buffers[key][0]}")
            else:
                print(f"{key}: NO TIMESTAMPS!")
        
        # Find common time range
        all_timestamps = []
        for topic_times in timestamps.values():
            if topic_times:
                all_timestamps.extend(topic_times)
        
        if not all_timestamps:
            print("❌ NO TIMESTAMPS FOUND!")
            return {'timestamps': [], 'observations': [], 'actions': []}
        
        start_time = min(all_timestamps)
        end_time = max(all_timestamps)
        duration = end_time - start_time
        
        print(f"Time range: {duration:.1f} seconds ({start_time:.1f} to {end_time:.1f})")
        
        # Create uniform time grid
        dt = 1.0 / self.target_hz
        time_grid = np.arange(start_time, end_time, dt)
        
        print(f"Creating {len(time_grid)} synchronized samples at {self.target_hz} Hz")
        
        # Synchronize each data stream
        sync_data = {
            'timestamps': time_grid,
            'observations': [],
            'actions': []
        }
        
        # Convert data buffers to numpy arrays for faster interpolation
        for topic, data_list in data_buffers.items():
            if data_list and topic in timestamps:
                try:
                    data_buffers[topic] = np.array(data_list)
                    timestamps[topic] = np.array(timestamps[topic])
                    print(f"Converted {topic}: {data_buffers[topic].shape}")
                except Exception as e:
                    print(f"Error converting {topic}: {e}")
        
        # Test interpolation on first few points
        print(f"\n=== TESTING INTERPOLATION ===")
        successful_interpolations = 0
        
        # Interpolate to common timeline
        for i, t in enumerate(time_grid):
            obs, action = self._interpolate_at_time(t, data_buffers, timestamps)
            
            if obs is not None and action is not None:
                sync_data['observations'].append(obs)
                sync_data['actions'].append(action)
                successful_interpolations += 1
                
                # Debug first few interpolations
                if i < 5:
                    print(f"t={t:.3f}: obs={obs[:3]}, action={action}")
            else:
                # Debug failed interpolations
                if i < 5:
                    print(f"t={t:.3f}: FAILED interpolation (obs={obs}, action={action})")
        
        print(f"Successful interpolations: {successful_interpolations}/{len(time_grid)}")
        print(f"=== END DEBUG ===\n")
        
        return sync_data

    def _interpolate_at_time(self, target_time, data_buffers, timestamps):
        """Fast interpolation at specific time"""
        
        # Get state data using topic names
        state_topic = self.required_topics['state']  # '/race2_auv/controller/process/value'
        error_topic = self.required_topics['error']  # '/race2_auv/controller/process/error'
        
        state_data = self._interpolate_topic_data(state_topic, target_time, data_buffers, timestamps)
        error_data = self._interpolate_topic_data(error_topic, target_time, data_buffers, timestamps)
        
        # DEBUG: Print issues for first few calls
        debug_count = getattr(self, '_debug_count', 0)
        if debug_count < 5:
            print(f"  Interpolation {debug_count}: state_data={state_data}, error_data={error_data}")
            self._debug_count = debug_count + 1
        
        if state_data is None or error_data is None:
            return None, None
        
        # Build observation (simplified)
        obs = []
        
        try:
            # Error components (first 9 elements)
            obs.extend([error_data[0]])  # depth error
            obs.extend(error_data[1:3])  # surge, sway error
            
            # Orientation errors as sin/cos (6 elements)
            for angle in error_data[3:6]:  # roll, pitch, yaw errors
                obs.extend([np.sin(angle), np.cos(angle)])
            
            # State velocities (7 elements) 
            obs.extend(state_data[1:4])  # surge, sway, heave velocity
            obs.extend(state_data[7:10])  # angular rates
            obs.extend([0.0])  # dummy acceleration
            
            # Pad to 17 dimensions if needed
            while len(obs) < 17:
                obs.append(0.0)
            obs = obs[:17]  # Truncate if too long
            
        except Exception as e:
            if debug_count < 5:
                print(f"  Error building observation: {e}")
            return None, None
        
        # Get thruster actions using topic names
        action = []
        for thruster_topic in self.required_topics['thrusters']:
            thruster_data = self._interpolate_topic_data(thruster_topic, target_time, data_buffers, timestamps)
            action.append(thruster_data if thruster_data is not None else 0.0)
        
        return np.array(obs, dtype=np.float32), np.array(action, dtype=np.float32)
    
    def _interpolate_topic_data(self, topic_key, target_time, data_buffers, timestamps):
        """Fast linear interpolation for a specific topic"""
        
        if topic_key not in data_buffers or topic_key not in timestamps:
            return None
        
        data = data_buffers[topic_key]
        times = timestamps[topic_key]
        
        if len(data) == 0 or len(times) == 0:
            return None
        
        # Find nearest indices
        idx = np.searchsorted(times, target_time)
        
        if idx == 0:
            return data[0] if isinstance(data[0], (int, float)) else data[0].copy()
        elif idx >= len(data):
            return data[-1] if isinstance(data[-1], (int, float)) else data[-1].copy()
        else:
            # Linear interpolation
            t0, t1 = times[idx-1], times[idx]
            if t1 == t0:
                return data[idx] if isinstance(data[idx], (int, float)) else data[idx].copy()
            
            alpha = (target_time - t0) / (t1 - t0)
            
            if isinstance(data[idx], (int, float)):
                # Scalar interpolation
                return (1 - alpha) * data[idx-1] + alpha * data[idx]
            else:
                # Vector interpolation
                return (1 - alpha) * data[idx-1] + alpha * data[idx]
    
    def _create_episodes_from_continuous_data(self, sync_data, max_episodes=None):
        """Create artificial episodes from continuous time series"""
        
        observations = sync_data['observations']
        actions = sync_data['actions']
        timestamps = sync_data['timestamps']
        
        if len(observations) != len(actions):
            min_len = min(len(observations), len(actions))
            observations = observations[:min_len]
            actions = actions[:min_len]
        
        total_samples = len(observations)
        print(f"    Creating episodes from {total_samples} samples")
        
        episodes = []
        episode_count = 0
        
        # Create overlapping episodes
        start_idx = 0
        while start_idx + self.min_episode_length < total_samples:
            end_idx = min(start_idx + self.episode_length, total_samples)
            
            # Extract episode data
            episode_obs = np.array(observations[start_idx:end_idx])
            episode_actions = np.array(actions[start_idx:end_idx])
            
            # Calculate simple rewards (negative error magnitude)
            episode_rewards = []
            for obs in episode_obs:
                # Simple reward: negative of error magnitude
                error_magnitude = np.abs(obs[0]) + np.abs(obs[1]) + np.abs(obs[2])  # depth, surge, sway errors
                episode_rewards.append(-error_magnitude)
            
            episode_rewards = np.array(episode_rewards, dtype=np.float32)
            
            # Terminal flags
            terminals = np.zeros(len(episode_obs), dtype=bool)
            terminals[-1] = True  # Last step is terminal
            
            episode = {
                'observations': episode_obs,
                'actions': episode_actions,
                'rewards': episode_rewards,
                'terminals': terminals
            }
            
            episodes.append(episode)
            episode_count += 1
            
            # Move to next episode (with 50% overlap for more data)
            start_idx += self.episode_length // 2
            
            if max_episodes and episode_count >= max_episodes:
                break
        
        print(f"    Created {len(episodes)} episodes (average length: {np.mean([len(ep['observations']) for ep in episodes]):.1f})")
        
        return episodes
    
    def _save_episodes(self, episodes, output_path):
        """Save episodes to HDF5 format"""
        
        output_file = output_path / "processed_episodes.h5"
        
        with h5py.File(output_file, 'w') as f:
            for i, episode in enumerate(episodes):
                group = f.create_group(f'episode_{i:06d}')
                
                for key, data in episode.items():
                    group.create_dataset(key, data=data, compression='gzip')
        
        print(f"    Saved to: {output_file}")
        
        # Create statistics
        stats = {
            'num_episodes': len(episodes),
            'total_steps': sum(len(ep['observations']) for ep in episodes),
            'avg_episode_length': np.mean([len(ep['observations']) for ep in episodes]),
            'avg_reward': np.mean([np.mean(ep['rewards']) for ep in episodes]),
            'obs_shape': episodes[0]['observations'].shape[1:] if episodes else None,
            'action_shape': episodes[0]['actions'].shape[1:] if episodes else None
        }
        
        stats_file = output_path / "data_statistics.yaml"
        with open(stats_file, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        print(f"    Statistics saved to: {stats_file}")
        print(f"    📊 {stats['num_episodes']} episodes, {stats['total_steps']} total steps")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fast PID data processor for DreamerV3')
    
    parser.add_argument('--bag_path', type=str, required=True,
                       help='Path to ROS bag directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--config', type=str, default='config/config_dreamerv3.yaml',
                       help='Config file path')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='Maximum episodes to create (for testing)')
    parser.add_argument('--episode_length', type=int, default=200,
                       help='Steps per episode')
    parser.add_argument('--target_hz', type=float, default=10.0,
                       help='Target sampling frequency')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = FastPIDProcessor(args.config)
    
    # Override parameters if provided
    if args.episode_length:
        processor.episode_length = args.episode_length
    if args.target_hz:
        processor.target_hz = args.target_hz
    
    # Process bag
    try:
        processor.process_bag_fast(args.bag_path, args.output_dir, args.max_episodes)
        print("🎉 Fast processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()