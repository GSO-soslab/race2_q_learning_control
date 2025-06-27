#rosbag_to_training.py

#!/usr/bin/env python3
"""
Direct MCAP/ROS2 bag to training CSV converter
Input: .mcap or .db3 ROS2 bag file
Output: Single CSV with timestamp, current_state, next_state, actions, reward
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from collections import defaultdict
import os
from coupling_rewards import CouplingAwareRewardCalculator

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    ROSBAG2_AVAILABLE = True
    print("✓ Using rosbag2_py (official ROS2 method)")
except ImportError:
    print("✗ ROS2 packages not found. Make sure ROS2 is sourced:")
    print("source /opt/ros/humble/setup.bash")
    ROSBAG2_AVAILABLE = False

class McapToTrainingCSV:
    def __init__(self, config_path=None):
        # Load config if available
        self.config = self._load_config(config_path)
        
        # Initialize coupling reward calculator
        self.coupling_calculator = CouplingAwareRewardCalculator(self.config) if self.config else None
        # Define topic mappings (update these to match your actual topics)
        self.topic_mappings = {
            '/race2_auv/controller/process/value': 'state',
            '/race2_auv/controller/process/error': 'error', 
            '/race2_auv/imu/data': 'imu',
            '/race2_auv/control/thruster/heave_bow': 'thrust_heave_bow',
            '/race2_auv/control/thruster/heave_stern': 'thrust_heave_stern',
            '/race2_auv/control/thruster/surge_port': 'thrust_surge_port',
            '/race2_auv/control/thruster/surge_starboard': 'thrust_surge_starboard',
            '/race2_auv/control/surge_port_servo': 'servo_port',
            '/race2_auv/control/surge_starboard_servo': 'servo_starboard'
        }
        
        # State observation components (24 elements)
        self.obs_columns = [
            'depth', 'depth_error', 'surge_vel_error', 'sway_vel_error',
            'roll_sin_err', 'roll_cos_err', 'pitch_sin_err', 'pitch_cos_err',
            'yaw_sin_err', 'yaw_cos_err', 'roll_sin_curr', 'roll_cos_curr',
            'pitch_sin_curr', 'pitch_cos_curr', 'yaw_sin_curr', 'yaw_cos_curr',
            'surge_vel', 'sway_vel', 'heave_vel', 'roll_rate', 'pitch_rate',
            'yaw_rate', 'x_accel', 'y_accel'
        ]
        
        # Action components (6 elements)
        self.action_columns = [
            'thrust_heave_bow', 'thrust_heave_stern',
            'thrust_surge_port', 'thrust_surge_starboard', 
            'servo_port', 'servo_starboard'
        ]
    
    def _load_config(self, config_path):
        """Load configuration file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return None
    
    def convert_bag_to_training_csv(self, bag_path, output_csv):
        """
        Main function: MCAP/ROS2 bag → Training CSV
        
        Args:
            bag_path: Path to .mcap or .db3 bag file
            output_csv: Output CSV file path
        """
        print(f"🎯 Converting {bag_path} → {output_csv}")
        
        # Step 1: Extract raw data from bag
        raw_data = self._extract_bag_data(bag_path)
        
        # Step 2: Synchronize topics to common timeline  
        sync_data = self._synchronize_topics(raw_data)
        
        # Step 3: Create training dataset
        training_df = self._create_training_dataset(sync_data)
        
        # Step 4: Save to CSV
        training_df.to_csv(output_csv, index=False)
        
        print(f"✅ Training CSV created: {output_csv}")
        print(f"   Rows: {len(training_df):,}")
        print(f"   Columns: {len(training_df.columns)}")
        if len(training_df) > 0:
            duration = training_df['timestamp'].max() - training_df['timestamp'].min()
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Sample rate: {len(training_df)/duration:.1f} Hz")
        
        return training_df
    
    def _extract_bag_data(self, bag_path):
        """Extract all relevant topic data from ROS2 bag using rosbag2_py"""
        print("📂 Extracting data from ROS2 bag...")
        
        if not ROSBAG2_AVAILABLE:
            raise ImportError("rosbag2_py not available. Source your ROS2 environment first.")
        
        raw_data = defaultdict(list)
        message_counts = defaultdict(int)
        
        # Try file path first, then directory (same as your old script)
        uri_options = [str(bag_path), str(Path(bag_path).parent)]
        
        for uri in uri_options:
            try:
                print(f"Trying URI: {uri}")
                
                storage_options = rosbag2_py.StorageOptions(
                    uri=uri,
                    storage_id='mcap'
                )
                converter_options = rosbag2_py.ConverterOptions(
                    input_serialization_format='cdr',
                    output_serialization_format='cdr'
                )
                
                reader = rosbag2_py.SequentialReader()
                reader.open(storage_options, converter_options)
                
                # Get available topics
                topic_types = reader.get_all_topics_and_types()
                type_map = {topic_metadata.name: topic_metadata.type for topic_metadata in topic_types}
                
                print(f"✓ Found {len(topic_types)} topics:")
                for topic_metadata in topic_types:
                    alias = self.topic_mappings.get(topic_metadata.name)
                    status = "✓" if alias else "○"
                    print(f"  {status} {topic_metadata.name} → {alias or 'unused'}")
                
                # Read messages
                while reader.has_next():
                    (topic, data, timestamp) = reader.read_next()
                    
                    if topic in self.topic_mappings:
                        try:
                            # Deserialize message using rosbag2_py method
                            msg_type = get_message(type_map[topic])
                            msg = deserialize_message(data, msg_type)
                            
                            # Convert timestamp (nanoseconds to seconds)
                            time_sec = timestamp / 1e9
                            
                            # Extract data based on message type
                            data_point = self._extract_message_data_rosbag2(msg, type_map[topic])
                            data_point['timestamp'] = time_sec
                            
                            alias = self.topic_mappings[topic]
                            raw_data[alias].append(data_point)
                            message_counts[alias] += 1
                            
                        except Exception as e:
                            if message_counts[alias] < 5:  # Only show first few errors
                                print(f"⚠️  Error processing {topic}: {e}")
                
                reader.close()
                
                # Summary
                print(f"\n📊 Extracted data summary:")
                for alias, count in message_counts.items():
                    print(f"   {alias}: {count:,} messages")
                
                return dict(raw_data)
                
            except Exception as e:
                print(f"Failed with URI {uri}: {e}")
                continue
        
        raise Exception("All bag reading attempts failed")
    
    def _extract_message_data_rosbag2(self, msg, msg_type):
        """Extract data from ROS message using rosbag2_py deserializer"""
        data = {}
        
        if 'ControlProcess' in msg_type:
            # Extract ControlProcess message (state/error)
            data.update({
                'position_x': msg.position.x,
                'position_y': msg.position.y, 
                'position_z': msg.position.z,
                'orientation_x': msg.orientation.x,
                'orientation_y': msg.orientation.y,
                'orientation_z': msg.orientation.z,
                'velocity_x': msg.velocity.x,
                'velocity_y': msg.velocity.y,
                'velocity_z': msg.velocity.z,
                'angular_rate_x': msg.angular_rate.x,
                'angular_rate_y': msg.angular_rate.y,
                'angular_rate_z': msg.angular_rate.z
            })
            
        elif 'Imu' in msg_type:
            # Extract IMU message
            data.update({
                'linear_accel_x': msg.linear_acceleration.x,
                'linear_accel_y': msg.linear_acceleration.y,
                'linear_accel_z': msg.linear_acceleration.z,
                'angular_vel_x': msg.angular_velocity.x,
                'angular_vel_y': msg.angular_velocity.y,
                'angular_vel_z': msg.angular_velocity.z
            })
            
        elif 'Float64' in msg_type:
            # Extract Float64 message (thrusters/servos)
            data['value'] = msg.data
            
        return data
    
    def _extract_message_data(self, msg, msg_type):
        """Extract data from ROS message based on type"""
        data = {}
        
        if 'ControlProcess' in msg_type:
            # Extract ControlProcess message (state/error)
            data.update({
                'position_x': msg.position.x,
                'position_y': msg.position.y, 
                'position_z': msg.position.z,
                'orientation_x': msg.orientation.x,
                'orientation_y': msg.orientation.y,
                'orientation_z': msg.orientation.z,
                'velocity_x': msg.velocity.x,
                'velocity_y': msg.velocity.y,
                'velocity_z': msg.velocity.z,
                'angular_rate_x': msg.angular_rate.x,
                'angular_rate_y': msg.angular_rate.y,
                'angular_rate_z': msg.angular_rate.z
            })
            
        elif 'Imu' in msg_type:
            # Extract IMU message
            data.update({
                'linear_accel_x': msg.linear_acceleration.x,
                'linear_accel_y': msg.linear_acceleration.y,
                'linear_accel_z': msg.linear_acceleration.z,
                'angular_vel_x': msg.angular_velocity.x,
                'angular_vel_y': msg.angular_velocity.y,
                'angular_vel_z': msg.angular_velocity.z
            })
            
        elif 'Float64' in msg_type:
            # Extract Float64 message (thrusters/servos)
            data['value'] = msg.data
            
        return data
    
    def _synchronize_topics(self, raw_data):
        """Synchronize all topics to common 10Hz timeline"""
        print("🔄 Synchronizing topics...")
        
        if 'state' not in raw_data:
            raise ValueError("No state data found - cannot synchronize")
        
        # Convert to DataFrames and sort by timestamp
        dataframes = {}
        for topic, messages in raw_data.items():
            if messages:
                df = pd.DataFrame(messages)
                df = df.sort_values('timestamp').reset_index(drop=True)
                dataframes[topic] = df
                print(f"   {topic}: {len(df)} samples")
        
        # Use state as reference (should be 10Hz)
        state_df = dataframes['state']
        print(f"Using state as reference: {len(state_df)} timesteps")
        
        # Synchronize other topics to state timestamps
        synchronized_data = []
        tolerance = 0.05  # 50ms tolerance
        
        for i, state_row in state_df.iterrows():
            target_time = state_row['timestamp']
            sync_point = {'timestamp': target_time, 'state': state_row.to_dict()}
            
            # Find matching data in other topics
            for topic, df in dataframes.items():
                if topic == 'state':
                    continue
                    
                # Find closest timestamp
                time_diffs = np.abs(df['timestamp'] - target_time)
                if len(time_diffs) > 0:
                    closest_idx = time_diffs.idxmin()
                    if time_diffs.iloc[closest_idx] <= tolerance:
                        sync_point[topic] = df.iloc[closest_idx].to_dict()
            
            synchronized_data.append(sync_point)
            
            if i % 1000 == 0:
                print(f"   Synchronized {i}/{len(state_df)} timesteps")
        
        print(f"✅ Synchronized to {len(synchronized_data)} timesteps")
        return synchronized_data
    
    def _create_training_dataset(self, sync_data):
        """Create training dataset with current_state, next_state, actions, rewards"""
        print("🏗️  Building training dataset...")
        
        training_rows = []
        
        # Process each timestep (except last, since we need next_state)
        for i in range(len(sync_data) - 1):
            current_point = sync_data[i]
            next_point = sync_data[i + 1]
            
            # Extract current and next state observations
            try:
                current_obs = self._extract_observation(current_point)
                next_obs = self._extract_observation(next_point)
                actions = self._extract_actions(current_point)
                # Calculate reward using coupling calculator
                reward = self._calculate_reward(current_point, episode_step=i)
                
                # Build training row
                row = {'timestamp': current_point['timestamp'], 'reward': reward}
                
                # Add current state
                for j, col in enumerate(self.obs_columns):
                    row[f'curr_{col}'] = current_obs[j] if j < len(current_obs) else 0.0
                
                # Add next state  
                for j, col in enumerate(self.obs_columns):
                    row[f'next_{col}'] = next_obs[j] if j < len(next_obs) else 0.0
                
                # Add actions
                for j, col in enumerate(self.action_columns):
                    row[f'action_{col}'] = actions[j] if j < len(actions) else 0.0
                
                training_rows.append(row)
                
            except Exception as e:
                print(f"⚠️  Error at timestep {i}: {e}")
                continue
            
            if i % 1000 == 0:
                print(f"   Processed {i}/{len(sync_data)-1} timesteps")
        
        df = pd.DataFrame(training_rows)
        print(f"✅ Created training dataset: {len(df)} samples")
        return df
    
    def _extract_observation(self, data_point):
        """Extract 24-element observation vector"""
        # Initialize with zeros
        obs = np.zeros(24)
        
        # Extract state data
        state = data_point.get('state', {})
        error = data_point.get('error', {})
        imu = data_point.get('imu', {})
        
        try:
            # Build observation vector (same structure as your original code)
            obs[0] = state.get('position_z', 0.0)  # depth
            obs[1] = error.get('position_z', 0.0)  # depth_error
            obs[2] = error.get('velocity_x', 0.0)  # surge_vel_error
            obs[3] = error.get('velocity_y', 0.0)  # sway_vel_error
            
            # Orientation errors (sin/cos representation)
            roll_err = error.get('orientation_x', 0.0)
            pitch_err = error.get('orientation_y', 0.0)
            yaw_err = error.get('orientation_z', 0.0)
            
            obs[4] = np.sin(roll_err)   # roll_sin_err
            obs[5] = np.cos(roll_err)   # roll_cos_err
            obs[6] = np.sin(pitch_err)  # pitch_sin_err
            obs[7] = np.cos(pitch_err)  # pitch_cos_err
            obs[8] = np.sin(yaw_err)    # yaw_sin_err
            obs[9] = np.cos(yaw_err)    # yaw_cos_err
            
            # Current orientation (sin/cos representation)
            roll_curr = state.get('orientation_x', 0.0)
            pitch_curr = state.get('orientation_y', 0.0)
            yaw_curr = state.get('orientation_z', 0.0)
            
            obs[10] = np.sin(roll_curr)   # roll_sin_curr
            obs[11] = np.cos(roll_curr)   # roll_cos_curr
            obs[12] = np.sin(pitch_curr)  # pitch_sin_curr
            obs[13] = np.cos(pitch_curr)  # pitch_cos_curr
            obs[14] = np.sin(yaw_curr)    # yaw_sin_curr
            obs[15] = np.cos(yaw_curr)    # yaw_cos_curr
            
            # Velocities and rates
            obs[16] = state.get('velocity_x', 0.0)      # surge_vel
            obs[17] = state.get('velocity_y', 0.0)      # sway_vel
            obs[18] = state.get('velocity_z', 0.0)      # heave_vel
            obs[19] = state.get('angular_rate_x', 0.0)  # roll_rate
            obs[20] = state.get('angular_rate_y', 0.0)  # pitch_rate
            obs[21] = state.get('angular_rate_z', 0.0)  # yaw_rate
            
            # Accelerations
            obs[22] = imu.get('linear_accel_x', 0.0)    # x_accel
            obs[23] = imu.get('linear_accel_y', 0.0)    # y_accel
            
        except Exception as e:
            print(f"Warning: Error extracting observation: {e}")
        
        return obs
    
    def _extract_actions(self, data_point):
        """Extract 6-element action vector"""
        actions = np.zeros(6)
        
        # Thruster values
        thrusters = ['thrust_heave_bow', 'thrust_heave_stern', 
                    'thrust_surge_port', 'thrust_surge_starboard']
        for i, thruster in enumerate(thrusters):
            if thruster in data_point:
                actions[i] = data_point[thruster].get('value', 0.0)
        
        # Servo values
        if 'servo_port' in data_point:
            actions[4] = data_point['servo_port'].get('value', 0.0)
        if 'servo_starboard' in data_point:
            actions[5] = data_point['servo_starboard'].get('value', 0.0)
        
        return actions
    
    def _calculate_reward(self, data_point, episode_step=0):
        """Calculate reward using coupling-aware reward calculator"""
        if 'error' not in data_point:
            return 0.0
        
        try:
            # Extract state error array (7 elements: depth, surge, sway, heave, roll, pitch, yaw)
            error = data_point['error']
            
            depth_error = error.get('position_z', 0.0)
            surge_error = error.get('velocity_x', 0.0) 
            sway_error = error.get('velocity_y', 0.0)
            heave_error = error.get('velocity_z', 0.0)
            roll_error = error.get('orientation_x', 0.0)
            pitch_error = error.get('orientation_y', 0.0)
            yaw_error = error.get('orientation_z', 0.0)
            
            # Create state error array for coupling calculator
            state_error_array = np.array([
                depth_error,
                surge_error,
                sway_error,
                heave_error,
                roll_error,
                pitch_error,
                yaw_error
            ])
            
            # Use coupling-aware reward if available
            if self.coupling_calculator is not None:
                # Get coupling method from config
                coupling_method = self.config.get('coupling', {}).get('method', 'v4')
                
                if coupling_method == 'v1':
                    reward = self.coupling_calculator.calculate_coupling_aware_reward_v1(
                        state_error_array, episode_step)
                elif coupling_method == 'v2':
                    reward = self.coupling_calculator.calculate_coupling_aware_reward_v2(
                        state_error_array, episode_step)
                elif coupling_method == 'v3':
                    reward = self.coupling_calculator.calculate_coupling_aware_reward_v3(
                        state_error_array, episode_step)
                elif coupling_method == 'v4':
                    reward = self.coupling_calculator.calculate_coupling_aware_reward_v4_enhanced(
                        state_error_array, episode_step)
                else:
                    # Fallback to simple reward
                    reward = self._calculate_simple_reward(state_error_array)
                
                return float(reward) if isinstance(reward, (int, float, np.number)) else 0.0
            else:
                # Fallback to simple reward calculation
                return self._calculate_simple_reward(state_error_array)
                
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0
    
    def _calculate_simple_reward(self, state_error_array):
        """Simple fallback reward calculation"""
        error_magnitude = np.sum(state_error_array**2)
        return -error_magnitude


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ROS2 bag to SAC training CSV")
    parser.add_argument("mcap_file", help="Path to .mcap or .db3 bag file")
    parser.add_argument("-o", "--output", help="Output CSV file path", 
                       default="sac_training_data.csv")
    parser.add_argument("-c", "--config", help="Config file path", 
                       default="config/config_sac.yaml")
    parser.add_argument("--list-topics", action="store_true", 
                       help="List topics without converting")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mcap_file):
        print(f"❌ Bag file not found: {args.mcap_file}")
        return 1
    
    converter = McapToTrainingCSV(config_path=args.config)
    
    if args.list_topics:
        # Add topic listing functionality if needed
        print("Topic listing not yet implemented")
        return 0
    
    try:
        df = converter.convert_bag_to_training_csv(args.mcap_file, args.output)
        print(f"\n🎉 Success! Training data ready for SAC")
        print(f"Output: {args.output}")
        print(f"Shape: {df.shape}")
        return 0
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    main()