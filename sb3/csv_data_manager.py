# csv_data_manager.py
import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Optional, Tuple
import random

class CSVDataManager:
    """
    Manages CSV data from rosbags for offline training.
    Each CSV file represents a different ROS topic from the same recording session.
    """
    
    def __init__(self, csv_directory: str, config: dict):
        """
        Initialize CSV data manager
        
        Args:
            csv_directory: Directory containing CSV files from rosbags (one CSV per topic)
            config: Configuration dictionary (same as used in online training)
        """
        self.csv_directory = csv_directory
        self.config = config
        
        # Load all CSV data by topic
        self.data = self._load_csv_files()
        
        # Synchronize all topics to common timeline
        self.synchronized_data = self._synchronize_topics()
        
        # Current position in the synchronized data
        self.current_step = 0
        self.total_steps = len(self.synchronized_data) if self.synchronized_data else 0
        
        # State variables (mimic AUVEnvNode structure)
        self.position_state = np.zeros(3)
        self.orientation_state = np.zeros(3)
        self.v_state = np.zeros(3)
        self.omega_ref_state = np.zeros(3)
        self.position_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)
        self.linear_acceleration = np.zeros(3)
        
        # Action tracking
        self.thrust_heave_bow = 0.0
        self.thrust_surge_port = 0.0
        self.thrust_surge_starboard = 0.0
        self.thrust_heave_stern = 0.0
        self.joint_angles_port = 0.0
        self.joint_angles_starboard = 0.0
        
        print(f"CSV Data Manager initialized:")
        print(f"  Total synchronized timesteps: {self.total_steps}")
        if self.total_steps > 0:
            duration = self.synchronized_data[-1]['timestamp'] - self.synchronized_data[0]['timestamp']
            print(f"  Total duration: {duration:.1f} seconds")
            print(f"  Potential episodes (500 steps each): {self.total_steps // 500}")
        
    def _load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files and organize by topic name"""
        data = {}
        
        # Map your actual CSV filenames to topic names
        file_mappings = {
            'race2_auv_controller_process_value.csv': 'state',
            'race2_auv_controller_process_error.csv': 'error',
            'race2_auv_imu_data.csv': 'imu',
            'race2_auv_control_thruster_heave_bow.csv': 'thruster_heave_bow',
            'race2_auv_control_thruster_heave_stern.csv': 'thruster_heave_stern', 
            'race2_auv_control_thruster_surge_port.csv': 'thruster_surge_port',
            'race2_auv_control_thruster_surge_starboard.csv': 'thruster_surge_starboard',
            'race2_auv_control_surge_port_servo.csv': 'servo_port',
            'race2_auv_control_surge_starboard_servo.csv': 'servo_starboard'
        }
        
        print("Loading CSV files by topic:")
        print("=" * 50)
        
        for filename, topic_name in file_mappings.items():
            file_path = os.path.join(self.csv_directory, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Handle timestamp column - your data has 'timestamp' column
                    if 'timestamp' in df.columns:
                        # Keep original timestamps for synchronization
                        pass
                    elif 'time' in df.columns:
                        df['timestamp'] = df['time']
                    else:
                        # Create artificial timestamps assuming 10Hz
                        print(f"  Warning: No timestamp in {filename}, creating artificial 10Hz timestamps")
                        df['timestamp'] = np.arange(len(df)) * 0.1
                    
                    # Calculate sample rate
                    if len(df) > 1:
                        dt = df['timestamp'].iloc[1] - df['timestamp'].iloc[0]
                        frequency = 1.0 / dt if dt > 0 else 10.0
                        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
                        print(f"✓ {topic_name}: {len(df)} samples, {frequency:.1f}Hz, {duration:.1f}s")
                    else:
                        print(f"✓ {topic_name}: {len(df)} samples")
                    
                    data[topic_name] = df
                    
                except Exception as e:
                    print(f"✗ {filename}: Error loading - {e}")
            else:
                print(f"✗ {filename}: File not found")
        
        return data
    
    def _synchronize_topics(self) -> List[Dict]:
        """
        Synchronize all topics to a common timeline - optimized version
        Returns list of synchronized data points
        """
        if not self.data:
            return []
        
        print(f"\nSynchronizing {len(self.data)} topics...")
        
        # Find common time range across all topics
        min_time = float('inf')
        max_time = float('-inf')
        
        for topic, df in self.data.items():
            if 'timestamp' in df.columns and len(df) > 0:
                topic_min = df['timestamp'].min()
                topic_max = df['timestamp'].max()
                min_time = min(min_time, topic_min)
                max_time = max(max_time, topic_max)
        
        if min_time == float('inf'):
            print("Error: No valid timestamps found")
            return []
        
        print(f"Common time range: {min_time:.1f}s to {max_time:.1f}s")
        
        # Use the state topic as reference (it has exactly 10Hz)
        if 'state' not in self.data:
            print("Error: No state data available for synchronization")
            return []
        
        state_df = self.data['state']
        print(f"Using state topic as reference: {len(state_df)} samples")
        
        synchronized_data = []
        
        # Pre-process all dataframes for faster lookup
        topic_indices = {}
        for topic, df in self.data.items():
            if 'timestamp' in df.columns:
                # Sort by timestamp and reset index for faster lookup
                df_sorted = df.sort_values('timestamp').reset_index(drop=True)
                topic_indices[topic] = {
                    'df': df_sorted,
                    'timestamps': df_sorted['timestamp'].values,
                    'current_idx': 0
                }
        
        # Process in chunks to show progress
        chunk_size = 5000
        total_chunks = (len(state_df) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(state_df))
            
            print(f"Processing chunk {chunk_idx + 1}/{total_chunks} (rows {start_idx}-{end_idx})")
            
            for i in range(start_idx, end_idx):
                state_timestamp = state_df.iloc[i]['timestamp']
                
                data_point = {'timestamp': state_timestamp}
                
                # Find matching data for each topic
                for topic, topic_data in topic_indices.items():
                    # Find closest timestamp using binary search-like approach
                    timestamps = topic_data['timestamps']
                    current_idx = topic_data['current_idx']
                    
                    # Find the best match within tolerance
                    tolerance = 0.05  # 50ms tolerance (half of 10Hz period)
                    best_idx = None
                    best_diff = float('inf')
                    
                    # Search around current position
                    search_range = 50  # Search within +/- 50 samples
                    start_search = max(0, current_idx - search_range)
                    end_search = min(len(timestamps), current_idx + search_range)
                    
                    for j in range(start_search, end_search):
                        diff = abs(timestamps[j] - state_timestamp)
                        if diff < best_diff:
                            best_diff = diff
                            best_idx = j
                    
                    # If we found a good match, use it
                    if best_idx is not None and best_diff <= tolerance:
                        data_point[topic] = topic_data['df'].iloc[best_idx].to_dict()
                        # Update current index for next search
                        topic_data['current_idx'] = best_idx
                
                # Only include if we have essential data (state and error)
                if 'state' in data_point and 'error' in data_point:
                    synchronized_data.append(data_point)
        
        print(f"Synchronized to {len(synchronized_data)} timesteps")
        
        if len(synchronized_data) == 0:
            print("Warning: No synchronized data points found!")
            print("This might indicate timestamp format issues or no overlapping data")
        
        return synchronized_data
    
    def reset_episode(self, episode_length: int = 500) -> Dict:
        """
        Reset to start a new episode from the synchronized data
        
        Args:
            episode_length: Maximum steps per episode
            
        Returns:
            Dictionary with episode info
        """
        if not self.synchronized_data:
            raise ValueError("No synchronized data available")
        
        # Randomly pick a starting point that allows for full episode
        max_start = max(0, self.total_steps - episode_length)
        
        if max_start <= 0:
            # Data is shorter than episode length, start from beginning
            self.current_step = 0
            actual_episode_length = self.total_steps
        else:
            # Random start position
            self.current_step = random.randint(0, max_start)
            actual_episode_length = min(episode_length, self.total_steps - self.current_step)
        
        # Initialize state from first step
        self._update_state_from_step(self.current_step)
        
        episode_info = {
            'episode_start_step': self.current_step,
            'episode_length': actual_episode_length,
            'start_timestamp': self.synchronized_data[self.current_step]['timestamp'],
            'data_source': 'synchronized_csv'
        }
        
        return episode_info
    
    def step_episode(self) -> Tuple[bool, Dict]:
        """
        Advance one step in the current episode
        
        Returns:
            (episode_done, step_info)
        """
        if not self.synchronized_data:
            return True, {'reason': 'no_data'}
        
        self.current_step += 1
        
        # Check if we've reached the end of available data
        if self.current_step >= self.total_steps:
            return True, {'reason': 'data_complete'}
        
        # Update state from current step
        self._update_state_from_step(self.current_step)
        
        step_info = {
            'current_step': self.current_step,
            'timestamp': self.synchronized_data[self.current_step]['timestamp'],
            'progress': self.current_step / self.total_steps
        }
        
        return False, step_info
    
    def _update_state_from_step(self, step: int):
        """Update internal state variables from synchronized data"""
        if step >= len(self.synchronized_data):
            return
        
        data_point = self.synchronized_data[step]
        
        # Update state data
        if 'state' in data_point:
            state_data = data_point['state']
            self.position_state = np.array([
                state_data.get('_position._x', 0.0),  # Note: using underscore format
                state_data.get('_position._y', 0.0),
                state_data.get('_position._z', 0.0)
            ])
            self.orientation_state = np.array([
                state_data.get('_orientation._x', 0.0),
                state_data.get('_orientation._y', 0.0),
                state_data.get('_orientation._z', 0.0)
            ])
            self.v_state = np.array([
                state_data.get('_velocity._x', 0.0),
                state_data.get('_velocity._y', 0.0),
                state_data.get('_velocity._z', 0.0)
            ])
            self.omega_ref_state = np.array([
                state_data.get('_angular_rate._x', 0.0),
                state_data.get('_angular_rate._y', 0.0),
                state_data.get('_angular_rate._z', 0.0)
            ])
        
        # Update error data
        if 'error' in data_point:
            error_data = data_point['error']
            self.position_err = np.array([
                error_data.get('_position._x', 0.0),
                error_data.get('_position._y', 0.0),
                error_data.get('_position._z', 0.0)
            ])
            self.orientation_err = np.array([
                error_data.get('_orientation._x', 0.0),
                error_data.get('_orientation._y', 0.0),
                error_data.get('_orientation._z', 0.0)
            ])
            self.v_err = np.array([
                error_data.get('_velocity._x', 0.0),
                error_data.get('_velocity._y', 0.0),
                error_data.get('_velocity._z', 0.0)
            ])
            self.omega_ref_err = np.array([
                error_data.get('_angular_rate._x', 0.0),
                error_data.get('_angular_rate._y', 0.0),
                error_data.get('_angular_rate._z', 0.0)
            ])
        
        # Update IMU data
        if 'imu' in data_point:
            imu_data = data_point['imu']
            self.linear_acceleration = np.array([
                imu_data.get('_linear_acceleration._x', 0.0),
                imu_data.get('_linear_acceleration._y', 0.0),
                imu_data.get('_linear_acceleration._z', 0.0)
            ])
        
        # Update thruster commands from recorded data (for reference)
        thruster_topics = ['thruster_heave_bow', 'thruster_heave_stern', 
                          'thruster_surge_port', 'thruster_surge_starboard']
        for topic in thruster_topics:
            if topic in data_point:
                value = data_point[topic].get('_data', 0.0)  # Note: using _data instead of data
                setattr(self, topic.replace('thruster_', 'thrust_'), value)
        
        # Update servo commands from recorded data (for reference)
        if 'servo_port' in data_point:
            self.joint_angles_port = data_point['servo_port'].get('_data', 0.0)
        if 'servo_starboard' in data_point:
            self.joint_angles_starboard = data_point['servo_starboard'].get('_data', 0.0)
        
    def _get_current_state_observation(self) -> np.ndarray:
        """Get current state observation in the same format as online training"""
        # Convert orientation to sin/cos representation (same as online version)
        roll_sin = np.sin(self.orientation_state[0])
        roll_cos = np.cos(self.orientation_state[0])
        pitch_sin = np.sin(self.orientation_state[1])
        pitch_cos = np.cos(self.orientation_state[1])
        yaw_sin = np.sin(self.orientation_state[2])
        yaw_cos = np.cos(self.orientation_state[2])
        
        # Error components with sin/cos representation
        roll_sin_err = np.sin(self.orientation_err[0])
        roll_cos_err = np.cos(self.orientation_err[0])
        pitch_sin_err = np.sin(self.orientation_err[1])
        pitch_cos_err = np.cos(self.orientation_err[1])
        yaw_sin_err = np.sin(self.orientation_err[2])
        yaw_cos_err = np.cos(self.orientation_err[2])
        
        # Build observation same as online training
        observation = np.concatenate([
            # Error components (9 elements) - note: removed heave_error to match your online version
            self.position_err[2:3],           # depth error
            self.v_err[:2],                   # surge, sway error
            np.array([roll_sin_err, roll_cos_err, pitch_sin_err, pitch_cos_err, yaw_sin_err, yaw_cos_err]),
            
            # Velocity components (3 elements)
            self.v_state[:3],                 # surge, sway, heave velocity
            
            # Angular rate components (3 elements)
            self.omega_ref_state[:3],         # roll, pitch, yaw rates
            
            # Acceleration components (2 elements)
            self.linear_acceleration[:2]      # x, y acceleration
        ])
        
        return observation
    
    def get_recorded_action_at_current_step(self) -> np.ndarray:
        """Get the recorded action from CSV data at current step (for reference/analysis)"""
        action = np.zeros(6)  # 4 thrusters + 2 servos
        
        if self.current_step < len(self.synchronized_data):
            data_point = self.synchronized_data[self.current_step]
            
            # Thruster commands
            thruster_names = ['thruster_heave_bow', 'thruster_heave_stern', 
                             'thruster_surge_port', 'thruster_surge_starboard']
            for i, topic in enumerate(thruster_names):
                if topic in data_point:
                    action[i] = data_point[topic].get('_data', 0.0)  # Note: using _data
            
            # Servo commands
            if 'servo_port' in data_point:
                action[4] = data_point['servo_port'].get('_data', 0.0)
            if 'servo_starboard' in data_point:
                action[5] = data_point['servo_starboard'].get('_data', 0.0)
        
        return action
    
    def get_state_error_array(self) -> np.ndarray:
        """Get state error array for reward calculation (same format as online training)"""
        # Convert orientation errors to sin/cos
        roll_sin_err = np.sin(self.orientation_err[0])
        roll_cos_err = np.cos(self.orientation_err[0])
        pitch_sin_err = np.sin(self.orientation_err[1])
        pitch_cos_err = np.cos(self.orientation_err[1])
        yaw_sin_err = np.sin(self.orientation_err[2])
        yaw_cos_err = np.cos(self.orientation_err[2])
        
        return np.concatenate([
            self.position_err[2:3],   # depth error
            self.v_err[:3],           # surge, sway, heave error
            np.array([roll_sin_err, roll_cos_err, pitch_sin_err, pitch_cos_err, yaw_sin_err, yaw_cos_err])
        ])
    
    def get_dataset_stats(self) -> Dict:
        """Get comprehensive statistics about the dataset"""
        if not self.synchronized_data:
            return {}
        
        stats = {
            'total_timesteps': len(self.synchronized_data),
            'duration_seconds': self.synchronized_data[-1]['timestamp'] - self.synchronized_data[0]['timestamp'],
            'sample_rate_hz': 10.0,  # Fixed at 10Hz
            'potential_episodes_500_steps': len(self.synchronized_data) // 500,
            'potential_episodes_300_steps': len(self.synchronized_data) // 300,
            'topics_available': []
        }
        
        # Check which topics are consistently available
        if self.synchronized_data:
            first_point = self.synchronized_data[0]
            for topic in first_point.keys():
                if topic != 'timestamp':
                    # Check availability across the dataset
                    available_count = sum(1 for point in self.synchronized_data if topic in point)
                    availability_pct = (available_count / len(self.synchronized_data)) * 100
                    stats['topics_available'].append({
                        'topic': topic,
                        'availability_percent': availability_pct
                    })
        
        return stats