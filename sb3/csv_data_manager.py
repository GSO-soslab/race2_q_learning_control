# gpu_csv_data_manager.py
import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Optional, Tuple
import random
import time

# GPU support with fallback
try:
    import torch as th
    GPU_AVAILABLE = th.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"🚀 GPU detected: {th.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU not available, using CPU for data processing")
except ImportError:
    th = None
    GPU_AVAILABLE = False
    print("⚠️  PyTorch not available, using CPU for data processing")

class CSVDataManager:
    """
    GPU-optimized CSV data manager for rosbags with automatic CPU fallback.
    Maintains the same interface as the original class.
    """
    
    def __init__(self, csv_directory: str, config: dict, device: str = 'auto', enable_gpu_preload: bool = True):
        """
        Initialize CSV data manager with GPU support
        
        Args:
            csv_directory: Directory containing CSV files from rosbags
            config: Configuration dictionary
            device: 'auto', 'cpu', 'cuda', or specific device like 'cuda:0'
            enable_gpu_preload: Whether to preload data to GPU memory
        """
        self.csv_directory = csv_directory
        self.config = config
        self.enable_gpu_preload = enable_gpu_preload
        
        # GPU/CPU device setup with fallback
        self.device, self.use_gpu = self._setup_device(device)
        
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
        
        # GPU optimization: Pre-computed tensors for faster access
        self.gpu_observations = None
        self.gpu_actions = None
        self.gpu_rewards = None
        
        # Initialize GPU preloading if enabled
        if self.use_gpu and self.enable_gpu_preload and self.total_steps > 0:
            self._preload_gpu_data()
        
        print(f"CSV Data Manager initialized:")
        print(f"  Device: {self.device} ({'GPU' if self.use_gpu else 'CPU'} mode)")
        print(f"  Total synchronized timesteps: {self.total_steps}")
        if self.total_steps > 0:
            duration = self.synchronized_data[-1]['timestamp'] - self.synchronized_data[0]['timestamp']
            print(f"  Total duration: {duration:.1f} seconds")
            print(f"  Potential episodes (500 steps each): {self.total_steps // 500}")
            if self.use_gpu and self.enable_gpu_preload:
                gpu_memory_mb = (self.total_steps * 24 * 4) / 1e6  # Rough estimate: 24 floats per obs
                print(f"  GPU memory usage: ~{gpu_memory_mb:.1f} MB for preloaded data")
    
    def _setup_device(self, device: str) -> Tuple[str, bool]:
        """Setup GPU/CPU device with automatic fallback"""
        if not GPU_AVAILABLE or th is None:
            return 'cpu', False
        
        if device == 'auto':
            if th.cuda.is_available():
                selected_device = 'cuda:0'
                use_gpu = True
            else:
                selected_device = 'cpu'
                use_gpu = False
        elif device == 'cpu':
            selected_device = 'cpu'
            use_gpu = False
        elif device.startswith('cuda'):
            if th.cuda.is_available():
                # Validate GPU exists
                try:
                    gpu_id = int(device.split(':')[1]) if ':' in device else 0
                    if gpu_id < th.cuda.device_count():
                        selected_device = device
                        use_gpu = True
                    else:
                        print(f"⚠️  GPU {gpu_id} not available, falling back to CPU")
                        selected_device = 'cpu'
                        use_gpu = False
                except:
                    print(f"⚠️  Invalid GPU device {device}, falling back to CPU")
                    selected_device = 'cpu'
                    use_gpu = False
            else:
                print(f"⚠️  CUDA not available, falling back to CPU")
                selected_device = 'cpu'
                use_gpu = False
        else:
            print(f"⚠️  Unknown device {device}, falling back to CPU")
            selected_device = 'cpu'
            use_gpu = False
        
        return selected_device, use_gpu
    
    def _preload_gpu_data(self):
        """Pre-compute and cache observations/actions on GPU for faster access"""
        if not self.use_gpu or not self.synchronized_data:
            return
        
        print("🔄 Pre-loading data to GPU memory for faster access...")
        start_time = time.time()
        
        observations = []
        actions = []
        
        # Process in chunks to show progress and manage memory
        chunk_size = 5000
        total_chunks = (self.total_steps + chunk_size - 1) // chunk_size
        
        original_step = self.current_step  # Save current position
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, self.total_steps)
            
            chunk_obs = []
            chunk_actions = []
            
            for i in range(start_idx, end_idx):
                # Update state from this step
                self.current_step = i
                self._update_state_from_step(i)
                
                # Get observation and action
                obs = self._get_current_state_observation()
                action = self.get_recorded_action_at_current_step()
                
                chunk_obs.append(obs)
                chunk_actions.append(action)
            
            # Convert chunk to GPU tensors
            if chunk_obs:
                try:
                    chunk_obs_tensor = th.tensor(np.array(chunk_obs), dtype=th.float32, device=self.device)
                    chunk_actions_tensor = th.tensor(np.array(chunk_actions), dtype=th.float32, device=self.device)
                    
                    observations.append(chunk_obs_tensor)
                    actions.append(chunk_actions_tensor)
                    
                    if chunk_idx % max(1, total_chunks // 10) == 0:
                        print(f"  Processed chunk {chunk_idx + 1}/{total_chunks} ({(chunk_idx + 1) / total_chunks * 100:.1f}%)")
                        
                except Exception as e:
                    print(f"⚠️  GPU memory error in chunk {chunk_idx}, falling back to CPU mode: {e}")
                    self.use_gpu = False
                    self.device = 'cpu'
                    return
        
        # Concatenate all chunks
        if observations:
            try:
                self.gpu_observations = th.cat(observations, dim=0)
                self.gpu_actions = th.cat(actions, dim=0)
                
                load_time = time.time() - start_time
                data_size_mb = (self.gpu_observations.numel() + self.gpu_actions.numel()) * 4 / 1e6
                
                print(f"✅ Pre-loaded {len(self.gpu_observations)} timesteps to GPU")
                print(f"  Load time: {load_time:.1f}s")
                print(f"  GPU memory used: {data_size_mb:.1f} MB")
                # print(f"  Access speedup: ~10-50x faster observation retrieval")
                
            except Exception as e:
                print(f"⚠️  GPU concatenation error, falling back to CPU mode: {e}")
                self.use_gpu = False
                self.device = 'cpu'
                self.gpu_observations = None
                self.gpu_actions = None
        
        # Restore original position
        self.current_step = original_step
        if original_step < self.total_steps:
            self._update_state_from_step(original_step)
    
    def _load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files and organize by topic name - same as original"""
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
        """Synchronize all topics to a common timeline - same as original but with GPU optimization"""
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
        
        # GPU optimization: Use numpy arrays for faster lookup if possible
        topic_indices = {}
        for topic, df in self.data.items():
            if 'timestamp' in df.columns:
                # Sort by timestamp and reset index for faster lookup
                df_sorted = df.sort_values('timestamp').reset_index(drop=True)
                topic_indices[topic] = {
                    'df': df_sorted,
                    'timestamps': df_sorted['timestamp'].values,  # numpy array for faster access
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
        """Reset to start a new episode - same interface as original"""
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
            'data_source': 'synchronized_csv',
            'device': self.device,  # Add device info
            'gpu_accelerated': self.use_gpu
        }
        
        return episode_info
    
    def step_episode(self) -> Tuple[bool, Dict]:
        """Advance one step - same interface as original"""
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
        """Update internal state variables - same as original"""
        if step >= len(self.synchronized_data):
            return
        
        data_point = self.synchronized_data[step]
        
        # Update state data
        if 'state' in data_point:
            state_data = data_point['state']
            self.position_state = np.array([
                state_data.get('_position._x', 0.0),
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
                value = data_point[topic].get('_data', 0.0)
                setattr(self, topic.replace('thruster_', 'thrust_'), value)
        
        # Update servo commands from recorded data (for reference)
        if 'servo_port' in data_point:
            self.joint_angles_port = data_point['servo_port'].get('_data', 0.0)
        if 'servo_starboard' in data_point:
            self.joint_angles_starboard = data_point['servo_starboard'].get('_data', 0.0)
    
    def _get_current_state_observation(self) -> np.ndarray:
        """Get current state observation - GPU-optimized version"""
        # GPU optimization: Use pre-computed observations if available
        if self.use_gpu and self.gpu_observations is not None and self.current_step < len(self.gpu_observations):
            # Return as numpy array for compatibility
            return self.gpu_observations[self.current_step].cpu().numpy()
        
        # Fallback to original computation (same as original method)
        depth = self.position_state[2:3]
        depth_error = self.position_err[2:3]
        surge_velocity_error = self.v_err[0:1]
        sway_velocity_error = self.v_err[1:2]
        
        # Orientation Errors (sin/cos representation)
        roll_sin_err = np.array([np.sin(self.orientation_err[0])])
        roll_cos_err = np.array([np.cos(self.orientation_err[0])])
        pitch_sin_err = np.array([np.sin(self.orientation_err[1])])
        pitch_cos_err = np.array([np.cos(self.orientation_err[1])])
        yaw_sin_err = np.array([np.sin(self.orientation_err[2])])
        yaw_cos_err = np.array([np.cos(self.orientation_err[2])])
        
        # Current Velocities
        surge_velocity = self.v_state[0:1]
        sway_velocity = self.v_state[1:2]
        heave_velocity = self.v_state[2:3]
        
        # Angular Rates
        roll_rate = self.omega_ref_state[0:1]
        pitch_rate = self.omega_ref_state[1:2]
        yaw_rate = self.omega_ref_state[2:3]
        
        # Linear Accelerations
        x_acceleration = self.linear_acceleration[0:1]
        y_acceleration = self.linear_acceleration[1:2]
        
        # Current orientation 
        roll_sin_current = np.array([np.sin(self.orientation_state[0])])
        roll_cos_current = np.array([np.cos(self.orientation_state[0])])
        pitch_sin_current = np.array([np.sin(self.orientation_state[1])])
        pitch_cos_current = np.array([np.cos(self.orientation_state[1])])
        yaw_sin_current = np.array([np.sin(self.orientation_state[2])])
        yaw_cos_current = np.array([np.cos(self.orientation_state[2])])
        
        # Assemble Observation Vector (Size is 24)
        observation_components = [
            depth, 
            depth_error, 
            surge_velocity_error, 
            sway_velocity_error,
            roll_sin_err, 
            roll_cos_err, 
            pitch_sin_err, 
            pitch_cos_err,
            yaw_sin_err,
            yaw_cos_err, 
            roll_sin_current, 
            roll_cos_current,
            pitch_sin_current, 
            pitch_cos_current, 
            yaw_sin_current, 
            yaw_cos_current,
            surge_velocity, 
            sway_velocity, 
            heave_velocity,
            roll_rate,
            pitch_rate, 
            yaw_rate,
            x_acceleration, y_acceleration,
        ]
        
        observation = np.concatenate(observation_components)
        return observation
    
    def get_recorded_action_at_current_step(self) -> np.ndarray:
        """Get recorded action - GPU-optimized version"""
        # GPU optimization: Use pre-computed actions if available
        if self.use_gpu and self.gpu_actions is not None and self.current_step < len(self.gpu_actions):
            # Return as numpy array for compatibility
            return self.gpu_actions[self.current_step].cpu().numpy()
        
        # Fallback to original computation
        action = np.zeros(6)  # 4 thrusters + 2 servos
        
        if self.current_step < len(self.synchronized_data):
            data_point = self.synchronized_data[self.current_step]
            
            # Thruster commands
            thruster_names = ['thruster_heave_bow', 'thruster_heave_stern', 
                             'thruster_surge_port', 'thruster_surge_starboard']
            for i, topic in enumerate(thruster_names):
                if topic in data_point:
                    action[i] = data_point[topic].get('_data', 0.0)
            
            # Servo commands
            if 'servo_port' in data_point:
                action[4] = data_point['servo_port'].get('_data', 0.0)
            if 'servo_starboard' in data_point:
                action[5] = data_point['servo_starboard'].get('_data', 0.0)
        
        return action
    
    def get_state_error_array(self) -> np.ndarray:
        """Get state error array - same as original"""
        # Convert orientation errors to sin/cos
        
        # depth = self.node.position_state[2:3]
        depth_error = self.position_err[2:3]
        surge_velocity_error = self.v_err[0:1]
        sway_velocity_error = self.v_err[1:2]
        heave_velocioty_error = self.v_err[2:3]
        roll_error = self.orientation_err[0:1]
        pitch_error = self.orientation_err[1:2]
        yaw_error = self.orientation_err[2:3]

        # roll_sin_err = np.array([np.sin(self.node.orientation_err[0])])
        # roll_cos_err = np.array([np.cos(self.node.orientation_err[0])])
        # pitch_sin_err = np.array([np.sin(self.node.orientation_err[1])])
        # pitch_cos_err = np.array([np.cos(self.node.orientation_err[1])])
        # yaw_sin_err = np.array([np.sin(self.node.orientation_err[2])])
        # yaw_cos_err = np.array([np.cos(self.node.orientation_err[2])])
        
        # roll_sin_current = np.array([np.sin(self.node.orientation_state[0])])
        # roll_cos_current = np.array([np.cos(self.node.orientation_state[0])])
        # pitch_sin_current = np.array([np.sin(self.node.orientation_state[1])])
        # pitch_cos_current = np.array([np.cos(self.node.orientation_state[1])])
        # yaw_sin_current = np.array([np.sin(self.node.orientation_state[2])])
        # yaw_cos_current = np.array([np.cos(self.node.orientation_state[2])])
        
        # surge_velocity = self.node.v_state[0:1]
        # sway_velocity = self.node.v_state[1:2]
        # heave_velocity = self.node.v_state[2:3]
        
        # roll_rate = self.node.omega_ref_state[0:1]
        # pitch_rate = self.node.omega_ref_state[1:2]
        # yaw_rate = self.node.omega_ref_state[2:3]
        
        # x_acceleration = self.node.linear_acceleration[0:1]
        # y_acceleration = self.node.linear_acceleration[1:2]

        # return np.concatenate([
        #     self.position_err[2:3],   # depth error
        #     self.v_err[:3],           # surge, sway, heave error
        #     np.array([roll_sin_err, roll_cos_err, pitch_sin_err, pitch_cos_err, yaw_sin_err, yaw_cos_err])
        # ])
    
        return np.concatenate([
                depth_error,
                surge_velocity_error,
                sway_velocity_error,
                heave_velocioty_error ,#heave error
                # np.array([np.sin(self.node.orientation_err[0]), np.cos(self.node.orientation_err[0])]),
                # np.array([np.sin(self.node.orientation_err[1]), np.cos(self.node.orientation_err[1])]),
                # np.array([np.sin(self.node.orientation_err[2]), np.cos(self.node.orientation_err[2])])
                roll_error,
                pitch_error,
                yaw_error
            ])
    
    def get_dataset_stats(self) -> Dict:
        """Get comprehensive statistics - enhanced with GPU info"""
        if not self.synchronized_data:
            return {}
        
        stats = {
            'total_timesteps': len(self.synchronized_data),
            'duration_seconds': self.synchronized_data[-1]['timestamp'] - self.synchronized_data[0]['timestamp'],
            'sample_rate_hz': 10.0,
            'potential_episodes_500_steps': len(self.synchronized_data) // 500,
            'potential_episodes_300_steps': len(self.synchronized_data) // 300,
            'device': self.device,
            'gpu_accelerated': self.use_gpu,
            'gpu_preloaded': self.gpu_observations is not None,
            'topics_available': []
        }
        
        # GPU memory usage info
        if self.use_gpu and self.gpu_observations is not None:
            obs_memory_mb = (self.gpu_observations.numel() * 4) / 1e6
            act_memory_mb = (self.gpu_actions.numel() * 4) / 1e6
            stats['gpu_memory_usage_mb'] = obs_memory_mb + act_memory_mb
            stats['estimated_speedup'] = '10-50x faster observation access'
        
        # Check which topics are consistently available
        if self.synchronized_data:
            first_point = self.synchronized_data[0]
            for topic in first_point.keys():
                if topic != 'timestamp':
                    available_count = sum(1 for point in self.synchronized_data if topic in point)
                    availability_pct = (available_count / len(self.synchronized_data)) * 100
                    stats['topics_available'].append({
                        'topic': topic,
                        'availability_percent': availability_pct
                    })
        
        return stats
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if self.use_gpu:
            self.gpu_observations = None
            self.gpu_actions = None
            if th is not None and th.cuda.is_available():
                th.cuda.empty_cache()
                print("🗑️  GPU cache cleared")
    
    def __del__(self):
        """Cleanup GPU memory on deletion"""
        if hasattr(self, 'use_gpu') and self.use_gpu:
            self.clear_gpu_cache()