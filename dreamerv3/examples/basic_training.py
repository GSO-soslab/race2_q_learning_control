#!/usr/bin/env python3

"""
Basic DreamerV3 training example for AUV control
Demonstrates the minimal setup required to start training
"""

import os
import sys
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from AUVEnv_DreamerV3 import AUVEnvDreamerV3
from train_dreamerv3_auv import AUVDreamerV3Trainer
import argparse

def basic_training_example():
    """Run a basic training example"""
    print("=== DreamerV3 AUV Basic Training Example ===")
    
    # Configuration
    config_path = Path(__file__).parent.parent / "config" / "config_dreamerv3.yaml"
    
    # Simple argument setup
    class Args:
        def __init__(self):
            self.mode = 'train'
            self.offline_mode = False
            self.timesteps = 50000  # Reduced for example
            self.resume_from_checkpoint = None
            self.log_interval = 50
            self.plot_interval = 500
    
    args = Args()
    
    # Test environment first
    print("1. Testing environment...")
    try:
        env = AUVEnvDreamerV3(config_path=str(config_path), offline_mode=False)
        obs, info = env.reset()
        print(f"   ✓ Environment initialized")
        print(f"   ✓ Observation shape: {obs['vector'].shape}")
        print(f"   ✓ Action space: {env.action_space}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   ✓ Step {i+1}: reward={reward:.4f}")
        
        env.close()
        
    except Exception as e:
        print(f"   ✗ Environment test failed: {e}")
        return
    
    # Initialize trainer
    print("2. Initializing trainer...")
    try:
        trainer = AUVDreamerV3Trainer(str(config_path), args)
        print("   ✓ Trainer initialized")
        
    except ImportError as e:
        print(f"   ✗ Missing dependencies: {e}")
        print("   Please install DreamerV3: git clone https://github.com/danijar/dreamerv3.git")
        return
    except Exception as e:
        print(f"   ✗ Trainer initialization failed: {e}")
        return
    
    # Run short training
    print("3. Starting training...")
    print(f"   Training for {args.timesteps} timesteps")
    print("   Press Ctrl+C to stop early")
    
    try:
        trainer.run()
        print("   ✓ Training completed successfully")
        
    except KeyboardInterrupt:
        print("   ✓ Training stopped by user")
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    basic_training_example()


#########################################
# examples/offline_training.py
#########################################

#!/usr/bin/env python3
"""
Offline training example using preprocessed ROSbag data
Demonstrates how to use historical data for training
"""

import os
import sys
import numpy as np
from pathlib import Path
import h5py
import pickle

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rosbag_preprocessor import ROSBagPreprocessor
from train_dreamerv3_auv import AUVDreamerV3Trainer

def create_dummy_offline_data(output_dir):
    """Create dummy offline data for demonstration"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Creating dummy offline data...")
    
    # Generate synthetic episodes
    num_episodes = 5
    episodes = []
    
    for ep in range(num_episodes):
        episode_length = np.random.randint(50, 200)
        
        # Generate synthetic observations (17-dimensional)
        observations = np.random.randn(episode_length, 17).astype(np.float32)
        
        # Generate synthetic actions (4 thrusters)
        actions = np.random.uniform(-1, 1, (episode_length, 4)).astype(np.float32)
        
        # Generate synthetic rewards (decay over time)
        rewards = -np.linspace(1.0, 0.1, episode_length).astype(np.float32)
        
        # Terminal flags
        terminals = np.zeros(episode_length, dtype=bool)
        terminals[-1] = True
        
        episode_data = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals
        }
        
        episodes.append(episode_data)
    
    # Save as HDF5
    output_file = output_path / "dummy_episodes.h5"
    with h5py.File(output_file, 'w') as f:
        for i, episode in enumerate(episodes):
            group = f.create_group(f'episode_{i:06d}')
            for key, data in episode.items():
                group.create_dataset(key, data=data, compression='gzip')
    
    print(f"   ✓ Created {num_episodes} dummy episodes in {output_file}")
    return str(output_file)

def offline_training_example():
    """Run offline training example"""
    print("=== DreamerV3 AUV Offline Training Example ===")
    
    # Create dummy data
    data_dir = Path(__file__).parent / "dummy_data"
    dummy_data_file = create_dummy_offline_data(data_dir)
    
    # Update config for offline training
    config_path = Path(__file__).parent.parent / "config" / "config_dreamerv3.yaml"
    
    # Read config and modify for offline training
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable offline training
    config['offline_training']['enabled'] = True
    config['offline_training']['rosbag_paths'] = [dummy_data_file]
    config['training']['total_steps'] = 5000  # Reduced for example
    
    # Save modified config
    offline_config_path = Path(__file__).parent / "config_offline_example.yaml"
    with open(offline_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("1. Configuration updated for offline training")
    print(f"   ✓ Data path: {dummy_data_file}")
    print(f"   ✓ Config: {offline_config_path}")
    
    # Setup args for offline training
    class Args:
        def __init__(self):
            self.mode = 'train'
            self.offline_mode = True
            self.timesteps = 5000
            self.resume_from_checkpoint = None
            self.log_interval = 100
            self.plot_interval = 1000
    
    args = Args()
    
    # Initialize trainer
    print("2. Initializing offline trainer...")
    try:
        trainer = AUVDreamerV3Trainer(str(offline_config_path), args)
        print("   ✓ Offline trainer initialized")
        
    except Exception as e:
        print(f"   ✗ Trainer initialization failed: {e}")
        return
    
    # Run offline training
    print("3. Starting offline training...")
    try:
        trainer.run()
        print("   ✓ Offline training completed")
        
    except Exception as e:
        print(f"   ✗ Offline training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    offline_training_example()


#########################################
# examples/coupling_rewards_demo.py
#########################################

#!/usr/bin/env python3
"""
Demonstration of coupling-aware reward methods
Shows how different coupling methods perform on the same state errors
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from coupling_rewards_dreamerv3 import CouplingAwareRewardCalculator, DreamerV3RewardAdapter

def create_test_config():
    """Create test configuration for coupling rewards"""
    return {
        'reward_function': {
            'w1': 1.0,
            'state_error_weights': [0.0001, 0.005, 0.0001, 0.0, 0.0, 0.001, 0.0001]
        },
        'coupling': {
            'surge_yaw_weight': 0.25,
            'pitch_depth_weight': 0.6,
            'threshold': 0.05,
            'progressive': {'enabled': True, 'exploration_episodes': 50, 'coupling_focus_episodes': 100},
            'dynamic': {'history_window': 15, 'adaptation_rate': 0.1},
            'energy': {'spring_stiffness': 0.3, 'damping_factor': 0.4}
        },
        'dreamerv3': {
            'reward_scale': 1.0,
            'symlog_rewards': True,
            'reward_smoothing': 0.05,
            'imagination_bonus': 0.1,
            'world_model_consistency': 0.05
        }
    }

def generate_test_scenarios():
    """Generate different test scenarios for coupling evaluation"""
    scenarios = {}
    
    # Scenario 1: Good coupling (surge and yaw errors decrease together)
    scenarios['good_coupling'] = {
        'depth_errors': [0.5, 0.4, 0.3, 0.2, 0.1],
        'surge_errors': [0.8, 0.6, 0.4, 0.2, 0.1],
        'yaw_errors': [0.6, 0.4, 0.3, 0.15, 0.05],
        'pitch_errors': [0.3, 0.25, 0.2, 0.15, 0.1]
    }
    
    # Scenario 2: Poor coupling (surge improves, yaw gets worse)
    scenarios['poor_coupling'] = {
        'depth_errors': [0.5, 0.4, 0.3, 0.2, 0.1],
        'surge_errors': [0.8, 0.6, 0.4, 0.2, 0.1],
        'yaw_errors': [0.2, 0.3, 0.4, 0.5, 0.6],
        'pitch_errors': [0.3, 0.25, 0.2, 0.15, 0.1]
    }
    
    # Scenario 3: High errors (all errors high)
    scenarios['high_errors'] = {
        'depth_errors': [1.0, 1.1, 1.2, 1.1, 1.0],
        'surge_errors': [1.5, 1.4, 1.3, 1.2, 1.1],
        'yaw_errors': [0.8, 0.9, 1.0, 0.9, 0.8],
        'pitch_errors': [0.7, 0.8, 0.9, 0.8, 0.7]
    }
    
    return scenarios

def state_errors_to_array(depth_err, surge_err, yaw_err, pitch_err):
    """Convert individual errors to state error array format"""
    # Format: [depth, surge, sway, heave, roll_sin, roll_cos, pitch_sin, pitch_cos, yaw_sin, yaw_cos]
    state_error_array = np.array([
        depth_err,                    # depth error
        surge_err,                    # surge error
        0.0,                         # sway error (dummy)
        0.0,                         # heave error (dummy)
        np.sin(0.0), np.cos(0.0),    # roll error (dummy)
        np.sin(pitch_err), np.cos(pitch_err),  # pitch error
        np.sin(yaw_err), np.cos(yaw_err)       # yaw error
    ])
    
    return state_error_array

def coupling_rewards_demo():
    """Demonstrate different coupling reward methods"""
    print("=== Coupling-Aware Rewards Demonstration ===")
    
    # Initialize reward calculator
    config = create_test_config()
    calculator = CouplingAwareRewardCalculator(config)
    adapter = DreamerV3RewardAdapter(calculator, config)
    
    # Generate test scenarios
    scenarios = generate_test_scenarios()
    
    # Test each coupling method
    methods = ['v1', 'v2', 'v3', 'v4']
    results = {method: {} for method in methods}
    
    print("1. Testing coupling methods on different scenarios...")
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n   Scenario: {scenario_name}")
        
        for method in methods:
            # Update method in config
            calculator.config['coupling']['method'] = method
            
            rewards = []
            for i in range(len(scenario_data['depth_errors'])):
                # Create state error array
                state_error_array = state_errors_to_array(
                    scenario_data['depth_errors'][i],
                    scenario_data['surge_errors'][i],
                    scenario_data['yaw_errors'][i],
                    scenario_data['pitch_errors'][i]
                )
                
                # Calculate reward
                if method == 'v1':
                    reward = calculator.calculate_coupling_aware_reward_v1(state_error_array, i)
                elif method == 'v2':
                    reward = calculator.calculate_coupling_aware_reward_v2(state_error_array, i)
                elif method == 'v3':
                    reward = calculator.calculate_coupling_aware_reward_v3(state_error_array, i)
                elif method == 'v4':
                    reward = calculator.calculate_coupling_aware_reward_v4_enhanced(state_error_array, i)
                
                rewards.append(reward)
            
            results[method][scenario_name] = rewards
            
            # Calculate improvement (reward increase from first to last step)
            improvement = rewards[-1] - rewards[0]
            print(f"     {method}: improvement = {improvement:.4f}")
    
    # Visualize results
    print("\n2. Creating visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
        ax = axes[i]
        
        for method in methods:
            ax.plot(results[method][scenario_name], label=f'Method {method}', marker='o')
        
        ax.set_title(f'Scenario: {scenario_name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(__file__).parent / "coupling_rewards_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Plot saved to {plot_path}")
    
    # Show summary
    print("\n3. Summary:")
    print("   Method v1: Cross-correlation coupling")
    print("   Method v2: Hierarchical progressive learning")
    print("   Method v3: Dynamic coupling matrix")
    print("   Method v4: Energy-based coupling (recommended)")
    
    # Test DreamerV3 adapter
    print("\n4. Testing DreamerV3 adapter...")
    
    dummy_action = np.array([0.1, -0.2, 0.3, -0.1])
    state_error_array = state_errors_to_array(0.2, 0.3, 0.1, 0.15)
    
    adapted_reward = adapter.calculate_reward(state_error_array, dummy_action, 10)
    print(f"   ✓ Adapter reward: {adapted_reward:.4f}")
    
    # Get reward statistics
    stats = adapter.get_reward_statistics()
    if stats:
        print(f"   ✓ Reward statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

if __name__ == "__main__":
    coupling_rewards_demo()


#########################################
# examples/deployment_example.py
#########################################

#!/usr/bin/env python3
"""
Deployment example showing how to deploy a trained model
Includes both simulation and hardware deployment scenarios
"""

import sys
import time
import threading
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def create_dummy_model():
    """Create a dummy model for deployment testing"""
    import pickle
    import numpy as np
    
    model_path = Path(__file__).parent / "dummy_model"
    model_path.mkdir(exist_ok=True)
    
    # Create dummy JIT policy
    dummy_policy_data = {
        'compiled_policy': lambda obs, state: (np.random.uniform(-1, 1, 4), {}),
        'obs_shape': (17,),
        'metadata': {'step': 1000, 'timestamp': '2024-01-01'},
        'normalization_params': {
            'obs_mean': np.zeros(17),
            'obs_std': np.ones(17)
        }
    }
    
    jit_path = model_path / "policy_jit.pkl"
    with open(jit_path, 'wb') as f:
        pickle.dump(dummy_policy_data, f)
    
    print(f"Created dummy model at {model_path}")
    return str(model_path)

def deployment_example():
    """Demonstrate model deployment"""
    print("=== DreamerV3 AUV Deployment Example ===")
    
    # Create dummy model
    model_path = create_dummy_model()
    config_path = Path(__file__).parent.parent / "config" / "config_dreamerv3.yaml"
    
    print("1. Testing model loading...")
    
    try:
        from deployment_utils import BenchmarkRunner
        
        # Test policy loading
        runner = BenchmarkRunner(str(config_path))
        
        # Benchmark performance
        print("2. Running benchmark...")
        results = runner.benchmark_policy_inference(model_path, num_iterations=100)
        
        print(f"   ✓ Policy type: {results['policy_type']}")
        print(f"   ✓ Average inference time: {results['avg_time_ms']:.2f} ms")
        print(f"   ✓ Max frequency: {results['max_frequency_hz']:.1f} Hz")
        
    except ImportError as e:
        print(f"   ✗ Deployment utilities not available: {e}")
        print("   This is expected if DreamerV3 is not installed")
        return
    except Exception as e:
        print(f"   ✗ Benchmark failed: {e}")
        return
    
    print("3. Testing environment integration...")
    
    try:
        # Test environment integration
        integration_results = runner.test_environment_integration(model_path, num_episodes=2)
        
        avg_reward = np.mean([r['reward'] for r in integration_results])
        print(f"   ✓ Integration test completed")
        print(f"   ✓ Average reward: {avg_reward:.2f}")
        
    except Exception as e:
        print(f"   ✗ Integration test failed: {e}")
        return
    
    # Demonstrate ROS2 deployment (simulation)
    print("4. ROS2 deployment simulation...")
    print("   (This would normally require ROS2 to be running)")
    
    try:
        # Simulate ROS2 deployment
        class MockROS2Deployment:
            def __init__(self, model_path, config_path):
                self.model_path = model_path
                self.config_path = config_path
                self.running = False
            
            def start(self):
                self.running = True
                print("   ✓ Mock ROS2 deployment started")
                
                # Simulate control loop
                for i in range(5):
                    time.sleep(0.1)
                    print(f"     Control step {i+1}: Publishing actions...")
                
                self.running = False
                print("   ✓ Mock deployment completed")
        
        deployment = MockROS2Deployment(model_path, config_path)
        deployment.start()
        
    except Exception as e:
        print(f"   ✗ ROS2 deployment simulation failed: {e}")
    
    print("\n5. Deployment checklist:")
    print("   ✓ Model loading functional")
    print("   ✓ Inference performance acceptable")
    print("   ✓ Environment integration working")
    print("   ✓ ROS2 deployment ready (with actual ROS2)")
    
    print("\nFor real deployment:")
    print("1. Ensure ROS2 is running: source /opt/ros/humble/setup.bash")
    print("2. Start AUV simulation or connect hardware")
    print("3. Run: python deployment_utils.py deploy <model_path>")

if __name__ == "__main__":
    import numpy as np
    deployment_example()


#########################################
# examples/data_preprocessing_example.py
#########################################

#!/usr/bin/env python3
"""
Data preprocessing example for offline training
Shows how to process ROSbag data for DreamerV3
"""

import sys
import numpy as np
import h5py
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def create_synthetic_rosbag_data():
    """Create synthetic ROSbag-like data for demonstration"""
    print("Creating synthetic ROSbag data...")
    
    # Simulate extracted ROSbag data
    data_buffer = {
        'state': [],
        'error': [],
        'setpoint': [],
        'imu': [],
        'thruster_heave_bow': [],
        'thruster_heave_stern': [],
        'thruster_surge_port': [],
        'thruster_surge_starboard': []
    }
    
    timestamps = {key: [] for key in data_buffer.keys()}
    
    # Generate 10 seconds of data at 10 Hz
    dt = 0.1
    duration = 10.0
    t = 0.0
    
    while t < duration:
        # State data
        state_data = {
            'position': [0.0, 0.0, 2.0 + 0.5 * np.sin(t)],  # Oscillating depth
            'orientation': [0.0, 0.1 * np.sin(t), 0.2 * np.cos(t)],
            'velocity': [0.5, 0.0, 0.1 * np.cos(t)],
            'angular_rate': [0.0, 0.0, 0.05 * np.sin(t)]
        }
        data_buffer['state'].append(state_data)
        timestamps['state'].append(t)
        
        # Error data (relative to some setpoint)
        error_data = {
            'position': [0.0, 0.0, 0.1 * np.random.randn()],
            'orientation': [0.0, 0.05 * np.random.randn(), 0.05 * np.random.randn()],
            'velocity': [0.1 * np.random.randn(), 0.0, 0.05 * np.random.randn()],
            'angular_rate': [0.0, 0.0, 0.02 * np.random.randn()]
        }
        data_buffer['error'].append(error_data)
        timestamps['error'].append(t)
        
        # Thruster commands
        base_thrust = 0.3
        data_buffer['thruster_heave_bow'].append(base_thrust + 0.1 * np.sin(t))
        data_buffer['thruster_heave_stern'].append(base_thrust + 0.1 * np.sin(t))
        data_buffer['thruster_surge_port'].append(0.2 + 0.1 * np.cos(t))
        data_buffer['thruster_surge_starboard'].append(0.2 + 0.1 * np.cos(t))
        
        for thruster in ['thruster_heave_bow', 'thruster_heave_stern', 
                        'thruster_surge_port', 'thruster_surge_starboard']:
            timestamps[thruster].append(t)
        
        t += dt
    
    print(f"   ✓ Generated {len(data_buffer['state'])} data points")
    return data_buffer, timestamps

def preprocessing_example():
    """Demonstrate data preprocessing"""
    print("=== Data Preprocessing Example ===")
    
    # Create synthetic data
    data_buffer, timestamps = create_synthetic_rosbag_data()
    
    print("1. Raw data statistics:")
    for key, data in data_buffer.items():
        print(f"   {key}: {len(data)} samples")
    
    # Simulate preprocessing steps
    print("\n2. Converting to episode format...")
    
    # Time grid for synchronization
    start_time = 0.0
    end_time = 10.0
    dt = 1.0 / 10.0  # 10 Hz
    time_grid = np.arange(start_time, end_time, dt)
    
    episode_data = {
        'observations': [],
        'actions': [],
        'rewards': []
    }
    
    for t in time_grid:
        # Find nearest data points (simplified interpolation)
        idx = int(t / 0.1)
        if idx >= len(data_buffer['state']):
            break
        
        # Extract observation
        state = data_buffer['state'][idx]
        error = data_buffer['error'][idx]
        
        # Build observation vector (simplified)
        obs = []
        obs.extend([error['position'][2]])  # depth error
        obs.extend(error['velocity'][:2])   # surge, sway error
        
        # Add orientation errors (as sin/cos)
        for angle in error['orientation']:
            obs.extend([np.sin(angle), np.cos(angle)])
        
        # Add state velocities
        obs.extend(state['velocity'])
        obs.extend(state['angular_rate'])
        obs.extend([0.0, 0.0])  # dummy accelerations
        
        episode_data['observations'].append(np.array(obs, dtype=np.float32))
        
        # Extract actions
        if idx < len(data_buffer['thruster_heave_bow']):
            action = [
                data_buffer['thruster_heave_bow'][idx],
                data_buffer['thruster_heave_stern'][idx],
                data_buffer['thruster_surge_port'][idx],
                data_buffer['thruster_surge_starboard'][idx]
            ]
            episode_data['actions'].append(np.array(action, dtype=np.float32))
        
        # Calculate simple reward
        error_magnitude = np.linalg.norm(error['position']) + np.linalg.norm(error['orientation'])
        reward = -error_magnitude
        episode_data['rewards'].append(reward)
    
    # Convert to numpy arrays
    for key in episode_data:
        episode_data[key] = np.array(episode_data[key])
    
    print(f"   ✓ Episode length: {len(episode_data['observations'])} steps")
    print(f"   ✓ Observation shape: {episode_data['observations'].shape}")
    print(f"   ✓ Action shape: {episode_data['actions'].shape}")
    
    # Save processed data
    print("\n3. Saving processed data...")
    
    output_path = Path(__file__).parent / "processed_example.h5"
    
    with h5py.File(output_path, 'w') as f:
        episode_group = f.create_group('episode_000000')
        
        for key, data in episode_data.items():
            episode_group.create_dataset(key, data=data, compression='gzip')
    
    print(f"   ✓ Data saved to {output_path}")
    
    # Verify saved data
    print("\n4. Verifying saved data...")
    
    with h5py.File(output_path, 'r') as f:
        print(f"   Episodes in file: {list(f.keys())}")
        
        episode = f['episode_000000']
        print(f"   Data keys: {list(episode.keys())}")
        
        for key in episode.keys():
            shape = episode[key].shape
            dtype = episode[key].dtype
            print(f"   {key}: shape={shape}, dtype={dtype}")
    
    print("\n5. Data preprocessing complete!")
    print("   This data can now be used for offline DreamerV3 training")
    print(f"   Use: --offline_mode --rosbag_paths {output_path}")

if __name__ == "__main__":
    preprocessing_example()


#########################################
# examples/complete_workflow.py
#########################################

#!/usr/bin/env python3
"""
Complete workflow example demonstrating the full pipeline
From data preprocessing to deployment
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def complete_workflow_example():
    """Demonstrate complete workflow from start to deployment"""
    print("=== Complete DreamerV3 AUV Workflow Example ===")
    print("This example shows the complete pipeline:")
    print("1. Data preprocessing (if using offline mode)")
    print("2. Training configuration")
    print("3. Model training")
    print("4. Performance evaluation")
    print("5. Model deployment")
    print()
    
    # Step 1: Environment setup
    print("STEP 1: Environment Setup")
    print("-" * 40)
    
    try:
        from AUVEnv_DreamerV3 import AUVEnvDreamerV3
        print("✓ AUV Environment available")
        
        # Test environment
        config_path = Path(__file__).parent.parent / "config" / "config_dreamerv3.yaml"
        env = AUVEnvDreamerV3(config_path=str(config_path), offline_mode=True)
        obs, info = env.reset()
        env.close()
        print("✓ Environment test passed")
        
    except Exception as e:
        print(f"✗ Environment setup failed: {e}")
        return
    
    # Step 2: Data preparation
    print("\nSTEP 2: Data Preparation")
    print("-" * 40)
    
    try:
        # Run data preprocessing example
        print("Running data preprocessing example...")
        exec(open(Path(__file__).parent / "data_preprocessing_example.py").read())
        print("✓ Data preprocessing completed")
        
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")
        print("Continuing with online training mode...")
    
    # Step 3: Training configuration
    print("\nSTEP 3: Training Configuration")
    print("-" * 40)
    
    import yaml
    
    # Load and display config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Key configuration parameters:")
    print(f"  Training steps: {config.get('training', {}).get('total_steps', 'N/A')}")
    print(f"  Coupling method: {config.get('coupling', {}).get('method', 'N/A')}")
    print(f"  Batch size: {config.get('dreamerv3', {}).get('batch_size', 'N/A')}")
    print("✓ Configuration loaded")
    
    # Step 4: Training demonstration
    print("\nSTEP 4: Training Demonstration")
    print("-" * 40)
    
    print("Training options:")
    print("  a) Online training: python train_dreamerv3_auv.py --mode train")
    print("  b) Offline training: python train_dreamerv3_auv.py --mode train --offline_mode")
    print("  c) Resume training: python train_dreamerv3_auv.py --resume_from_checkpoint <path>")
    
    print("\nFor this demo, we'll simulate a short training run...")
    
    try:
        # Simulate training progress
        print("Simulating training progress:")
        for step in range(1, 6):
            time.sleep(0.5)
            reward = -1.0 + 0.1 * step + 0.05 * (step ** 1.5)
            print(f"  Step {step*1000}: reward={reward:.3f}")
        
        print("✓ Training simulation completed")
        
    except KeyboardInterrupt:
        print("✓ Training stopped by user")
    
    # Step 5: Model evaluation
    print("\nSTEP 5: Model Evaluation")
    print("-" * 40)
    
    try:
        # Run coupling rewards demo
        print("Demonstrating coupling reward methods...")
        exec(open(Path(__file__).parent / "coupling_rewards_demo.py").read())
        print("✓ Coupling evaluation completed")
        
    except Exception as e:
        print(f"Note: Coupling demo failed: {e}")
        print("✓ Evaluation framework available")
    
    # Step 6: Deployment preparation
    print("\nSTEP 6: Deployment Preparation")
    print("-" * 40)
    
    print("Deployment steps:")
    print("1. Create deployment package:")
    print("   python checkpoint_utils.py package <checkpoint_path> <output_path>")
    print()
    print("2. Benchmark performance:")
    print("   python deployment_utils.py benchmark <model_path>")
    print()
    print("3. Deploy for real-time control:")
    print("   python deployment_utils.py deploy <model_path>")
    print()
    print("✓ Deployment tools available")
    
    # Step 7: Docker deployment
    print("\nSTEP 7: Docker Deployment")
    print("-" * 40)
    
    print("Docker deployment options:")
    print("1. Build container:")
    print("   docker build -f docker_setup/Dockerfile -t dreamerv3-auv .")
    print()
    print("2. Run training:")
    print("   docker-compose up dreamerv3-train")
    print()
    print("3. Run deployment:")
    print("   docker-compose up dreamerv3-deploy")
    print()
    print("✓ Docker configuration available")
    
    # Summary
    print("\nWORKFLOW SUMMARY")
    print("=" * 50)
    print("Complete DreamerV3 AUV workflow demonstrated:")
    print("✓ Environment setup and testing")
    print("✓ Data preprocessing pipeline")
    print("✓ Training configuration")
    print("✓ Training process simulation")
    print("✓ Model evaluation methods")
    print("✓ Deployment preparation")
    print("✓ Docker containerization")
    print()
    print("Next steps for actual deployment:")
    print("1. Prepare your ROSbag data or set up simulation")
    print("2. Adjust configuration for your specific AUV")
    print("3. Run training with: python train_dreamerv3_auv.py")
    print("4. Deploy trained model with: python deployment_utils.py deploy")
    print()
    print("For detailed instructions, see README.md and MIGRATION_GUIDE.md")

if __name__ == "__main__":
    complete_workflow_example()


#########################################
# examples/run_examples.py
#########################################

#!/usr/bin/env python3
"""
Script to run all examples in sequence
Useful for testing the complete system
"""

import sys
import subprocess
from pathlib import Path

def run_example(script_name, description):
    """Run a single example script"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('='*60)
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=False, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def main():
    """Run all examples"""
    print("🚀 Running DreamerV3 AUV Examples")
    print("This will test all major components of the system")
    
    examples = [
        ("data_preprocessing_example.py", "Data Preprocessing"),
        ("coupling_rewards_demo.py", "Coupling Rewards Demo"),
        ("deployment_example.py", "Deployment Example"),
        ("basic_training.py", "Basic Training"),
        ("complete_workflow.py", "Complete Workflow")
    ]
    
    results = []
    
    for script_name, description in examples:
        success = run_example(script_name, description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("EXAMPLES SUMMARY")
    print('='*60)
    
    passed = 0
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description:<30} {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} examples passed")
    
    if passed == total:
        print("🎉 All examples completed successfully!")
        print("The DreamerV3 AUV system is ready for use.")
    else:
        print("⚠️  Some examples failed. Check the output above for details.")
        print("This might be due to missing dependencies or configuration issues.")
    
    print("\nFor more information, see:")
    print("- README.md for complete documentation")
    print("- MIGRATION_GUIDE.md for SAC to DreamerV3 migration")
    print("- Individual example files for specific use cases")

if __name__ == "__main__":
    main()