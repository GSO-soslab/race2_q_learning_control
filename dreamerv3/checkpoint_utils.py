#!/usr/bin/env python3

import os
import sys
import json
import pickle
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml

# JAX and DreamerV3 imports
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import haiku as hk
    import optax
    JAX_AVAILABLE = True
except ImportError:
    print("Warning: JAX not available")
    JAX_AVAILABLE = False

# Add DreamerV3 to path
DREAMERV3_PATH = os.path.join(os.path.dirname(__file__), 'dreamerv3')
if os.path.exists(DREAMERV3_PATH):
    sys.path.insert(0, DREAMERV3_PATH)

try:
    import dreamerv3
    from dreamerv3 import embodied
    DREAMERV3_AVAILABLE = True
except ImportError:
    DREAMERV3_AVAILABLE = False

class CheckpointManager:
    """
    Comprehensive checkpoint management for DreamerV3 models
    Handles saving, loading, conversion, and deployment utilities
    """
    
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        self.deployment_config = self.config.get('deployment', {})
        
    def save_checkpoint_with_metadata(self, agent, step, log_dir, metrics=None):
        """Save checkpoint with comprehensive metadata"""
        checkpoint_dir = Path(log_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save agent checkpoint - Fixed for DreamerV3 API
        try:
            if hasattr(agent, 'save'):
                try:
                    agent.save(str(checkpoint_dir))
                    print(f"Agent saved to {checkpoint_dir}")
                except TypeError:
                    # Try parameterless save
                    agent.save()
                    print(f"Agent saved (parameterless)")
        except Exception as e:
            print(f"Error saving agent: {e}")
        
        # Create metadata
        metadata = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'jax_version': jax.__version__ if JAX_AVAILABLE else 'N/A',
            'python_version': sys.version,
            'checkpoint_path': str(checkpoint_dir),
        }
        
        if metrics:
            metadata['training_metrics'] = metrics
        
        # Save metadata
        metadata_file = checkpoint_dir / f"metadata_step_{step}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Create symlink to latest checkpoint
        latest_link = checkpoint_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        try:
            latest_link.symlink_to(f"step_{step}")
        except:
            # Fallback for systems without symlink support
            pass
        
        print(f"Checkpoint saved: step {step} at {checkpoint_dir}")
        
        return str(checkpoint_dir)
    
    def load_checkpoint_with_validation(self, checkpoint_path, agent=None):
        """Load checkpoint with validation and compatibility checks"""
        checkpoint_path = Path(checkpoint_path)
        
        # Find metadata file
        metadata_files = list(checkpoint_path.glob("metadata_step_*.json"))
        if not metadata_files:
            print("Warning: No metadata found for checkpoint")
            metadata = {}
        else:
            # Use latest metadata
            metadata_file = sorted(metadata_files)[-1]
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Validate compatibility
        self._validate_checkpoint_compatibility(metadata)
        
        # Load checkpoint - Fixed for DreamerV3 API
        if agent and DREAMERV3_AVAILABLE:
            try:
                if hasattr(agent, 'load'):
                    try:
                        agent.load(str(checkpoint_path))
                        print(f"Loaded checkpoint from {checkpoint_path}")
                    except TypeError:
                        # Try with directory only
                        agent.load(str(checkpoint_path.parent))
                        print(f"Loaded checkpoint from {checkpoint_path.parent}")
                else:
                    print("Agent has no load method")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
            
        return metadata
    
    def _validate_checkpoint_compatibility(self, metadata):
        """Validate checkpoint compatibility with current environment"""
        warnings = []
        
        # Check JAX version compatibility
        if 'jax_version' in metadata and JAX_AVAILABLE:
            saved_version = metadata['jax_version']
            current_version = jax.__version__
            if saved_version != current_version:
                warnings.append(f"JAX version mismatch: saved={saved_version}, current={current_version}")
        
        # Check config compatibility
        if 'config' in metadata and self.config:
            saved_config = metadata['config']
            
            # Check critical parameters
            critical_params = [
                ('environment', 'thruster_size'),
                ('environment', 'servo_joints_size'),
                ('dreamerv3', 'dyn_stoch'),
                ('dreamerv3', 'dyn_deter')
            ]
            
            for param_path in critical_params:
                saved_val = self._get_nested_config_value(saved_config, param_path)
                current_val = self._get_nested_config_value(self.config, param_path)
                
                if saved_val != current_val:
                    warnings.append(f"Config mismatch {'.'.join(param_path)}: saved={saved_val}, current={current_val}")
        
        # Print warnings
        if warnings:
            print("Checkpoint compatibility warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        return len(warnings) == 0
    
    def _get_nested_config_value(self, config, path):
        """Get nested configuration value"""
        current = config
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def convert_checkpoint_for_deployment(self, checkpoint_path, output_path, format='jit'):
        """Convert checkpoint for deployment in different formats"""
        if not DREAMERV3_AVAILABLE or not JAX_AVAILABLE:
            raise ImportError("DreamerV3 and JAX required for checkpoint conversion")
        
        checkpoint_path = Path(checkpoint_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint metadata
        metadata = self.load_checkpoint_with_validation(checkpoint_path)
        
        # Initialize agent for conversion - Fixed for correct API
        config = self._create_agent_config(metadata)
        obs_space, act_space = self._get_spaces_from_config()
        
        # Create agent with correct signature
        agent = embodied.Agent(obs_space, act_space, config)
        
        # Load the checkpoint
        try:
            agent.load(str(checkpoint_path))
        except Exception as e:
            print(f"Error loading checkpoint for conversion: {e}")
            return
        
        if format == 'jit':
            self._convert_to_jit(agent, output_path, metadata)
        elif format == 'onnx':
            self._convert_to_onnx(agent, output_path, metadata)
        elif format == 'tflite':
            self._convert_to_tflite(agent, output_path, metadata)
        else:
            raise ValueError(f"Unsupported conversion format: {format}")
        
        print(f"Checkpoint converted to {format} format at {output_path}")
    
    def _create_agent_config(self, metadata):
        """Create agent configuration from metadata"""
        config = {
            'logdir': '/tmp/dreamerv3_conversion',
            'seed': 42,
            'precision': 16,
            'task': 'auv_control',
            'env_amount': 1,
            'env_parallel': False,
            'batch_size': 16,
            'batch_length': 64,
        }
        
        # Override with saved configuration if available
        if 'config' in metadata and 'dreamerv3' in metadata['config']:
            saved_config = metadata['config']['dreamerv3']
            config.update(saved_config)
        
        return config
    
    def _get_spaces_from_config(self):
        """Get observation and action spaces from config"""
        try:
            from gymnasium.spaces import Dict, Box
            import numpy as np
            
            # Default spaces for AUV
            obs_space = Dict({
                'vector': Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
            })
            
            thruster_size = self.config.get('environment', {}).get('thruster_size', 4)
            servo_size = self.config.get('environment', {}).get('servo_joints_size', 0)
            action_dim = thruster_size + servo_size
            
            act_space = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
            
            return obs_space, act_space
            
        except ImportError:
            # Fallback if gymnasium not available
            return {'vector': (17,)}, 4
    
    def _convert_to_jit(self, agent, output_path, metadata):
        """Convert to JAX JIT compiled format"""
        # Create JIT compiled policy function
        def policy_fn(obs, state):
            try:
                if hasattr(agent, 'policy'):
                    return agent.policy(obs, state, mode='eval')
                elif hasattr(agent, 'act'):
                    return agent.act(obs), state
                else:
                    # Fallback
                    action_dim = 4  # Default AUV action dimension
                    return jnp.zeros(action_dim), state
            except Exception as e:
                print(f"Error in policy function: {e}")
                action_dim = 4
                return jnp.zeros(action_dim), state
        
        # Create compiled policy with error handling
        try:
            compiled_policy = jax.jit(policy_fn)
        except Exception as e:
            print(f"JIT compilation failed: {e}")
            compiled_policy = policy_fn  # Use non-JIT version
        
        # Save JIT compiled function
        jit_path = output_path / "policy_jit.pkl"
        
        # Create example inputs for compilation
        obs_shape = self.config.get('environment', {}).get('obs_shape', (17,))
        example_obs = {'vector': jnp.zeros(obs_shape, dtype=jnp.float32)}
        example_state = {}
        
        # Save compiled policy and metadata
        deployment_data = {
            'compiled_policy': compiled_policy,
            'obs_shape': obs_shape,
            'metadata': metadata,
            'normalization_params': getattr(agent, 'normalization_params', None)
        }
        
        with open(jit_path, 'wb') as f:
            pickle.dump(deployment_data, f)
        
        # Create deployment wrapper script
        self._create_jit_deployment_script(output_path, obs_shape)
    
    def _convert_to_onnx(self, agent, output_path, metadata):
        """Convert to ONNX format (placeholder - requires additional setup)"""
        print("ONNX conversion not implemented - requires jax2onnx or similar")
        # This would require additional dependencies like jax2onnx
        # and is more complex due to JAX's functional nature
        pass
    
    def _convert_to_tflite(self, agent, output_path, metadata):
        """Convert to TensorFlow Lite format (placeholder)"""
        print("TFLite conversion not implemented - requires jax2tf")
        # This would require jax2tf conversion first
        pass
    
    def _create_jit_deployment_script(self, output_path, obs_shape):
        """Create a deployment script for JIT compiled model"""
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated deployment script for JIT compiled DreamerV3 policy
"""

import pickle
import numpy as np
import jax.numpy as jnp

class JITDreamerV3Policy:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.compiled_policy = data['compiled_policy']
        self.obs_shape = data['obs_shape']
        self.metadata = data['metadata']
        self.normalization_params = data.get('normalization_params')
        
        # Initialize policy state
        self.policy_state = {{}}
    
    def predict(self, observation):
        """
        Predict action for given observation
        
        Args:
            observation: dict with 'vector' key containing observation array
        
        Returns:
            action: numpy array of actions
        """
        # Ensure observation is in correct format
        if isinstance(observation, np.ndarray):
            obs = {{'vector': jnp.array(observation, dtype=jnp.float32)}}
        else:
            obs = {{k: jnp.array(v, dtype=jnp.float32) for k, v in observation.items()}}
        
        # Apply normalization if available
        if self.normalization_params:
            obs_mean = self.normalization_params.get('obs_mean')
            obs_std = self.normalization_params.get('obs_std')
            if obs_mean is not None and obs_std is not None:
                obs['vector'] = (obs['vector'] - obs_mean) / obs_std
        
        # Get action
        try:
            action, self.policy_state = self.compiled_policy(obs, self.policy_state)
            return np.array(action)
        except Exception as e:
            print(f"Policy inference error: {{e}}")
            # Return zero actions as fallback
            return np.zeros({obs_shape[0] if len(obs_shape) == 1 else 4}, dtype=np.float32)
    
    def reset(self):
        """Reset policy state"""
        self.policy_state = {{}}

# Example usage
if __name__ == "__main__":
    # Load policy
    policy = JITDreamerV3Policy("policy_jit.pkl")
    
    # Example observation
    obs = np.zeros({obs_shape}, dtype=np.float32)
    
    # Get action
    action = policy.predict(obs)
    print(f"Action: {{action}}")
'''
        
        script_path = output_path / "deploy_policy.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        try:
            script_path.chmod(0o755)
        except:
            pass  # Skip if chmod not supported
    
    def migrate_sac_checkpoint(self, sac_checkpoint_path, output_path):
        """Migrate SAC checkpoint data to DreamerV3 format (partial migration)"""
        print("Migrating SAC checkpoint data...")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # This is a simplified migration - only transfers what's possible
        migration_data = {
            'timestamp': datetime.now().isoformat(),
            'source_format': 'SAC',
            'target_format': 'DreamerV3',
            'migration_notes': 'Partial migration - only hyperparameters and metadata transferred'
        }
        
        try:
            # Try to load SAC checkpoint (if using stable-baselines3 format)
            if sac_checkpoint_path.endswith('.zip'):
                # This would require stable-baselines3 to be available
                print("SAC checkpoint detected - extracting metadata only")
                migration_data['sac_checkpoint_path'] = str(sac_checkpoint_path)
                migration_data['transferable_components'] = [
                    'episode_count',
                    'total_timesteps', 
                    'config_parameters'
                ]
                migration_data['non_transferable_components'] = [
                    'actor_network_weights',
                    'critic_network_weights',
                    'replay_buffer_data'  # Could be used for offline training
                ]
            
        except Exception as e:
            print(f"Error reading SAC checkpoint: {e}")
            migration_data['error'] = str(e)
        
        # Save migration metadata
        migration_file = output_path / "sac_migration_metadata.json"
        with open(migration_file, 'w') as f:
            json.dump(migration_data, f, indent=2)
        
        print(f"Migration metadata saved to {migration_file}")
        print("Note: Complete model migration requires retraining with DreamerV3")
        print("Consider using SAC replay buffer data for DreamerV3 offline pre-training")
    
    def create_deployment_package(self, checkpoint_path, output_path, include_config=True):
        """Create a complete deployment package"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Creating deployment package...")
        
        # Copy checkpoint files
        checkpoint_path = Path(checkpoint_path)
        checkpoint_dest = output_path / "checkpoint"
        if checkpoint_dest.exists():
            shutil.rmtree(checkpoint_dest)
        shutil.copytree(checkpoint_path, checkpoint_dest)
        
        # Copy configuration
        if include_config and self.config:
            config_dest = output_path / "config.yaml"
            with open(config_dest, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        
        # Create deployment instructions
        instructions = self._create_deployment_instructions()
        instructions_file = output_path / "DEPLOYMENT_INSTRUCTIONS.md"
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        # Create requirements file
        requirements = self._create_requirements_file()
        requirements_file = output_path / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write(requirements)
        
        # Convert to optimized format if requested
        if self.deployment_config.get('export_jit', True):
            jit_output = output_path / "optimized"
            try:
                self.convert_checkpoint_for_deployment(
                    checkpoint_path, jit_output, format='jit'
                )
            except Exception as e:
                print(f"JIT conversion failed: {e}")
        
        print(f"Deployment package created at {output_path}")
        
        return str(output_path)
    
    def _create_deployment_instructions(self):
        """Create deployment instructions"""
        return '''# DreamerV3 AUV Model Deployment Instructions

## Prerequisites
1. Install required dependencies: `pip install -r requirements.txt`
2. Ensure ROS2 environment is properly set up
3. Verify AUV simulation/hardware is accessible

## Quick Start
1. Use the optimized JIT model for best performance:
   ```python
   from optimized.deploy_policy import JITDreamerV3Policy
   policy = JITDreamerV3Policy("optimized/policy_jit.pkl")
   ```

2. For full model loading:
   ```python
   import dreamerv3
   from dreamerv3 import embodied
   
   # Create agent config and spaces
   config = {...}  # Your config
   obs_space = {...}  # Your observation space
   act_space = {...}  # Your action space
   
   agent = embodied.Agent(obs_space, act_space, config)
   agent.load("checkpoint")
   ```

## Integration with AUV Environment
1. Replace the policy in your AUV control loop
2. Ensure observation format matches training format
3. Apply any normalization parameters if used during training

## Performance Optimization
- Use JIT compiled model for fastest inference
- Consider batch processing for multiple predictions
- Monitor GPU memory usage if using GPU inference

## Safety Considerations
- Implement safety bounds checking on actions
- Monitor for unusual behavior during deployment
- Have emergency stop procedures in place

## Troubleshooting
- Check JAX/DreamerV3 version compatibility
- Verify observation shapes match training configuration
- Ensure ROS2 topics are publishing correctly
'''
    
    def _create_requirements_file(self):
        """Create requirements.txt for deployment"""
        return '''# DreamerV3 AUV Deployment Requirements
jax[cuda]>=0.4.0
dm-haiku>=0.0.9
optax>=0.1.4
numpy>=1.21.0
pyyaml>=6.0
matplotlib>=3.5.0

# ROS2 dependencies (install via apt or conda)
# rclpy
# rosbag2_py

# Optional: for optimized deployment
# onnxruntime
# tensorflow-lite

# AUV-specific dependencies
# mvp_msgs (custom ROS2 messages)
'''

def analyze_checkpoint_directory(checkpoint_dir):
    """Analyze checkpoint directory and provide summary"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    print(f"Analyzing checkpoint directory: {checkpoint_path}")
    print("=" * 50)
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_path.glob("step_*"))
    metadata_files = list(checkpoint_path.glob("metadata_step_*.json"))
    
    print(f"Found {len(checkpoint_files)} checkpoint steps")
    print(f"Found {len(metadata_files)} metadata files")
    
    if checkpoint_files:
        steps = []
        for f in checkpoint_files:
            try:
                step_num = int(f.name.split('_')[1])
                steps.append(step_num)
            except (ValueError, IndexError):
                continue
        
        if steps:
            steps.sort()
            print(f"Steps available: {steps[0]} to {steps[-1]}")
            print(f"Latest step: {steps[-1]}")
    
    # Analyze metadata if available
    if metadata_files:
        latest_metadata_file = sorted(metadata_files)[-1]
        try:
            with open(latest_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print("\nLatest checkpoint metadata:")
            print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
            print(f"  Step: {metadata.get('step', 'N/A')}")
            print(f"  JAX version: {metadata.get('jax_version', 'N/A')}")
            
            if 'config' in metadata:
                config = metadata['config']
                if 'dreamerv3' in config:
                    d3_config = config['dreamerv3']
                    print(f"  Model architecture:")
                    print(f"    Stochastic size: {d3_config.get('dyn_stoch', 'N/A')}")
                    print(f"    Deterministic size: {d3_config.get('dyn_deter', 'N/A')}")
                    print(f"    Batch size: {d3_config.get('batch_size', 'N/A')}")
            
            if 'training_metrics' in metadata:
                metrics = metadata['training_metrics']
                print(f"  Training metrics: {list(metrics.keys())}")
        
        except Exception as e:
            print(f"Error reading metadata: {e}")
    
    # Check disk usage
    total_size = sum(f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file())
    print(f"\nTotal disk usage: {total_size / (1024**3):.2f} GB")

def main():
    """Main function for checkpoint utilities"""
    parser = argparse.ArgumentParser(description='DreamerV3 Checkpoint Management Utilities')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze checkpoint directory')
    analyze_parser.add_argument('checkpoint_dir', type=str, help='Checkpoint directory path')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert checkpoint for deployment')
    convert_parser.add_argument('checkpoint_path', type=str, help='Checkpoint path')
    convert_parser.add_argument('output_path', type=str, help='Output path')
    convert_parser.add_argument('--format', type=str, default='jit', 
                               choices=['jit', 'onnx', 'tflite'], help='Conversion format')
    convert_parser.add_argument('--config', type=str, help='Config file path')
    
    # Package command
    package_parser = subparsers.add_parser('package', help='Create deployment package')
    package_parser.add_argument('checkpoint_path', type=str, help='Checkpoint path')
    package_parser.add_argument('output_path', type=str, help='Output path')
    package_parser.add_argument('--config', type=str, help='Config file path')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate SAC checkpoint')
    migrate_parser.add_argument('sac_checkpoint', type=str, help='SAC checkpoint path')
    migrate_parser.add_argument('output_path', type=str, help='Output path')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize checkpoint manager
    config_path = getattr(args, 'config', None)
    manager = CheckpointManager(config_path)
    
    try:
        if args.command == 'analyze':
            analyze_checkpoint_directory(args.checkpoint_dir)
            
        elif args.command == 'convert':
            manager.convert_checkpoint_for_deployment(
                args.checkpoint_path, args.output_path, args.format
            )
            
        elif args.command == 'package':
            manager.create_deployment_package(
                args.checkpoint_path, args.output_path
            )
            
        elif args.command == 'migrate':
            manager.migrate_sac_checkpoint(
                args.sac_checkpoint, args.output_path
            )
        
    except Exception as e:
        print(f"Error executing command: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()