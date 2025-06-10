#!/usr/bin/env python3

import os
import sys
import argparse
import yaml
import time
import numpy as np
from datetime import datetime
import threading
import pickle
import json
from pathlib import Path
import subprocess
import tempfile

# Add the path manually
DREAMERV3_REPO_PATH = os.path.expanduser('~/dreamerv3')
if os.path.exists(DREAMERV3_REPO_PATH):
    sys.path.insert(0, DREAMERV3_REPO_PATH)

try:
    import dreamerv3
    import embodied
    DREAMERV3_AVAILABLE = True
    print("DreamerV3 and embodied successfully imported!")
except ImportError as e:
    print(f"Warning: DreamerV3 or embodied not found: {e}")
    DREAMERV3_AVAILABLE = False

# Import custom environment
from AUVEnv_DreamerV3 import AUVEnvDreamerV3

class DreamerV3ProperTrainer:
    """
    Proper DreamerV3 trainer that uses the official training pipeline
    """
    
    def __init__(self, config_path, args):
        self.args = args
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize environment for data preparation
        self.setup_environment()
        
        # DreamerV3 specific setup
        self.dreamerv3_config_path = None
        self.dreamerv3_data_path = None
        
    def setup_logging(self):
        """Setup logging directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.args.resume_from_checkpoint:
            checkpoint_name = os.path.splitext(os.path.basename(self.args.resume_from_checkpoint))[0]
            self.log_dir = os.path.join("logs", f"dreamerv3_resumed_{checkpoint_name}_{timestamp}")
        else:
            mode_suffix = "offline" if self.args.offline_mode else "online"
            self.log_dir = os.path.join("logs", f"dreamerv3_{mode_suffix}_{timestamp}")
        
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Logging to: {self.log_dir}")
        
        # Save config
        config_save_path = os.path.join(self.log_dir, "config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def setup_environment(self):
        """Setup the AUV environment for data preparation"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config_dreamerv3.yaml')
        
        self.env = AUVEnvDreamerV3(
            config_path=config_path, 
            offline_mode=self.args.offline_mode
        )
        
        print(f"Environment initialized in {'offline' if self.args.offline_mode else 'online'} mode")
        print(f"Action space: {self.env.action_space}")
        print(f"Observation space: {self.env.observation_space}")

    def create_dreamerv3_config_file(self):
        """Create a proper DreamerV3 config file"""
        dreamerv3_settings = self.config.get('dreamerv3', {})
        
        # Create config in the format expected by dreamerv3/main.py
        dreamerv3_config = {
            # Basic settings
            'task': 'auv_control',
            'logdir': self.log_dir,
            'seed': self.config['others']['random_seed'],
            
            # Environment
            'env': {
                'name': 'auv_control',
                'amount': 1,
                'parallel': False,
                'reset': True,
                'restart': True,
            },
            
            # Training
            'batch_size': dreamerv3_settings.get('batch_size', 16),
            'batch_length': dreamerv3_settings.get('batch_length', 64),
            'replay_size': int(dreamerv3_settings.get('replay_size', 2e6)),
            'train_ratio': dreamerv3_settings.get('train_ratio', 32),
            'pretrain': 100,
            'train_steps': self.args.timesteps or 1000000,
            
            # Model
            'dyn_stoch': dreamerv3_settings.get('dyn_stoch', 32),
            'dyn_deter': dreamerv3_settings.get('dyn_deter', 512),
            'dyn_hidden': dreamerv3_settings.get('dyn_hidden', 512),
            'dyn_rec_depth': dreamerv3_settings.get('dyn_rec_depth', 1),
            
            # Optimizer
            'model_lr': dreamerv3_settings.get('model_lr', 1e-4),
            'actor_lr': dreamerv3_settings.get('actor_lr', 8e-5),
            'critic_lr': dreamerv3_settings.get('critic_lr', 2e-4),
            'grad_clip': 1000.0,
            'weight_decay': 0.0,
            
            # Behavior
            'discount': 0.99,
            'lambda_': 0.95,
            'horizon': 15,
            'actor_grad': 'dynamics',
            'actor_dist': 'normal',
            'actor_entropy': 1e-4,
            'baseline': 'critic',
            
            # Exploration
            'expl_amount': dreamerv3_settings.get('exploration_noise', 0.3),
            'expl_decay': dreamerv3_settings.get('exploration_decay', 0.0),
            'expl_min': dreamerv3_settings.get('exploration_min', 0.1),
            'expl_behavior': 'epsilon_greedy',
            
            # JAX
            'jax': {
                'platform': 'gpu',
                'precision': 'float32',
                'prealloc': False,
            },
            
            # Logging
            'run': {
                'train_ratio': dreamerv3_settings.get('train_ratio', 32),
                'log_every': 1000,
                'eval_every': 10000,
                'save_every': 10000,
                'eval_episodes': 10,
                'eval_samples': 1,
            }
        }
        
        # Save config file
        self.dreamerv3_config_path = os.path.join(self.log_dir, 'dreamerv3_config.yaml')
        with open(self.dreamerv3_config_path, 'w') as f:
            yaml.dump(dreamerv3_config, f, default_flow_style=False)
        
        print(f"Created DreamerV3 config at: {self.dreamerv3_config_path}")
        return self.dreamerv3_config_path

    def prepare_offline_data(self):
        """Prepare offline data in DreamerV3 format"""
        if not hasattr(self.env, 'data_loader') or not self.env.data_loader.episodes:
            print("No offline data available. Please provide ROSbag data paths in config.")
            return None
        
        print(f"Preparing {len(self.env.data_loader.episodes)} episodes for DreamerV3...")
        
        # Create data directory
        data_dir = os.path.join(self.log_dir, 'offline_data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Convert episodes to DreamerV3 format and save as NPZ files
        episode_files = []
        
        for episode_idx, episode_data in enumerate(self.env.data_loader.episodes):
            try:
                # Extract episode data
                observations = episode_data['observations']
                actions = episode_data['actions']
                rewards = episode_data['rewards']
                terminals = episode_data['terminals']
                
                episode_length = len(observations)
                if episode_length < 2:
                    continue
                
                # Ensure proper shapes
                if observations.ndim == 1:
                    observations = observations.reshape(-1, 1)
                if actions.ndim == 1:
                    actions = actions.reshape(-1, 1)
                
                # Create episode data in DreamerV3 format
                episode_dict = {
                    'observation': observations.astype(np.float32),
                    'action': actions.astype(np.float32),
                    'reward': rewards.astype(np.float32),
                    'discount': np.where(terminals, 0.0, 0.99).astype(np.float32),
                    'is_first': np.zeros(episode_length, dtype=bool),
                    'is_last': terminals.astype(bool),
                    'is_terminal': terminals.astype(bool),
                }
                
                # Mark first step
                episode_dict['is_first'][0] = True
                
                # Save episode
                episode_file = os.path.join(data_dir, f'episode_{episode_idx:06d}.npz')
                np.savez_compressed(episode_file, **episode_dict)
                episode_files.append(episode_file)
                
                if (episode_idx + 1) % 10 == 0:
                    print(f"Processed {episode_idx + 1}/{len(self.env.data_loader.episodes)} episodes")
                    
            except Exception as e:
                print(f"Error processing episode {episode_idx}: {e}")
                continue
        
        print(f"Prepared {len(episode_files)} episodes for offline training")
        
        # Create episode list file
        episode_list_file = os.path.join(data_dir, 'episodes.txt')
        with open(episode_list_file, 'w') as f:
            for episode_file in episode_files:
                f.write(f"{episode_file}\n")
        
        self.dreamerv3_data_path = data_dir
        return data_dir

    def create_custom_environment_wrapper(self):
        """Create a custom environment wrapper for DreamerV3"""
        
        # Create a wrapper that DreamerV3 can use
        wrapper_code = f'''
import numpy as np
import os
import sys
sys.path.append("{os.path.dirname(os.path.abspath(__file__))}")

from AUVEnv_DreamerV3 import AUVEnvDreamerV3

class AUVDreamerV3Wrapper:
    def __init__(self, **kwargs):
        config_path = "{os.path.join(os.path.dirname(__file__), 'config', 'config_dreamerv3.yaml')}"
        
        # Check if we should use offline mode
        offline_data_path = os.environ.get('DREAMERV3_OFFLINE_DATA', None)
        offline_mode = offline_data_path is not None
        
        self.env = AUVEnvDreamerV3(config_path=config_path, offline_mode=offline_mode)
        self._obs_space = self.env.observation_space
        self._act_space = self.env.action_space
        
        # For offline mode, we may need to handle data differently
        self.offline_data_path = offline_data_path
        self.offline_mode = offline_mode
        
        print(f"AUV Environment Wrapper initialized in {{'offline' if offline_mode else 'online'}} mode")
        if offline_mode:
            print(f"Offline data path: {{offline_data_path}}")
        
    @property
    def obs_space(self):
        return self._obs_space
    
    @property
    def act_space(self):
        return self._act_space
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # DreamerV3 expects specific format
        return {{
            'observation': obs,
            'reward': float(reward),
            'is_first': False,
            'is_last': done,
            'is_terminal': terminated
        }}
    
    def reset(self):
        obs, info = self.env.reset()
        
        # DreamerV3 expects specific format
        return {{
            'observation': obs,
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False
        }}
    
    def close(self):
        self.env.close()

# Register the environment
def make_env(**kwargs):
    return AUVDreamerV3Wrapper(**kwargs)

# Also create a function that DreamerV3 can discover
def auv_control(**kwargs):
    """Environment factory function for DreamerV3"""
    return AUVDreamerV3Wrapper(**kwargs)
'''
        
        # Save wrapper
        wrapper_file = os.path.join(self.log_dir, 'auv_env_wrapper.py')
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_code)
        
        # Also create an __init__.py file to make it a package
        init_file = os.path.join(self.log_dir, '__init__.py')
        with open(init_file, 'w') as f:
            f.write('# DreamerV3 AUV Environment Package\n')
        
        print(f"Created environment wrapper at: {wrapper_file}")
        return wrapper_file

    def run_dreamerv3_training(self):
        """Run DreamerV3 training using the official script"""
        
        if not DREAMERV3_AVAILABLE:
            raise ImportError("DreamerV3 not available")
        
        # Check if the official main.py exists
        main_script = os.path.join(DREAMERV3_REPO_PATH, 'dreamerv3', 'main.py')
        if not os.path.exists(main_script):
            raise FileNotFoundError(f"DreamerV3 main.py not found at {main_script}")
        
        # Create config file
        config_file = self.create_dreamerv3_config_file()
        
        # Prepare data if offline mode
        if self.args.offline_mode:
            data_dir = self.prepare_offline_data()
            if data_dir is None:
                return
        
        # Create environment wrapper
        env_wrapper = self.create_custom_environment_wrapper()
        
        # Build command line arguments - using correct flag format
        cmd = [
            sys.executable,
            main_script,
            f'--logdir={self.log_dir}',
            f'--configs=debug',
            f'--task=auv_control',
        ]
        
        # Add config overrides with correct flag format (dotted notation)
        dreamerv3_settings = self.config.get('dreamerv3', {})
        
        # Basic training settings from the log
        cmd.extend([
            f'--batch_size={dreamerv3_settings.get("batch_size", 16)}',
            f'--batch_length={dreamerv3_settings.get("batch_length", 50)}',
            f'--run.train_ratio={dreamerv3_settings.get("train_ratio", 32)}',
        ])
        
        # Learning rates (FIXED: using the correct nested keys like model_opt.lr)
        cmd.extend([
            f'--model_opt.lr={dreamerv3_settings.get("model_lr", 0.0002)}',
            f'--actor_opt.lr={dreamerv3_settings.get("actor_lr", 0.0001)}',
            f'--critic_opt.lr={dreamerv3_settings.get("critic_lr", 0.0003)}',
        ])
        
        # Model settings
        cmd.extend([
            f'--dyn_stoch={dreamerv3_settings.get("dyn_stoch", 32)}',
            f'--dyn_deter={dreamerv3_settings.get("dyn_deter", 512)}',
            f'--dyn_hidden={dreamerv3_settings.get("dyn_hidden", 512)}',
        ])
        
        # JAX and run settings
        cmd.extend([
            '--jax.platform=gpu',
            f'--run.log_every=1000',
            f'--run.eval_every=10000',
            f'--run.save_every=10000',
        ])
        
        # Add timesteps if specified
        if self.args.timesteps:
            cmd.append(f'--run.steps={self.args.timesteps}')
        
        if self.args.offline_mode:
            print("Note: Offline data will be handled through environment variables.")
            print(f"Environment variable DREAMERV3_OFFLINE_DATA set to: {self.dreamerv3_data_path}")
        
        print("Running DreamerV3 with command:")
        print(" \\\n    ".join(cmd))
        
        # Set environment variables
        env_vars = os.environ.copy()
        # Add current directory and the log directory (for the wrapper) to PYTHONPATH
        env_vars['PYTHONPATH'] = f"{self.log_dir}:{os.path.dirname(os.path.abspath(__file__))}:{env_vars.get('PYTHONPATH', '')}"
        
        if self.args.offline_mode and self.dreamerv3_data_path:
            env_vars['DREAMERV3_OFFLINE_DATA'] = self.dreamerv3_data_path
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env_vars,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            return_code = process.returncode
            
            if return_code == 0:
                print("DreamerV3 training completed successfully!")
            else:
                print(f"DreamerV3 training failed with return code: {return_code}")
                
        except KeyboardInterrupt:
            print("Training interrupted by user")
            process.terminate()
            process.wait()
        except Exception as e:
            print(f"Error running DreamerV3 training: {e}")
            import traceback
            traceback.print_exc()

    def run_alternative_training(self):
        """Alternative training approach using embodied framework directly"""
        
        print("Attempting alternative training using embodied framework...")
        
        try:
            import embodied
            
            # Create config
            config = embodied.Config()
            
            # Set basic parameters
            config.logdir = self.log_dir
            config.seed = self.config['others']['random_seed']
            config.task = 'auv_control'
            
            # Training settings
            dreamerv3_settings = self.config.get('dreamerv3', {})
            config.batch_size = dreamerv3_settings.get('batch_size', 16)
            config.batch_length = dreamerv3_settings.get('batch_length', 64)
            config.train_ratio = dreamerv3_settings.get('train_ratio', 32)
            
            # Try to run training using embodied.run
            if hasattr(embodied, 'run'):
                # Create environment function
                def make_env():
                    return self.env
                
                # Register environment
                embodied.envs.register('auv_control', make_env)
                
                # Run training
                embodied.run.train(config)
                
            else:
                print("embodied.run not available")
                return False
                
        except Exception as e:
            print(f"Alternative training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True

    def run(self):
        """Main training runner"""
        
        print("Starting DreamerV3 training...")
        
        try:
            # Try official DreamerV3 training first
            if os.path.exists(os.path.join(DREAMERV3_REPO_PATH, 'dreamerv3', 'main.py')):
                print("Using official DreamerV3 training script...")
                self.run_dreamerv3_training()
            else:
                print("Official DreamerV3 main.py not found, trying alternative...")
                if not self.run_alternative_training():
                    print("All training methods failed.")
                    print("\nTo use DreamerV3 properly, please:")
                    print("1. Clone the official repository: git clone https://github.com/danijar/dreamerv3.git ~/dreamerv3")
                    print("2. Install requirements: pip install -r ~/dreamerv3/requirements.txt")
                    print("3. Re-run this script")
                    
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if hasattr(self, 'env'):
                self.env.close()
            print("Training session completed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train DreamerV3 agent for AUV control (Proper Integration)')
    
    # Basic arguments
    parser.add_argument('--config', type=str, default='config/config_dreamerv3.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help='Training or evaluation mode')
    
    # Training mode arguments
    parser.add_argument('--offline_mode', action='store_true',
                       help='Use offline training with ROSbag data')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total training timesteps')
    
    # Checkpoint arguments
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint for resuming training')
    
    args = parser.parse_args()
    
    # Check DreamerV3 availability
    if not DREAMERV3_AVAILABLE:
        print("Error: DreamerV3 not available. Please install it:")
        print("1. Clone the repository: git clone https://github.com/danijar/dreamerv3.git ~/dreamerv3")
        print("2. Install dependencies: pip install -r ~/dreamerv3/requirements.txt")
        return
    
    # Load and validate config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    # Initialize and run trainer
    trainer = DreamerV3ProperTrainer(config_path, args)
    trainer.run()

if __name__ == "__main__":
    main()