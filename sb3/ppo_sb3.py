#!/usr/bin/env python3

import os
import numpy as np
import yaml
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Stable Baselines 3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
import torch as th

# For environment compatibility
import gym
from gym.wrappers import TimeLimit

# Import custom environment
from AUVEnv import AUVEnv

class RewardPlottingCallback(BaseCallback):
    """
    Custom callback for plotting rewards during training
    """
    def __init__(self, plot_interval=10, verbose=0):
        super(RewardPlottingCallback, self).__init__(verbose)
        self.plot_interval = plot_interval
        self.rewards = []
        self.moving_avg_rewards = []
        self.episodes = []
        self.episode_count = 0
        self.episode_reward = 0
        self.window_size = 20  # For moving average
        
    def _on_step(self) -> bool:
        # Accumulate reward
        self.episode_reward += self.locals["rewards"][0]
        
        # Check if episode is done
        if self.locals["dones"][0]:
            self.episode_count += 1
            self.rewards.append(self.episode_reward)
            self.episodes.append(self.episode_count)
            
            # Calculate moving average
            if len(self.rewards) >= self.window_size:
                avg = np.mean(self.rewards[-self.window_size:])
            else:
                avg = np.mean(self.rewards)
            self.moving_avg_rewards.append(avg)
            
            # Plot at specified intervals
            if self.episode_count % self.plot_interval == 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.episodes, self.rewards, 'b-', alpha=0.3, label='Episode Reward')
                plt.plot(self.episodes, self.moving_avg_rewards, 'r-', label=f'Moving Avg ({self.window_size} episodes)')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Training Rewards')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{self.model.logger.dir}/reward_plot.png")
                plt.close()
                
                # Log to stable-baselines logger
                self.logger.record("reward/episode_reward", self.rewards[-1])
                self.logger.record("reward/moving_avg", self.moving_avg_rewards[-1])
                
                if self.verbose > 0:
                    print(f"Episode {self.episode_count}, Reward: {self.rewards[-1]:.2f}, Moving Avg: {self.moving_avg_rewards[-1]:.2f}")
        
        return True

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for AUV control')
    parser.add_argument('--config', type=str, default='config/config_ppo.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Training or testing mode')
    parser.add_argument('--model', type=str, default=None, help='Path to model file for testing')
    parser.add_argument('--timesteps', type=int, default=None, help='Total timesteps for training')
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)
    
    # Set random seed
    random_seed = config['others']['random_seed']
    np.random.seed(random_seed)
    th.manual_seed(random_seed)
    
    # Create environment
    env = AUVEnv()
    
    # Limit episode length
    max_episode_steps = config['training']['max_t']
    # env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    # Set up logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"ppo_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    if args.mode == 'train':
        # Configure PPO hyperparameters from config
        learning_rate = config['agent']['learning_rate']
        n_steps = config['agent']['n_steps']
        n_epochs = config['agent']['n_epochs']
        batch_size = config['agent']['batch_size']
        gamma = config['agent']['gamma']
        gae_lambda = config['agent']['gae_lambda']
        clip_range = config['agent']['clip_range']
        
        # Policy network architecture
        policy_kwargs = {
            "net_arch": config['qnetwork']['actor_hidden_layers'] # or specify pi and vf nets separately
        }
        
        # Initialize the PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            seed = random_seed,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir
        )
        
        # Set up callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=max(max_episode_steps // 10, 1),  # Save every 1/10 of max timesteps
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="ppo_auv_model"
        )
        
        # Setup reward plotting callback
        plot_callback = RewardPlottingCallback(
            plot_interval=config['plotting']['plot_interval'],
            verbose=1
        )
        
        # Configure custom logger
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        
        # Determine total timesteps
        if args.timesteps:
            total_timesteps = args.timesteps
        else:
            total_timesteps = config['training']['max_episodes'] * config['training']['max_t']
        
        # Start training
        print(f"Starting PPO training with {total_timesteps} timesteps...")
        print(f"Learning rate: {learning_rate}, n_steps: {n_steps}, n_epochs: {n_epochs}, batch_size: {batch_size}")
        print(f"Policy network: {config['qnetwork']['actor_hidden_layers']}")
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback, plot_callback],
                log_interval=1
            )
            
            # Save the final model
            final_model_path = os.path.join(log_dir, "final_model")
            model.save(final_model_path)
            print(f"Training completed. Final model saved to {final_model_path}")
            
        except KeyboardInterrupt:
            print("Training interrupted. Saving current model...")
            interrupted_model_path = os.path.join(log_dir, "interrupted_model")
            model.save(interrupted_model_path)
            print(f"Interrupted model saved to {interrupted_model_path}")
    
    elif args.mode == 'test':
        if args.model is None:
            raise ValueError("Model path must be provided for testing mode")
        
        print(f"Loading model from {args.model}")
        model = PPO.load(args.model)
        
        # Test the model
        print("Starting evaluation...")
        test_episodes = config['evaluation']['num_episodes']
        episode_rewards = []
        
        for episode in range(test_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                step += 1
                
                time.sleep(0.01)  # Small delay to not overwhelm ROS
                
                # Print step information
                if step % 10 == 0:
                    print(f"Episode {episode+1}/{test_episodes}, Step {step}, Action: {action}, Reward: {reward:.4f}")
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode+1} completed with total reward: {episode_reward:.4f}")
        
        # Plot evaluation results
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, test_episodes+1), episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Evaluation Results')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(log_dir, "evaluation_results.png"))
        
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f"Evaluation completed. Average reward: {avg_reward:.4f} ± {std_reward:.4f}")

if __name__ == "__main__":
    main()