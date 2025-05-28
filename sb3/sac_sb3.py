#!/usr/bin/env python3

import os
import numpy as np
import yaml
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Stable Baselines 3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
import torch as th

# For environment compatibility
import gym
# from gym.wrappers import TimeLimit # Removed, as AUVEnv might handle max steps

# Import custom environment
from AUVEnv import AUVEnv

class WeightUpdateInspectCallback(BaseCallback):
    """
    Callback to inspect if network weights are changing.
    Checks a sample weight from actor and critic.
    """
    def __init__(self, check_freq: int, verbose: int = 0):
        super(WeightUpdateInspectCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.actor_weight_prev = None
        self.critic_weight_prev = None
        # It's good practice to check a specific layer, e.g., the first linear layer's weights
        # Or just the first parameter overall.

    def _on_step(self) -> bool:
        # self.n_calls is the number of times this callback's _on_step has been called
        # self.num_timesteps is the total number of environment steps collected so far
        # SAC updates happen every `train_freq` steps, and perform `gradient_steps` updates.
        # We'll check periodically based on `self.num_timesteps` or `self.n_calls`.
        # Let's use self.n_calls for simplicity with the callback's check_freq.

        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                print(f"\n--- Weight Inspection at Timestep: {self.num_timesteps} (Callback Call: {self.n_calls}) ---")

            # --- Actor Weights ---
            # Access actor parameters (weights and biases)
            actor_params = list(self.model.policy.actor.parameters())
            if actor_params: # Check if actor has parameters
                # Get a sample weight (e.g., first element of the first parameter tensor)
                # .data is important to get the tensor data, not the Parameter object
                # .clone().detach().cpu().numpy() is to safely convert to numpy for comparison
                current_actor_weight_sample = actor_params[0].data.clone().detach().cpu().numpy().flatten()[0]

                if self.actor_weight_prev is not None:
                    if not np.isclose(self.actor_weight_prev, current_actor_weight_sample):
                        if self.verbose > 0:
                            print(f"Actor weights HAVE changed. Sample: {self.actor_weight_prev:.6f} -> {current_actor_weight_sample:.6f}")
                    else:
                        if self.verbose > 0:
                            print(f"Actor weights (sample) appear UNCHANGED: {current_actor_weight_sample:.6f}")
                elif self.verbose > 0:
                    print(f"Actor initial weight sample: {current_actor_weight_sample:.6f}")
                self.actor_weight_prev = current_actor_weight_sample
            else:
                if self.verbose > 0:
                    print("Actor has no parameters to check.")


            # --- Critic Weights ---
            # Access critic parameters
            critic_params = list(self.model.policy.critic.parameters())
            if critic_params:
                current_critic_weight_sample = critic_params[0].data.clone().detach().cpu().numpy().flatten()[0]
                if self.critic_weight_prev is not None:
                    if not np.isclose(self.critic_weight_prev, current_critic_weight_sample):
                        if self.verbose > 0:
                            print(f"Critic weights HAVE changed. Sample: {self.critic_weight_prev:.6f} -> {current_critic_weight_sample:.6f}")
                    else:
                        if self.verbose > 0:
                            print(f"Critic weights (sample) appear UNCHANGED: {current_critic_weight_sample:.6f}")
                elif self.verbose > 0:
                    print(f"Critic initial weight sample: {current_critic_weight_sample:.6f}")
                self.critic_weight_prev = current_critic_weight_sample
            else:
                if self.verbose > 0:
                    print("Critic has no parameters to check.")
            if self.verbose > 0:
                print("--- End Weight Inspection ---")
        return True
    
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
        self.initial_episode_offset = 0 # For resuming plots

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        If resuming, we might want to offset the episode count on the plot.
        """
        # If model.num_timesteps > 0, it's likely a resumed training.
        # However, accurately getting the *episode* count from just timesteps is tricky
        # if episode lengths vary. For simplicity, we'll restart plot numbering,
        # or the user could manually pass an offset if they track episodes externally.
        # For now, we'll just reset, meaning plots are per-training-run.
        # If you need continuous plots, you'd need to save/load callback state.
        if self.model.num_timesteps > 0 and self.model.logger:
             # Try to get previous episode count if logged. This is an approximation.
            try:
                # This depends on how you log episodes.
                # If you have a 'rollout/ep_len_mean' and 'rollout/ep_rew_mean'
                # you could estimate, but it's not straightforward.
                # For simplicity, the plot will restart its episode count.
                print(f"Resuming training. Plot episode count will restart from 1 for this session.")
            except Exception:
                pass # Oh well, can't get it.

    def _on_step(self) -> bool:
        # Accumulate reward
        self.episode_reward += self.locals["rewards"][0]

        # Check if episode is done
        if self.locals["dones"][0]:
            self.episode_count += 1
            self.rewards.append(self.episode_reward)
            self.episodes.append(self.initial_episode_offset + self.episode_count)
            self.episode_reward = 0 # Reset for next episode

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
                plt.title('Training Rewards (Current Session)')
                plt.legend()
                plt.grid(True)
                # Ensure logger dir exists before saving
                if self.model.logger and self.model.logger.dir:
                    os.makedirs(self.model.logger.dir, exist_ok=True)
                    plt.savefig(f"{self.model.logger.dir}/reward_plot_session.png")
                plt.close()

                # Log to stable-baselines logger
                if self.model.logger:
                    self.model.logger.record("reward/episode_reward", self.rewards[-1])
                    self.model.logger.record("reward/moving_avg", self.moving_avg_rewards[-1])
                    self.model.logger.record("reward/episode_count_session", self.episode_count)


                if self.verbose > 0:
                    print(f"Episode {self.episode_count} (Session), Reward: {self.rewards[-1]:.2f}, Moving Avg: {self.moving_avg_rewards[-1]:.2f}")
        return True

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train SAC agent for AUV control')
    parser.add_argument('--config', type=str, default='config/config_sac.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Training or testing mode')
    parser.add_argument('--model', type=str, default=None, help='Path to model file for testing OR initial model for transfer learning (not resume)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to a .zip model file to resume training from')
    parser.add_argument('--timesteps', type=int, default=None, help='Total timesteps for training (overall target)')
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config) # Use abspath for robustness
    config = load_config(config_path)

    # Set random seed
    random_seed = config['others']['random_seed']
    np.random.seed(random_seed)
    th.manual_seed(random_seed)
    # th.cuda.manual_seed_all(random_seed) # if using CUDA

    # Create environment
    env = AUVEnv()

    # Limit episode length (AUVEnv should handle this internally via its _max_episode_steps)
    max_episode_steps = config['training']['max_t']
    # env = TimeLimit(env, max_episode_steps=max_episode_steps) # Usually not needed if env has its own max steps

    # Set up logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_from_checkpoint:
        # Option 1: Log to a new directory for the resumed session
        log_dir_base_name = os.path.splitext(os.path.basename(args.resume_from_checkpoint))[0]
        log_dir = os.path.join("logs", f"sac_resumed_{log_dir_base_name}_{timestamp}")
        # Option 2: Try to log to the same directory (more complex to manage)
        # log_dir = os.path.dirname(os.path.dirname(args.resume_from_checkpoint)) # e.g. logs/sac_XXXX
    else:
        log_dir = os.path.join("logs", f"sac_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")


    if args.mode == 'train':
        # Configure SAC hyperparameters from config
        # These are used if starting fresh or if you want to override specific things
        # when loading (though many params are part of the loaded model)
        learning_rate = config['agent']['learning_rate']
        buffer_size = config['agent']['buffer_size']
        batch_size = config['agent']['batch_size']
        gamma = config['agent']['gamma']
        tau = config['agent']['tau']
        ent_coef = config['agent']['ent_coef']
        target_update_interval = config['agent']['target_update_interval']
        replay_buffer_kwargs = config['agent'].get('replay_buffer_kwargs', None)
        policy_kwargs = {
            "net_arch": config['qnetwork']['actor_hidden_layers'],
            # "net_arch": dict(pi=config['qnetwork']['actor_hidden_layers'], qf=config['qnetwork']['critic_hidden_layers']) # If you define separate critic arch
        }

        if args.resume_from_checkpoint:
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            model = SAC.load(
                args.resume_from_checkpoint,
                env=env, # CRITICAL: Must provide env for continued training
                tensorboard_log=log_dir, # Point tensorboard to the new log directory
                # device='auto' # Or specific device
                # You can also override some parameters here if needed, e.g., learning_rate
                # learning_rate=new_lr_for_resumed_training,
            )
            print(f"Model loaded. Current timesteps: {model.num_timesteps}. Training will continue up to the target total_timesteps.")
            print(f"Original learning rate (from loaded model): {model.learning_rate}")
             # If you want to change the learning rate for the resumed session:
            # model.learning_rate = 0.0001 # Example: new learning rate
            # print(f"Set new learning rate for resumed session: {model.learning_rate}")

        else:
            print("Starting new training session.")
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                gamma=gamma,
                tau=tau,
                ent_coef=ent_coef,
                target_update_interval=target_update_interval,
                seed=random_seed,
                policy_kwargs=policy_kwargs,
                replay_buffer_kwargs=replay_buffer_kwargs,
                verbose=1, # Changed to 1 for less spam, 2 is very verbose
                tensorboard_log=log_dir
            )
            print(f"Learning rate: {learning_rate}, buffer_size: {buffer_size}, batch_size: {batch_size}")
            print(f"Actor network: {config['qnetwork']['actor_hidden_layers']}")


        # Configure custom logger (do this AFTER model is created or loaded)
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        # Set up callbacks
        # Checkpoint callback saves relative to model.save_path, which is based on logger.dir
        checkpoint_callback = CheckpointCallback(
            save_freq=max(config['training']['save_freq_timesteps'], max_episode_steps), # Save e.g. every N timesteps
            save_path=os.path.join(log_dir, "checkpoints"), # Explicitly save in the current log_dir
            name_prefix="sac_auv_model"
        )

        plot_callback = RewardPlottingCallback(
            plot_interval=config['plotting']['plot_interval'],
            verbose=1
        )

        weight_inspect_callback = WeightUpdateInspectCallback(check_freq=100, verbose=1) 

        # Determine total timesteps
        if args.timesteps:
            total_timesteps = args.timesteps
        else:
            total_timesteps = config['training']['max_episodes'] * max_episode_steps # Use max_episode_steps from config

        remaining_timesteps = total_timesteps - model.num_timesteps
        if remaining_timesteps <= 0:
            print(f"Model already trained for {model.num_timesteps} timesteps. Target total_timesteps {total_timesteps} already met or exceeded.")
            print("If you want to train further, increase --timesteps or config['training']['max_episodes'].")
        else:
            print(f"Starting SAC training. Current timesteps: {model.num_timesteps}. Target timesteps: {total_timesteps}. Remaining: {remaining_timesteps}")
            try:
                model.learn(
                    total_timesteps=total_timesteps, # This is the CUMULATIVE total
                    callback=[checkpoint_callback, plot_callback,weight_inspect_callback],
                    log_interval=1, # Log every N rollouts/episodes (depends on n_envs)
                    reset_num_timesteps=False # IMPORTANT: Do NOT reset timesteps when resuming
                )

                final_model_path = os.path.join(log_dir, "final_model")
                model.save(final_model_path)
                print(f"Training completed. Final model saved to {final_model_path}. Total timesteps: {model.num_timesteps}")

            except KeyboardInterrupt:
                print("Training interrupted. Saving current model...")
                interrupted_model_path = os.path.join(log_dir, "interrupted_model")
                model.save(interrupted_model_path)
                print(f"Interrupted model saved to {interrupted_model_path}. Total timesteps: {model.num_timesteps}")

    elif args.mode == 'test':
        if args.model is None:
            # Try to find the latest model in the log_dir if resuming for test
            if args.resume_from_checkpoint:
                # A bit heuristic: assume 'final_model.zip' or 'interrupted_model.zip' might exist
                # in the log_dir associated with the checkpoint's parent.
                # This part can be made more robust.
                potential_log_dir = os.path.dirname(os.path.dirname(args.resume_from_checkpoint))
                final_model_path = os.path.join(potential_log_dir, "final_model.zip")
                interrupted_model_path = os.path.join(potential_log_dir, "interrupted_model.zip")
                if os.path.exists(final_model_path):
                    model_path_to_test = final_model_path
                elif os.path.exists(interrupted_model_path):
                    model_path_to_test = interrupted_model_path
                else:
                    # Fallback to the checkpoint itself if no final model found in its original log dir
                    model_path_to_test = args.resume_from_checkpoint
                print(f"Testing with model: {model_path_to_test} (derived from resume_from_checkpoint)")
            else:
                raise ValueError("Model path (--model) must be provided for testing mode if not resuming.")
        else:
            model_path_to_test = args.model

        print(f"Loading model from {model_path_to_test}")
        model = SAC.load(model_path_to_test, env=env) 

        # Test the model
        print("Starting evaluation...")
        test_episodes = config['evaluation']['num_episodes']
        episode_rewards = []
        episode_lengths = []

        for episode in range(test_episodes):
            # Correctly unpack the return from env.reset()
            obs, info = env.reset() # obs is now the NumPy array
            episode_reward = 0
            # Use terminated and truncated flags from Gymnasium
            terminated = False
            truncated = False
            step = 0

            # Update loop condition
            while not (terminated or truncated):
                # Pass only the observation array to model.predict()
                action, _ = model.predict(obs, deterministic=True)
                # action, _ = env.last_action
                # Correctly unpack the return from env.step()
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step += 1
                
                # Update obs for the next iteration
                obs = next_obs

                # Your env.render call (ensure it's implemented or comment out if not needed)
                # env.render(mode='human')
                # time.sleep(0.01)

                if step % 50 == 0 and config['evaluation']['verbose']:
                    print(f"Episode {episode+1}/{test_episodes}, Step {step}, Action: {action}, Reward: {reward:.4f}")
                if step >= max_episode_steps :
                    print(f"Warning: Episode {episode+1} reached max_episode_steps ({max_episode_steps}) set in test script.")
                    truncated = True


            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            print(f"Episode {episode+1} completed. Reward: {episode_reward:.4f}, Length: {step}")

        # Plot evaluation results
        eval_plot_path = os.path.join(log_dir, "evaluation_results.png") # Save in current session's log_dir
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, test_episodes+1), episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Evaluation Results ({os.path.basename(model_path_to_test)})')
        plt.grid(True, axis='y')
        plt.savefig(eval_plot_path)
        print(f"Evaluation plot saved to {eval_plot_path}")
        plt.close()

        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        print(f"\nEvaluation Summary ({os.path.basename(model_path_to_test)}):")
        print(f"Number of episodes: {test_episodes}")
        print(f"Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"Average Length: {avg_length:.2f} steps")

if __name__ == "__main__":
    main()