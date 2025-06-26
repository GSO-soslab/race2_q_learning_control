#!/usr/bin/env python3

import os
import numpy as np
import yaml
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm  
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
import torch as th
import torch.nn as nn

import gym

from AUVEnv import AUVEnv

# class LayerNormMLP(nn.Module):
#     """MLP with Layer Normalization"""
    
#     def __init__(self, input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU):
#         super().__init__()
        
#         layers = []
#         prev_dim = input_dim
        
#         # Hidden layers with LayerNorm
#         for hidden_dim in net_arch:
#             layers.extend([
#                 nn.Linear(prev_dim, hidden_dim),
#                 nn.LayerNorm(hidden_dim),  # Add Layer Norm
#                 activation_fn()
#             ])
#             prev_dim = hidden_dim
        
#         # Output layer (no norm on output)
#         layers.append(nn.Linear(prev_dim, output_dim))
        
#         self.network = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.network(x)

# class LayerNormSACPolicy(SACPolicy):
#     """SAC Policy with Layer Normalization"""
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def make_actor(self, features_extractor: BaseFeaturesExtractor) -> "Actor":
#         actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        
#         # Get network architecture
#         net_arch = actor_kwargs.get("net_arch", [256, 256])
#         activation_fn = actor_kwargs.get("activation_fn", nn.ReLU)
        
#         # Create custom actor with LayerNorm
#         from stable_baselines3.sac.policies import Actor
        
#         class LayerNormActor(Actor):
#             def __init__(self, *args, **kwargs):
#                 super().__init__(*args, **kwargs)
                
#                 # Replace latent_pi with LayerNorm version
#                 obs_dim = self.features_extractor.features_dim
#                 self.latent_pi = LayerNormMLP(
#                     input_dim=obs_dim,
#                     output_dim=net_arch[-1],  # Last hidden layer size
#                     net_arch=net_arch[:-1],   # All but last layer
#                     activation_fn=activation_fn
#                 )
        
#         return LayerNormActor(**actor_kwargs)
    
#     def make_critic(self, features_extractor: BaseFeaturesExtractor) -> "ContinuousCritic":
#         critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        
#         # Get network architecture  
#         net_arch = critic_kwargs.get("net_arch", [256, 256])
#         activation_fn = critic_kwargs.get("activation_fn", nn.ReLU)
        
#         from stable_baselines3.sac.policies import ContinuousCritic
        
#         class LayerNormCritic(ContinuousCritic):
#             def __init__(self, *args, **kwargs):
#                 super().__init__(*args, **kwargs)
                
#                 # Replace q_networks with LayerNorm versions
#                 obs_dim = self.features_extractor.features_dim
#                 action_dim = self.action_space.shape[0]
#                 input_dim = obs_dim + action_dim
                
#                 self.q_networks = nn.ModuleList([
#                     LayerNormMLP(
#                         input_dim=input_dim,
#                         output_dim=1,
#                         net_arch=net_arch,
#                         activation_fn=activation_fn
#                     )
#                     for _ in range(2)  # SAC uses 2 Q-networks
#                 ])
        
#         return LayerNormCritic(**critic_kwargs)

class PIDActionLearningCallback(BaseCallback):
    """Callback that makes SAC learn from PID actions in CSV mode"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # Check if we're in CSV mode and have recorded action
        info = self.locals.get("infos", [{}])[0]
        
        if 'recorded_action' in info:
            # Replace the action in the replay buffer with PID action
            if self.model.replay_buffer.size() > 0:
                pid_action = info['recorded_action']
                
                # Get the last added transition
                buffer = self.model.replay_buffer
                last_idx = (buffer.pos - 1) % buffer.buffer_size
                
                # Replace SAC action with PID action
                buffer.actions[last_idx] = pid_action
                
                if self.verbose > 0 and self.num_timesteps % 1000 == 0:
                    print(f"Step {self.num_timesteps}: Learning from PID actions")
        
        return True
    
class WarmupMonitoringCallback(BaseCallback):
    """
    Callback to monitor warmup phase and training transitions
    """
    def __init__(self, verbose=1):
        super(WarmupMonitoringCallback, self).__init__(verbose)
        self.warmup_completed = False
        self.warmup_completion_step = None
        
    def _on_step(self) -> bool:
        # Check if we've transitioned from warmup to training
        if not self.warmup_completed and self.model.num_timesteps >= self.model.learning_starts:
            self.warmup_completed = True
            self.warmup_completion_step = self.model.num_timesteps
            
            if self.verbose > 0:
                print(f"\n WARMUP COMPLETED at step {self.warmup_completion_step}")
                print(f"   Replay buffer size: {self.model.replay_buffer.size()}")
                print(f"   Training will now begin with batch size: {self.model.batch_size}")
                
            # Log warmup completion
            if self.model.logger:
                self.model.logger.record("warmup/completion_step", self.warmup_completion_step)
                self.model.logger.record("warmup/buffer_size_at_completion", self.model.replay_buffer.size())
                
        return True

class GradientMonitoringCallback(BaseCallback):
    """
    Custom callback for monitoring gradients and network weight updates
    Enhanced to handle warmup period
    """
    def __init__(self, monitor_interval=100, verbose=1):
        super(GradientMonitoringCallback, self).__init__(verbose)
        self.monitor_interval = monitor_interval
        self.step_count = 0
        self.previous_weights = {}
        self.gradient_norms = {'actor': [], 'critic_1': [], 'critic_2': []}
        self.weight_changes = {'actor': [], 'critic_1': [], 'critic_2': []}
        self.update_counts = {'actor': 0, 'critic_1': 0, 'critic_2': 0}
        self.warmup_phase = True
        
    def _on_training_start(self) -> None:
        """Initialize weight tracking"""
        if hasattr(self.model, 'policy'):
            # Store initial weights
            self._store_current_weights()
            print("Gradient and weight monitoring initialized.")
            print(f"Network architecture:")
            self._print_network_info()
            print(f"Warmup period: {self.model.learning_starts} steps")

    def _store_current_weights(self):
        """Store current network weights for comparison"""
        if hasattr(self.model.policy, 'actor'):
            self.previous_weights['actor'] = self._get_network_weights(self.model.policy.actor)
        if hasattr(self.model.policy, 'critic') and hasattr(self.model.policy.critic, 'q_networks'):
            self.previous_weights['critic_1'] = self._get_network_weights(self.model.policy.critic.q_networks[0])
            self.previous_weights['critic_2'] = self._get_network_weights(self.model.policy.critic.q_networks[1])

    def _get_network_weights(self, network):
        """Extract weights from a network"""
        weights = {}
        for name, param in network.named_parameters():
            weights[name] = param.data.clone().detach()
        return weights

    def _calculate_weight_changes(self):
        """Calculate the magnitude of weight changes"""
        changes = {}
        
        # Actor network
        if 'actor' in self.previous_weights:
            current_actor_weights = self._get_network_weights(self.model.policy.actor)
            actor_change = 0
            for name, current_weight in current_actor_weights.items():
                if name in self.previous_weights['actor']:
                    diff = current_weight - self.previous_weights['actor'][name]
                    actor_change += th.norm(diff).item()
            changes['actor'] = actor_change
            
        # Critic networks
        if hasattr(self.model.policy, 'critic') and hasattr(self.model.policy.critic, 'q_networks'):
            for i, q_net in enumerate(self.model.policy.critic.q_networks):
                critic_key = f'critic_{i+1}'
                if critic_key in self.previous_weights:
                    current_critic_weights = self._get_network_weights(q_net)
                    critic_change = 0
                    for name, current_weight in current_critic_weights.items():
                        if name in self.previous_weights[critic_key]:
                            diff = current_weight - self.previous_weights[critic_key][name]
                            critic_change += th.norm(diff).item()
                    changes[critic_key] = critic_change
                    
        return changes

    def _get_gradient_norms(self):
        """Calculate gradient norms for each network"""
        grad_norms = {}
        
        # Actor gradients
        if hasattr(self.model.policy, 'actor'):
            actor_grad_norm = 0
            for param in self.model.policy.actor.parameters():
                if param.grad is not None:
                    actor_grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norms['actor'] = actor_grad_norm ** 0.5
            
        # Critic gradients
        if hasattr(self.model.policy, 'critic') and hasattr(self.model.policy.critic, 'q_networks'):
            for i, q_net in enumerate(self.model.policy.critic.q_networks):
                critic_grad_norm = 0
                for param in q_net.parameters():
                    if param.grad is not None:
                        critic_grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norms[f'critic_{i+1}'] = critic_grad_norm ** 0.5
                
        return grad_norms

    def _print_network_info(self):
        """Print network architecture information"""
        if hasattr(self.model.policy, 'actor'):
            print("Actor Network:")
            for name, module in self.model.policy.actor.named_modules():
                if isinstance(module, nn.Linear):
                    print(f"  {name}: {module.in_features} -> {module.out_features}")
                    
        if hasattr(self.model.policy, 'critic') and hasattr(self.model.policy.critic, 'q_networks'):
            for i, q_net in enumerate(self.model.policy.critic.q_networks):
                print(f"Critic {i+1} Network:")
                for name, module in q_net.named_modules():
                    if isinstance(module, nn.Linear):
                        print(f"  {name}: {module.in_features} -> {module.out_features}")

    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Check if we're still in warmup
        in_warmup = self.model.num_timesteps < self.model.learning_starts
        if self.warmup_phase and not in_warmup:
            self.warmup_phase = False
            if self.verbose > 0:
                print(f"\n📊 Gradient monitoring: Training phase started at step {self.model.num_timesteps}")
        
        # Monitor at specified intervals, but only during training phase
        if not in_warmup and self.step_count % self.monitor_interval == 0:
            # Calculate gradient norms
            grad_norms = self._get_gradient_norms()
            
            # Calculate weight changes
            weight_changes = self._calculate_weight_changes()
            
            # Store gradient norms
            for network, norm in grad_norms.items():
                self.gradient_norms[network].append(norm)
                
            # Store weight changes and update counts
            for network, change in weight_changes.items():
                self.weight_changes[network].append(change)
                if change > 1e-8:  # Threshold for considering a weight update
                    self.update_counts[network] += 1
                    
            # Log information
            if self.verbose > 0:
                print(f"\n--- Step {self.step_count} Monitoring (Training Phase) ---")
                print("Gradient Norms:")
                for network, norm in grad_norms.items():
                    print(f"  {network}: {norm:.6f}")
                    
                print("Weight Changes (L2 norm):")
                for network, change in weight_changes.items():
                    print(f"  {network}: {change:.6f}")
                    
                print("Update Counts:")
                for network, count in self.update_counts.items():
                    print(f"  {network}: {count} updates")
                    
            # Log to tensorboard
            if self.model.logger:
                for network, norm in grad_norms.items():
                    self.model.logger.record(f"gradients/{network}_norm", norm)
                for network, change in weight_changes.items():
                    self.model.logger.record(f"weights/{network}_change", change)
                for network, count in self.update_counts.items():
                    self.model.logger.record(f"updates/{network}_count", count)
                self.model.logger.record("training/in_warmup", 0)
                    
            # Update stored weights for next comparison
            self._store_current_weights()
            
            # Save gradient plots
            if self.model.logger and self.model.logger.dir:
                self._save_gradient_plots()
        
        elif in_warmup:
            # Log warmup status occasionally
            if self.step_count % (self.monitor_interval * 5) == 0 and self.verbose > 0:
                progress = (self.model.num_timesteps / self.model.learning_starts) * 100
                print(f"🔄 Warmup phase: {self.model.num_timesteps}/{self.model.learning_starts} steps ({progress:.1f}%)")
                
            # Log warmup status to tensorboard
            if self.model.logger and self.step_count % self.monitor_interval == 0:
                self.model.logger.record("training/in_warmup", 1)
                self.model.logger.record("warmup/progress", self.model.num_timesteps / self.model.learning_starts)
                
        return True
        
    def _save_gradient_plots(self):
        """Save gradient and weight change plots"""
        if not self.gradient_norms['actor']:  # No data yet
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gradient norms plot
        axes[0, 0].set_title('Gradient Norms Over Time (Training Phase Only)')
        for network, norms in self.gradient_norms.items():
            if norms:
                axes[0, 0].plot(norms, label=network)
        axes[0, 0].set_xlabel('Monitoring Steps (Post-Warmup)')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_yscale('log')
        
        # Weight changes plot
        axes[0, 1].set_title('Weight Changes Over Time (Training Phase Only)')
        for network, changes in self.weight_changes.items():
            if changes:
                axes[0, 1].plot(changes, label=network)
        axes[0, 1].set_xlabel('Monitoring Steps (Post-Warmup)')
        axes[0, 1].set_ylabel('Weight Change (L2 Norm)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_yscale('log')
        
        # Update counts bar plot
        networks = list(self.update_counts.keys())
        counts = list(self.update_counts.values())
        axes[1, 0].bar(networks, counts)
        axes[1, 0].set_title('Total Weight Updates by Network')
        axes[1, 0].set_ylabel('Update Count')
        axes[1, 0].grid(True, axis='y')
        
        # Recent gradient norms (last 20 monitoring steps)
        axes[1, 1].set_title('Recent Gradient Norms (Last 20 Steps)')
        for network, norms in self.gradient_norms.items():
            if norms:
                recent_norms = norms[-20:] if len(norms) > 20 else norms
                axes[1, 1].plot(recent_norms, label=network, marker='o')
        axes[1, 1].set_xlabel('Recent Monitoring Steps')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plot_path = os.path.join(self.model.logger.dir, "gradient_monitoring.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

class EnhancedRewardPlottingCallback(BaseCallback):
    """
    Enhanced callback for plotting rewards with setpoint tracking during training
    Enhanced to handle warmup period and CSV mode
    """
    def __init__(self, plot_interval=10, verbose=0):
        super(EnhancedRewardPlottingCallback, self).__init__(verbose)
        self.plot_interval = plot_interval
        self.rewards = []
        self.moving_avg_rewards = []
        self.episodes = []
        self.episode_count = 0
        self.episode_reward = 0
        self.window_size = 20  # For moving average
        self.initial_episode_offset = 0
        
        # Setpoint tracking (for online mode)
        self.setpoint_history = []
        self.episode_setpoints = {}
        
        # CSV mode tracking
        self.csv_episode_info = []
        
        # Warmup tracking
        self.warmup_episodes = 0
        self.training_episodes = 0

    def _on_training_start(self) -> None:
        """Initialize tracking"""
        if self.model.num_timesteps > 0 and self.model.logger:
            try:
                print(f"Resuming training. Plot episode count will restart from 1 for this session.")
                print(f"Warmup period: {self.model.learning_starts} steps")
            except Exception:
                pass

    def _on_step(self) -> bool:
        # Accumulate reward
        self.episode_reward += self.locals["rewards"][0]

        # Check if episode is done
        if self.locals["dones"][0]:
            self.episode_count += 1
            self.rewards.append(self.episode_reward)
            self.episodes.append(self.initial_episode_offset + self.episode_count)
            
            # Track warmup vs training episodes
            if self.model.num_timesteps < self.model.learning_starts:
                self.warmup_episodes += 1
                episode_phase = "WARMUP"
            else:
                self.training_episodes += 1
                episode_phase = "TRAINING"
            
            # Extract episode information
            info = self.locals.get("infos", [{}])[0]
            
            # Handle both online and CSV modes
            if 'setpoint_values' in info:
                # Online mode
                self.episode_setpoints[self.episode_count] = info['setpoint_values']
                self.setpoint_history.append(info['setpoint_values'])
                
                if self.verbose > 0:
                    setpoint_info = info.get('setpoint_info', 'No setpoint info')
                    print(f"Episode {self.episode_count} ({episode_phase}): {setpoint_info}, Reward: {self.episode_reward:.2f}")
            
            elif 'csv_episode_info' in info:
                # CSV mode
                self.csv_episode_info.append(info['csv_episode_info'])
                
                if self.verbose > 0:
                    csv_info = info.get('setpoint_info', 'CSV Episode')
                    print(f"Episode {self.episode_count} ({episode_phase}): {csv_info}, Reward: {self.episode_reward:.2f}")
            
            self.episode_reward = 0  # Reset for next episode

            # Calculate moving average
            if len(self.rewards) >= self.window_size:
                avg = np.mean(self.rewards[-self.window_size:])
            else:
                avg = np.mean(self.rewards)
            self.moving_avg_rewards.append(avg)

            # Plot at specified intervals
            if self.episode_count % self.plot_interval == 0:
                self._create_enhanced_plots()

                # Log to stable-baselines logger
                if self.model.logger:
                    self.model.logger.record("reward/episode_reward", self.rewards[-1])
                    self.model.logger.record("reward/moving_avg", self.moving_avg_rewards[-1])
                    self.model.logger.record("reward/episode_count_session", self.episode_count)
                    self.model.logger.record("reward/warmup_episodes", self.warmup_episodes)
                    self.model.logger.record("reward/training_episodes", self.training_episodes)
                    
                    # Log mode-specific metrics
                    if self.setpoint_history:
                        # Online mode setpoint diversity metrics
                        recent_setpoints = self.setpoint_history[-self.plot_interval:]
                        pos_z_std = np.std([sp['pos_z'] for sp in recent_setpoints])
                        ori_z_std = np.std([sp['ori_z'] for sp in recent_setpoints])
                        vel_x_std = np.std([sp['vel_x'] for sp in recent_setpoints])
                        
                        self.model.logger.record("setpoint/pos_z_diversity", pos_z_std)
                        self.model.logger.record("setpoint/ori_z_diversity", ori_z_std)
                        self.model.logger.record("setpoint/vel_x_diversity", vel_x_std)
                    
                    elif self.csv_episode_info:
                        # CSV mode metrics
                        self.model.logger.record("csv/episodes_completed", len(self.csv_episode_info))

                if self.verbose > 0:
                    print(f"Episode {self.episode_count} (Session), Reward: {self.rewards[-1]:.2f}, Moving Avg: {self.moving_avg_rewards[-1]:.2f}")
                    print(f"  Warmup episodes: {self.warmup_episodes}, Training episodes: {self.training_episodes}")
        return True
    
    def _create_enhanced_plots(self):
        """Create enhanced plots for both online and CSV modes"""
        if not self.model.logger or not self.model.logger.dir:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward plot with warmup indicator
        axes[0, 0].plot(self.episodes, self.rewards, 'b-', alpha=0.3, label='Episode Reward')
        axes[0, 0].plot(self.episodes, self.moving_avg_rewards, 'r-', label=f'Moving Avg ({self.window_size} episodes)')
        
        # Add vertical line to indicate warmup completion (approximate)
        if self.training_episodes > 0:
            warmup_end_episode = self.warmup_episodes
            if warmup_end_episode < len(self.episodes):
                axes[0, 0].axvline(x=self.episodes[warmup_end_episode], color='green', linestyle='--', 
                                 alpha=0.7, label='Training Started')
        
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title(f'Training Rewards (Warmup: {self.warmup_episodes}, Training: {self.training_episodes})')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Mode-specific plots
        if self.setpoint_history:
            # Online mode: setpoint diversity plots
            recent_episodes = min(50, len(self.setpoint_history))
            recent_setpoints = self.setpoint_history[-recent_episodes:]
            episode_nums = list(range(len(self.setpoint_history) - recent_episodes + 1, len(self.setpoint_history) + 1))
            
            # Position Z (depth) targets
            pos_z_values = [sp['pos_z'] for sp in recent_setpoints]
            axes[0, 1].scatter(episode_nums, pos_z_values, alpha=0.6, s=20)
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Depth Target (m)')
            axes[0, 1].set_title(f'Depth Setpoints (Last {recent_episodes} episodes)')
            axes[0, 1].grid(True)
            
            # Orientation Z (yaw) targets
            ori_z_values = [sp['ori_z'] for sp in recent_setpoints]
            axes[1, 0].scatter(episode_nums, ori_z_values, alpha=0.6, s=20, color='orange')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Yaw Target (rad)')
            axes[1, 0].set_title(f'Yaw Setpoints (Last {recent_episodes} episodes)')
            axes[1, 0].grid(True)
            
            # Velocity X (surge) targets
            vel_x_values = [sp['vel_x'] for sp in recent_setpoints]
            axes[1, 1].scatter(episode_nums, vel_x_values, alpha=0.6, s=20, color='green')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Surge Target (m/s)')
            axes[1, 1].set_title(f'Surge Velocity Setpoints (Last {recent_episodes} episodes)')
            axes[1, 1].grid(True)
            
        elif self.csv_episode_info:
            # CSV mode: data usage plots
            recent_episodes = min(50, len(self.csv_episode_info))
            recent_info = self.csv_episode_info[-recent_episodes:]
            episode_nums = list(range(len(self.csv_episode_info) - recent_episodes + 1, len(self.csv_episode_info) + 1))
            
            # Episode start timestamps
            start_times = [info['start_timestamp'] for info in recent_info]
            axes[0, 1].scatter(episode_nums, start_times, alpha=0.6, s=20)
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Start Time (s)')
            axes[0, 1].set_title(f'CSV Data Usage (Last {recent_episodes} episodes)')
            axes[0, 1].grid(True)
            
            # Episode lengths
            episode_lengths = [info['episode_length'] for info in recent_info]
            axes[1, 0].scatter(episode_nums, episode_lengths, alpha=0.6, s=20, color='orange')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Episode Length (steps)')
            axes[1, 0].set_title(f'Episode Lengths')
            axes[1, 0].grid(True)
            
            # CSV mode summary
            axes[1, 1].axis('off')
            csv_text = f"""CSV Training Mode:
Episodes: {len(self.csv_episode_info)}
Data Coverage: {min(start_times):.1f}s to {max(start_times):.1f}s
Avg Episode Length: {np.mean(episode_lengths):.1f} steps
Data-driven learning from real AUV recordings"""
            axes[1, 1].text(0.1, 0.9, csv_text, transform=axes[1, 1].transAxes, 
                            fontsize=12, verticalalignment='top', fontfamily='monospace')
            
        else:
            # No specific mode data available
            for i in range(1, 4):
                ax = axes.flat[i]
                ax.text(0.5, 0.5, 'No mode-specific data available', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'Mode Plot {i}')
        
        plt.tight_layout()
        plot_path = os.path.join(self.model.logger.dir, "enhanced_training_plots.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save mode-specific summary
        if self.setpoint_history:
            self._save_setpoint_summary()
        elif self.csv_episode_info:
            self._save_csv_summary()
    
    def _save_setpoint_summary(self):
        """Save a summary of setpoint diversity (online mode)"""
        if not self.setpoint_history:
            return
            
        summary_path = os.path.join(self.model.logger.dir, "setpoint_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Online Mode Setpoint Summary for {len(self.setpoint_history)} episodes\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Warmup episodes: {self.warmup_episodes}\n")
            f.write(f"Training episodes: {self.training_episodes}\n\n")
            
            # Calculate statistics
            pos_z_values = [sp['pos_z'] for sp in self.setpoint_history]
            ori_z_values = [sp['ori_z'] for sp in self.setpoint_history]
            vel_x_values = [sp['vel_x'] for sp in self.setpoint_history]
            
            f.write("Depth (pos_z) statistics:\n")
            f.write(f"  Range: {min(pos_z_values):.2f} to {max(pos_z_values):.2f} m\n")
            f.write(f"  Mean: {np.mean(pos_z_values):.2f} m\n")
            f.write(f"  Std: {np.std(pos_z_values):.2f} m\n\n")
            
            f.write("Yaw (ori_z) statistics:\n")
            f.write(f"  Range: {min(ori_z_values):.2f} to {max(ori_z_values):.2f} rad\n")
            f.write(f"  Mean: {np.mean(ori_z_values):.2f} rad\n")
            f.write(f"  Std: {np.std(ori_z_values):.2f} rad\n\n")
            
            f.write("Surge velocity (vel_x) statistics:\n")
            f.write(f"  Range: {min(vel_x_values):.2f} to {max(vel_x_values):.2f} m/s\n")
            f.write(f"  Mean: {np.mean(vel_x_values):.2f} m/s\n")
            f.write(f"  Std: {np.std(vel_x_values):.2f} m/s\n\n")
            
            f.write("Recent 10 episodes setpoints:\n")
            for i, sp in enumerate(self.setpoint_history[-10:], 1):
                episode_num = len(self.setpoint_history) - 10 + i
                episode_type = "WARMUP" if episode_num <= self.warmup_episodes else "TRAINING"
                f.write(f"  Episode {episode_num} ({episode_type}): depth={sp['pos_z']:.2f}m, yaw={sp['ori_z']:.2f}rad, surge={sp['vel_x']:.2f}m/s\n")
    
    def _save_csv_summary(self):
        """Save a summary of CSV episode usage"""
        if not self.csv_episode_info:
            return
            
        summary_path = os.path.join(self.model.logger.dir, "csv_training_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"CSV Mode Training Summary for {len(self.csv_episode_info)} episodes\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Warmup episodes: {self.warmup_episodes}\n")
            f.write(f"Training episodes: {self.training_episodes}\n\n")
            
            # Calculate statistics
            start_times = [info['start_timestamp'] for info in self.csv_episode_info]
            episode_lengths = [info['episode_length'] for info in self.csv_episode_info]
            
            f.write("CSV Data Usage Statistics:\n")
            f.write(f"  Time range covered: {min(start_times):.1f}s to {max(start_times):.1f}s\n")
            f.write(f"  Episode length range: {min(episode_lengths)} to {max(episode_lengths)} steps\n")
            f.write(f"  Average episode length: {np.mean(episode_lengths):.1f} steps\n")
            f.write(f"  Total data duration: {max(start_times) - min(start_times):.1f}s\n\n")
            
            f.write("Recent 10 episodes:\n")
            for i, info in enumerate(self.csv_episode_info[-10:], 1):
                episode_num = len(self.csv_episode_info) - 10 + i
                episode_type = "WARMUP" if episode_num <= self.warmup_episodes else "TRAINING"
                f.write(f"  Episode {episode_num} ({episode_type}): start={info['start_timestamp']:.1f}s, length={info['episode_length']} steps\n")


def prefill_replay_buffer_with_csv_data(model, csv_directory, config, num_epochs=3):
    """Fill replay buffer with CSV data multiple times with shuffling"""
    from csv_data_manager import CSVDataManager
    
    # Load all CSV data once
    csv_manager = CSVDataManager(csv_directory, config)
    total_transitions = csv_manager.total_steps - 1  # Need next_obs
    
    print(f"Pre-filling replay buffer with {total_transitions} transitions across {num_epochs} epochs...")
    print(f"Total data exposure: {total_transitions * num_epochs} transition samples")
    
    transitions_added = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}: Shuffling and loading data...")
        
        # Create shuffled indices for this epoch
        indices = list(range(total_transitions))
        np.random.shuffle(indices)
        
        epoch_added = 0
        for i, data_idx in enumerate(indices):
            # Check if buffer is full
            if model.replay_buffer.size() >= model.replay_buffer.buffer_size:
                print(f"Replay buffer full at {model.replay_buffer.size()} samples")
                return transitions_added
            
            # Get current transition
            csv_manager.current_step = data_idx
            csv_manager._update_state_from_step(data_idx)
            obs = csv_manager._get_current_state_observation()
            pid_action = csv_manager.get_recorded_action_at_current_step()
            
            # Get next observation  
            csv_manager.current_step = data_idx + 1
            csv_manager._update_state_from_step(data_idx + 1)
            next_obs = csv_manager._get_current_state_observation()
            
            # Calculate reward for this transition
            csv_manager.current_step = data_idx  # Reset to current for reward calculation
            csv_manager._update_state_from_step(data_idx)
            state_error_array = csv_manager.get_state_error_array()
            
            # Use simple reward calculation to avoid format mismatch
            # Apply standard quadratic penalty on errors
            w = config['reward_function']
            state_error_weights = w.get('state_error_weights', [1.0] * len(state_error_array))
            
            # Ensure weights match error array size
            if len(state_error_weights) != len(state_error_array):
                # Use first weight for all if mismatch
                weight_val = state_error_weights[0] if state_error_weights else 1.0
                state_error_weights = [weight_val] * len(state_error_array)
            
            # Calculate simple quadratic reward
            performance_error = 0
            for i, (error, weight) in enumerate(zip(state_error_array, state_error_weights)):
                performance_error += weight * (error ** 2)
            
            reward = -performance_error  # Negative because we want to minimize errors
            
            # Determine if this is a terminal state (only at very end of data)
            done = (data_idx == total_transitions - 1)
            
            # Add to replay buffer
            obs = np.array(obs, dtype=np.float32).flatten()
            next_obs = np.array(next_obs, dtype=np.float32).flatten()  
            pid_action = np.array(pid_action, dtype=np.float32).flatten()[:4] #only 4 cause of thrusters with no servos. 
            reward = float(reward)
            done = bool(done)

            model.replay_buffer.add(obs, next_obs, pid_action, reward, done, [{}])
            
            transitions_added += 1
            epoch_added += 1
            
            # Progress indicator
            if i % 5000 == 0:
                progress = (i / len(indices)) * 100
                print(f"  Epoch {epoch + 1} progress: {i}/{len(indices)} ({progress:.1f}%)")
        
        print(f"  Epoch {epoch + 1} completed: {epoch_added} transitions added")
        print(f"  Buffer size: {model.replay_buffer.size()}/{model.replay_buffer.buffer_size}")
        
        # If buffer is full, stop
        if model.replay_buffer.size() >= model.replay_buffer.buffer_size:
            break
    
    print(f"\n✅ Multi-epoch loading completed!")
    print(f"  Total transitions added: {transitions_added}")
    print(f"  Buffer utilization: {model.replay_buffer.size()}/{model.replay_buffer.buffer_size} ({model.replay_buffer.size()/model.replay_buffer.buffer_size*100:.1f}%)")
    print(f"  Data seen {transitions_added / total_transitions:.1f}x times on average")
    
    return transitions_added


def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train SAC agent for AUV control with CSV or online training')
    parser.add_argument('--config', type=str, default='config/config_sac.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Training or testing mode')
    parser.add_argument('--model', type=str, default=None, help='Path to model file for testing')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--timesteps', type=int, default=None, help='Total timesteps for training')
    parser.add_argument('--gradient_monitor_interval', type=int, default=100, help='Gradient monitoring interval')
    parser.add_argument('--learning_starts', type=int, default=None, help='Warmup period steps')
    
    parser.add_argument('--csv_directory', type=str, default=None, 
                       help='Directory containing CSV files from rosbags for offline training')
    parser.add_argument('--csv_mode', action='store_true', 
                       help='Enable CSV mode (alternative to --csv_directory)')
    
    parser.add_argument('--csv_epochs', type=int, default=3, 
                   help='Number of epochs to load CSV data (default: 3)')
    
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    config = load_config(config_path)

    # Set random seed
    # random_seed = config['others']['random_seed']
    random_seed = int(time.time()) % 10000  # Random seed each run
    np.random.seed(random_seed)
    th.manual_seed(random_seed)

    if args.csv_directory or args.csv_mode:
        if args.csv_directory:
            csv_dir = args.csv_directory
        else:
            # Look for CSV files in default location
            csv_dir = os.path.join(os.path.dirname(__file__), 'csv_data')
            if not os.path.exists(csv_dir):
                raise ValueError("CSV mode enabled but no CSV directory found. Use --csv_directory to specify path.")
        
        print(f"Creating AUV environment in CSV mode using data from: {csv_dir}")
        env = AUVEnv(csv_directory=csv_dir)  # Pass CSV directory to enable CSV mode

        training_mode = "CSV"
        
    else:
        print("Creating AUV environment in online ROS2 mode...")
        env = AUVEnv()  # No CSV directory = online mode
        training_mode = "Online ROS2"
    
    print(f"Environment created successfully in {training_mode} mode!")

    # Display configuration based on mode
    if args.csv_directory or args.csv_mode:
        print(f"\nCSV Training Configuration:")
        print(f"  CSV Directory: {csv_dir}")
        if hasattr(env, 'csv_manager'):
            stats = env.csv_manager.get_dataset_stats()
            print(f"  Total timesteps available: {stats.get('total_timesteps', 'Unknown')}")
            print(f"  Duration: {stats.get('duration_seconds', 0):.1f} seconds")
            print(f"  Potential episodes (500 steps): {stats.get('potential_episodes_500_steps', 0)}")
        print(f"  Training will cycle through available CSV data segments.")
    else:
        # Display setpoint configuration for online mode
        setpoint_config = config.get('setpoint', {})
        print(f"\nOnline Setpoint Configuration:")
        print(f"  Depth range: {setpoint_config.get('pos_z_range', [1.0, 8.0])} m")
        print(f"  Yaw range: {setpoint_config.get('ori_z_range', [-2.14, 2.14])} rad")
        print(f"  Pitch range: {setpoint_config.get('ori_y_range', [-0.1, 0.1])} rad")
        print(f"  Surge velocity range: {setpoint_config.get('vel_x_range', [-0.6, 0.6])} m/s")
        print("Setpoints will change at the beginning of each episode.")

    # Set up logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_from_checkpoint:
        log_dir_base_name = os.path.splitext(os.path.basename(args.resume_from_checkpoint))[0]
        log_dir = os.path.join("logs", f"sac_resumed_{log_dir_base_name}_{timestamp}")
    else:
        mode_suffix = "csv" if (args.csv_directory or args.csv_mode) else "online"
        log_dir = os.path.join("logs", f"sac_{mode_suffix}_warmup_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    if args.mode == 'train':
        # Configure SAC hyperparameters from config
        learning_rate = config['agent']['learning_rate']
        buffer_size = config['agent']['buffer_size']
        batch_size = config['agent']['batch_size']
        gamma = config['agent']['gamma']
        tau = config['agent']['tau']
        ent_coef = config['agent']['ent_coef']
        target_update_interval = config['agent']['target_update_interval']
        replay_buffer_kwargs = config['agent'].get('replay_buffer_kwargs', None)
        
        # Set learning_starts (warmup period)
        if args.learning_starts:
            learning_starts = args.learning_starts
        else:
            # Use config value if available, otherwise use a reasonable default
            learning_starts = config['agent'].get('learning_starts', min(1000, batch_size * 4))
        
        optimizer_kwargs = config['agent'].get('optimizer_kwargs', {
            "betas": (0.9, 0.9),
            "weight_decay": 1e-4
        })

        policy_kwargs = {
            "net_arch": config['qnetwork']['actor_hidden_layers'],
            "optimizer_class": th.optim.Adam,
            "optimizer_kwargs": optimizer_kwargs
        }

        if args.resume_from_checkpoint:
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            model = SAC.load(
                args.resume_from_checkpoint,
                env=env,
                tensorboard_log=log_dir,
            )

            if args.csv_directory or args.csv_mode:
                print("🔄 Pre-filling replay buffer with CSV data...")
                # csv_data_points = prefill_replay_buffer_with_csv_data(model, csv_dir, config)
                csv_epochs = getattr(args, 'csv_epochs', 3) 
                csv_data_points = prefill_replay_buffer_with_csv_data(model, csv_dir, config, csv_epochs)

                print(f"✅ Loaded {csv_data_points} transitions from CSV data")
                print(f"   Buffer utilization: {model.replay_buffer.size()}/{buffer_size} ({model.replay_buffer.size()/buffer_size*100:.1f}%)")
        
            print(f"Model loaded. Current timesteps: {model.num_timesteps}")
            print(f"Warmup period: {model.learning_starts} steps")
            print(f"Training will continue in {training_mode} mode.")

        else:
            print(f"Starting new training session in {training_mode} mode with warmup period.")
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                gamma=gamma,
                tau=tau,
                ent_coef=ent_coef,
                target_update_interval=target_update_interval,
                seed=random_seed,
                policy_kwargs=policy_kwargs,
                replay_buffer_kwargs=replay_buffer_kwargs,
                verbose=1,
                tensorboard_log=log_dir
            )

            # model = SAC(
            #     LayerNormSACPolicy,  # Use custom policy with LayerNorm
            #     env,
            #     learning_rate=learning_rate,
            #     buffer_size=buffer_size,
            #     learning_starts=learning_starts,
            #     batch_size=batch_size,
            #     gamma=gamma,
            #     tau=tau,
            #     ent_coef=ent_coef,
            #     target_update_interval=target_update_interval,
            #     seed=random_seed,
            #     policy_kwargs={
            #         "net_arch": config['qnetwork']['actor_hidden_layers'],
            #         "activation_fn": th.nn.ReLU,
            #     },
            #     replay_buffer_kwargs=replay_buffer_kwargs,
            #     verbose=1,
            #     tensorboard_log=log_dir
            # )
            print(f"Learning rate: {learning_rate}, buffer_size: {buffer_size}, batch_size: {batch_size}")
            print(f"Warmup period (learning_starts): {learning_starts} steps")
            print(f"Actor network: {config['qnetwork']['actor_hidden_layers']}")


        if args.csv_directory or args.csv_mode:
                print("🔄 Pre-filling replay buffer with CSV data...")
                csv_epochs = getattr(args, 'csv_epochs', 3) 
                csv_data_points = prefill_replay_buffer_with_csv_data(model, csv_dir, config, csv_epochs)
                print(f"✅ Loaded {csv_data_points} transitions from CSV data")
                print(f"   Buffer utilization: {model.replay_buffer.size()}/{buffer_size} ({model.replay_buffer.size()/buffer_size*100:.1f}%)")

        # Configure custom logger
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        # Set up callbacks - same for both modes
        max_episode_steps = config['training']['max_t']
        checkpoint_callback = CheckpointCallback(
            save_freq=max(config['training']['save_freq_timesteps'], max_episode_steps),
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix=f"sac_auv_{training_mode.lower().replace(' ', '_')}_warmup"
        )

        # Use enhanced plotting callback (now handles both modes)
        plot_callback = EnhancedRewardPlottingCallback(
            plot_interval=config['plotting']['plot_interval'],
            verbose=1
        )
        
        # Add gradient monitoring callback
        gradient_callback = GradientMonitoringCallback(
            monitor_interval=args.gradient_monitor_interval,
            verbose=1
        )
        
        # Add warmup monitoring callback
        warmup_callback = WarmupMonitoringCallback(verbose=1)


        # Collect all callbacks
        callbacks = [checkpoint_callback, plot_callback, gradient_callback, warmup_callback]
        
        # Add PID action learning callback for CSV mode
        if args.csv_directory or args.csv_mode:
            callbacks.append(PIDActionLearningCallback(verbose=1))
            print("PID action learning callback for CSV mode")

        # Determine total timesteps
        if args.timesteps:
            total_timesteps = args.timesteps
        else:
            total_timesteps = config['training']['max_episodes'] * max_episode_steps

        remaining_timesteps = total_timesteps - model.num_timesteps
        if remaining_timesteps <= 0:
            print(f"Model already trained for {model.num_timesteps} timesteps. Target total_timesteps {total_timesteps} already met or exceeded.")
            print("If you want to train further, increase --timesteps or config['training']['max_episodes'].")
        else:
            print(f"Starting SAC training in {training_mode} mode with warmup period.")
            print(f"Current timesteps: {model.num_timesteps}. Target timesteps: {total_timesteps}. Remaining: {remaining_timesteps}")
            print(f"Warmup period: {model.learning_starts} steps (training starts after this)")
            print(f"Gradient monitoring interval: {args.gradient_monitor_interval} steps")
            
            if args.csv_directory or args.csv_mode:
                if hasattr(env, 'csv_manager'):
                    stats = env.csv_manager.get_dataset_stats()
                    print(f"Training will use {stats.get('total_timesteps', 0)} timesteps of CSV data.")
            else:
                print(f"Each episode will have a new random setpoint generated at reset.")
            
            # Display warmup strategy
            if model.num_timesteps < model.learning_starts:
                warmup_remaining = model.learning_starts - model.num_timesteps
                print(f"🔄 Currently in WARMUP phase. {warmup_remaining} steps remaining before training begins.")
                print(f"   During warmup: collecting experiences, no gradient updates")
                print(f"   Replay buffer will be filled to {model.learning_starts} samples before training")
            else:
                print(f"✅ Warmup completed. Currently in TRAINING phase.")
            
            try:
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=[checkpoint_callback, plot_callback, gradient_callback, warmup_callback],
                    log_interval=1,
                    reset_num_timesteps=False
                )

                final_model_path = os.path.join(log_dir, "final_model")
                model.save(final_model_path)
                print(f"Training completed. Final model saved to {final_model_path}. Total timesteps: {model.num_timesteps}")

                # Print final summaries
                print(f"\n--- Final Training Summary ---")
                print(f"Training mode: {training_mode}")
                print(f"Total episodes completed: {plot_callback.episode_count}")
                print(f"  - Warmup episodes: {plot_callback.warmup_episodes}")
                print(f"  - Training episodes: {plot_callback.training_episodes}")
                
                if args.csv_directory or args.csv_mode:
                    print(f"CSV episodes completed: {len(plot_callback.csv_episode_info)}")
                    if plot_callback.csv_episode_info:
                        start_times = [info['start_timestamp'] for info in plot_callback.csv_episode_info]
                        print(f"CSV data range used: {min(start_times):.1f}s to {max(start_times):.1f}s")
                else:
                    print(f"Setpoint diversity achieved: {len(plot_callback.setpoint_history)} unique setpoints")
                    if plot_callback.setpoint_history:
                        print(f"Depth range explored: {min(sp['pos_z'] for sp in plot_callback.setpoint_history):.2f} to {max(sp['pos_z'] for sp in plot_callback.setpoint_history):.2f} m")
                        print(f"Yaw range explored: {min(sp['ori_z'] for sp in plot_callback.setpoint_history):.2f} to {max(sp['ori_z'] for sp in plot_callback.setpoint_history):.2f} rad")

                print(f"\n--- Final Gradient Monitoring Summary ---")
                print("Total Updates by Network (Training Phase Only):")
                for network, count in gradient_callback.update_counts.items():
                    print(f"  {network}: {count} updates")
                
                if warmup_callback.warmup_completion_step:
                    print(f"Warmup completed at step: {warmup_callback.warmup_completion_step}")

            except KeyboardInterrupt:
                print("Training interrupted. Saving current model...")
                interrupted_model_path = os.path.join(log_dir, "interrupted_model")
                model.save(interrupted_model_path)
                print(f"Interrupted model saved to {interrupted_model_path}. Total timesteps: {model.num_timesteps}")

    elif args.mode == 'test':
        # Testing mode - works for both CSV and online modes
        if args.model is None:
            if args.resume_from_checkpoint:
                potential_log_dir = os.path.dirname(os.path.dirname(args.resume_from_checkpoint))
                final_model_path = os.path.join(potential_log_dir, "final_model.zip")
                interrupted_model_path = os.path.join(potential_log_dir, "interrupted_model.zip")
                if os.path.exists(final_model_path):
                    model_path_to_test = final_model_path
                elif os.path.exists(interrupted_model_path):
                    model_path_to_test = interrupted_model_path
                else:
                    model_path_to_test = args.resume_from_checkpoint
                print(f"Testing with model: {model_path_to_test} (derived from resume_from_checkpoint)")
            else:
                raise ValueError("Model path (--model) must be provided for testing mode if not resuming.")
        else:
            model_path_to_test = args.model

        print(f"Loading model from {model_path_to_test}")
        model = SAC.load(model_path_to_test, env=env) 

        # Test the model
        print(f"Starting evaluation in {training_mode} mode...")
        max_episode_steps = config['training']['max_t']
        test_episodes = config['evaluation']['num_episodes']
        episode_rewards = []
        episode_lengths = []
        episode_info_list = []

        for episode in range(test_episodes):
            obs, info = env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            step = 0
            
            # Store episode info
            episode_info_list.append(info.get('setpoint_info', 'No info available'))
            print(f"Episode {episode+1}/{test_episodes}: {info.get('setpoint_info', 'No info available')}")

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step += 1
                obs = next_obs

                if step % 50 == 0 and config['evaluation']['verbose']:
                    print(f"  Step {step}, Action: {action}, Reward: {reward:.4f}")
                if step >= max_episode_steps:
                    print(f"  Warning: Episode {episode+1} reached max_episode_steps ({max_episode_steps})")
                    truncated = True

            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            print(f"  Episode {episode+1} completed. Reward: {episode_reward:.4f}, Length: {step}")

        # Enhanced evaluation plotting - now mode-aware
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].bar(range(1, test_episodes+1), episode_rewards)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title(f'Evaluation Results - {training_mode} Mode ({os.path.basename(model_path_to_test)})')
        axes[0, 0].grid(True, axis='y')
        
        # Episode lengths
        axes[0, 1].bar(range(1, test_episodes+1), episode_lengths, color='orange')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Length (steps)')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True, axis='y')
        
        # Reward vs Length scatter
        axes[1, 0].scatter(episode_lengths, episode_rewards, alpha=0.7)
        axes[1, 0].set_xlabel('Episode Length (steps)')
        axes[1, 0].set_ylabel('Total Reward')
        axes[1, 0].set_title('Reward vs Episode Length')
        axes[1, 0].grid(True)
        
        # Summary statistics
        axes[1, 1].axis('off')
        mode_specific_text = ""
        if args.csv_directory or args.csv_mode:
            mode_specific_text = f"CSV Mode: Real AUV data\nData-driven evaluation"
        else:
            mode_specific_text = "Online Mode: Random setpoints\nLive ROS2 evaluation"
            
        stats_text = f"""Evaluation Summary:
Episodes: {test_episodes}
Mode: {training_mode}
Avg Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}
Avg Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}
Best Reward: {max(episode_rewards):.2f}
Worst Reward: {min(episode_rewards):.2f}

{mode_specific_text}
Model trained with warmup period"""
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        mode_suffix = "csv" if (args.csv_directory or args.csv_mode) else "online"
        eval_plot_path = os.path.join(log_dir, f"evaluation_results_{mode_suffix}.png")
        plt.savefig(eval_plot_path, dpi=150, bbox_inches='tight')
        print(f"Enhanced evaluation plot saved to {eval_plot_path}")
        plt.close()

        # Save detailed evaluation report
        eval_report_path = os.path.join(log_dir, f"evaluation_report_{mode_suffix}.txt")
        with open(eval_report_path, 'w') as f:
            f.write(f"Evaluation Report - {training_mode} Mode\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {os.path.basename(model_path_to_test)}\n")
            f.write(f"Training Mode: {training_mode}\n")
            f.write(f"Episodes: {test_episodes}\n")
            f.write(f"Average Reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}\n")
            f.write(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}\n\n")
            
            if args.csv_directory or args.csv_mode:
                f.write(f"CSV Directory: {csv_dir}\n")
                if hasattr(env, 'csv_manager'):
                    stats = env.csv_manager.get_dataset_stats()
                    f.write(f"Dataset Duration: {stats.get('duration_seconds', 0):.1f}s\n")
                    f.write(f"Dataset Timesteps: {stats.get('total_timesteps', 0)}\n\n")
            
            f.write("Episode Details:\n")
            for i, (reward, length, info) in enumerate(zip(episode_rewards, episode_lengths, episode_info_list)):
                f.write(f"Episode {i+1}: Reward={reward:.4f}, Length={length}, {info}\n")

        print(f"\nEvaluation Summary:")
        print(f"Mode: {training_mode}")
        print(f"Number of episodes: {test_episodes}")
        print(f"Average Reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
        print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        if args.csv_directory or args.csv_mode:
            print(f"Evaluated using real AUV data from CSV files")
        else:
            print(f"Each episode tested with a different random setpoint")
        print(f"Model was trained with proper warmup period")
        print(f"Detailed report saved to: {eval_report_path}")

    # Clean up
    env.close()

if __name__ == "__main__":
    main()