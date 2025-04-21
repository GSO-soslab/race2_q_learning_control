#!/usr/bin/env python3
import os
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from OUActionNoise import OUActionNoise
from ReplayBuffer import ReplayBuffer
from Actor import Actor
from Critic import Critic

from config_utils import load_config 


if __name__ == "__main__":
    config = load_config()
else:
    # When imported, the config will be passed to the class
    config = None

class DDPG:
    """DDPG Agent for AUV control using PyTorch"""
    def __init__(self, actor_state_dim, critic_state_dim, action_dim, action_bound, config , device="cuda" if torch.cuda.is_available() else "cpu"):
        self.actor_state_dim = actor_state_dim
        self.critic_state_dim = critic_state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device
        
        # Initialize actor and critic networks with separate state dimensions
        self.actor = Actor(actor_state_dim, action_dim, action_bound, config).to(device)
        self.actor_target = Actor(actor_state_dim, action_dim, action_bound, config).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(critic_state_dim, action_dim, config).to(device)
        self.critic_target = Critic(critic_state_dim, action_dim, config).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        critic_change = self.critic.check_if_stuck()
        print(f"Critic weight change: {critic_change:.8f}")

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        
        # Initialize replay buffer (modified to store both actor and critic states)
        self.buffer = ReplayBuffer(actor_state_dim, critic_state_dim)
        
        # Initialize noise process
        self.noise = OUActionNoise(
            mean=np.zeros(action_dim),
            std_deviation=0.1 * np.ones(action_dim)
        )
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001 # Target network update rate (0.01 for depth only)


    def update_learning_rates(self, episode, avg_recent_rewards, decay_factor=0.5, patience=100000):
        """
        Reduce learning rate when performance plateaus
        """
        if not hasattr(self, 'best_reward'):
            self.best_reward = float('-inf')
            self.plateau_counter = 0
        
        # Check if we've improved
        if avg_recent_rewards > self.best_reward:
            self.best_reward = avg_recent_rewards
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
        
        # If we've plateaued for 'patience' episodes, reduce learning rate
        if self.plateau_counter >= patience:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] *= decay_factor
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] *= decay_factor
            
            self.plateau_counter = 0  # Reset counter
            return True  # Return True if LR was updated
        
        return False
    
    def get_action(self, state, add_noise=True):
        """Get action from actor with optional noise for exploration"""
        state = torch.FloatTensor(state).to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0) 
        self.actor.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            action = self.actor(state).cpu().detach().numpy()
        action = np.reshape(action, -1) 
        self.actor.train()  # Back to training mode
        
        if add_noise:
            # Update noise standard deviation before adding noise
            self.noise.update_std()
            noise = self.noise()
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
        else:
            action = np.clip(action, -self.action_bound, self.action_bound)
            
        return action
    
    def remember(self, actor_state, critic_state, action, reward, next_actor_state, next_critic_state, done):
        """Store experience in replay buffer with separate states for actor and critic"""
        self.buffer.add(actor_state, critic_state, action, reward, next_actor_state, next_critic_state, done)
    
    def learn(self):
        """Update actor and critic networks from replay buffer"""
        if self.buffer.size() < self.buffer.batch_size:
            return None, None, None, None
        
        # Sample a batch from replay buffer
        actor_states, critic_states, actions, rewards, next_actor_states, next_critic_states, dones = self.buffer.sample()
        
        # Move tensors to device
        actor_states = actor_states.to(self.device)
        critic_states = critic_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_actor_states = next_actor_states.to(self.device)
        next_critic_states = next_critic_states.to(self.device)
        dones = dones.to(self.device)
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_actor_states)
            # next_q_values = self.critic_target(next_critic_states, next_actions)
            next_q_values = self.critic(critic_states, actions)
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
            # print("Target Q Values!!!", target_q)
        

        current_q = self.critic.forward(critic_states, actions)
        self.critic.train()
        self.critic_optimizer.zero_grad()
        critic_loss = nn.MSELoss()(target_q,current_q)
        
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor using deterministic policy gradient
        self.critic.eval()
        self.actor_optimizer.zero_grad()
        actions_pred = self.actor.forward(actor_states)  #mu
        self.actor.train()
        actor_loss = -self.critic.forward(critic_states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_targets()
        return critic_loss.item(), actor_loss.item(), rewards.mean().item(), current_q.mean().item()    
    
    def update_targets(self):
        """Soft update target networks"""
        # Update actor target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_weights(self, path):
        """Save model weights"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)
    
    def load_weights(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

