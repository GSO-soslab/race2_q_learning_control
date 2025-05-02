from collections import deque
import random
import numpy as np
import torch
class ReplayBuffer:
    """Experience replay buffer with separate states for actor and critic"""
    def __init__(self, actor_state_dim, critic_state_dim,n_actions, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.mem_cntr = 0

        # Pre-allocate memory
        self.actor_state_memory = np.zeros((self.buffer_capacity, actor_state_dim))
        self.critic_state_memory = np.zeros((self.buffer_capacity, critic_state_dim))
        self.action_memory = np.zeros((self.buffer_capacity, n_actions)) 
        self.reward_memory = np.zeros(self.buffer_capacity)
        self.next_actor_state_memory = np.zeros((self.buffer_capacity, actor_state_dim))
        self.next_critic_state_memory = np.zeros((self.buffer_capacity, critic_state_dim))
        self.done_memory = np.zeros(self.buffer_capacity, dtype=np.float32)
        
    def add(self, actor_state, critic_state, action, reward, next_actor_state, next_critic_state, done):
        """Store transition in the buffer"""
        index = self.mem_cntr % self.buffer_capacity
        
        self.actor_state_memory[index] = actor_state
        self.critic_state_memory[index] = critic_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_actor_state_memory[index] = next_actor_state
        self.next_critic_state_memory[index] = next_critic_state
        self.done_memory[index] = 1.0 - done  # Storing terminal flag (1 - done) like reference
        
        self.mem_cntr += 1
        
    def sample(self):
        """Sample a batch of experiences with separate state arrays"""
        max_mem = min(self.mem_cntr, self.buffer_capacity)
        batch_indices = np.random.choice(max_mem, self.batch_size)
        
        # Get samples from memory using indices
        actor_states = self.actor_state_memory[batch_indices]
        critic_states = self.critic_state_memory[batch_indices]
        actions = self.action_memory[batch_indices]
        rewards = self.reward_memory[batch_indices]
        next_actor_states = self.next_actor_state_memory[batch_indices]
        next_critic_states = self.next_critic_state_memory[batch_indices]
        dones = self.done_memory[batch_indices]
        
        # Convert to PyTorch tensors
        actor_states = torch.FloatTensor(actor_states)
        critic_states = torch.FloatTensor(critic_states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_actor_states = torch.FloatTensor(next_actor_states)
        next_critic_states = torch.FloatTensor(next_critic_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        return actor_states, critic_states, actions, rewards, next_actor_states, next_critic_states, dones
    
    def size(self):
        """Return the current size of the buffer"""
        return min(self.mem_cntr, self.buffer_capacity)