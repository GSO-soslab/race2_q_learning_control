from collections import deque
import random
import numpy as np
import torch
class ReplayBuffer:
    """Experience replay buffer with separate states for actor and critic"""
    def __init__(self, actor_state_dim, critic_state_dim, buffer_capacity=100000, batch_size=128):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_capacity)
        self.actor_state_dim = actor_state_dim
        self.critic_state_dim = critic_state_dim
    
    def add(self, actor_state, critic_state, action, reward, next_actor_state, next_critic_state, done):
        """Add experience to buffer"""
        self.buffer.append((actor_state, critic_state, action, reward, next_actor_state, next_critic_state, done))
    
    def sample(self):
        """Sample a batch of experiences with separate state arrays"""
        batch = random.sample(self.buffer, self.batch_size)
        
        # Separate the experiences
        actor_states = np.array([experience[0] for experience in batch])
        critic_states = np.array([experience[1] for experience in batch])
        actions = np.array([experience[2] for experience in batch])
        rewards = np.array([experience[3] for experience in batch])
        next_actor_states = np.array([experience[4] for experience in batch])
        next_critic_states = np.array([experience[5] for experience in batch])
        dones = np.array([experience[6] for experience in batch])
        
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
        return len(self.buffer)