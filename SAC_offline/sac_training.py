import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import argparse
from pathlib import Path
import wandb
from collections import deque
import random
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

class AUVDataset(Dataset):
    """Dataset for AUV training data from CSV"""
    
    def __init__(self, csv_file, device='cpu'):
        print(f"Loading dataset from {csv_file}...")
        self.data = pd.read_csv(csv_file)
        self.device = device
        
        # Validate data
        print(f"Dataset shape: {self.data.shape}")
        print(f"Duration: {self.data['timestamp'].max() - self.data['timestamp'].min():.1f}s")
        
        # Extract column groups
        self.curr_cols = sorted([c for c in self.data.columns if c.startswith('curr_')])
        self.next_cols = sorted([c for c in self.data.columns if c.startswith('next_')])
        self.action_cols = sorted([c for c in self.data.columns if c.startswith('action_')])
        
        print(f"State dimensions: {len(self.curr_cols)} (current) + {len(self.next_cols)} (next)")
        print(f"Action dimensions: {len(self.action_cols)}")
        
        # Convert to tensors - KEEP ON CPU for DataLoader compatibility
        self.current_states = torch.FloatTensor(self.data[self.curr_cols].values)
        self.next_states = torch.FloatTensor(self.data[self.next_cols].values)
        self.actions = torch.FloatTensor(self.data[self.action_cols].values)
        self.rewards = torch.FloatTensor(self.data['reward'].values)
        
        # Calculate statistics on CPU tensors
        self.state_mean = self.current_states.mean(dim=0)
        self.state_std = self.current_states.std(dim=0) + 1e-8
        self.action_mean = self.actions.mean(dim=0)
        self.action_std = self.actions.std(dim=0) + 1e-8
        
        print(f"✅ Dataset loaded: {len(self)} samples")
        print(f"Reward range: [{self.rewards.min():.3f}, {self.rewards.max():.3f}]")
        print(f"Reward mean: {self.rewards.mean():.3f} ± {self.rewards.std():.3f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return tensors on CPU - DataLoader will handle GPU transfer
        return {
            'current_state': self.current_states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'next_state': self.next_states[idx],
            'done': torch.tensor(0.0)  # Assume no terminal states in continuous data
        }
    
    def get_normalization_params(self):
        """Get normalization parameters for the model"""
        return {
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std
        }

class OfflineSACTrainer:
    """Offline SAC trainer for AUV data"""
    
    def __init__(self, config_path=None, device='auto'):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        self._setup_logging()
        
        # Device setup
        self.device = self._setup_device(device)
        
        # Setup reproducibility
        self._setup_reproducibility()
        
        # Setup checkpoint directory with timestamp (must be before tensorboard)
        self._setup_checkpoint_dir()
        
        # Extract config sections
        self.data_config = self.config.get('data', {})
        self.model_config = self.config.get('model', {})
        self.sac_config = self.config.get('sac', {})
        self.training_config = self.config.get('training', {})
        self.logging_config = self.config.get('logging', {})
        
        # Training parameters from config
        self.batch_size = self.training_config.get('batch_size', 256)
        self.num_epochs = self.training_config.get('num_epochs', 200)
        self.validate_every = self.training_config.get('validate_every', 5)
        self.save_every = self.training_config.get('save_every', 25)
        
        # SAC parameters from config
        self.actor_lr = self.sac_config.get('actor_lr', 3e-4)
        self.critic_lr = self.sac_config.get('critic_lr', 3e-4)
        self.alpha_lr = self.sac_config.get('alpha_lr', 3e-4)
        self.tau = self.sac_config.get('tau', 0.005)
        self.gamma = self.sac_config.get('gamma', 0.99)
        self.alpha = self.sac_config.get('alpha', 0.2)
        self.auto_entropy_tuning = self.sac_config.get('auto_entropy_tuning', True)
        
        # Initialize logging (after checkpoint dir is set up)
        self.writer = None
        self.wandb_run = None
        self._setup_tensorboard()
        self._setup_wandb()
        
        # Model will be initialized after we know dimensions
        self.model = None
        self.dataset = None
        
        self.logger.info(f"SAC Trainer initialized with device: {self.device}")
        self.logger.info(f"Run name: {self.run_name}")
        self.logger.info(f"Checkpoints: {self.checkpoint_dir}")
        self.logger.info(f"TensorBoard: {self.tensorboard_dir}")
    
    def _setup_checkpoint_dir(self):
        """Setup checkpoint directory with timestamp"""
        from datetime import datetime
        
        checkpoint_config = self.config.get('checkpointing', {})
        base_dir = checkpoint_config.get('save_dir', 'runs/checkpoints')
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"sac_training_{timestamp}"
        self.checkpoint_dir = os.path.join(base_dir, self.run_name)
        self.tensorboard_dir = os.path.join('runs', self.run_name, 'tensorboard')
        
        # Create directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save config to checkpoint directory
        config_path = os.path.join(self.checkpoint_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"📁 Checkpoint directory created: {self.checkpoint_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {}).get('console', {})
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SAC_Trainer')
    
    def _setup_device(self, device):
        """Setup training device"""
        device_config = self.config.get('device', {})
        
        if device == 'auto' and device_config.get('auto_select', True):
            if torch.cuda.is_available():
                device_id = device_config.get('cuda_device_id', 0)
                selected_device = torch.device(f'cuda:{device_id}')
            else:
                selected_device = torch.device('cpu')
        else:
            selected_device = torch.device(device)
        
        self.logger.info(f"Using device: {selected_device}")
        return selected_device
    
    def _setup_reproducibility(self):
        """Setup reproducibility settings"""
        repro_config = self.config.get('reproducibility', {})
        seed = repro_config.get('seed', 42)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        if repro_config.get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = repro_config.get('benchmark', False)
        
        self.logger.info(f"Reproducibility setup with seed: {seed}")
    
    def _setup_tensorboard(self):
        """Setup tensorboard logging"""
        tb_config = self.logging_config.get('tensorboard', {})
        
        if tb_config.get('enabled', True):
            log_dir = tb_config.get('log_dir', self.tensorboard_dir)
            self.writer = SummaryWriter(log_dir=log_dir)
            self.logger.info(f"Tensorboard logging enabled: {log_dir}")
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb_config = self.logging_config.get('wandb', {})
        
        if wandb_config.get('enabled', False):
            self.wandb_run = wandb.init(
                project=wandb_config.get('project', 'auv-sac-offline'),
                entity=wandb_config.get('entity'),
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', ''),
                config=self.config
            )
            self.logger.info("Wandb logging enabled")
        
    def _load_config(self, config_path):
        """Load configuration file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                print(f"✅ Loaded config from: {config_path}")
                return config
        else:
            print("⚠️  No config file found, using minimal defaults")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return minimal default configuration"""
        return {
            'data': {'train_split': 0.8, 'normalize_states': True},
            'model': {
                'actor': {'hidden_layers': [256, 256]},
                'critic': {'hidden_layers': [256, 256]}
            },
            'sac': {
                'actor_lr': 3e-4, 'critic_lr': 3e-4, 'alpha_lr': 3e-4,
                'gamma': 0.99, 'tau': 0.005, 'alpha': 0.2
            },
            'training': {
                'batch_size': 256, 'num_epochs': 100,
                'validate_every': 10, 'save_every': 25
            },
            'logging': {'tensorboard': {'enabled': True}},
            'device': {'auto_select': True}
        }
    
    def load_data(self, csv_file):
        """Load and split data"""
        self.logger.info(f"Loading data from: {csv_file}")
        
        # Load full dataset - keep on CPU
        full_dataset = AUVDataset(csv_file, device='cpu')
        
        # Split into train/validation
        train_split = self.data_config.get('train_split', 0.8)
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        # Use torch's random_split for proper shuffling
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.get('reproducibility', {}).get('seed', 42))
        )
        
        self.logger.info(f"Data split: {train_size} train, {val_size} validation")
        
        # DataLoader configuration
        dataloader_config = self.config.get('dataloader', {})
        
        # Create dataloaders with proper pin_memory settings
        pin_memory = dataloader_config.get('pin_memory', True) and self.device.type == 'cuda'
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.data_config.get('shuffle_data', True),
            num_workers=dataloader_config.get('num_workers', 0),
            pin_memory=pin_memory,
            drop_last=dataloader_config.get('drop_last', False)
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation
            num_workers=dataloader_config.get('num_workers', 0),
            pin_memory=pin_memory
        )
        
        # Store dataset info
        self.dataset = full_dataset
        self.state_dim = len(full_dataset.curr_cols)
        self.action_dim = len(full_dataset.action_cols)
        
        self.logger.info(f"Model dimensions: state={self.state_dim}, action={self.action_dim}")
        
        return full_dataset.get_normalization_params()
    
    def init_model(self):
        """Initialize SAC model using config parameters"""
        self.logger.info("Initializing SAC model from config...")
        
        # Get model configuration
        actor_config = self.model_config.get('actor', {})
        critic_config = self.model_config.get('critic', {})
        
        # Create networks
        self.actor = ActorNetwork(
            self.state_dim, 
            self.action_dim, 
            hidden_layers=actor_config.get('hidden_layers', [256, 256]),
            activation=actor_config.get('activation', 'relu'),
            dropout=actor_config.get('dropout', 0.0)
        ).to(self.device)
        
        self.critic1 = CriticNetwork(
            self.state_dim, 
            self.action_dim,
            hidden_layers=critic_config.get('hidden_layers', [256, 256]),
            activation=critic_config.get('activation', 'relu'),
            dropout=critic_config.get('dropout', 0.0)
        ).to(self.device)
        
        self.critic2 = CriticNetwork(
            self.state_dim, 
            self.action_dim,
            hidden_layers=critic_config.get('hidden_layers', [256, 256]),
            activation=critic_config.get('activation', 'relu'),
            dropout=critic_config.get('dropout', 0.0)
        ).to(self.device)
        
        # Create target networks
        self.target_critic1 = CriticNetwork(
            self.state_dim, 
            self.action_dim,
            hidden_layers=critic_config.get('hidden_layers', [256, 256]),
            activation=critic_config.get('activation', 'relu'),
            dropout=critic_config.get('dropout', 0.0)
        ).to(self.device)
        
        self.target_critic2 = CriticNetwork(
            self.state_dim, 
            self.action_dim,
            hidden_layers=critic_config.get('hidden_layers', [256, 256]),
            activation=critic_config.get('activation', 'relu'),
            dropout=critic_config.get('dropout', 0.0)
        ).to(self.device)
        
        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Setup optimizers
        optimizer_config = self.sac_config.get('optimizer', 'adam')
        weight_decay = float(self.sac_config.get('weight_decay', 1e-4))
        betas = self.sac_config.get('betas', [0.9, 0.999])
        eps = float(self.sac_config.get('eps', 1e-8))

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.actor_lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), 
            lr=self.critic_lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), 
            lr=self.critic_lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
        
        # Automatic entropy tuning
        if self.auto_entropy_tuning:
            target_entropy_scale = self.sac_config.get('target_entropy_scale', 1.0)
            self.target_entropy = -target_entropy_scale * self.action_dim
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None
        
        self.logger.info("SAC model initialized successfully")

    def train_offline(self):
        """Train SAC model offline using the CSV data with full logging"""
        
        self.logger.info("Starting offline SAC training...")
        self.logger.info(f"Configuration: {self.num_epochs} epochs, batch size {self.batch_size}")
        
        # Initialize training tracking
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_config = self.training_config.get('early_stopping', {})
        early_stopping_enabled = early_stopping_config.get('enabled', True)
        patience = int(early_stopping_config.get('patience', 20))
        min_delta = float(early_stopping_config.get('min_delta', 1e-4))
        
        # Training loop
        console_config = self.logging_config.get('console', {})
        print_every = console_config.get('print_every', 5)
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch  # Track current epoch for saving
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = {}
            if epoch % self.validate_every == 0:
                val_metrics = self._validate_epoch()
                
                # Logging
                if epoch % print_every == 0:
                    self.logger.info(
                        f"Epoch {epoch:3d}: "
                        f"Actor: {train_metrics['actor_loss']:.4f}, "
                        f"Critic: {train_metrics['critic_loss']:.4f}, "
                        f"Alpha: {train_metrics['alpha']:.3f}, "
                        f"Val Loss: {val_metrics.get('loss', 0):.4f}"
                    )
                
                # Tensorboard logging
                if self.writer:
                    self._log_to_tensorboard(epoch, train_metrics, val_metrics)
                
                # Wandb logging
                if self.wandb_run:
                    self._log_to_wandb(epoch, train_metrics, val_metrics)
                
                # Early stopping check
                current_val_loss = val_metrics.get('loss', float('inf'))
                if early_stopping_enabled:
                    if current_val_loss < best_val_loss - min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        self.save_model('best_model.pth')
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping triggered after {epoch} epochs")
                        break
                elif current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    self.save_model('best_model.pth')
            
            # Periodic saves
            if epoch % self.save_every == 0 and epoch > 0:
                self.save_model(f'checkpoint_epoch_{epoch:04d}.pth')
        
        self.logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        self.save_model('final_model.pth')
        
        # Print summary
        print(f"\n🎉 Training completed!")
        print(f"📂 Run directory: runs/{self.run_name}/")
        print(f"🏆 Best model: {os.path.join(self.checkpoint_dir, 'best_model.pth')}")
        print(f"📊 View results: tensorboard --logdir runs/{self.run_name}/tensorboard")
        
        # Cleanup
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()

    def _train_epoch(self):
        """Train for one epoch - following the 12 SAC steps exactly"""
        total_actor_loss = 0
        total_critic_loss = 0
        total_alpha_loss = 0
        num_batches = 0
        
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        
        for batch in self.train_loader:
            # Step 1: Sample batch from offline dataset (s_i, a_i, r_i, s'_i) ~ D
            current_states = batch['current_state'].to(self.device)  # s_i
            actions = batch['action'].to(self.device)                # a_i  
            rewards = batch['reward'].unsqueeze(1).to(self.device)   # r_i
            next_states = batch['next_state'].to(self.device)        # s'_i
            dones = batch['done'].unsqueeze(1).to(self.device)       # terminal flags
            
            # Step 2: Actor inference on next states (for critic target)
            with torch.no_grad():
                next_actions, next_log_probs, _ = self.actor.sample(next_states)  # a'_i ~ π_φ(a|s'_i)
                
                # Step 3: Target Q-value (Bellman target)
                target_q1 = self.target_critic1(next_states, next_actions)  # Q_θ̄₁(s'_i, a'_i)
                target_q2 = self.target_critic2(next_states, next_actions)  # Q_θ̄₂(s'_i, a'_i)
                target_q_min = torch.min(target_q1, target_q2)              # min(Q_θ̄₁, Q_θ̄₂)
                
                # y_i = r_i + γ(min(Q_θ̄₁(s'_i, a'_i), Q_θ̄₂(s'_i, a'_i)) - α log π_φ(a'_i | s'_i))
                target_q = rewards + (1 - dones) * self.gamma * (target_q_min - self.alpha * next_log_probs)
            
            # Step 4: Critic forward pass on current data
            current_q1 = self.critic1(current_states, actions)  # Q_θ₁(s_i, a_i)
            current_q2 = self.critic2(current_states, actions)  # Q_θ₂(s_i, a_i)
            
            # Step 5: Critic losses
            critic1_loss = nn.MSELoss()(current_q1, target_q)  # L_Q₁ = 1/B Σ(Q_θ₁(s_i, a_i) - y_i)²
            critic2_loss = nn.MSELoss()(current_q2, target_q)  # L_Q₂ = 1/B Σ(Q_θ₂(s_i, a_i) - y_i)²
            
            # Step 6: Backpropagate critic losses
            # θ₁ ← θ₁ - η_Q ∇_θ₁ L_Q₁
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            
            # θ₂ ← θ₂ - η_Q ∇_θ₂ L_Q₂  
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()
            
            # Step 7: Actor inference on current states
            new_actions, log_probs, _ = self.actor.sample(current_states)  # a_i^new ~ π_φ(a|s_i)
            
            # Step 8: Actor Q-value evaluation  
            q1_new = self.critic1(current_states, new_actions)    # Q_θ₁(s_i, a_i^new)
            q2_new = self.critic2(current_states, new_actions)    # Q_θ₂(s_i, a_i^new)
            q_min = torch.min(q1_new, q2_new)                    # Q_min,i = min(Q_θ₁, Q_θ₂)
            
            # Step 9: Actor (policy) loss
            # L_π = 1/B Σ[α log π_φ(a_i^new | s_i) - Q_min,i]
            actor_loss = (self.alpha * log_probs - q_min).mean()
            
            # Step 10: Backpropagate actor loss  
            # φ ← φ - η_π ∇_φ L_π
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Step 11: Temperature loss and update (if α is learnable)
            # L_α = 1/B Σ α(-log π_φ(a_i^new | s_i) - H_target)
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                
                # α ← α - η_α ∇_α L_α
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.tensor(0.0)
            
            # Step 12: Target network update (Polyak averaging)
            # θ̄_j ← τθ_j + (1-τ)θ̄_j, j=1,2
            self._soft_update_target_networks()
            
            # Accumulate losses for logging
            total_actor_loss += actor_loss.item()
            total_critic_loss += (critic1_loss.item() + critic2_loss.item()) / 2
            total_alpha_loss += alpha_loss.item()
            num_batches += 1
        
        return {
            'actor_loss': total_actor_loss / num_batches,
            'critic_loss': total_critic_loss / num_batches, 
            'alpha_loss': total_alpha_loss / num_batches,
            'alpha': self.alpha.item(),
            'loss': (total_actor_loss + total_critic_loss) / num_batches
        }

    def _validate_epoch(self):
        """Validate for one epoch"""
        total_q_loss = 0
        total_reward = 0
        num_batches = 0
        
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                current_states = batch['current_state'].to(self.device)
                actions = batch['action'].to(self.device)
                rewards = batch['reward'].to(self.device)
                
                q1 = self.critic1(current_states, actions)
                q2 = self.critic2(current_states, actions)
                q_pred = torch.min(q1, q2).squeeze()
                
                q_loss = nn.MSELoss()(q_pred, rewards)
                total_q_loss += q_loss.item()
                total_reward += rewards.mean().item()
                num_batches += 1
        
        return {
            'q_loss': total_q_loss / num_batches,
            'avg_reward': total_reward / num_batches,
            'loss': total_q_loss / num_batches
        }

    def _soft_update_target_networks(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _log_to_tensorboard(self, epoch, train_metrics, val_metrics):
        """Log metrics to tensorboard"""
        tb_config = self.logging_config.get('tensorboard', {})
        log_every = tb_config.get('log_every', 1)
        
        if epoch % log_every == 0:
            # Training metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            # Validation metrics
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Log histograms if enabled
            if tb_config.get('log_histograms', True):
                for name, param in self.actor.named_parameters():
                    self.writer.add_histogram(f'Actor/{name}', param, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Actor/{name}_grad', param.grad, epoch)
                
                for name, param in self.critic1.named_parameters():
                    self.writer.add_histogram(f'Critic1/{name}', param, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Critic1/{name}_grad', param.grad, epoch)
    
    def _log_to_wandb(self, epoch, train_metrics, val_metrics):
        """Log metrics to Weights & Biases"""
        log_dict = {'epoch': epoch}
        
        # Add training metrics
        for key, value in train_metrics.items():
            log_dict[f'train_{key}'] = value
        
        # Add validation metrics
        for key, value in val_metrics.items():
            log_dict[f'val_{key}'] = value
        
        self.wandb_run.log(log_dict)
    
    def save_model(self, filename):
        """Save the trained model to checkpoint directory"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': getattr(self, 'current_epoch', 0),
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            'log_alpha': self.log_alpha,
            'alpha': self.alpha,
            'config': self.config,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'normalization_params': getattr(self.dataset, 'get_normalization_params', lambda: {})()
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"💾 Model saved: {filepath}")
        
        # Create symlink to latest checkpoint
        if 'best' in filename or 'final' in filename:
            latest_path = os.path.join(self.checkpoint_dir, f"latest_{filename}")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(filepath), latest_path)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Recreate networks
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.init_model()
        
        # Load state dicts
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        # Load optimizers
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        if checkpoint['alpha_optimizer'] and self.alpha_optimizer:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        self.log_alpha = checkpoint['log_alpha']
        if self.log_alpha is not None:
            self.alpha = self.log_alpha.exp()
        
        self.logger.info(f"📁 Model loaded: {filepath}")
        return checkpoint.get('epoch', 0)
    
    def predict(self, state):
        """Predict action for given state"""
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, _, action = self.actor.sample(state_tensor)
            return action.cpu().numpy()[0]


class ActorNetwork(nn.Module):
    """SAC Actor Network with configurable architecture"""
    
    def __init__(self, state_dim, action_dim, hidden_layers=[256, 256], activation='relu', dropout=0.0):
        super(ActorNetwork, self).__init__()
        
        # Get activation function
        activation_fn = self._get_activation(activation)
        
        # Build hidden layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)
        
        # Action bounds for AUV thrusters/servos
        self.action_scale = 1.0
        self.action_bias = 0.0
    
    def _get_activation(self, activation):
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU,
            'swish': nn.SiLU
        }
        return activations.get(activation.lower(), nn.ReLU)
    
    def forward(self, state):
        x = self.network(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        
        # Enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


class CriticNetwork(nn.Module):
    """SAC Critic Network with configurable architecture"""
    
    def __init__(self, state_dim, action_dim, hidden_layers=[256, 256], activation='relu', dropout=0.0):
        super(CriticNetwork, self).__init__()
        
        # Get activation function
        activation_fn = self._get_activation(activation)
        
        # Build network
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU,
            'swish': nn.SiLU
        }
        return activations.get(activation.lower(), nn.ReLU)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


def main():
    """Main training function with config-driven execution"""
    parser = argparse.ArgumentParser(description="Train SAC from CSV data")
    parser.add_argument("csv_file", help="Path to training CSV file")
    parser.add_argument("-c", "--config", help="Config file path", 
                   default="config/config_sac.yaml")
    parser.add_argument("--device", help="Device (cpu/cuda/auto)", default="auto")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable tensorboard logging")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"❌ CSV file not found: {args.csv_file}")
        return 1
    
    try:
        # Initialize trainer
        trainer = OfflineSACTrainer(config_path=args.config, device=args.device)
        
        # Override config with command line args
        if args.epochs:
            trainer.num_epochs = args.epochs
        if args.batch_size:
            trainer.batch_size = args.batch_size
        if args.lr:
            trainer.actor_lr = trainer.critic_lr = trainer.alpha_lr = args.lr
        if args.no_tensorboard and trainer.writer:
            trainer.writer.close()
            trainer.writer = None
        if args.wandb:
            if trainer.wandb_run is None:
                trainer._setup_wandb()
        if args.debug:
            trainer.logger.setLevel(logging.DEBUG)
        
        # Load data
        trainer.logger.info("Loading and preprocessing data...")
        norm_params = trainer.load_data(args.csv_file)
        
        # Initialize model
        trainer.logger.info("Initializing SAC model...")
        trainer.init_model()
        
        # Log model info
        total_params = sum(p.numel() for p in trainer.actor.parameters())
        total_params += sum(p.numel() for p in trainer.critic1.parameters())
        total_params += sum(p.numel() for p in trainer.critic2.parameters())
        trainer.logger.info(f"Total parameters: {total_params:,}")
        
        # Train
        trainer.logger.info("Starting training...")
        trainer.train_offline()
        
        trainer.logger.info("🎉 Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())