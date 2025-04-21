
import torch.nn as nn
import torch

class Actor(nn.Module):
    """Actor Network for DDPG using PyTorch, configured from config file"""
    def __init__(self, state_dim, action_dim, action_bound, config):
        # Validate config is provided and contains required keys
        if config is None:
            raise ValueError("Configuration must be provided for Actor initialization")
        if 'qnetwork' not in config:
            raise ValueError("Configuration must contain 'qnetwork' key")
        if 'actor_hidden_layers' not in config.get('qnetwork', {}):
            raise ValueError("Configuration must specify 'actor_hidden_layers' in 'qnetwork'")
        
        super(Actor, self).__init__()
        self.action_bound = action_bound
        hidden_dims = config.get('qnetwork', {}).get('actor_hidden_layers')
        
        # Dynamically create layers
        layers = []
        current_dim = state_dim
        
        # Create hidden layers dynamically
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU())
            current_dim = hidden_dim
        
        # Create network
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for Linear layers
                nn.init.xavier_uniform_(module.weight)
                # Initialize bias to small values
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
        
        # Special initialization for the final output layer
        # Using smaller values for the final layer can help with initial exploration
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)
    
    def forward(self, state):
        x = self.layers(state)
        x = torch.tanh(self.output_layer(x))
        return x * self.action_bound