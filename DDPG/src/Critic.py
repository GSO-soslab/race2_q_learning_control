
import torch.nn as nn
import torch

# class Critic(nn.Module):
#     """Critic Network for DDPG using PyTorch, configured from config file"""
#     def __init__(self, state_dim, action_dim, config):
#         # Validate config is provided and contains required keys
#         if config is None:
#             raise ValueError("Configuration must be provided for Critic initialization")
#         if 'qnetwork' not in config:
#             raise ValueError("Configuration must contain 'qnetwork' key")
#         if 'critic_hidden_layers' not in config.get('qnetwork', {}):
#             raise ValueError("Configuration must specify 'critic_hidden_layers' in 'qnetwork'")
        
#         super(Critic, self).__init__()
#         hidden_dims = config.get('qnetwork', {}).get('critic_hidden_layers')
        
#         # Ensure there's at least one hidden layer for state and action
#         if len(hidden_dims) < 1:
#             raise ValueError("Critic must have at least one hidden layer")
        
#         # Separate processing for state and action
#         # Dynamically create state layers
#         state_layers = []
#         current_state_dim = state_dim
#         for hidden_dim in hidden_dims[:2]:  # Up to first two layers for state
#             state_layers.append(nn.Linear(current_state_dim, hidden_dim))
#             state_layers.append(nn.LayerNorm(hidden_dim))
#             state_layers.append(nn.ReLU())
#             current_state_dim = hidden_dim
        
#         self.state_layers = nn.Sequential(*state_layers)
        
#         # Action processing layer
#         self.action_layer = nn.Sequential(
#             nn.Linear(action_dim, hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0]),
#             nn.ReLU()
#         )
        
#         # Combined processing layers
#         combined_dim = hidden_dims[1] * 2 if len(hidden_dims) > 1 else sum(hidden_dims)
#         combined_layers = []
#         for hidden_dim in hidden_dims[2:] if len(hidden_dims) > 2 else []:
#             combined_layers.append(nn.Linear(combined_dim, hidden_dim))
#             combined_layers.append(nn.LayerNorm(hidden_dim))
#             combined_layers.append(nn.ReLU())
#             combined_dim = hidden_dim
        
#         # Final output layer with Tanh activation
#         combined_layers.append(nn.Linear(combined_dim, 1))
#         combined_layers.append(nn.Tanh())
        
#         self.combined_layers = nn.Sequential(*combined_layers)
        
#         # Initialize weights
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         """Initialize network weights with Xavier/Glorot initialization"""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 # Xavier/Glorot initialization for Linear layers
#                 nn.init.xavier_uniform_(module.weight)
#                 # Initialize bias to small values
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0.01)
                    
#     def check_if_stuck(self):
#         """One-liner to check if network weights are changing"""
#         return sum(p.grad.abs().mean().item() if p.grad is not None else 0 for p in self.parameters())

#     def forward(self, state, action):
#         state_features = self.state_layers(state)
#         action_features = self.action_layer(action)
#         combined = torch.cat([state_features, action_features], dim=1)
#         return self.combined_layers(combined)


class Critic(nn.Module):
    """Critic Network for DDPG using PyTorch with a truly unified architecture"""
    def __init__(self, state_dim, action_dim, config):
        # Validate config is provided and contains required keys
        if config is None:
            raise ValueError("Configuration must be provided for Critic initialization")
        if 'qnetwork' not in config:
            raise ValueError("Configuration must contain 'qnetwork' key")
        if 'critic_hidden_layers' not in config.get('qnetwork', {}):
            raise ValueError("Configuration must specify 'critic_hidden_layers' in 'qnetwork'")
        
        super(Critic, self).__init__()
        hidden_dims = config.get('qnetwork', {}).get('critic_hidden_layers')
        
        # Ensure there's at least one hidden layer
        if len(hidden_dims) < 1:
            raise ValueError("Critic must have at least one hidden layer")
        
        # Input layer takes both state and action as input
        self.input_layer = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0])
        self.input_activation = nn.LeakyReLU()
        
        # Hidden layers
        hidden_layers = []
        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            hidden_layers.append(nn.LayerNorm(hidden_dims[i+1]))
            hidden_layers.append(nn.LeakyReLU())
        
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        # self.output_activation = nn.Tanh()
        
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
                    
    def check_if_stuck(self):
        """One-liner to check if network weights are changing"""
        return sum(p.grad.abs().mean().item() if p.grad is not None else 0 for p in self.parameters())
    
    def forward(self, state, action):
        # Concatenate state and action once at input
        x = torch.cat([state, action], dim=1)
        
        # Process through network
        x = self.input_activation(self.input_norm(self.input_layer(x)))
        x = self.hidden_layers(x)
        # x = self.output_activation(self.output_layer(x))
        x = self.output_layer(x) 
        
        return x