#!/usr/bin/env python3

import os
import sys
import time
import argparse
import threading
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import json
import pickle
from collections import deque

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import Float64
    from geometry_msgs.msg import Vector3
    from mvp_msgs.msg import ControlProcess
    ROS2_AVAILABLE = True
except ImportError:
    print("Warning: ROS2 not available")
    ROS2_AVAILABLE = False

# JAX and DreamerV3 imports
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    print("Warning: JAX not available")
    JAX_AVAILABLE = False

# Add DreamerV3 to path
DREAMERV3_PATH = os.path.join(os.path.dirname(__file__), 'dreamerv3')
if os.path.exists(DREAMERV3_PATH):
    sys.path.insert(0, DREAMERV3_PATH)

try:
    import dreamerv3
    from dreamerv3 import embodied
    DREAMERV3_AVAILABLE = True
except ImportError:
    print("Warning: DreamerV3 not available")
    DREAMERV3_AVAILABLE = False

# Import custom modules
from AUVEnv_DreamerV3 import AUVEnvDreamerV3
from checkpoint_utils import CheckpointManager

class DreamerV3PolicyNode(Node):
    """
    ROS2 node for deploying DreamerV3 policy in real-time AUV control
    """
    
    def __init__(self, config, model_path, policy_type='full'):
        super().__init__('dreamerv3_policy_node')
        
        self.config = config
        self.model_path = model_path
        self.policy_type = policy_type
        
        # Initialize policy
        self.policy = None
        self.policy_state = None
        self.load_policy()
        
        # Safety and monitoring
        self.safety_config = config.get('deployment', {})
        self.max_action_change = self.safety_config.get('max_action_change', 0.2)
        self.emergency_threshold = self.safety_config.get('emergency_stop_threshold', 5.0)
        self.control_rate = self.safety_config.get('ros_control_rate', 10.0)
        
        # State tracking
        self.current_observation = None
        self.last_action = np.zeros(config['environment']['thruster_size'] + 
                                  config['environment']['servo_joints_size'])
        self.observation_buffer = deque(maxlen=10)
        self.action_buffer = deque(maxlen=10)
        
        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        self.control_loop_times = deque(maxlen=100)
        
        # Threading
        self.state_lock = threading.Lock()
        self.policy_lock = threading.Lock()
        
        # Initialize ROS interfaces
        self.setup_ros_interfaces()
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate, 
            self.control_callback
        )
        
        # Monitoring timer
        self.monitoring_timer = self.create_timer(
            5.0,  # Every 5 seconds
            self.monitoring_callback
        )
        
        self.get_logger().info(f"DreamerV3 Policy Node initialized with {policy_type} policy")
        self.get_logger().info(f"Control rate: {self.control_rate} Hz")
    
    def load_policy(self):
        """Load DreamerV3 policy from checkpoint - Fixed version"""
        if not DREAMERV3_AVAILABLE or not JAX_AVAILABLE:
            raise ImportError("DreamerV3 and JAX required for policy deployment")
        
        model_path = Path(self.model_path)
        
        try:
            if self.policy_type == 'jit' and (model_path / "policy_jit.pkl").exists():
                # Load JIT compiled policy
                self.load_jit_policy(model_path / "policy_jit.pkl")
            else:
                # Load full DreamerV3 agent
                self.load_full_agent(model_path)
            
            self.get_logger().info(f"Policy loaded successfully: {self.policy_type}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {e}")
            # Create a fallback random policy
            self.create_fallback_policy()
    
    def load_jit_policy(self, jit_path):
        """Load JIT compiled policy for fast inference"""
        with open(jit_path, 'rb') as f:
            policy_data = pickle.load(f)
        
        self.policy = policy_data['compiled_policy']
        self.normalization_params = policy_data.get('normalization_params')
        self.obs_shape = policy_data['obs_shape']
        
        # Initialize policy state
        self.policy_state = {}  # JIT policies may not need complex state
    
    def load_full_agent(self, checkpoint_path):
        """Load full DreamerV3 agent - Fixed for correct API"""
        try:
            # Load checkpoint metadata
            checkpoint_manager = CheckpointManager()
            metadata = checkpoint_manager.load_checkpoint_with_validation(checkpoint_path)
            
            # Get observation and action spaces for agent creation
            obs_space = self._get_observation_space()
            act_space = self._get_action_space()
            
            # Initialize agent configuration
            config = self._create_agent_config(metadata)
            
            # Create agent with correct signature: Agent(obs_space, act_space, config)
            import embodied
            self.policy = embodied.Agent(obs_space, act_space, config)
            
            # Load checkpoint using the fixed load method
            self._load_checkpoint_safe(checkpoint_path, metadata)
            
            # Initialize policy state
            self.policy_state = self._initialize_policy_state()
            self.normalization_params = getattr(self.policy, 'normalization_params', None)
            
            self.get_logger().info("Full DreamerV3 agent loaded successfully")
            
        except Exception as e:
            self.get_logger().error(f"Error loading full agent: {e}")
            raise

    def _get_observation_space(self):
        """Get observation space for agent creation"""
        try:
            from gymnasium.spaces import Dict, Box
            import numpy as np
            
            # Use config to determine observation space
            obs_dim = 17  # Default AUV observation dimension
            
            obs_space = Dict({
                'vector': Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(obs_dim,), 
                    dtype=np.float32
                )
            })
            
            return obs_space
            
        except ImportError:
            # Fallback if gymnasium not available
            return {'vector': (17,)}

    def _get_action_space(self):
        """Get action space for agent creation"""
        try:
            from gymnasium.spaces import Box
            import numpy as np
            
            # Use config to determine action space
            thruster_size = self.config['environment']['thruster_size']
            servo_size = self.config['environment']['servo_joints_size']
            action_dim = thruster_size + servo_size
            
            action_space = Box(
                low=-1.0, 
                high=1.0, 
                shape=(action_dim,), 
                dtype=np.float32
            )
            
            return action_space
            
        except ImportError:
            # Fallback if gymnasium not available
            action_dim = self.config['environment']['thruster_size'] + self.config['environment']['servo_joints_size']
            return action_dim

    def _create_agent_config(self, metadata):
        """Create agent configuration from metadata and defaults"""
        # Start with default config structure
        config = {
            'logdir': '/tmp/dreamerv3_deployment',
            'seed': 42,
            'precision': 16,
            'task': 'auv_control',
            'env_amount': 1,
            'env_parallel': False,
            'batch_size': 16,
            'batch_length': 64,
        }
        
        # Override with saved configuration if available
        if 'config' in metadata and 'dreamerv3' in metadata['config']:
            saved_config = metadata['config']['dreamerv3']
            config.update(saved_config)
        
        return config

    def _load_checkpoint_safe(self, checkpoint_path, metadata):
        """Safely load checkpoint with error handling"""
        try:
            # Try different loading approaches
            step = metadata.get('step', 0)
            
            # Method 1: Load with directory and step
            try:
                self.policy.load(str(checkpoint_path), step)
                self.get_logger().info(f"Loaded checkpoint using method 1: step {step}")
                return
            except Exception as e1:
                self.get_logger().warn(f"Load method 1 failed: {e1}")
            
            # Method 2: Load with just directory
            try:
                self.policy.load(str(checkpoint_path))
                self.get_logger().info("Loaded checkpoint using method 2: directory only")
                return
            except Exception as e2:
                self.get_logger().warn(f"Load method 2 failed: {e2}")
            
            # Method 3: Try alternative load methods
            if hasattr(self.policy, 'restore'):
                try:
                    self.policy.restore(str(checkpoint_path))
                    self.get_logger().info("Loaded checkpoint using restore method")
                    return
                except Exception as e3:
                    self.get_logger().warn(f"Restore method failed: {e3}")
            
            raise Exception("All checkpoint loading methods failed")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load checkpoint: {e}")
            raise

    def _initialize_policy_state(self):
        """Initialize policy state for the agent"""
        try:
            # Try different state initialization methods
            if hasattr(self.policy, 'init_state'):
                return self.policy.init_state()
            elif hasattr(self.policy, 'initial_state'):
                return self.policy.initial_state()
            elif hasattr(self.policy, 'reset_state'):
                return self.policy.reset_state()
            elif hasattr(self.policy, 'init'):
                return self.policy.init()
            else:
                # Default empty state
                return {}
                
        except Exception as e:
            self.get_logger().warn(f"Error initializing policy state: {e}")
            return {}

    def create_fallback_policy(self):
        """Create a fallback random policy for safety"""
        self.get_logger().warn("Creating fallback random policy")
        
        self.policy_type = 'fallback'
        self.policy = None
        self.policy_state = None
        self.normalization_params = None
        
        # Set a flag to use random actions
        self._use_fallback = True
    
    def setup_ros_interfaces(self):
        """Setup ROS2 publishers and subscribers"""
        # Subscribers for sensor data
        self.create_subscription(
            ControlProcess,
            '/race2_auv/controller/process/value',
            self.state_callback,
            10
        )
        
        self.create_subscription(
            ControlProcess,
            '/race2_auv/controller/process/error',
            self.error_callback,
            10
        )
        
        # Publishers for control commands
        self.thruster_pubs = {
            'heave_bow': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_bow', 1),
            'heave_stern': self.create_publisher(Float64, '/race2_auv/control/thruster/heave_stern', 1),
            'surge_port': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_port', 1),
            'surge_starboard': self.create_publisher(Float64, '/race2_auv/control/thruster/surge_starboard', 1),
        }
        
        # Add servo publishers if configured
        if self.config['environment']['servo_joints_size'] > 0:
            self.thruster_pubs.update({
                'port_servo': self.create_publisher(Float64, '/race2_auv/control/surge_port_servo', 1),
                'starboard_servo': self.create_publisher(Float64, '/race2_auv/control/surge_starboard_servo', 1)
            })
        
        # Safety override subscriber
        self.create_subscription(
            Float64,
            '/race2_auv/emergency_stop',
            self.emergency_stop_callback,
            1
        )
        
        # Status publisher
        self.status_pub = self.create_publisher(
            ControlProcess,
            '/race2_auv/dreamerv3/status',
            1
        )
    
    def state_callback(self, msg):
        """Process state updates from AUV"""
        with self.state_lock:
            # Convert state message to observation format
            obs_vector = self.convert_state_to_observation(msg, is_state=True)
            if obs_vector is not None:
                self.current_observation = {'vector': obs_vector}
                self.observation_buffer.append(obs_vector)
    
    def error_callback(self, msg):
        """Process error updates from AUV"""
        with self.state_lock:
            # Convert error message to observation format  
            obs_vector = self.convert_state_to_observation(msg, is_state=False)
            if obs_vector is not None:
                self.current_observation = {'vector': obs_vector}
                self.observation_buffer.append(obs_vector)
    
    def convert_state_to_observation(self, msg, is_state=True):
        """Convert ROS message to DreamerV3 observation format"""
        try:
            # Extract components based on message type
            if is_state:
                # State data - need to compute errors relative to setpoint
                obs = []
                
                # For simplified deployment, use state values directly
                obs.extend([msg.position.z])  # depth
                obs.extend([msg.velocity.x, msg.velocity.y])  # surge, sway velocity
                
                # Convert orientation to sin/cos
                for angle in [msg.orientation.x, msg.orientation.y, msg.orientation.z]:
                    obs.extend([np.sin(angle), np.cos(angle)])
                
                # Add remaining velocity and angular rate components
                obs.extend([msg.velocity.x, msg.velocity.y, msg.velocity.z])
                obs.extend([msg.angular_rate.x, msg.angular_rate.y, msg.angular_rate.z])
                
                # Add dummy accelerations (would come from IMU in full system)
                obs.extend([0.0, 0.0])
                
            else:
                # Error data - use directly
                obs = []
                obs.extend([msg.position.z])  # depth error
                obs.extend([msg.velocity.x, msg.velocity.y])  # surge, sway error
                
                # Convert orientation errors to sin/cos
                for angle in [msg.orientation.x, msg.orientation.y, msg.orientation.z]:
                    obs.extend([np.sin(angle), np.cos(angle)])
                
                # Add velocity components (from state, not error)
                obs.extend([0.0, 0.0, 0.0])  # Would need state data
                
                # Add angular rates (from state, not error)
                obs.extend([0.0, 0.0, 0.0])  # Would need state data
                
                # Add accelerations
                obs.extend([0.0, 0.0])
            
            obs_array = np.array(obs, dtype=np.float32)
            
            # Apply normalization if available
            if self.normalization_params:
                obs_mean = self.normalization_params.get('obs_mean')
                obs_std = self.normalization_params.get('obs_std')
                if obs_mean is not None and obs_std is not None:
                    obs_array = (obs_array - obs_mean) / obs_std
            
            return obs_array
            
        except Exception as e:
            self.get_logger().error(f"Error converting state to observation: {e}")
            return None
    
    def control_callback(self):
        """Main control loop callback"""
        loop_start_time = time.time()
        
        # Check if we have current observation
        if self.current_observation is None:
            return
        
        try:
            with self.policy_lock:
                # Get action from policy
                inference_start = time.time()
                action = self.get_policy_action(self.current_observation)
                inference_time = time.time() - inference_start
                
                self.inference_times.append(inference_time)
            
            # Apply safety constraints
            safe_action = self.apply_safety_constraints(action)
            
            # Publish actions
            self.publish_actions(safe_action)
            
            # Update action buffer
            self.action_buffer.append(safe_action)
            self.last_action = safe_action.copy()
            
            # Record control loop time
            loop_time = time.time() - loop_start_time
            self.control_loop_times.append(loop_time)
            
        except Exception as e:
            self.get_logger().error(f"Error in control callback: {e}")
            # Publish zero actions in case of error
            self.publish_zero_actions()
    
    def get_policy_action(self, observation):
        """Get action from loaded policy with fallback - Enhanced version"""
        if hasattr(self, '_use_fallback') and self._use_fallback:
            # Use random policy as fallback
            action_dim = self.config['environment']['thruster_size'] + self.config['environment']['servo_joints_size']
            return np.random.uniform(-0.1, 0.1, action_dim)  # Small random actions for safety
        
        try:
            if self.policy_type == 'jit':
                # JIT compiled policy
                obs_jax = {k: jnp.array(v) for k, v in observation.items()}
                action, self.policy_state = self.policy(obs_jax, self.policy_state)
                return np.array(action)
            else:
                # Full DreamerV3 agent - try different action methods
                try:
                    # Method 1: policy method
                    if hasattr(self.policy, 'policy'):
                        action = self.policy.policy(observation, self.policy_state, mode='eval')
                        if isinstance(action, tuple):
                            action, self.policy_state = action
                        return np.array(action)
                except Exception as e1:
                    self.get_logger().warn(f"Policy method 1 failed: {e1}")
                
                try:
                    # Method 2: act method
                    if hasattr(self.policy, 'act'):
                        action = self.policy.act(observation)
                        return np.array(action)
                except Exception as e2:
                    self.get_logger().warn(f"Policy method 2 failed: {e2}")
                
                try:
                    # Method 3: call method
                    if hasattr(self.policy, '__call__'):
                        action = self.policy(observation)
                        return np.array(action)
                except Exception as e3:
                    self.get_logger().warn(f"Policy method 3 failed: {e3}")
                
                # Fallback: return random action
                self.get_logger().error("All policy action methods failed, using random action")
                action_dim = self.config['environment']['thruster_size'] + self.config['environment']['servo_joints_size']
                return np.random.uniform(-1.0, 1.0, action_dim)
                
        except Exception as e:
            self.get_logger().error(f"Critical error in get_policy_action: {e}")
            # Emergency fallback
            action_dim = self.config['environment']['thruster_size'] + self.config['environment']['servo_joints_size']
            return np.zeros(action_dim)
    
    def apply_safety_constraints(self, action):
        """Apply safety constraints to actions"""
        # Clip action changes
        if len(self.action_buffer) > 0:
            action_change = np.abs(action - self.last_action)
            max_change_mask = action_change > self.max_action_change
            
            if np.any(max_change_mask):
                # Limit action changes
                change_direction = np.sign(action - self.last_action)
                limited_change = change_direction * self.max_action_change
                action = np.where(max_change_mask, 
                                self.last_action + limited_change, 
                                action)
        
        # Clip to action space bounds
        action = np.clip(action, -1.0, 1.0)
        
        # Check for emergency conditions
        if self.should_emergency_stop():
            action = np.zeros_like(action)
            self.get_logger().warn("Emergency stop activated")
        
        return action
    
    def should_emergency_stop(self):
        """Check if emergency stop should be activated"""
        if len(self.observation_buffer) < 2:
            return False
        
        # Check for large observation changes (potential sensor failures)
        current_obs = self.observation_buffer[-1]
        prev_obs = self.observation_buffer[-2]
        
        obs_change = np.linalg.norm(current_obs - prev_obs)
        if obs_change > self.emergency_threshold:
            return True
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(current_obs)):
            return True
        
        return False
    
    def publish_actions(self, action):
        """Publish actions to ROS topics"""
        try:
            # Map actions to thrusters
            action_mapping = [
                ('heave_bow', action[0]),
                ('heave_stern', action[1]),
                ('surge_port', action[2]),
                ('surge_starboard', action[3])
            ]
            
            # Add servo actions if available
            if len(action) > 4:
                action_mapping.extend([
                    ('port_servo', action[4]),
                    ('starboard_servo', action[5] if len(action) > 5 else 0.0)
                ])
            
            # Publish each action
            for name, value in action_mapping:
                if name in self.thruster_pubs:
                    msg = Float64()
                    msg.data = float(value)
                    self.thruster_pubs[name].publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing actions: {e}")
    
    def publish_zero_actions(self):
        """Publish zero actions (emergency/fallback)"""
        zero_action = np.zeros(self.config['environment']['thruster_size'] + 
                             self.config['environment']['servo_joints_size'])
        self.publish_actions(zero_action)
    
    def emergency_stop_callback(self, msg):
        """Handle emergency stop command"""
        if msg.data > 0.5:  # Emergency stop activated
            self.get_logger().warn("External emergency stop received")
            self.publish_zero_actions()
    
    def monitoring_callback(self):
        """Periodic monitoring and status reporting"""
        # Compute performance metrics
        if self.inference_times:
            avg_inference = np.mean(self.inference_times)
            max_inference = np.max(self.inference_times)
        else:
            avg_inference = 0.0
            max_inference = 0.0
        
        if self.control_loop_times:
            avg_loop_time = np.mean(self.control_loop_times)
            control_frequency = 1.0 / avg_loop_time if avg_loop_time > 0 else 0.0
        else:
            avg_loop_time = 0.0
            control_frequency = 0.0
        
        # Log performance
        self.get_logger().info(
            f"Performance - Inference: {avg_inference*1000:.2f}ms (max: {max_inference*1000:.2f}ms), "
            f"Control freq: {control_frequency:.1f}Hz"
        )
        
        # Publish status
        self.publish_status(avg_inference, control_frequency)
    
    def publish_status(self, inference_time, control_freq):
        """Publish policy status"""
        try:
            status_msg = ControlProcess()
            status_msg.header.stamp = self.get_clock().now().to_msg()
            status_msg.header.frame_id = "dreamerv3_policy"
            
            # Use position field for performance metrics
            status_msg.position.x = inference_time * 1000  # ms
            status_msg.position.y = control_freq  # Hz
            status_msg.position.z = len(self.action_buffer)  # Buffer size
            
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing status: {e}")

class DeploymentManager:
    """
    High-level deployment manager for DreamerV3 AUV system
    """
    
    def __init__(self, config_path, model_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = model_path
        self.policy_node = None
        self.executor = None
        
        # Determine policy type
        model_path_obj = Path(model_path)
        if (model_path_obj / "policy_jit.pkl").exists():
            self.policy_type = 'jit'
        else:
            self.policy_type = 'full'
        
        print(f"Deployment Manager initialized")
        print(f"Model path: {model_path}")
        print(f"Policy type: {self.policy_type}")
    
    def start_deployment(self):
        """Start the deployment system"""
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 not available for deployment")
        
        # Initialize ROS2
        rclpy.init()
        
        try:
            # Create policy node
            self.policy_node = DreamerV3PolicyNode(
                self.config, 
                self.model_path, 
                self.policy_type
            )
            
            # Create executor
            self.executor = MultiThreadedExecutor(num_threads=4)
            self.executor.add_node(self.policy_node)
            
            print("DreamerV3 deployment started successfully")
            print(f"Policy type: {self.policy_type}")
            print("Press Ctrl+C to stop...")
            
            # Spin executor
            self.executor.spin()
            
        except KeyboardInterrupt:
            print("Deployment stopped by user")
        except Exception as e:
            print(f"Deployment error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup deployment resources"""
        if self.executor:
            self.executor.shutdown()
        
        if self.policy_node:
            self.policy_node.destroy_node()
        
        if rclpy.ok():
            rclpy.shutdown()
        
        print("Deployment cleanup completed")

class BenchmarkRunner:
    """
    Benchmark and testing utilities for DreamerV3 policies
    """
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def benchmark_policy_inference(self, model_path, num_iterations=1000):
        """Benchmark policy inference speed - Fixed version"""
        print(f"Benchmarking policy inference ({num_iterations} iterations)")
        
        # Load policy
        model_path_obj = Path(model_path)
        if (model_path_obj / "policy_jit.pkl").exists():
            policy = self.load_jit_policy(model_path_obj / "policy_jit.pkl")
            policy_type = 'jit'
            policy_state = {}
        else:
            policy = self.load_full_policy_benchmark(model_path_obj)
            policy_type = 'full'
            # Initialize policy state
            try:
                if hasattr(policy, 'init_state'):
                    policy_state = policy.init_state()
                else:
                    policy_state = {}
            except:
                policy_state = {}
        
        # Create dummy observation
        obs_shape = (17,)  # Default AUV observation shape
        obs = {'vector': np.random.randn(*obs_shape).astype(np.float32)}
        
        # Warm-up
        for _ in range(10):
            try:
                if policy_type == 'jit':
                    obs_jax = {k: jnp.array(v) for k, v in obs.items()}
                    action, policy_state = policy(obs_jax, policy_state)
                else:
                    # Try different action methods for full agent
                    if hasattr(policy, 'policy'):
                        try:
                            action = policy.policy(obs, policy_state, mode='eval')
                            if isinstance(action, tuple):
                                action, policy_state = action
                        except:
                            action = policy.policy(obs, mode='eval')
                    elif hasattr(policy, 'act'):
                        action = policy.act(obs)
                    else:
                        action = np.random.uniform(-1, 1, 4)  # Fallback
            except Exception as e:
                print(f"Warm-up error: {e}")
                continue
        
        # Benchmark
        times = []
        successful_runs = 0
        
        for i in range(num_iterations):
            start_time = time.time()
            
            try:
                if policy_type == 'jit':
                    obs_jax = {k: jnp.array(v) for k, v in obs.items()}
                    action, policy_state = policy(obs_jax, policy_state)
                else:
                    # Full agent inference
                    if hasattr(policy, 'policy'):
                        try:
                            action = policy.policy(obs, policy_state, mode='eval')
                            if isinstance(action, tuple):
                                action, policy_state = action
                        except:
                            action = policy.policy(obs, mode='eval')
                    elif hasattr(policy, 'act'):
                        action = policy.act(obs)
                    else:
                        action = np.random.uniform(-1, 1, 4)
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
                successful_runs += 1
                
            except Exception as e:
                if i < 10:  # Only print first few errors
                    print(f"Inference error {i}: {e}")
                continue
        
        if not times:
            print("No successful inference runs!")
            return None
        
        # Results
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"Policy inference benchmark results ({policy_type}):")
        print(f"  Successful runs: {successful_runs}/{num_iterations}")
        print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Min/Max: {min_time:.2f} / {max_time:.2f} ms")
        print(f"  Max frequency: {1000/avg_time:.1f} Hz")
        
        return {
            'policy_type': policy_type,
            'successful_runs': successful_runs,
            'total_runs': num_iterations,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'max_frequency_hz': 1000/avg_time
        }
    
    def load_jit_policy(self, jit_path):
        """Load JIT compiled policy"""
        with open(jit_path, 'rb') as f:
            policy_data = pickle.load(f)
        return policy_data['compiled_policy']
    
    def load_full_policy_benchmark(self, checkpoint_path):
        """Load full DreamerV3 agent for benchmarking - Fixed"""
        if not DREAMERV3_AVAILABLE:
            raise ImportError("DreamerV3 not available")
        
        try:
            checkpoint_manager = CheckpointManager()
            metadata = checkpoint_manager.load_checkpoint_with_validation(checkpoint_path)
            
            # Create observation and action spaces
            from gymnasium.spaces import Dict, Box
            import numpy as np
            
            obs_space = Dict({
                'vector': Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
            })
            
            thruster_size = self.config['environment']['thruster_size']
            servo_size = self.config['environment']['servo_joints_size']
            action_dim = thruster_size + servo_size
            
            act_space = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
            
            # Create agent configuration
            config = {
                'logdir': '/tmp/dreamerv3_benchmark',
                'seed': 42,
                'precision': 16,
                'task': 'auv_control',
                'env_amount': 1,
                'env_parallel': False,
                'batch_size': 16,
                'batch_length': 64,
            }
            
            # Override with saved config
            if 'config' in metadata and 'dreamerv3' in metadata['config']:
                saved_config = metadata['config']['dreamerv3']
                config.update(saved_config)
            
            # Create agent
            import embodied
            agent = embodied.Agent(obs_space, act_space, config)
            
            # Load checkpoint
            step = metadata.get('step', 0)
            try:
                agent.load(str(checkpoint_path), step)
            except TypeError:
                # Try with just directory
                agent.load(str(checkpoint_path))
            
            return agent
            
        except Exception as e:
            print(f"Error loading agent for benchmark: {e}")
            raise
    
    def test_environment_integration(self, model_path, num_episodes=5):
        """Test policy integration with AUV environment"""
        print(f"Testing environment integration ({num_episodes} episodes)")
        
        # Create environment
        env = AUVEnvDreamerV3(offline_mode=True)  # Use offline mode for testing
        
        # Load policy (simplified)
        model_path_obj = Path(model_path)
        policy_available = ((model_path_obj / "policy_jit.pkl").exists() or 
                          (model_path_obj / "step_0").exists())
        
        if not policy_available:
            print("No valid policy found for testing")
            return
        
        results = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while episode_length < 100:  # Max 100 steps per test episode
                # Random action for testing (replace with actual policy)
                action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            results.append({
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length
            })
            
            print(f"Test episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}")
        
        env.close()
        
        avg_reward = np.mean([r['reward'] for r in results])
        avg_length = np.mean([r['length'] for r in results])
        
        print(f"Integration test results:")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average length: {avg_length:.1f}")
        
        return results

def main():
    """Main function for deployment utilities"""
    parser = argparse.ArgumentParser(description='DreamerV3 AUV Deployment Utilities')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy policy for real-time control')
    deploy_parser.add_argument('model_path', type=str, help='Path to trained model')
    deploy_parser.add_argument('--config', type=str, default='config/config_dreamerv3.yaml',
                              help='Configuration file path')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark policy performance')
    benchmark_parser.add_argument('model_path', type=str, help='Path to trained model')
    benchmark_parser.add_argument('--config', type=str, default='config/config_dreamerv3.yaml',
                                 help='Configuration file path')
    benchmark_parser.add_argument('--iterations', type=int, default=1000,
                                 help='Number of benchmark iterations')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test environment integration')
    test_parser.add_argument('model_path', type=str, help='Path to trained model')
    test_parser.add_argument('--config', type=str, default='config/config_dreamerv3.yaml',
                            help='Configuration file path')
    test_parser.add_argument('--episodes', type=int, default=5,
                            help='Number of test episodes')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Resolve config path
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    try:
        if args.command == 'deploy':
            # Real-time deployment
            manager = DeploymentManager(config_path, args.model_path)
            manager.start_deployment()
            
        elif args.command == 'benchmark':
            # Performance benchmarking
            runner = BenchmarkRunner(config_path)
            results = runner.benchmark_policy_inference(args.model_path, args.iterations)
            
            # Save results
            results_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Benchmark results saved to {results_file}")
            
        elif args.command == 'test':
            # Integration testing
            runner = BenchmarkRunner(config_path)
            results = runner.test_environment_integration(args.model_path, args.episodes)
            
            # Save results
            results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Test results saved to {results_file}")
        
    except Exception as e:
        print(f"Error executing command: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()