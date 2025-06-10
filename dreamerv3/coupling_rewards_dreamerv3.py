# coupling_rewards_dreamerv3.py - Enhanced with DreamerV3 compatibility
import numpy as np
import jax.numpy as jnp
import jax

class DreamerV3RewardAdapter:
    """
    Adapter class to make coupling-aware rewards compatible with DreamerV3
    Handles symlog transformation and JAX compatibility
    """
    
    def __init__(self, coupling_calculator, config):
        self.coupling_calculator = coupling_calculator
        self.config = config
        
        # DreamerV3 specific parameters
        self.symlog_enabled = config.get('dreamerv3', {}).get('symlog_rewards', True)
        self.reward_scale = config.get('dreamerv3', {}).get('reward_scale', 1.0)
        self.reward_clip = config.get('dreamerv3', {}).get('reward_clip', None)
        
        # Enhanced smoothing for more stable learning
        self.reward_smoothing = config.get('dreamerv3', {}).get('reward_smoothing', 0.05)
        self.reward_history = []
        self.max_history_len = 20
        
    def symlog(self, x):
        """Apply symlog transformation for DreamerV3 compatibility"""
        if not self.symlog_enabled:
            return x
        
        # JAX-compatible symlog transformation
        return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)
    
    def inverse_symlog(self, x):
        """Inverse symlog transformation"""
        if not self.symlog_enabled:
            return x
        
        return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)
    
    def calculate_reward(self, state_error_array, action, episode_step):
        """
        Calculate coupling-aware reward with DreamerV3 adaptations
        
        Args:
            state_error_array: Current state errors
            action: Action taken 
            episode_step: Current episode step
            
        Returns:
            Transformed reward suitable for DreamerV3
        """
        # Calculate base coupling-aware reward
        base_reward = self.coupling_calculator.calculate_coupling_aware_reward_v4_enhanced(
            state_error_array, episode_step
        )
        
        # Apply smoothing if enabled
        if self.reward_smoothing > 0:
            base_reward = self._apply_smoothing(base_reward)
        
        # Scale reward
        scaled_reward = base_reward * self.reward_scale
        
        # Apply clipping if specified
        if self.reward_clip is not None:
            scaled_reward = np.clip(scaled_reward, -self.reward_clip, self.reward_clip)
        
        # Apply symlog transformation for DreamerV3
        if self.symlog_enabled:
            transformed_reward = float(self.symlog(scaled_reward))
        else:
            transformed_reward = float(scaled_reward)
        
        return transformed_reward
    
    def _apply_smoothing(self, reward):
        """Apply exponential smoothing to reward signal"""
        self.reward_history.append(reward)
        
        if len(self.reward_history) > self.max_history_len:
            self.reward_history.pop(0)
        
        if len(self.reward_history) > 1:
            # Exponential moving average
            smoothed = self.reward_smoothing * reward + (1 - self.reward_smoothing) * np.mean(self.reward_history[:-1])
            return smoothed
        else:
            return reward
    
    def get_reward_statistics(self):
        """Get reward statistics for monitoring"""
        if len(self.reward_history) < 2:
            return None
        
        return {
            'mean': np.mean(self.reward_history),
            'std': np.std(self.reward_history),
            'min': np.min(self.reward_history),
            'max': np.max(self.reward_history),
            'latest': self.reward_history[-1]
        }

class CouplingAwareRewardCalculator:
    """
    Enhanced coupling-aware reward functions for AUV control with DreamerV3 compatibility
    Handles surge-yaw and pitch-depth coupling for underwater vehicles
    """
    
    def __init__(self, config):
        self.config = config
        self.w = config.get('reward_function', {})
        
        # Get coupling configuration with defaults
        coupling_config = config.get('coupling', {})
        self.surge_yaw_coupling = coupling_config.get('surge_yaw_weight', 0.4)
        self.pitch_depth_coupling = coupling_config.get('pitch_depth_weight', 0.5)
        self.coupling_threshold = coupling_config.get('threshold', 0.1)
        
        # Progressive learning parameters
        progressive_config = coupling_config.get('progressive', {})
        self.exploration_episodes = progressive_config.get('exploration_episodes', 50)
        self.coupling_focus_episodes = progressive_config.get('coupling_focus_episodes', 100)
        
        # Dynamic coupling parameters (for v3)
        dynamic_config = coupling_config.get('dynamic', {})
        self.history_window = dynamic_config.get('history_window', 15)
        self.adaptation_rate = dynamic_config.get('adaptation_rate', 0.1)
        
        # Energy-based parameters (for v4)
        energy_config = coupling_config.get('energy', {})
        self.spring_stiffness = energy_config.get('spring_stiffness', 0.5)
        self.damping_factor = energy_config.get('damping_factor', 0.1)
        
        # SURGE-SPECIFIC FIXES
        surge_config = coupling_config.get('surge_fixes', {})
        self.surge_priority_mode = surge_config.get('priority_mode', True)
        self.surge_error_threshold = surge_config.get('error_threshold', 0.15)
        self.surge_coupling_reduction = surge_config.get('coupling_reduction', 0.3)
        self.surge_progress_bonus_scale = surge_config.get('progress_bonus_scale', 0.2)
        self.surge_setpoint_tracking_bonus = surge_config.get('setpoint_tracking_bonus', 0.15)
        
        # History for coupling detection
        self.error_history = []
        self.max_history = max(20, self.history_window)
        
        # Dynamic coupling matrix (for v3)
        self.coupling_matrix = np.eye(4)
        
        # DreamerV3 specific enhancements
        dreamerv3_config = config.get('dreamerv3', {})
        self.imagination_bonus = dreamerv3_config.get('imagination_bonus', 0.1)
        self.world_model_consistency_weight = dreamerv3_config.get('world_model_consistency', 0.05)
    
    def calculate_coupling_aware_reward_v1(self, state_error_array, episode_num=0):
        """
        Method 1: Cross-Correlation Coupling Reward with DreamerV3 enhancements
        """
        # Extract errors (assuming sin/cos representation)
        depth_error = state_error_array[0]
        surge_error = state_error_array[1] 
        sway_error = state_error_array[2]
        heave_error = state_error_array[3]
        
        # For angles, use the actual error magnitude, not sin/cos magnitude
        roll_sin_err, roll_cos_err = state_error_array[4], state_error_array[5]
        pitch_sin_err, pitch_cos_err = state_error_array[6], state_error_array[7]
        yaw_sin_err, yaw_cos_err = state_error_array[8], state_error_array[9]
        
        # Convert sin/cos back to actual angle error magnitude
        pitch_error_mag = abs(np.arctan2(pitch_sin_err, pitch_cos_err))
        yaw_error_mag = abs(np.arctan2(yaw_sin_err, yaw_cos_err))
        
        # Store error history
        current_errors = {
            'depth': depth_error,
            'surge': surge_error,
            'pitch': pitch_error_mag,
            'yaw': yaw_error_mag
        }
        self.error_history.append(current_errors)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Base performance error with enhanced coupling awareness
        original_weights = np.array(self.config['reward_function']['state_error_weights'])
        state_error_weights = self._expand_weights_for_sincos(original_weights, state_error_array)
        
        # Apply coupling-aware weighting
        coupling_enhanced_weights = self._apply_coupling_weights(state_error_weights, current_errors)
        
        error_column = state_error_array.reshape(-1, 1)
        error_row = state_error_array.reshape(1, -1)
        weights_diag = np.diag(coupling_enhanced_weights)
        base_performance = -error_row @ weights_diag @ error_column
        
        # Coupling bonuses
        surge_yaw_coupling_bonus = self._calculate_surge_yaw_coupling(surge_error, yaw_error_mag)
        pitch_depth_coupling_bonus = self._calculate_pitch_depth_coupling(pitch_error_mag, depth_error)
        
        # Adaptive coupling strength based on error magnitude
        coupling_strength = self._adaptive_coupling_strength(current_errors)
        
        # DreamerV3 imagination bonus (rewards exploration in safe directions)
        imagination_bonus = self._calculate_imagination_bonus(current_errors, episode_num)
        
        total_reward = (base_performance + 
                       coupling_strength * self.surge_yaw_coupling * surge_yaw_coupling_bonus +
                       coupling_strength * self.pitch_depth_coupling * pitch_depth_coupling_bonus +
                       self.imagination_bonus * imagination_bonus)
        
        return float(total_reward.item())
    
    def calculate_coupling_aware_reward_v2(self, state_error_array, episode_num=0):
        """
        Method 2: Hierarchical Progressive Learning with DreamerV3 adaptations
        """
        # Extract errors
        depth_error = state_error_array[0]
        surge_error = state_error_array[1]
        
        pitch_sin_err, pitch_cos_err = state_error_array[6], state_error_array[7]
        yaw_sin_err, yaw_cos_err = state_error_array[8], state_error_array[9]
        
        # Use actual angle error magnitude
        pitch_error_mag = abs(np.arctan2(pitch_sin_err, pitch_cos_err))
        yaw_error_mag = abs(np.arctan2(yaw_sin_err, yaw_cos_err))
        
        # Store error history
        current_errors = {
            'depth': depth_error,
            'surge': surge_error,
            'pitch': pitch_error_mag,
            'yaw': yaw_error_mag
        }
        self.error_history.append(current_errors)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Progressive learning phases adapted for DreamerV3
        phase = self._determine_learning_phase(episode_num)
        
        if phase == "coupling_focus":
            # Focus only on coupled pairs with world model consistency bonus
            surge_yaw_error = np.sqrt(surge_error**2 + yaw_error_mag**2)
            pitch_depth_error = np.sqrt(pitch_error_mag**2 + depth_error**2)
            
            coupling_reward = -(self.surge_yaw_coupling * surge_yaw_error**2 + 
                              self.pitch_depth_coupling * pitch_depth_error**2)
            
            # Bonus for synchronized reduction
            if len(self.error_history) > 1:
                prev_surge_yaw = np.sqrt(self.error_history[-2]['surge']**2 + self.error_history[-2]['yaw']**2)
                prev_pitch_depth = np.sqrt(self.error_history[-2]['pitch']**2 + self.error_history[-2]['depth']**2)
                
                surge_yaw_improvement = prev_surge_yaw - surge_yaw_error
                pitch_depth_improvement = prev_pitch_depth - pitch_depth_error
                
                synchronization_bonus = 0.1 * (surge_yaw_improvement * pitch_depth_improvement)
                coupling_reward += synchronization_bonus
            
            # Add world model consistency bonus for DreamerV3
            consistency_bonus = self._calculate_world_model_consistency_bonus(current_errors)
            coupling_reward += self.world_model_consistency_weight * consistency_bonus
            
            return coupling_reward
            
        elif phase == "refinement":
            # Include all errors but weight coupled pairs higher
            return self.calculate_coupling_aware_reward_v1(state_error_array, episode_num)
            
        else:  # "exploration" phase
            # Standard reward with exploration bonuses for DreamerV3
            base_reward = self._calculate_standard_reward(state_error_array)
            exploration_bonus = self._calculate_exploration_bonus(current_errors, episode_num)
            return base_reward + self.imagination_bonus * exploration_bonus
    
    def calculate_coupling_aware_reward_v3(self, state_error_array, episode_num=0):
        """
        Method 3: Dynamic Coupling Matrix Reward with DreamerV3 adaptations
        """
        # Extract current errors
        depth_error = state_error_array[0]
        surge_error = state_error_array[1]
        
        pitch_sin_err, pitch_cos_err = state_error_array[6], state_error_array[7]
        yaw_sin_err, yaw_cos_err = state_error_array[8], state_error_array[9]
        
        # Use actual angle error magnitude
        pitch_error_mag = abs(np.arctan2(pitch_sin_err, pitch_cos_err))
        yaw_error_mag = abs(np.arctan2(yaw_sin_err, yaw_cos_err))
        
        # Store current state
        current_errors = {
            'depth': depth_error,
            'surge': surge_error,
            'pitch': pitch_error_mag,
            'yaw': yaw_error_mag
        }
        self.error_history.append(current_errors)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Update dynamic coupling matrix
        self._update_dynamic_coupling_matrix()
        
        # Apply coupling-aware error weighting
        current_error_vector = np.array([depth_error, surge_error, pitch_error_mag, yaw_error_mag])
        coupled_error = current_error_vector.T @ self.coupling_matrix @ current_error_vector
        
        # Base reward with dynamic coupling
        base_reward = -np.sum(current_error_vector**2)
        coupling_penalty = -self.w.get('w1', 1.0) * coupled_error
        
        # DreamerV3 specific: add temporal consistency reward
        temporal_consistency = self._calculate_temporal_consistency(current_errors)
        
        return base_reward + coupling_penalty + self.world_model_consistency_weight * temporal_consistency
    
    def calculate_coupling_aware_reward_v4_enhanced(self, state_error_array, episode_num=0):
        """
        Enhanced Method 4: Energy-Based Coupling with DreamerV3 optimizations
        """
        # ERROR HANDLING: Check for None or invalid input
        if state_error_array is None:
            print("Warning: state_error_array is None, returning default reward")
            return -1.0
        
        if len(state_error_array) < 10:
            print(f"Warning: state_error_array too short ({len(state_error_array)}), expected 10")
            return -1.0
        
        try:
            # Extract errors with safe indexing
            depth_error = float(state_error_array[0])
            surge_error = float(state_error_array[1])
            
            pitch_sin_err, pitch_cos_err = float(state_error_array[6]), float(state_error_array[7])
            yaw_sin_err, yaw_cos_err = float(state_error_array[8]), float(state_error_array[9])
            
            # Use actual angle error magnitude
            pitch_error_mag = abs(np.arctan2(pitch_sin_err, pitch_cos_err))
            yaw_error_mag = abs(np.arctan2(yaw_sin_err, yaw_cos_err))
            
            # Store error history
            current_errors = {
                'depth': depth_error,
                'surge': surge_error,
                'pitch': pitch_error_mag,
                'yaw': yaw_error_mag
            }
            self.error_history.append(current_errors)
            if len(self.error_history) > self.max_history:
                self.error_history.pop(0)
            
            # ENHANCED: Adaptive coupling springs based on error magnitude
            surge_yaw_spring_stiffness = self.spring_stiffness
            pitch_depth_spring_stiffness = self.spring_stiffness
            
            # Increase stiffness for problematic areas
            if pitch_error_mag > 0.2:  # High pitch error
                pitch_depth_spring_stiffness *= 2.0
            
            if abs(yaw_error_mag) > 0.3:  # Yaw offset issue
                surge_yaw_spring_stiffness *= 1.5
            
            # Energy calculations with adaptive stiffness
            surge_yaw_spring_energy = 0.5 * surge_yaw_spring_stiffness * self.surge_yaw_coupling * (surge_error - yaw_error_mag)**2
            pitch_depth_spring_energy = 0.5 * pitch_depth_spring_stiffness * self.pitch_depth_coupling * (pitch_error_mag - depth_error)**2
            
            # Individual error energies with enhanced weights
            individual_energy = 0.5 * (
                depth_error**2 + 
                surge_error**2 + 
                2.0 * pitch_error_mag**2 +  # 2x weight for pitch
                1.5 * yaw_error_mag**2       # 1.5x weight for yaw offset
            )
            
            # ENHANCED: Asymmetric surge penalty
            surge_asymmetry_penalty = self._calculate_surge_asymmetry_penalty()
            surge_progress_bonus = self._calculate_surge_progress_bonus()
            
            # ENHANCED: Yaw bias penalty
            yaw_bias_penalty = self._calculate_yaw_bias_penalty()
            
            # Enhanced damping energy
            damping_energy = self._calculate_damping_energy(current_errors, pitch_error_mag)
            
            # DreamerV3 specific: World model predictive bonus
            predictive_bonus = self._calculate_predictive_accuracy_bonus(current_errors, episode_num)
            
            # Total enhanced energy
            total_energy = (individual_energy + 
                          surge_yaw_spring_energy + 
                          pitch_depth_spring_energy + 
                          damping_energy +
                          surge_asymmetry_penalty +
                          yaw_bias_penalty)
            
            # ENHANCED: Adaptive energy scaling for DreamerV3
            if episode_num < 1000:
                energy_scale = 0.5  # Gentler penalty during early learning
            elif episode_num < 5000:
                energy_scale = 0.8  # Medium penalty during intermediate learning
            else:
                energy_scale = 1.0  # Full penalty for mature learning
            
            # Return negative scaled energy with bonuses
            final_reward = (-total_energy * energy_scale + 
                          surge_progress_bonus + 
                          self.imagination_bonus * predictive_bonus)
            
            return float(final_reward)
        
        except Exception as e:
            print(f"Error in v4_enhanced calculation: {e}")
            print(f"state_error_array: {state_error_array}")
            return -1.0
    
    # DreamerV3 specific helper methods
    
    def _calculate_imagination_bonus(self, current_errors, episode_num):
        """Calculate bonus for safe exploration in imagination"""
        if len(self.error_history) < 5:
            return 0.0
        
        # Reward consistent improvement patterns
        recent_errors = self.error_history[-5:]
        error_trends = []
        
        for error_type in ['depth', 'surge', 'pitch', 'yaw']:
            values = [e[error_type] for e in recent_errors]
            if len(values) > 1:
                # Calculate trend (negative means improvement)
                trend = np.polyfit(range(len(values)), values, 1)[0]
                error_trends.append(trend)
        
        # Bonus for consistent improvement across error types
        if error_trends and all(trend < 0 for trend in error_trends):
            return 0.2 * abs(np.mean(error_trends))
        
        return 0.0
    
    def _calculate_world_model_consistency_bonus(self, current_errors):
        """Bonus for maintaining world model consistency"""
        if len(self.error_history) < 3:
            return 0.0
        
        # Reward smooth transitions (beneficial for world model learning)
        recent_errors = self.error_history[-3:]
        smoothness_score = 0.0
        
        for error_type in ['depth', 'surge', 'pitch', 'yaw']:
            values = [e[error_type] for e in recent_errors]
            if len(values) == 3:
                # Calculate second derivative (acceleration)
                second_deriv = values[2] - 2*values[1] + values[0]
                smoothness_score += 1.0 / (1.0 + abs(second_deriv))
        
        return smoothness_score / 4.0  # Normalize by number of error types
    
    def _calculate_temporal_consistency(self, current_errors):
        """Calculate temporal consistency for world model accuracy"""
        if len(self.error_history) < 2:
            return 0.0
        
        prev_errors = self.error_history[-2]
        consistency_score = 0.0
        
        for error_type in ['depth', 'surge', 'pitch', 'yaw']:
            # Reward predictable changes
            error_change = abs(current_errors[error_type] - prev_errors[error_type])
            # Exponential decay for large changes (unpredictable)
            consistency_score += np.exp(-error_change * 5.0)
        
        return consistency_score / 4.0
    
    def _calculate_predictive_accuracy_bonus(self, current_errors, episode_num):
        """Bonus for maintaining predictive accuracy in world model"""
        if len(self.error_history) < 3:
            return 0.0
        
        # Look at prediction vs actual error evolution
        recent_errors = self.error_history[-3:]
        prediction_accuracy = 0.0
        
        for error_type in ['depth', 'surge', 'pitch', 'yaw']:
            values = [e[error_type] for e in recent_errors]
            if len(values) == 3:
                # Simple linear prediction from previous two points
                predicted = 2 * values[1] - values[0]
                actual = values[2]
                accuracy = 1.0 / (1.0 + abs(predicted - actual))
                prediction_accuracy += accuracy
        
        return prediction_accuracy / 4.0
    
    def _calculate_exploration_bonus(self, current_errors, episode_num):
        """Calculate exploration bonus for early training phases"""
        # Encourage diverse error patterns early in training
        if episode_num > self.exploration_episodes:
            return 0.0
        
        error_diversity = np.std([current_errors[k] for k in current_errors.keys()])
        return min(0.1, error_diversity * 0.1)
    
    # Enhanced helper methods with improved calculations
    
    def _calculate_surge_asymmetry_penalty(self):
        """Enhanced surge asymmetry penalty calculation"""
        if len(self.error_history) < 10:
            return 0.0
        
        recent_surges = [h['surge'] for h in self.error_history[-10:]]
        positive_surges = [abs(s) for s in recent_surges if s > 0.1]
        negative_surges = [abs(s) for s in recent_surges if s < -0.1]
        
        if len(positive_surges) > 0 and len(negative_surges) > 0:
            positive_response = np.mean(positive_surges)
            negative_response = np.mean(negative_surges)
            asymmetry = abs(positive_response - negative_response) / max(positive_response, negative_response)
            return 0.15 * asymmetry  # Increased penalty
        
        return 0.0
    
    def _calculate_surge_progress_bonus(self):
        """Enhanced surge progress bonus"""
        if len(self.error_history) < 5:
            return 0.0
        
        recent_surges = [abs(h['surge']) for h in self.error_history[-5:]]
        if len(recent_surges) >= 2:
            improvement = recent_surges[0] - recent_surges[-1]
            if improvement > 0:  # Improvement
                return 0.15 * improvement  # Increased bonus
        
        return 0.0
    
    def _calculate_yaw_bias_penalty(self):
        """Enhanced yaw bias penalty"""
        if len(self.error_history) < 15:
            return 0.0
        
        recent_yaws = [h['yaw'] for h in self.error_history[-15:]]
        mean_yaw_error = np.mean(recent_yaws)
        if abs(mean_yaw_error) > 0.15:  # Reduced threshold for earlier detection
            return 0.08 * abs(mean_yaw_error)  # Increased penalty
        
        return 0.0
    
    def _calculate_damping_energy(self, current_errors, pitch_error_mag):
        """Enhanced damping energy calculation"""
        if len(self.error_history) < 2:
            return 0.0
        
        prev_errors = self.error_history[-2]
        
        # Velocity calculations
        velocity_surge = current_errors['surge'] - prev_errors['surge']
        velocity_yaw = current_errors['yaw'] - prev_errors['yaw']
        velocity_pitch = pitch_error_mag - prev_errors['pitch']
        velocity_depth = current_errors['depth'] - prev_errors['depth']
        
        # Enhanced damping for problem areas
        enhanced_damping = self.damping_factor
        if pitch_error_mag > 0.2:
            enhanced_damping *= 2.0  # More damping for high pitch errors
        
        damping_energy = enhanced_damping * (
            velocity_surge**2 + 
            velocity_yaw**2 + 
            2.0 * velocity_pitch**2 +  # Extra damping for pitch
            velocity_depth**2
        )
        
        return damping_energy
    
    # Keep all existing methods from the original coupling_rewards.py
    # (All the helper methods like _determine_learning_phase, _update_dynamic_coupling_matrix, etc.)
    # Just adding the methods that were in the original file
    
    def _determine_learning_phase(self, episode_num):
        """Determine current learning phase for progressive training"""
        progressive_config = self.config.get('coupling', {}).get('progressive', {})
        
        if not progressive_config.get('enabled', False):
            return "refinement"
        
        if episode_num < self.exploration_episodes:
            return "exploration"
        elif episode_num < self.exploration_episodes + self.coupling_focus_episodes:
            return "coupling_focus"
        else:
            return "refinement"
    
    def _update_dynamic_coupling_matrix(self):
        """Update the dynamic coupling matrix based on observed correlations"""
        if len(self.error_history) < self.history_window:
            return
        
        # Get recent history
        recent_history = self.error_history[-self.history_window:]
        history_array = np.array([[h['depth'], h['surge'], h['pitch'], h['yaw']] for h in recent_history])
        
        if len(history_array) > 1:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(history_array.T)
            
            # Update coupling matrix with adaptation rate
            new_coupling_matrix = np.eye(4)
            
            # Enhance coupling based on observed correlations
            new_coupling_matrix[0, 2] = new_coupling_matrix[2, 0] = abs(corr_matrix[0, 2]) * self.pitch_depth_coupling
            new_coupling_matrix[1, 3] = new_coupling_matrix[3, 1] = abs(corr_matrix[1, 3]) * self.surge_yaw_coupling
            
            # Apply adaptation rate
            self.coupling_matrix = (1 - self.adaptation_rate) * self.coupling_matrix + self.adaptation_rate * new_coupling_matrix
    
    def _calculate_standard_reward(self, state_error_array):
        """Standard reward calculation for exploration phase"""
        original_weights = np.array(self.config['reward_function']['state_error_weights'])
        state_error_weights = self._expand_weights_for_sincos(original_weights, state_error_array)
        
        error_column = state_error_array.reshape(-1, 1)
        error_row = state_error_array.reshape(1, -1)
        weights_diag = np.diag(state_error_weights)
        
        return float((-error_row @ weights_diag @ error_column).item())
    
    def _apply_coupling_weights(self, base_weights, current_errors):
        """Apply coupling-aware weight adjustments"""
        enhanced_weights = base_weights.copy()
        
        # If surge and yaw errors are both significant, increase their coupling
        surge_yaw_magnitude = np.sqrt(current_errors['surge']**2 + current_errors['yaw']**2)
        if surge_yaw_magnitude > self.coupling_threshold:
            if len(enhanced_weights) > 1:
                enhanced_weights[1] *= (1.0 + 0.5 * self.surge_yaw_coupling)
            if len(enhanced_weights) > 8:
                enhanced_weights[8] *= (1.0 + 0.5 * self.surge_yaw_coupling)
            if len(enhanced_weights) > 9:
                enhanced_weights[9] *= (1.0 + 0.5 * self.surge_yaw_coupling)
        
        # If pitch and depth errors are both significant, increase their coupling
        pitch_depth_magnitude = np.sqrt(current_errors['pitch']**2 + current_errors['depth']**2)
        if pitch_depth_magnitude > self.coupling_threshold:
            if len(enhanced_weights) > 0:
                enhanced_weights[0] *= (1.0 + 0.5 * self.pitch_depth_coupling)
            if len(enhanced_weights) > 6:
                enhanced_weights[6] *= (1.0 + 0.5 * self.pitch_depth_coupling)
            if len(enhanced_weights) > 7:
                enhanced_weights[7] *= (1.0 + 0.5 * self.pitch_depth_coupling)
        
        return enhanced_weights
    
    def _calculate_surge_yaw_coupling(self, surge_error, yaw_error):
        """Calculate coupling bonus for surge-yaw pair"""
        if len(self.error_history) < 2:
            return 0
        
        prev_surge = self.error_history[-2]['surge']
        prev_yaw = self.error_history[-2]['yaw']
        
        surge_direction = np.sign(surge_error - prev_surge)
        yaw_direction = np.sign(yaw_error - prev_yaw)
        
        synchronization = surge_direction * yaw_direction
        coupling_magnitude = np.sqrt(surge_error**2 + yaw_error**2)
        
        return synchronization * np.exp(-coupling_magnitude * 5.0)
    
    def _calculate_pitch_depth_coupling(self, pitch_error, depth_error):
        """Calculate coupling bonus for pitch-depth pair"""
        if len(self.error_history) < 2:
            return 0
        
        prev_pitch = self.error_history[-2]['pitch']
        prev_depth = self.error_history[-2]['depth']
        
        pitch_direction = np.sign(pitch_error - prev_pitch)
        depth_direction = np.sign(depth_error - prev_depth)
        
        synchronization = pitch_direction * depth_direction
        coupling_magnitude = np.sqrt(pitch_error**2 + depth_error**2)
        
        return synchronization * np.exp(-coupling_magnitude * 5.0)
    
    def _adaptive_coupling_strength(self, current_errors):
        """Adapt coupling strength based on current error magnitudes"""
        error_magnitude = sum(abs(error) for error in current_errors.values())
        return np.tanh(error_magnitude * 10)
    
    def _expand_weights_for_sincos(self, original_weights, state_error_array):
        """Dynamically expand weights array to account for sin/cos representation"""
        original_weights = np.array(original_weights)
        target_size = len(state_error_array)
        state_error_weights = np.zeros(target_size)
        
        if len(original_weights) == 7 and target_size == 10:
            # Standard AUV case: 7 original -> 10 expanded
            state_error_weights[0] = original_weights[0]  # depth
            state_error_weights[1] = original_weights[1]  # surge
            state_error_weights[2] = original_weights[2]  # sway
            state_error_weights[3] = original_weights[3]  # heave
            
            # Orientation errors (distribute weights between sin/cos pairs)
            state_error_weights[4] = original_weights[4] / 2  # roll sin
            state_error_weights[5] = original_weights[4] / 2  # roll cos
            state_error_weights[6] = original_weights[5] / 2  # pitch sin
            state_error_weights[7] = original_weights[5] / 2  # pitch cos
            state_error_weights[8] = original_weights[6] / 2  # yaw sin
            state_error_weights[9] = original_weights[6] / 2  # yaw cos
            
        elif len(original_weights) == target_size:
            # Already the right size
            state_error_weights = original_weights.copy()
            
        else:
            # General case: fill what we can, pad the rest
            min_size = min(len(original_weights), target_size)
            state_error_weights[:min_size] = original_weights[:min_size]
            
            if target_size > len(original_weights):
                default_weight = original_weights[-1] if len(original_weights) > 0 else 0.001
                state_error_weights[len(original_weights):] = default_weight
        
        return state_error_weights
    
    def get_diagnostics(self, episode_num=0):
        """Enhanced diagnostics for DreamerV3 monitoring"""
        if len(self.error_history) < 2:
            return None
        
        current = self.error_history[-1]
        
        diagnostics = {
            'current_errors': current,
            'surge_yaw_magnitude': np.sqrt(current['surge']**2 + current['yaw']**2),
            'pitch_depth_magnitude': np.sqrt(current['pitch']**2 + current['depth']**2),
            'coupling_weights': {
                'surge_yaw': self.surge_yaw_coupling,
                'pitch_depth': self.pitch_depth_coupling
            },
            'learning_phase': self._determine_learning_phase(episode_num)
        }
        
        # Enhanced diagnostics for DreamerV3
        if len(self.error_history) > 5:
            recent_errors = self.error_history[-5:]
            
            # Surge asymmetry analysis
            positive_surges = [h['surge'] for h in recent_errors if h['surge'] > 0.1]
            negative_surges = [h['surge'] for h in recent_errors if h['surge'] < -0.1]
            
            diagnostics['surge_asymmetry'] = {
                'positive_avg': np.mean(positive_surges) if positive_surges else 0,
                'negative_avg': np.mean(negative_surges) if negative_surges else 0,
                'asymmetry_ratio': len(positive_surges) / max(len(negative_surges), 1)
            }
            
            # Yaw bias analysis
            yaw_errors = [h['yaw'] for h in recent_errors]
            diagnostics['yaw_bias'] = {
                'mean_error': np.mean(yaw_errors),
                'bias_magnitude': abs(np.mean(yaw_errors)),
                'variability': np.std(yaw_errors)
            }
            
            # Pitch persistence analysis
            pitch_errors = [h['pitch'] for h in recent_errors]
            diagnostics['pitch_analysis'] = {
                'mean_error': np.mean(pitch_errors),
                'high_error_persistence': sum(1 for p in pitch_errors if p > 0.2) / len(pitch_errors),
                'improvement_trend': pitch_errors[0] - pitch_errors[-1]
            }
            
            # DreamerV3 specific metrics
            diagnostics['world_model_metrics'] = {
                'temporal_consistency': self._calculate_temporal_consistency(current),
                'prediction_accuracy': self._calculate_predictive_accuracy_bonus(current, episode_num),
                'imagination_bonus': self._calculate_imagination_bonus(current, episode_num)
            }
        
        return diagnostics