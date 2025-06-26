# coupling_rewards.py - Complete Working Version
import numpy as np

class CouplingAwareRewardCalculator:
    """
    Novel coupling-aware reward functions for AUV control where:
    - Surge and heading (yaw) are coupled
    - Pitch and depth are coupled
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
        
    def calculate_coupling_aware_reward_v1(self, state_error_array, episode_num=0):
        """
        Method 1: Cross-Correlation Coupling Reward
        Rewards when coupled states move in the same direction
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
        
        # FIXED: Convert sin/cos back to actual angle error magnitude
        # Use atan2 to get the actual angle, then take absolute value
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
        
        total_reward = (base_performance + 
                       coupling_strength * self.surge_yaw_coupling * surge_yaw_coupling_bonus +
                       coupling_strength * self.pitch_depth_coupling * pitch_depth_coupling_bonus)
        
        return float(total_reward.item())
    
    def calculate_coupling_aware_reward_v2(self, state_error_array, episode_num=0):
        """
        Method 2: Hierarchical Progressive Learning
        Prioritizes learning coupled pairs with phase-based training
        """
        # Extract errors
        depth_error = state_error_array[0]
        surge_error = state_error_array[1]
        
        pitch_sin_err, pitch_cos_err = state_error_array[6], state_error_array[7]
        yaw_sin_err, yaw_cos_err = state_error_array[8], state_error_array[9]
        
        # FIXED: Use actual angle error magnitude
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
        
        # Progressive learning phases
        phase = self._determine_learning_phase(episode_num)
        
        if phase == "coupling_focus":
            # Focus only on coupled pairs
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
            
            return coupling_reward
            
        elif phase == "refinement":
            # Include all errors but weight coupled pairs higher
            return self.calculate_coupling_aware_reward_v1(state_error_array, episode_num)
            
        else:  # "exploration" phase
            # Standard reward to encourage exploration
            return self._calculate_standard_reward(state_error_array)
    
    def calculate_coupling_aware_reward_v3(self, state_error_array, episode_num=0):
        """
        Method 3: Dynamic Coupling Matrix Reward
        Uses time-varying coupling weights based on error correlation
        """
        # Extract current errors
        depth_error = state_error_array[0]
        surge_error = state_error_array[1]
        
        pitch_sin_err, pitch_cos_err = state_error_array[6], state_error_array[7]
        yaw_sin_err, yaw_cos_err = state_error_array[8], state_error_array[9]
        
        # FIXED: Use actual angle error magnitude
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
        
        return base_reward + coupling_penalty
    
    def calculate_coupling_aware_reward_v4(self, state_error_array, episode_num=0):
        """
        Method 4: Energy-Based Coupling Reward
        Models coupling as energy minimization problem
        """
        # Extract errors
        depth_error = state_error_array[0]
        surge_error = state_error_array[1]
        
        pitch_sin_err, pitch_cos_err = state_error_array[6], state_error_array[7]
        yaw_sin_err, yaw_cos_err = state_error_array[8], state_error_array[9]
        
        # FIXED: Use actual angle error magnitude
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
        
        # Define coupling "springs" - stronger coupling = stiffer spring
        surge_yaw_spring_energy = 0.5 * self.spring_stiffness * self.surge_yaw_coupling * (surge_error - yaw_error_mag)**2
        pitch_depth_spring_energy = 0.5 * self.spring_stiffness * self.pitch_depth_coupling * (pitch_error_mag - depth_error)**2
        
        # Individual error energies
        individual_energy = 0.5 * (depth_error**2 + surge_error**2 + pitch_error_mag**2 + yaw_error_mag**2)
        
        # Damping energy (if we have velocity information)
        damping_energy = 0
        if len(self.error_history) > 1:
            prev_errors = self.error_history[-2]
            velocity_surge = surge_error - prev_errors['surge']
            velocity_yaw = yaw_error_mag - prev_errors['yaw']
            velocity_pitch = pitch_error_mag - prev_errors['pitch']
            velocity_depth = depth_error - prev_errors['depth']
            
            damping_energy = self.damping_factor * (velocity_surge**2 + velocity_yaw**2 + 
                                                  velocity_pitch**2 + velocity_depth**2)
        
        # Total system energy
        total_energy = individual_energy + surge_yaw_spring_energy + pitch_depth_spring_energy + damping_energy
        
        # Reward is negative energy (minimize energy = maximize reward)
        return -total_energy
    
    def calculate_coupling_aware_reward_v4_enhanced(self, state_error_array, episode_num=0):
        """
        Enhanced Method 4: Energy-Based Coupling with specific issue fixes
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
            surge_error = float(state_error_array[1]) # to scaLe surge up compared to other guys
            
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
            if pitch_error_mag > 0.2:  # High pitch error (your 0.3 rad issue)
                pitch_depth_spring_stiffness *= 2.0  # Stronger coupling
            
            if abs(yaw_error_mag) > 0.3:  # Yaw offset issue
                surge_yaw_spring_stiffness *= 1.5  # Stronger coupling
            
            # Energy calculations with adaptive stiffness
            surge_yaw_spring_energy = 0.5 * surge_yaw_spring_stiffness * self.surge_yaw_coupling * (surge_error - yaw_error_mag)**2
            pitch_depth_spring_energy = 0.5 * pitch_depth_spring_stiffness * self.pitch_depth_coupling * (pitch_error_mag - depth_error)**2
            
            # Individual error energies with enhanced weights for problem areas
            individual_energy = 0.5 * (
                depth_error**2 + 
                ((np.exp( surge_error))**2) + 
                2.0 * pitch_error_mag**2 +  # ENHANCED: 2x weight for pitch
                1.5 * yaw_error_mag**2       # ENHANCED: 1.5x weight for yaw offset
            )

            # ENHANCED: Asymmetric surge penalty
            surge_asymmetry_penalty = 0
            # if len(self.error_history) > 5:
            #     recent_surges = [h['surge'] for h in self.error_history[-5:]]
            #     positive_surges = [abs(s) for s in recent_surges if s > 0.1]
            #     negative_surges = [abs(s) for s in recent_surges if s < -0.1]
                
            #     if len(positive_surges) > 0 and len(negative_surges) > 0:
            #         positive_response = np.mean(positive_surges)
            #         negative_response = np.mean(negative_surges)
            #         asymmetry = abs(positive_response - negative_response) / max(positive_response, negative_response)
            #         surge_asymmetry_penalty = 0.1 * asymmetry
            
            surge_progress_bonus = 0
            if len(self.error_history) > 2:
                current_surge_error = abs(self.error_history[-1]['surge'])
                prev_surge_error = abs(self.error_history[-2]['surge'])
                if prev_surge_error > current_surge_error:  # Improvement
                    surge_progress_bonus = 10 * (prev_surge_error - current_surge_error)
                else:
                    surge_progress_bonus = - 10 * (prev_surge_error - current_surge_error)

            print("Surge bonus",surge_progress_bonus)
            # ENHANCED: Yaw bias penalty
            yaw_bias_penalty = 0
            if len(self.error_history) > 10:
                recent_yaws = [h['yaw'] for h in self.error_history[-10:]]
                mean_yaw_error = np.mean(recent_yaws)
                if abs(mean_yaw_error) > 0.2:  # Systematic offset
                    yaw_bias_penalty = 0.05 * abs(mean_yaw_error)
            
            # Enhanced damping energy
            damping_energy = 0
            if len(self.error_history) > 1:
                prev_errors = self.error_history[-2]
                
                # Velocity calculations
                velocity_surge = 10 * (surge_error - prev_errors['surge'])
                velocity_yaw = yaw_error_mag - prev_errors['yaw']
                velocity_pitch = pitch_error_mag - prev_errors['pitch']
                velocity_depth = depth_error - prev_errors['depth']
                
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
            
            # print("individual_energy:", individual_energy)
            # print("surge_yaw_spring_energy:", surge_yaw_spring_energy)
            # print("pitch_depth_spring_energy:", pitch_depth_spring_energy)
            # print("damping_energy:", damping_energy)
            # print("surge_asymmetry_penalty:", surge_asymmetry_penalty)
            # print("yaw_bias_penalty:", yaw_bias_penalty)

            # Total enhanced energy
            total_energy = (individual_energy + 
                        surge_yaw_spring_energy + 
                        pitch_depth_spring_energy + 
                        damping_energy +
                        surge_asymmetry_penalty +
                        yaw_bias_penalty)
            # print("total_energy:", total_energy)
            # ENHANCED: Adaptive energy scaling
            # # Scale energy penalty based on learning progress
            # if episode_num < 1000:
            #     energy_scale = 0.5  # Gentler penalty during early learning
            # elif episode_num < 5000:
            #     energy_scale = 0.8  # Medium penalty during intermediate learning
            # else:
            #     energy_scale = 1.0  # Full penalty for mature learning
            
            # Return negative scaled energy
            # return float(-total_energy * energy_scale)
            # return float(-total_energy * energy_scale + surge_progress_bonus)
            print("Total energy guy" , float(-total_energy) + surge_progress_bonus)
            return float(-total_energy) + surge_progress_bonus
        
        except Exception as e:
            print(f"Error in v4_enhanced calculation: {e}")
            print(f"state_error_array: {state_error_array}")
            return -1.0
    
    def _determine_learning_phase(self, episode_num):
        """Determine current learning phase for progressive training"""
        progressive_config = self.config.get('coupling', {}).get('progressive', {})
        
        if not progressive_config.get('enabled', False):
            return "refinement"  # Skip phases if progressive learning disabled
        
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
            new_coupling_matrix[0, 2] = new_coupling_matrix[2, 0] = abs(corr_matrix[0, 2]) * self.pitch_depth_coupling  # depth-pitch
            new_coupling_matrix[1, 3] = new_coupling_matrix[3, 1] = abs(corr_matrix[1, 3]) * self.surge_yaw_coupling    # surge-yaw
            
            # Apply adaptation rate
            self.coupling_matrix = (1 - self.adaptation_rate) * self.coupling_matrix + self.adaptation_rate * new_coupling_matrix
    
    def _calculate_standard_reward(self, state_error_array):
        """Standard reward calculation for exploration phase"""
        original_weights = np.array(self.config['reward_function']['state_error_weights'])
        # state_error_weights = self._expand_weights_for_sincos(original_weights, state_error_array)
        state_error_weights = original_weights
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
            # Increase weights for surge and yaw (indices 1, 8, 9) - check bounds first
            if len(enhanced_weights) > 1:
                enhanced_weights[1] *= (1.0 + 0.5 * self.surge_yaw_coupling)  # surge
            if len(enhanced_weights) > 8:
                enhanced_weights[8] *= (1.0 + 0.5 * self.surge_yaw_coupling)  # yaw sin
            if len(enhanced_weights) > 9:
                enhanced_weights[9] *= (1.0 + 0.5 * self.surge_yaw_coupling)  # yaw cos
        
        # If pitch and depth errors are both significant, increase their coupling
        pitch_depth_magnitude = np.sqrt(current_errors['pitch']**2 + current_errors['depth']**2)
        if pitch_depth_magnitude > self.coupling_threshold:
            # Increase weights for depth and pitch (indices 0, 6, 7) - check bounds first
            if len(enhanced_weights) > 0:
                enhanced_weights[0] *= (1.0 + 0.5 * self.pitch_depth_coupling)  # depth
            if len(enhanced_weights) > 6:
                enhanced_weights[6] *= (1.0 + 0.5 * self.pitch_depth_coupling)  # pitch sin
            if len(enhanced_weights) > 7:
                enhanced_weights[7] *= (1.0 + 0.5 * self.pitch_depth_coupling)  # pitch cos
        
        return enhanced_weights
    
    def _calculate_surge_yaw_coupling(self, surge_error, yaw_error):
        """Calculate coupling bonus for surge-yaw pair"""
        if len(self.error_history) < 2:
            return 0
        
        # Check if errors are moving in the same direction (coupled behavior)
        prev_surge = self.error_history[-2]['surge']
        prev_yaw = self.error_history[-2]['yaw']
        
        surge_direction = np.sign(surge_error - prev_surge)
        yaw_direction = np.sign(yaw_error - prev_yaw)
        
        # Reward synchronized movement
        synchronization = surge_direction * yaw_direction
        coupling_magnitude = np.sqrt(surge_error**2 + yaw_error**2)
        
        # Higher bonus for synchronized, low-error states
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
        
        # Stronger coupling when errors are large (need coordination)
        # Weaker coupling when errors are small (allow fine-tuning)
        return np.tanh(error_magnitude * 10)
    
    def _expand_weights_for_sincos(self, original_weights, state_error_array):
        """
        Dynamically expand weights array to account for sin/cos representation
        Sizes the output array based on state_error_array length
        """
        original_weights = np.array(original_weights)
        target_size = len(state_error_array)
        state_error_weights = np.zeros(target_size)
        
        if len(original_weights) == 7 and target_size == 10:
            # Standard AUV case: 7 original -> 10 expanded
            # Position and velocity errors (unchanged)
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
            
            # If we need more weights, use the last available weight or a default
            if target_size > len(original_weights):
                default_weight = original_weights[-1] if len(original_weights) > 0 else 0.001
                state_error_weights[len(original_weights):] = default_weight
        
        return state_error_weights
    
    def _expand_weights_for_sincos_flexible(self, original_weights, state_error_array):
        """
        Alternative: Completely flexible version that adapts to actual state_error_array size
        """
        original_weights = np.array(original_weights)
        target_size = len(state_error_array)
        
        if len(original_weights) == target_size:
            # Already the right size
            return original_weights
        
        if len(original_weights) * 2 - 3 == target_size:
            # Likely sin/cos expansion: 7 original -> 10 expanded (3 angles become 6 sin/cos)
            return self._expand_weights_for_sincos(original_weights)
        
        # Fallback: repeat or truncate to match target size
        if len(original_weights) < target_size:
            # Pad with the last weight value
            expanded = np.pad(original_weights, (0, target_size - len(original_weights)), 
                            mode='constant', constant_values=original_weights[-1])
            return expanded
        else:
            # Truncate to target size
            return original_weights[:target_size]
    
    def _get_state_structure_info(self):
        """
        Get information about the expected state structure from config
        Returns dict with structure information
        """
        # This could be made configurable in the future
        return {
            'position_errors': ['depth'],  # 1 element
            'velocity_errors': ['surge', 'sway', 'heave'],  # 3 elements  
            'orientation_errors': ['roll', 'pitch', 'yaw'],  # 3 elements -> 6 with sin/cos
            'total_original': 7,  # 1 + 3 + 3
            'total_expanded': 10  # 1 + 3 + 6
        }
    
    # def get_diagnostics(self, episode_num=0):
    #     """Get coupling diagnostics for monitoring"""
    #     if len(self.error_history) < 2:
    #         return None
        
    #     current = self.error_history[-1]
    #     diagnostics = {
    #         'current_errors': current,
    #         'surge_yaw_magnitude': np.sqrt(current['surge']**2 + current['yaw']**2),
    #         'pitch_depth_magnitude': np.sqrt(current['pitch']**2 + current['depth']**2),
    #         'coupling_weights': {
    #             'surge_yaw': self.surge_yaw_coupling,
    #             'pitch_depth': self.pitch_depth_coupling
    #         },
    #         'learning_phase': self._determine_learning_phase(episode_num)
    #     }
        
    #     # Add method-specific diagnostics
    #     if hasattr(self, 'coupling_matrix'):
    #         diagnostics['dynamic_coupling_matrix'] = self.coupling_matrix.tolist()
        
    #     if hasattr(self, 'spring_stiffness'):
    #         diagnostics['energy_parameters'] = {
    #             'spring_stiffness': self.spring_stiffness,
    #             'damping_factor': self.damping_factor
    #         }
        
    #     return diagnostics
    
    def get_diagnostics(self, episode_num=0):
        """Enhanced diagnostics for v4 troubleshooting"""
        if len(self.error_history) < 2:
            return None
        
        current = self.error_history[-1]
        
        # Calculate additional metrics
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
        
        # Enhanced diagnostics for your specific issues
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
        
        return diagnostics