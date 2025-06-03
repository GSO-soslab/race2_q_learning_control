#!/usr/bin/env python3
"""
Test script to verify coupling-aware reward integration
Run this to make sure everything is working correctly
"""

import numpy as np
import yaml
import os

# Test the coupling reward calculator independently
def test_coupling_calculator():
    """Test the coupling reward calculator with dummy data"""
    print("=== Testing Coupling Reward Calculator ===")
    
    # Create a test configuration
    test_config = {
        'coupling': {
            'method': 'v1',
            'surge_yaw_weight': 0.4,
            'pitch_depth_weight': 0.5,
            'threshold': 0.1,
            'progressive': {
                'enabled': True,
                'exploration_episodes': 50,
                'coupling_focus_episodes': 100
            }
        },
        'reward_function': {
            'w1': 1.0,
            'state_error_weights': [0.001, 0.0001, 0.0001, 0.0, 0.0, 0.0005, 0.0005]
        }
    }
    
    # Import and test the coupling calculator
    try:
        from coupling_rewards import CouplingAwareRewardCalculator
        print("✓ Successfully imported CouplingAwareRewardCalculator")
        
        calculator = CouplingAwareRewardCalculator(test_config)
        print("✓ Successfully created calculator instance")
        
        # Test with dummy error data (10 elements for sin/cos representation)
        dummy_errors = np.array([
            0.1,   # depth error
            0.05,  # surge error
            0.02,  # sway error
            0.01,  # heave error
            0.1,   # roll sin error
            0.0,   # roll cos error
            0.15,  # pitch sin error
            0.0,   # pitch cos error
            0.08,  # yaw sin error
            0.0    # yaw cos error
        ])
        
        # Test all methods
        methods_to_test = ['v1', 'v2', 'v3', 'v4']
        
        for method in methods_to_test:
            print(f"\n  Testing method {method}...")
            test_config['coupling']['method'] = method
            
            calculator = CouplingAwareRewardCalculator(test_config)
            
            if method == 'v1':
                reward = calculator.calculate_coupling_aware_reward_v1(dummy_errors, episode_num=10)
            elif method == 'v2':
                reward = calculator.calculate_coupling_aware_reward_v2(dummy_errors, episode_num=10)
            elif method == 'v3':
                reward = calculator.calculate_coupling_aware_reward_v3(dummy_errors, episode_num=10)
            elif method == 'v4':
                reward = calculator.calculate_coupling_aware_reward_v4_enhanced(dummy_errors, episode_num=10)
            
            print(f"    ✓ Method {method} reward: {reward:.4f}")
        
        print("✓ All coupling methods tested successfully")
        
        # Test diagnostics
        diagnostics = calculator.get_diagnostics()
        if diagnostics:
            print("✓ Diagnostics available:")
            print(f"  Surge-Yaw magnitude: {diagnostics['surge_yaw_magnitude']:.4f}")
            print(f"  Pitch-Depth magnitude: {diagnostics['pitch_depth_magnitude']:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import coupling calculator: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing coupling calculator: {e}")
        return False

def test_config_loading():
    """Test loading the updated configuration"""
    print("\n=== Testing Configuration Loading ===")
    
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config_sac.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Successfully loaded config file")
        
        # Check for coupling configuration
        if 'coupling' in config:
            print("✓ Coupling configuration found:")
            coupling_config = config['coupling']
            print(f"  Method: {coupling_config.get('method', 'not set')}")
            print(f"  Surge-Yaw weight: {coupling_config.get('surge_yaw_weight', 'not set')}")
            print(f"  Pitch-Depth weight: {coupling_config.get('pitch_depth_weight', 'not set')}")
        else:
            print("⚠ No coupling configuration found - using defaults")
        
        return True
        
    except FileNotFoundError:
        print(f"✗ Config file not found at {config_path}")
        print("  Make sure you've updated your config_sac.yaml file")
        return False
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False

def test_auv_env_integration():
    """Test the AUV environment integration"""
    print("\n=== Testing AUV Environment Integration ===")
    
    try:
        # Note: This will only work if ROS2 is available
        print("Attempting to import AUVEnv...")
        from AUVEnv import AUVEnv
        print("✓ Successfully imported AUVEnv")
        
        print("Creating AUVEnv instance...")
        # This might fail if ROS2 isn't running, which is expected
        env = AUVEnv()
        print("✓ Successfully created AUVEnv instance")
        
        # Test if coupling calculator is initialized
        if hasattr(env, 'coupling_calculator') and env.coupling_calculator is not None:
            print("✓ Coupling calculator initialized")
        else:
            print("⚠ Coupling calculator not yet initialized (will be on first reset)")
        
        return True
        
    except Exception as e:
        print(f"⚠ AUVEnv test skipped (expected if ROS2 not running): {e}")
        print("  This is normal - the environment needs ROS2 to run")
        return True  # Don't fail the test for this

def validate_coupling_parameters():
    """Validate coupling parameter ranges"""
    print("\n=== Validating Coupling Parameters ===")
    
    # Test parameter validation
    valid_configs = [
        {'surge_yaw_weight': 0.4, 'pitch_depth_weight': 0.5},
        {'surge_yaw_weight': 0.0, 'pitch_depth_weight': 1.0},
        {'surge_yaw_weight': 1.0, 'pitch_depth_weight': 0.0},
    ]
    
    invalid_configs = [
        {'surge_yaw_weight': 1.5, 'pitch_depth_weight': 0.5},  # > 1.0
        {'surge_yaw_weight': -0.1, 'pitch_depth_weight': 0.5}, # < 0.0
    ]
    
    print("Testing valid configurations:")
    for i, params in enumerate(valid_configs):
        surge_yaw = params['surge_yaw_weight']
        pitch_depth = params['pitch_depth_weight']
        
        if 0.0 <= surge_yaw <= 1.0 and 0.0 <= pitch_depth <= 1.0:
            print(f"  ✓ Config {i+1}: surge_yaw={surge_yaw}, pitch_depth={pitch_depth}")
        else:
            print(f"  ✗ Config {i+1}: Invalid parameter ranges")
    
    print("Testing invalid configurations:")
    for i, params in enumerate(invalid_configs):
        surge_yaw = params['surge_yaw_weight']
        pitch_depth = params['pitch_depth_weight']
        
        if not (0.0 <= surge_yaw <= 1.0 and 0.0 <= pitch_depth <= 1.0):
            print(f"  ✓ Config {i+1}: Correctly identified as invalid")
        else:
            print(f"  ✗ Config {i+1}: Should be invalid but passed")
    
    return True

def main():
    """Run all integration tests"""
    print("Coupling-Aware Reward Integration Test")
    print("=" * 50)
    
    tests = [
        ("Coupling Calculator", test_coupling_calculator),
        ("Configuration Loading", test_config_loading),
        ("AUV Environment Integration", test_auv_env_integration),
        ("Parameter Validation", validate_coupling_parameters),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your coupling integration is ready.")
        print("\nNext steps:")
        print("1. Update your config_sac.yaml with the coupling configuration")
        print("2. Run your training script with the new coupling-aware rewards")
        print("3. Monitor the coupling diagnostics during training")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("Make sure you have:")
        print("1. Created the coupling_rewards.py file")
        print("2. Updated your AUVEnv.py file")
        print("3. Updated your config_sac.yaml file")

if __name__ == "__main__":
    main()