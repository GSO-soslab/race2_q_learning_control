#!/usr/bin/env python3
"""
Debug script to analyze CSV data loading and synchronization issues
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def debug_csv_directory(csv_directory):
    """Debug CSV directory and files"""
    print(f"🔍 Debugging CSV directory: {csv_directory}")
    print("=" * 60)
    
    if not os.path.exists(csv_directory):
        print(f"❌ ERROR: Directory does not exist!")
        return False
    
    # List all files in directory
    all_files = os.listdir(csv_directory)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    print(f"📁 Directory contents:")
    print(f"   Total files: {len(all_files)}")
    print(f"   CSV files: {len(csv_files)}")
    
    if not csv_files:
        print(f"❌ ERROR: No CSV files found!")
        return False
    
    print(f"\n📋 CSV files found:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"   {i:2d}. {csv_file}")
    
    return True

def debug_csv_file_loading(csv_directory):
    """Debug individual CSV file loading"""
    print(f"\n🔍 Analyzing individual CSV files...")
    print("=" * 60)
    
    # Expected file mappings from csv_data_manager.py
    file_mappings = {
        'race2_auv_controller_process_value.csv': 'state',
        'race2_auv_controller_process_error.csv': 'error',
        'race2_auv_imu_data.csv': 'imu',
        'race2_auv_control_thruster_heave_bow.csv': 'thruster_heave_bow',
        'race2_auv_control_thruster_heave_stern.csv': 'thruster_heave_stern',
        'race2_auv_control_thruster_surge_port.csv': 'thruster_surge_port',
        'race2_auv_control_thruster_surge_starboard.csv': 'thruster_surge_starboard',
        'race2_auv_control_surge_port_servo.csv': 'servo_port',
        'race2_auv_control_surge_starboard_servo.csv': 'servo_starboard'
    }
    
    loaded_data = {}
    
    for filename, topic_name in file_mappings.items():
        file_path = os.path.join(csv_directory, filename)
        
        if os.path.exists(file_path):
            try:
                print(f"\n✅ Found: {filename}")
                df = pd.read_csv(file_path)
                
                print(f"   📊 Shape: {df.shape} (rows × columns)")
                print(f"   📋 Columns: {list(df.columns)}")
                
                # Check for timestamp column
                has_timestamp = False
                timestamp_col = None
                if 'timestamp' in df.columns:
                    timestamp_col = 'timestamp'
                    has_timestamp = True
                elif 'time' in df.columns:
                    timestamp_col = 'time' 
                    has_timestamp = True
                
                if has_timestamp:
                    print(f"   ⏰ Timestamp column: '{timestamp_col}'")
                    
                    # Analyze timestamps
                    timestamps = df[timestamp_col]
                    if len(timestamps) > 1:
                        dt = timestamps.iloc[1] - timestamps.iloc[0]
                        freq = 1.0 / dt if dt > 0 else 0
                        duration = timestamps.iloc[-1] - timestamps.iloc[0]
                        
                        print(f"   📈 Time info:")
                        print(f"      Start: {timestamps.iloc[0]}")
                        print(f"      End: {timestamps.iloc[-1]}")
                        print(f"      Duration: {duration:.2f} seconds")
                        print(f"      Sample rate: {freq:.1f} Hz")
                        print(f"      Time step: {dt:.3f} seconds")
                    else:
                        print(f"   ⚠️  Only one timestamp value")
                else:
                    print(f"   ❌ No timestamp column found!")
                
                # Show first few rows
                print(f"   📝 First 3 rows:")
                for idx, row in df.head(3).iterrows():
                    print(f"      Row {idx}: {dict(row)}")
                
                loaded_data[topic_name] = df
                
            except Exception as e:
                print(f"   ❌ Error loading {filename}: {e}")
        else:
            print(f"❌ Missing: {filename}")
    
    return loaded_data

def debug_synchronization(loaded_data):
    """Debug the synchronization process"""
    print(f"\n🔍 Debugging synchronization process...")
    print("=" * 60)
    
    if not loaded_data:
        print("❌ No data loaded - cannot debug synchronization")
        return
    
    print(f"📊 Loaded topics: {list(loaded_data.keys())}")
    
    # Find common time range
    min_time = float('inf')
    max_time = float('-inf')
    
    print(f"\n⏰ Time range analysis:")
    for topic, df in loaded_data.items():
        if 'timestamp' in df.columns:
            timestamp_col = 'timestamp'
        elif 'time' in df.columns:
            timestamp_col = 'time'
        else:
            print(f"   ❌ {topic}: No timestamp column")
            continue
        
        if len(df) == 0:
            print(f"   ❌ {topic}: Empty dataframe")
            continue
        
        topic_min = df[timestamp_col].min()
        topic_max = df[timestamp_col].max()
        duration = topic_max - topic_min
        
        print(f"   📈 {topic}: {topic_min:.2f}s to {topic_max:.2f}s (duration: {duration:.2f}s)")
        
        min_time = min(min_time, topic_min)
        max_time = max(max_time, topic_max)
    
    if min_time == float('inf'):
        print("❌ No valid timestamps found across all files!")
        return
    
    print(f"\n🎯 Common time range: {min_time:.2f}s to {max_time:.2f}s")
    common_duration = max_time - min_time
    print(f"   Duration: {common_duration:.2f}s")
    
    # Calculate expected synchronized timesteps
    dt = 0.1  # 10Hz
    expected_timesteps = int(common_duration / dt)
    print(f"   Expected timesteps at 10Hz: {expected_timesteps}")
    
    if expected_timesteps <= 0:
        print("❌ No valid time range for synchronization!")
        return
    
    # Check essential topics
    essential_topics = ['state', 'error']
    missing_essential = []
    for topic in essential_topics:
        if topic not in loaded_data:
            missing_essential.append(topic)
    
    if missing_essential:
        print(f"❌ Missing essential topics: {missing_essential}")
        print("   Cannot proceed without 'state' and 'error' data")
        return
    
    print(f"✅ Synchronization should work!")
    print(f"   Essential topics present: {essential_topics}")
    print(f"   Expected synchronized timesteps: {expected_timesteps}")

def debug_csv_manager_creation(csv_directory):
    """Debug the actual CSV manager creation"""
    print(f"\n🔍 Testing CSV Manager creation...")
    print("=" * 60)
    
    try:
        # Import here to avoid issues if files don't exist
        from csv_data_manager import CSVDataManager
        
        # Create a minimal config for testing
        test_config = {
            'environment': {
                'thruster_size': 4,
                'servo_joints_size': 2
            }
        }
        
        print("📦 Creating CSVDataManager...")
        csv_manager = CSVDataManager(csv_directory, test_config)
        
        print(f"✅ CSV Manager created successfully!")
        print(f"   Total synchronized timesteps: {csv_manager.total_steps}")
        
        if csv_manager.total_steps > 0:
            print(f"   First timestamp: {csv_manager.synchronized_data[0]['timestamp']:.2f}s")
            print(f"   Last timestamp: {csv_manager.synchronized_data[-1]['timestamp']:.2f}s")
            print(f"   Available topics in first timestep: {list(csv_manager.synchronized_data[0].keys())}")
        else:
            print("❌ No synchronized data created!")
            
            # Debug why synchronization failed
            print("\n🔍 Debugging synchronization failure...")
            if hasattr(csv_manager, 'data'):
                print(f"   Raw data topics loaded: {list(csv_manager.data.keys())}")
                for topic, df in csv_manager.data.items():
                    print(f"   {topic}: {len(df)} rows")
        
        return csv_manager
        
    except ImportError as e:
        print(f"❌ Cannot import CSVDataManager: {e}")
        print("   Make sure csv_data_manager.py is in the same directory")
        return None
    except Exception as e:
        print(f"❌ Error creating CSV Manager: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug CSV data loading for AUV training')
    parser.add_argument('csv_directory', help='Directory containing CSV files')
    args = parser.parse_args()
    
    print("🐛 CSV Data Debug Tool")
    print("=" * 60)
    
    # Step 1: Check directory and files
    if not debug_csv_directory(args.csv_directory):
        return
    
    # Step 2: Load and analyze individual files
    loaded_data = debug_csv_file_loading(args.csv_directory)
    
    # Step 3: Debug synchronization logic
    debug_synchronization(loaded_data)
    
    # Step 4: Test actual CSV manager creation
    csv_manager = debug_csv_manager_creation(args.csv_directory)
    
    print(f"\n✅ Debug complete!")
    
    if csv_manager and csv_manager.total_steps > 0:
        print(f"🎉 CSV data is ready for training!")
        print(f"   Use: python sac_sb3.py --csv_directory {args.csv_directory}")
    else:
        print(f"❌ CSV data has issues that need to be fixed.")
        print(f"\n💡 Suggested fixes:")
        print(f"   1. Make sure you have race2_auv_controller_process_value.csv")
        print(f"   2. Make sure you have race2_auv_controller_process_error.csv") 
        print(f"   3. Check that CSV files have 'timestamp' or 'time' columns")
        print(f"   4. Verify timestamps are in seconds and have reasonable values")

if __name__ == "__main__":
    main()
    