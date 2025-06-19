#!/usr/bin/env python3
"""
CSV Data Organizer for AUV Training
Helps organize CSV files from rosbags into the expected structure
"""

import os
import pandas as pd
import glob
import shutil
from pathlib import Path
import argparse

def organize_csv_files(source_directory, target_directory):
    """
    Organize CSV files from rosbags into the expected structure for training
    
    Based on your actual CSV files from the screenshot:
    - Multiple CSV files, each representing a different ROS topic
    - Naming pattern appears to be: race2_auv_[topic_path].csv
    """
    
    # Create target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)
    
    # Define mapping from your actual file patterns to standardized names
    file_mappings = {
        # State and error data
        'race2_auv_controller_process_value.csv': 'race2_auv_controller_process_value.csv',
        'race2_auv_controller_process_error.csv': 'race2_auv_controller_process_error.csv',
        
        # IMU data
        'race2_auv_imu_data.csv': 'race2_auv_imu_data.csv',
        
        # Thruster commands - looking for your actual file names
        '*race2_auv_control_thruster_heave_bow*.csv': 'race2_auv_control_thruster_heave_bow.csv',
        '*race2_auv_control_thruster_heave_stern*.csv': 'race2_auv_control_thruster_heave_stern.csv', 
        '*race2_auv_control_thruster_surge_port*.csv': 'race2_auv_control_thruster_surge_port.csv',
        '*race2_auv_control_thruster_surge_starboard*.csv': 'race2_auv_control_thruster_surge_starboard.csv',
        
        # Servo commands
        '*race2_auv_control_surge_port_servo*.csv': 'race2_auv_control_surge_port_servo.csv',
        '*race2_auv_control_surge_starboard_servo*.csv': 'race2_auv_control_surge_starboard_servo.csv',
        
        # Alternative patterns based on your screenshot
        '*control_force_heave_bow*.csv': 'race2_auv_control_thruster_heave_bow.csv',
        '*control_force_heave_stern*.csv': 'race2_auv_control_thruster_heave_stern.csv',
        '*control_force_surge_port*.csv': 'race2_auv_control_thruster_surge_port.csv', 
        '*control_force_surge_starboard*.csv': 'race2_auv_control_thruster_surge_starboard.csv',
        
        # Additional patterns from your screenshot
        '*controller_process_set_point*.csv': 'race2_auv_controller_process_set_point.csv',
        '*stonefish_servo_desired*.csv': 'race2_auv_control_surge_port_servo.csv',
        '*stonefish_servo_joint_states*.csv': 'race2_auv_control_surge_starboard_servo.csv',
    }
    
    found_files = {}
    
    print(f"Searching for CSV files in: {source_directory}")
    print("=" * 60)
    
    # First, list all CSV files found
    all_csv_files = glob.glob(os.path.join(source_directory, "*.csv"))
    print(f"Found {len(all_csv_files)} CSV files:")
    for f in sorted(all_csv_files):
        print(f"  {os.path.basename(f)}")
    print()
    
    # Search for files matching each pattern
    for pattern, target_name in file_mappings.items():
        if '*' in pattern:
            # Pattern matching
            matching_files = glob.glob(os.path.join(source_directory, pattern))
        else:
            # Exact filename match
            exact_path = os.path.join(source_directory, pattern)
            matching_files = [exact_path] if os.path.exists(exact_path) else []
        
        if matching_files:
            # Use the first matching file
            source_file = matching_files[0]
            target_file = os.path.join(target_directory, target_name)
            
            # Skip if target already exists (avoid duplicates)
            if target_name in found_files:
                print(f"⚠ {target_name} already mapped, skipping {os.path.basename(source_file)}")
                continue
            
            # Copy file to target location
            shutil.copy2(source_file, target_file)
            found_files[target_name] = source_file
            
            print(f"✓ {target_name}")
            print(f"  Source: {os.path.basename(source_file)}")
            print(f"  Copied to: {target_file}")
            
            # Show basic file info
            try:
                df = pd.read_csv(target_file)
                print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
                if len(df.columns) <= 10:  # Only show columns if reasonable number
                    print(f"  Columns: {list(df.columns)}")
                else:
                    print(f"  Columns: {list(df.columns[:5])} ... (and {len(df.columns)-5} more)")
                    
                # Check for timestamp info
                if 'timestamp' in df.columns:
                    if len(df) > 1:
                        dt = df['timestamp'].iloc[1] - df['timestamp'].iloc[0]
                        freq = 1.0/dt if dt > 0 else 0
                        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
                        print(f"  Duration: {duration:.1f}s, Frequency: ~{freq:.1f}Hz")
                        
            except Exception as e:
                print(f"  Warning: Could not read CSV: {e}")
            print()
            
        else:
            print(f"✗ {target_name}")
            print(f"  Pattern: {pattern}")
            print(f"  No matching files found")
            print()
    
    print("=" * 60)
    print(f"Summary: {len(found_files)}/{len(file_mappings)} expected files found")
    
    # Show unmapped files
    mapped_sources = set(os.path.basename(f) for f in found_files.values())
    unmapped_files = [f for f in all_csv_files if os.path.basename(f) not in mapped_sources]
    
    if unmapped_files:
        print(f"\nUnmapped CSV files ({len(unmapped_files)}):")
        for f in sorted(unmapped_files):
            print(f"  {os.path.basename(f)}")
        print("\nYou may need to manually map these files or update the patterns.")
    
    if len(found_files) >= 3:  # At least state, error, and one actuator file
        print("✓ Sufficient files found for training")
        
        # Create a simple validation script
        create_validation_script(target_directory)
        
    else:
        print("✗ Insufficient files for training")
        print("  Minimum required: state data, error data, and at least one actuator file")
    
    return found_files

def create_validation_script(csv_directory):
    """Create a simple script to validate the CSV data"""
    
    validation_script = f'''#!/usr/bin/env python3
"""
Validate CSV data for AUV training
Generated automatically by csv_organizer.py
"""

import pandas as pd
import numpy as np
import os

def validate_csv_data():
    csv_dir = "{csv_directory}"
    
    print("Validating CSV data...")
    print("=" * 50)
    
    # Check each expected file
    expected_files = [
        'race2_auv_controller_process_value.csv',
        'race2_auv_controller_process_error.csv',
        'race2_auv_imu_data.csv',
        'race2_auv_control_thruster_heave_bow.csv',
        'race2_auv_control_thruster_heave_stern.csv',
        'race2_auv_control_thruster_surge_port.csv',
        'race2_auv_control_thruster_surge_starboard.csv',
        'race2_auv_control_surge_port_servo.csv',
        'race2_auv_control_surge_starboard_servo.csv'
    ]
    
    file_info = {{}}
    
    for filename in expected_files:
        filepath = os.path.join(csv_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                file_info[filename] = {{
                    'exists': True,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'time_span': None
                }}
                
                # Check for time information
                if 'timestamp' in df.columns:
                    time_span = df['timestamp'].max() - df['timestamp'].min()
                    file_info[filename]['time_span'] = time_span
                elif 'time' in df.columns:
                    time_span = df['time'].max() - df['time'].min()
                    file_info[filename]['time_span'] = time_span
                
                print(f"✓ {{filename}}")
                print(f"  Rows: {{len(df)}}")
                print(f"  Columns: {{df.columns.tolist()}}")
                if file_info[filename]['time_span']:
                    print(f"  Duration: {{file_info[filename]['time_span']:.2f}} seconds")
                print()
                
            except Exception as e:
                file_info[filename] = {{'exists': True, 'error': str(e)}}
                print(f"✗ {{filename}} - Error: {{e}}")
        else:
            file_info[filename] = {{'exists': False}}
            print(f"✗ {{filename}} - File not found")
    
    # Check data consistency
    print("\\nData Consistency Checks:")
    print("-" * 30)
    
    state_file = 'race2_auv_controller_process_value.csv'
    error_file = 'race2_auv_controller_process_error.csv'
    
    if (state_file in file_info and file_info[state_file]['exists'] and 
        error_file in file_info and file_info[error_file]['exists']):
        
        state_rows = file_info[state_file]['rows']
        error_rows = file_info[error_file]['rows']
        
        print(f"State data rows: {{state_rows}}")
        print(f"Error data rows: {{error_rows}}")
        
        if abs(state_rows - error_rows) < 0.1 * max(state_rows, error_rows):
            print("✓ State and error data have compatible lengths")
        else:
            print("⚠ Warning: State and error data have very different lengths")
    
    print("\\nValidation complete!")
    print("You can now use this data for training with:")
    print(f"python train_sac.py --csv_directory {{csv_dir}}")

if __name__ == "__main__":
    validate_csv_data()
'''
    
    validation_file = os.path.join(csv_directory, "validate_data.py")
    with open(validation_file, 'w') as f:
        f.write(validation_script)
    
    # Make it executable
    os.chmod(validation_file, 0o755)
    
    print(f"Created validation script: {validation_file}")
    print("Run it with: python validate_data.py")

def main():
    parser = argparse.ArgumentParser(description='Organize CSV files from rosbags for AUV training')
    parser.add_argument('source_directory', help='Directory containing CSV files from rosbags')
    parser.add_argument('target_directory', help='Directory to organize CSV files for training')
    parser.add_argument('--validate', action='store_true', help='Run validation after organizing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_directory):
        print(f"Error: Source directory does not exist: {args.source_directory}")
        return
    
    print("CSV Data Organizer for AUV Training")
    print("=" * 60)
    print(f"Source: {args.source_directory}")
    print(f"Target: {args.target_directory}")
    print()
    
    # Organize the files
    found_files = organize_csv_files(args.source_directory, args.target_directory)
    
    # Run validation if requested
    if args.validate and found_files:
        print("\\nRunning validation...")
        validation_script = os.path.join(args.target_directory, "validate_data.py")
        if os.path.exists(validation_script):
            os.system(f"python {validation_script}")

if __name__ == "__main__":
    main()