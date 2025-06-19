#!/usr/bin/env python3
"""
Direct MCAP File Converter
Converts MCAP files to CSV using rosbag2_py directly with the file path
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    print("Error: ROS2 packages not found.")
    print("Make sure ROS2 is installed and sourced:")
    print("source /opt/ros/humble/setup.bash  # or your ROS2 distro")
    sys.exit(1)


class DirectMcapConverter:
    def __init__(self, mcap_file: str, output_dir: str = None):
        self.mcap_file = Path(mcap_file)
        self.output_dir = Path(output_dir) if output_dir else self.mcap_file.parent / f"{self.mcap_file.stem}_csv"
        self.topic_writers = {}
        self.topic_headers = {}
        
    def create_output_directory(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
    
    def sanitize_filename(self, topic_name: str) -> str:
        """Convert topic name to valid filename."""
        sanitized = topic_name.lstrip('/').replace('/', '_')
        sanitized = sanitized.replace(' ', '_').replace(':', '_').replace('?', '_')
        return sanitized
    
    def flatten_message(self, msg_dict: Dict, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested message structure for CSV output."""
        flattened = {}
        
        for key, value in msg_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self.flatten_message(value, full_key))
            elif isinstance(value, list):
                if len(value) > 0:
                    if isinstance(value[0], dict):
                        # For small lists, flatten each element
                        for i, item in enumerate(value[:5]):  # Limit to first 5 items
                            flattened.update(self.flatten_message(item, f"{full_key}[{i}]"))
                    else:
                        # For simple lists
                        if len(value) <= 10:
                            for i, item in enumerate(value):
                                flattened[f"{full_key}[{i}]"] = item
                        else:
                            flattened[full_key] = str(value)
                else:
                    flattened[full_key] = ""
            else:
                flattened[full_key] = value
                
        return flattened
    
    def message_to_dict(self, message) -> Dict[str, Any]:
        """Convert ROS2 message to dictionary."""
        try:
            if hasattr(message, '__slots__'):
                msg_dict = {}
                for slot in message.__slots__:
                    value = getattr(message, slot)
                    msg_dict[slot] = self.convert_value_to_serializable(value)
                return msg_dict
            else:
                return {"raw_data": str(message)}
        except Exception as e:
            return {"error": str(e), "raw_data": str(message)}
    
    def convert_value_to_serializable(self, value) -> Any:
        """Convert ROS2 message values to serializable format."""
        if hasattr(value, '__slots__'):
            result = {}
            for slot in value.__slots__:
                result[slot] = self.convert_value_to_serializable(getattr(value, slot))
            return result
        elif isinstance(value, list):
            return [self.convert_value_to_serializable(item) for item in value]
        elif hasattr(value, 'sec') and hasattr(value, 'nanosec'):
            return value.sec + value.nanosec * 1e-9
        elif hasattr(value, 'x') and hasattr(value, 'y') and hasattr(value, 'z'):
            return {"x": value.x, "y": value.y, "z": value.z}
        elif hasattr(value, 'w') and hasattr(value, 'x') and hasattr(value, 'y') and hasattr(value, 'z'):
            return {"w": value.w, "x": value.x, "y": value.y, "z": value.z}
        else:
            return value
    
    def get_csv_writer(self, topic_name: str, headers: List[str]):
        """Get or create CSV writer for a topic."""
        if topic_name not in self.topic_writers:
            filename = self.sanitize_filename(topic_name) + ".csv"
            filepath = self.output_dir / filename
            
            file_handle = open(filepath, 'w', newline='', encoding='utf-8')
            writer = csv.DictWriter(file_handle, fieldnames=headers)
            writer.writeheader()
            
            self.topic_writers[topic_name] = {
                'writer': writer,
                'file': file_handle,
                'headers': headers
            }
            print(f"Created CSV file: {filepath}")
        
        return self.topic_writers[topic_name]['writer']
    
    def update_headers(self, topic_name: str, flattened_data: Dict[str, Any]):
        """Update headers for a topic based on new data."""
        new_headers = set(flattened_data.keys())
        
        if topic_name not in self.topic_headers:
            self.topic_headers[topic_name] = set()
        
        base_headers = {'timestamp', 'ros_timestamp'}
        self.topic_headers[topic_name].update(base_headers)
        self.topic_headers[topic_name].update(new_headers)
    
    def list_topics(self):
        """List topics in the MCAP file."""
        print(f"Analyzing MCAP file: {self.mcap_file}")
        
        # Try to open with rosbag2_py using the file directly
        try:
            # Use the file path directly as URI
            storage_options = rosbag2_py.StorageOptions(
                uri=str(self.mcap_file), 
                storage_id='mcap'
            )
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )
            
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)
            
            topic_types = reader.get_all_topics_and_types()
            print(f"\n✓ Successfully opened MCAP file")
            print(f"Found {len(topic_types)} topics:")
            
            for topic_metadata in topic_types:
                print(f"  {topic_metadata.name}: {topic_metadata.type}")
            
            reader.close()
            return True
            
        except Exception as e:
            print(f"Error opening MCAP file: {e}")
            
            # Try with the directory instead
            try:
                print("\nTrying with directory path...")
                storage_options = rosbag2_py.StorageOptions(
                    uri=str(self.mcap_file.parent), 
                    storage_id='mcap'
                )
                
                reader = rosbag2_py.SequentialReader()
                reader.open(storage_options, converter_options)
                
                topic_types = reader.get_all_topics_and_types()
                print(f"✓ Successfully opened using directory path")
                print(f"Found {len(topic_types)} topics:")
                
                for topic_metadata in topic_types:
                    print(f"  {topic_metadata.name}: {topic_metadata.type}")
                
                reader.close()
                return True
                
            except Exception as e2:
                print(f"Directory approach also failed: {e2}")
                return False
    
    def convert(self):
        """Convert MCAP file to CSV files."""
        print(f"Converting MCAP file: {self.mcap_file}")
        self.create_output_directory()
        
        # Try file path first, then directory
        uri_options = [str(self.mcap_file), str(self.mcap_file.parent)]
        
        for uri in uri_options:
            try:
                print(f"Trying URI: {uri}")
                
                storage_options = rosbag2_py.StorageOptions(
                    uri=uri, 
                    storage_id='mcap'
                )
                converter_options = rosbag2_py.ConverterOptions(
                    input_serialization_format='cdr',
                    output_serialization_format='cdr'
                )
                
                reader = rosbag2_py.SequentialReader()
                reader.open(storage_options, converter_options)
                
                # Get topic types
                topic_types = reader.get_all_topics_and_types()
                type_map = {topic_metadata.name: topic_metadata.type for topic_metadata in topic_types}
                
                print(f"✓ Found {len(topic_types)} topics")
                
                # First pass: analyze structure
                message_count = 0
                topic_counts = {}
                
                print("First pass: analyzing message structure...")
                while reader.has_next():
                    (topic, data, timestamp) = reader.read_next()
                    
                    if topic in type_map:
                        try:
                            msg_type = get_message(type_map[topic])
                            msg = deserialize_message(data, msg_type)
                            
                            msg_dict = self.message_to_dict(msg)
                            flattened_data = self.flatten_message(msg_dict)
                            self.update_headers(topic, flattened_data)
                            
                            topic_counts[topic] = topic_counts.get(topic, 0) + 1
                            message_count += 1
                            
                            if message_count % 10000 == 0:
                                print(f"Analyzed {message_count} messages...")
                                
                        except Exception as e:
                            if message_count < 10:  # Only show first few errors
                                print(f"Warning: Could not analyze message for topic {topic}: {e}")
                
                reader.close()
                
                print(f"Found {len(topic_counts)} topics with {message_count} total messages:")
                for topic, count in topic_counts.items():
                    print(f"  {topic}: {count} messages")
                
                # Second pass: write data
                print("\nSecond pass: writing CSV files...")
                reader = rosbag2_py.SequentialReader()
                reader.open(storage_options, converter_options)
                
                processed_count = 0
                while reader.has_next():
                    (topic, data, timestamp) = reader.read_next()
                    
                    if topic in type_map and topic in self.topic_headers:
                        try:
                            msg_type = get_message(type_map[topic])
                            msg = deserialize_message(data, msg_type)
                            
                            msg_dict = self.message_to_dict(msg)
                            flattened_data = self.flatten_message(msg_dict)
                            
                            # Add timestamp information
                            timestamp_sec = timestamp / 1e9
                            flattened_data['timestamp'] = timestamp_sec
                            flattened_data['ros_timestamp'] = datetime.fromtimestamp(timestamp_sec).isoformat()
                            
                            # Write to CSV
                            headers = sorted(list(self.topic_headers[topic]))
                            writer = self.get_csv_writer(topic, headers)
                            
                            row_data = {}
                            for header in headers:
                                row_data[header] = flattened_data.get(header, "")
                            
                            writer.writerow(row_data)
                            
                            processed_count += 1
                            if processed_count % 10000 == 0:
                                print(f"Processed {processed_count}/{message_count} messages...")
                                
                        except Exception as e:
                            if processed_count < 10:  # Only show first few errors
                                print(f"Warning: Could not process message for topic {topic}: {e}")
                
                reader.close()
                
                # Close all file handles
                for topic_info in self.topic_writers.values():
                    topic_info['file'].close()
                
                print(f"\nConversion complete!")
                print(f"Processed {message_count} messages from {len(topic_counts)} topics")
                print(f"CSV files saved to: {self.output_dir}")
                
                return True
                
            except Exception as e:
                print(f"Failed with URI {uri}: {e}")
                continue
        
        print("All conversion attempts failed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert MCAP files to CSV using rosbag2_py directly")
    parser.add_argument("mcap_file", help="Path to the .mcap file")
    parser.add_argument("-o", "--output-dir", help="Output directory for CSV files")
    parser.add_argument("--list-topics", action="store_true", help="List topics without converting")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mcap_file):
        print(f"Error: File {args.mcap_file} does not exist")
        return 1
    
    converter = DirectMcapConverter(args.mcap_file, args.output_dir)
    
    if args.list_topics:
        if not converter.list_topics():
            return 1
    else:
        if not converter.convert():
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())