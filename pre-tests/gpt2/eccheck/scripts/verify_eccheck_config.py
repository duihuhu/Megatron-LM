#!/usr/bin/env python3
"""
Quick verification script for EC-CHECK configuration file.
Checks format, structure, and values.
"""

import json
import sys
import os

def verify_config(config_path):
    """Verify EC-CHECK configuration file."""
    print(f"Verifying EC-CHECK configuration: {config_path}")
    print("=" * 60)
    
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}")
        return False
    
    # Check persist section
    if 'persist' not in config:
        print("WARNING: 'persist' section not found, using defaults")
    else:
        persist = config['persist']
        print(f"Persist settings:")
        print(f"  recv: {persist.get('recv', False)}")
        print(f"  parity: {persist.get('parity', False)}")
    
    # Check columns section
    if 'columns' not in config:
        print("ERROR: 'columns' section not found")
        return False
    
    columns = config['columns']
    if not isinstance(columns, list):
        print(f"ERROR: 'columns' must be a list, got {type(columns)}")
        return False
    
    num_columns = len(columns)
    print(f"\nColumns: {num_columns}")
    
    if num_columns == 0:
        print("ERROR: At least one column must be defined")
        return False
    
    if num_columns > 32:
        print(f"WARNING: {num_columns} columns exceeds MAX_COLUMNS (32)")
    
    # Verify each column
    all_valid = True
    for i, col in enumerate(columns):
        print(f"\nColumn {i}:")
        if not isinstance(col, dict):
            print(f"  ERROR: Column {i} is not a dictionary")
            all_valid = False
            continue
        
        # Check required fields
        required_fields = ['coefficient', 'send_peer', 'recv_peer']
        for field in required_fields:
            if field not in col:
                print(f"  ERROR: Missing required field '{field}'")
                all_valid = False
            else:
                value = col[field]
                if not isinstance(value, int):
                    print(f"  WARNING: '{field}' should be integer, got {type(value)}")
        
        if all(field in col for field in required_fields):
            print(f"  coefficient: {col['coefficient']}")
            send_peer = col['send_peer']
            recv_peer = col['recv_peer']
            send_info = f"{send_peer} (auto-calculate)" if send_peer == -1 else str(send_peer)
            recv_info = f"{recv_peer} (auto-calculate)" if recv_peer == -1 else str(recv_peer)
            print(f"  send_peer: {send_info}")
            print(f"  recv_peer: {recv_info}")
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ Configuration is valid!")
        print(f"✓ Ready for {num_columns}+{num_columns} EC-CHECK test")
        return True
    else:
        print("✗ Configuration has errors, please fix before testing")
        return False

if __name__ == '__main__':
    # Default config path
    default_path = "/workspace/Megatron-LM/pre-tests/gpt2/eccheck_2x2.json"
    
    # Use command line argument or default
    config_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    # Also check environment variable
    env_path = os.environ.get('ECCHECK_CONFIG_PATH')
    if env_path and os.path.exists(env_path):
        config_path = env_path
    
    success = verify_config(config_path)
    sys.exit(0 if success else 1)

