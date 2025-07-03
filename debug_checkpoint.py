#!/usr/bin/env python3
"""
Debug script to see what's in the checkpoint directory
"""

import os
import glob

def debug_checkpoint_directory():
    """Debug what's in the checkpoint directory"""
    
    checkpoint_path = "/home/paperspace/checkpoint/CIHP_pgn"
    
    print("🔍 CIHP_PGN Checkpoint Debug")
    print("=" * 50)
    print(f"📁 Checking path: {checkpoint_path}")
    
    # Check if directory exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Directory does not exist: {checkpoint_path}")
        return
    
    if not os.path.isdir(checkpoint_path):
        print(f"❌ Path exists but is not a directory: {checkpoint_path}")
        return
    
    print(f"✅ Directory exists")
    
    # List all files
    print("\n📋 All files in directory:")
    try:
        for root, dirs, files in os.walk(checkpoint_path):
            level = root.replace(checkpoint_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"{subindent}{file} ({file_size} bytes)")
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return
    
    # Check for common checkpoint patterns
    print("\n🔍 Looking for checkpoint patterns:")
    
    patterns = [
        "checkpoint",
        "*.ckpt*",
        "*.pth",
        "*.pkl", 
        "*.pb",
        "model.ckpt*",
        "*.index",
        "*.data-*",
        "*.meta"
    ]
    
    for pattern in patterns:
        full_pattern = os.path.join(checkpoint_path, pattern)
        matches = glob.glob(full_pattern)
        if matches:
            print(f"  ✅ {pattern}: {len(matches)} files")
            for match in matches[:3]:  # Show first 3 matches
                print(f"    - {os.path.basename(match)}")
            if len(matches) > 3:
                print(f"    - ... and {len(matches) - 3} more")
        else:
            print(f"  ❌ {pattern}: not found")
    
    # Check checkpoint file content if it exists
    checkpoint_file = os.path.join(checkpoint_path, "checkpoint")
    if os.path.exists(checkpoint_file):
        print(f"\n📄 Checkpoint file content:")
        try:
            with open(checkpoint_file, 'r') as f:
                content = f.read()
                print(content)
        except Exception as e:
            print(f"❌ Error reading checkpoint file: {e}")
    
    # Try to determine checkpoint format
    print(f"\n🎯 Checkpoint format analysis:")
    
    # TensorFlow checkpoint
    tf_files = glob.glob(os.path.join(checkpoint_path, "*.index")) + \
               glob.glob(os.path.join(checkpoint_path, "*.data-*")) + \
               glob.glob(os.path.join(checkpoint_path, "*.meta"))
    
    if tf_files:
        print(f"  🔧 TensorFlow checkpoint format detected ({len(tf_files)} files)")
    
    # PyTorch checkpoint
    pth_files = glob.glob(os.path.join(checkpoint_path, "*.pth")) + \
                glob.glob(os.path.join(checkpoint_path, "*.pt"))
    
    if pth_files:
        print(f"  🔧 PyTorch checkpoint format detected ({len(pth_files)} files)")
    
    # Other formats
    other_files = glob.glob(os.path.join(checkpoint_path, "*.pkl")) + \
                  glob.glob(os.path.join(checkpoint_path, "*.npz")) + \
                  glob.glob(os.path.join(checkpoint_path, "*.h5"))
    
    if other_files:
        print(f"  🔧 Other format detected ({len(other_files)} files)")
    
    if not (tf_files or pth_files or other_files):
        print(f"  ❓ Unknown or no checkpoint format detected")

if __name__ == "__main__":
    debug_checkpoint_directory() 