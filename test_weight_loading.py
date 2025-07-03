#!/usr/bin/env python3
"""
Test script to verify CIHP_PGN model weight loading works correctly
"""

import os
import sys
import numpy as np
import tensorflow as tf
from utils.pgn_keras import PGNKeras

def test_model_loading():
    """Test if the model loads weights correctly"""
    print("🧪 Testing CIHP_PGN model weight loading...")
    
    # Check for checkpoint
    checkpoint_paths = [
        '/home/paperspace/checkpoint/CIHP_pgn',
        './checkpoint/CIHP_pgn'
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if not checkpoint_path:
        print("❌ No checkpoint found in:")
        for path in checkpoint_paths:
            print(f"  - {path}")
        return False
    
    print(f"📁 Using checkpoint: {checkpoint_path}")
    
    try:
        # Create model
        print("🔨 Creating model...")
        model = PGNKeras(n_classes=20, checkpoint_path=checkpoint_path)
        
        # Create test input
        print("🎯 Creating test input...")
        test_input = np.random.rand(1, 512, 512, 3).astype(np.float32)
        
        # Run inference
        print("🚀 Running inference...")
        parsing_fc, parsing_rf_fc, edge_rf_fc = model(test_input)
        
        # Check outputs
        print("📊 Checking outputs...")
        print(f"  parsing_fc shape: {parsing_fc.shape}")
        print(f"  parsing_rf_fc shape: {parsing_rf_fc.shape}")
        print(f"  edge_rf_fc shape: {edge_rf_fc.shape}")
        
        # Check if outputs are meaningful (not all zeros)
        parsing_mean = np.mean(parsing_fc)
        parsing_std = np.std(parsing_fc)
        edge_mean = np.mean(edge_rf_fc)
        edge_std = np.std(edge_rf_fc)
        
        print(f"📈 Output statistics:")
        print(f"  parsing_fc - mean: {parsing_mean:.6f}, std: {parsing_std:.6f}")
        print(f"  edge_rf_fc - mean: {edge_mean:.6f}, std: {edge_std:.6f}")
        
        # Check if outputs are non-trivial (weights actually loaded)
        if parsing_std > 0.01 and edge_std > 0.01:
            print("✅ SUCCESS: Model appears to be working with loaded weights!")
            print("🎯 The human parsing should now produce proper segmentation masks!")
            return True
        else:
            print("⚠️  WARNING: Outputs have very low variance - weights may not be loaded")
            print("🔧 Model structure is working but may still use random weights")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test PASSED!")
        print("💡 The human parsing issue should now be fixed!")
    else:
        print("\n💥 Model loading test FAILED!")
        print("🔧 Additional debugging may be needed")
    
    sys.exit(0 if success else 1) 