#!/usr/bin/env python3
"""
Simple CIHP_PGN test script for single image inference
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.pgn_keras import PGNKeras
except ImportError as e:
    print(f"[ERROR] Failed to import PGNKeras: {e}")
    print("[INFO] Please ensure all dependencies are installed")
    sys.exit(1)

# Configuration
N_CLASSES = 20
INPUT_SIZE = (512, 512)
IMG_MEAN = np.array((125.0, 114.4, 107.9), dtype=np.float32)

def preprocess_image(image_path, target_size=INPUT_SIZE):
    """Preprocess image for inference"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        print(f"[INFO] Original image shape: {img.shape}")
        
        # Convert BGR to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        
        # Convert to float32 and apply mean subtraction (BGR order for mean)
        img = img.astype(np.float32)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr -= IMG_MEAN
        
        # Add batch dimension
        img_batch = np.expand_dims(img_bgr, axis=0)
        
        print(f"[INFO] Preprocessed image shape: {img_batch.shape}")
        return img_batch
        
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        return None

def run_inference(input_path, output_dir, checkpoint_path):
    """Run inference on a single image"""
    
    print("="*60)
    print("🤖 CIHP_PGN Simple Test")
    print("="*60)
    
    # Check input
    if not os.path.exists(input_path):
        print(f"[ERROR] Input image not found: {input_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Preprocess image
        print(f"[INFO] Preprocessing image: {input_path}")
        img_batch = preprocess_image(input_path)
        if img_batch is None:
            return False
        
        # Load model
        print(f"[INFO] Loading PGN model...")
        model = PGNKeras(n_classes=N_CLASSES, checkpoint_path=checkpoint_path)
        model.summary()
        
        # Run inference
        print(f"[INFO] Running inference...")
        img_tensor = tf.constant(img_batch, dtype=tf.float32)
        
        try:
            parsing_fc, parsing_rf_fc, edge_rf_fc = model(img_tensor)
            
            # Process outputs
            parsing_out = tf.argmax(parsing_fc, axis=-1)
            edge_out = tf.sigmoid(edge_rf_fc)
            
            # Convert to numpy
            parsing_np = parsing_out.numpy().astype(np.uint8)[0]  # Remove batch dim
            edge_np = (edge_out.numpy() > 0.5).astype(np.uint8)[0] * 255
            
            print(f"[INFO] Inference successful!")
            print(f"[INFO] Parsing output shape: {parsing_np.shape}")
            print(f"[INFO] Edge output shape: {edge_np.shape}")
            print(f"[INFO] Unique parsing values: {np.unique(parsing_np)}")
            
            # Save outputs
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            parsing_path = os.path.join(output_dir, f"{base_name}_parsing.png")
            edge_path = os.path.join(output_dir, f"{base_name}_edge.png")
            
            cv2.imwrite(parsing_path, parsing_np)
            cv2.imwrite(edge_path, edge_np.squeeze() if len(edge_np.shape) > 2 else edge_np)
            
            print(f"[SUCCESS] Results saved:")
            print(f"  - Parsing: {parsing_path}")
            print(f"  - Edge: {edge_path}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            print(f"[INFO] Model may be running with random weights")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple CIHP_PGN Test')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', default='./output', help='Output directory')
    parser.add_argument('--checkpoint', '-c', default='./checkpoint/CIHP_pgn', 
                       help='Checkpoint path')
    
    args = parser.parse_args()
    
    # Run test
    success = run_inference(args.input, args.output, args.checkpoint)
    
    if success:
        print("\n✅ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

if __name__ == '__main__':
    main() 