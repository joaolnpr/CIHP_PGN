#!/usr/bin/env python3
"""
Single image inference script for CIHP_PGN human parsing
Usage: python single_image_inference.py --input path/to/image.jpg --output path/to/output.png
"""

import argparse
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.pgn_keras import PGNKeras

IMG_MEAN = np.array((125.0, 114.4, 107.9), dtype=np.float32)

def preprocess_image(img_path, target_size=(512, 512)):
    """
    Preprocess a single image for inference
    """
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, target_size)
        
        # Convert to float32 and normalize
        img = img.astype(np.float32)
        
        # Apply BGR mean subtraction (model was trained with BGR)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr -= IMG_MEAN
        
        # Add batch dimension
        img_batch = np.expand_dims(img_bgr, axis=0)
        
        return img_batch
        
    except Exception as e:
        print(f"[ERROR] Failed to preprocess image {img_path}: {str(e)}")
        return None

def run_inference(input_path, output_path, checkpoint_path='./checkpoint/CIHP_pgn'):
    """
    Run human parsing inference on a single image
    """
    try:
        print(f"[INFO] Processing image: {input_path}")
        
        # Check if input exists
        if not os.path.exists(input_path):
            print(f"[ERROR] Input image not found: {input_path}")
            return False
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            return False
        
        # Preprocess image
        img_batch = preprocess_image(input_path)
        if img_batch is None:
            return False
        
        # Load model
        print(f"[INFO] Loading model from: {checkpoint_path}")
        model = PGNKeras(n_classes=20, checkpoint_path=checkpoint_path)
        
        # Run inference
        print(f"[INFO] Running inference...")
        img_tensor = tf.constant(img_batch, dtype=tf.float32)
        parsing_fc, parsing_rf_fc, edge_rf_fc = model(img_tensor)
        
        # Get parsing result
        parsing_out = tf.argmax(parsing_fc, axis=-1)
        parsing_np = parsing_out.numpy().astype(np.uint8)[0]  # Remove batch dimension
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save result
        cv2.imwrite(output_path, parsing_np)
        print(f"[INFO] Parsing result saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='CIHP_PGN Single Image Inference')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', required=True, help='Output parsing mask path')
    parser.add_argument('--checkpoint', '-c', default='./checkpoint/CIHP_pgn', 
                       help='Path to model checkpoint (default: ./checkpoint/CIHP_pgn)')
    
    args = parser.parse_args()
    
    # Run inference
    success = run_inference(args.input, args.output, args.checkpoint)
    
    if success:
        print(f"[SUCCESS] Human parsing completed successfully")
        sys.exit(0)
    else:
        print(f"[FAILURE] Human parsing failed")
        sys.exit(1)

if __name__ == '__main__':
    main() 