from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import scipy.io as sio
import cv2
import numpy as np
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from PIL import Image
from utils.pgn_keras import PGNKeras

N_CLASSES = 20

# Check for checkpoint in multiple locations
if os.path.exists('/home/paperspace/checkpoint/CIHP_pgn'):
    RESTORE_FROM = '/home/paperspace/checkpoint/CIHP_pgn'
elif os.path.exists('./checkpoint/CIHP_pgn'):
    RESTORE_FROM = './checkpoint/CIHP_pgn'
else:
    RESTORE_FROM = './checkpoint/CIHP_pgn'  # Default, will show error
    print(f"[WARNING] Checkpoint not found. Attempting quick download...")
    try:
        # Try to run quick download script
        import subprocess
        result = subprocess.run([sys.executable, 'quick_download.py'], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"[INFO] Quick download successful!")
            # Re-check checkpoint locations
            if os.path.exists('/home/paperspace/checkpoint/CIHP_pgn'):
                RESTORE_FROM = '/home/paperspace/checkpoint/CIHP_pgn'
            elif os.path.exists('./checkpoint/CIHP_pgn'):
                RESTORE_FROM = './checkpoint/CIHP_pgn'
        else:
            print(f"[WARNING] Quick download failed: {result.stderr}")
    except Exception as e:
        print(f"[WARNING] Could not run quick download: {e}")

def preprocess_image(image_path, target_size=(512, 512)):
    """Preprocess image for CIHP_PGN inference"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB and swap to BGR for model compatibility
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_b, img_g, img_r = img[:,:,2], img[:,:,1], img[:,:,0]
    img = np.stack([img_b, img_g, img_r], axis=2)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Convert to float and normalize
    img = img.astype(np.float32)
    img -= np.array([125.0, 114.4, 107.9])  # IMG_MEAN
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

try:
    # Load model
    model = PGNKeras(n_classes=N_CLASSES, checkpoint_path=RESTORE_FROM)

    # Check if running from API (single image mode)
    if os.path.exists('/home/paperspace/datasets/input.jpg'):
        # API mode - process single image
        input_image_path = '/home/paperspace/datasets/input.jpg'
        
        # Preprocess image
        image_batch = preprocess_image(input_image_path)
        
        # Run inference
        parsing_fc, parsing_rf_fc, edge_rf_fc = model(image_batch)
        
        # Process outputs
        parsing_out = np.argmax(parsing_fc, axis=-1)
        edge_out = 1.0 / (1.0 + np.exp(-edge_rf_fc))  # sigmoid
        parsing_np = parsing_out.astype(np.uint8)
        edge_np = (edge_out > 0.5).astype(np.uint8) * 255
        
        # Save outputs
        output_parsing = '/home/paperspace/output/input.png'
        output_edge = '/home/paperspace/output/input_edge.png'
        
        # Ensure output directory exists
        os.makedirs('/home/paperspace/output', exist_ok=True)
        
        cv2.imwrite(output_parsing, parsing_np[0])
        cv2.imwrite(output_edge, edge_np[0])
        
        print(f"✅ Saved parsing mask: {output_parsing}")
        print(f"✅ Saved edge mask: {output_edge}")
        print(f"✅ Human parsing completed successfully!")
        
    else:
        print("❌ No input image found at /home/paperspace/datasets/input.jpg")
        print("This script expects to be called from the API mode")
        sys.exit(1)

except Exception as e:
    print(f"Error during processing: {str(e)}")
    sys.exit(1)


##############################################################333
