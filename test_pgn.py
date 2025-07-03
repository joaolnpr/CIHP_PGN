from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import scipy.io as sio
import cv2
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
from PIL import Image
from utils import *
from utils.image_reader import create_dataset
from utils.pgn_keras import PGNKeras

N_CLASSES = 20
DATA_DIR = './datasets/CIHP'
LIST_PATH = './datasets/CIHP/list/val.txt'
DATA_ID_LIST = './datasets/CIHP/list/val_id.txt'

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

BATCH_SIZE = 1

# Check if required files exist
if not os.path.exists(DATA_ID_LIST):
    print(f"Error: Required file {DATA_ID_LIST} not found")
    sys.exit(1)

if not os.path.exists(LIST_PATH):
    print(f"Error: Required file {LIST_PATH} not found")
    sys.exit(1)

with open(DATA_ID_LIST, 'r') as f:
    NUM_STEPS = len(f.readlines())

try:
    # Prepare dataset
    val_dataset = create_dataset(
        data_dir=DATA_DIR,
        data_list=LIST_PATH,
        data_id_list=DATA_ID_LIST,
        input_size=(512, 512),
        random_scale=False,
        random_mirror=False,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Load model
    model = PGNKeras(n_classes=N_CLASSES, checkpoint_path=RESTORE_FROM)

    # Check if running from API (expects output in parent directory)
    if os.path.exists('/home/paperspace/output'):
        # API mode - save to expected location
        parsing_dir = '/home/paperspace/output'
        edge_dir = '/home/paperspace/output'
    else:
        # Standalone mode - use local output
        parsing_dir = './output/cihp_parsing_maps'
        edge_dir = './output/cihp_edge_maps'
        os.makedirs(parsing_dir, exist_ok=True)
        os.makedirs(edge_dir, exist_ok=True)

    # Get image paths for saving
    image_list = [line.strip() for line in open(LIST_PATH)]
    image_paths = [os.path.join(DATA_DIR, p.split()[0]) for p in image_list]

    # Create TF1.x iterator for the dataset
    iterator = tf.compat.v1.data.make_one_shot_iterator(val_dataset)
    next_element = iterator.get_next()
    
    # Process each image
    step = 0
    try:
        while step < NUM_STEPS:
            # Get the next batch from dataset
            image_batch, label_batch, edge_batch = model.sess.run(next_element)
            
            # Run inference
            parsing_fc, parsing_rf_fc, edge_rf_fc = model(image_batch)
            
            # Process outputs
            parsing_out = np.argmax(parsing_fc, axis=-1)
            edge_out = 1.0 / (1.0 + np.exp(-edge_rf_fc))  # sigmoid
            parsing_np = parsing_out.astype(np.uint8)
            edge_np = (edge_out > 0.5).astype(np.uint8) * 255
            
            for i in range(parsing_np.shape[0]):
                img_id = os.path.splitext(os.path.basename(image_paths[step * BATCH_SIZE + i]))[0]
                
                # Check if running from API (expects specific filename)
                if os.path.exists('/home/paperspace/output'):
                    # API mode - use expected filename
                    parsing_filename = 'input.png'
                    edge_filename = 'input_edge.png'
                else:
                    # Standalone mode - use image ID
                    parsing_filename = f'{img_id}.png'
                    edge_filename = f'{img_id}.png'
                
                cv2.imwrite(f'{parsing_dir}/{parsing_filename}', parsing_np[i])
                cv2.imwrite(f'{edge_dir}/{edge_filename}', edge_np[i])
                print(f"Saved: {img_id} -> {parsing_filename}")
            
            step += 1
            
    except tf.errors.OutOfRangeError:
        print("Finished processing all images in dataset")

except Exception as e:
    print(f"Error during processing: {str(e)}")
    sys.exit(1)


##############################################################333
