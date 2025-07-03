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
RESTORE_FROM = './checkpoint/CIHP_pgn'
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

    parsing_dir = './output/cihp_parsing_maps'
    os.makedirs(parsing_dir, exist_ok=True)
    edge_dir = './output/cihp_edge_maps'
    os.makedirs(edge_dir, exist_ok=True)

    # Get image paths for saving
    image_list = [line.strip() for line in open(LIST_PATH)]
    image_paths = [os.path.join(DATA_DIR, p.split()[0]) for p in image_list]

    for step, (image_batch, label_batch, edge_batch) in enumerate(val_dataset):
        parsing_fc, parsing_rf_fc, edge_rf_fc = model(image_batch)
        parsing_out = tf.argmax(parsing_fc, axis=-1)
        edge_out = tf.sigmoid(edge_rf_fc)
        parsing_np = parsing_out.numpy().astype(np.uint8)
        edge_np = (edge_out.numpy() > 0.5).astype(np.uint8) * 255
        for i in range(parsing_np.shape[0]):
            img_id = os.path.splitext(os.path.basename(image_paths[step * BATCH_SIZE + i]))[0]
            cv2.imwrite(f'{parsing_dir}/{img_id}.png', parsing_np[i])
            cv2.imwrite(f'{edge_dir}/{img_id}.png', edge_np[i])
            print(f"Saved: {img_id}")

except Exception as e:
    print(f"Error during processing: {str(e)}")
    sys.exit(1)


##############################################################333
