from __future__ import print_function
import argparse
import os
import sys
import cv2
from glob import glob
import numpy as np
import tensorflow as tf
from utils.image_reade_inf import create_inference_dataset
from utils.pgn_keras import PGNKeras

argp = argparse.ArgumentParser(description="Inference pipeline")
argp.add_argument('-i', '--directory', type=str, help='Path of the input dir', default='./datasets/images')
argp.add_argument('-o', '--output', type=str, help='Path of the output dir', default='./datasets/output')
argp.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
args = argp.parse_args()

image_list_inp = [i for i in glob(os.path.join(args.directory, '**'), recursive=True) if os.path.isfile(i)]
image_list_inp = image_list_inp[:5]
N_CLASSES = 20
RESTORE_FROM = './checkpoint/CIHP_pgn'

# Create dataset
dataset = create_inference_dataset(image_list_inp, input_size=(512, 512), random_scale=False, random_mirror=False, batch_size=args.batch_size, shuffle=False)

# Load model
model = PGNKeras(n_classes=N_CLASSES, checkpoint_path=RESTORE_FROM)

parsing_dir = os.path.join(args.output, 'cihp_parsing_maps')
os.makedirs(parsing_dir, exist_ok=True)
edge_dir = os.path.join(args.output, 'cihp_edge_maps')
os.makedirs(edge_dir, exist_ok=True)

for step, batch in enumerate(dataset):
    parsing_fc, parsing_rf_fc, edge_rf_fc = model(batch)
    parsing_out = tf.argmax(parsing_fc, axis=-1)
    edge_out = tf.sigmoid(edge_rf_fc)
    parsing_np = parsing_out.numpy().astype(np.uint8)
    edge_np = (edge_out.numpy() > 0.5).astype(np.uint8) * 255
    for i in range(parsing_np.shape[0]):
        img_id = os.path.splitext(os.path.basename(image_list_inp[step * args.batch_size + i]))[0]
        cv2.imwrite(f'{parsing_dir}/{img_id}.png', parsing_np[i])
        cv2.imwrite(f'{edge_dir}/{img_id}.png', edge_np[i])
        print(f"Saved: {img_id}")


##############################################################333
