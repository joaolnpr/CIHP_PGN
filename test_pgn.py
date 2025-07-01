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

N_CLASSES = 20
DATA_DIR = './datasets/CIHP'
LIST_PATH = './datasets/CIHP/list/val.txt'
DATA_ID_LIST = './datasets/CIHP/list/val_id.txt'
with open(DATA_ID_LIST, 'r') as f:
    NUM_STEPS = len(f.readlines()) 
RESTORE_FROM = './checkpoint/CIHP_pgn'

IMG_MEAN = np.array((125.0, 114.4, 107.9), dtype=np.float32)

def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img_r, img_g, img_b = tf.split(img, 3, axis=2)
    img = tf.concat([img_b, img_g, img_r], 2)
    img -= IMG_MEAN
    return img

def main():
    # Load and preprocess images
    image_list = [line.strip() for line in open(LIST_PATH)]
    image_paths = [os.path.join(DATA_DIR, p.split()[0]) for p in image_list]
    label_paths = [os.path.join(DATA_DIR, p.split()[1]) for p in image_list]
    images = [preprocess_image(p) for p in image_paths]
    images = [tf.image.resize(img, [512, 512]) for img in images]
    images = tf.stack(images)
    images_rev = tf.reverse(images, axis=[2])
    image_batch = tf.stack([images, images_rev], axis=1)
    image_batch = tf.reshape(image_batch, [-1, 512, 512, 3])

    h_orig, w_orig = 512.0, 512.0
    def resize_batch(batch, scale):
        new_h = int(h_orig * scale)
        new_w = int(w_orig * scale)
        return tf.image.resize(batch, [new_h, new_w])
    image_batch050 = resize_batch(image_batch, 0.50)
    image_batch075 = resize_batch(image_batch, 0.75)
    image_batch125 = resize_batch(image_batch, 1.25)
    image_batch150 = resize_batch(image_batch, 1.50)
    image_batch175 = resize_batch(image_batch, 1.75)

    # Model (must refactor PGNModel to be a tf.keras.Model subclass for best practice)
    net_100 = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    net_050 = PGNModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)
    net_075 = PGNModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    net_125 = PGNModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
    net_150 = PGNModel({'data': image_batch150}, is_training=False, n_classes=N_CLASSES)
    net_175 = PGNModel({'data': image_batch175}, is_training=False, n_classes=N_CLASSES)

    parsing_out1_050 = net_050.layers['parsing_fc']
    parsing_out1_075 = net_075.layers['parsing_fc']
    parsing_out1_100 = net_100.layers['parsing_fc']
    parsing_out1_125 = net_125.layers['parsing_fc']
    parsing_out1_150 = net_150.layers['parsing_fc']
    parsing_out1_175 = net_175.layers['parsing_fc']

    parsing_out2_050 = net_050.layers['parsing_rf_fc']
    parsing_out2_075 = net_075.layers['parsing_rf_fc']
    parsing_out2_100 = net_100.layers['parsing_rf_fc']
    parsing_out2_125 = net_125.layers['parsing_rf_fc']
    parsing_out2_150 = net_150.layers['parsing_rf_fc']
    parsing_out2_175 = net_175.layers['parsing_rf_fc']

    edge_out2_100 = net_100.layers['edge_rf_fc']
    edge_out2_125 = net_125.layers['edge_rf_fc']
    edge_out2_150 = net_150.layers['edge_rf_fc']
    edge_out2_175 = net_175.layers['edge_rf_fc']

    parsing_out1 = tf.reduce_mean(tf.stack([
        tf.image.resize(parsing_out1_050, [512, 512]),
        tf.image.resize(parsing_out1_075, [512, 512]),
        tf.image.resize(parsing_out1_100, [512, 512]),
        tf.image.resize(parsing_out1_125, [512, 512]),
        tf.image.resize(parsing_out1_150, [512, 512]),
        tf.image.resize(parsing_out1_175, [512, 512])
    ]), axis=0)

    parsing_out2 = tf.reduce_mean(tf.stack([
        tf.image.resize(parsing_out2_050, [512, 512]),
        tf.image.resize(parsing_out2_075, [512, 512]),
        tf.image.resize(parsing_out2_100, [512, 512]),
        tf.image.resize(parsing_out2_125, [512, 512]),
        tf.image.resize(parsing_out2_150, [512, 512]),
        tf.image.resize(parsing_out2_175, [512, 512])
    ]), axis=0)

    edge_out2_100 = tf.image.resize(edge_out2_100, [512, 512])
    edge_out2_125 = tf.image.resize(edge_out2_125, [512, 512])
    edge_out2_150 = tf.image.resize(edge_out2_150, [512, 512])
    edge_out2_175 = tf.image.resize(edge_out2_175, [512, 512])
    edge_out2 = tf.reduce_mean(tf.stack([edge_out2_100, edge_out2_125, edge_out2_150, edge_out2_175]), axis=0)

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, axis=[1])

    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, axis=0)
    pred_scores = tf.reduce_max(raw_output_all, axis=3)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(raw_output_all, axis=3) # Create 4-d tensor.

    raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
    tail_output_rev = tf.reverse(tail_output, axis=[1])
    raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_edge_all = tf.expand_dims(raw_edge_all, axis=0)
    pred_edge = tf.sigmoid(raw_edge_all)
    res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)

    # TODO: Update metrics to use tf.keras.metrics or tf.metrics in TF2.x
    # For now, skip metrics and focus on inference output

    parsing_dir = './output/cihp_parsing_maps'
    os.makedirs(parsing_dir, exist_ok=True)
    edge_dir = './output/cihp_edge_maps'
    os.makedirs(edge_dir, exist_ok=True)
    for step in range(NUM_STEPS):
        print(step)
        # TODO: Replace with model inference call
        img_id = os.path.splitext(os.path.basename(image_paths[step]))[0]
        parsing_im = np.zeros((512, 512), dtype=np.uint8)
        cv2.imwrite(f'{parsing_dir}/{img_id}.png', parsing_im)
        cv2.imwrite(f'{edge_dir}/{img_id}.png', parsing_im)
        print("here")

if __name__ == '__main__':
    main()


##############################################################333
