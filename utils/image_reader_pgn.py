import os
import numpy as np
import tensorflow as tf
import random

IGNORE_LABEL = 255
IMG_MEAN = np.array((125.0, 114.4, 107.9), dtype=np.float32)

def image_scaling(img, label, edge):
    scale = tf.random.uniform([], minval=0.5, maxval=2.0, dtype=tf.float32)
    h_new = tf.cast(tf.cast(tf.shape(img)[0], tf.float32) * scale, tf.int32)
    w_new = tf.cast(tf.cast(tf.shape(img)[1], tf.float32) * scale, tf.int32)
    new_shape = tf.stack([h_new, w_new])
    img = tf.image.resize(img, new_shape)
    label = tf.image.resize(label, new_shape, method='nearest')
    edge = tf.image.resize(edge, new_shape, method='nearest')
    return img, label, edge

def image_mirroring(img, label, label_rev, edge):
    do_flip = tf.less(tf.random.uniform([]), 0.5)
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
    label = tf.cond(do_flip, lambda: tf.image.flip_left_right(label_rev), lambda: label)
    edge = tf.cond(do_flip, lambda: tf.image.flip_left_right(edge), lambda: edge)
    return img, label, edge

def random_resize_img_labels(image, label, resized_h, resized_w):
    scale = tf.random.uniform([], minval=0.75, maxval=1.25, dtype=tf.float32)
    h_new = tf.cast(resized_h * scale, tf.int32)
    w_new = tf.cast(resized_w * scale, tf.int32)
    new_shape = tf.stack([h_new, w_new])
    img = tf.image.resize(image, new_shape)
    label = tf.image.resize(label, new_shape, method='nearest')
    return img, label

def resize_img_labels(image, label, resized_h, resized_w):
    new_shape = tf.stack([tf.cast(resized_h, tf.int32), tf.cast(resized_w, tf.int32)])
    img = tf.image.resize(image, new_shape)
    label = tf.image.resize(label, new_shape, method='nearest')
    return img, label

def random_crop_and_pad_image_and_labels(image, label, edge, crop_h, crop_w, ignore_label=255):
    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label
    edge = tf.cast(edge, dtype=tf.float32)
    combined = tf.concat([image, label, edge], axis=2)
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.image.random_crop(combined_pad, [crop_h, crop_w, last_image_dim + last_label_dim + 1])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:last_image_dim+last_label_dim]
    edge_crop = combined_crop[:, :, last_image_dim+last_label_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    edge_crop = tf.cast(edge_crop, dtype=tf.uint8)
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))
    edge_crop.set_shape((crop_h, crop_w, 1))
    return img_crop, label_crop, edge_crop

def read_labeled_image_reverse_list(data_dir, data_list):
    f = open(data_list, 'r')
    images = []
    masks = []
    masks_rev = []
    for line in f:
        try:
            image, mask, mask_rev = line.strip("\n").split(' ')
        except ValueError:
            image = mask = mask_rev = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
        masks_rev.append(data_dir + mask_rev)
    return images, masks, masks_rev

def read_edge_list(data_dir, data_id_list):
    f = open(data_id_list, 'r')
    edges = []
    for line in f:
        edge = line.strip("\n")
        edges.append(data_dir + '/edges/' + edge + '.png')
    return edges

def parse_image(img_path, label_path, label_rev_path, edge_path, input_size, random_scale, random_mirror):
    img_contents = tf.io.read_file(img_path)
    label_contents = tf.io.read_file(label_path)
    label_contents_rev = tf.io.read_file(label_rev_path)
    edge_contents = tf.io.read_file(edge_path)
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(img, 3, axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    img -= IMG_MEAN
    label = tf.image.decode_png(label_contents, channels=1)
    label_rev = tf.image.decode_png(label_contents_rev, channels=1)
    edge = tf.image.decode_png(edge_contents, channels=1)
    if input_size is not None:
        h, w = input_size
        if random_mirror:
            img, label, edge = image_mirroring(img, label, label_rev, edge)
        if random_scale:
            img, label, edge = image_scaling(img, label, edge)
        img, label, edge = random_crop_and_pad_image_and_labels(img, label, edge, h, w, IGNORE_LABEL)
    return img, label, edge

def create_pgn_dataset(data_dir, data_list, data_id_list, input_size, random_scale, random_mirror, batch_size, shuffle=True):
    image_list, label_list, label_rev_list = read_labeled_image_reverse_list(data_dir, data_list)
    edge_list = read_edge_list(data_dir, data_id_list)
    dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list, label_rev_list, edge_list))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_list))
    dataset = dataset.map(lambda img, lbl, lbl_rev, edge: parse_image(img, lbl, lbl_rev, edge, input_size, random_scale, random_mirror),
                         num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
