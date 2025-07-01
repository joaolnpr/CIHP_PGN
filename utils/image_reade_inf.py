import os
import numpy as np
import tensorflow as tf
import random

IGNORE_LABEL = 255
IMG_MEAN = np.array((125.0, 114.4, 107.9), dtype=np.float32)


def image_scaling(img):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.
    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """

    scale = tf.random.uniform([], minval=0.5, maxval=2.0, dtype=tf.float32)
    h_new = tf.cast(tf.cast(tf.shape(img)[0], tf.float32) * scale, tf.int32)
    w_new = tf.cast(tf.cast(tf.shape(img)[1], tf.float32) * scale, tf.int32)
    new_shape = tf.stack([h_new, w_new])
    img = tf.image.resize(img, new_shape)
    # label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0),
    #                                          new_shape)
    # label = tf.squeeze(label, squeeze_dims=[0])
    # edge = tf.image.resize_nearest_neighbor(tf.expand_dims(edge, 0), new_shape)
    # edge = tf.squeeze(edge, squeeze_dims=[0])

    return img


def image_mirroring(img):
    """
    Randomly mirrors the images.
    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    do_flip = tf.less(tf.random.uniform([]), 0.5)
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
    # label = tf.reverse(label, mirror)
    # edge = tf.reverse(edge, mirror)
    return img


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


def random_crop_and_pad_image_and_labels(image, crop_h, crop_w,
                                         ignore_label=255):
    """
    Randomly crop and pads the input images.
    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    # label = tf.cast(label, dtype=tf.float32)
    # label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.
    # edge = tf.cast(edge, dtype=tf.float32)
    # edge = edge - 0

    img_shape = tf.shape(image)
    img_pad = tf.image.pad_to_bounding_box(image, 0, 0, tf.maximum(crop_h, img_shape[0]), tf.maximum(crop_w, img_shape[1]))
    img_crop = tf.image.random_crop(img_pad, [crop_h, crop_w, 3])
    img_crop.set_shape((crop_h, crop_w, 3))
    return img_crop


def read_labeled_image_reverse_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    masks_rev = []
    for line in f:
        try:
            image, mask, mask_rev = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = mask_rev = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
        masks_rev.append(data_dir + mask_rev)
    return images, masks, masks_rev


def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks


def read_edge_list(data_dir, data_id_list):
    f = open(data_id_list, 'r')
    edges = []
    for line in f:
        edge = line.strip("\n")
        edges.append(data_dir + '/edges/' + edge + '.png')
    return edges


def parse_image(img_path, input_size, random_scale, random_mirror):
    img_contents = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(img, 3, axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    img -= IMG_MEAN
    if input_size is not None:
        h, w = input_size
        if random_scale:
            img = image_scaling(img)
        if random_mirror:
            img = image_mirroring(img)
        img = random_crop_and_pad_image_and_labels(img, h, w, IGNORE_LABEL)
    return img


def create_inference_dataset(image_list, input_size, random_scale, random_mirror, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(image_list)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_list))
    dataset = dataset.map(lambda img: parse_image(img, input_size, random_scale, random_mirror),
                         num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
