import os
import time
import tensorflow as tf
import numpy as np
import random
from utils.utils import prepare_label
from utils.model_pgn import PGNModel

# Set gpus
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

### parameters setting
DATA_DIR = './datasets/CIHP'
LIST_PATH = './datasets/CIHP/list/train_rev.txt'
DATA_ID_LIST = './datasets/CIHP/list/train_id.txt'
SNAPSHOT_DIR = './checkpoint/CIHP_pgn'
LOG_DIR = './logs/CIHP_pgn'

N_CLASSES = 20
INPUT_SIZE = (512, 512)
BATCH_SIZE = 1
SHUFFLE = True
RANDOM_SCALE = True
RANDOM_MIRROR = True
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
POWER = 0.9
p_Weight = 50
e_Weight = 0.005
Edge_Pos_W = 2
with open(DATA_ID_LIST, 'r') as f:
    TRAIN_SET = len(f.readlines())
SAVE_PRED_EVERY = TRAIN_SET // BATCH_SIZE + 1
NUM_STEPS = SAVE_PRED_EVERY * 100 + 1

IMG_MEAN = np.array((125.0, 114.4, 107.9), dtype=np.float32)

def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img_r, img_g, img_b = tf.split(img, 3, axis=2)
    img = tf.concat([img_b, img_g, img_r], 2)
    img -= IMG_MEAN
    img = tf.image.resize(img, INPUT_SIZE)
    return img

def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load and preprocess images
    image_list = [line.strip() for line in open(LIST_PATH)]
    image_paths = [os.path.join(DATA_DIR, p.split()[0]) for p in image_list]
    label_paths = [os.path.join(DATA_DIR, p.split()[1]) for p in image_list]
    images = [preprocess_image(p) for p in image_paths]
    labels = [tf.io.read_file(p) for p in label_paths]
    labels = [tf.image.decode_png(l, channels=1) for l in labels]
    labels = [tf.image.resize(l, INPUT_SIZE, method='nearest') for l in labels]
    images = tf.stack(images)
    labels = tf.stack(labels)

    # Model (must refactor PGNModel to be a tf.keras.Model subclass for best practice)
    model = PGNModel({'data': images}, is_training=True, n_classes=N_CLASSES, keep_prob=0.9)
    parsing_out1 = model.layers['parsing_fc']
    parsing_out2 = model.layers['parsing_rf_fc']
    edge_out = model.layers['edge_rf_fc']

    # Loss (example, you may need to adapt for your model)
    label_proc = prepare_label(labels, tf.shape(parsing_out1)[1:3], one_hot=False)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=parsing_out1, labels=label_proc))

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            logits = model({'data': images}, training=True)
            loss_value = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_proc))
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value

    # Training loop
    for step in range(NUM_STEPS):
        start_time = time.time()
        loss_value = train_step().numpy()
        duration = time.time() - start_time
        if step % SAVE_PRED_EVERY == 0:
            model.save_weights(os.path.join(SNAPSHOT_DIR, f'ckpt_{step}'))
        print(f'step {step} \t loss = {loss_value:.3f} ({duration:.3f} sec/step)')

if __name__ == '__main__':
    main()


##########################################
