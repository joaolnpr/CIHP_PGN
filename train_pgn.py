import os
import time
import tensorflow as tf
import numpy as np
import random
from utils.utils import prepare_label
from utils.pgn_keras import PGNKeras
from utils.image_reader import create_dataset

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

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Prepare dataset
train_dataset = create_dataset(
    data_dir=DATA_DIR,
    data_list=LIST_PATH,
    data_id_list=DATA_ID_LIST,
    input_size=INPUT_SIZE,
    random_scale=RANDOM_SCALE,
    random_mirror=RANDOM_MIRROR,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE
)

# Model
model = PGNKeras(n_classes=N_CLASSES)
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)

@tf.function
def train_step(image_batch, label_batch, edge_batch):
    with tf.GradientTape() as tape:
        parsing_fc, parsing_rf_fc, edge_rf_fc = model(image_batch, training=True)
        label_proc = prepare_label(label_batch, tf.shape(parsing_fc)[1:3], one_hot=False)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=parsing_fc, labels=label_proc))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Training loop
step = 0
for epoch in range(NUM_STEPS // SAVE_PRED_EVERY + 1):
    for image_batch, label_batch, edge_batch in train_dataset:
        start_time = time.time()
        loss_value = train_step(image_batch, label_batch, edge_batch).numpy()
        duration = time.time() - start_time
        if step % SAVE_PRED_EVERY == 0:
            model.save_weights(os.path.join(SNAPSHOT_DIR, f'ckpt_{step}'))
        print(f'step {step} \t loss = {loss_value:.3f} ({duration:.3f} sec/step)')
        step += 1
        if step >= NUM_STEPS:
            break
    if step >= NUM_STEPS:
        break


##########################################
