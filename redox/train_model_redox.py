import os
import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import nfp

from preprocessor import preprocessor

from loss import AtomInfMask, KLWithLogits, RedoxAttention
from model import build_embedding_model

def parse_example(example):
    parsed = tf.io.parse_single_example(example, features={
        **preprocessor.tfrecord_features,
        **{'redox': tf.io.FixedLenFeature([], dtype=tf.string)}})

    # All of the array preprocessor features are serialized integer arrays
    for key, val in preprocessor.tfrecord_features.items():
        if val.dtype == tf.string:
            parsed[key] = tf.io.parse_tensor(
                parsed[key], out_type=preprocessor.output_types[key])
    
    # Pop out the prediction target from the stored dictionary as a seperate dict
    parsed['redox'] = tf.io.parse_tensor(parsed['redox'], out_type=tf.float64)
    redox = parsed.pop('redox')        
    return parsed, redox


max_atoms = 80
max_bonds = 100
batch_size = 128

# Here, we have to add the prediction target padding onto the input padding
padded_shapes = (preprocessor.padded_shapes(max_atoms=None, max_bonds=None), [2])
padding_values = (preprocessor.padding_values, tf.constant(np.nan, dtype=tf.float64))

num_train = len(np.load('redox_split.npz', allow_pickle=True)['train'])

train_dataset = tf.data.TFRecordDataset('tfrecords_redox/train.tfrecord.gz', compression_type='GZIP')
train_new_dataset = tf.data.TFRecordDataset('tfrecords_redox/train_new.tfrecord.gz', compression_type='GZIP')

train_dataset = train_dataset.concatenate(train_new_dataset) \
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache().shuffle(buffer_size=num_train).repeat()\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = tf.data.TFRecordDataset('tfrecords_redox/valid.tfrecord.gz', compression_type='GZIP')\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache().shuffle(buffer_size=5000).repeat()\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)


input_tensors, atom_states, bond_states, global_states = build_embedding_model(
    preprocessor,
    dropout=0.0,
    atom_features=128,
    num_messages=6,
    num_heads=8,
    name='atom_embedding_model')

atom_class, bond_class, connectivity, n_atom = input_tensors
output = layers.Dense(2)(global_states[-1])

redox_model = tf.keras.Model(input_tensors, output)

if __name__ == "__main__":

    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(1E-4, 1, 1E-5)
    weight_decay  = tf.keras.optimizers.schedules.InverseTimeDecay(1E-6, 1, 1E-5)
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay) 
    
    redox_model.compile(
        loss=nfp.masked_mean_absolute_error,
        optimizer=optimizer)
    
    redox_model.summary()

    model_name = '20210602_redox_model_tempo'

    if not os.path.exists(model_name):
        os.makedirs(model_name)

    filepath = model_name + "/best_model"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, verbose=0)
    csv_logger = tf.keras.callbacks.CSVLogger(model_name + '/log.csv')

    redox_model.fit(train_dataset,
                    validation_data=valid_dataset,
                    steps_per_epoch=math.ceil(num_train/batch_size),
                    validation_steps=math.ceil(5000/batch_size),
                    epochs=500,
                    callbacks=[checkpoint, csv_logger],
                    verbose=1)
