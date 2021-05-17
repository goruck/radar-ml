"""
Train a deep neural network based classifier on radar data.

Example usage:
    $ python3 ./dnn.py \
        --datasets datasets/radar_samples_25Nov20.pickle datasets/radar_samples.pickle

Copyright (c) 2021 Lindo St. Angel
"""

import os
import collections
import pickle
import argparse
import logging
import sys
import functools
import itertools
import time
import datetime

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn import (model_selection, metrics, preprocessing, linear_model,
                     svm, utils, calibration)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import Model
from tensorflow.python.keras.utils.layer_utils import convert_dense_weights_data_format
#from IPython import display

import common

# Uncomment line below to print all elements of numpy arrays.
# np.set_printoptions(threshold=sys.maxsize)

# Uncomment line below to disable TF warnings.
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)

# Define a seed so random operations are the same from run to run.
RANDOM_SEED = 1234

# Create the models.


def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(1*1*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 256)

    model.add(layers.Reshape((1, 256, 1)))
    # Note: None is the batch size
    assert model.output_shape == (None, 1, 256, 1)

    # Input is a 4D tensor with shape: (batch_size, rows, cols, channels)
    model.add(layers.Conv2DTranspose(
        512, (5, 5), strides=(1, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 1, 512, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(
        256, (5, 5), strides=(1, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 1, 1024, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 1, 2048, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(1, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 1, 4096, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 2),
              padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 1, 8192, 1)

    return model

#generator = make_generator_model()
# generator.summary()

#noise = tf.random.normal([1, 100])
#generated_image = generator(noise, training=False)

#plt.imshow(generated_image[0, :, :, 0], cmap='gray')


def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(1, 4),
              padding='same', input_shape=[1, 8192, 1]))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    assert model.output_shape == (None, 1, 2048, 64)

    model.add(layers.Conv2D(128, (5, 5), strides=(1, 4), padding='same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    assert model.output_shape == (None, 1, 512, 128)

    model.add(layers.Conv2D(256, (5, 5), strides=(1, 4), padding='same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    assert model.output_shape == (None, 1, 128, 256)

    model.add(layers.Conv2D(512, (5, 5), strides=(1, 4), padding='same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    assert model.output_shape == (None, 1, 32, 512)

    model.add(layers.Conv2D(1024, (5, 5), strides=(1, 2), padding='same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    assert model.output_shape == (None, 1, 16, 1024)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    assert model.output_shape == (None, 1)

    return model

#discriminator = make_discriminator_model()
# discriminator.summary()
#decision = discriminator(generated_image)
# print(decision)

### Functional D/C model ####
# custom activation function


def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    return logexpsum / (logexpsum + 1.0)

def create_conv_layers(input_scan):
    input_shape = input_scan.shape[1:]

    conv = layers.Conv2D(32, (4, 4), strides=(
        2, 2), padding='same', input_shape=input_shape)(input_scan)
    conv = layers.BatchNormalization()(conv)
    conv = layers.LeakyReLU(alpha=0.2)(conv)
    #conv = layers.Dropout(0.2)(conv)

    conv = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.LeakyReLU(alpha=0.2)(conv)
    #conv = layers.Dropout(0.2)(conv)

    conv = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.LeakyReLU(alpha=0.2)(conv)
    #conv = layers.Dropout(0.2)(conv)

    conv = layers.GlobalMaxPooling2D()(conv)

    return conv

# define the standalone supervised and unsupervised discriminator models
# Input ordering is xz, yz, xy.
def define_discriminator_m(xz_shape=(80, 80, 1), yz_shape=(80, 80, 1), xy_shape=(80, 80, 1), n_classes=3):
    xz_input = layers.Input(shape=xz_shape)
    xz_model = create_conv_layers(xz_input)
    yz_input = layers.Input(shape=yz_shape)
    yz_model = create_conv_layers(yz_input)
    xy_input = layers.Input(shape=xy_shape)
    xy_model = create_conv_layers(xy_input)

    conv = layers.concatenate([yz_model, xz_model, xy_model])

    conv = layers.Dense(64)(conv)
    conv = layers.LeakyReLU(alpha=0.2)(conv)
    conv = layers.Dropout(0.3)(conv)

    conv = layers.Dense(64)(conv)
    conv = layers.LeakyReLU(alpha=0.2)(conv)
    conv = layers.Dropout(0.3)(conv)

    conv = layers.Dense(n_classes)(conv)

    c_out_layer = layers.Activation('softmax')(conv)

    c_model = Model(inputs=[xz_input, yz_input, xy_input], outputs=[c_out_layer])
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
        lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    # unsupervised output
    d_out_layer = layers.Lambda(custom_activation)(conv)
    # define and compile unsupervised discriminator model
    d_model = Model(inputs=[xz_input, yz_input, xy_input], outputs=[d_out_layer])
    d_model.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    return d_model, c_model

# define the standalone supervised and unsupervised discriminator models


def define_discriminator(in_shape=(1, 8192, 1), n_classes=3):
    # image input
    in_image = layers.Input(shape=in_shape)
    # downsample
    fe = layers.Conv2D(64, (5, 5), strides=(1, 4), padding='same')(in_image)
    fe = layers.LeakyReLU(alpha=0.3)(fe)
    #fe = layers.Dropout(0.2)(fe)
    # downsample
    fe = layers.Conv2D(128, (5, 5), strides=(1, 4), padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.3)(fe)
    #fe = layers.Dropout(0.2)(fe)
    # downsample
    fe = layers.Conv2D(256, (5, 5), strides=(1, 4), padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.3)(fe)
    #fe = layers.Dropout(0.2)(fe)
    # downsample
    fe = layers.Conv2D(512, (5, 5), strides=(1, 4), padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.3)(fe)
    #fe = layers.Dropout(0.2)(fe)
    # downsample
    fe = layers.Conv2D(1024, (5, 5), strides=(1, 2), padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.3)(fe)
    #fe = layers.Dropout(0.2)(fe)
    # flatten feature maps
    fe = layers.Flatten()(fe)
    # dropout
    #fe = layers.Dropout(0.4)(fe)
    # output layer nodes
    fe = layers.Dense(n_classes)(fe)
    # supervised output
    c_out_layer = layers.Activation('softmax')(fe)
    # define and compile supervised discriminator model
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
        lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    # unsupervised output
    d_out_layer = layers.Lambda(custom_activation)(fe)
    # define and compile unsupervised discriminator model
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
    return d_model, c_model


# Define the loss and optimizers.
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
cat_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Save checkpoints.
#checkpoint_dir = './training_checkpoints'
#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
# discriminator_optimizer=discriminator_optimizer,
# generator=generator,
# discriminator=discriminator)

# Define the training loop.
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Select metrics to measure the loss and the accuracy of the model.
gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
#disc_accy_metric = tf.keras.metrics.BinaryAccuracy(name='disc_accy')
disc_accy_metric = tf.keras.metrics.CategoricalAccuracy(name='disc_accy')
#test_accy_metric = tf.keras.metrics.BinaryAccuracy(name='test_accy')
test_accy_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accy')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
gen_log_dir = 'logs/' + current_time + '/gen'
disc_log_dir = 'logs/' + current_time + '/disc'
graph_log_dir = 'logs/' + current_time + '/graph'
gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)
graph_writer = tf.summary.create_file_writer(graph_log_dir)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# Ref: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch


@tf.function
def train_step(dataset):
    real_scans, real_labels = dataset

    half_dataset = int(len(real_scans) / 2)

    # Update supervised discriminator (c).
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        predictions = c_model(real_scans[:half_dataset], training=True)
        #discriminator_loss = loss_fn(labels, predictions)
        c_model_loss = cat_loss_fn(real_labels[:half_dataset], predictions)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    gradients = tape.gradient(c_model_loss, c_model.trainable_weights)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    discriminator_optimizer.apply_gradients(
        zip(gradients, c_model.trainable_weights))

    disc_loss_metric.update_state(c_model_loss)
    disc_accy_metric.update_state(real_labels[:half_dataset], predictions)

    # Update un-supervised discriminator (d).
    # Sample random points in the latent space.
    random_latent_vectors = tf.random.normal(
        shape=[len(real_scans) - half_dataset, noise_dim])
    # Decode them to fake radar scans.
    fake_scans = generator(random_latent_vectors)
    # Combine them with real radar scans.
    combined_scans = tf.concat([fake_scans, real_scans[half_dataset:]], axis=0)

    # Assemble labels, assigning label=0 to fake radar scans.
    combined_labels = tf.concat(
        [tf.zeros((len(real_scans) - half_dataset, 1)), tf.ones((len(real_scans) - half_dataset, 1))], axis=0
    )
    # Add random noise to the labels - important trick!
    #combined_labels += 0.05 * tf.random.uniform(combined_labels.shape)

    with tf.GradientTape() as tape:
        predictions = d_model(combined_scans, training=True)
        #discriminator_loss = loss_fn(labels, predictions)
        d_model_loss = loss_fn(combined_labels, predictions)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    gradients = tape.gradient(d_model_loss, d_model.trainable_weights)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    discriminator_optimizer.apply_gradients(
        zip(gradients, d_model.trainable_weights))

    # disc_loss_metric.update_state(c_model_loss)
    #disc_accy_metric.update_state(combined_labels, predictions)

    # Sample random points in the latent space.
    random_latent_vectors = tf.random.normal(
        shape=[len(real_scans), noise_dim])
    # Assemble labels that say "all real images".
    misleading_labels = tf.ones((len(real_scans), 1))

    # Train the generator (note that we should *not* update the weights
    # of the discriminator)!
    with tf.GradientTape() as tape:
        predictions = d_model(
            generator(random_latent_vectors, training=True), training=False)
        generator_loss = loss_fn(misleading_labels, predictions)
    gradients = tape.gradient(generator_loss, generator.trainable_weights)
    generator_optimizer.apply_gradients(
        zip(gradients, generator.trainable_weights))

    gen_loss_metric.update_state(generator_loss)


@tf.function
def train_step_c(dataset):
    real_scans, real_labels = dataset

    # Update supervised discriminator (c).
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        predictions = c_model(real_scans, training=True)
        #discriminator_loss = loss_fn(labels, predictions)
        c_model_loss = cat_loss_fn(real_labels, predictions)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    gradients = tape.gradient(c_model_loss, c_model.trainable_weights)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    discriminator_optimizer.apply_gradients(
        zip(gradients, c_model.trainable_weights))

    disc_loss_metric.update_state(c_model_loss)
    disc_accy_metric.update_state(real_labels, predictions)


@tf.function
def test_step(dataset):
    scans, labels = dataset
    predictions = c_model(scans, training=False)
    loss = cat_loss_fn(labels, predictions)
    test_loss_metric.update_state(loss)
    test_accy_metric.update_state(labels, predictions)


def train_old(train_dataset, val_dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for step, train_batch in enumerate(train_dataset):
            # train_step_c(train_batch)
            X, y = train_batch
            c_loss, c_acc = c_model.train_on_batch(X, y)
            #print(f'step: {step} c_loss: {c_loss}, c_acc: {c_acc}')

        print(f'epoch: {epoch} c_loss: {c_loss}, c_acc: {c_acc}')

        # with gen_summary_writer.as_default():
        #tf.summary.scalar('gen_loss', gen_loss_metric.result(), step=epoch)

        # with disc_summary_writer.as_default():
        #tf.summary.scalar('disc_loss', disc_loss_metric.result(), step=epoch)

        # Produce images for the GIF as we go
        # display.clear_output(wait=True)
        #generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        # if (epoch + 1) % 15 == 0:
        #checkpoint.save(file_prefix = checkpoint_prefix)

        # Run a validation loop at the end of each epoch.
        # for step, val_batch in enumerate(val_dataset):
        # test_step(val_batch)
        #X, y = val_batch
        #c_model.evaluate(X, y=y, batch_size=1)

        X, y = val_dataset
        c_model.evaluate(X, y=y, batch_size=32)

        """
    print(
        f'Epoch: {epoch + 1}, '
        f'Time: {time.time()- start}, '
        f'Generator Loss: {gen_loss_metric.result()}, '
        f'Discriminator Test Loss: {disc_loss_metric.result()}, '
        f'Discriminator Test Accy: {disc_accy_metric.result()}, '
        f'Discriminator Val Loss: {test_loss_metric.result()}, '
        f'Discriminator Val Accy: {test_accy_metric.result()}, '
    )

    # Reset metrics.
    gen_loss_metric.reset_states()
    disc_loss_metric.reset_states()
    disc_accy_metric.reset_states()
    test_accy_metric.reset_states()
    test_loss_metric.reset_states()
    """

    # Generate after the final epoch
    # display.clear_output(wait=True)
    #generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


def augment_data(x, rotation_range=1.0, zoom_range=0.3, noise_sd=1.0):
    rg = np.random.Generator(np.random.PCG64())

    def clamp(p):
        p[p > 1.0] = 1.0
        p[p < -1.0] = -1.0
        return p

    def rotate(p):
        """Rotate projection."""
        angle = np.random.uniform(-1*rotation_range, rotation_range)
        p = ndimage.rotate(p, angle, reshape=False)
        return clamp(p)

    def clipped_zoom(img, zoom_factor, **kwargs):
        """Generate zoomed versions of radar scans keeping array size constant.

        Note:
        https://stackoverflow.com/questions/37119071/
        """
        h, w = img.shape[:2]

        # For multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

        # Zooming out
        if zoom_factor < 1:

            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            # Zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left +
                zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

        # Zooming in
        elif zoom_factor > 1:

            # Bounding box of the zoomed-in region within the input array
            zh = int(np.ceil(h / zoom_factor))
            zw = int(np.ceil(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            out = ndimage.zoom(
                img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]

        # If zoom_factor == 1, just return the input array
        else:
            out = img

        return clamp(out)

    def add_noise(p, sd):
        """Add Gaussian noise."""
        p += rg.normal(scale=sd)
        return clamp(p)

    # Generate new tuple of rotated projections.
    # Rotates each projection independently.
    if rotation_range is not None:
        x = tuple(rotate(p) for p in x)

    # Generate new tuple of zoomed projections.
    # Use same zoom scale for all projections.
    if zoom_range is not None:
        zoom_factor = np.random.uniform(
            1.0 - zoom_range,
            1.0 + zoom_range
        )
        x = tuple(clipped_zoom(p, zoom_factor) for p in x)

    # Generate new tuple of projections with Gaussian noise.
    # Adds noise to each projection independently.
    if noise_sd is not None:
        x = tuple(add_noise(p, noise_sd) for p in x)

    return x


def prepare(data, labels, shuffle=False, augment=False, balance=False,
            proj_mask=[True, True, True], batch_size=32):
    """
    Prepare radar data for training and validation.
    """
    if augment:
        data = [augment_data(d) for d in data]

    #data = common.process_samples(data, proj_mask=proj_mask)

    if balance:
        data, labels = balance_classes(data, labels)

    # Gather up projections in each sample point.
    xz, yz, xy = [], [], []
    for d in data:
        xz.append(d[0])
        yz.append(d[1])
        xy.append(d[2])

    xz, yz, xy = np.array(xz), np.array(yz), np.array(xy)

    labels = tf.cast(labels, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(((xz, yz, xy), labels))

    if shuffle:
        dataset = dataset.shuffle(len(labels), reshuffle_each_iteration=True)

    # Resize images and add channel demension. 
    def dataset_map(d, l):
        xz, yz, xy = d
        xz = tf.image.resize(xz[..., tf.newaxis], [30, 200])
        yz = tf.image.resize(yz[..., tf.newaxis], [30, 200])
        xy = tf.image.resize(xy[..., tf.newaxis], [30, 200])
        return (xz, yz, xy), l
    dataset = dataset.map(dataset_map)

    dataset = dataset.batch(batch_size)

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def balance_classes(data, labels):
    """Balance classess."""
    # Most common classes and their counts from the most common to the least.
    c = collections.Counter(labels)
    mc = c.most_common()

    # Return if already balanced.
    if len(set([c for _, c in mc])) == 1:
        return labels, data

    print(f'Unbalanced most common: {mc}')
    print(f'Unbalanced label len: {len(labels)}')
    print(f'Unbalanced data len: {len(data)}')

    # Build a list of class indices from most common rankings.
    indices = [np.nonzero(labels == i)[0] for (i, _) in mc]
    # Use that list to build a list of label sets corresponding to each class.
    labels_list = [labels[i] for i in indices]
    # Use that list again to build a list of data sets corresponding to each class.
    data_list = [data[i] for i in indices]

    # Upsample data and label sets.
    _, majority_size = mc[0]

    def upsample(samples):
        return utils.resample(
            samples,
            replace=True,               # sample with replacement
            n_samples=majority_size,    # to match majority class
            random_state=RANDOM_SEED    # reproducible results
        )
    data_upsampled = [upsample(data) for data in data_list]
    labels_upsampled = [upsample(label) for label in labels_list]

    # Recombine the separate, and now upsampled, label and data sets.
    data_balanced = functools.reduce(
        lambda a, b: np.vstack((a, b)), data_upsampled
    )
    labels_balanced = functools.reduce(
        lambda a, b: np.concatenate((a, b)), labels_upsampled
    )

    c = collections.Counter(labels_balanced)
    mc = c.most_common()

    print(f'Balanced most common: {mc}')
    print(f'Balanced label len: {len(labels_balanced)}')
    print(f'Balanced data len: {len(data_balanced)}')

    return data_balanced, labels_balanced


if __name__ == '__main__':
    # Log file name.
    default_log_file = 'train-results/train.log'
    # Training datasets.
    default_datasets = ['datasets/radar_samples.pickle']
    # SVM confusion matrix name.
    default_svm_cm = 'train-results/svm_cm.png'
    # SVM model name.
    default_svm_model = 'train-results/svm_radar_classifier.pickle'
    # Label encoder name.
    default_label_encoder = 'train-results/radar_labels.pickle'
    # Radar 2-D projections to use for predictions (xy, xz, yz).
    default_proj_mask = [True, True, True]
    # Labels to use for training.
    default_desired_labels = ['person', 'dog', 'cat', 'pet']
    # Each epoch augments entire data set (zero disables).
    default_epochs = 0
    # Fraction of data set used for training, validation, testing.
    # Must sum to 1.0.
    default_train_val_test_frac = [0.8, 0.2, 0.0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', type=int,
        help='number of augementation epochs',
        default=default_epochs
    )
    parser.add_argument(
        '--datasets', nargs='+', type=str,
        help='paths to training datasets',
        default=default_datasets
    )
    parser.add_argument(
        '--desired_labels', nargs='+', type=str,
        help='labels to use for training',
        default=default_desired_labels
    )
    parser.add_argument(
        '--proj_mask', nargs='+', type=bool,
        help='projection mask (xy, xz, yz)',
        default=default_proj_mask
    )
    parser.add_argument(
        '--svm_cm', type=str,
        help='path of output svm confusion matrix',
        default=os.path.join(common.PRJ_DIR, default_svm_cm)
    )
    parser.add_argument(
        '--svm_model', type=str,
        help='path of output svm model name',
        default=os.path.join(common.PRJ_DIR, default_svm_model)
    )
    parser.add_argument(
        '--label_encoder', type=str,
        help='path of output label encoder',
        default=os.path.join(common.PRJ_DIR, default_label_encoder)
    )
    parser.add_argument(
        '--logging_level', type=str,
        help='logging level, "info" or "debug"',
        default='info'
    )
    parser.add_argument(
        '--online_learn', action='store_true',
        help='use dataset(s) for online learning (ignored if --use_svc'
    )
    parser.add_argument(
        '--use_svc', action='store_true',
        help='use svm.SVC instead of linear_model.SGDClassifier'
    )
    parser.add_argument(
        '--train_val_test_frac', nargs='+', type=float,
        help='train, val, test fraction of data set. must sum to 1.0',
        default=default_train_val_test_frac
    )
    parser.add_argument(
        '--log_file', type=str,
        help='path of output svm model name',
        default=os.path.join(common.PRJ_DIR, default_log_file)
    )
    parser.set_defaults(online_learn=False)
    parser.set_defaults(use_svc=False)
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        level=logging.DEBUG if args.logging_level == 'debug' else logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Combine multiple datasets if given.
    samples = []
    labels = []

    for dataset in args.datasets:
        logger.info(f'Opening dataset: {dataset}')
        try:
            with open(os.path.join(common.PRJ_DIR, dataset), 'rb') as fp:
                data_pickle = pickle.load(fp)
        except FileNotFoundError as e:
            logger.error(f'Dataset not found: {e}')
            exit(1)
        logger.debug(f'Found class labels: {set(data_pickle["labels"])}.')
        samples.extend(data_pickle['samples'])
        labels.extend(data_pickle['labels'])

    # Use alised class names.
    CLASS_ALIAS = {'polly': 'dog', 'rebel': 'cat'}
    #CLASS_ALIAS = {'polly': 'pet', 'rebel': 'pet', 'dog': 'pet', 'cat': 'pet'}
    keys = CLASS_ALIAS.keys()
    if set(labels) - set(keys):
        print('Using alised class names.')
        labels = list(
            map(
                lambda x: CLASS_ALIAS[x] if x in list(keys) else x,
                labels
            )
        )

    data = {'samples': samples, 'labels': labels}

    # Filter desired classes.
    logger.info('Maybe filtering classes.')
    desired = list(
        map(lambda x: 1 if x in args.desired_labels else 0, data['labels'])
    )
    # Samples are in the form [(xz, yz, xy), ...] in range [0, RADAR_MAX].
    samples = [s for i, s in enumerate(data['samples']) if desired[i]]

    # Scale each feature to the [-1, 1] range.
    logger.info('Scaling samples.')
    samples = [tuple(
        (p - common.RADAR_MAX / 2) / (common.RADAR_MAX / 2) for p in s
    ) for s in samples
    ]

    # Encode the labels.
    logger.info('Encoding labels.')
    le = preprocessing.LabelEncoder()
    desired_labels = [l for i, l in enumerate(data['labels']) if desired[i]]
    encoded_labels = le.fit_transform(desired_labels)
    class_names = list(le.classes_)

    # Data set summary.
    logger.info(
        f'Found {len(class_names)} classes and {len(desired_labels)} samples:')
    for i, c in enumerate(class_names):
        logger.info(
            f'...class: {i} "{c}" count: {np.count_nonzero(encoded_labels==i)}')

    # Split data and labels up into train, validation and test sets.
    logger.info(f'Splitting data set:')
    train_frac, val_frac, test_frac = args.train_val_test_frac
    X_train, X_val_test, y_train, y_val_test = model_selection.train_test_split(
        samples, encoded_labels, test_size=val_frac + test_frac,
        random_state=RANDOM_SEED, shuffle=True
    )
    val_split = int(len(X_val_test) * val_frac / (val_frac + test_frac))
    X_val, y_val = X_val_test[:val_split], y_val_test[:val_split]
    X_test, y_test = X_val_test[val_split:], y_val_test[val_split:]
    logger.info(f'...training samples: {len(X_train)}')
    logger.info(f'...validation samples: {len(X_val)}')
    logger.info(f'...test samples: {len(X_test)}')

    proj_mask = common.ProjMask(*args.proj_mask)
    logger.info(f'Projection mask: {proj_mask}')
    logger.info(f'Augment epochs: {args.epochs}')
    logger.info(f'Online learning: {args.online_learn}')

    #zf = 8192 / 10010
    #proj_zoom = common.ProjZoom([zf, zf], [zf, zf], [zf, zf])

    BATCH_SIZE = 64

    #y_train = tf.cast(y_train, dtype=tf.float32)
    #y_val = tf.cast(y_val, dtype=tf.float32)

    # Prepare datasets for training and validation.
    #train_dataset = prepare(X_train, y_train, shuffle=True,
                            #augment=False, batch_size=64, balance=False)
    # for d, l in train_dataset:
    #print(d, l)
    # exit()
    #val_dataset = prepare(X_val, y_val)
    # for d, l in val_dataset:
    #print(d, l)
    # exit()

    #train(train_dataset, val_dataset, EPOCHS)

    # Gather up projections in each sample point.
    xz, yz, xy = [], [], []
    for s in samples:
        xz.append(np.resize(s[0], (80, 80)))
        yz.append(np.resize(s[1], (80, 80)))
        xy.append(np.resize(s[2], (80, 80)))

    xz, yz, xy = np.array(xz), np.array(yz), np.array(xy)

    xz = xz[..., np.newaxis]
    yz = yz[..., np.newaxis]
    xy = xy[..., np.newaxis]

    print(xz.shape, yz.shape, xy.shape)

    # channel 0 = xz, ch 1 = yz, ch2 = xy
    X = np.concatenate((xz, yz, xy), axis=3)
    print(X.shape)

    y = np.array(encoded_labels)

    print(y.shape)

    rng = np.random.default_rng()
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    counter = collections.Counter(encoded_labels)
    max_v = float(max(counter.values()))
    class_weight = {cls: max_v / v for cls, v in counter.items()}
    logger.info(f'class weight: {class_weight}')
    num_classes = len(list(counter))
    logger.info(f'number of classes: {num_classes}')

    d_model, c_model = define_discriminator_m(n_classes=num_classes)
    d_model.summary()
    c_model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=10
    )

    c_model.fit(
        #train_dataset,
        x=[X[...,0], X[...,1], X[...,2]],
        y=y,
        batch_size=64,
        validation_split=0.2,
        epochs=100,
        #validation_data=val_dataset,
        class_weight=class_weight,
        callbacks=[early_stop]
    )

    # train model
    #train(g_model, d_model, c_model, gan_model, dataset)

    # c_model.evaluate(val_dataset)
