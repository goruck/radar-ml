"""
radar-ml using GAN.

Copyright (c) 2020 Lindo St. Angel
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
#from IPython import display

import common

logger = logging.getLogger(__name__)

# Define a seed so random operations are the same from run to run.
RANDOM_SEED = 1234

# Create the models.
def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(1*1*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    #model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 256)

    model.add(layers.Reshape((1, 256, 1)))
    assert model.output_shape == (None, 1, 256, 1) # Note: None is the batch size

    # Input is a 4D tensor with shape: (batch_size, rows, cols, channels)
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 1, 512, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    #model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 1, 1024, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    #model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 1, 2048, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    #model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 1, 4096, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    #model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 1, 8192, 1)

    return model

generator = make_generator_model()
generator.summary()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

#plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(1, 4), padding='same', input_shape=[1, 8192, 1]))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    assert model.output_shape == (None, 1, 2048, 64)

    model.add(layers.Conv2D(128, (5, 5), strides=(1, 4), padding='same'))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    assert model.output_shape == (None, 1, 512, 128)

    model.add(layers.Conv2D(256, (5, 5), strides=(1, 4), padding='same'))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    assert model.output_shape == (None, 1, 128, 256)

    model.add(layers.Conv2D(512, (5, 5), strides=(1, 4), padding='same'))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    assert model.output_shape == (None, 1, 32, 512)

    model.add(layers.Conv2D(1024, (5, 5), strides=(1, 2), padding='same'))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    assert model.output_shape == (None, 1, 16, 1024)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    assert model.output_shape == (None, 1)

    return model

discriminator = make_discriminator_model()
discriminator.summary()
decision = discriminator(generated_image)
print(decision)

# Define the loss and optimizers.
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

# Save checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Define the training loop.
EPOCHS = 25
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Select metrics to measure the loss and the accuracy of the model.
gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
disc_accy_metric = tf.keras.metrics.BinaryAccuracy(name='disc_accy')
test_accy_metric = tf.keras.metrics.BinaryAccuracy(name='test_accy')

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
def train_step(real_images):
    # Sample random points in the latent space.
    random_latent_vectors = tf.random.normal(shape=[BATCH_SIZE, noise_dim])
    # Decode them to fake images.
    generated_images = generator(random_latent_vectors)
    # Combine them with real images.
    combined_images = tf.concat([generated_images, real_images], axis=0)

    # Assemble labels discriminating real (label=1) from fake (label=0) images.
    labels = tf.concat(
        [tf.zeros((BATCH_SIZE, 1)), tf.ones((real_images.shape[0], 1))], axis=0
    )
    # Add random noise to the labels - important trick!
    #labels += 0.05 * tf.random.uniform(labels.shape)

    # Train the discriminator.
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images, training=True)
        discriminator_loss = loss_fn(labels, predictions)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    gradients = tape.gradient(discriminator_loss, discriminator.trainable_weights)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_weights))

    disc_loss_metric.update_state(discriminator_loss)
    disc_accy_metric.update_state(labels, predictions)

    # Sample random points in the latent space.
    random_latent_vectors = tf.random.normal(shape=[BATCH_SIZE, noise_dim])
    # Assemble labels that say "all real images".
    misleading_labels = tf.ones((BATCH_SIZE, 1))

    # Train the generator (note that we should *not* update the weights
    # of the discriminator)!
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(random_latent_vectors, training=True), training=False)
        generator_loss = loss_fn(misleading_labels, predictions)
    gradients = tape.gradient(generator_loss, generator.trainable_weights)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_weights))

    gen_loss_metric.update_state(generator_loss)

    return discriminator_loss, generator_loss

@tf.function
def test_step(real_images):
    labels = tf.ones((real_images.shape[0], 1))
    predictions = discriminator(real_images, training=False)
    loss = loss_fn(labels, predictions)
    test_loss_metric.update_state(loss)
    test_accy_metric.update_state(labels, predictions)

def train(train_dataset, val_dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for step, image_batch in enumerate(train_dataset):
        d_loss, g_loss = train_step(image_batch)
        #print("discriminator loss at step %d: %.2f" % (step, d_loss))
        #print("adversarial loss at step %d: %.2f" % (step, g_loss))

    with gen_summary_writer.as_default():
        tf.summary.scalar('gen_loss', gen_loss_metric.result(), step=epoch)

    with disc_summary_writer.as_default():
        tf.summary.scalar('disc_loss', disc_loss_metric.result(), step=epoch)

    # Produce images for the GIF as we go
    #display.clear_output(wait=True)
    #generate_and_save_images(generator, epoch + 1, seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    # Run a validation loop at the end of each epoch.
    for image_batch in val_dataset:
        test_step(image_batch)

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

  # Generate after the final epoch
  #display.clear_output(wait=True)
  #generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()

def augment_data(x, rotation_range=10.0, zoom_range=0.5, noise_sd=1.0):
    rg = np.random.Generator(np.random.PCG64())

    def clamp(p):
        p[p>1.0] = 1.0
        p[p<-1.0] = -1.0
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
            out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

        # Zooming in
        elif zoom_factor > 1:

            # Bounding box of the zoomed-in region within the input array
            zh = int(np.ceil(h / zoom_factor))
            zw = int(np.ceil(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

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

def prepare(data, shuffle=False, augment=False,
    proj_mask = [True, True, True], batch_size=32):
    """
    Prepare radar data for training and validation.
    """
    if augment:
        data = [augment_data(x) for x in data]

    fv = common.process_samples(data, proj_mask=proj_mask)

    dataset = tf.data.Dataset.from_tensor_slices(fv)

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.map(lambda x: tf.image.resize(
        x[tf.newaxis, ..., tf.newaxis], [1, 8192])
    )

    dataset = dataset.batch(batch_size)

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

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
    default_desired_labels = ['person']
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
        level=logging.DEBUG if args.logging_level=='debug' else logging.INFO,
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
    logger.info(f'Found {len(class_names)} classes and {len(desired_labels)} samples:')
    for i, c in enumerate(class_names):
        logger.info(f'...class: {i} "{c}" count: {np.count_nonzero(encoded_labels==i)}')

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

    proj_mask=common.ProjMask(*args.proj_mask)
    logger.info(f'Projection mask: {proj_mask}')
    logger.info(f'Augment epochs: {args.epochs}')
    logger.info(f'Online learning: {args.online_learn}')

    #zf = 8192 / 10010
    #proj_zoom = common.ProjZoom([zf, zf], [zf, zf], [zf, zf])

    BATCH_SIZE = 64

    # Prepare datasets for training and validation.
    train_dataset = prepare(X_train, shuffle=True, augment=True, batch_size=BATCH_SIZE)
    val_dataset = prepare(X_val, shuffle=True, batch_size=BATCH_SIZE)

    #X_train_fv = common.process_samples(X_train, proj_mask=proj_mask)
    #print(X_train_fv)
    #for fv in X_train_fv:
        #print(f'min: {min(fv)} max: {max(fv)}, \nfv: {fv}')

    #X_val_fv = common.process_samples(X_val, proj_mask=proj_mask)

    # Batch size.
    #BATCH_SIZE = 64

    # Create dataset.
    #train_dataset = tf.data.Dataset.from_tensor_slices(X_train_fv)
    #val_dataset = tf.data.Dataset.from_tensor_slices(X_val_fv)

    # Reshape and resize to [None, 1, 8192, 1] (add Batch and Channels dimensions).
    #train_dataset = train_dataset.map(lambda x: tf.image.resize(x[tf.newaxis, ..., tf.newaxis], [1, 8192]))
    #val_dataset = val_dataset.map(lambda x: tf.image.resize(x[tf.newaxis, ..., tf.newaxis], [1, 8192]))

    # Batch.
    #train_dataset = train_dataset.batch(BATCH_SIZE)
    #val_dataset = val_dataset.batch(BATCH_SIZE)

    #for element in train_dataset:
        #print(element)

    #exit()

    train(train_dataset, val_dataset, EPOCHS)