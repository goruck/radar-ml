"""
Train a Semi-Supervised GAN on radar data.

Example usage:
    $ python3 ./sgan.py \
        --datasets datasets/radar_samples_25Nov20.pickle datasets/radar_samples.pickle \
        --datasets_as_sup datasets/radar_samples_25Nov20.pickle

Note:
    Partially based on "Generative Adversarial Networks with Python" by Jason Brownlee.
    See https://machinelearningmastery.com/generative_adversarial_networks/.

Copyright (c) 2021 Lindo St. Angel
"""

import os
import collections
import pickle
import argparse
import logging
import sys
import functools
import pickle

import numpy as np
from scipy import ndimage
from sklearn import preprocessing, utils
import tensorflow as tf
from PIL import Image

import common

logger = logging.getLogger(__name__)

RANDOM_SEED = 1234
rng = np.random.default_rng(RANDOM_SEED)

# Radar projection rescaling factor.
RESCALE = (128, 128)

# Original shapes (cols, rows) of radar projections.
# todo: change so that the code determines this from the data set.
XZ_SIZE = (176, 22)
YZ_SIZE = (176, 31)
XY_SIZE = (31, 22)

# Class aliases.
# Some data sets used pet names instead of pet type so this makes them consistent.
CLASS_ALIAS = {'polly': 'dog', 'rebel': 'cat'}
# Make dogs and cats as pets.
#CLASS_ALIAS = {'polly': 'pet', 'rebel': 'pet', 'dog': 'pet', 'cat': 'pet'}

# Uncomment line below to print all elements of numpy arrays.
# np.set_printoptions(threshold=sys.maxsize)


def create_g_conv_layers(input, init):
    """Creates generator convolutional layers."""
    n_nodes = 8 * 8 * 128
    conv = tf.keras.layers.Dense(n_nodes, kernel_initializer=init)(input)
    conv = tf.keras.layers.ReLU()(conv)
    conv = tf.keras.layers.Reshape((8, 8, 128))(conv)

    # Upsample to 16x16.
    conv = tf.keras.layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.ReLU()(conv)

    # Upsample to 32x32.
    conv = tf.keras.layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.ReLU()(conv)

    # Upsample to 64x64.
    conv = tf.keras.layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.ReLU()(conv)

    # Upsample to 128x128.
    conv = tf.keras.layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.ReLU()(conv)

    # Single filter conv with tanh activation to make data fall in [-1, 1]
    conv = tf.keras.layers.Conv2D(1, (7, 7), activation='tanh', padding='same',
                                  kernel_initializer=init)(conv)

    return conv


def define_generator(latent_dim=100):
    """Define the standalone generator model.

    Args:
        latent_dim (int): Latent space dimension,
        e.g. a 100-element vector of Gaussian random numbers.

    Returns:
        model (Keras object): Generator model.

    Note:
        Model output is a list of xz, yz, xy generated radar projections
        each with shape (n_batch, 128, 128, 1) with values in [-1,1]
    """
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    input = tf.keras.layers.Input(shape=(latent_dim,))

    # Create convolutional layers per projection.
    xz_conv = create_g_conv_layers(input, init)
    yz_conv = create_g_conv_layers(input, init)
    xy_conv = create_g_conv_layers(input, init)

    # Define model.
    out = [xz_conv, yz_conv, xy_conv]
    model = tf.keras.Model(inputs=input, outputs=out, name='generator')

    return model


def custom_activation(output):
    """Custom activation function for discriminator."""
    logexpsum = tf.keras.backend.sum(
        tf.keras.backend.exp(output), axis=-1, keepdims=True)
    return logexpsum / (logexpsum + 1.0)


def create_d_conv_layers(input, init):
    """Creates discriminator conv layers."""
    input_shape = input.shape[1:]

    # Downsample to 64x64.
    conv = tf.keras.layers.Conv2D(128, (3, 3), strides=(
        2, 2), padding='same', input_shape=input_shape, kernel_initializer=init)(input)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)

    # Downsample to 32x32.
    conv = tf.keras.layers.Conv2D(64, (3, 3), strides=(
        2, 2), padding='same', kernel_initializer=init)(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)

    # Downsample to 16x16.
    conv = tf.keras.layers.Conv2D(32, (3, 3), strides=(
        2, 2), padding='same', kernel_initializer=init)(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)

    return conv


def define_discriminator(xz_shape, yz_shape, xy_shape, n_classes):
    """Define supervised and unsupervised discriminator models.

    Args:
        xz_shape, yz_shape, xy_shape (tuple): Shapes of radar projections.
        n_classes (int): Number of classes for supervised disc model.

    Returns:
        d_model (Keras object): Unsupervised discriminator model.
        c_model (Keras object): Supervised discriminator model.

    Note:
        Input ordering is xz, yz, xy.
    """
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    # Create conv layers for each radar projection.
    xz_input = tf.keras.layers.Input(shape=xz_shape)
    xz_model = create_d_conv_layers(xz_input, init)
    yz_input = tf.keras.layers.Input(shape=yz_shape)
    yz_model = create_d_conv_layers(yz_input, init)
    xy_input = tf.keras.layers.Input(shape=xy_shape)
    xy_model = create_d_conv_layers(xy_input, init)

    # Concat convolutions.
    conv = tf.keras.layers.concatenate([xz_model, yz_model, xy_model])

    # Flatten to get feature vector.
    fv = tf.keras.layers.Flatten()(conv)

    # Pass feature vector to dense layers.
    dense = tf.keras.layers.Dense(64, kernel_initializer=init)(fv)
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.LeakyReLU(alpha=0.2)(dense)
    dense = tf.keras.layers.Dropout(0.5)(dense)

    dense = tf.keras.layers.Dense(64, kernel_initializer=init)(dense)
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.LeakyReLU(alpha=0.2)(dense)
    dense = tf.keras.layers.Dropout(0.5)(dense)

    # Classifier.
    cls = tf.keras.layers.Dense(n_classes, kernel_initializer=init)(dense)

    # Supervised output.
    c_out_layer = tf.keras.layers.Activation('softmax')(cls)
    # Define and compile supervised discriminator model.
    c_model = tf.keras.Model(inputs=[xz_input, yz_input, xy_input],
                             outputs=[c_out_layer], name='classifier')
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
        lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    # Unsupervised output.
    d_out_layer = tf.keras.layers.Lambda(custom_activation)(cls)
    # Define and compile unsupervised discriminator model.
    d_model = tf.keras.Model(inputs=[xz_input, yz_input, xy_input], outputs=[
        d_out_layer], name='discriminator')
    d_model.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    return d_model, c_model


def define_gan(g_model, d_model):
    """Define combined generator and discriminator model for updating the generator."""
    # Make weights in the discriminator not trainable.
    for layer in d_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    # Connect image output from generator as input to discriminator.
    gan_output = d_model(g_model.output)
    # Define gan model as taking noise and outputting a classification.
    model = tf.keras.Model(inputs=g_model.input,
                           outputs=gan_output, name='gan')
    # Compile model.
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model


def augment_data(x, rotation_range=1.0, zoom_range=0.3, noise_sd=1.0):
    """Augment a tuple of radar projections."""
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
        p += rng.normal(scale=sd)
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


def balance_classes(data, labels, samples_sup, shuffle=True):
    """Balance classess."""
    # Most common classes and their counts from the most common to the least.
    c = collections.Counter(labels)
    mc = c.most_common()

    # Return if already balanced.
    if len(set([c for _, c in mc])) == 1:
        return data, labels, samples_sup

    logger.debug(f'Unbalanced most common: {mc}')
    logger.debug(f'Unbalanced label len: {len(labels)}')
    logger.debug(f'Unbalanced data len: {len(data)}')
    logger.debug(f'Unbalanced samples_sup len: {len(samples_sup)}')

    # Build a list of class indices from most common rankings.
    indices = [np.nonzero(labels == i)[0] for (i, _) in mc]
    # Use that list to build a list of label sets corresponding to each class.
    labels_list = [labels[i] for i in indices]
    # Use that list again to build a list of data sets corresponding to each class.
    data_list = [data[i] for i in indices]
    # Use that list again to build a list of samples_sup sets corresponding to each class.
    samples_sup_list = [samples_sup[i] for i in indices]

    # Upsample.
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
    samples_sup_upsampled = [upsample(samples_sup)
                             for samples_sup in samples_sup_list]

    # Recombine the separate, and now upsampled, label and data sets.
    data_balanced = functools.reduce(
        lambda a, b: np.vstack((a, b)), data_upsampled
    )
    labels_balanced = functools.reduce(
        lambda a, b: np.concatenate((a, b)), labels_upsampled
    )
    samples_sup_balanced = functools.reduce(
        lambda a, b: np.concatenate((a, b)), samples_sup_upsampled
    )

    if shuffle:
        idx = np.arange(labels_balanced.size)
        rng.shuffle(idx)
        data_balanced, labels_balanced, samples_sup_balanced = data_balanced[
            idx], labels_balanced[idx], samples_sup_balanced[idx]

    c = collections.Counter(labels_balanced)
    mc = c.most_common()

    logger.debug(f'Balanced most common: {mc}')
    logger.debug(f'Balanced label len: {len(labels_balanced)}')
    logger.debug(f'Balanced data len: {len(data_balanced)}')
    logger.debug(f'Balanced samples_sup len: {len(samples_sup_balanced)}')

    return data_balanced, labels_balanced, samples_sup_balanced


def smooth_positive_labels(y):
    """Smooths class=1 to [0.7, 1.2]"""
    return y - 0.3 + (np.random.random(y.shape) * 0.5)


def smooth_negative_labels(y):
    """Smooths class=0 to [0.0, 0.3]"""
    return y + np.random.random(y.shape) * 0.3


def select_supervised_samples(dataset, n_samples=150, n_classes=3):
    """Select a supervised subset of the dataset, ensures classes are balanced."""
    X, y, sup = dataset
    X_list, y_list = [], []
    n_per_class = int(n_samples / n_classes)

    for i in range(n_classes):
        # get all images for this class IF sup is True
        X_with_class = X[(y == i) & sup]
        # choose random instances
        ix = np.random.randint(0, X_with_class.shape[0], n_per_class)
        # Ensure enough samples in each class.
        assert len(ix) == n_per_class, f'Not enough class {i} sup samples'
        # add to list
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return np.asarray(X_list), np.asarray(y_list)


def generate_real_samples(dataset, n_samples):
    """Select real smaples."""
    # split into images and labels
    images, labels, *_ = dataset
    # choose random instances
    ix = np.random.randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    y = smooth_positive_labels(y)
    return [X, labels], y


def generate_latent_points(latent_dim, n_samples):
    """Generate points in latent space as input for the generator."""
    # generate points in the latent space
    return rng.standard_normal(size=(n_samples, latent_dim))


def generate_fake_samples(generator, latent_dim, n_samples):
    """Use the generator to generate n fake examples, with class labels."""
    # generate points in latent space
    input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict(input)
    # create class labels
    y = np.zeros((n_samples, 1))
    y = smooth_negative_labels(y)
    return images, y


def summarize_performance(step, g_model, c_model, latent_dim, dataset, n_samples=100):
    """Generate samples, saving as tuple of projections, and save models."""
    # Prepare fake examples.
    # The Generator returns 4D tensors with the last axis of length 1.
    # i.e., (n_samples, rows, cols, 1)
    fake, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # Scale from [-1,1] to [0, RADAR_MAX].
    fake = [common.RADAR_MAX * (v + 1.) / 2. for v in fake]
    # Convert back to [(xz, yz, xy), ...] from [[XZ],[YZ],[XY]].
    XZ, YZ, XY = fake[0], fake[1], fake[2]
    samples = [(XZ[i, :, :, 0], YZ[i, :, :, 0], XY[i, :, :, 0])
               for i in range(n_samples)]
    # Resize samples back to original dimensions.
    out = []
    for s in samples:
        # Convert numpy ndarrays to PIL Image objects.
        # Note: Using PIL because its really fast.
        xz, yz, xy = Image.fromarray(s[0]), Image.fromarray(
            s[1]), Image.fromarray(s[2])
        # Scale PIL Images, convert back to numpy ndarrys.
        xz = np.asarray(xz.resize(XZ_SIZE, resample=Image.BICUBIC))
        yz = np.asarray(yz.resize(YZ_SIZE, resample=Image.BICUBIC))
        xy = np.asarray(xy.resize(XY_SIZE, resample=Image.BICUBIC))
        # Append tuple of projections to output.
        out.append((xz, yz, xy))
    # Create data set.
    data = {'samples': out, 'labels': ['generated_data'] * n_samples}
    # Write serialized data set to disk.
    filename1 = os.path.join(
        args.results_dir, f'generated_data_{step+1:04d}.pickle')
    with open(filename1, 'wb') as fp:
        pickle.dump(data, fp)
    # Evaluate the classifier model.
    X, y, *_ = dataset
    _, acc = c_model.evaluate([X[..., 0], X[..., 1], X[..., 2]], y)
    logger.info(f'Classifier accuracy at step {step+1}: {acc*100:.2f}%')
    # Reset metrics to avoid accumulation in next model operation.
    c_model.reset_metrics()
    # Save the generator model.
    filename2 = os.path.join(args.results_dir, 'g_model_%04d.h5' % (step+1))
    g_model.save(filename2)
    # Save the classifier model.
    filename3 = os.path.join(args.results_dir, 'c_model_%04d.h5' % (step+1))
    c_model.save(filename3)
    logger.info('Saved: %s, %s, and %s' % (filename1, filename2, filename3))


def train(g_model, d_model, c_model, gan_model,
          train_set, val_set, n_classes, w_classes=None,
          latent_dim=100, n_epochs=15, n_batch=32,
          ):
    """Train the generator and discriminator."""
    # Select supervised dataset.
    X_sup, y_sup = select_supervised_samples(train_set, n_classes=n_classes)
    # Calculate the number of batches per training epoch.
    bat_per_epo = int(train_set[0].shape[0] / n_batch)
    # Calculate the number of training iterations.
    n_steps = bat_per_epo * n_epochs
    # Calculate the size of half a batch of samples.
    half_batch = int(n_batch / 2)
    logger.info(f'Starting training loop.')
    logger.info('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' %
                (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    # Manually enumerate epochs.
    for i in range(n_steps):
        # Update supervised discriminator (c).
        [Xsup_real, ysup_real], _ = generate_real_samples(
            [X_sup, y_sup], half_batch)
        c_loss, c_acc = c_model.train_on_batch(
            [Xsup_real[..., 0], Xsup_real[..., 1], Xsup_real[..., 2]], ysup_real)
        # Update unsupervised discriminator (d).
        [X_real, _], y_real = generate_real_samples(train_set, half_batch)
        dr_loss = d_model.train_on_batch(
            [X_real[..., 0], X_real[..., 1], X_real[..., 2]], y_real, class_weight=w_classes)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        df_loss = d_model.train_on_batch(X_fake, y_fake)
        # Update generator (g).
        X_gan, y_gan = generate_latent_points(
            latent_dim, n_batch), np.ones((n_batch, 1))
        y_gan = smooth_positive_labels(y_gan)
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # Summarize loss and acc on this batch.
        logger.debug('Training results at step %d: c[%.3f,%.0f], d_r[%.3f], d_f[%.3f], g[%.3f]' %
                    (i+1, c_loss, c_acc*100, dr_loss, df_loss, g_loss))
        # Evaluate the model performance every so often.
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, c_model, latent_dim, val_set)


def get_datasets(args):
    """Gets and parses dataset(s) from command line.

    Args:
        args (parser object): command line arguments.

    Returns:
        samples (list of tuples of np.arrays): Radar samples in the form [(xz, yz, xy), ...] in range [0, RADAR_MAX].
        labels (list of strings): Radar sample labels.
        samples_sup: (list of bool): Radar samples to use for supervised learning.

    Note:
        Causes program to exit if data set not found on filesystem. 
    """
    samples, labels, samples_sup = [], [], []

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
        samples_sup.extend([True] * len(data_pickle['samples'])
                           if dataset in args.datasets_as_sup or not args.datasets_as_sup
                           else [False] * len(data_pickle['samples']))

    return samples, labels, samples_sup


def filter_data(args, samples, labels):
    """Filter desired classes and apply aliases. 

    Args:
        args (parser object): command line arguments.
        samples (list of tuples of np.arrays): Radar samples in the form [(xz, yz, xy), ...] in range [0, RADAR_MAX].
        labels (list of strings): Radar sample labels.

    Returns:
        filtered_samples (list of tuples of np.arrays): Filtered and aliased radar samples. 
        filtered_labels (list of strings): Filtered and aliased radar sample labels.

    Note:
        Returns original samples and labels if no filter and no aliases.
    """
    # Alias class names.
    keys = CLASS_ALIAS.keys()
    if set(labels) - set(keys):
        logger.info('Using aliased class names.')
        aliased_labels = list(
            map(
                lambda x: CLASS_ALIAS[x] if x in list(keys) else x,
                labels
            )
        )
    else:
        aliased_labels = labels

    # Filter desired samples and classes.
    logger.info('Maybe filtering data set.')
    filtered_labels = [l for l in aliased_labels if l in args.desired_labels]
    filtered_samples = [s for i, s in enumerate(
        samples) if aliased_labels[i] in args.desired_labels]

    return filtered_samples, filtered_labels


def preprocess_data(args, data, labels, samples_sup):
    """Preprocess data set for use in training the models. 

    Args:
        args (parser object): command line arguments.
        data (list of tuples of np.arrays): Radar samples in the form [(xz, yz, xy), ...] in range [0, RADAR_MAX].
        labels (list of strings): Radar sample labels.
        samples_sup: (list of bool): Mask representing radar samples to use for supervised learning.
        augment (bool): Flag indicating radar projections will be augmented.

    Returns:
        train_set (tuple of list of np.array): Training set and mask. 
        val_set (tuple of list of np.array): Normalized validation set.
        n_classes (int): Number of classes in data set.
        w_classes (dict): Class weight dict.

    Note:
        If split == 1.0 validation set = pre-balanced training set. 
    """
    # Scale each feature to the [-1, 1] range from [0, RADAR_MAX]
    logger.info('Scaling samples.')
    scaled_data = [tuple(
        (p - common.RADAR_MAX / 2.) / (common.RADAR_MAX / 2.) for p in s
    ) for s in data
    ]

    if args.augment:
        logger.info('Augmenting data.')
        scaled_data = [augment_data(d) for d in scaled_data]

    # Encode the labels.
    logger.info('Encoding labels.')
    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    class_names = list(le.classes_)

    counter = collections.Counter(encoded_labels)
    max_v = float(max(counter.values()))
    w_classes = {cls: round(max_v / v, 2) for cls, v in counter.items()}
    n_classes = len(list(counter))

    # Output data set summary.
    logger.info(
        f'Found {n_classes} classes and {len(labels)} samples:'
    )
    for i, c in enumerate(class_names):
        logger.info(
            f'...class: {i} "{c}" count: {np.count_nonzero(encoded_labels==i)}'
        )
    logger.info(f'Class weights: {w_classes}')

    # Convert radar samples from [(xz, yz, xy), ...] to [[XZ],[YZ],[XY]].
    # i.e., from tuples of projections per sample to arrays of projections.
    # This is required for downstream processing.
    # Gather up projections in each sample. Resize to make same shape.
    XZ, YZ, XY = [], [], []
    for d in scaled_data:
        # Convert numpy ndarrays to PIL Image objects.
        # Note : Using PIL because its really fast.
        xz, yz, xy = Image.fromarray(d[0]), Image.fromarray(
            d[1]), Image.fromarray(d[2])
        # Scale PIL Images, convert back to numpy ndarrys and add to lists.
        XZ.append(np.asarray(xz.resize(RESCALE, resample=Image.BICUBIC)))
        YZ.append(np.asarray(yz.resize(RESCALE, resample=Image.BICUBIC)))
        XY.append(np.asarray(xy.resize(RESCALE, resample=Image.BICUBIC)))
    # Make each array 3D so that they can be concatenated.
    XZ, YZ, XY = np.array(XZ), np.array(YZ), np.array(XY)
    XZ = XZ[..., np.newaxis]
    YZ = YZ[..., np.newaxis]
    XY = XY[..., np.newaxis]
    # Form the samples array that will be used for training.
    # It will have the shape (n_samples, rows, cols, projections).
    # Note: samples[...,0]=XZ, samples[...,1]=YZ, samples[...,2]=XY
    samples = np.concatenate((XZ, YZ, XY), axis=3)

    # Encode labels.
    encoded_labels = np.array(encoded_labels)

    # Make boolean nparray from supervised samples mask.
    samples_sup = np.array(samples_sup, dtype=bool)

    # Shuffle dataset.
    idx = np.arange(samples.shape[0])
    rng.shuffle(idx)
    samples, encoded_labels, samples_sup = samples[idx], encoded_labels[idx], samples_sup[idx]

    # Split dataset.
    split = min(int(samples.shape[0] * args.train_split), samples.shape[0])
    X_train, y_train, sup_train = samples[:
                                          split], encoded_labels[:split], samples_sup[:split]
    X_val, y_val = samples[split:], encoded_labels[split:]

    # Balance training set.
    X_train_bal, y_train_bal, sup_train_bal = balance_classes(
        X_train, y_train, sup_train)

    logger.debug(
        f'Shape summary...'
        f'  X_train_bal: {X_train_bal.shape}'
        f'  y_train_bal: {y_train_bal.shape}'
        f'  sup_train_bal: {sup_train_bal.shape}'
        f'  X_val: {X_val.shape}'
        f'  y_val: {y_val.shape}'
    )

    # If validation set is empty use pre-balanced training set instead.
    val_set = (X_val, y_val) if X_val.size > 0 else (X_train, y_train)

    train_set = X_train_bal, y_train_bal, sup_train_bal

    return train_set, val_set, n_classes, w_classes


def instantiate_models(n_classes):
    """Instantiate models. 

    Args:
        n_classes (int): Number of classes in data set.

    Returns:
        d_model (keras object): Discriminator.
        c_model (keras object): Classifier.
        g_model (keras object): Generator.
        gan_model (keras object): GAN model.

    Note:
        Logs text representations of model if 'debug' is set.
        Saves graphical representations of models to 'images' dir.
    """
    # Create classifier and discriminator.
    shape = RESCALE + (1,)
    d_model, c_model = define_discriminator(
        xz_shape=shape, yz_shape=shape, xy_shape=shape, n_classes=n_classes)
    d_model.summary(print_fn=logger.debug)
    tf.keras.utils.plot_model(d_model, to_file=os.path.join(
        common.PRJ_DIR, 'images', 'sgan_d_model.png'), show_shapes=True, show_layer_names=True)
    c_model.summary(print_fn=logger.debug)
    tf.keras.utils.plot_model(c_model, to_file=os.path.join(
        common.PRJ_DIR, 'images', 'sgan_c_model.png'), show_shapes=True, show_layer_names=True)
    # Create generator.
    g_model = define_generator()
    g_model.summary(print_fn=logger.debug)
    tf.keras.utils.plot_model(g_model, to_file=os.path.join(
        common.PRJ_DIR, 'images', 'sgan_g_model.png'), show_shapes=True, show_layer_names=True)
    # Create gan.
    gan_model = define_gan(g_model, d_model)
    gan_model.summary(print_fn=logger.debug)
    tf.keras.utils.plot_model(gan_model, to_file=os.path.join(
        common.PRJ_DIR, 'images', 'sgan_gan_model.png'), show_shapes=True, show_layer_names=True)
    return d_model, c_model, g_model, gan_model


def main(args):
    """Main program.

    Args:
        args (parser object): command line arguments.

    """
    # Log to both stdout and a file.
    log_file = os.path.join(args.results_dir, 'train.log')
    logging.basicConfig(
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        level=logging.DEBUG if args.logging_level == 'debug' else logging.INFO,
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Get samples, labels and supervised training mask.
    samples, labels, samples_sup = get_datasets(args)
    # Filter and alias samples and labels.
    filtered_samples, filtered_labels = filter_data(args, samples, labels)
    # Prepare data for training.
    train_set, val_set, n_classes, _ = preprocess_data(
        args, filtered_samples, filtered_labels, samples_sup)
    # Create the models.
    d_model, c_model, g_model, gan_model = instantiate_models(n_classes)
    # Actual training.
    train(g_model, d_model, c_model, gan_model, train_set,
          val_set=val_set, n_classes=n_classes)


if __name__ == '__main__':
    # Directory to save training results.
    default_results_dir = 'train-results/sgan'
    # Training datasets.
    default_datasets = []
    # Supervised datasets.
    default_datasets_as_sup = []
    # Labels to use for training.
    default_desired_labels = ['person', 'dog', 'cat', 'pet']
    # Fraction of data set used for training, must be <=1.0.
    default_train_split = 1.0

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets', nargs='+', type=str,
        help='paths to training datasets',
        default=default_datasets
    )
    parser.add_argument(
        '--datasets_as_sup', nargs='+', type=str,
        help='dataset(s) for supervised training',
        default=default_datasets_as_sup
    )
    parser.add_argument(
        '--desired_labels', nargs='+', type=str,
        help='labels to use for training',
        default=default_desired_labels
    )
    parser.add_argument(
        '--logging_level', type=str,
        help='logging level, "info" or "debug"',
        default='info'
    )
    parser.add_argument(
        '--train_split', type=float,
        help='train fraction of data set',
        default=default_train_split
    )
    parser.add_argument(
        '--results_dir', type=str,
        help='training results path',
        default=os.path.join(common.PRJ_DIR, default_results_dir)
    )
    parser.add_argument(
        '--augment', action='store_true',
        help='if true data set will be augmented',
    )
    parser.set_defaults(augment=False)
    args = parser.parse_args()

    main(args)
