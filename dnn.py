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
import pickle

import numpy as np
from scipy import ndimage
from sklearn import preprocessing
import tensorflow as tf
from PIL import Image

import common

logger = logging.getLogger(__name__)

RANDOM_SEED = 1234
rng = np.random.default_rng(RANDOM_SEED)

# Radar projection rescaling factor.
RESCALE = (80, 80)

# Class aliases.
# Some data sets used pet names instead of pet type so this makes them consistent.
CLASS_ALIAS = {'polly': 'dog', 'rebel': 'cat'}
# Make dogs and cats as pets.
#CLASS_ALIAS = {'polly': 'pet', 'rebel': 'pet', 'dog': 'pet', 'cat': 'pet'}

# Uncomment line below to print all elements of numpy arrays.
# np.set_printoptions(threshold=sys.maxsize)


def create_conv_layers(input):
    """Creates convolutional layers."""
    input_shape = input.shape[1:]
    conv = tf.keras.layers.Conv2D(64, (3, 3), strides=(
        2, 2), padding='same', activation='relu', input_shape=input_shape)(input)
    conv = tf.keras.layers.Conv2D(
        32, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    return conv


def define_classifier(xz_shape, yz_shape, xy_shape, n_classes):
    """Define classifier model.

    Args:
        xz_shape, yz_shape, xy_shape (tuple): Shapes of radar projections.
        n_classes (int): Number of classes for model.

    Returns:
        model (Keras object): Classifier model.

    Note:
        Input ordering is xz, yz, xy.
    """
    # Make convolutional layers for each radar projection.
    xz_input = tf.keras.layers.Input(shape=xz_shape)
    xz_conv = create_conv_layers(xz_input)
    yz_input = tf.keras.layers.Input(shape=yz_shape)
    yz_conv = create_conv_layers(yz_input)
    xy_input = tf.keras.layers.Input(shape=xy_shape)
    xy_conv = create_conv_layers(xy_input)
    # Concat convolutions.
    conv = tf.keras.layers.concatenate([xz_conv, yz_conv, xy_conv])
    # Create a feature vector.
    fv = tf.keras.layers.Flatten()(conv)
    # Create dense layers and operate on the feature vector.
    dense = tf.keras.layers.Dense(64, activation='relu')(fv)
    dense = tf.keras.layers.Dropout(0.5)(dense)
    dense = tf.keras.layers.Dense(64, activation='relu')(dense)
    dense = tf.keras.layers.Dropout(0.5)(dense)
    # Classifier.
    cls = tf.keras.layers.Dense(units=n_classes, activation='softmax')(dense)
    # Create model and compile it.
    model = tf.keras.Model(
        inputs=[xz_input, yz_input, xy_input], outputs=[cls])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
        lr=0.0002, beta_1=0.5), metrics=['accuracy'])
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


def preprocess_data(args, data, labels):
    """Preprocess data set for use in training the model. 

    Args:
        args (parser object): command line arguments.
        data (list of tuples of np.arrays): Radar samples in the form [(xz, yz, xy), ...] in range [0, RADAR_MAX].
        labels (list of strings): Radar sample labels.
        augment (bool): Flag indicating radar projections will be augmented.

    Returns:
        train_set (tuple of list of np.array): Training set. 
        val_set (tuple of list of np.array): Validation set.
        n_classes (int): Number of classes in data set.
        w_classes (dict): Class weight dict.
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

    # Shuffle dataset.
    idx = np.arange(samples.shape[0])
    rng.shuffle(idx)
    samples, encoded_labels = samples[idx], encoded_labels[idx]

    # Split dataset.
    split = min(int(samples.shape[0] * args.train_split), samples.shape[0])
    X_train, y_train = samples[:split], encoded_labels[:split]
    X_val, y_val = samples[split:], encoded_labels[split:]

    logger.debug(
        f'Shape summary...'
        f'  X_train: {X_train.shape}'
        f'  y_train: {y_train.shape}'
        f'  X_val: {X_val.shape}'
        f'  y_val: {y_val.shape}'
    )

    return X_train, y_train, X_val, y_val, n_classes, w_classes


def get_datasets(args):
    """Gets and parses dataset(s) from command line.

    Args:
        args (parser object): command line arguments.

    Returns:
        samples (list of tuples of np.arrays): Radar samples in the form [(xz, yz, xy), ...] in range [0, RADAR_MAX].
        labels (list of strings): Radar sample labels.

    Note:
        Causes program to exit if data set not found on filesystem. 
    """
    samples, labels = [], []

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

    return samples, labels


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


def train(model, X, y, X_val, y_val, w_classes):
    """Train model, save best model and log summary.

    Args:
        model (Keras object): dnn model to be trained.
        X (numpy array): Training data, 4-D numpy array.
        y (numpy array): Training data labels.
        X_val (numpy array): Validation data, 4-D numpy array.
        y_val (numpy array): Validation data labels.
        w_classes (dict): Class weights. 
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=10
    )
    fp = os.path.join(args.results_dir, 'c_model.h5')
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=fp,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    X_train = [X[..., 0], X[..., 1], X[..., 2]]
    X_val = [X_val[..., 0], X_val[..., 1], X_val[..., 2]]
    hist = model.fit(
        x=X_train,
        y=y,
        batch_size=64,
        epochs=100,
        validation_data=(X_val, y_val),
        class_weight=w_classes,
        callbacks=[early_stop, model_ckpt]
    )
    best_val_loss = min(hist.history['val_loss'])
    best_val_loss_idx = hist.history['val_loss'].index(best_val_loss)
    best_val_acc = hist.history['val_accuracy'][best_val_loss_idx]
    best_loss = hist.history['loss'][best_val_loss_idx]
    best_acc = hist.history['accuracy'][best_val_loss_idx]
    logger.info(f'Best loss: {best_loss:.4f}, Best acc: {best_acc*100:.2f}%')
    logger.info(
        f'Best val loss: {best_val_loss:.4f}, Best val acc: {best_val_acc*100:.2f}%')
    logger.info(f'Saved best model to {args.results_dir}')


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
    samples, labels = get_datasets(args)
    # Filter and alias samples and labels.
    filtered_samples, filtered_labels = filter_data(args, samples, labels)
    # Prepare data for training.
    X, y, X_val, y_val, n_classes, w_classes = preprocess_data(
        args, filtered_samples, filtered_labels)
    # Create the model.
    logger.info('Creating model.')
    shape = RESCALE + (1,)
    model = define_classifier(
        xz_shape=shape, yz_shape=shape, xy_shape=shape, n_classes=n_classes)
    model.summary(print_fn=logger.debug)
    # Actual training.
    logger.info('Training model.')
    train(model, X, y, X_val, y_val, w_classes)


if __name__ == '__main__':
    # Directory to save training results.
    default_results_dir = 'train-results/dnn'
    # Training datasets.
    default_datasets = []
    # Labels to use for training.
    default_desired_labels = ['person', 'dog', 'cat', 'pet']
    # Fraction of data set used for training, must be <=1.0.
    default_train_split = 0.8

    parser = argparse.ArgumentParser()
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
