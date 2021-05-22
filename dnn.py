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
RESCALE = (128, 128)

# Class aliases.
# Some data sets used pet names instead of pet type so this makes them consistent.
CLASS_ALIAS = {'polly': 'dog', 'rebel': 'cat'}
# Make dogs and cats as pets.
#CLASS_ALIAS = {'polly': 'pet', 'rebel': 'pet', 'dog': 'pet', 'cat': 'pet'}

# Uncomment line below to print all elements of numpy arrays.
# np.set_printoptions(threshold=sys.maxsize)


def create_conv_layers(input_scan):
    """Creates convolutional layers."""
    input_shape = input_scan.shape[1:]

    conv = tf.keras.layers.Conv2D(32, (4, 4), strides=(
        2, 2), padding='same', input_shape=input_shape)(input_scan)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)

    conv = tf.keras.layers.Conv2D(
        64, (4, 4), strides=(2, 2), padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)

    conv = tf.keras.layers.Conv2D(
        128, (4, 4), strides=(2, 2), padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)

    conv = tf.keras.layers.GlobalMaxPooling2D()(conv)

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
    xz_input = tf.keras.layers.Input(shape=xz_shape)
    xz_model = create_conv_layers(xz_input)
    yz_input = tf.keras.layers.Input(shape=yz_shape)
    yz_model = create_conv_layers(yz_input)
    xy_input = tf.keras.layers.Input(shape=xy_shape)
    xy_model = create_conv_layers(xy_input)

    conv = tf.keras.layers.concatenate([yz_model, xz_model, xy_model])

    conv = tf.keras.layers.Dense(64)(conv)
    conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)
    conv = tf.keras.layers.Dropout(0.3)(conv)

    conv = tf.keras.layers.Dense(64)(conv)
    conv = tf.keras.layers.LeakyReLU(alpha=0.2)(conv)
    conv = tf.keras.layers.Dropout(0.3)(conv)

    conv = tf.keras.layers.Dense(n_classes)(conv)

    out_layer = tf.keras.layers.Activation('softmax')(conv)

    model = tf.keras.Model(
        inputs=[xz_input, yz_input, xy_input], outputs=[out_layer])
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
    shape = RESCALE + (1,)
    model = define_classifier(
        xz_shape=shape, yz_shape=shape, xy_shape=shape, n_classes=n_classes)
    model.summary(print_fn=logger.debug)
    # Actual training.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=10
    )
    X_train = [X[...,0], X[...,1], X[...,2]]
    X_val = [X_val[...,0], X_val[...,1], X_val[...,2]]
    hist = model.fit(
        x=X_train,
        y=y,
        batch_size=64,
        epochs=100,
        validation_data=(X_val, y_val),
        class_weight=w_classes,
        callbacks=[early_stop]
    )
    low_val_loss = min(hist.history['val_loss'])
    low_val_loss_idx = hist.history.val_loss.index(low_val_loss)
    logger.info(f'low_val_loss: {low_val_loss}, low_val_loss_idx: {low_val_loss_idx}')


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

"""
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
"""
