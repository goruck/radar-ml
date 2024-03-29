"""
Train SVM classifier using SGD or SVC on radar data.

Example usage:
    $ python3 ./train.py \
        --datasets datasets/radar_samples_25Nov20.pickle datasets/radar_samples.pickle \
        --epochs 4

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

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn import (model_selection, metrics, preprocessing, linear_model,
    svm, utils, calibration)

import common

logger = logging.getLogger(__name__)

# Define a seed so random operations are the same from run to run.
RANDOM_SEED = 1234

class DataGenerator(object):
    """Generate augmented radar data."""
    def __init__(
            self, rotation_range=None, zoom_range=None,
            noise_sd=None, balance=False):
        """Initialize generator behavior.

        Args:
            rotation_range (float): Range of angles to rotate data [-rotation_range, rotation_range].
            zoom_range (float): Range of scale factors to zoom data [1-zoom_range, 1+zoom_range].
            noise_sd (float): Standard deviation of Gaussian noise added to data.
            balance: (bool): Balance classes while augmenting data. This will make all class samples
                equal to the minority class samples which may lead to large data sets for highly
                imbalanced classes.

        Note:
            Augmenting data with balance set may not result in perfect balance.
        """
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.noise_sd = noise_sd
        self.balance = balance

    def flow(
            self, x, y, batch_size=32, save_to_dir=None,
            save_prefix='./datasets/augment'):
        """Yield batches of augmented radar data.

        Args:
            x (list of np.array): Data to augment.
            y (list of int): Labels of data to augment.
            batch_size (int): Size of sub-groups of data to process.
            save_to_dir (bool): If true save augmented data to disk.
            save_prefix (str): Location to save augmented data.

        Yields:
            x_batch (list of np.array): Batch of augmented data.
            y_batch (np.array): Batch of augmented lables.

        Examples:
            for e in range(EPOCHS):
                batch = 0
                for X_batch, y_batch in data_gen.flow(xc, yc, batch_size=BATCH_SIZE):
                    X_train.extend(X_batch)
                    y_train.extend(y_batch)
                    batch += 1
                        if batch >= len(xc) / BATCH_SIZE:
                        break
        """

        def augment(x_batch, y_batch, class_weights):
            rg = np.random.Generator(np.random.PCG64())

            def rotate(p):
                """Rotate projection."""
                angle = np.random.uniform(-1*self.rotation_range, self.rotation_range)
                out = ndimage.rotate(p, angle, reshape=False)
                # Clamp to [0,1].
                out[out>1.0] = 1.0
                out[out<0.0] = 0.0
                return out

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

                # Clamp to [0,1].
                out[out>1.0] = 1.0
                out[out<0.0] = 0.0

                return out

            def sparse_noise(q, sd):
                """Add Gaussian noise w/o breaking sparsity."""
                qc = q.copy() # do not mutate original list
                qc[qc!=0] += rg.normal(scale=sd)
                # Clamp to [0,1].
                qc[qc>1.0] = 1.0
                qc[qc<0.0] = 0.0
                return qc

            aug_x = []
            aug_y = []

            for xb, yb in zip(x_batch, y_batch):
                for _ in range(int(np.round(class_weights[yb]))):
                    # Generate new tuple of rotated projections.
                    # Rotates each projection independently.
                    if self.rotation_range is not None:
                        new_t = tuple(rotate(p) for p in xb)
                        aug_x.append(new_t)
                        aug_y.append(yb)
                    # Generate new tuple of zoomed projections.
                    # Use same zoom scale for all projections.
                    if self.zoom_range is not None:
                        zoom_factor = np.random.uniform(
                            1.0 - self.zoom_range,
                            1.0 + self.zoom_range
                            )
                        new_t = tuple(clipped_zoom(p, zoom_factor) for p in xb)
                        aug_x.append(new_t)
                        aug_y.append(yb)
                    # Generate new tuple of projections with Gaussian noise.
                    # Adds noise to each projection independently.
                    if self.noise_sd is not None:
                        new_t = tuple(sparse_noise(p, self.noise_sd) for p in xb)
                        aug_x.append(new_t)
                        aug_y.append(yb)
            return aug_x, np.array(aug_y)

        # Determine parameters to balance data.
        # Most common classes and their counts from the most common to the least.
        c = collections.Counter(y)
        mc = c.most_common()
        logger.debug(f'class most common: {mc}')
        if self.balance:
            class_weights = {c : mc[0][1] / cnt for c, cnt in mc}
        else:
            class_weights = {c : 1 for c, _ in mc}
        logger.debug(f'class_weights: {class_weights}')

        # Generate augmented data.
        # Runs forever. Loop needs to be broken by calling function.
        batch = 0
        while True:
            for pos in range(0, len(x), batch_size):
                remaining = len(x) - pos
                end = remaining if remaining < batch_size else batch_size
                x_batch = x[pos:pos + end]
                y_batch = y[pos:pos + end]
                yield augment(x_batch, y_batch, class_weights)
                # Save augmented batches to disk if desired.
                if save_to_dir is not None:
                    fname = f'batch_{str(batch)}_{str(pos)}.pickle'
                    with open(os.path.join(save_prefix, fname), 'wb') as fp:
                        pickle.dump({'x_batch':x_batch, 'y_batch':y_batch}, fp)
            batch += 1

def evaluate_model(model, X_test, y_test, target_names, cm_name):
    """Generate model confusion matrix and classification report."""
    y_pred = model.predict(X_test)
    logger.info(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')
    cm = metrics.confusion_matrix(y_test, y_pred)
    logger.info(f'Confusion matrix:\n{cm}')
    cm_figure = plot_confusion_matrix(cm, class_names=target_names)
    logger.info(f'Saving confusion matrix plot to: {cm_name}')
    cm_figure.savefig(cm_name)
    cm_figure.clf()
    cls_report = metrics.classification_report(
        y_test, y_pred, target_names=target_names
        )
    logger.info(f'Classification report:\n{cls_report}')

def balance_classes(labels, data):
    """Balance classess."""
    # Most common classes and their counts from the most common to the least.
    c = collections.Counter(labels)
    mc = c.most_common()

    # Return if already balanced.
    if len(set([c for _, c in mc])) == 1: return labels, data

    #print(f'Unbalanced most common: {mc}')
    #print(f'Unbalanced label len: {len(labels)}')
    #print(f'Unbalanced data len: {len(data)}')
    
    # Build a list of class indices from most common rankings.
    indices = [np.nonzero(labels==i)[0] for (i, _) in mc]
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
            random_state=RANDOM_SEED)   # reproducible results
    data_upsampled = [upsample(data) for data in data_list]
    labels_upsampled = [upsample(label) for label in labels_list]

    # Recombine the separate, and now upsampled, label and data sets. 
    data_balanced = functools.reduce(
        lambda a, b: np.vstack((a, b)), data_upsampled)
    labels_balanced = functools.reduce(
        lambda a, b: np.concatenate((a, b)), labels_upsampled)

    c = collections.Counter(labels_balanced)
    mc = c.most_common()

    #print(f'Balanced most common: {mc}')
    #print(f'Balanced label len: {len(labels_balanced)}')
    #print(f'Balanced data len: {len(data_balanced)}')
    
    return labels_balanced, data_balanced

def plot_dataset(labels, data):
    plt.matshow(data.transpose(), fignum='all classes', aspect='auto')

    c = collections.Counter(labels)
    mc = c.most_common()

    # Build a list of class indices from most common rankings. 
    indices = [np.nonzero(labels==i)[0] for (i, _) in mc]
    # Use that list to build a list of data sets corresponding to each class.
    data_list = [data[i] for i in indices]

    # Plot data set per class. 
    for c, d in enumerate(data_list):
        plt.matshow(d.transpose(), fignum=f'class {str(c)}', aspect='auto')

    plt.show()

def plot_confusion_matrix(cm, class_names):
    """Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i, j] > threshold else 'black'
        plt.text(j, i, cm[i, j], horizontalalignment='center', color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure

def sgd_fit(
        train,
        test,
        proj_mask,
        online_learn,
        svm_model,
        epochs,
        folds=5,
        batch_size=32):
    """ Fit SVM using SGD on data set. 

    Args:
        train (tuple of list): (X, y) train data.
        test (tuple of list): (X, y) test data.
        proj_mask (Namedtuple): Radar projections to use for training.
        online_learn (bool): If True perform online learning with data.
        svm_model (str): Name of existing svm model for online learning.
        epochs (int): Number of times to augment data.
        folds (int, optional): Number of folds for the Stratified K-Folds
            cross-validator. Default=5
        batch_size (int, optional): Augment batch size. Default=32.

    Returns:
        estimator: Estimator that was chosen by grid search.
    """

    def find_best_sgd_svm_estimator(X, y, cv, random_seed):
        """Exhaustive search over specified parameter values for svm using sgd.

        Returns:
            optimized svm estimator.
        """
        max_iter = max(np.ceil(10**6 / len(X)), 1000)
        small_alphas = [10.0e-08, 10.0e-09, 10.0e-10]
        alphas = [10.0e-04, 10.0e-05, 10.0e-06, 10.0e-07]
        l1_ratios = [0.075, 0.15, 0.30]
        param_grid = [
            {'alpha': alphas, 'penalty': ['l1', 'l2'], 'average':[False]},
            {'alpha': alphas, 'penalty': ['elasticnet'], 'average':[False],
            'l1_ratio': l1_ratios},
            {'alpha': small_alphas, 'penalty': ['l1', 'l2'], 'average':[True]},
            {'alpha': small_alphas, 'penalty': ['elasticnet'], 'average':[True],
            'l1_ratio': l1_ratios}
            ]
        init_est = linear_model.SGDClassifier(loss='log', max_iter=max_iter,
            random_state=random_seed, n_jobs=-1, warm_start=True)
        grid_search = model_selection.GridSearchCV(estimator=init_est,
            param_grid=param_grid, verbose=2, n_jobs=-1, cv=cv)
        grid_search.fit(X, y)
        #print('\n All results:')
        #print(grid_search.cv_results_)
        logger.info('\n Best estimator:')
        logger.info(grid_search.best_estimator_)
        logger.info('\n Best score for {}-fold search:'.format(folds))
        logger.info(grid_search.best_score_)
        logger.info('\n Best hyperparameters:')
        logger.info(grid_search.best_params_)
        return grid_search.best_estimator_

    X_train, y_train = train
    X_test, y_test = test

    # Make a copy of train set for later use in augmentation. 
    if epochs:
        xc = X_train.copy()
        yc = y_train.copy()

    # Generate feature vectors from radar projections.
    logger.info('Generating feature vectors.')
    X_train = common.process_samples(X_train, proj_mask=proj_mask)
    X_test = common.process_samples(X_test, proj_mask=proj_mask)
    logger.info(f'Feature vector length: {X_train.shape[1]}')

    # Balance classes.
    logger.info('Balancing classes.')
    y_train, X_train = balance_classes(y_train, X_train)

    if not online_learn: 
        # Find best initial classifier.
        logger.info('Running best fit with new data.')
        skf = model_selection.StratifiedKFold(n_splits=folds)
        clf = find_best_sgd_svm_estimator(
            X_train, y_train,
            skf.split(X_train, y_train), RANDOM_SEED
            )
    else:
        # Fit existing classifier with new data.
        logger.info('Running partial fit with new data.')
        with open(os.path.join(common.PRJ_DIR, svm_model), 'rb') as fp:
            clf = pickle.load(fp)
        max_iter = max(np.ceil(10**6 / len(X_train)), 1000)
        for _ in range(max_iter):
            clf.partial_fit(X_train, y_train)

    # Augment training set and use to run partial fits on classifier.
    if epochs:
        logger.info(f'Running partial fit with augmented data (epochs: {epochs}).')
        y_predicted = clf.predict(X_test)
        logger.debug(f'Un-augmented accuracy: {metrics.accuracy_score(y_test, y_predicted)}.')
        data_gen = DataGenerator(
            rotation_range=5.0, zoom_range=0.2, noise_sd=0.1, balance=True)
        for e in range(epochs):
            logger.debug(f'Augment epoch: {e}.')
            batch = 0
            for X_batch, y_batch in data_gen.flow(xc, yc, batch_size=batch_size):
                logger.debug(f'Augment batch: {batch}.')
                X_batch = common.process_samples(X_batch, proj_mask=proj_mask)
                y_batch, X_batch = balance_classes(y_batch, X_batch)
                clf.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
                y_predicted = clf.predict(X_test)
                acc = metrics.accuracy_score(y_test, y_predicted)
                logger.debug(f'Augmented accuracy: {acc}.')
                batch += 1
                if batch >= len(xc) / batch_size:
                    break

    return clf

def svc_fit(
        train,
        proj_mask,
        epochs,
        folds=5,
        batch_size=32):
    """ Fit SVM using SVC on data set. 

    Args:
        train (tuple of list): (X, y) train data.
        proj_mask (Namedtuple): Radar projections to use for training.
        epochs (int): Number of times to augment data.
        folds (int, optional): Number of folds for the Stratified K-Folds
            cross-validator. Default=5
        batch_size (int, optional): Augment batch size. Default=32.

    Returns:
        estimator: Estimator that was chosen by grid search.
    """

    def find_best_svm_estimator(X, y, cv, random_seed):
        """Exhaustive search over specified parameter values for svm.

        Returns:
            optimized svm estimator.

        Note:
            https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
        """
        print('\n Finding best svm estimator...')
        Cs = [0.01, 0.1, 1, 10, 100]
        gammas = [0.001, 0.01, 0.1, 1, 10]
        param_grid = [
            {'C': Cs, 'kernel': ['linear']},
            {'C': Cs, 'gamma': gammas, 'kernel': ['rbf']}
            ]
        init_est = svm.SVC(probability=True, class_weight='balanced',
            random_state=random_seed, cache_size=1000, verbose=False)
        grid_search = model_selection.GridSearchCV(estimator=init_est,
            param_grid=param_grid, verbose=2, n_jobs=4, cv=cv)
        grid_search.fit(X, y)
        #print('\n All results:')
        #print(grid_search.cv_results_)
        logger.info('\n Best estimator:')
        logger.info(grid_search.best_estimator_)
        logger.info('\n Best score for {}-fold search:'.format(folds))
        logger.info(grid_search.best_score_)
        logger.info('\n Best hyperparameters:')
        logger.info(grid_search.best_params_)
        return grid_search.best_estimator_

    X_train, y_train = train

     # Augment training set.
    if epochs:
        data_gen = DataGenerator(rotation_range=15.0, zoom_range=0.3, noise_sd=0.2)
        logger.info('Augmenting data set.')
        logger.info(f'Original number of training samples: {y_train.shape[0]}')

        # Faster to use a list in below ops.
        y_train = y_train.tolist()

        # Do not mutate original lists.
        xc = X_train.copy()
        yc = y_train.copy()

        for e in range(epochs):
            logger.debug(f'epoch: {e}')
            batch = 0
            for X_batch, y_batch in data_gen.flow(xc, yc, batch_size=batch_size):
                logger.debug(f'batch: {batch}')
                X_train.extend(X_batch)
                y_train.extend(y_batch)
                batch += 1
                if batch >= len(xc) / batch_size:
                    break
    
        # Sanity check if augmentation introduced a scaling problem.
        max = np.amax([[np.concatenate(t, axis=None)] for t in X_train])
        assert abs(max - 1.0) < 1e-6, 'scale error'

        # Convert y_train back to np array.
        y_train = np.array(y_train, dtype=np.int8)

        logger.info(f'Augmented number of training samples: {y_train.shape[0]}')

    logger.info('Generating feature vectors from radar projections.')
    X_train = common.process_samples(X_train, proj_mask=proj_mask)
    logger.info(f'Feature vector length: {X_train.shape[1]}')

    # Balance classes.
    logger.info('Balancing classes.')
    y_train, X_train = balance_classes(y_train, X_train)

    skf = model_selection.StratifiedKFold(n_splits=folds)

    # Find best classifier.
    logger.info('Finding best classifier.')
    clf = find_best_svm_estimator(
        X_train, y_train,
        skf.split(X_train, y_train), RANDOM_SEED
        )

    return clf

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
    default_desired_labels = ['person', 'dog', 'cat']
    # Each epoch augments entire data set (zero disables).
    default_epochs = 0
    # Fraction of data set used for training, validation, testing.
    # Must sum to 1.0.
    default_train_val_test_frac = [0.8, 0.1, 0.1]

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

    # Scale each feature to the [0, 1] range without breaking the sparsity.
    logger.info('Scaling samples.')
    samples = [[p / common.RADAR_MAX for p in s] for s in samples]

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

    if not args.use_svc:
        logger.info('Using SVM algo: SGDClassifier.')
        clf = sgd_fit(
            train=(X_train, y_train),
            test=(X_test, y_test),
            proj_mask=args.proj_mask,
            online_learn=args.online_learn,
            svm_model=args.svm_model,
            epochs=args.epochs
            )
    else:
        logger.info('Using SVM algo: SVC.')
        clf = svc_fit(
            train=(X_train, y_train),
            proj_mask=args.proj_mask,
            epochs=args.epochs
            )

    # Generate feature vectors.
    X_val_fv = common.process_samples(X_val, proj_mask=proj_mask)
    X_test_fv = common.process_samples(X_test, proj_mask=proj_mask)

    logger.info('Calibrating classifier.')
    cal_clf = calibration.CalibratedClassifierCV(base_estimator=clf, cv='prefit')
    cal_clf.fit(X_val_fv, y_val)

    logger.info('Evaluating final classifier on test set.')
    evaluate_model(cal_clf, X_test_fv, y_test, class_names, args.svm_cm)

    logger.info(f'Saving svm model to: {args.svm_model}.')
    with open(args.svm_model, 'wb') as outfile:
        outfile.write(pickle.dumps(cal_clf))

    # Do not overwrite label encoder if online learning was performed.
    if not args.online_learn or args.use_svc:
        logger.info(f'Saving label encoder to: {args.label_encoder}.')
        with open(args.label_encoder, 'wb') as outfile:
            outfile.write(pickle.dumps(le))