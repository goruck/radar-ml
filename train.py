"""
Train SVM- and XGB-based classifiers on radar data.

Copyright (c) 2020 Lindo St. Angel
"""

import os
#import sys
import collections
import itertools
import functools
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics, preprocessing, utils, svm
from scipy import ndimage
#import xgboost as xgb

import common

#np.set_printoptions(threshold=sys.maxsize)

# Output SVM confusion matrix name.
SVM_CM = 'train-results/svm_cm.png'
# Output XGBoost confusion matrix name.
XGB_CM = 'train-results/xgb_cm.png'
# Define a seed so random operations are the same from run to run.
RANDOM_SEED = 1234
# Define number of folds for the Stratified K-Folds cross-validator.
FOLDS = 5
# Number of parameters to combine for xgb random search. 
PARA_COMB = 20
# Radar 2-D projections to use for predictions.
PROJ_MASK = common.ProjMask(xy=True, xz=True, yz=True)

class DataGenerator:
    def __init__(self, rotation_range=None, zoom_range=None, noise_sd=None):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.noise_sd = noise_sd

    def flow(self, x, y, batch_size=32, save_to_dir=None, save_prefix='./datasets/augment'):
        def augment(x_batch, y_batch):
            aug_x = []
            aug_y = []
            for xb, yb in zip(x_batch, y_batch):
                # Generate new tuple of rotated projections.
                # Rotate each projection independenly.
                if self.rotation_range is not None:
                    new_t = tuple(ndimage.rotate(p, random.uniform(0, self.rotation_range))
                            for p in xb)
                    aug_x.append(new_t)
                    aug_y.append(yb)
                # Generate new tuple of zoomed projections.
                # Use same zoom scale for all projections.
                if self.zoom_range is not None:
                    scale = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
                    new_t = tuple(ndimage.zoom(p, scale) for p in xb)
                    aug_x.append(new_t)
                    aug_y.append(yb)
                # Generate new tuple of projections with Gaussian noise.
                # Add noise to each projection independently.
                if self.noise_sd is not None:
                    new_t = tuple(np.random.normal(p, self.noise_sd) for p in xb)
                    aug_x.append(new_t)
                    aug_y.append(yb)
            return aug_x, aug_y

        # Generate augmented data.
        # Runs forever. Loop needs to be broken by calling function.
        batch = 0
        while True:
            for pos in range(0, len(x), batch_size):
                remaining = len(x) - pos
                end = remaining if remaining < batch_size else batch_size
                x_batch = x[pos:pos + end]
                y_batch = y[pos:pos + end]
                yield augment(x_batch, y_batch)
                # Save augmented batches to disk if desired.
                if save_to_dir is not None:
                    fname = f'batch_{str(batch)}_{str(pos)}.pickle'
                    with open(os.path.join(save_prefix, fname), 'wb') as fp:
                        pickle.dump({'x_batch':x_batch, 'y_batch':y_batch}, fp)
            batch += 1

def evaluate_model(model, X_test, y_test, target_names, cm_name):
    """ Generate model confusion matrix and classification report. """
    print('\n Evaluating model.')
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(f'\n Confusion matrix:\n{cm}')
    cm_figure = plot_confusion_matrix(cm, class_names=target_names)
    cm_figure.savefig(os.path.join(common.PRJ_DIR, cm_name))
    cm_figure.clf()
    print('\n Classification matrix:')
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))
    return

def balance_classes(labels, data):
    """ balance classess """
    # Most common classes and their counts from the most common to the least.
    c = collections.Counter(labels)
    mc = c.most_common()

    # Return if already balanced.
    if len(set([c for _, c in mc])) == 1: return labels, data

    print(f'Unbalanced most common: {mc}')
    print(f'Unbalanced label shape: {labels.shape}')
    print(f'Unbalanced data shape: {data.shape}')
    
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
    data_balanced = functools.reduce(lambda a, b: np.vstack((a, b)), data_upsampled)
    labels_balanced = functools.reduce(lambda a, b: np.concatenate((a, b)), labels_upsampled)

    c = collections.Counter(labels_balanced)
    mc = c.most_common()

    print(f'Balanced most common: {mc}')
    print(f'Balanced label shape: {labels_balanced.shape}')
    print(f'Balanced data shape: {data_balanced.shape}')
    
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

    return

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Parameters:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure

def find_best_svm_estimator(X, y, cv, random_seed):
    # Exhaustive search over specified parameter values for svm.
    # Returns optimized svm estimator.
    print('\n Finding best svm estimator...')
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = [
        {'C': Cs, 'kernel': ['linear']},
        {'C': Cs, 'gamma': gammas, 'kernel': ['rbf']}]
    init_est = svm.SVC(probability=True, class_weight='balanced',
        random_state=random_seed, cache_size=1000, verbose=False)
    grid_search = model_selection.GridSearchCV(estimator=init_est,
        param_grid=param_grid, verbose=1, n_jobs=4, cv=cv)
    grid_search.fit(X, y)
    #print('\n All results:')
    #print(grid_search.cv_results_)
    print('\n Best estimator:')
    print(grid_search.best_estimator_)
    print('\n Best score for {}-fold search:'.format(FOLDS))
    print(grid_search.best_score_)
    print('\n Best hyperparameters:')
    print(grid_search.best_params_)
    return grid_search.best_estimator_

def find_best_xgb_estimator(X, y, cv, param_comb, random_seed):
    # Random search over specified parameter values for XGBoost.
    # Exhaustive search takes many more cycles w/o much benefit.
    # Returns optimized XGBoost estimator.
    # Ref: https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost
    print('\n Finding best XGBoost estimator...')
    param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
    init_est = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='multi:softprob',
        verbose=1, n_jobs=1, random_state=random_seed)
    random_search = model_selection.RandomizedSearchCV(estimator=init_est,
        param_distributions=param_grid, n_iter=param_comb, n_jobs=4,
        cv=cv, verbose=1, random_state=random_seed)
    random_search.fit(X, y)
    #print('\n All results:')
    #print(random_search.cv_results_)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best score for {}-fold search with {} parameter combinations:'
        .format(FOLDS, PARA_COMB))
    print(random_search.best_score_)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    return random_search.best_estimator_

def main():
    # Load radar observations and labels. 
    with open(os.path.join(common.PRJ_DIR, common.RADAR_DATA), 'rb') as fp:
        data_pickle = pickle.load(fp)

    samples = data_pickle['samples']

    # Encode the labels.
    print('Encoding labels.')
    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(data_pickle['labels'])
    class_names = list(le.classes_)
    print(f'class names: {class_names}')

    # Split data and labels up into train and test sets.
    print('Splitting data into train and test sets.')
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        samples, encoded_labels, test_size=0.20, random_state=RANDOM_SEED, shuffle=True)
    #print(f'X_train: {X_train} X_test: {X_test} y_train: {y_train} y_test: {y_test}')

    # Balance the dataset. 
    #balanced_labels, balanced_data = balance_classes(encoded_labels, processed_samples)

    # Plot the dataset.
    #plot_data = balanced_data
    #plot_dataset(balanced_labels, plot_data)

    # Augment training set.
    data_gen = DataGenerator(rotation_range=20.0, zoom_range=0.2, noise_sd=1.0)
    print('Augmenting data set.')
    print(f'X len: {len(X_train)}, y len: {len(y_train)}')
    EPOCHES = 2 # Each epoch augments entire data set. 
    BATCH_SIZE = 32 # Augment batch size.
    y_train = y_train.tolist() # Faster to use a list in below ops.
    xc = X_train.copy()
    yc = y_train.copy()
    for e in range(EPOCHES):
        print(f'epoch: {e}')
        batch = 0
        for X_batch, y_batch in data_gen.flow(xc, yc, batch_size=BATCH_SIZE):
            print(f'batch: {batch}')
            X_train.extend(X_batch)
            y_train.extend(y_batch)
            batch += 1
            if batch >= len(xc) / BATCH_SIZE:
                break

    print(f'aug X len: {len(X_train)}, aug y len: {len(y_train)}')

    y_train = np.array(y_train, dtype=np.int8) # Convert back since sklearn wants np.array.

    print('Preparing samples for training.')
    X_train = common.process_samples(X_train, proj_mask=PROJ_MASK)
    print(f'X_train shape: {X_train.shape}')

    skf = model_selection.StratifiedKFold(n_splits=FOLDS)

    # Find best svm classifier, evaluate and then save it.
    best_svm = find_best_svm_estimator(X_train, y_train,
        skf.split(X_train, y_train), RANDOM_SEED)

    print('Preparing samples for testimg.')
    X_test = common.process_samples(X_test, proj_mask=PROJ_MASK)
    print(f'X_test shape: {X_test.shape}')
    evaluate_model(best_svm, X_test, y_test, class_names, SVM_CM)

    print('\n Saving svm model...')
    with open(os.path.join(common.PRJ_DIR, common.SVM_MODEL), 'wb') as outfile:
        outfile.write(pickle.dumps(best_svm))
    """
    # Find best XGBoost classifier, evaluate and save it. 
    best_xgb = find_best_xgb_estimator(X_train, y_train, skf.split(X_train, y_train),
        PARA_COMB, RANDOM_SEED)

    evaluate_model(best_xgb, X_test, y_test, class_names, XGB_CM)

    print('\n Saving xgb model...')
    with open(os.path.join(common.PRJ_DIR, common.XGB_MODEL), 'wb') as outfile:
        outfile.write(pickle.dumps(best_xgb))
    """
    # Write the label encoder to disk.
    print('\n Saving label encoder.')
    with open(os.path.join(common.PRJ_DIR, common.LABELS), 'wb') as outfile:
        outfile.write(pickle.dumps(le))

if __name__ == '__main__':
    main()