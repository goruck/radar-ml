"""
Common module for radar-ml.

Copyright (c) 2020 Lindo St. Angel.
"""

import WalabotAPI as radar
import numpy as np
import scipy.ndimage
import collections
import sklearn.preprocessing

# Radar scan arena in spherical (r-Θ-Φ) coordinates from radar unit origin.
# Radial distance (R) is along the Z axis. In cm.
# Angle Theta (Θ) is from the Z axis. In degrees.
# Angle Phi (Φ) is from the X axis to orthogonal projection on the XY plane. In degrees.
# NB: (max - min) / res should be an integer.
# NB: The camera's angle of view must match the arena.
R_MIN, R_MAX, R_RES = 10, 360, 2
THETA_MIN, THETA_MAX, THETA_RES = -42, 42, 4
PHI_MIN, PHI_MAX, PHI_RES = -30, 30, 2

# Set scan profile.
RADAR_PROFILE = radar.PROF_SENSOR

# Project directory path. 
PRJ_DIR = './'

# Radar samples with ground truth lables.
RADAR_DATA = 'datasets/radar_samples.pickle'
# SVM model name.
SVM_MODEL = 'train-results/svm_radar_classifier_all.pickle'
# XGBoost model name.
XGB_MODEL = 'train-results/xgb_radar_classifier.pickle'
# Label encoder name.
LABELS = 'train-results/radar_labels.pickle'

# Radar 2-D projections to use for predictions.
ProjMask = collections.namedtuple('ProjMask', ['xz', 'yz', 'xy'])

# Radar 2-D projection zoom factors.
ProjZoom = collections.namedtuple('ProjZoom', ['xz', 'yz', 'xy'])

class DerivedTarget(collections.namedtuple('DerivedTarget',
    ['xPosCm', 'yPosCm', 'zPosCm', 'amplitude','i', 'j', 'k'])):
    """ Radar targets. Replaces Walabot getSensorTargets(). """
    @staticmethod
    def get_derived_targets(radar_data, size_x, size_y, size_z, num_targets=1):
        """ Derive targets from raw radar data. """
        def find_max_indices(inner_axis, outer_axis):
            """ Sum over specified rows and columns in data. """
            sums = np.sum(np.sum(radar_data, axis=inner_axis), axis=outer_axis)
            max_indices = np.argpartition(sums, -num_targets)[-num_targets:]
            return (max_indices[np.argsort(sums[max_indices])])

        # Find indices where radar signal is strongest.
        max_theta_indices = find_max_indices(1, 1)
        max_phi_indices = find_max_indices(0, 1)
        max_r_indices = find_max_indices(0, 0)

        def make(i, j, k):
            """
            Calculate coordinates of target from max indices.
            TODO: change to find clusters of targets.
            """
            theta = THETA_MIN + i * (THETA_MAX - THETA_MIN) / (size_x - 1)
            phi = PHI_MIN + j * (PHI_MAX - PHI_MIN) / (size_y - 1)
            r = R_MIN + k * (R_MAX - R_MIN) / (size_z - 1)
            x, y, z = spherical_to_cartesian(r, theta, phi)
            return DerivedTarget(
                xPosCm = x,
                yPosCm = y,
                zPosCm = z,
                amplitude = None, # TODO - implement,
                i = i,
                j = j,
                k = k
            )
        return [make(i, j, k) for i, j, k in zip(max_theta_indices, max_phi_indices, max_r_indices)]

def calibrate():
    """ Calibrate radar. """
    radar.StartCalibration()

    app_status, calibration_process = radar.GetStatus()
    while app_status == radar.STATUS_CALIBRATING:
        radar.Trigger()
        app_status, calibration_process = radar.GetStatus()

    return

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
    phi = np.arctan2(y, z)
    theta = np.arcsin(x / r)
    return (r, np.rad2deg(theta), np.rad2deg(phi))

def spherical_to_cartesian(r, theta, phi):
    theta_rad, phi_rad = np.deg2rad(theta), np.deg2rad(phi)
    x = r * np.sin(theta_rad)
    y = r * np.cos(theta_rad) * np.sin(phi_rad)
    z = r * np.cos(theta_rad) * np.cos(phi_rad)
    return (x, y, z)

def calculate_matrix_indices(x, y, z, size_x, size_y, size_z):
    """ Calculate 3-D radar image matrix indices from target (x, y, z) coordinates.
    Ref: https://api.walabot.com/_walabot_a_p_i_8h.html#afdbf56d82d99682627d3f781f9815f83

    Args:
        x, y, z (float, float, float): target coordinates
        size_x, size_y, size_z (int, int, int): dimensions of target signal matrix

    Returns:
        (i, j, k) (int, int, int): indices of target in signal matrix
    """
    r, theta, phi = cartesian_to_spherical(x, y, z)
    i = int((theta - THETA_MIN) * (size_x - 1) / (THETA_MAX - THETA_MIN))
    j = int((phi - PHI_MIN) * (size_y - 1) / (PHI_MAX - PHI_MIN))
    k = int((r - R_MIN) * (size_z - 1) / (R_MAX - R_MIN))
    return (i, j, k)

def process_samples(samples, proj_mask=ProjMask(xz=True,yz=True,xy=True),
    proj_zoom=ProjZoom(xz=[1.0, 1.0],yz=[1.0, 1.0],xy=[1.0, 1.0])):
    """ Prepare samples for training or predictions.

    Get projections of interest, zoom them to fit a radar arena then scale and flatten.

    Note zoom is usually used to size new samples into a different sized radar arena from
    that used for training. This is useful to use a common ml model across applications. 

    Args:
        samples (list of tuples of np arrays): Samples of radar projections [(xz, yz, xy)].
        proj_mask (tuple of bools): Projection(s) to use (xz, yz, xy).
        proj_zoom (tuple of list of floats): Projection zoom factors (xz, yz, xy).

    Returns:
        np.array: processed samples
    """
    def make(t):
        # Get projections of interest and zoom them.
        wanted_projections = tuple(scipy.ndimage.zoom(p, proj_zoom[i])
            for i, p in enumerate(t) if proj_mask[i])
        # Concatenate into a flattened feature vector.
        concat_projections = np.concatenate(wanted_projections, axis=None)
        # Scale features to the [-1, 1] range.
        return sklearn.preprocessing.maxabs_scale(concat_projections, axis=0, copy=True)
    return np.array([make(t) for t in samples])