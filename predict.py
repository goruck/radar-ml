"""
Predict objects from a radar waveform.

Copyright (c) 2020 Lindo St. Angel.
"""

import WalabotAPI as radar
import numpy as np
import pickle
import common
import logging
import argparse
import os
import sys
from sklearn import svm, linear_model

logger = logging.getLogger(__name__)

# Radar detection threshold. 
RADAR_THRESHOLD = 5
# Set to True if using Moving Target Identification (MTI) filter.
MTI = True
# Radar scan arena in polar coords.
# This can be different than arena used for training. 
R_MIN, R_MAX, R_RES = 10, 360, 2
THETA_MIN, THETA_MAX, THETA_RES = -42, 42, 4
PHI_MIN, PHI_MAX, PHI_RES = -30, 30, 2

# Radar 2-D projections to use for predictions.
PROJ_MASK = common.ProjMask(xy=True, xz=True, yz=True)

LOG_FILE = 'predict.log'

def calc_proj_zoom(train_size_x, train_size_y, train_size_z,
    size_x, size_y, size_z):
    """ Calculate projection zoom factors for prediction radar arena.

    Args:
        train_size_{x,y,z} (int): Size of image array used for training.
        size_{x,y,z} (int): Size of sample image array.

    Returns:
        ProjZoom (tuple of list of floats): Zoom factors per projection.
    """

    x_zoom = train_size_x / size_x
    y_zoom = train_size_y / size_y
    z_zoom = train_size_z / size_z
    logger.debug(f'zoom: {x_zoom}, {y_zoom}, {z_zoom}')

    return common.ProjZoom(
        xy=[x_zoom, y_zoom],
        xz=[x_zoom, z_zoom],
        yz=[y_zoom, z_zoom])

def classifier(observation, model, le, min_proba=0.7):
    """ Perform classification on a single radar image. """

    # note: reshape(1,-1) converts 1D array into 2D
    preds = model.predict_proba(observation.reshape(1, -1))[0]
    j = np.argmax(preds)
    proba = preds[j]
    logger.debug('classifier proba {} name {}'.format(proba, le.classes_[j]))

    if proba >= min_proba:
        name = le.classes_[j]
    else:
        name = 'Unknown'

    return name, proba

def predict(min_proba, model, le, proj_mask):
    # Calculate size of radar image data array used for training. 
    train_size_z = int((common.R_MAX - common.R_MIN) / common.R_RES) + 1
    train_size_y = int((common.PHI_MAX - common.PHI_MIN) / common.PHI_RES) + 1
    train_size_x = int((common.THETA_MAX - common.THETA_MIN) / common.THETA_RES) + 1
    logger.debug(f'train_size: {train_size_x}, {train_size_y}, {train_size_z}')

    try:
        while True:
            # Scan according to profile and record targets.
            radar.Trigger()

            # Retrieve any targets from the last recording.
            targets = radar.GetSensorTargets()
            if not targets:
                continue

            # Retrieve the last completed triggered recording
            raw_image, size_x, size_y, size_z, _ = radar.GetRawImage()
            raw_image_np = np.array(raw_image, dtype=np.float32)

            for t, target in enumerate(targets):
                logger.info('**********')
                logger.info('Target #{}:\nx: {}\ny: {}\nz: {}\namplitude: {}\n'.format(
                    t + 1, target.xPosCm, target.yPosCm, target.zPosCm, target.amplitude))

                i, j, k = common.calculate_matrix_indices(
                    target.xPosCm, target.yPosCm, target.zPosCm,
                    size_x, size_y, size_z)

                # projection_yz is the 2D projection of target in y-z plane.
                projection_yz = raw_image_np[i,:,:]
                # projection_xz is the 2D projection of target in x-z plane.
                projection_xz = raw_image_np[:,j,:]
                # projection_xy is 2D projection of target signal in x-y plane.
                projection_xy = raw_image_np[:,:,k]

                proj_zoom = calc_proj_zoom(train_size_x, train_size_y, train_size_z,
                    size_x, size_y, size_z)

                observation = common.process_samples(
                    [(projection_xz, projection_yz, projection_xy)],
                    proj_mask=proj_mask,
                    proj_zoom=proj_zoom,
                    scale=True)

                # Make a prediction. 
                name, prob = classifier(observation, model, le, min_proba)
                logger.info(f'Detected {name} with probability {prob}')
                logger.info('**********')
    except KeyboardInterrupt:
        pass
    finally:
        # Stop and Disconnect.
        radar.Stop()
        radar.Disconnect()
        radar.Clean()
        logger.info('Successful radar shutdown.')

    return

if __name__ == '__main__':
    # SVM model name.
    default_svm_model = 'train-results/svm_radar_classifier.pickle'
    # Label encoder name.
    default_label_encoder = 'train-results/radar_labels.pickle'
    # Radar 2-D projections to use for predictions (xy, xz, yz).
    default_proj_mask = [True, True, True]

    parser = argparse.ArgumentParser()
    parser.add_argument('--logging_level', type=str,
        help='logging level, "info" or "debug"',
        default='info')
    parser.add_argument('--min_proba', type=float,
        help='minimum prediction probability',
        default=0.7)
    parser.add_argument('--svm_model', type=str,
        help='path of output svm model name',
        default=os.path.join(common.PRJ_DIR, default_svm_model))
    parser.add_argument('--label_encoder', type=str,
        help='path of output label encoder',
        default=os.path.join(common.PRJ_DIR, default_label_encoder))
    parser.add_argument('--proj_mask', nargs='+', type=bool,
        help='projection mask (xy, xz, yz)',
        default=default_proj_mask)
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        level=logging.DEBUG if args.logging_level=='debug' else logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(common.PRJ_DIR, LOG_FILE), mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    radar.Init()

    # Configure Walabot database install location.
    radar.SetSettingsFolder()

    # Establish communication with walabot.
    try:
        radar.ConnectAny()
    except radar.WalabotError as err:
        logger.error(f'Failed to connect to Walabot.\nerror code: {str(err.code)}')
        sys.exit(1)

    api_version = radar.GetVersion()
    logger.info(f'Walabot api version: {api_version}')

    # Set radar scan profile.
    radar.SetProfile(common.RADAR_PROFILE)

    # Set scan arena in polar coords
    radar.SetArenaR(R_MIN, R_MAX, R_RES)
    radar.SetArenaPhi(PHI_MIN, PHI_MAX, PHI_RES)
    radar.SetArenaTheta(THETA_MIN, THETA_MAX, THETA_RES)
    r_min, r_max, _ = radar.GetArenaR()
    logger.info(f'predict r min: {r_min}, r max: {r_max} (cm)')
    phi_min, phi_max, _ = radar.GetArenaPhi()
    logger.info(f'predict phi min: {phi_min}, phi max: {phi_max} (deg)')
    theta_min, theta_max, _ = radar.GetArenaTheta()
    logger.info(f'predict theta min: {theta_min}, theta max: {theta_max} (deg)')

    logger.info(f'train r min: {common.R_MIN}, r max: {common.R_MAX} (cm)')
    logger.info(f'train phi min: {common.PHI_MIN}, phi max: {common.PHI_MAX} (deg)')
    logger.info(f'train theta min: {common.THETA_MIN}, theta max: {common.THETA_MAX} (deg)')

    # Threshold
    radar.SetThreshold(RADAR_THRESHOLD)

    # radar filtering
    filter_type = radar.FILTER_TYPE_MTI if MTI else radar.FILTER_TYPE_NONE
    radar.SetDynamicImageFilter(filter_type)

    # Start the system in preparation for scanning.
    radar.Start()

    # Calibrate scanning to ignore or reduce the signals if not in MTI mode.
    if not MTI:
        common.calibrate()

    frame_rate = radar.GetAdvancedParameter('FrameRate')
    logger.info(f'radar frame rate: {frame_rate}')

    # Load classifier along with the label encoder.
    with open(args.svm_model, 'rb') as fp:
        model = pickle.load(fp)
    with open(args.label_encoder, 'rb') as fp:
        le = pickle.load(fp)

    proj_mask=common.ProjMask(*args.proj_mask)
    logger.info(f'Projection mask: {proj_mask}')

    predict(min_proba=args.min_proba, model=model, le=le, proj_mask=proj_mask)