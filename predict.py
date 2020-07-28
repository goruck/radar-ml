"""
Predict objects from a radar waveform.

Copyright (c) 2020 Lindo St. Angel.
"""

import WalabotAPI as radar
import numpy as np
import pickle
import common
from sklearn.svm import SVC
from os import path
from sys import exit

# Radar detection threshold. 
RADAR_THRESHOLD = 5
# Set to True if using Moving Target Identification (MTI) filter.
MTI = True

# Radar 2-D projections to use for predictions.
PROJ_MASK = common.ProjMask(xy=True, xz=True, yz=True)

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
    #print(f'zoom: {x_zoom}, {y_zoom}, {z_zoom}')

    return common.ProjZoom(
        xy=[x_zoom, y_zoom],
        xz=[x_zoom, z_zoom],
        yz=[y_zoom, z_zoom])

def classifier(observation, model, le, min_proba=0.98):
    """ Perform classification on a single radar image. """

    # note: reshape(1,-1) converts 1D array into 2D
    preds = model.predict_proba(observation.reshape(1, -1))[0]
    j = np.argmax(preds)
    proba = preds[j]
    #print('classifier proba {} name {}'.format(proba, le.classes_[j]))

    if proba >= min_proba:
        name = le.classes_[j]
    else:
        name = 'Unknown'

    return name, proba

def main():
    # Load classifier along with the label encoder.
    with open(path.join(common.PRJ_DIR, common.SVM_MODEL), 'rb') as fp:
        model = pickle.load(fp)
    with open(path.join(common.PRJ_DIR, common.LABELS), 'rb') as fp:
        le = pickle.load(fp)

    radar.Init()

    # Configure Walabot database install location.
    radar.SetSettingsFolder()

    # Establish communication with walabot.
    try:
        radar.ConnectAny()
    except radar.WalabotError as err:
        print(f'Failed to connect to Walabot.\nerror code: {str(err.code)}')
        exit(1)

    # Set radar scan profile.
    radar.SetProfile(common.RADAR_PROFILE)

    # Set scan arena in polar coords
    # This can be different than arena used for training. 
    R_MIN, R_MAX, R_RES = 15, 331, 4
    THETA_MIN, THETA_MAX, THETA_RES = -40, 35, 8
    PHI_MIN, PHI_MAX, PHI_RES = -27, 31, 4
    radar.SetArenaR(R_MIN, R_MAX, R_RES)
    radar.SetArenaPhi(PHI_MIN, PHI_MAX, PHI_RES)
    radar.SetArenaTheta(THETA_MIN, THETA_MAX, THETA_RES)

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

    # Calculate size of radar image data array used for training. 
    train_size_z = int((common.R_MAX - common.R_MIN) / common.R_RES) + 1
    train_size_y = int((common.PHI_MAX - common.PHI_MIN) / common.PHI_RES) + 1
    train_size_x = int((common.THETA_MAX - common.THETA_MIN) / common.THETA_RES) + 1
    #print(f'train_size: {train_size_x}, {train_size_y}, {train_size_z}')

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
                print('Target #{}:\nx: {}\ny: {}\nz: {}\namplitude: {}\n'.format(
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
                    proj_mask=PROJ_MASK,
                    proj_zoom=proj_zoom)

                # Make a prediction. 
                name, prob = classifier(observation, model, le)
                print(f'Detected {name} with probability {prob}\n')
    except KeyboardInterrupt:
        pass
    finally:
        # Stop and Disconnect.
        radar.Stop()
        radar.Disconnect()
        radar.Clean()
        print('Successful termination.')

if __name__ == '__main__':
    main()