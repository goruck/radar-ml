"""
Ground truth radar samples with objects from detection server.

Copyright (c) 2020 Lindo St. Angel.
"""

import WalabotAPI as radar
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import animation
from collections import namedtuple
import common
import numpy as np
import pickle
import logging
import grpc
import detection_server_pb2
import detection_server_pb2_grpc
import os.path
import argparse

logger = logging.getLogger(__name__)

# grpc detection server address.
#DETECT_SERVER_IP = '192.168.1.131:50051'
DETECT_SERVER_IP = '10.0.0.20:50051'

# If radar unit is placed with usb facing right then set True.
# Else set false if radar is placed with usb facing bottom.
# See https://api.walabot.com/_features.html#_coordination.
RADAR_HORIZONTAL = True
# Radar detection threshold. 
RADAR_THRESHOLD = 5
# Set to True if using Moving Target Identification (MTI) filter.
MTI = True

# Offset between camera and radar physical centers in cm.
CAMERA_X_OFFSET = 1.13
CAMERA_Y_OFFSET = 5.08
CAMERA_Z_OFFSET = -1.2

# Threshold for match between radar and camera detected object.
# Defined as a percentage of radar target depth. 
DETECTION_THRESHOLD_PERCENT = 0.25
# Threshold for match between radar and camera detected object. In (cm).
#MAX_TARGET_OBJECT_DISTANCE = 20.0

# Minimum detection score to qualify as ground truth. 
MIN_DETECTED_OBJECT_SCORE = 0.50

LOG_FILE = 'ground_truth_samples.log'

Centroid = namedtuple('Centroid', ['x', 'y'])

DetectedObject = namedtuple(
    'DetectedObject', ['label', 'score', 'area', 'centroid']
)

def compute_distance(p0, p1):
    """ Compute distance between two points. """
    x0, y0 = p0
    x1, y1 = p1

    return np.sqrt(np.power(x1 - x0, 2) + np.power(y1 - y0, 2))

def convert_coordinates(camera_point, target_z, fx, fy, cx, cy):
    """ Convert a point from camera to radar coordinates.

    Radar cordinate system is used as the reference.

    Assumes camera z optical axis is aligned with radar z-axis.

    Camera coordinate limits (OpenCV style):
        top, left is at (x, y) = (0, 0)
        bottom, right is at (x, y) = (w, h)
        top, right is at (x, y) = (w, 0)
        bottom, left is at (x, y) = (0, h)
    Where (w, h) is the camera's resolution in pixels.

    Radar coordinates:
        Center is at (x, y, z) = (0, 0, 0)
        Limits of (x, y, z) are set by the Arena.

    Args:
        camera_point (float, float): (x, y) coordinate of point to convert in pixels.
        target_z (float): z of detected target from radar in cm.
        fx, fx (float, float): calibrated camera x and y focal points in pixels.
        cx, cy (float, float): calibrated x and y of camera's principal point in pixels.

    Returns:
        (radar_x, radar_y) (float, float): camera point in radar coord system. Units cm.
    """

    # Point in camera coordinate system.
    cam_x, cam_y = camera_point
    
    # Same point in world coordinates. 
    world_x = (cam_x - cx) * (target_z - CAMERA_Z_OFFSET) / fx
    world_y = (cam_y - cy) * (target_z - CAMERA_Z_OFFSET) / fy

    # Rotate and translate to convert point from world to radar coordinates.
    if RADAR_HORIZONTAL:
        radar_x = world_y - CAMERA_Y_OFFSET
        radar_y = world_x - CAMERA_X_OFFSET
    else:
        radar_x = world_x - CAMERA_X_OFFSET
        radar_y = -world_y - CAMERA_Y_OFFSET

    return (radar_x, radar_y)

def get_camera_resolution(stub):
    """ Get camera resolution from gprc detection server. """
    request = detection_server_pb2.Empty()
    try:
        response = stub.GetCameraResolution(request)
        return response
    except grpc.RpcError as err:
        logger.error(err.details()) #pylint: disable=no-member
        logger.error('{}, {}'.format(err.code().name, err.code().value)) #pylint: disable=no-member
        exit(1)

def get_camera_intrinsic_parameters(stub):
    """ Get camera intrinsic params from gprc detection server. """
    request = detection_server_pb2.Empty()
    try:
        response = stub.GetCameraIntrinsicParameters(request)
        return response
    except grpc.RpcError as err:
        logger.error(err.details()) #pylint: disable=no-member
        logger.error('{}, {}'.format(err.code().name, err.code().value)) #pylint: disable=no-member
        exit(1)

def get_detected_objects(stub, desired_labels):
    """ Get detected objects from gprc detection server. """
    request = detection_server_pb2.DesiredLabels(labels=desired_labels)
    try:
        response = stub.GetDetectedObjects(request)
    except grpc.RpcError as err:
        logger.error(f'grpc error: {err.details()}') #pylint: disable=no-member
        logger.error('{}, {}'.format(err.code().name, err.code().value)) #pylint: disable=no-member
        exit(1)

    # Find valid detected objects in the response from the grpc server.
    # An invalid object is one with label == '' which the server will
    #   send if its stack is empty. This serves as a kind of flow control.
    # Note: ListFields returns a list of (FieldDescriptor, value) tuples
    #   for all fields in the protobuf message which are not empty.
    def make(obj):
        return DetectedObject(
            label = obj.label,
            score = obj.score,
            area = obj.area,
            centroid = Centroid(
                x = obj.centroid.x,
                y = obj.centroid.y
            )
        )
    return [make(obj) for (fd,v) in response.ListFields() for obj in v if obj.label != '']

def plot_and_capture_data(num_samples, realtime_plot, save_plot, save_plot_path, desired_labels):
    def pol_2_cart_deg(a, r):
        ''' Convert polar coordinates, in degrees, to cartesian. '''
        a_rad = np.deg2rad(a)
        return (r * np.sin(a_rad), r * np.cos(a_rad))

    def gen_pos_map():
        ''' Create position coordinates map for plotting. '''
        arr_r = list(range(common.R_MIN, common.R_MAX, common.R_RES)) + [common.R_MAX]
        arr_t = list(range(common.THETA_MIN, common.THETA_MAX, common.THETA_RES)) + [common.THETA_MAX]
        arr_p = list(range(common.PHI_MIN, common.PHI_MAX, common.PHI_RES)) + [common.PHI_MAX]

        # Format of pmap_xz is [[list of x],[list of z],[list of dot size]].
        # Used to plot points on the XZ plane. 
        pmap_xz = np.array([list(pol_2_cart_deg(p, ra)) + [ra * 0.75] for ra in arr_r for p in arr_p]).T

        # Format of pmap_yz is [[list of y],[list of z],[list of dot size]].
        # Used to plot points on the YZ plane.
        pmap_yz = np.array([list(pol_2_cart_deg(t, ra)) + [ra * 0.75] for ra in arr_r for t in arr_t]).T

        return pmap_yz, pmap_xz

    def init_position_markers(ax):
        """Initialize position markers and annotations."""
        target_pt, = ax.plot(0, 0, 'ro', zorder=2)
        target_ant = ax.annotate('target', xy=(0,0), color='red', zorder=2)
        centroid_pt, = ax.plot(0, 0, 'go', zorder=3)
        centroid_ant = ax.annotate('', xy=(0,0), color='green', zorder=3)
        return (target_pt, target_ant, centroid_pt, centroid_ant)

    def init_axis(ax, title, xlabel, ylabel):
        """Initialize axis labels."""
        face_color = ScalarMappable(cmap='coolwarm').to_rgba(0)
        ax.set_title(title)
        ax.set_facecolor(face_color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def update_plot(data):
        '''Update plot for animation.'''
        projections, name, target_position, centroid_position = data
        projection_xz, projection_yz, projection_xy = projections
        target_x, target_y, target_z = target_position
        centroid_x, centroid_y = centroid_position

        # Update target position and annotation from radar on plots. 
        target_ant_xz.set_position(xy=(target_x, target_z))
        target_pt_xz.set_data(target_x, target_z)

        target_ant_yz.set_position(xy=(target_y, target_z))
        target_pt_yz.set_data(target_y, target_z)

        target_ant_xy.set_position(xy=(target_x, target_y))
        target_pt_xy.set_data(target_x, target_y)

        # Update name and postion annotations of centroid from camera on plots
        centroid_ant_xz.set_text(s=name)
        centroid_ant_xz.set_position(xy=(centroid_x, target_z))
        centroid_pt_xz.set_data(centroid_x, target_z)

        centroid_ant_yz.set_text(s=name)
        centroid_ant_yz.set_position(xy=(centroid_y, target_z))
        centroid_pt_yz.set_data(centroid_y, target_z)

        centroid_ant_xy.set_text(s=name)
        centroid_ant_xy.set_position(xy=(centroid_x, centroid_y))
        centroid_pt_xy.set_data(centroid_x, centroid_y)

        # Update image colors according to return signal strength on plots.
        sm = ScalarMappable(cmap='coolwarm')
        signal_pts_xz.set_color(sm.to_rgba(projection_xz.T.flatten()))

        sm = ScalarMappable(cmap='coolwarm')
        signal_pts_yz.set_color(sm.to_rgba(projection_yz.T.flatten()))

        # Scale xy image data relative to target distance. 
        signal_pts_xy.set_extent([v*target_z/(zmax-zmin) for v in [xmin,xmax,ymin,ymax]])
        # Rotate xy image if radar horizontal since x and y axis are rotated 90 deg CCW.
        if RADAR_HORIZONTAL:
            projection_xy = np.rot90(projection_xy)

        sm = ScalarMappable(cmap='coolwarm')
        signal_pts_xy.set_data(sm.to_rgba(projection_xy))

        return (signal_pts_xz, target_ant_xz, target_pt_xz, centroid_ant_xz, centroid_pt_xz,
            signal_pts_yz, target_ant_yz, target_pt_yz, centroid_ant_yz, centroid_pt_yz,
            signal_pts_xy, target_ant_xy, target_pt_xy, centroid_ant_xy, centroid_pt_xy)

    if realtime_plot or save_plot:
        if save_plot:
            # Set up formatting for movie files.
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='lindo'), bitrate=1800)

        # Get position maps. 
        pmap_xz, pmap_yz  = gen_pos_map()

        # Initial scan.
        radar.Trigger()
        raw_image, _, _, _, _ = radar.GetRawImage()
        raw_image_np = np.array(raw_image, dtype=np.float32)
        # projection_yz is the 2D projection of target in y-z plane.
        # Transpose to match ordering of position map.
        projection_yz = raw_image_np[0,:,:].transpose().flatten()
        # projection_xz is the 2D projection of target in x-z plane.
        projection_xz = raw_image_np[:,0,:].transpose().flatten()

        #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig = plt.figure()
        fig.suptitle('Target Return Signal, Target Position, Object Position and ID')
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,:])

        # Setup x-z plane plot.
        # Radar target return signal strength and camera centroid in x-z plane.
        init_axis(ax1, 'X-Z Plane', 'X (cm)', 'Z (cm)')
        signal_pts_xz = ax1.scatter(pmap_xz[0], pmap_xz[1], s=pmap_xz[2],
            c=projection_xz, cmap='coolwarm', zorder=1)
        (target_pt_xz, target_ant_xz, centroid_pt_xz,
            centroid_ant_xz) = init_position_markers(ax1)

        # Setup y-z plane plot.
        # Radar target return signal strength and camera centroid in y-z plane.
        init_axis(ax2, 'Y-Z Plane', 'Y (cm)', 'Z (cm)')
        signal_pts_yz = ax2.scatter(pmap_yz[0], pmap_yz[1], s=pmap_yz[2],
            c=projection_yz, cmap='coolwarm', zorder=1)
        (target_pt_yz, target_ant_yz, centroid_pt_yz,
            centroid_ant_yz) = init_position_markers(ax2)

        # Setup x-y plane plot.
        # Radar target return signal strength and camera centroid in x-z plane.
        # NB: When radar is placed horizontally, a rotated image will be shown.
        init_axis(ax3, 'X-Y Plane', 'X (cm)', 'Y (cm)')

        # Calculate axis range to set axis limits and plot extent.
        # Plot extent will change as a function of target distance.
        xmax = np.amax(pmap_xz[0]).astype(np.int)
        xmin = np.amin(pmap_xz[0]).astype(np.int)
        ymax = np.amax(pmap_yz[0]).astype(np.int)
        ymin = np.amin(pmap_yz[0]).astype(np.int)
        zmax = np.amax(pmap_yz[1]).astype(np.int)
        zmin = np.amin(pmap_yz[1]).astype(np.int)

        ax3.set_xlim(xmax, xmin)
        ax3.set_ylim(ymax, ymin)
        init_img = np.zeros((xmax-xmin, ymax-ymin))
        signal_pts_xy = ax3.imshow(init_img, cmap='coolwarm',
            extent=[xmin,xmax,ymin,ymax], zorder=1)
        (target_pt_xy, target_ant_xy, centroid_pt_xy,
            centroid_ant_xy) = init_position_markers(ax3)

    # Initialize ground truth data.
    samples = []
    labels = []

    with grpc.insecure_channel(DETECT_SERVER_IP) as channel:
        stub = detection_server_pb2_grpc.DetectionServerStub(channel)

        res = get_camera_resolution(stub)
        width, height = res.width, res.height
        logger.info(f'camera resolution: {width, height}')

        res = get_camera_intrinsic_parameters(stub)
        fx, fy, cx, cy = res.fx, res.fy, res.cx, res.cy
        logger.debug(f'camera intrinsics fx: {fx} fy:{fy} cx:{cx} cy:{cy}')

        # Calculate camera field (aka angle) of view from intrinsics.
        fov_hor = 2 * np.arctan(width / (2 * fx)) * 180.0 / np.pi
        fov_ver = 2 * np.arctan(height / (2 * fy)) * 180.0 / np.pi
        logger.info(f'camera hor fov: {fov_hor:.1f} (deg) ver fov: {fov_ver:.1f} (deg)')

        def get_samples():
            active = True
            sample_num = 1

            while active:
                # Scan according to profile and record targets (if any).
                radar.Trigger()

                # Get object detection results from server (if any). 
                detected_objects = get_detected_objects(stub, desired_labels)
                if not detected_objects:
                    continue

                # Retrieve any targets from the last recording.
                #targets = radar.GetTrackerTargets()
                targets = radar.GetSensorTargets()
                if not targets:
                    continue

                # raw_image ordering: (theta, phi, r)
                raw_image, size_x, size_y, size_z, _ = radar.GetRawImage()
                raw_image_np = np.array(raw_image, dtype=np.float32)
                logger.debug(f'Raw image np shape: {raw_image_np.shape}')

                #targets = get_derived_targets(raw_image_np, size_x, size_y, size_z)

                logger.info(f'Sample number {sample_num} of {num_samples}'.center(60, '-'))
                # Find the detected object closest to each radar target.
                for t, target in enumerate(targets):
                    logger.info(f'Target #{t + 1}:')
                    logger.debug('\nx: {}\ny: {}\nz: {}\namplitude: {}\n'.format(
                        target.xPosCm, target.yPosCm, target.zPosCm, target.amplitude))

                    i, j, k = common.calculate_matrix_indices(
                        target.xPosCm, target.yPosCm, target.zPosCm,
                        size_x, size_y, size_z)
                    logger.debug(f'i: {i}, j: {j}, k: {k}')

                    # Init distance between radar and camera target as a % of radar target depth.
                    # This is used as a threshold to declare correspondence. 
                    current_distance = DETECTION_THRESHOLD_PERCENT * target.zPosCm
                    #current_distance = MAX_TARGET_OBJECT_DISTANCE
                    logger.debug(f'Initial threshold: {current_distance:.1f} (cm)')

                    target_object_close = False

                    for obj in detected_objects:
                        if obj.score < MIN_DETECTED_OBJECT_SCORE:
                            logger.debug(f'Object ({obj.label}) score ({obj.score:.1f}) too low...skipping.')
                            continue

                        # Convert position of detected object's centroid to radar coordinate system.
                        centroid_camera = (width*obj.centroid.x, height*obj.centroid.y)
                        logger.debug(f'Centroid camera: {centroid_camera}')
                        centroid_radar = convert_coordinates(centroid_camera,
                            target.zPosCm, fx, fy, cx, cy)
                        logger.debug(f'Centroid radar: {centroid_radar}')

                        # Calculate distance between detected object and radar target. 
                        distance = compute_distance((target.xPosCm, target.yPosCm), centroid_radar)
                        logger.debug(f'Distance: {distance}')

                        # Find the detected object closest to the radar target.
                        if distance < current_distance:
                            target_object_close = True
                            current_distance = distance
                            current_score = obj.score
                            target_name = obj.label
                            target_position = target.xPosCm, target.yPosCm, target.zPosCm
                            centroid_position = centroid_radar
                            # Calculate 3D to 2D projections of target return signal.
                            # Signal in raw_image_np with shape (size_x, size_y, size_z).
                            # axis 1 and 2 of the matrix (j, k) contain the projections
                            #   represented in angle phi and distance r, respectively.
                            #   These are 2D projections the y-z plane.
                            # axis 0 and 2 of the matrix (i, k) contain the projections
                            #   represented in angle theta and distance r, respectively.
                            #   These are 2D projections the x-z plane.
                            #
                            # projection_yz is the 2D projection of target in y-z plane.
                            projection_yz = raw_image_np[i,:,:]
                            logger.debug(f'Projection_yz shape: {projection_yz.shape}')
                            # projection_xz is the 2D projection of target in x-z plane.
                            projection_xz = raw_image_np[:,j,:]
                            logger.debug(f'Projection_xz shape: {projection_xz.shape}')
                            # projection_xy is 2D projection of target signal in x-y plane.
                            projection_xy = raw_image_np[:,:,k]
                            logger.debug(f'Projection_xy shape: {projection_xy.shape}')
                        else:
                            msg = (
                                f'Found "{obj.label}" with score {obj.score:.1f} at {distance:.1f} (cm)'
                                f' too far from target at z {target.zPosCm:.1f} (cm)...skipping.'
                            )
                            logger.info(msg)

                    if target_object_close:
                        msg = (
                            f'Found "{target_name}" with score {current_score:.1f} at {distance:.1f} (cm)'
                            f' from target at z {target.zPosCm:.1f} (cm)...storing.'
                        )
                        logger.info(msg)

                        yield ((projection_xz, projection_yz, projection_xy),
                            target_name, target_position, centroid_position)

                logger.info('-'*60+'\n')

                if sample_num < num_samples:
                    sample_num += 1
                else:
                    active = False
                    if realtime_plot:
                        print('\n**** Close plot window to continue. ****\n')

        if realtime_plot or save_plot:
            # Animate but do not save data.
            ani = animation.FuncAnimation(fig, update_plot, frames=get_samples,
                repeat=False, interval=100, blit=True)

            try:
                if realtime_plot:
                    plt.show()
                elif save_plot:
                    ani.save(save_plot_path, writer=writer)
            except Exception as e:
                print(f'Unhandled animation exception: {e}')
                pass
        else:
            # Save data but do not animate.
            for data in get_samples():
                projections, target_name, _, _ = data
                samples.append(projections)
                labels.append(target_name)

    return samples, labels

if __name__ == '__main__':
    # Desired labels from detection server.
    # These must be all or a subset of the class labels. 
    default_desired_labels = ['person', 'dog', 'cat']

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int,
        help='number of samples to capture',
        default=500)
    parser.add_argument('--desired_labels', nargs='+', type=str,
        help='Labels to use from detection server.',
        default=default_desired_labels)
    parser.add_argument('--realtime_plot', action='store_true',
        help='plot radar results in real-time')
    parser.add_argument('--save_plot', action='store_true',
        help='save radar realtime plot as movie')
    parser.add_argument('--save_plot_path', type=str,
        help='radar plot movie file name',
        default=os.path.join(common.PRJ_DIR, 'ground-truth-samples.mp4'))
    parser.add_argument('--logging_level', type=str,
        help='logging level, "info" or "debug"',
        default='info')
    parser.set_defaults(realtime_plot=False)
    parser.set_defaults(save_plot=False)
    args = parser.parse_args()

    logging.basicConfig(filename=os.path.join(common.PRJ_DIR, LOG_FILE),
        filemode='w',
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        level=logging.DEBUG if args.logging_level=='debug' else logging.INFO)

    radar.Init()

    # Configure radar database install location.
    radar.SetSettingsFolder()

    # Establish communication with radar.
    try:
        radar.ConnectAny()
    except radar.WalabotError as err:
        logger.error(f'Failed to connect to radar.\nerror code: {str(err.code)}')
        exit(1)

    api_version = radar.GetVersion()
    logger.info(f'Walabot api version: {api_version}')

    # Set Profile.
    radar.SetProfile(common.RADAR_PROFILE)

    # Set scan arena coordinates. 
    radar.SetArenaR(common.R_MIN, common.R_MAX, common.R_RES)
    radar.SetArenaPhi(common.PHI_MIN, common.PHI_MAX, common.PHI_RES)
    radar.SetArenaTheta(common.THETA_MIN, common.THETA_MAX, common.THETA_RES)
    r_min, r_max, _ = radar.GetArenaR()
    logger.info(f'radar r min: {r_min}, r max: {r_max} (cm)')
    phi_min, phi_max, _ = radar.GetArenaPhi()
    logger.info(f'radar phi min: {phi_min}, phi max: {phi_max} (deg)')
    theta_min, theta_max, _ = radar.GetArenaTheta()
    logger.info(f'radar theta min: {theta_min}, theta max: {theta_max} (deg)')

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

    logger.info(f'desired labels: {args.desired_labels}')

    samples, labels = plot_and_capture_data(args.num_samples, args.realtime_plot,
        args.save_plot, args.save_plot_path, args.desired_labels)

    # Append data file if it already exists, else create a new one.
    if samples and labels:
        logger.info(f'Captured {len(labels)} new samples with label(s) {set(labels)}.')
        try:
            with open(os.path.join(common.PRJ_DIR, common.RADAR_DATA), 'rb') as fp:
                data = pickle.load(fp)

            msg = (
                f'Appending existing data file with new samples.'
                f' Existing data file has {len(data["labels"])} samples'
                f' with label(s) {set(data["labels"])}.'
            )
            logger.info(msg)
            data['samples'].extend(samples)
            data['labels'] += labels
        except (ValueError, AttributeError) as e:
            logger.error(f'Got error "{e}"" while trying to append data file, exiting.')
            exit(1)
        except FileNotFoundError:
            logger.info('Existing data file not found, creating.')

        # Write data to disc. 
        logger.debug(f'Data dump:\n{data}')
        with open(os.path.join(common.PRJ_DIR, common.RADAR_DATA), 'wb') as fp:
            logger.info('Saving data file.')
            pickle.dump(data, fp)
    else:
        logger.info('No data was captured.')

    # Stop and Disconnect radar.
    radar.Stop()
    radar.Disconnect()
    logger.info('Successful radar shutdown.')