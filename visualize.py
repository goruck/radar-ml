"""
Visualize radar samples.

'n' keypress advances sample; 'b' goes back' 'escape' exits.

Copyright (c) 2020 Lindo St. Angel
"""
import os
import pickle
import argparse

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np

import common

# If radar unit is placed with usb facing right then set True.
# Else set false if radar is placed with usb facing bottom.
# See https://api.walabot.com/_features.html#_coordination.
RADAR_HORIZONTAL = True

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
    """Initialize target name."""
    target_pt, = ax.plot(0, 0, 'ro', zorder=2)
    target_ant = ax.annotate('target', xy=(0,0), color='red', zorder=2)
    return target_pt, target_ant

def init_axis(ax, title, xlabel, ylabel):
    """Initialize axis labels."""
    face_color = ScalarMappable(cmap='coolwarm').to_rgba(0)
    ax.set_title(title)
    ax.set_facecolor(face_color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def visualize(samples, labels):
    """Main function.
    
    Note:
        First init plots then update per user keypress.
        Samples are in the form [(xz, yz, xy), ...] in range [0, common.RADAR_MAX].

    Args:
        samples (list of tuples of np.array): Data to visualize.
        labels (list of strings): Sample labels.
    """
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:])

    # Get position maps. 
    pmap_xz, pmap_yz  = gen_pos_map()

    idx = 0

    # Initial sample.
    xz, yz, xy = samples[idx]

    title = fig.suptitle(f'Target Return Signal. Label "{labels[idx]}", Sample {idx}.')

    # Setup x-z plane plot.
    # Radar target return signal strength in x-z plane.
    init_axis(ax1, 'X-Z Plane', 'X (cm)', 'Z (cm)')
    sm = ScalarMappable(cmap='coolwarm')
    init_c = sm.to_rgba(xz.T.flatten())
    pts_xz = ax1.scatter(pmap_xz[0], pmap_xz[1], s=pmap_xz[2],
        c=init_c, cmap='coolwarm', zorder=1)

    # Setup y-z plane plot.
    # Radar target return signal strength in y-z plane.
    init_axis(ax2, 'Y-Z Plane', 'Y (cm)', 'Z (cm)')
    sm = ScalarMappable(cmap='coolwarm')
    init_c = sm.to_rgba(yz.T.flatten())
    pts_yz = ax2.scatter(pmap_yz[0], pmap_yz[1], s=pmap_yz[2],
        c=init_c, cmap='coolwarm', zorder=1)

    # Setup x-y plane plot.
    # Radar target return signal strength in x-z plane.
    init_axis(ax3, 'X-Y Plane', 'X (cm)', 'Y (cm)')

    # Calculate axis range to set axis limits and plot extent.
    xmax = np.amax(pmap_xz[0]).astype(np.int)
    xmin = np.amin(pmap_xz[0]).astype(np.int)
    ymax = np.amax(pmap_yz[0]).astype(np.int)
    ymin = np.amin(pmap_yz[0]).astype(np.int)
    zmax = np.amax(pmap_yz[1]).astype(np.int)
    zmin = np.amin(pmap_yz[1]).astype(np.int)

    ax3.set_xlim(xmax, xmin)
    ax3.set_ylim(ymax, ymin)
    # Rotate xy image if radar horizontal since x and y axis are rotated 90 deg CCW.
    if RADAR_HORIZONTAL:
        xy = np.rot90(xy)
    sm = ScalarMappable(cmap='coolwarm')
    init_img = sm.to_rgba(xy)
    pts_xy = ax3.imshow(init_img, cmap='coolwarm',
        extent=[xmin,xmax,ymin,ymax], zorder=1)

    def update(event):
        """Update plot per keypress."""
        nonlocal idx

        if event.key == 'n': # next sample
            if idx >= len(samples) - 1:
                idx = len(samples) - 1
            else:
                idx+=1
        elif event.key == 'b': # prev sample
            if idx <= 0:
                idx = 0
            else:
                idx-=1
        elif event.key == 'escape': # all done
            plt.close()
            return

        # Get next sample.
        xz, yz, xy = samples[idx]

        # Update title.
        title.set_text(f'Target Return Signal. Label "{labels[idx]}", Sample {idx}.')

        # Update image colors according to return signal strength on plots.
        sm = ScalarMappable(cmap='coolwarm')
        pts_xz.set_color(sm.to_rgba(xz.T.flatten()))

        sm = ScalarMappable(cmap='coolwarm')
        pts_yz.set_color(sm.to_rgba(yz.T.flatten()))

        if RADAR_HORIZONTAL:
            xy = np.rot90(xy)
        sm = ScalarMappable(cmap='coolwarm')
        pts_xy.set_data(sm.to_rgba(xy))

        # Actual update of plot.
        plt.draw()

        return

    fig.canvas.mpl_connect('key_press_event', update)

    plt.show()

    return

if __name__ == '__main__':
    default_dataset = 'datasets/radar_samples_20Oct20.pickle'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
        help='dateset name to visualize',
        default=os.path.join(common.PRJ_DIR, default_dataset))
    args = parser.parse_args()

    try:
        with open(args.dataset, 'rb') as fp:
            data_pickle = pickle.load(fp)
    except FileNotFoundError as e:
        print(f'error: {e}')
        exit(1)

    samples = data_pickle['samples']
    labels = data_pickle['labels']

    visualize(samples, labels)