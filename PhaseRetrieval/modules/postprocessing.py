"""
This module contains functions for "post-processing" of phase-retrieved images
"""

import numpy as np
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
import os
from PIL import Image

def phase_alignment_gerchberg_saxton(amplitude_filename = None,
                                     phase_filenames = None,
                                     num_files_to_align = None,
                                     delimiter = '\t',
                                     ref_coordinates=None,
                                     symmetric_phase = True,
                                     plot_progress=True,
                                     plot_every_kth_iteration=1,
                                     zoom=1):
    """
    Alignment of phase images yielded by Gerchberg-Saxton algorithm (in csv file format) (of either pbject-domain or Fourier domain phase images).
    It is assumed that the corresponding amplitude is known and the phases are all in spatial registry.
    ---
    Parameters
    ---
    amplitude_filename : list
        Path to file containing amplitude distribution
        Default is None
    phase_filenames : list
        List of paths to files with phase distributions to be aligned
        Default is None
    delimiter : str, optional
        Delimiter in csv files
        Default is '\t'
    ref_coordinates: list [1x2], optional
        Coordinates of a reference phase value.
        If set to None, the ref_coordinates will be set the image's centre coordinates (i.e. [501,501] for 1000x1000 pixels image)
        If a list is provided, the values will be taken from there.
    symmetric_phase ; bool, optional
        If set to True, pixel values in the final phase distribution will be shifted symmetrically w.r.t zero
        If False, pixels values in the final phase distribution will be left as they are after alignment
        Default is True
    plot_progress : bool, optional
        False will prevent algorithm from plotting the progress.
        True will plot the progress of the algorithm.
        Default is False.
    plot_every_kth_iteration : int, optional
        Plot the progress each k-th iteration, where k=plot_every_kth_iteration
        Default is 1.
    zoom: int, optional
        Zoom factor to zoom into the 2D plot
        Default is 1 (no zoom)
    """

    #read amplitude image
    if os.path.exists(amplitude_filename):
        # read amplitude image
        if amplitude_filename[-4:] == '.tif':
            #amplitude = io.imread(amplitude_filename)
            amplitude = np.asarray(Image.open(amplitude_filename))
        elif amplitude_filename[-4:] == '.csv':
            amplitude = pd.read_csv(amplitude_filename, delimiter='\t', header=None).values
        else:
            raise ValueError("Data must be either in tif or csv format.")
    else:
        raise ValueError("Path to an amplitude file does not exist")
    #
    for phase_idx, phase_file in enumerate(sorted(phase_filenames)):
        #
        # read phase images
        phase = pd.read_csv(phase_file, delimiter=delimiter, header=None)
        phase = phase.values
        #
        #determine image size and set the reference pixel to one located in the middle of the image
        if ref_coordinates == None:
            ref_coordinates = [0,0]
            ref_coordinates[0] = int(phase.shape[0]) // 2 + 1
            ref_coordinates[1] = int(phase.shape[0]) // 2 + 1
            #
        #or use given values instead
        else:
            ref_coordinates = ref_coordinates
        #
        #
        #set the number of files to align
        if num_files_to_align is None:
            len_phase_filenames = len(phase_filenames)
        else:
            len_phase_filenames = num_files_to_align
        #
        if phase_idx == 0:
            # no aligned / reference phases at first
            reference_phase = np.zeros(phase.shape)
            #
            # let the first file be the "offset" phase
            offset_phase = np.copy(phase)
            #
            # aligned phases is the sum of reference and offset phases
            aligned_phases = reference_phase + offset_phase
        #
        elif phase_idx > 0 and phase_idx < len_phase_filenames - 1:
            # set the reference phase to be equal to the sum of already aligned phases
            reference_phase = np.copy(aligned_phases)
            #
            # treat the next file as the offset phase  which must be aligned
            offset_phase = np.copy(phase)
            #
            # get reference phase value
            ref_phase_value = np.copy(reference_phase[ref_coordinates[0], ref_coordinates[1]])
            # shift offset phase distributions depending on the sign of the reference phase value
            if ref_phase_value + np.abs(0.5 * ref_phase_value) < 0:
                offset_phase = np.abs(offset_phase + np.abs(ref_phase_value))
            elif ref_phase_value + np.abs(0.5 * ref_phase_value) > 0:
                offset_phase = np.abs(offset_phase - np.abs(ref_phase_value))
            else:
                offset_phase = np.abs(offset_phase)
            #
            # obtain the sum of aligned phases by adding phase-shifted offset phase to the reference phase
            aligned_phases = reference_phase + offset_phase
        #
        #compute the average of the aligned phases
        elif phase_idx > 0 and phase_idx == len_phase_filenames - 1:
            print("averaging ")
            aligned_phases = aligned_phases / len(phase_filenames) - int(symmetric_phase) * 0.5 * np.max(np.max(aligned_phases)) / len(phase_filenames)
            #
            # plot progress
        if plot_progress is True:
            if (phase_idx + 1) % plot_every_kth_iteration == 0:
                #
                print("Image %d \r" % (phase_idx + 1), " (is being aligned)")
                #
                # some manipulations with phase distribution to be able to plot it weighted with amplitude values
                # important to normalise to 1, otherwise the plot will not be displayed correctly!
                #
                offset_phase_weighted = plt.cm.bwr(
                    rescale_intensity(- offset_phase.min() + offset_phase, out_range=(0, 1)))
                offset_phase_weighted[..., -1] = rescale_intensity(amplitude, out_range=(0, 1))
                #
                aligned_phases_weighted = plt.cm.bwr(
                    rescale_intensity(- aligned_phases.min() + aligned_phases, out_range=(0, 1)))
                aligned_phases_weighted[..., -1] = rescale_intensity(amplitude, out_range=(0, 1))
                #
                fig, ax = plt.subplots(1, 2, figsize=(10, 20))
                ax = ax.ravel()
                theshape = offset_phase_weighted.shape
                #
                # current offset phase
                im00 = ax[0].imshow(offset_phase_weighted, cmap='seismic', vmin=offset_phase.min(),
                                    vmax=offset_phase.max())
                ax[0].set_title("offset phase")
                plt.colorbar(im00, ax=ax[0], fraction=0.046, pad=0.04)
                ax[0].axis([theshape[0] // 2 - theshape[0] // 2 // zoom,
                            theshape[0] // 2 + theshape[0] // 2 // zoom,
                            theshape[1] // 2 - theshape[1] // 2 // zoom,
                            theshape[1] // 2 + theshape[1] // 2 // zoom])
                #
                # computed Fourier phase
                im01 = ax[1].imshow(aligned_phases_weighted, cmap='seismic', vmin=aligned_phases.min(),
                                    vmax=aligned_phases.max())
                ax[1].set_title("aligned phase")
                plt.colorbar(im01, ax=ax[1], fraction=0.046, pad=0.04)
                ax[1].axis([theshape[0] // 2 - theshape[0] // 2 // zoom,
                            theshape[0] // 2 + theshape[0] // 2 // zoom,
                            theshape[1] // 2 - theshape[1] // 2 // zoom,
                            theshape[1] // 2 + theshape[1] // 2 // zoom])
                #
                fig.tight_layout()
                display.clear_output(wait=True)
                plt.show()
            else:
                print("Alignment of %d \r" % (phase_idx + 1) / len(phase_filenames),
                     " % of images completed")
        else:
            print("Alignment of %d \r" % (phase_idx + 1) / len(phase_filenames),
                  " % of images completed")
        #
    #save aligned images
    filename_full = amplitude_filename[:-4] + '_phase.csv'
    np.savetxt(filename_full, aligned_phases, delimiter='\t')
    print('Aligned phase distribution was saved as ', filename_full)
    #