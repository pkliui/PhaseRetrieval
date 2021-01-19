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
from skimage import io
import time


def phase_alignment_gerchberg_saxton(amplitude_filename = None,
                                     delimiter=None,
                                     phase_filenames = None,
                                     ref_coordinates = None,
                                     num_files_to_align = None,
                                     symmetric_phase = True,
                                     plot_progress = True,
                                     plot_every_kth_iteration=1,
                                     zoom = 1):
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
        num_files_to_align : int
            Integer number of files with phase distributions to be aligned
            If set to None, all files will be used for alignment.
            Default is None.
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
    #
    #read amplitude image
    if os.path.exists(amplitude_filename):
        # read amplitude image
        if amplitude_filename[-4:] == '.tif':
            # make sure to convert intensities to amplitudes
            amplitude = np.sqrt(io.imread(amplitude_filename))
            #amplitude = np.asarray(Image.open(amplitude_filename))
        elif amplitude_filename[-4:] == '.csv':
            amplitude = np.sqrt(pd.read_csv(amplitude_filename, delimiter=delimiter, header=None).values)
        else:
            raise ValueError("Data must be either in tif or csv format.")
    else:
        raise ValueError("Path to an amplitude file does not exist")

    for phase_idx, phase_file in enumerate(sorted(phase_filenames)):
        #
        # read phase images
        if os.path.exists(phase_file):
            phase = pd.read_csv(phase_file, delimiter=delimiter, header=None, error_bad_lines=False).values
            print('image being aligned: ',phase_file)
            #
        else:
            raise ValueError("Path to an amplitude file does not exist")
        #
        # make sure the shape of the phase  image is equal to that of the amplitude image
        try:
            print(phase.shape, amplitude.shape)
            assert phase.shape == amplitude.shape
            print("phase shape = amplitude shape")
            #
            #compute reference coordinates
            if ref_coordinates == None: #applies to phase_idx==0 only
                # determine image size and set the reference pixel to the ones located in the middle of the image
                ref_coordinates = [0,0]
                ref_coordinates[0] = int(phase.shape[0]) // 2
                ref_coordinates[1] = int(phase.shape[0]) // 2
                #
            #applies to all phase_idx
            #if phase_idx==0 and ref_coordinates are not None, use values given by user / or if phase_idx>0, use  ref_coordinates computed at phase_idx==0 iteration
            else:
                ref_coordinates = ref_coordinates
            #
            # set the number of files to align either to total number of files or just a part of thereof
            if num_files_to_align is None:
                len_phase_filenames = len(phase_filenames)
            else:
                len_phase_filenames = num_files_to_align
            #
            #align phases
            if phase_idx == 0:
                #
                #set the reference phase value to 0 in the first 'aligned' phase distribution
                aligned_phase = np.copy(phase) - np.copy(phase[ref_coordinates[0], ref_coordinates[1]])
                offset_phase = np.zeros(aligned_phase.shape)
            #
            elif phase_idx > 0 and phase_idx < len_phase_filenames:
                #
                # treat the next phase distribution as the offset phase distribution which must be aligned
                offset_phase = np.copy(phase)
                #
                # get current reference phase value in the offset phase distribution
                ref_phase_value = np.copy(offset_phase[ref_coordinates[0], ref_coordinates[1]])
                #
                # shift offset phase distributions depending on the sign of the reference phase value
                if ref_phase_value + np.abs(0.5 * ref_phase_value) < 0:
                    offset_phase = offset_phase + np.abs(ref_phase_value)
                elif ref_phase_value + np.abs(0.5 * ref_phase_value) > 0:
                    offset_phase = offset_phase - np.abs(ref_phase_value)
                else:
                    offset_phase = offset_phase
                #
                # obtain the sum of aligned phases by adding phase-shifted offset phase to the reference phase
                aligned_phase = np.copy(aligned_phase) + np.copy(offset_phase)
            # finish alignment
            else:
                break
            #
            #plot
            if plot_progress is True:
                if (phase_idx + 1) % plot_every_kth_iteration == 0:
                    #
                    print("Image " + str(int(phase_idx + 1)) + " is being aligned")
                    #
                    # some manipulations with phase distribution to be able to plot it weighted with amplitude values
                    # important to normalise to 1, otherwise the plot will not be displayed correctly!
                    #
                    offset_phase_weighted = plt.cm.bwr(
                        rescale_intensity(- offset_phase.min() + offset_phase, out_range=(0, 1)))
                    offset_phase_weighted[..., -1] = rescale_intensity(amplitude, out_range=(0, 1))
                    #
                    aligned_phases_weighted = plt.cm.bwr(
                        rescale_intensity(- aligned_phase.min() + aligned_phase, out_range=(0, 1)))
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
                    ax[0].set_ylim(ax[0].get_ylim()[::-1])
                    #
                    # computed Fourier phase
                    im01 = ax[1].imshow(aligned_phases_weighted, cmap='seismic', vmin=aligned_phase.min(),
                                        vmax=aligned_phase.max())
                    ax[1].set_title("aligned phase")
                    plt.colorbar(im01, ax=ax[1], fraction=0.046, pad=0.04)
                    ax[1].axis([theshape[0] // 2 - theshape[0] // 2 // zoom,
                                theshape[0] // 2 + theshape[0] // 2 // zoom,
                                theshape[1] // 2 - theshape[1] // 2 // zoom,
                                theshape[1] // 2 + theshape[1] // 2 // zoom])
                    ax[1].set_ylim(ax[1].get_ylim()[::-1])
                    #
                    fig.tight_layout()
                    display.clear_output(wait=True)
                    plt.show()
                else:
                    print("Alignment of " + str(100 * int(phase_idx + 1) / num_files_to_align) + " % of images completed")
            else:
                print("Alignment of " + str(100 * int(phase_idx + 1) / num_files_to_align) + " % of images completed")
        #
        # skip the phase image if its shape is different from the shape of the amplitude image
        except AssertionError as e:
            print(e)
    #
    # compute the average of the aligned phases
    print("averaging ")
    aligned_phase_norm = aligned_phase / num_files_to_align - int(symmetric_phase) * 0.5 * np.max(np.max(aligned_phase)) / num_files_to_align
    #
    #save aligned images
    filename_full = amplitude_filename[:-14] + '_phase.csv'
    np.savetxt(filename_full, aligned_phase_norm, delimiter='\t', fmt='%1.2f')
    print('Aligned phase distribution was saved as ', filename_full)
    #
    return amplitude, phase, ref_coordinates, len_phase_filenames, aligned_phase, aligned_phase_norm

def plot_reconstruction(amplitude_filename = None,
                        phase_filename = None,
                        delimiter = ',',
                        zoom = 1,
                        save_as_eps = False):
    """
    Plot aligned amplitude and phase images yielded by phase_alignment_gerchberg_saxton method
    ---
    Parameters
    ---
    amplitude_filename : list
        Path to file containing amplitude distribution
        Default is None
    phase_filenames : list
        Path to file with aligned phase distributions
        Default is None
    delimiter : str, optional
        Delimiter in csv files
        Default is '\t'
    zoom: int, optional
        Zoom factor to zoom into the 2D plot
        Default is 1 (no zoom)
    save_as_eps: bool, optional
        If set to False, the images are only displayed
        If set to True, the displayed image is saved as eps file
        Default is False.
    """
    #
    # read amplitude image
    if os.path.exists(amplitude_filename):
        # read amplitude image
        if amplitude_filename[-4:] == '.tif':
            # convert intensity to amplitude
            amplitude0 = np.sqrt(io.imread(amplitude_filename))
            # amplitude0 = np.asarray(Image.open(amplitude_filename))
            amplitude = amplitude0 / np.max(amplitude0)
        elif amplitude_filename[-4:] == '.csv':
            # convert intensity to amplitude
            amplitude0 = np.sqrt(pd.read_csv(amplitude_filename, delimiter='\t', header=None).values)
            amplitude = amplitude0 / np.max(amplitude0)
        else:
            raise ValueError("Data must be either in tif or csv format.")
    else:
        raise ValueError("Path to an amplitude file does not exist")
    #
    #read phases
    if phase_filename[-4:] == ".csv":
        phase = pd.read_csv(phase_filename, delimiter=delimiter, header=None)
        phase = phase.values
    else:
        raise ValueError("Phase data must be in csv format!")
    #
    phase_weighted = plt.cm.bwr(
        rescale_intensity(- phase.min() + phase, out_range=(0, 1)))
    phase_weighted[..., -1] = rescale_intensity(amplitude, out_range=(0, 1))
    #
    fig, ax = plt.subplots(1, 2, figsize=(10, 20))
    ax = ax.ravel()
    theshape = phase_weighted.shape
    #
    # current offset phase
    im00 = ax[0].imshow(amplitude)
    ax[0].set_title("amplitude")
    plt.colorbar(im00, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].axis([theshape[0] // 2 - theshape[0] // 2 // zoom,
                theshape[0] // 2 + theshape[0] // 2 // zoom,
                theshape[1] // 2 - theshape[1] // 2 // zoom,
                theshape[1] // 2 + theshape[1] // 2 // zoom])
    ax[0].set_ylim(ax[0].get_ylim()[::-1])
    #
    # computed Fourier phase
    im01 = ax[1].imshow(phase_weighted, cmap='seismic', vmin=phase.min(),
                        vmax=phase.max())
    ax[1].set_title("weighted phase")
    plt.colorbar(im01, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].axis([theshape[0] // 2 - theshape[0] // 2 // zoom,
                theshape[0] // 2 + theshape[0] // 2 // zoom,
                theshape[1] // 2 - theshape[1] // 2 // zoom,
                theshape[1] // 2 + theshape[1] // 2 // zoom])
    ax[1].set_ylim(ax[1].get_ylim()[::-1])

    if save_as_eps is True:
        #ax[1].set_rasterization_zorder(0)
        ax[1].set_rasterized(True)
        #plt.savefig(phase_filename[:-4] + '.eps', format = 'eps', bbox_inches='tight', pad_inches=0)
        plt.savefig(phase_filename[:-4] + '.eps', format = 'eps', bbox_inches='tight', pad_inches=0)
        #plt.imsave(phase_filename[:-4] + '.png', phase_weighted)
    else:
        pass

    #
    fig.tight_layout()
    #display.clear_output(wait=True)
    plt.show()
