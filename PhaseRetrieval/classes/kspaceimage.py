#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for Fourier-domain images.

"""
from copy import deepcopy
import os
import numpy as np
import pandas as pd
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from skimage import io
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.transform import AffineTransform, warp
from skimage.util import pad

from PhaseRetrieval.classes.kspacemetadata import KSpaceMetadata


class KSpaceImage(object):

    def __init__(self, filename=None, delimiter=None, image=None, renorm_factor = None):
        """
        Initializes Fourier-domain image class

        ---
        Parameters
        ---
        filename: str, optional
            Path used to load the image.
            If None, an empty class is created.
            If `image` argument is provided, the image will be initialized from the `image` argument.
            Default is None.
        delimiter: str, optional
            Delimiter used in the csv file
            Delimiter needs to be specified if the data are to be loaded from a csv file.
            If None, then it is assumed that the data are to be loaded from a tif file.
            Default is None.
        image : ndarray, optional
            2D array to initialize the image.
            If None, an empty class is created.
            Default is None.
        renorm_factor : float
            Renormalisation factor to fulfill Parseval's theorem
            If None, an empty class is created.
            Default is None.
        """
        self.filename = filename
        self.delimiter = delimiter
        self.image = image
        self.renorm_factor = renorm_factor
        #
        self.metadata = KSpaceMetadata()
        self.metadata['Background subtracted?'] = None
        self.metadata['Image centred and padded?'] = None
        self.metadata["Parseval's theorem fulfilled?"] = None
        #
        if self.image is None and filename is not None and delimiter is not None and os.path.exists(filename):
            # then read data from file
            self.read_from_csv(filename, delimiter)
        elif self.image is None and filename is not None and delimiter is None and os.path.exists(filename):
            # then read data from tif file
            self.read_from_tif(filename)

    def __repr__(self):
        return "2D Fourier-domain image"

    def read_from_csv(self, filename, delimiter):
        """
        Reads an image from a csv file.

        ---
        Parameters
        ---
        filename: str
            Path used to load the image.
        delimiter: str
            Delimiter used in the csv file.
            Default is a comma ','.
        """
        if os.path.exists(filename) and delimiter is not None:
            # read the image from file
            # as there is no header in images, the header is set to None
            self.image = pd.read_csv(filename, delimiter, header=None)
            self.filename = filename
            self.delimiter = delimiter
            #
            self.metadata['Image centred and padded?'] = 'no'
            self.metadata['Background subtracted?'] = 'no'
            print("Fourier domain: Input image was read")
        else:
            raise ValueError('Invalid path! File does not exist!')

    def read_from_tif(self, filename):
        """
        Reads an image from a tif file.

        ---
        Parameters
        ---
        filename: str
            Path used to load the image.
        """
        if os.path.exists(filename):
            # read the image from file
            self.image = io.imread(filename)
            self.filename = filename
            #
            self.metadata['Background subtracted?'] = 'no'
            self.metadata['Image centred and padded?'] = 'no'
        else:
            raise ValueError('Invalid path! File does not exist!')

    def plot_image(self, zoom=1, log_scale=False):
        """
        Plots the image.

        ---
        Parameters
        ---
        zoom: int, optional
            Zoom factor to zoom into the plot
            Default is 1 (no zoom)
        log_scale: bool, optional
            To plot the image in log scale
            Default is False.
        """
        if self.image is not None:
            # plot the image from file
            if log_scale is True:
                plt.imshow(np.log(self.image))
                plt.title("Fourier-domain image (log)")
            else:
                plt.imshow(self.image)
                plt.title("Fourier-domain image")
                plt.axis([self.image.shape[0] // 2 - self.image.shape[0] // 2 // zoom,
                      self.image.shape[0] // 2 + self.image.shape[0] // 2 // zoom,
                      self.image.shape[1] // 2 - self.image.shape[1] // 2 // zoom,
                      self.image.shape[1] // 2 + self.image.shape[1] // 2 // zoom])
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.show()
        else:
            raise ValueError('Read the image data first!')

    def rotate_image(self, times_rot=1, axes=(0, 1), zoom = 1, estimate_only = True,  plot_progress = False):
        """
        Rotates the image by 90°.

        ---
        Parameters
        ---
            times_rot: int, optional
                Number of times the image is rotated by 90 degrees.
                Default is 1.
            axes: array, optional
                (0,1) - counter-clockwise
                (1,0) - clockwise
                Default is (0,1) - counter-clockwise
            zoom: int, optional
                Zoom factor to zoom into the plot
                Default is 1 (no zoom)
            estimate_only: bool, optional
                False will irreversibly rotate the input image.
                True will only estimate how the image will look like after the rotation.
                Default is True.
            plot_progress: bool, optional
                Plot original and rotated images for estimate_only = False.
                Default is False
        """
        if self.image is not None:
            if estimate_only is True:
                fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
                ax1, ax2 = ax
                im1 = ax1.imshow(self.image)
                ax1.set_title("Original")
                #ax1.invert_yaxis()
                plt.colorbar(im1,ax=ax1)
                #
                im2 = ax2.imshow(np.rot90(self.image, times_rot, axes))
                ax2.set_title("If rotated")
                plt.colorbar(im2,ax=ax2)
                plt.axis([self.image.shape[0] // 2 - self.image.shape[0] // 2 // zoom,
                          self.image.shape[0] // 2 + self.image.shape[0] // 2 // zoom,
                          self.image.shape[1] // 2 - self.image.shape[1] // 2 // zoom,
                          self.image.shape[1] // 2 + self.image.shape[1] // 2 // zoom])
                ax2.invert_yaxis()
                plt.show()
                print("Fourier domain: This is how the image looks like if rotated.")
            elif estimate_only is False:
                #
                rotated = np.rot90(self.image, times_rot, axes)
                print("Fourier domain: Input image was rotated")
                #
                if plot_progress is True:
                    #
                    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
                    ax1, ax2 = ax
                    im1 = ax1.imshow(self.image)
                    ax1.set_title("Original")
                    #ax1.invert_yaxis()
                    plt.colorbar(im1,ax=ax1)
                    #
                    im2 =ax2.imshow(rotated)
                    ax2.set_title("Rotated")
                    plt.colorbar(im2,ax=ax2)
                    plt.axis([self.image.shape[0] // 2 - self.image.shape[0] // 2 // zoom,
                              self.image.shape[0] // 2 + self.image.shape[0] // 2 // zoom,
                              self.image.shape[1] // 2 - self.image.shape[1] // 2 // zoom,
                              self.image.shape[1] // 2 + self.image.shape[1] // 2 // zoom])
                    ax2.invert_yaxis()
                    plt.show()
                #
                self.image = rotated
                #
        else:
            raise ValueError('Read the image data first!')

    def flip_image(self, axis=0, zoom = 1, estimate_only = True, plot_progress = False):
        """
        Flips the image along its vertical or horizontal axes.

        ---
        Parameters
        ---
        axis: None or int or tuple of ints, optional
            Axis or axes along which to flip over.
            The default, axis=None, will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes specified in the tuple.
            axis = 0 - flip upside-down
            axis = 1 - flip left-right
            Default is 0.
        zoom: int, optional
            Zoom factor to zoom into the plot
            Default is 1 (no zoom)
        estimate_only: bool, optional
            False will irreversibly flip the input image.
            True will only estimate how the image will look like after the flipping.
            Default is True.
        plot_progress: bool, optional
            Plot original and flipped images for estimate_only = False.
            Default is False
        """
        if self.image is not None:
            if estimate_only == True:
                fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
                ax1, ax2 = ax
                im1 = ax1.imshow(self.image)
                ax1.set_title("Original")
                plt.colorbar(im1,ax=ax1)
                #
                if axis == 0:
                    # flip upside-down
                    im2 = ax2.imshow(np.flipud(self.image))
                    ax2.set_title("If flipped")
                    plt.colorbar(im2, ax=ax2)
                elif axis == 1:
                    # flip left-right
                    im2 = ax2.imshow(np.fliplr(self.image))
                    ax2.set_title("If flipped")
                    plt.colorbar(im2, ax=ax2)
                plt.axis([self.image.shape[0] // 2 - self.image.shape[0] // 2 // zoom,
                          self.image.shape[0] // 2 + self.image.shape[0] // 2 // zoom,
                          self.image.shape[1] // 2 - self.image.shape[1] // 2 // zoom,
                          self.image.shape[1] // 2 + self.image.shape[1] // 2 // zoom])
                ax2.invert_yaxis()
                plt.show()
            #
            elif estimate_only == False:
                #
                if axis == 0:
                    # flip upside-down
                    flipped = np.flipud(self.image)
                elif axis == 1:
                    # flip left-right
                    flipped = np.fliplr(self.image)
                print("Fourier domain: Input image was flipped")
                #
                if plot_progress is True:
                    #
                    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
                    ax1, ax2 = ax
                    im1 = ax1.imshow(self.image)
                    ax1.set_title("Original")
                    plt.colorbar(im1,ax=ax1)
                    #
                    im2 = ax2.imshow(flipped)
                    ax2.set_title("Flipped")
                    plt.axis([self.image.shape[0] // 2 - self.image.shape[0] // 2 // zoom,
                              self.image.shape[0] // 2 + self.image.shape[0] // 2 // zoom,
                              self.image.shape[1] // 2 - self.image.shape[1] // 2 // zoom,
                              self.image.shape[1] // 2 + self.image.shape[1] // 2 // zoom])
                    ax2.invert_yaxis()
                    plt.colorbar(im2,ax=ax2)
                    plt.show()
                #
                self.image = flipped
        else:
            raise ValueError('Read the image data first!')

    def centre_image_manually(self, manual_centroid = (500, 500), npixels_final = 2000):
        """
        Centers the image using manually provided its centroid local_max_coordinates.
        Pads the image with zeros to a larger linear number of pixels.

        ---
        Parameters
        ---
            manual_centroid: tuple, optional
                Tuple's elements are centroid's local_max_coordinates (integers).
                Default is (500, 500).
            npixels_final: int, optional
                Final linear number of pixels in the centered image.
                The input image will be padded by zeros in both dimensions.
                Default is 2000.
        """
        #
        # convert to numpy array in case it was not yet
        if self.image is not None:
            if isinstance(npixels_final, int) is True:
                if npixels_final > self.image.shape[0] or self.image.shape[0]:
                    if type(self.image) is not np.ndarray:
                        im = self.image.to_numpy()
                    else:
                        im = self.image
                    # zero-pad to a given linear number of pixels
                    im_pad = pad(im,
                                 ((0, npixels_final - im.shape[0]), (0, npixels_final - im.shape[1])),
                                 mode='constant')
                    # compute centroid in the zero-padded image
                    manual_centroid = (round(manual_centroid[0] - im_pad.shape[0]/ 2),
                                       round(manual_centroid[1] - im_pad.shape[1] / 2))
                    # centre image
                    im_centred = warp(im_pad, AffineTransform(translation = manual_centroid), mode='wrap', preserve_range=True)
                    # save the changes
                    self.image = im_centred
                    #
                    self.metadata['Image centred and padded?'] = 'yes'
                else:
                    raise ValueError('Final pixel number must be greater than the current one!')
            else:
                raise ValueError('Number of pixels must be an integer!')
        else:
            raise ValueError('Read the image data first!')

    def centre_image(self, roi = (0,10,0,10), centre = (1,1), gaussian_filter = False, sigma = 1, min_distance = 10,
                     threshold_abs = 0, num_peaks = 1, npixels_pad = 2000, zoom = 1, estimate_only = True, plot_progress = False):
        """
        Centers the Fourier-domain image whose centre is located at one of its local maxima
        Completes zero-padding of the original image to a specified linear number of pixels

        ---
        Parameters
        ---
        roi: tuple, optional
            Region of interest (ROI) used to search for local maxima.
            Default is (0,10,0,10).
        centre: tuple, optional
            Centre of the image.
            Default is (1,1), which must be changed by user once the centre of the image (one of the local maxima) is found.
        gaussian_filter : bool, optional
            Apply Gaussian filter to filter noise
            Default is False
        sigma : float, optional
            Standard deviation of a Gaussian filter
            Defailt is 1.0
        min_distance, int, optional
            Minimal distance between the local maxima
            Must be tuned by user to make the search most effective.
            Default is 10.
        threshold_abs: float, optional
            Minimum intensity of peaks.
            Default is 0.
        num_peaks: int, optional
            Maximum number of peaks.
            When the number of peaks exceeds num_peaks, return num_peaks peaks based on highest peak intensity.
            Default is 1.
        npixels_pad: int, optional
            Linear number of pixels in the zero-padded Fourier-domain image.
            Need be set to the one in the zero-padded object-domain image after it was resampled.
            Default is 2000.
        zoom: int, optional
            Zoom factor to zoom into the plot
            Default is 1 (no zoom)
        estimate_only: bool, optional
            False will irreversibly centre the image using local_max_coordinates specified in 'centre' argument
            True will only find the local_max_coordinates of local maxima
            Default is True.
        plot_progress: bool, optional
            Plot centred image for estimate_only = False.
            Default is False
        """
        if self.image is not None:
            if npixels_pad >= self.image.shape[0] or npixels_pad >= self.image.shape[1]:
                #
                # convert to numpy array
                if type(self.image) is not np.ndarray:
                    self.image = self.image.to_numpy()
                else:
                    pass
                #apply Gaussian filter
                if gaussian_filter is True:
                    image_filtered = gaussian(deepcopy(self.image),
                                              sigma = sigma,
                                              preserve_range = True)

                else:
                    image_filtered = deepcopy(self.image)
                #
                # set ROI and find local maxima
                mask = np.zeros(self.image.shape)
                mask[roi[1]:roi[3], roi[0]:roi[2]] = 1
                im_masked = image_filtered * mask
                local_max_coordinates = peak_local_max(im_masked,
                                             min_distance = min_distance,
                                             threshold_abs = threshold_abs,
                                             num_peaks = num_peaks)
                #
                if plot_progress is True:
                    # plot local maxima
                    fig, ax = plt.subplots(1)
                    im = ax.imshow(self.image)
                    rect = patches.Rectangle((roi[0], roi[1]),
                                             roi[2] - roi[0],
                                             roi[3] - roi[1],
                                             linewidth=1,
                                             edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)
                    ax.plot(local_max_coordinates[:, 1],
                            local_max_coordinates[:, 0],
                            'r.')
                    local_max_coordinates_temp = deepcopy(local_max_coordinates)
                    local_max_coordinates[:,0]=local_max_coordinates_temp[:,1]
                    local_max_coordinates[:,1] = local_max_coordinates_temp[:,0]
                    plt.text(int(roi[2]),
                             int(roi[3]),
                             local_max_coordinates,
                             fontsize=11,
                             color='r')
                    plt.title('Original image, ROI and local maxima')
                    plt.axis([self.image.shape[1] // 2 - self.image.shape[1] // 2 // zoom,
                              self.image.shape[1] // 2 + self.image.shape[1] // 2 // zoom,
                              self.image.shape[0] // 2 - self.image.shape[0] // 2 // zoom,
                              self.image.shape[0] // 2 + self.image.shape[0] // 2 // zoom])
                    plt.colorbar(im)
                    plt.gca().invert_yaxis()
                    plt.show()
                #
                # centre image
                if estimate_only is not True:
                    #
                    # zero-pad to a given linear number of pixels
                    im_pad = pad(self.image,
                                 ((0, npixels_pad - self.image.shape[0]), (0, npixels_pad - self.image.shape[1])),
                                 mode='constant')
                    #
                    # compute centroid in the zero-padded image
                    centroid = (round(centre[0] - im_pad.shape[0]/2 - 1),
                                       round(centre[1] - im_pad.shape[1]/2 - 1))
                    #
                    # centre image
                    im_centred = warp(im_pad, AffineTransform(translation=centroid), mode='wrap',
                                      preserve_range=True)
                    print("Fourier domain: Input image was padded to ", im_centred.shape[0], "X", im_centred.shape[1],"pixels and centred.")
                    self.metadata['Linear number of pixels in the zero-padded real-space image'] = npixels_pad
                    #
                    if plot_progress is True:
                        # plot centred image
                        fig1, ax1 = plt.subplots(1)
                        plt.imshow(im_centred)
                        ax1.plot(centre[1]-centroid[1],
                                centre[0]-centroid[0],
                                'r.')
                        plt.text(centre[1]-centroid[1] + 10,
                                 centre[0]-centroid[0],
                                 (centre[1]-centroid[1],
                                centre[0]-centroid[0]),
                                 fontsize=11,
                                 color='r')
                        plt.title('Centred image')
                        plt.colorbar()
                        plt.axis([im_centred.shape[1] // 2 - im_centred.shape[1] // 2 // (zoom * im_centred.shape[1]/self.image.shape[0]),
                                  im_centred.shape[1] // 2 + im_centred.shape[1] // 2 // (zoom * im_centred.shape[1]/self.image.shape[0]),
                                  im_centred.shape[0] // 2 - im_centred.shape[0] // 2 // (zoom * im_centred.shape[1]/self.image.shape[0]),
                                  im_centred.shape[0] // 2 + im_centred.shape[0] // 2 // (zoom * im_centred.shape[1]/self.image.shape[0])])
                        plt.gca().invert_yaxis()
                        plt.show()
                    self.image = im_centred
                    self.metadata['Image centred and padded?'] = 'yes'
                #
            else:
                raise ValueError('Specified number of pixels for zero-padding is too low!')
        else:
            raise ValueError('Read the image data first!')

    def subtract_background(self, counts = 0, zoom = 1, log_scale = False, estimate_only = True, plot_progress = False):
        """
        Subtracts constant background given a number of counts

        ---
        Parameters
        ---
        counts: int, optional
            Number of background counts to subtract from the image
            Estimated manually by the user (e.g. from a histogram)
            Default is 0.
        zoom: int, optional
            Zoom factor to zoom into the 2D plot
            Default is 1 (no zoom)
        log_scale: bool, optional
            To plot the image in log scale
            Default is False.
        estimate_only: bool, optional
            False will irreversibly subtract the background from the image.
            True will only estimate how the image will look like after the background subtraction.
            Default is True.
        plot_progress: bool, optional
            Plot background-free image for estimate_only = False.
            Default is False
        """
        if self.image is not None and self.metadata['Image centred and padded?'] == 'yes':
            im_bgfree = self.image
            im_bgfree = im_bgfree - counts
            im_bgfree[im_bgfree < 0] = 0
            #
            if estimate_only == True:
                #estimate how the image would look like if the background were subtracted
                if log_scale == True:
                    plt.imshow(np.log(im_bgfree))
                    plt.title("If background-subtracted (log scale)")
                else:
                    plt.imshow(im_bgfree)
                    plt.title("If background-subtracted")
                plt.axis([im_bgfree.shape[0]//2-im_bgfree.shape[0]//2//zoom,
                          im_bgfree.shape[0]//2+im_bgfree.shape[0]//2//zoom,
                          im_bgfree.shape[1]//2-im_bgfree.shape[1]//2//zoom,
                          im_bgfree.shape[1]//2+im_bgfree.shape[1]//2//zoom])
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.show()
                print("Fourier domain: This is how the image would look like if the background were subtracted.")
                print("Fourier domain: Background was set to ", counts)
            else:
                #subtract background and save changes
                self.image = im_bgfree
                self.metadata['Background subtracted?'] = 'yes'
                #
                if plot_progress is True:
                    if log_scale == True:
                        plt.imshow(np.log(self.image))
                        print('Fourier domain: Plotted in log scale')
                    else:
                        plt.imshow(self.image)
                    plt.axis([im_bgfree.shape[0]//2-im_bgfree.shape[0]//2//zoom,
                              im_bgfree.shape[0]//2+im_bgfree.shape[0]//2//zoom,
                              im_bgfree.shape[1]//2-im_bgfree.shape[1]//2//zoom,
                              im_bgfree.shape[1]//2+im_bgfree.shape[1]//2//zoom])
                    plt.gca().invert_yaxis()
                    plt.title("Image after the background subtraction.")
                    plt.colorbar()
                    plt.show()
                print("Fourier domain: Background of", counts, "counts has been subtracted.")
        else:
            raise ValueError('Read the image data and centre it first!')

    def renormalise_image(self, energy_rspace = None):
        """
        Renormalise Fourier-domain image to fulfill Parseval's theorem

        ---
        Parameters
        ---
        energy_rspace: float
            The sum of all pixels in the object-domain image after background subtraction and rescaling
            Must be estimated manually by the user
            Default is None.
        ---
        """
        if energy_rspace is not None:
            #compute the renormalisation factor
            renorm_factor = energy_rspace * self.image.shape[0] * self.image.shape[1] / np.sum(np.sum(np.array(self.image)))
            #renormalise the Fourier-domain image
            self.renorm_factor = renorm_factor
            self.image = self.image * renorm_factor
            self.metadata["Parseval's theorem fulfilled?"] = 'yes'
            #
            print ("Fourier domain: Energy = ", np.sum(np.sum(self.image)))
            print("Energy in object's domain * total number of pixels: ", self.image.shape[0] * self.image.shape[1] * energy_rspace)
            if round(np.sum(np.sum(self.image))) == round(self.image.shape[0] * self.image.shape[1] * energy_rspace):
                return print("Fourier domain: Image was renormalised. Parseval's theorem is fulfilled.")
            else:
                return print("Fourier domain: Image was renormalised. Parseval's theorem is NOT fulfilled.")
        else:
            raise ValueError('Energy in the object domain cannot be None!')

    def save_as_tif(self,  pathtosave = None, outputfilename = None):
        """
        Saves Fourier-space image as tif

        ---
        Parameters
        ---
        outputfilename: str
            Path to save the image as tif
            Default is None
        """
        if self.image is not None:
            if outputfilename is not None and os.path.exists(pathtosave):
                #append tif if needed
                if not outputfilename.endswith('.tif'):
                    outputfilename += '.tif'
                #save the image
                # convert to numpy array
                if type(self.image) is not np.ndarray:
                    self.image = self.image.to_numpy()
                else:
                    self.image = self.image
                io.imsave(os.path.join(pathtosave, outputfilename), self.image.astype(np.float32))
                #save its metadata
                self.metadata.to_csv(os.path.join(pathtosave, outputfilename[:-4] + '.csv'), sep='\t', header=False)
                self.filename = outputfilename
                print('Fourier domain: Image was saved in tif file under ', os.path.join(pathtosave, outputfilename))
                print('Fourier domain: Metadata was saved in csv file under ', os.path.join(pathtosave, outputfilename[:-4] + '.csv'))
            else:
                raise ValueError('Invalid path! Please specify a valid path to save data as tif file.')
        else:
            raise ValueError('Read the image data first!')