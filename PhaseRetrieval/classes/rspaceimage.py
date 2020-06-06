#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for object-domain images.

"""

import os
import mahotas as mh
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
from skimage import io
from skimage import measure
from skimage.filters import gaussian
from skimage.segmentation import watershed

from skimage.transform import AffineTransform, resize, warp
from skimage.util import pad


from PhaseRetrieval.classes.rspacemetadata import RSpaceMetadata


class RSpaceImage(object):

    def __init__(self, filename=None, delimiter=None, image=None, image_apodization_filter=None):
        """
        Initializes the object-domain image class

        ---
        Parameters
        ---
        filename: str, optional
            Path used to load the image.
            If None, an empty class is created.
            If `image` argument is provided, the image will be initialized from the `image` argument.
            Default is None.
        delimiter: str, optional
            Delimiter used in the csv file (need to be specified if data are loaded from a csv file).
            If None, an empty class is created.
            Default is None.
        image : ndarray, optional
            2D array to initialize the image.
            If None, an empty class is created.
            Default is None.
        image_apodization_filter: ndarray, optional
            2D array to initialize the thresholded version of the image (i.e. image apodization filter).
            If None, an empty class is created.
            Default is None.
        """
        self.filename = filename
        self.delimiter = delimiter
        self.image = image
        self.image_apodization_filter = image_apodization_filter
        #
        self.metadata = RSpaceMetadata()
        self.metadata['Apodization filter applied?'] = None
        self.metadata['Image centred and padded?'] = None
        self.metadata['Linear number of pixels in the zero-padded real-space image'] = None
        self.metadata["Parseval's theorem fulfilled?"] = None
        self.metadata['Pixel size object domain, nm'] = None
        #
        if self.image is None and filename is not None and delimiter is not None and os.path.exists(filename):
            # then read data from file
            self.read_from_csv(filename, delimiter)

    def __repr__(self):
        return "2D object-domain image"

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
            self.image = self.image.values
            self.filename = filename
            self.delimiter = delimiter
            #
            self.metadata['Image centred and padded?'] = 'no'
            self.metadata['Background subtracted?'] = 'no'
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

    def plot_image(self, zoom = 1):
        """
        Plots the image

        ---
        Parameters
        ---
        zoom: int, optional
            Zoom factor to zoom into the plot
            Default is 1 (no zoom)
        """
        if self.image is not None:
            # convert to numpy array
            #if type(self.image) is not np.ndarray:
            #    self.image = self.image.to_numpy()
            #else:
            #    pass
            # plot the image from file
            plt.imshow(self.image)
            plt.axis([self.image.shape[0] // 2 - self.image.shape[0] // 2 // zoom,
                      self.image.shape[0] // 2 + self.image.shape[0] // 2 // zoom,
                      self.image.shape[1] // 2 - self.image.shape[1] // 2 // zoom,
                      self.image.shape[1] // 2 + self.image.shape[1] // 2 // zoom])
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.title("Object-domain image")
        else:
            raise ValueError('Read the image data first!')

    def rotate_image(self, times_rot=1, axes=(0, 1), estimate_only = True):
        """
        Rotates the image by 90°.

        ---
        Parameters
        ---
            times_rot: int, optional
                Number of times the array is rotated by 90 degrees.
                Default is 1.
            axes: array, optional
                (0,1) - counter-clockwise
                (1,0) - clockwise
                Default is (0,1) - counter-clockwise
            estimate_only: bool, optional
                False will irreversibly rotate the input image.
                True will only estimate how the image will look like after the rotation.
                Default is True.
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
                #
                im2= ax2.imshow(np.rot90(self.image, times_rot, axes))
                ax2.set_title("If rotated")
                #ax2.invert_yaxis()
                plt.colorbar(im2,ax=ax2)
                plt.show()
            elif estimate_only is False:
                #
                fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
                ax1, ax2 = ax
                im1 = ax1.imshow(self.image)
                ax1.set_title("Original")
                #ax1.invert_yaxis()
                plt.colorbar(im1,ax=ax1)
                #
                temp = np.rot90(self.image, times_rot, axes)
                self.image = temp
                print("Input image was rotated")
                #
                im2 = ax2.imshow(self.image)
                ax2.set_title("Rotated")
                #ax2.invert_yaxis()
                plt.colorbar(im2,ax=ax2)
                plt.show()
                #
        else:
            raise ValueError('Read the image data first!')

    def flip_image(self, axis=0, estimate_only = True):
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
        estimate_only: bool, optional
            False will irreversibly flip the input image.
            True will only estimate how the image will look like after the flipping.
            Default is True.
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
                plt.show()
            elif estimate_only == False:
                #
                fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
                ax1, ax2 = ax
                im1 = ax1.imshow(self.image)
                ax1.set_title("Original")
                plt.colorbar(im1,ax=ax1)
                #
                if axis == 0:
                    # flip upside-down
                    self.image = np.flipud(self.image)
                elif axis == 1:
                    # flip left-right
                    self.image = np.fliplr(self.image)
                print("Input image was flipped")
                #
                im2 = ax2.imshow(self.image)
                ax2.set_title("Flipped")
                plt.colorbar(im2,ax=ax2)
                plt.show()
        else:
            raise ValueError('Read the image data first!')

    def centre_image_watershed(self, linear_object_size=100, npixels_pad=2000, apodization=False):
        """
        Centers the object-domain image using watershed algorithm (the existence of only a single blob is assumed).
        Finds its physical linear pixel size.
        Completes zero-padding of the original image to a specified linear number of pixels
        Applies an apodization filter to smooth boundaries of the object distribution (optional)

        ---
        Parameters
        ---
        linear_object_size: float, optional
            Physical linear size of the (non-zero-valued) input object distribution, in micrometers.
            Default is 100.
        npixels_pad: int, optional
            Linear number of pixels in the zero-padded object-domain image.
            Default is 2000.
        apodization: bool, optional
            If True, the boundaries of the object distribution will be smoothed using Gaussian filter
            with standard deviation = 1 and truncation of the filter's boundaries to 2
            (fixed at the moment, but may/should be changed in the future)
            Default is False
        """
        if self.image is not None:
            if linear_object_size is not None and npixels_pad is not None:
                if npixels_pad >= self.image.shape[0] or npixels_pad >= self.image.shape[1]:
                    #
                    #convert to numpy array
                    if type(self.image) is not np.ndarray:
                        self.image = self.image.to_numpy()
                    else:
                        pass
                    #
                    plt.imshow(self.image)
                    plt.title('self.image')
                    plt.colorbar()
                    plt.show()
                    # compute Euclidean distance transform
                    im_distance = ndi.distance_transform_edt(self.image)
                    plt.imshow(im_distance)
                    plt.title('Output of the distance transform algorithm')
                    plt.colorbar()
                    plt.show()
                    #
                    # find local maximum (only 1 in this case)
                    im_local_maxi = feature.peak_local_max(im_distance,
                                                           indices=False,
                                                           num_peaks=1)
                    #
                    # watershed
                    im_watershed = 1 * watershed(-im_distance,
                                                 markers=im_local_maxi,
                                                 mask=self.image)
                    plt.imshow(im_watershed)
                    plt.title('Output of the watershed algorithm')
                    plt.colorbar()
                    plt.show()
                    #
                    #now pad original image and watershed image
                    im_pad = pad(self.image,
                                ((0, npixels_pad - self.image.shape[0]),
                                (0, npixels_pad - self.image.shape[1])),
                                mode='constant')
                    im_watershed_pad = pad(im_watershed,
                                           ((0, npixels_pad - self.image.shape[0]),
                                           (0, npixels_pad - self.image.shape[1])),
                                           mode='constant')
                    plt.imshow(im_watershed_pad)
                    plt.title('Output of the watershed algorithm - padded')
                    plt.colorbar()
                    plt.show()
                    print("Input and watershed images have been padded to ", im_pad.shape[0], "X", im_pad.shape[1], "pixels.")
                    self.metadata['Linear number of pixels in the zero-padded real-space image']= npixels_pad
                    #
                    # compute the object's centre as a centre of mass of the padded watershed distribution
                    im_moments = measure.moments(im_watershed_pad)
                    im_centroid = (round(im_moments[0, 1] / im_moments[0, 0] - im_pad.shape[0] / 2),
                                   round(im_moments[1, 0] / im_moments[0, 0] - im_pad.shape[1] / 2))
                    #
                    # find out physical linear pixel size
                    pixelsize_dr0 = round(1e3 * np.sqrt(linear_object_size**2 / np.count_nonzero(im_watershed_pad)), 0)
                    self.metadata['Pixel size object domain, nm'] = pixelsize_dr0
                    #
                    # centre object's distribution
                    # (with or without the application of the apodization filter)
                    if apodization is False:
                        self.image = warp(im_pad,
                                          AffineTransform(translation=im_centroid),
                                          mode='wrap',
                                          preserve_range=True)
                        print('Centred.\nApodization filter has NOT been applied.\nLinear pixel size is ', pixelsize_dr0, "nm")
                        self.metadata['Apodization filter applied?'] = 'no'
                        plt.imshow(self.image)
                        plt.title('Centred image')
                        plt.colorbar()
                        plt.show()
                    else:
                        # compute apodization filter
                        image_apodization_filter = warp(im_watershed_pad,
                                                        AffineTransform(translation=im_centroid),
                                                        mode='wrap',
                                                        preserve_range=True)
                        image_apodization_filter = gaussian(image_apodization_filter,
                                                            sigma=1,
                                                            preserve_range=True,
                                                            truncate=2.0)
                        image_apodization_filter = image_apodization_filter / np.max(np.max(image_apodization_filter))
                        self.image_apodization_filter = image_apodization_filter
                        plt.imshow(image_apodization_filter)
                        plt.title('Apodization filter')
                        plt.colorbar()
                        plt.show()
                        #
                        self.image = image_apodization_filter * warp(im_pad,
                                                                              AffineTransform(translation=im_centroid),
                                                                              mode='wrap',
                                                                              preserve_range=True)
                        print("Centred.\nApodization filter has been applied.\nLinear pixel size is ", pixelsize_dr0, "nm")
                        self.metadata['Apodization filter applied?'] = 'yes'
                        plt.imshow(self.image)
                        plt.title('Centred and apodized image')
                        plt.colorbar()
                        plt.show()
                    self.metadata['Image centred and padded?'] = 'yes'
                    #
                else:
                    raise ValueError('Specified number of pixels for zero-padding is too low!')

            else:
                raise ValueError('Argument(s) missing!')
        else:
            raise ValueError('Read the image data first!')

    def subtract_background(self, counts = 0, zoom = 1, log_scale = False, estimate_only = True):
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
            False will irreversibly subtract the background from the input image.
            True will only estimate how the image will look like after the background subtraction.
            Default is True.
        """
        if self.image is not None:
            im_bgfree = self.image
            im_bgfree = im_bgfree - counts
            im_bgfree[im_bgfree < 0] = 0
            #
            if estimate_only == True:
                # estimate how the image would look like if the background were subtracted
                if log_scale == True:
                    plt.imshow(np.log(im_bgfree))
                    plt.title("If background-subtracted (log scale)")
                else:
                    plt.imshow(im_bgfree)
                    plt.title("If background-subtracted")
                plt.axis([im_bgfree.shape[0] // 2 - im_bgfree.shape[0] // 2 // zoom,
                          im_bgfree.shape[0] // 2 + im_bgfree.shape[0] // 2 // zoom,
                          im_bgfree.shape[1] // 2 - im_bgfree.shape[1] // 2 // zoom,
                          im_bgfree.shape[1] // 2 + im_bgfree.shape[1] // 2 // zoom])
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.show()
                print("This is how the image looks like if the background were subtracted.")
                print("Background was set to ", counts)
            else:
                # subtract background and save changes
                self.image = im_bgfree
                self.metadata['Background subtracted?'] = 'yes'
                #
                if log_scale == True:
                    plt.imshow(np.log(self.image))
                    print('Plotted in log scale')
                else:
                    plt.imshow(self.image)
                plt.axis([im_bgfree.shape[0] // 2 - im_bgfree.shape[0] // 2 // zoom,
                          im_bgfree.shape[0] // 2 + im_bgfree.shape[0] // 2 // zoom,
                          im_bgfree.shape[1] // 2 - im_bgfree.shape[1] // 2 // zoom,
                          im_bgfree.shape[1] // 2 + im_bgfree.shape[1] // 2 // zoom])
                plt.gca().invert_yaxis()
                plt.title("Image after the background subtraction.")
                plt.colorbar()
                plt.show()
                print("Background of", counts, "counts has been subtracted.")
        else:
            raise ValueError('Read the image data and centre it first!')

    def resample_image(self, fieldofview = 17, npixels_kspace = 500, pixelsize_dr0 = None, lambd = 880, estimate_only = True):
        """
        Resamples object-space image to equalise its pixel size to the one set by
        digital Fourier transform and the pixel size in the experimental Fourier-space image

        ---
        Parameters
        ---
        fieldofview: int, optional
            One half of field of view in Fourier-space, in degrees
            Default is ±17°, i.e. fieldofview = 17
        npixels_kspace: int, optional
            One half of linear number of non-zeros-valued pixels in experimental Fourier-space image to be used together with the real-space image
            (= corresponds to the linear number of pixels within the 1/2 of field of view)
            Default is 500
        pixelsize_dr0: int, optional
            Pixel size in experimental object-domain image, in nm.
            If it is None, the value will be read from metadata (saved to metadata after centering and segmentation of the image).
            If the value in metadata is None, there will be an error message.
            Alternatively, one can set the pixel size manually.
            Default is None.
        lambd: float, optional
            Wavelength of light, in nm
            Default is 880
        estimate_only: bool, optional
            False will irreversibly resample the input image.
            True will only estimate the downsampling ratio.
            Default is True.
        """
        if self.image is not None:
            if self.metadata['Image centred and padded?'] == 'yes':
                # read out pixel size in object domain
                if pixelsize_dr0 is None:
                    if self.metadata['Pixel size object domain, nm'] is not None:
                    # if none, read from metadata of it's not none
                        pixelsize_dr0 = 1e-9 * self.metadata['Pixel size object domain, nm']
                    else:
                        raise ValueError('Pixel size object domain is None! Either specify the pixel size manually or determine it via centre_image method!')
                else:
                    #if not none save the pixel size in metadata and convert ot nm
                    self.metadata['Pixel size object domain, nm'] = pixelsize_dr0
                    pixelsize_dr0 = 1e-9 * pixelsize_dr0
                #
                npixels_pad = self.metadata['Linear number of pixels in the zero-padded real-space image']
                alpha = fieldofview * np.pi / 180
                #
                # compute pixel size in Fourier domain
                pixelsize_dk = (np.sin(alpha) * 2 * np.pi / (lambd*1e-9)) / npixels_kspace
                self.metadata['Pixel size Fourier domain, 1/nm'] = pixelsize_dk * 1e9
                #
                # compute pixel size in the object domain
                # as set by the discrete Fourier transform
                pixelsize_dr_pad = 2*np.pi / (npixels_pad * pixelsize_dk)
                self.metadata['Pixel size object domain from padded Fourier data, 1/nm'] = pixelsize_dr_pad * 1e9
                #
                #compute the downsampling ratio
                downsampling = round(pixelsize_dr_pad / pixelsize_dr0)
                #
                #compute final linear number of pixels as set by the downsampling ratio and experimental
                #pixel sizes in Fourier and object spaces
                npixels_final = int(round(2 * np.pi / (downsampling * pixelsize_dr0 * pixelsize_dk)))
                print('npixels_final =', npixels_final)
                #
                if estimate_only == False:
                    print('input image shape is ', self.image.shape[0],'X', self.image.shape[1])
                    print('resampling object domain image...')
                    #
                    self.image = resize(np.array(self.image), (self.image.shape[0] // downsampling, self.image.shape[1] // downsampling), anti_aliasing=True)
                    print('image has been resampled and its current shape is', self.image.shape)
                    npixels_to_pad_final0 = int((npixels_final - self.image.shape[0]) / 2)
                    npixels_to_pad_final1 = int((npixels_final - self.image.shape[1]) / 2)
                    self.image = pad(np.array(self.image), ((npixels_to_pad_final0, npixels_to_pad_final1), (npixels_to_pad_final0, npixels_to_pad_final1)), mode='constant')
                    print('image has been resampled with the downsampling ratio =', downsampling, 'and zero-padded to npixels_final X npixels_final=', self.image.shape[0], 'X', self.image.shape[1], 'pixels')
            #
                elif estimate_only == True:
                    print('estimating the downsampling ratio...')
                    print('Downsampling ratio =', downsampling)
            else:
                raise ValueError('Centre and zero-pad image distribution!')
        else:
            raise ValueError('Read the image data first!')

    def save_as_tif(self, pathtosave=None, outputfilename=None):
        """
        Saves object-domain image as tif

        ---
        Parameters
        ---
        outputfilename: str
            Path to save the image as tif
            Default is None
        """
        if self.image is not None:
            if outputfilename is not None and os.path.exists(pathtosave):
                #
                # convert to numpy array
                imagetosave = self.image
                if type(self.image) is not np.ndarray:
                    imagetosave = self.image.to_numpy()
                #
                # append tif if needed
                if not outputfilename.endswith('.tif'):
                    outputfilename += '.tif'
                # save the image
                io.imsave(os.path.join(pathtosave, outputfilename), imagetosave.astype(np.float32))
                # save its metadata
                self.metadata.to_csv(os.path.join(pathtosave, outputfilename[:-4] + '_meta.csv'), sep='\t', header=False)
                self.filename = outputfilename
                print('The image was saved in tif file. Its metadata was saved in csv file.')
            else:
                raise ValueError('Invalid path! Please specify a valid path to save data as tif file.')
        else:
            raise ValueError('Read the image data first!')