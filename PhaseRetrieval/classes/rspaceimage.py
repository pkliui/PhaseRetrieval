#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for object-domain images.

"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
from skimage import filters
from skimage import io
from skimage import measure
from skimage.filters import gaussian
from skimage.segmentation import watershed
from skimage.filters import rank
from skimage.morphology import disk
import cv2
from sklearn.cluster import KMeans

from skimage.transform import AffineTransform, resize, warp
from skimage.util import pad


from PhaseRetrieval.classes.rspacemetadata import RSpaceMetadata


class RSpaceImage(object):

    def __init__(self, filename=None, delimiter=None, image=None, image_binary=None, image_segmented=None, image_apodization_filter=None):
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
            Delimiter used in the csv file
            Delimiter needs to be specified if the data are to be loaded from a csv file.
            If None, then it is assumed that the data are to be loaded from a tif file.
            Default is None.
        image : ndarray, optional
            2D array to initialize the image.
            If None, an empty class is created.
            Default is None.
        image_binary: ndarray, optional
            2D array to initialize binary version the image.
            If None, an empty class is created.
            Default is None.
        image_segmented: ndarray, optional
            2D array to initialize segmented version the image.
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
        self.image_binary = image_binary
        self.image_segmented = image_segmented
        self.image_apodization_filter = image_apodization_filter
        #
        self.metadata = RSpaceMetadata()
        self.metadata['Wavelength of light, m'] = None
        self.metadata['Field of view, deg'] = None
        self.metadata['Number of pixels within the field of view'] = None
        self.metadata['Linear size of the object, m'] = None
        self.metadata['Pixel size object domain, m'] = None
        self.metadata['Linear number of pixels in the zero-padded real-space image'] = None
        self.metadata['Image centred and padded?'] = None
        self.metadata['Apodization filter applied?'] = None
        self.metadata["Parseval's theorem fulfilled?"] = None
        #
        if self.image is None and filename is not None and delimiter is not None and os.path.exists(filename):
            # then read data from csv file
            self.read_from_csv(filename, delimiter)
        elif self.image is None and filename is not None and delimiter is None and os.path.exists(filename):
            # then read data from tif file
            self.read_from_tif(filename)

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
        if os.path.exists(filename):
            # read the image from file
            # as there is no header in images, the header is set to None
            self.image = pd.read_csv(filename, delimiter, header=None)
            self.image = self.image.values
            self.filename = filename
            self.delimiter = delimiter
            #
            self.metadata['Image centred and padded?'] = 'no'
            self.metadata['Background subtracted?'] = 'no'
            print("Object domain: Input image was read")
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

    def rotate_image(self, times_rot=1, axes=(0, 1), estimate_only = True, plot_progress = False):
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
                plt.colorbar(im1,ax=ax1)
                #
                #
                im2= ax2.imshow(np.rot90(self.image, times_rot, axes))
                ax2.set_title("If rotated")
                plt.colorbar(im2,ax=ax2)
                plt.show()
            elif estimate_only is False:
                #
                rotated = np.rot90(self.image, times_rot, axes)
                print("Object-domain: Input image was rotated")
                #
                if plot_progress is True:
                    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
                    ax1, ax2 = ax
                    im1 = ax1.imshow(self.image)
                    ax1.set_title("Original")
                    plt.colorbar(im1,ax=ax1)
                    #
                    im2 = ax2.imshow(rotated)
                    ax2.set_title("Rotated")
                    plt.colorbar(im2,ax=ax2)
                    plt.show()
                    #
                # replace the image by its rotated version
                self.image = rotated
        else:
            raise ValueError('Read the image data first!')

    def flip_image(self, axis=0, estimate_only = True, plot_progress = False):
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
                plt.show()
            elif estimate_only == False:
                #
                if axis == 0:
                    # flip upside-down
                    flipped = np.flipud(self.image)
                elif axis == 1:
                    # flip left-right
                    flipped = np.fliplr(self.image)
                print("Object-domain: Image was flipped")
                if plot_progress is True:
                    #
                    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
                    ax1, ax2 = ax
                    im1 = ax1.imshow(self.image)
                    ax1.set_title("Original")
                    plt.colorbar(im1,ax=ax1)
                    im2 = ax2.imshow(flipped)
                    ax2.set_title("Flipped")
                    plt.colorbar(im2,ax=ax2)
                    plt.show()
                #
                # replace the image by its flipped version
                self.image = flipped
        else:
            raise ValueError('Read the image data first!')

    def centre_image_watershed(self, linear_object_size=100e-6, npixels_pad=2000, apodization=False, plot_progress = False):
        """
        Centers the object-domain image using watershed algorithm (the existence of only a single blob is assumed).
        Finds its physical linear pixel size and saves this information into image's metadata.
        Completes zero-padding of the original image to a specified linear number of pixels
        Applies an apodization filter to smooth boundaries of the object distribution (optional)

        ---
        Parameters
        ---
        linear_object_size: float, optional
            Physical linear size of the (non-zero-valued) input object distribution, in m.
            Default is 100 micrometer.
        npixels_pad: int, optional
            Linear number of pixels in the zero-padded object-domain image.
            Default is 2000.
        apodization: bool, optional
            If True, the boundaries of the object distribution will be smoothed using Gaussian filter
            with standard deviation = 3 pixels and truncation of the filter's boundaries to 2
            (fixed at the moment, but may/should be changed in the future)
            Default is False
        plot_progress: bool, optional
            Plot images.
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
                    if plot_progress is True:
                        plt.imshow(self.image)
                        plt.title('self.image')
                        plt.colorbar()
                        plt.show()
                    #
                    # compute Euclidean distance transform of the binary image to find local maxima
                    plt.imshow(self.image_binary)
                    plt.title('binary by bg subtraction and thresholding')
                    plt.colorbar()
                    plt.show()

                    # noise removal
                    kernel = np.ones((110, 110), np.float64)
                    print(self.image_binary)
                    opening = cv2.morphologyEx(self.image_binary.astype(np.float64), cv2.MORPH_OPEN, kernel, iterations=2)
                    plt.imshow(opening)
                    plt.title('opening')
                    plt.colorbar()
                    plt.show()
                    # sure foreground area
                    sure_fg = cv2.erode(opening, kernel, iterations=1)
                    plt.imshow(sure_fg)
                    plt.title('max pixel values show sure_fg')
                    plt.colorbar()
                    plt.show()
                    # sure background area
                    sure_bg = cv2.dilate(opening, kernel, iterations=1)
                    print('sure_bg ', sure_bg.dtype)
                    plt.imshow(sure_bg)
                    plt.title('min pixel values show sure_bg')
                    plt.colorbar()
                    plt.show()
                    # Finding unknown region
                    #sure_fg = np.uint8(sure_fg)
                    unknown = sure_bg - sure_fg
                    print('unknown ', unknown.dtype)
                    plt.imshow(unknown)
                    plt.title('unknown')
                    plt.colorbar()
                    plt.show()
                    #
                    #
                    # Marker labelling
                    _, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
                    # Add one to all labels so that sure background is not 0, but 1
                    markers = markers + 1
                    # Now, mark the region of unknown with zero
                    markers[unknown == 1] = 0
                    markers = markers.astype('int32')
                    plt.imshow(markers)
                    plt.title('markers')
                    plt.colorbar()
                    plt.show()
                    #
                    # watershed
                    print(self.image.dtype)


                    image4watershed = np.copy(self.image)
                    image4watershed = filters.gaussian(image4watershed, sigma = 3)
                    image4watershed = image4watershed/np.max(image4watershed)
                    image4watershed = 255 * image4watershed

                    image4watershed = cv2.cvtColor(image4watershed.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    plt.imshow(image4watershed)
                    plt.title('image4watershed')
                    plt.colorbar()
                    plt.show()

                    markers_watershed = cv2.watershed(image4watershed, markers)
                    plt.imshow(markers_watershed)
                    plt.title('markers after segmentation')
                    plt.colorbar()
                    plt.show()

                    markers_watershed[markers_watershed <2]=0
                    markers_watershed[markers_watershed == 2] = 1
                    plt.imshow(markers_watershed)
                    plt.title('markers after segmentation')
                    plt.colorbar()
                    plt.show()

                    image = markers_watershed * self.image
                    plt.imshow(image)
                    plt.title('image after segmentation with watershed')
                    plt.colorbar()
                    plt.show()

                    image = opening * self.image
                    plt.imshow(image)
                    plt.title('image after segmentation with opening')
                    plt.colorbar()
                    plt.show()

                    #self.image[markers == -1] = [255, 0, 0]
                    # Finding sure foreground area
                    #dist_transform = cv2.distanceTransform(opening, distanceType=cv2.DIST_L2, maskSize=5)
                    #ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

                    # Finding unknown region
                    #sure_fg = np.uint8(sure_fg)
                    #unknown = cv2.subtract(sure_bg, sure_fg)
                    #
                    # Marker labelling
                    #ret, markers = cv2.connectedComponents(sure_fg)

                    # Add one to all labels so that sure background is not 0, but 1
                    #markers = markers + 1

                    # Now, mark the region of unknown with zero
                    #markers[unknown == 255] = 0

                    im_distance = ndi.distance_transform_edt(self.image_binary)
                    #
                    if plot_progress is True:
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
                    # compute image gradient using Sobel filter
                    #im_gradient = filters.sobel(filters.gaussian(self.image, sigma=5))
                    #im_gradient = im_gradient / np.max(im_gradient)

                    # image gradient
                    denoised = rank.median(self.image, disk(2))
                    im_gradient = rank.gradient(denoised, disk(2))
                    plt.imshow(im_gradient)
                    plt.colorbar()
                    plt.title("gradient")
                    plt.show()


                    #
                    # apply watershed algorithm
                    # use the negative of the distance transform as "the landscape"
                    # use local maxima of the distance transform as "the markers of the valleys"
                    im_watershed = 1 * watershed(im_gradient,
                                                 markers=im_local_maxi)
                    #
                    if plot_progress is True:
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
                    #
                    if plot_progress is True:
                        plt.imshow(im_watershed_pad)
                        plt.title('Output of the watershed algorithm - padded')
                        plt.colorbar()
                        plt.show()
                    print("Object domain: Input and watershed images were padded to ", im_pad.shape[0], "X", im_pad.shape[1], "pixels.")
                    self.metadata['Linear number of pixels in the zero-padded real-space image']= npixels_pad
                    #
                    # compute the relative shift the segmented region's centroid w.r.t. the centre of the computational domain
                    # (the tuple's items must be multiplied by -1 to ensure that in AffineTransform the shift occurs in the right direction)
                    # the moments are rounded to ensure the centroid is not between the pixels
                    # add 0.1 to round(im_moments ... / im_moments ... ) to round as per mathematical definition
                    im_moments = measure.moments(im_watershed_pad)
                    im_centroids_shift = (-(int(im_pad.shape[0] / 2) - int(round(im_moments[1, 0] / im_moments[0, 0] + 0.1))),
                                   -(int(im_pad.shape[1] / 2)  - int(round(im_moments[0, 1] / im_moments[0, 0] + 0.1))))
                    #
                    # find out physical linear pixel size
                    pixelsize_dr0 = round(1e3 * np.sqrt(linear_object_size**2 / np.count_nonzero(im_watershed_pad)), 0)
                    self.metadata['Pixel size object domain, m'] = pixelsize_dr0
                    #
                    # centre object's distribution
                    # (with or without the application of the apodization filter)
                    if apodization is False:
                        self.image = warp(im_pad,
                                          AffineTransform(translation=im_centroids_shift),
                                          mode='wrap',
                                          preserve_range=True)
                        print('Object domain: Image centred. Apodization was NOT applied. Linear pixel size is ', pixelsize_dr0, "nm")
                        self.metadata['Apodization filter applied?'] = 'no'
                        #
                        if plot_progress is True:
                            plt.imshow(self.image)
                            plt.title('Centred image')
                            plt.colorbar()
                            plt.show()
                    else:
                        # compute apodization filter
                        image_apodization_filter = warp(im_watershed_pad,
                                                        AffineTransform(translation=im_centroids_shift),
                                                        mode='wrap',
                                                        preserve_range=True)
                        image_apodization_filter = gaussian(image_apodization_filter,
                                                            sigma=3,
                                                            preserve_range=True,
                                                            truncate=2.0)
                        image_apodization_filter = image_apodization_filter / np.max(np.max(image_apodization_filter))
                        self.image_apodization_filter = image_apodization_filter
                        #
                        if plot_progress is True:
                            plt.imshow(image_apodization_filter)
                            plt.title('Apodization filter')
                            plt.colorbar()
                            plt.show()
                        #
                        self.image = image_apodization_filter * warp(im_pad,
                                                                              AffineTransform(translation=im_centroids_shift),
                                                                              mode='wrap',
                                                                              preserve_range=True)
                        print("Object domain: Image centred. Apodization filter was applied. Linear pixel size is ", pixelsize_dr0, "nm")
                        self.metadata['Apodization filter applied?'] = 'yes'
                        #
                        if plot_progress is True:
                            plt.imshow(self.image)
                            plt.title('Centred and apodized image')
                            plt.colorbar()
                            plt.show()
                    self.metadata['Image centred and padded?'] = 'yes'
                    self.metadata['Linear size of the object, m'] = linear_object_size

                    return im_centroids_shift, pixelsize_dr0
                    #
                else:
                    raise ValueError('Specified number of pixels for zero-padding is too low!')

            else:
                raise ValueError('Argument(s) missing!')
        else:
            raise ValueError('Read the image data first!')

    def segment_image_watershed(self, str_element_size = 100, linear_object_size=100e-6, plot_progress = False):
        """
        Segments object-domain image using watershed algorithm (the existence of only a single blob is assumed).
        Finds its physical linear pixel size and saves this information into image's metadata.
        ---
        Parameters
        ---
        str_element_size : int, optional
            Linear size of the structuring element
            Default is 100 pixels.
        linear_object_size: float, optional
            Physical linear size of the (non-zero-valued) input object distribution, in m.
            Default is 100 micrometer.
        plot_progress: bool, optional
            Plot images.
            Default is False
        """
        if self.image is not None:
            if self.image_binary is not None:
                #
                if plot_progress is True:
                    plt.imshow(self.image)
                    plt.title('image before segmentation')
                    plt.colorbar()
                    plt.show()
                    plt.imshow(self.image_binary)
                    plt.title('binary image')
                    plt.colorbar()
                    plt.show()
                #
                # get structuring element
                kernel = np.ones((str_element_size, str_element_size), np.float64)
                #
                # perform morphological opening to clean up the binary image
                opening = cv2.morphologyEx(self.image_binary.astype(np.float64), cv2.MORPH_OPEN, kernel, iterations=1)
                if plot_progress is True:
                    plt.imshow(opening)
                    plt.title('binary image after opening')
                    plt.colorbar()
                    plt.show()
                # determine sure foreground area
                sure_fg = cv2.erode(opening, kernel, iterations=1)
                if plot_progress is True:
                    plt.imshow(sure_fg)
                    plt.title('max pixel values show sure_fg')
                    plt.colorbar()
                    plt.show()
                # determine sure background area
                sure_bg = cv2.dilate(opening, kernel, iterations=1)
                if plot_progress is True:
                    plt.imshow(sure_bg)
                    plt.title('min pixel values show sure_bg')
                    plt.colorbar()
                    plt.show()
                # determine bondary (unknown) region
                unknown = sure_bg - sure_fg
                if plot_progress is True:
                    plt.imshow(unknown)
                    plt.title('unknown')
                    plt.colorbar()
                    plt.show()
                #
                # label markers
                _, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
                # add one to all markers to ensure background is not 0, but 1
                markers = markers + 1
                # mark the region of unknown with zero
                markers[unknown == 1] = 0
                markers = markers.astype('int32')
                if plot_progress is True:
                    plt.imshow(markers)
                    plt.title('markers')
                    plt.colorbar()
                    plt.show()
                #
                # filter, renormalise and convert input image to 8 bit BGR
                image4watershed = np.copy(self.image)
                image4watershed = filters.gaussian(image4watershed, sigma = 3)
                image4watershed = image4watershed/np.max(image4watershed)
                image4watershed = 255 * image4watershed
                image4watershed = cv2.cvtColor(image4watershed.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                if plot_progress is True:
                    plt.imshow(image4watershed)
                    plt.title('image converted to 8 bit BGR used in watershed')
                    plt.colorbar()
                    plt.show()
                #
                # apply watershed algorithm
                image_segmented = cv2.watershed(image4watershed, markers)
                if plot_progress is True:
                    plt.imshow(image_segmented)
                    plt.title('segmented regions')
                    plt.colorbar()
                    plt.show()
                #
                # normalise segmented image
                image_segmented[image_segmented <2]=0
                image_segmented[image_segmented == 2] = 1
                if plot_progress is True:
                    plt.imshow(image_segmented)
                    plt.title('image with segmented background set to 0')
                    plt.colorbar()
                    plt.show()
                #
                # save normalised segmented image
                self.image_segmented = image_segmented
                #
                # apply normalised segmented image as a mask
                image = image_segmented * self.image
                if plot_progress is True:
                    plt.imshow(image)
                    plt.title('image with segmented background set to 0')
                    plt.colorbar()
                    plt.show()
                #
                # find out physical linear pixel size
                pixelsize_dr0 = round(1e3 * np.sqrt(linear_object_size**2 / np.count_nonzero(image_segmented)), 0)
                self.metadata['Pixel size object domain, m'] = pixelsize_dr0
                #
                self.metadata['Linear size of the object, m'] = linear_object_size

                return pixelsize_dr0
                    #
            else:
                raise ValueError("Binarize the input image first!")
        else:
            raise ValueError('Read the image data first!')


    def segment_image_opening(self, str_element_size = 100, linear_object_size=100e-6, plot_progress = False):
        """
        Segments object-domain image using morphological opening algorithm (the existence of only a single blob is assumed).
        Finds its physical linear pixel size and saves this information into image's metadata.

        ---
        Parameters
        ---
        str_element_size : int, optional
            Linear size of the structuring element
            Default is 100 pixels.
        linear_object_size: float, optional
            Physical linear size of the (non-zero-valued) input object distribution, in m.
            Default is 100 micrometer.
        plot_progress: bool, optional
            Plot images.
            Default is False
        """
        if self.image is not None:
            if self.image_binary is not None:
                #
                if plot_progress is True:
                    plt.imshow(self.image)
                    plt.title('self.image')
                    plt.colorbar()
                    plt.show()
                #
                # get the structuring element
                kernel = np.ones((str_element_size, str_element_size), np.float64)
                #
                # get normalised segmented image by morphological opening of the binary image
                image_segmented = cv2.morphologyEx(self.image_binary.astype(np.float64), cv2.MORPH_OPEN, kernel, iterations=1)
                # plot the result
                if plot_progress is True:
                    plt.imshow(self.image_binary)
                    plt.title('binary image to be segmented')
                    plt.colorbar()
                    plt.show()
                    plt.imshow(image_segmented)
                    plt.title('binary image after morphological opening')
                    plt.colorbar()
                    plt.show()
                #
                # save normalised segmented image
                self.image_segmented = image_segmented
                #
                # use normalised segmented image as a mask
                self.image = image_segmented * self.image
                if plot_progress is True:
                    plt.imshow(self.image)
                    plt.title('image after segmentation by opening')
                    plt.colorbar()
                    plt.show()
                #
                # find out physical linear pixel size using segmented image
                pixelsize_dr0 = round(1e3 * np.sqrt(linear_object_size**2 / np.count_nonzero(image_segmented)), 0)
                self.metadata['Pixel size object domain, m'] = pixelsize_dr0
                self.metadata['Linear size of the object, m'] = linear_object_size

                return pixelsize_dr0
            else:
                raise ValueError('Compute the binary image first!')
        else:
            raise ValueError('Read the image data first!')

    def segment_image_kmeans(self,
                             n_clusters=2, init="k-means++",  n_init = 10, max_iter=300, tol = 1e-4,
                             verbose=0, random_state=None, copy_x=True,
                             algorithm='auto',  str_element_size = 10,  linear_object_size=100e-6, plot_progress = False):
        """
        Segments image using k-means clustering.
        Finds its physical linear pixel size and saves this information into image's metadata.

        ---
        Parameters
        ---
        n_clusters: int, optional
            Number of clusters in the image
            Default is 2 (background and foreground).
        max_iter: int, optional
            Maximum number of iterations of the k-means algorithm
            Default is 300
        algorithm : str, optional
            k-means algorithm.
            Deafult is "auto".
        str_element_size : int, optional
            Linear size of the structuring element of the morphological closing applied after the completion of clustering
            Default is 10 pixels.
        linear_object_size: float, optional
            Physical linear size of the (non-zero-valued) input object distribution, in m.
            Default is 100 micrometer.
        plot_progress: bool, optional
            Plot images.
            Default is False
        """
        if self.image is not None:
            #
            # normalise input image and reshape it
            image = np.copy(self.image)
            image = image / np.max(image)
            image_reshaped = image.reshape(self.image.shape[0] * self.image.shape[1], 1)
            #
            # apply k-means algorithm and reshape clustered image
            kmeans = KMeans(n_clusters=n_clusters, init=init, n_init = n_init, max_iter=max_iter, tol = tol,
                            verbose=verbose, random_state=random_state, copy_x=copy_x,
                            algorithm=algorithm).fit(image_reshaped)
            print("labels ",np.unique(kmeans.labels_))
            print("centeres ",np.unique(kmeans.cluster_centers_))
            #
            image_clustered = kmeans.labels_
            #
            # because the labels are assigned randomly, we must re-label the image so that the central region (the smallest region) == 1
            # count the number of pixels in each cluster and re-assign the label accordingly
            _, counts_0 = np.unique(image_clustered[image_clustered == 0], return_counts=True)
            _, counts_1 = np.unique(image_clustered[image_clustered == 1], return_counts=True)
            image_clustered = image_clustered * (counts_1 < counts_0) + (np.abs(image_clustered - 1))*(counts_1 > counts_0)
            image_clustered_reshaped = image_clustered.reshape(self.image.shape[0], self.image.shape[1])
            #
            if plot_progress is True:
                plt.imshow(image_clustered_reshaped)
                plt.title('image after segmentation by k-means clustering')
                plt.colorbar()
                plt.show()
            #
            # fill in the gaps in the clustered image by morphological closing
            kernel = np.ones((str_element_size, str_element_size), np.float64)
            image_segmented  = cv2.morphologyEx(image_clustered_reshaped, cv2.MORPH_CLOSE, kernel)
            if plot_progress is True:
                plt.imshow(image_segmented)
                plt.title('segmented image after morphological closing')
                plt.colorbar()
                plt.show()
            #
            # save normalised segmented image
            self.image_segmented = image_segmented
            #
            # find out physical linear pixel size using segmented image
            pixelsize_dr0 = round(1e3 * np.sqrt(linear_object_size ** 2 / np.count_nonzero(image_segmented)), 0)
            self.metadata['Pixel size object domain, m'] = pixelsize_dr0
            self.metadata['Linear size of the object, m'] = linear_object_size
            #
            return pixelsize_dr0
        #
        else:
            raise ValueError('Read the image data first!')

    def centre_image(self, npixels_pad=2000, apodization=False, plot_progress = False):
        """
        Centers object-domain image after its segmentation.
        Completes zero-padding of the original image to a specified linear number of pixels
        Applies an apodization filter to smooth boundaries of the object distribution (optional)

        ---
        Parameters
        ---
        npixels_pad: int, optional
            Linear number of pixels in the zero-padded object-domain image.
            Default is 2000.
        apodization: bool, optional
            If True, the boundaries of the object distribution will be smoothed using Gaussian filter
            with standard deviation = 3 pixels and truncation of the filter's boundaries to 2
            (fixed at the moment, but may/should be changed in the future)
            Default is False
        plot_progress: bool, optional
            Plot images.
            Default is False
        """
        if self.image is not None:
            if self.image_segmented is not None:
                if npixels_pad is not None:
                    if npixels_pad >= self.image.shape[0] or npixels_pad >= self.image.shape[1]:
                        #
                        if plot_progress is True:
                            plt.imshow(self.image)
                            plt.title('self.image')
                            plt.colorbar()
                            plt.show()
                        #
                        #now pad original image and watershed image
                        image_pad = pad(self.image,
                                    ((0, npixels_pad - self.image.shape[0]),
                                    (0, npixels_pad - self.image.shape[1])),
                                    mode='constant')
                        image_segmented_pad = pad(self.image_segmented,
                                               ((0, npixels_pad - self.image.shape[0]),
                                               (0, npixels_pad - self.image.shape[1])),
                                               mode='constant')
                        #
                        if plot_progress is True:
                            plt.imshow(image_segmented_pad)
                            plt.title('Padded segmented image')
                            plt.colorbar()
                            plt.show()
                        print("Object domain: Input and segmented images were padded to ", image_pad.shape[0], "X", image_pad.shape[1], "pixels.")
                        self.metadata['Linear number of pixels in the zero-padded real-space image']= npixels_pad
                        #
                        # compute the relative shift the segmented region's centroid w.r.t. the centre of the computational domain
                        # (the tuple's items must be multiplied by -1 to ensure that in AffineTransform the shift occurs in the right direction)
                        # the moments are rounded to ensure the centroid is not between the pixels
                        # add 0.1 to round(im_moments ... / im_moments ... ) to round as per mathematical definition
                        im_moments = measure.moments(image_segmented_pad)
                        im_centroids_shift = (-(int(image_segmented_pad.shape[0] / 2) - int(round(im_moments[1, 0] / im_moments[0, 0] + 0.1))),
                                       -(int(image_segmented_pad.shape[1] / 2)  - int(round(im_moments[0, 1] / im_moments[0, 0] + 0.1))))
                        #
                        # centre object's distribution
                        # without application of apodization filter
                        if apodization is False:
                            # center padded image
                            self.image = warp(image_pad,
                                              AffineTransform(translation=im_centroids_shift),
                                              mode='wrap',
                                              preserve_range=True)
                            print('Object domain: Image centred. Apodization was NOT applied. Linear pixel size is ', self.metadata['Pixel size object domain, m'], "nm")
                            self.metadata['Apodization filter applied?'] = 'no'
                            #
                            if plot_progress is True:
                                plt.imshow(self.image)
                                plt.title('Centred image')
                                plt.colorbar()
                                plt.show()
                        else:
                            # with application of apodization filter
                            # first center padded segmented image
                            image_segmented_centered = warp(image_segmented_pad,
                                                            AffineTransform(translation=im_centroids_shift),
                                                            mode='wrap',
                                                            preserve_range=True)
                            # then compute apodization filter
                            image_apodization_filter = gaussian(image_segmented_centered ,
                                                                sigma=3,
                                                                preserve_range=True,
                                                                truncate=2.0)
                            image_apodization_filter = image_apodization_filter / np.max(np.max(image_apodization_filter))
                            self.image_apodization_filter = image_apodization_filter
                            #
                            if plot_progress is True:
                                plt.imshow(image_apodization_filter)
                                plt.title('Apodization filter')
                                plt.colorbar()
                                plt.show()
                            #
                            # center padded image and apply apodization filter to it
                            self.image = image_apodization_filter * warp(image_pad,
                                                                                  AffineTransform(translation=im_centroids_shift),
                                                                                  mode='wrap',
                                                                                  preserve_range=True)
                            print("Object domain: Image centred. Apodization filter was applied. Linear pixel size is ", self.metadata['Pixel size object domain, m'], "nm")
                            self.metadata['Apodization filter applied?'] = 'yes'
                            #
                            if plot_progress is True:
                                plt.imshow(self.image)
                                plt.title('Centred and apodized image')
                                plt.colorbar()
                                plt.show()
                        self.metadata['Image centred and padded?'] = 'yes'

                        return im_centroids_shift
                        #
                    else:
                        raise ValueError('Specified number of pixels for zero-padding is too low!')

                else:
                    raise ValueError('Number of pixels to zero-pad to is None!')
            else:
                raise ValueError('Segment the image data first!')
        else:
            raise ValueError('Read the image data first!')

    def subtract_background(self, noise_mean = False, patch_corner = (0,0), patch_size = (100,100), counts = 0, zoom = 1,  log_scale = False, estimate_only = True, plot_progress = False):
        """
        Subtracts constant background given a number of counts

        ---
        Parameters
        ---
        noise_mean : bool, optional
            If False, the number of counts to subtract is set by "counts"
            If True, the number of counts is the mean pixel value in the patch set by "patch_corner" and "patch_size"
            Default is False.
        patch_corner : tuple of int, optional
            Sets the upper left corner of the patch to estimate the noise mean
            Default is (0,0)
        patch_size : tuple of int, optional
            Sets the size of the path to estimate the noise mean
            Default is (100,100).
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
        plot_progress: bool, optional
            Plot background-free image for estimate_only = False.
            Default is False
        """
        if self.image is not None:
            #
            if noise_mean is True:
                counts_to_subtract = np.mean(self.image[patch_corner[0]:patch_corner[0]+patch_size[0], patch_corner[1]:patch_corner[1]+patch_size[1]])
            else:
                counts_to_subtract = counts
            #
            # subtract background and threshold the resulting image
            im_bgfree = np.copy(self.image)
            im_bgfree = im_bgfree - counts_to_subtract
            im_bgfree[im_bgfree < 0] = 0
            #
            if estimate_only == True:
                # estimate how the image would look like if the background were subtracted
                if plot_progress == True:
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
                    print("Object domain: This is how the image looks like if the background were subtracted.")
                    print("Object domain: Background was set to ", counts)
            else:
                #
                # subtract background and save changes
                self.image = im_bgfree
                self.metadata['Background subtracted?'] = 'yes'
                #
                if plot_progress is True:
                    if log_scale == True:
                        plt.imshow(np.log(self.image))
                        print('Object domain: Plotted in log scale')
                    else:
                        plt.imshow(self.image)
                    plt.axis([im_bgfree.shape[0] // 2 - im_bgfree.shape[0] // 2 // zoom,
                              im_bgfree.shape[0] // 2 + im_bgfree.shape[0] // 2 // zoom,
                              im_bgfree.shape[1] // 2 - im_bgfree.shape[1] // 2 // zoom,
                              im_bgfree.shape[1] // 2 + im_bgfree.shape[1] // 2 // zoom])
                    plt.gca().invert_yaxis()
                    plt.title("Object domain: Image after the background subtraction.")
                    plt.colorbar()
                    plt.show()
                print("Object domain: Background of", counts, "counts was subtracted.")
            #
            # binarize image
            image_binary = np.copy(im_bgfree)
            image_binary[image_binary > 0] = 1
            self.image_binary = image_binary
            #
            return im_bgfree
        else:
            raise ValueError('Read the image data and centre it first!')

    def resample_image(self, fieldofview = 17, npixels_kspace = 500, pixelsize_dr0 = None, lambd = 880e-9, estimate_only = True):
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
            Pixel size in experimental object-domain image, in m.
            If it is None, the value will be read from metadata (saved to metadata after centering and segmentation of the image).
            If the value in metadata is None, there will be an error message.
            Alternatively, one can set the pixel size manually.
            Default is None.
        lambd: float, optional
            Wavelength of light, in m
            Default is 880e-9
        estimate_only: bool, optional
            False will irreversibly resample the input image.
            True will only estimate the downsampling ratio.
            Default is True.
        ---
        Returns
        ---
        npixels_final : int
            Final linear number of pixels in the object-domain image
        """
        if self.image is not None:
            if self.metadata['Image centred and padded?'] == 'yes':
                #
                # compute pixel size in Fourier domain
                alpha = fieldofview * np.pi / 180
                pixelsize_dk = (np.sin(alpha) * 2 * np.pi / lambd) / npixels_kspace
                print("pixelssize_dk = ",pixelsize_dk)
                self.metadata['Pixel size Fourier domain, 1/nm'] = pixelsize_dk * 1e9
                #
                # read out linear number of pixels in object domain
                npixels_pad = self.metadata['Linear number of pixels in the zero-padded real-space image']
                #
                # compute pixel size in object domain as set by the discrete Fourier transform
                pixelsize_dr_pad = 2*np.pi / (npixels_pad * pixelsize_dk)
                self.metadata['Pixel size object domain from padded Fourier data, 1/nm'] = pixelsize_dr_pad * 1e9
                #
                #compute the downsampling ratio
                downsampling = round(pixelsize_dr_pad / pixelsize_dr0)
                #
                #compute preliminary final linear number of pixels as set by the downsampling ratio and experimental
                #pixel sizes in Fourier and object spaces
                npixels_final = int(round(2 * np.pi / (downsampling * pixelsize_dr0 * pixelsize_dk)))
                #
                if estimate_only == False:
                    print('Object domain: Input image shape is ', self.image.shape[0],'X', self.image.shape[1])
                    #
                    self.image = resize(np.array(self.image), (self.image.shape[0] // downsampling, self.image.shape[1] // downsampling), anti_aliasing=True)
                    print('Object domain: Image was resampled and its current shape is', self.image.shape)
                    npixels_to_pad_final0 = int((npixels_final - self.image.shape[0]) / 2)
                    npixels_to_pad_final1 = int((npixels_final - self.image.shape[1]) / 2)
                    self.image = pad(np.array(self.image), ((npixels_to_pad_final0, npixels_to_pad_final1), (npixels_to_pad_final0, npixels_to_pad_final1)), mode='constant')
                    #
                    #check if the final dimensions are of equal length
                    if self.image.shape[0] == self.image.shape[1]:
                        #
                        #set the final linear number of pixels by the actual image size
                        npixels_final = self.image.shape[0]
                        print('Object domain: Image was resampled with the downsampling ratio =', downsampling,
                              'and zero-padded to npixels_final X npixels_final=', self.image.shape[0], 'X', self.image.shape[1], 'pixels')
                    #
                    #if the final dimensions are not of the equal length, fix them by deleting the corresponding last column
                    else:
                        if self.image.shape[0] > self.image.shape[1]:
                            self.image = self.image[:-1,:]
                        elif self.image.shape[0] < self.image.shape[1]:
                            self.image = self.image[:,:-1]
                        #
                        # set the final linear number of pixels by the actual image size
                        npixels_final = self.image.shape[0]
                        print('Object domain: Image was resampled with the downsampling ratio =', downsampling,
                              'and zero-padded to npixels_final X npixels_final=', self.image.shape[0], 'X',
                              self.image.shape[1], 'pixels')
                #
                elif estimate_only == True:
                    print('Object domain: Downsampling ratio =', downsampling)
                self.metadata['Wavelength of light, m'] = lambd
                self.metadata['Field of view, deg'] = fieldofview
                self.metadata['Number of pixels within the field of view'] = npixels_kspace
                #
            else:
                raise ValueError('Centre and zero-pad image distribution!')
        else:
            raise ValueError('Read the image data first!')
        return pixelsize_dk, pixelsize_dr_pad, downsampling, npixels_final

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
                self.metadata.to_csv(os.path.join(pathtosave, outputfilename[:-4] + '.csv'), sep='\t', header=False)
                self.filename = outputfilename
                print('Object domain: Image was saved in tif file under ', os.path.join(pathtosave, outputfilename))
                print('Object domain: Metadata was saved in csv file under ', os.path.join(pathtosave, outputfilename[:-4] + '.csv'))
            else:
                raise ValueError('Invalid path! Please specify a valid path to save data as tif file.')
        else:
            raise ValueError('Read the image data first!')