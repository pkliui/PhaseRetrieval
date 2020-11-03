#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for phase retrieval

"""

import glob
import os
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from IPython import display
from matplotlib import pyplot as plt
from skimage import io
from skimage.exposure import rescale_intensity

from PhaseRetrieval.classes.rspacemetadata import RSpaceMetadata
from PhaseRetrieval.classes.kspacemetadata import KSpaceMetadata

class PhaseRetrieval(object):

    def __init__(self, filename_rspace=None,
                 filename_kspace=None,
                 amp_rspace = None,
                 amp_kspace = None,
                 delimiter = None,
                 image_rspace_kk_prime = None,
                 image_rspace_kk1 = None,
                 image_kspace_kk = None,
                 image_kspace_kk_prime = None,
                 rms_error = None):
        """
        Initializes phase retrieval class

        ---
        Parameters
        ---
        filename_rspace: str, optional
            Path used to load the object-domain image.
            If None, an empty class is created.
            If `amp_rspace` argument is provided, the image will be initialized from the `amp_rspace` argument.
            Default is None.
        filename_kspace: str, optional
            Path used to load the Fourier-domain image.
            If None, an empty class is created.
            If `amp_kspace` argument is provided, the image will be initialized from the `amp_kspace` argument.
            Default is None.
        amp_rspace : ndarray, optional
            2D array to initialize the object-domain image.
            If None, an empty class is created.
            Default is None.
        amp_kspace : ndarray, optional
            2D array to initialize the Fourier-domain image.
            If None, an empty class is created.
            Default is None.
        delimiter: str, optional
            Delimiter used in the csv file.
            If None, an empty class is created.
            Default is None.
        image_rspace_kk_prime : ndarray
            Reconstructed (computed) complex-valued object-domain distribution
            Default is None.
        image_rspace_kk1 : ndarray
            Constrained complex-valued object-domain distribution
            Default is None.
        image_kspace_kk : ndarray
            Reconstructed (computed) complex-valued Fourier-domain distribution
            Default is None.
        image_kspace_kk_prime : ndarray
            Constrained complex-valued Fourier-domain distribution
            Default is None.
        rms_error : 1D array
            Root mean squared (RMS) error distribution
            Default is None.
        """

        self.filename_rspace = filename_rspace
        self.filename_kspace = filename_kspace
        self.delimiter = delimiter
        self.amp_rspace = amp_rspace
        self.amp_kspace = amp_kspace
        self.image_rspace_kk_prime = image_rspace_kk_prime
        self.image_rspace_kk1 = image_rspace_kk1
        self.image_kspace_kk = image_kspace_kk
        self.image_kspace_kk_prime = image_kspace_kk_prime
        self.rms_error = rms_error
        self.metadata_rspace = RSpaceMetadata()
        self.metadata_kspace = KSpaceMetadata()
        #
        #read object-domain image
        if amp_rspace is None and filename_rspace is not None and os.path.exists(filename_rspace):
            if filename_rspace[-4:] == '.csv':
                if delimiter is not None:
                    print('file type is csv')
                    print()
                    # then read data from file
                    self.read_from_csv(filename=filename_rspace, delimiter=delimiter, image_domain='object')
                else:
                    raise ValueError('Delimiter must be specified')
            elif filename_rspace[-4:] == '.tif':
                print('file type is tif')
                self.read_from_tif(filename=filename_rspace, image_domain='object')
            else:
                raise ValueError('Invalid file type! Must be either csv or tif!')
        #
        #read Fourier-domain image
        if amp_kspace is None and filename_kspace is not None and os.path.exists(filename_kspace):
            if filename_kspace[-4:] == '.csv':
                if delimiter is not None:
                    print('file type is csv')
                    print()
                    # then read data from file
                    self.read_from_csv(filename=filename_kspace, delimiter=delimiter, image_domain='Fourier')
                else:
                    raise ValueError('Delimiter must be specified')
            elif filename_kspace[-4:] == '.tif':
                print('file type is tif')
                self.read_from_tif(filename=filename_kspace, image_domain='Fourier')
            else:
                raise ValueError('Invalid file type! Must be either csv or tif!')
        #
        #convert read intensities to amplitudes
        self.amp_rspace = np.sqrt(self.amp_rspace)
        self.amp_kspace =  np.sqrt(self.amp_kspace)
        #
        #check energy conservation
        print("checking energy conservation ... ")
        print("rs energy", (np.power(self.amp_rspace,2)).sum())
        print("ks energy", (np.power(self.amp_kspace,2)).sum())
        print("rs energy * Ntot", (np.power(self.amp_rspace,2)).sum() *self.amp_rspace.shape[0] * self.amp_rspace.shape[1])

    def __repr__(self):
        return "Class for phase retrieval"

    @staticmethod
    def ft2d_centered(input_function):
        """
        Centred Fourier transform  as described in
        T. Latyvevskaia and H.-W. Fink  "Practical algorithms for simulation and reconstruction of digital in-line holograms",
        Appl. Optics 54, 2424 - 2434 (2015)

        ---
        Parameters
        ---
        input_function: 2D array
            2D array to compute the Fourier transform of
        ---
        Return
        ---
        output_function: 2D array
            Centred Fourier transform of the input 2D array
        """
        ii = np.linspace(0, input_function.shape[0] - 1, input_function.shape[0])
        jj = np.linspace(0, input_function.shape[1] - 1, input_function.shape[1])
        x,y = np.meshgrid(ii, jj, sparse=True)
        phase_factor = np.exp(1j * np.pi * (x + y))
        #
        #compute Fourier transform
        ft = np.fft.fft2(phase_factor * input_function)
        output_function = np.array(phase_factor * ft).reshape(input_function.shape)
        #
        return output_function

    @staticmethod
    def ift2d_centered(input_function):
        """
        Centred inverse Fourier transform  as described in
        T. Latyvevskaia and H.-W. Fink  "Practical algorithms for simulation and reconstruction of digital in-line holograms",
        Appl. Optics 54, 2424 - 2434 (2015)

        ---
        Parameters
        ---
        input_function: 2D array
            2D array to compute the inverse Fourier transform of
        ---
        Return
        ---
        output_function: 2D array
            Centred inverse Fourier transform of the input 2D array
        """
        ii = np.linspace(0, input_function.shape[0] - 1, input_function.shape[0])
        jj = np.linspace(0, input_function.shape[1] - 1, input_function.shape[1])
        x,y = np.meshgrid(ii, jj, sparse=True)
        phase_factor = np.exp(-1j * np.pi * (x + y))
        #
        #compute inverse Fourier transform
        ift = np.fft.ifft2(phase_factor * input_function)
        output_function = np.array( phase_factor * ift).reshape(input_function.shape)
        #
        return output_function

    def read_from_tif(self, filename, image_domain=None):
        """
        Reads image data from a tif file.

        ---
        Parameters
        ---
        filename: str
            Path used to load the image.
        image_domain: str
            Image domain: either 'object' or 'Fourier'
            Default is None
        """
        if os.path.exists(filename):
                if image_domain is not None:
                    # read the image from file
                    # as there is no header in images, the header is set to None
                    if image_domain is 'object':
                        self.amp_rspace = io.imread(filename)
                        print('object-domain data were read as ', filename)
                        print()
                        self.filename_rspace = filename
                        if os.path.exists(filename[:-4] + '.csv'):
                            self.metadata_rspace.read_from_csv(filename[:-4] + '.csv')
                            print('object-domain metadata were read as ', filename[:-4] + '.csv')
                            print()
                        else:
                            warnings.warn("Object-domain metadata file is not found!", Warning)
                        #
                    elif image_domain is 'Fourier':
                        self.amp_kspace = io.imread(filename)
                        print('Fourier-domain data were read as ', filename)
                        print()
                        self.filename_kspace = filename
                        #
                    else:
                        ValueError('Image domain must be either "object" or "Fourier"!')
                else:
                    ValueError('Specify the image type!')
        else:
            raise ValueError('Invalid path! File does not exist!')

    def read_from_csv(self, filename, delimiter = ',', image_domain=None):
        """
        Reads image data from a csv file.

        ---
        Parameters
        ---
        filename: str
            Path used to load the image.
        delimiter: str
            Delimiter used in the csv file.
            Default is comma ','.
        image_domain: str
            Image domain: either 'object' or 'Fourier'
            Default is None
        """
        if os.path.exists(filename):
            if delimiter is not None:
                if image_domain is not None:
                        # read the image from file
                        # as there is no header in images, the header is set to None
                        if image_domain is 'object':
                            self.amp_rspace = pd.read_csv(filename, delimiter, header=None)
                            print('object-domain data were read as ', filename)
                            print()
                            #get the values
                            self.amp_rspace = self.amp_rspace.values
                            self.filename_rspace = filename
                            self.delimiter = delimiter
                            if os.path.exists(filename[:-4] + '.csv'):
                                self.metadata = pd.read_csv(filename[:-4] + '_meta.csv')
                                print('object-domain metadata were read as ', filename[:-4] + '_meta.csv')
                            else:
                                warnings.warn("Object-domain metadata file is not found!", Warning)
                            #
                        elif image_domain is 'Fourier':
                            self.amp_kspace = pd.read_csv(filename, delimiter, header=None)
                            print('Fourier-domain data were read as ', filename)
                            #get the values
                            self.amp_kspace = self.amp_kspace.values
                            self.filename_kspace = filename
                            self.delimiter = delimiter
                            if os.path.exists(filename[:-4] + '.csv'):
                                self.metadata = pd.read_csv(filename[:-4] + '_meta.csv')
                                print('Fourier-domain metadata were read as ', filename[:-4] + '_meta.csv')
                            else:
                                warnings.warn("Fourier-domain metadata file is not found!", Warning)
                            #
                        else:
                            ValueError('Image domain must be either "object" or "Fourier"!')
                else:
                    ValueError('Specify the image type!')
            else:
                ValueError('Specify the delimiter!')
        else:
            raise ValueError('Invalid path! File does not exist!')

    def plot_input_images(self, zoom = 1, image_domain = None):
        """
        Plots input image data

        ---
        Parameters
        ---
        zoom: int, optional
            Zoom factor to zoom into the plot
            Default is 1 (no zoom)
        image_domain: str
            Image domain: either 'object' or 'Fourier'
            Default is None
        """
        if image_domain is not None:
            if image_domain is 'object' and self.amp_rspace is not None:
                # plot the image from file
                image = self.amp_rspace
                plt.imshow(image)
                plt.axis([image.shape[0] // 2 - image.shape[0] // 2 // zoom,
                          image.shape[0] // 2 + image.shape[0] // 2 // zoom,
                          image.shape[1] // 2 - image.shape[1] // 2 // zoom,
                          image.shape[1] // 2 + image.shape[1] // 2 // zoom])
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.title("Input object-domain amplitude")
            elif image_domain is 'Fourier' and self.amp_kspace is not None:
                # plot the image from file
                image = self.amp_kspace
                plt.imshow(image)
                plt.axis([image.shape[0] // 2 - image.shape[0] // 2 // zoom,
                          image.shape[0] // 2 + image.shape[0] // 2 // zoom,
                          image.shape[1] // 2 - image.shape[1] // 2 // zoom,
                          image.shape[1] // 2 + image.shape[1] // 2 // zoom])
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.title("Input Fourier-domain amplitude")
                plt.show()
            else:
                raise ValueError('Read data first.')
        else:
            raise ValueError('Specify image domain first. Must be either image_domain="object" or image_domain= "Fourier"!')

    def gerchberg_saxton(self, gs_steps = None, plot_progress = False, plot_every_kth_iteration = 1, zoom = 1):
        """
        Gechberg_Saxton algorithm
        as described in R. W. Gerchberg and W. O. Saxton,
        "A practical algorithm for the determination of the phase from image and diffraction plane pictures”,
         Optik 35, 237 (1972).

        ---
        Parameters
        ---
        gs_steps: int
            Number of iterations in GS algorithm
            Default is None.
        plot_progress : bool, optional
            False will prevent algorithm from plotting the progress.
            True will plot the progress of the algorithm.
            Default is False.
        plot_every_kth_iteration : int, optional
            Plot the progress at each k-th iteration, where k is given by the argument "plot_every_kth_iteration"
            Default is 1.
        zoom: int, optional
            Zoom factor to zoom into the 2D plot
            Default is 1 (no zoom)
        ---
        Returns (optional)
        ---
        np.abs(image_kspace_kk_prime): 2D array
            Amplitude of the reconstructed Fourier data
        np.angle(image_kspace_kk_prime): 2D array
            Phase of the reconstructed Fourier data
        np.abs(image_rspace_kk_prime): 2D array
            Amplitude of the reconstructed object distribution
        np.angle(image_rspace_kk_prime): 2D array
            Phase of the reconstructed object distribution
        rms_error: 1D array
            Root mean squared error distribution in Fourier domain
        """
        #
        #random phase distribution in Fourier space (between -pi and pi) and initial guesses in Fourier and object domains
        phase_kspace0 = (2.0 * np.random.rand(self.amp_kspace.shape[0], self.amp_kspace.shape[1]) - 1) * np.pi
        image_kspace0 = self.amp_kspace * np.exp(1j*phase_kspace0)
        image_rspace0 = self.ft2d_centered(input_function = image_kspace0)
        #
        #initial input object distribution
        image_rspace_kk = np.copy(image_rspace0)
        #
        #initialise object-domain output image
        image_rspace_kk_prime = None
        #
        #initialise root mean squared error (in Fourier domain)
        rms_error = []
        #
        #Gerchberg-Saxton algorithm
        for kk_gs in range(0, gs_steps):
            #
            print("Iteration %d \r" % (kk_gs+1))
            #
            #compute Fourier transform and apply constraints in Fourier domain
            image_kspace_kk = self.ft2d_centered(input_function = image_rspace_kk)
            image_kspace_kk_prime = self.amp_kspace * np.exp(1j * np.angle(image_kspace_kk))
            #
            #compute inverse Fourier transform
            image_rspace_kk_prime = self.ift2d_centered(input_function = image_kspace_kk_prime)
            #
            #apply constraints in object domain
            image_rspace_kk1 = self.amp_rspace * np.exp(1j * np.angle(image_rspace_kk_prime))
            #
            #use image_rspace_kk1 as a new input to compute image_kspace_kk
            image_rspace_kk = np.copy(image_rspace_kk1)
            #
            #root mean squared error (in Fourier domain)
            rms_error.append(np.sqrt(np.sum((np.abs(image_kspace_kk) - np.abs(self.amp_kspace))**2) /
                                         np.sum(np.abs(self.amp_kspace)**2)))
            #
            #plot results
            #
            if plot_progress is True:
                if (kk_gs+1) % plot_every_kth_iteration == 0:
                    #
                    print("Iteration %d \r" % (kk_gs + 1), " (plotting)")
                    #
                    # some manipulations with phase distribution to be able to plot it weighted with amplitude values
                    # important to normalise to 1, otherwise the plot will not be displayed correctly!
                    phase_kspace_kk_weighted = plt.cm.bwr(rescale_intensity(- np.angle(image_kspace_kk).min() + np.angle(image_kspace_kk), out_range=(0, 1)))
                    phase_kspace_kk_weighted[..., -1] = rescale_intensity(np.abs(image_kspace_kk), out_range=(0, 1))
                    #
                    phase_rspace_kk_prime_weighted = plt.cm.bwr(rescale_intensity(- np.angle(image_rspace_kk_prime).min() + np.angle(image_rspace_kk_prime), out_range=(0, 1)))
                    phase_rspace_kk_prime_weighted[..., -1] = rescale_intensity(np.abs(image_rspace_kk_prime), out_range=(0, 1))
                    #
                    phase_rspace_kk1_weighted = plt.cm.bwr(rescale_intensity(- np.angle(image_rspace_kk1).min() + np.angle(image_rspace_kk1), out_range=(0, 1)))
                    phase_rspace_kk1_weighted[..., -1] = rescale_intensity(np.abs(image_rspace_kk1), out_range=(0, 1))
                    #
                    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
                    #
                    # computed Fourier amplitude
                    im00 = ax[0,0].imshow(np.abs(image_kspace_kk))
                    ax[0,0].set_title("$|\mathrm{G}_k|$ - computed Fourier amplitude")
                    plt.colorbar(im00, ax=ax[0,0], fraction=0.046, pad=0.04)
                    ax[0,0].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    # computed Fourier phase
                    im01 = ax[0,1].imshow(phase_kspace_kk_weighted, cmap='seismic', vmin=np.angle(image_kspace_kk).min(), vmax=np.angle(image_kspace_kk).max())
                    ax[0,1].set_title("$arg(\mathrm{G}_k)$ - computed Fourier phase")
                    plt.colorbar(im01, ax=ax[0,1], fraction=0.046, pad=0.04)
                    ax[0,1].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    # constrained Fourier amplitude
                    im02 = ax[0,2].imshow(self.amp_kspace)
                    ax[0,2].set_title("$|\mathrm{F}|$ - constrained Fourier amplitude")
                    plt.colorbar(im02, ax=ax[0,2], fraction=0.046, pad=0.04)
                    ax[0,2].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    # RMSE
                    im03 = ax[0,3].plot(rms_error)
                    ax[0,3].set_ylim([0, 1])
                    ax[0,3].set_title("RMS error in Fourier domain")
                    #
                    # computed object-domain amplitude
                    im10 = ax[1,0].imshow(np.abs(image_rspace_kk_prime))
                    ax[1,0].set_title("$|\mathrm{g}_k'|$ - computed object amplitude")
                    plt.colorbar(im10, ax=ax[1,0], fraction=0.046, pad=0.04)
                    ax[1,0].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    # computed object-domain phase
                    im11 = ax[1,1].imshow(phase_rspace_kk_prime_weighted, cmap='seismic', vmin=np.angle(image_rspace_kk_prime).min(), vmax=np.angle(image_rspace_kk_prime).max())
                    ax[1,1].set_title("$arg(\mathrm{g}_k')$ - computed object phase")
                    plt.colorbar(im11, ax=ax[1,1], fraction=0.046, pad=0.04)
                    ax[1,1].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    # constrained object-domain amplitude
                    im12 = ax[1,2].imshow(self.amp_rspace)
                    ax[1,2].set_title("$|\mathrm{g}_{k+1}|$ - constrained object amplitude")
                    plt.colorbar(im10, ax=ax[1,2], fraction=0.046, pad=0.04)
                    ax[1,2].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    # constrained object-domain phase
                    im13 = ax[1,3].imshow(phase_rspace_kk1_weighted, cmap='seismic', vmin=np.angle(image_rspace_kk1).min(), vmax=np.angle(image_rspace_kk1).max())
                    ax[1,3].set_title("$arg(\mathrm{g}_{k+1})$ - constrained  object phase")
                    plt.colorbar(im11, ax=ax[1,3], fraction=0.046, pad=0.04)
                    ax[1,3].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    fig.tight_layout()
                    display.clear_output(wait=True)
                    plt.show()
                    #
                    print("Iteration %d \r" % (kk_gs + 1), " completed")
                else:
                    print("Iteration %d \r" % (kk_gs + 1), " completed")
            else:
                print("Iteration %d \r" % (kk_gs + 1), " completed")
        #
        #save results
        self.image_rspace_kk_prime = image_rspace_kk_prime
        self.image_rspace_kk1 = image_rspace_kk1
        self.image_kspace_kk = image_kspace_kk
        self.image_kspace_kk_prime = image_kspace_kk_prime
        self.rms_error = rms_error
        #
        #return np.abs(self.image_kspace_kk_prime), np.angle(self.image_kspace_kk_prime), np.abs(self.image_rspace_kk_prime),  np.angle(self.image_rspace_kk_prime), self.rms_error

    def check_oversampling(self, extrapolation = False):
        """
        Checks if the oversampling ratio is fulfilled for extrapolation in Fourier domain
        Following T. Latychevskaia, Reconstruction of missing information in diffraction patterns and holograms by iterative phase retrieval,
        Optics Communications, Vol. 452,  p. 56-67, 2019.
        """
        #
        #find the number of "missing" (zero-valued) pixels in Fourier domain amplitude image
        # and check whether the oversampling condition is fulfilled
        N_missing = len(np.argwhere(self.amp_kspace == 0))
        N_total = self.amp_kspace.shape[0] *  self.amp_kspace.shape[1]
        ff = N_missing / N_total
        #
        # pixel size in Fourier domain
        print(self.metadata_rspace)
        dk = (2 * np.pi / self.metadata_rspace['Wavelength of light, m']) * np.sin(self.metadata_rspace['Field of view, deg'] * np.pi / 180) / self.metadata_rspace['Number of pixels within the field of view']
        #
        #linear oversampling
        oversampling = 2 * np.pi / (dk * self.metadata_rspace['Linear size of the object, m'])
        print('linear oversampling ratio = ', oversampling)
        if oversampling > np.sqrt(2):
            print('linear oversampling condition is fulfilled')
        else:
            print('linear oversampling condition is NOT fulfilled')
        #
        if extrapolation is True:
            #check if the oversampling condition is fulfilled
            if ff < (1 - 2 / oversampling ** 2):
                print ('Oversampling condition for extrapolation is fulfilled.')
                print( 'The ratio of the number of missing pixels to the total number of pixels is f = ', ff)
                print( 'Reconstruction with phase retrieval (with extrapolation) is possible if f < ', 1 - 2 / oversampling ** 2)
            else:
                print ('Oversampling condition for extrapolation is NOT fulfilled.')
                print( 'The ratio of the number of missing pixels to the total number of pixels is f = ', ff)
                print( 'Reconstruction with phase retrieval (with extrapolation) is possible only if f < ', 1 - 2 / oversampling ** 2)
                print('Keep in mind that typically quality of reconstructions drastically degrades for high missing pixel numbers')

    def gerchberg_saxton_extrapolation(self, gs_steps = None, plot_progress = False, plot_every_kth_iteration = 1, zoom = 1):
        """
        Gechberg_Saxton algorithm with extrapolation in Fourier domain
        #
        GS algorithm is implemented as described in
        R. W. Gerchberg and W. O. Saxton,
        "A practical algorithm for the determination of the phase from image and diffraction plane pictures”,
         Optik 35, 237 (1972).
        #
        Extrapolation of the Fourier domain data follows some of the procedures described in T. Latychevskaia et al.,
        "Imaging outside the box: Resolution enhancement in X-ray coherent diffraction imaging by extrapolation of diffraction patterns"
        Appl. Phys. Lett. 107, 183102 (2015).

        ---
        Parameters
        ---
        gs_steps: int
            Number of iterations in GS algorithm
            Default is None.
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
        ---
        Returns (optional)
        ---
        np.abs(image_kspace_kk_prime): 2D array
            Amplitude of the reconstructed Fourier data
        np.angle(image_kspace_kk_prime): 2D array
            Phase of the reconstructed Fourier data
        np.abs(image_rspace_kk_prime): 2D array
            Amplitude of the reconstructed object distribution
        np.angle(image_rspace_kk_prime): 2D array
            Phase of the reconstructed object distribution
        rms_error: 1D array
            Root mean squared error distribution in Fourier domain

        """
        #
        #random phase distribution in Fourier space (between -pi and pi) and initial guesses in Fourier and object domains
        phase_kspace0= (2.0 * np.random.rand(self.amp_kspace.shape[0], self.amp_kspace.shape[1]) - 1) * np.pi
        image_kspace0 = self.amp_kspace * np.exp(1j*phase_kspace0)
        image_rspace0 = self.ft2d_centered(input_function = image_kspace0)
        #
        #initial input object distribution
        image_rspace_kk = np.copy(image_rspace0)
        #
        #initialise output object distribution
        image_rspace_kk_prime = None
        #
        #initialise root mean squared error (in Fourier domain)
        rms_error = []
        #
        #Gerchberg-Saxton algorithm
        for kk_gs in range(0, gs_steps):
            #
            print("Iteration %d \r" % (kk_gs+1))
            #
            #compute Fourier transform and apply constraints in Fourier domain
            image_kspace_kk = self.ft2d_centered(input_function = image_rspace_kk)
            image_kspace_kk_prime = self.amp_kspace * np.exp(1j * np.angle(image_kspace_kk))
            #
            #compute energy in Fourier domain
            energy_kspace_kk_prime = np.sum(np.sum(np.abs(image_kspace_kk_prime)**2))
            #check if the energies in both domains are equal
            #print('energy k space ', np.sum(np.sum(np.abs(image_kspace_kk) ** 2)))
            #print('energy real space ', image_rspace_kk.shape[0] * image_rspace_kk.shape[1] * np.sum(np.sum(np.abs(image_rspace_kk) ** 2)))
            #
            #replace missing pixels (here we set them to 0) by computed ones
            image_kspace_kk_prime[image_kspace_kk_prime==0] = image_kspace_kk[image_kspace_kk_prime==0]
            #
            #compute energy of the updated Fourier image and normalise it to account for energy conservation
            energy_kspace_kk_prime_new = np.sum(np.sum(np.abs(image_kspace_kk_prime)**2))
            image_kspace_kk_prime = image_kspace_kk_prime *  np.sqrt( energy_kspace_kk_prime / energy_kspace_kk_prime_new )
            #
            #compute inverse Fourier transform
            image_rspace_kk_prime = self.ift2d_centered(input_function = image_kspace_kk_prime)
            #
            #check if the energies in both domains are equal
            #print('energy k space ', np.sum(np.sum(np.abs(image_kspace_kk_prime)**2)))
            #print('energy real space ', image_rspace_kk.shape[0] * image_rspace_kk.shape[1] * np.sum(np.sum(np.abs(image_rspace_kk_prime)**2)))
            #
            #apply constraints in object domain
            image_rspace_kk1 = self.amp_rspace * np.exp(1j * np.angle(image_rspace_kk_prime))
            #
            #use image_rspace_kk1 as a new input to compute image_kspace_kk
            image_rspace_kk = np.copy(image_rspace_kk1)
            #
            #root mean squared error (in Fourier domain)
            rms_error.append(np.sqrt(np.sum((np.abs(image_kspace_kk) - np.abs(self.amp_kspace))**2) / np.sum(np.abs(self.amp_kspace)**2)))
            #
            #plot results
            if plot_progress is True:
                if (kk_gs+1) % plot_every_kth_iteration == 0:
                    #
                    print("Iteration %d \r" % (kk_gs + 1), " (plotting)")
                    #
                    # some manipulations with phase distribution to be able to plot it weighted with amplitude values
                    # important to normalise to 1, otherwise the plot will not be displayed correctly!
                    phase_kspace_kk_weighted = plt.cm.bwr(rescale_intensity(- np.angle(image_kspace_kk).min() + np.angle(image_kspace_kk), out_range=(0, 1)))
                    phase_kspace_kk_weighted[..., -1] = rescale_intensity(np.abs(image_kspace_kk), out_range=(0, 1))
                    #
                    phase_rspace_kk_prime_weighted = plt.cm.bwr(rescale_intensity(- np.angle(image_rspace_kk_prime).min() + np.angle(image_rspace_kk_prime), out_range=(0, 1)))
                    phase_rspace_kk_prime_weighted[..., -1] = rescale_intensity(np.abs(image_rspace_kk_prime), out_range=(0, 1))
                    #
                    phase_rspace_kk1_weighted = plt.cm.bwr(rescale_intensity(- np.angle(image_rspace_kk1).min() + np.angle(image_rspace_kk1), out_range=(0, 1)))
                    phase_rspace_kk1_weighted[..., -1] = rescale_intensity(np.abs(image_rspace_kk1), out_range=(0, 1))
                    #
                    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
                    #
                    # computed Fourier amplitude
                    im00 = ax[0,0].imshow(np.abs(image_kspace_kk))
                    ax[0,0].set_title("$|\mathrm{G}_k|$ - computed Fourier amplitude")
                    plt.colorbar(im00, ax=ax[0,0], fraction=0.046, pad=0.04)
                    ax[0,0].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom])
                    #
                    # computed Fourier phase
                    im01 = ax[0,1].imshow(phase_kspace_kk_weighted, cmap='seismic', vmin=np.angle(image_kspace_kk).min(), vmax=np.angle(image_kspace_kk).max())
                    ax[0,1].set_title("$arg(\mathrm{G}_k)$ - computed Fourier phase")
                    plt.colorbar(im01, ax=ax[0,1], fraction=0.046, pad=0.04)
                    ax[0,1].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    # constrained Fourier amplitude
                    im02 = ax[0,2].imshow(np.log(np.abs(image_kspace_kk_prime)))
                    ax[0,2].set_title("$|\mathrm{G}_k'|$ - constrained Fourier amplitude (log)")
                    plt.colorbar(im02, ax=ax[0,2], fraction=0.046, pad=0.04)
                    ax[0,2].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    # RMSE
                    im03 = ax[0,3].plot(rms_error)
                    ax[0,3].set_ylim([0, 1])
                    ax[0,3].set_title("RMS error in Fourier domain")
                    #
                    # computed object-domain amplitude
                    im10 = ax[1,0].imshow(np.abs(image_rspace_kk_prime))
                    ax[1,0].set_title("$|\mathrm{g}_k'|$ - computed object amplitude")
                    plt.colorbar(im10, ax=ax[1,0], fraction=0.046, pad=0.04)
                    ax[1,0].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    # computed object-domain phase
                    im11 = ax[1,1].imshow(phase_rspace_kk_prime_weighted, cmap='seismic', vmin=np.angle(image_rspace_kk_prime).min(), vmax=np.angle(image_rspace_kk_prime).max())
                    ax[1,1].set_title("$arg(\mathrm{g}_k')$ - computed object phase")
                    plt.colorbar(im11, ax=ax[1,1], fraction=0.046, pad=0.04)
                    ax[1,1].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    # constrained object-domain amplitude
                    im12 = ax[1,2].imshow(self.amp_rspace)
                    ax[1,2].set_title("$|\mathrm{g}_{k+1}|$ - constrained object amplitude")
                    plt.colorbar(im10, ax=ax[1,2], fraction=0.046, pad=0.04)
                    ax[1,2].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    # constrained object-domain phase
                    im13 = ax[1,3].imshow(phase_rspace_kk1_weighted, cmap='seismic', vmin=np.angle(image_rspace_kk1).min(), vmax=np.angle(image_rspace_kk1).max())
                    ax[1,3].set_title("$arg(\mathrm{g}_{k+1})$ - constrained  object phase")
                    plt.colorbar(im11, ax=ax[1,3], fraction=0.046, pad=0.04)
                    ax[1,3].axis([image_rspace0.shape[0] // 2 - image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[0] // 2 + image_rspace0.shape[0] // 2 // zoom,
                              image_rspace0.shape[1] // 2 + image_rspace0.shape[1] // 2 // zoom,
                              image_rspace0.shape[1] // 2 - image_rspace0.shape[1] // 2 // zoom])
                    #
                    fig.tight_layout()
                    display.clear_output(wait=True)
                    plt.show()
                    #
                    print("Iteration %d \r" % (kk_gs + 1), " completed")
                else:
                    print("Iteration %d \r" % (kk_gs + 1), " completed")
            else:
                print("Iteration %d \r" % (kk_gs + 1), " completed")
        #
        #save results
        self.image_rspace_kk_prime = image_rspace_kk_prime
        self.image_rspace_kk1 = image_rspace_kk1
        self.image_kspace_kk = image_kspace_kk
        self.image_kspace_kk_prime = image_kspace_kk_prime
        self.rms_error = rms_error
        #
        #return np.abs(self.image_kspace_kk_prime), np.angle(self.image_kspace_kk_prime), np.abs(self.image_rspace_kk_prime),  np.angle(self.image_rspace_kk_prime), self.rms_error

    def plot_reconstructed_images(self, zoom = 1):
        """
        Plot images reconstructed by Gerchberg-Saxton algorithm

        ---
        Parameters
        ---
        zoom: int, optional
            Zoom factor to zoom into the 2D plot
            Default is 1 (no zoom)
        ---
        """
        # some manipulations with phase distribution to be able to plot it weighted with amplitude values
        # important to normalise to 1, otherwise the plot will not be displayed correctly!
        phase_kspace_kk_weighted = plt.cm.bwr(
            rescale_intensity(- np.angle(self.image_kspace_kk).min() + np.angle(self.image_kspace_kk), out_range=(0, 1)))
        phase_kspace_kk_weighted[..., -1] = rescale_intensity(np.abs(self.image_kspace_kk), out_range=(0, 1))
        #
        phase_rspace_kk_prime_weighted = plt.cm.bwr(
            rescale_intensity(- np.angle(self.image_rspace_kk_prime).min() + np.angle(self.image_rspace_kk_prime),
                              out_range=(0, 1)))
        phase_rspace_kk_prime_weighted[..., -1] = rescale_intensity(np.abs(self.image_rspace_kk_prime), out_range=(0, 1))
        #
        phase_rspace_kk1_weighted = plt.cm.bwr(
            rescale_intensity(- np.angle(self.image_rspace_kk1).min() + np.angle(self.image_rspace_kk1), out_range=(0, 1)))
        phase_rspace_kk1_weighted[..., -1] = rescale_intensity(np.abs(self.image_rspace_kk1), out_range=(0, 1))
        #
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        #
        # computed Fourier amplitude
        im00 = ax[0, 0].imshow(np.abs(self.image_kspace_kk))
        ax[0, 0].set_title("$|\mathrm{G}_k|$ - computed Fourier amplitude")
        plt.colorbar(im00, ax=ax[0, 0], fraction=0.046, pad=0.04)
        ax[0, 0].axis([self.image_rspace_kk1.shape[0] // 2 - self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[0] // 2 + self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 - self.image_rspace_kk1.shape[1] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 + self.image_rspace_kk1.shape[1] // 2 // zoom])
        #
        # computed Fourier phase
        im01 = ax[0, 1].imshow(phase_kspace_kk_weighted, cmap='seismic', vmin=np.angle(self.image_kspace_kk).min(),
                               vmax=np.angle(self.image_kspace_kk).max())
        ax[0, 1].set_title("$arg(\mathrm{G}_k)$ - computed Fourier phase")
        plt.colorbar(im01, ax=ax[0, 1], fraction=0.046, pad=0.04)
        ax[0, 1].axis([self.image_rspace_kk1.shape[0] // 2 - self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[0] // 2 + self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 - self.image_rspace_kk1.shape[1] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 + self.image_rspace_kk1.shape[1] // 2 // zoom])
        #
        # constrained Fourier amplitude
        im02 = ax[0, 2].imshow(np.log(np.abs(self.image_kspace_kk_prime)))
        ax[0, 2].set_title("$|\mathrm{F}|$ - constrained Fourier amplitude")
        plt.colorbar(im02, ax=ax[0, 2], fraction=0.046, pad=0.04)
        ax[0, 2].axis([self.image_rspace_kk1.shape[0] // 2 - self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[0] // 2 + self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 - self.image_rspace_kk1.shape[1] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 + self.image_rspace_kk1.shape[1] // 2 // zoom])
        #
        # RMSE
        im03 = ax[0, 3].plot(self.rms_error)
        ax[0, 3].set_ylim([0, 1])
        ax[0, 3].set_title("RMS error in Fourier domain")
        #
        # computed object-domain amplitude
        im10 = ax[1, 0].imshow(np.abs(self.image_rspace_kk_prime))
        ax[1, 0].set_title("$|\mathrm{g}_k'|$ - computed object amplitude")
        plt.colorbar(im10, ax=ax[1, 0], fraction=0.046, pad=0.04)
        ax[1, 0].axis([self.image_rspace_kk1.shape[0] // 2 - self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[0] // 2 + self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 - self.image_rspace_kk1.shape[1] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 + self.image_rspace_kk1.shape[1] // 2 // zoom])
        #
        # computed object-domain phase
        im11 = ax[1, 1].imshow(phase_rspace_kk_prime_weighted, cmap='seismic',
                               vmin=np.angle(self.image_rspace_kk_prime).min(), vmax=np.angle(self.image_rspace_kk_prime).max())
        ax[1, 1].set_title("$arg(\mathrm{g}_k')$ - computed object phase")
        plt.colorbar(im11, ax=ax[1, 1], fraction=0.046, pad=0.04)
        ax[1, 1].axis([self.image_rspace_kk1.shape[0] // 2 - self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[0] // 2 + self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 - self.image_rspace_kk1.shape[1] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 + self.image_rspace_kk1.shape[1] // 2 // zoom])
        #
        # constrained object-domain amplitude
        im12 = ax[1, 2].imshow(self.amp_rspace)
        ax[1, 2].set_title("$|\mathrm{g}_{k+1}|$ - constrained object amplitude")
        plt.colorbar(im10, ax=ax[1, 2], fraction=0.046, pad=0.04)
        ax[1, 2].axis([self.image_rspace_kk1.shape[0] // 2 - self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[0] // 2 + self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 - self.image_rspace_kk1.shape[1] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 + self.image_rspace_kk1.shape[1] // 2 // zoom])
        #
        # constrained object-domain phase
        im13 = ax[1, 3].imshow(phase_rspace_kk1_weighted, cmap='seismic', vmin=np.angle(self.image_rspace_kk1).min(),
                               vmax=np.angle(self.image_rspace_kk1).max())
        ax[1, 3].set_title("$arg(\mathrm{g}_{k+1})$ - constrained  object phase")
        plt.colorbar(im11, ax=ax[1, 3], fraction=0.046, pad=0.04)
        ax[1, 3].axis([self.image_rspace_kk1.shape[0] // 2 - self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[0] // 2 + self.image_rspace_kk1.shape[0] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 - self.image_rspace_kk1.shape[1] // 2 // zoom,
                       self.image_rspace_kk1.shape[1] // 2 + self.image_rspace_kk1.shape[1] // 2 // zoom])
        #
        fig.tight_layout()
        plt.show()

    def save_as_tif(self,
                    filename = None,
                    pathtosave = None,
                    Fourier_amplitude = True,
                    Fourier_phase = True,
                    object_amplitude = True,
                    object_phase = True):
        """
        Saves images reconstructed with Gechberg_Saxton algorithm
        as float 32 tif images

        Filenames of the saved images indicate the corresponding saved image type (i.e. _object_phase_),
        followed by reciprocal space error recorded at the last iteration of the retrieval algorithm,
        date (YYYYMMDD) and time (HHMMSS). The error is given in format (1X000), where X are for digits of
        the decimal part of the error.

        ---
        Parameters
        ---
        filename: str
            Name of the images
            Default is None.
        pathtosave: str
            Path to save the images as tif
            Default is None.
        Fourier_amplitude : bool, optional
            If set to True, constrained Fourier amplitude will be saved
            Default is True
        Fourier_phase : bool, optional
            If set to True, constrained(=computed) Fourier phase will be saved
            Default is True
        object_amplitude : bool, optional
            If set to True, computed object amplitude will be saved
            Default is True
        object_phase : bool, optional
            If set to True, computed object phase will be saved
            Default is True
        """
        #
        #modify the last value of RMS error and append it to the filename
        if self.rms_error is not None:
            rms_final = str(float(self.rms_error[-1])).replace('.','')
            if len(rms_final) > 6:
                rms_final = rms_final[1:6]
            rms_final = '1' + rms_final + '000_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            raise ValueError('Error cannot be None. Run phase retrieval algorithm first!')
        #
        if self.image_rspace_kk_prime is not None and self.image_kspace_kk_prime is not None:
            if filename is not None and pathtosave is not None:
                print('saving data as float32 in ', pathtosave)
                #
                if Fourier_amplitude is True:
                    # convert to numpy array
                    if type(self.image_kspace_kk_prime) is not np.ndarray:
                        image_kspace_kk_prime_abs = np.abs(self.image_kspace_kk_prime.to_numpy())
                    else:
                        image_kspace_kk_prime_abs = np.abs(self.image_kspace_kk_prime)
                    #
                    # append tif if needed
                    if not filename.endswith('.tif'):
                        filename_full = filename + '_Fourier_amplitude_' + rms_final + '.tif'
                    else:
                        filename_full = filename[:-4] + '_Fourier_amplitude_' + rms_final + '.tif'
                    # save the image
                    io.imsave(os.path.join(pathtosave, filename_full), image_kspace_kk_prime_abs.astype(np.float32))
                    print('Fourier amplitude saved as ', filename_full)
                #
                if Fourier_phase is True:
                    # convert to numpy array
                    if type(self.image_kspace_kk_prime) is not np.ndarray:
                        image_kspace_kk_prime_angle = np.angle(self.image_kspace_kk_prime.to_numpy())
                    else:
                        image_kspace_kk_prime_angle = np.angle(self.image_kspace_kk_prime)
                    #
                    # append tif if needed
                    if not filename.endswith('.tif'):
                        filename_full = filename + '_Fourier_phase_' + rms_final + '.tif'
                    else:
                        filename_full = filename[:-4] + '_Fourier_phase_' + rms_final + '.tif'
                    # save the image
                    io.imsave(os.path.join(pathtosave, filename_full), image_kspace_kk_prime_angle.astype(np.float32))
                    print('Fourier phase saved as ', filename_full)
                #
                if object_amplitude is True:
                    # convert to numpy array
                    if type(self.image_rspace_kk_prime) is not np.ndarray:
                        image_rspace_kk_prime_abs = np.abs(self.image_rspace_kk_prime.to_numpy())
                    else:
                        image_rspace_kk_prime_abs = np.abs(self.image_rspace_kk_prime)
                    #
                    # append tif if needed
                    if not filename.endswith('.tif'):
                        filename_full = filename + '_object_amplitude_' + rms_final + '.tif'
                    else:
                        filename_full = filename[:-4] + '_object_amplitude_' + rms_final + '.tif'
                    # save the image
                    io.imsave(os.path.join(pathtosave, filename_full), image_rspace_kk_prime_abs.astype(np.float32))
                    print('object amplitude saved as ', filename_full)
                #
                if object_phase is True:
                    # convert to numpy array
                    if type(self.image_rspace_kk_prime) is not np.ndarray:
                        image_rspace_kk_prime_angle = np.angle(self.image_rspace_kk_prime.to_numpy())
                    else:
                        image_rspace_kk_prime_angle = np.angle(self.image_rspace_kk_prime)
                    #
                    # append tif if needed
                    if not filename.endswith('.tif'):
                        filename_full = filename + '_object_phase_' + rms_final + '.tif'
                    else:
                        filename_full = filename[:-4] + '_object_phase_' + rms_final + '.tif'
                    # save the image
                    io.imsave(os.path.join(pathtosave, filename_full), image_rspace_kk_prime_angle.astype(np.float32))
                    print('object phase saved as ', filename_full)
            #
                print('data saved as ', filename)
            else:
                raise ValueError("Filename or path cannot be None!")
        else:
            raise ValueError("Run phase retrieval algorithm first!")

    def save_as_csv(self,
                    filename=None,
                    pathtosave=None,
                    Fourier_amplitude = True,
                    Fourier_phase=True,
                    object_amplitude=True,
                    object_phase=True):

        #
        #modify the last value of RMS error and append it to the filename
        if self.rms_error is not None:
            rms_final = str(float(self.rms_error[-1])).replace('.','')
            if len(rms_final) > 6:
                rms_final = rms_final[1:6]
            rms_final = '1' + rms_final + '000_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            raise ValueError('Error cannot be None. Run phase retrieval algorithm first!')
        #
        #
        if Fourier_amplitude is True:
            # convert to numpy array
            if type(self.image_kspace_kk_prime) is not np.ndarray:
                image_kspace_kk_prime_abs = np.abs(self.image_kspace_kk_prime.to_numpy())
            else:
                image_kspace_kk_prime_abs = np.abs(self.image_kspace_kk_prime)
            #
            # append csv if needed
            if not filename.endswith('.csv'):
                filename_full = filename + '_Fourier_amplitude_' + rms_final + '.csv'
            else:
                filename_full = filename[:-4] + '_Fourier_amplitude_' + rms_final + '.csv'
            np.savetxt(os.path.join(pathtosave, filename_full), image_kspace_kk_prime_abs, delimiter = '\t', fmt='%1.2f')
            print('Fourier amplitude saved as ', filename_full)
            #
        if Fourier_phase is True:
            # convert to numpy array
            if type(self.image_kspace_kk_prime) is not np.ndarray:
                image_kspace_kk_prime_angle = np.angle(self.image_kspace_kk_prime.to_numpy())
            else:
                image_kspace_kk_prime_angle = np.angle(self.image_kspace_kk_prime)
            #
            # append csv if needed
            if not filename.endswith('.csv'):
                filename_full = filename + '_Fourier_phase_' + rms_final + '.csv'
            else:
                filename_full = filename[:-4] + '_Fourier_phase_' + rms_final + '.csv'
            np.savetxt(os.path.join(pathtosave, filename_full), image_kspace_kk_prime_angle, delimiter = '\t', fmt='%1.2f')
            print('Fourier phase saved as ', filename_full)
        #
        if object_amplitude is True:
            # convert to numpy array
            if type(self.image_rspace_kk_prime) is not np.ndarray:
                image_rspace_kk_prime_abs = np.abs(self.image_rspace_kk_prime.to_numpy())
            else:
                image_rspace_kk_prime_abs = np.abs(self.image_rspace_kk_prime)
            #
            # append csv if needed
            if not filename.endswith('.csv'):
                filename_full = filename + '_object_amplitude_' + rms_final + '.csv'
            else:
                filename_full = filename[:-4] + '_object_amplitude_' + rms_final + '.csv'
            np.savetxt(os.path.join(pathtosave, filename_full), image_rspace_kk_prime_abs, delimiter = '\t', fmt='%1.2f')
            print('object amplitude saved as ', filename_full)
            #
        if object_phase is True:
            # convert to numpy array
            if type(self.image_rspace_kk_prime) is not np.ndarray:
                image_rspace_kk_prime_angle = np.angle(self.image_rspace_kk_prime.to_numpy())
            else:
                image_rspace_kk_prime_angle = np.angle(self.image_rspace_kk_prime)
            #
            # append csv if needed
            if not filename.endswith('.csv'):
                filename_full = filename + '_object_phase_' + rms_final + '.csv'
            else:
                filename_full = filename[:-4] + '_object_phase_' + rms_final + '.csv'
            np.savetxt(os.path.join(pathtosave, filename_full), image_rspace_kk_prime_angle, delimiter = '\t', fmt='%1.2f')
            print('object phase saved as', filename_full)

def save_as_eps(self,
                filename=None,
                pathtosave=None,
                Fourier_amplitude = True,
                Fourier_phase=True,
                object_amplitude=True,
                object_phase=True):

    #
    #modify the last value of RMS error and append it to the filename
    if self.rms_error is not None:
        rms_final = str(float(self.rms_error[-1])).replace('.','')
        if len(rms_final) > 6:
            rms_final = rms_final[1:6]
        rms_final = '1' + rms_final + '000_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        raise ValueError('Error cannot be None. Run phase retrieval algorithm first!')
    #
    #
    if Fourier_amplitude is True:
        # convert to numpy array
        if type(self.image_kspace_kk_prime) is not np.ndarray:
            image_kspace_kk_prime_abs = np.abs(self.image_kspace_kk_prime.to_numpy())
        else:
            image_kspace_kk_prime_abs = np.abs(self.image_kspace_kk_prime)
        #
        # append eps if needed
        if not filename.endswith('.eps'):
            filename_full = filename + '_Fourier_amplitude_' + rms_final + '.eps'
        else:
            filename_full = filename[:-4] + '_Fourier_amplitude_' + rms_final + '.eps'
        np.savetxt(os.path.join(pathtosave, filename_full), image_kspace_kk_prime_abs, delimiter = '\t')
        print('Fourier amplitude saved as ', filename_full)
        #
    if Fourier_phase is True:
        # convert to numpy array
        if type(self.image_kspace_kk_prime) is not np.ndarray:
            image_kspace_kk_prime_angle = np.angle(self.image_kspace_kk_prime.to_numpy())
        else:
            image_kspace_kk_prime_angle = np.angle(self.image_kspace_kk_prime)
        #
        # append eps if needed
        if not filename.endswith('.eps'):
            filename_full = filename + '_Fourier_phase_' + rms_final + '.eps'
        else:
            filename_full = filename[:-4] + '_Fourier_phase_' + rms_final + '.eps'
        np.savetxt(os.path.join(pathtosave, filename_full), image_kspace_kk_prime_angle, delimiter = '\t')
        print('Fourier phase saved as ', filename_full)
    #
    if object_amplitude is True:
        # convert to numpy array
        if type(self.image_rspace_kk_prime) is not np.ndarray:
            image_rspace_kk_prime_abs = np.abs(self.image_rspace_kk_prime.to_numpy())
        else:
            image_rspace_kk_prime_abs = np.abs(self.image_rspace_kk_prime)
        #
        # append eps if needed
        if not filename.endswith('.eps'):
            filename_full = filename + '_object_amplitude_' + rms_final + '.'
        else:
            filename_full = filename[:-4] + '_object_amplitude_' + rms_final + '.eps'
        np.savetxt(os.path.join(pathtosave, filename_full), image_rspace_kk_prime_abs, delimiter = '\t')
        print('object amplitude saved as ', filename_full)
        #
    if object_phase is True:
        # convert to numpy array
        if type(self.image_rspace_kk_prime) is not np.ndarray:
            image_rspace_kk_prime_angle = np.angle(self.image_rspace_kk_prime.to_numpy())
        else:
            image_rspace_kk_prime_angle = np.angle(self.image_rspace_kk_prime)
        #
        # append eps if needed
        if not filename.endswith('.eps'):
            filename_full = filename + '_object_phase_' + rms_final + '.eps'
        else:
            filename_full = filename[:-4] + '_object_phase_' + rms_final + '.eps'
        np.savetxt(os.path.join(pathtosave, filename_full), image_rspace_kk_prime_angle, delimiter = '\t')
        plt.savefig('destination_path.eps', format='eps')
        print('object phase saved as ', filename_full)