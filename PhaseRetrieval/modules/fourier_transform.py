"""
This module contains functions for computing direct and inverse centred Fourier transforms of 2D images.
The transforms are as described in
    T. Latyvevskaia and H.-W. Fink  "Practical algorithms for simulation and reconstruction of digital in-line holograms",
    Appl. Optics 54, 2424 - 2434 (2015)
"""

import numpy as np

def ft2d_centered(input_function):
    """
    Centred Fourier transform
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


def ift2d_centered(input_function):
    """
    Centred inverse Fourier transform
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