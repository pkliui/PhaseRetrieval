import unittest

from ddt import ddt

import shutil, tempfile
import pandas as pd
import numpy as np

from PhaseRetrieval.modules import fourier_transform

@ddt
class TestFourierTransform(unittest.TestCase):
    """
    Class to test the fourier_transform module
    Tests linearity of FT and its response to a unit impulse
    """
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        #import postprocessing module
        self.ft = fourier_transform
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        #
        self.test_image_energy_conservation = pd.DataFrame([[0, 0, 0, 0, 0],
                             [0, 4, 4, 4, 0],
                             [0, 4, 16, 4, 0],
                             [0, 4, 4, 4, 0],
                             [0, 0, 0, 0, 0]])
        #
        # fft of unit impulse 2x2
        # input image, real part
        self.test_unit_impulse_input_real1 = pd.DataFrame([[1.0, 0.0],
                             [0.0, 0.0]])
        #
        # input image, imaginary part
        self.test_unit_impulse_input_imag1 = pd.DataFrame([[0.0, 0.0],
                             [0.0, 0.0]])
        #
        # output image, real part
        self.test_unit_impulse_output_real1 = pd.DataFrame([[1.0, -1.0],
                             [-1.0, 1.0]])
        #
        # output image, imaginary part
        self.test_unit_impulse_output_imag1 = pd.DataFrame([[0.0, 0.0],
                             [0.0, 0.0]])
        #
        #
        # fft of complex-valued unit impulse 2x2
        # input image, real part
        self.test_unit_impulse_input_real2 = pd.DataFrame([[1.0, 0.0],
                             [0.0, 0.0]])
        #
        # input image, imaginary part
        self.test_unit_impulse_input_imag2 = pd.DataFrame([[1.0, 0.0],
                             [0.0, 0.0]])
        #
        # output image, real part
        self.test_unit_impulse_output_real2 = pd.DataFrame([[1.0, -1.0],
                             [-1.0, 1.0]])
        #
        # output image, imaginary part
        self.test_unit_impulse_output_imag2 = pd.DataFrame([[1.0, -1.0],
                             [-1.0, 1.0]])
        #
        #
        # fft of unit impulse 3x3
        # input image, real part
        self.test_unit_impulse_input_real3 = pd.DataFrame([[0.0, 0.0, 0.0],
                                                          [0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0]])
        #
        # input image, imaginary part
        self.test_unit_impulse_input_imag3 = pd.DataFrame([[0.0, 0.0, 0.0],
                                                          [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])
        #
        # output image, real part
        self.test_unit_impulse_output_real3 = pd.DataFrame([[1.0, 0.5, -0.5],
                                                          [0.5, -0.5, -1.0],
                             [-0.5, -1.0, -0.5]])
        #
        # output image, imaginary part
        self.test_unit_impulse_output_imag3 = pd.DataFrame([[0.0, 0.866, 0.866],
                                                           [0.866, 0.866, 0.0],
                             [0.866, 0.0, -0.866]])
        #

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_energy_conservation(self):
        """
        testing energy conservation
        energies in both domains must be equal
        """
        #
        #compute Fourier transform of the input image
        image_ft = self.ft.ft2d_centered(np.sqrt(self.test_image_energy_conservation))
        #
        # check if the energies in both domains are equal
        energy_kspace = int(round(np.sum(np.sum(np.abs(image_ft) ** 2))))
        energy_rspace = int(round(self.test_image_energy_conservation.shape[0] * self.test_image_energy_conservation.shape[1] * np.sum(np.sum(np.abs(self.test_image_energy_conservation)))))
        #
        self.assertEqual(energy_kspace,energy_rspace)
        #
        #
        #compute inverse Fourier transform of the input image
        image_ft_ft = self.ft.ift2d_centered(image_ft)
        #
        # check if the energies in both domains are equal
        energy_kspace = int(round(np.sum(np.sum(np.abs(image_ft) ** 2))))
        energy_rspace = int(round(image_ft_ft.shape[0] * image_ft_ft.shape[1] * np.sum(np.sum(np.abs(image_ft_ft)**2))))
        #
        self.assertEqual(energy_kspace,energy_rspace)

    def test_unit_impulse1(self):
        """
        testing FT of 2x2 image of a unit impulse
        """
        #
        # compute Fourier transform of the input image
        image_ft = self.ft.ft2d_centered(self.test_unit_impulse_input_real1 + 1j * self.test_unit_impulse_input_imag1)
        #
        # test real part
        self.assertTrue(np.array_equal(np.round(np.real(image_ft),1), self.test_unit_impulse_output_real1))
        # test imaginary part
        self.assertTrue(np.array_equal(np.round(np.imag(image_ft),1), self.test_unit_impulse_output_imag1))


    def test_unit_impulse2(self):
        """
        testing FT of a 2x2 complex-valued unit impulse
        """
        #
        # compute Fourier transform of the input image
        image_ft = self.ft.ft2d_centered(self.test_unit_impulse_input_real2 + 1j * self.test_unit_impulse_input_imag2)
        #
        # test real part
        self.assertTrue(np.array_equal(np.round(np.real(image_ft),1), self.test_unit_impulse_output_real2))
        # test imaginary part
        self.assertTrue(np.array_equal(np.round(np.imag(image_ft),1), self.test_unit_impulse_output_imag2))

    def test_unit_impulse3(self):
        """
        testing FT of 3x3 image of a unit impulse
        """
        #
        # compute Fourier transform of the input image
        image_ft = self.ft.ft2d_centered(self.test_unit_impulse_input_real3 + 1j * self.test_unit_impulse_input_imag3)
        #
        # test real part - rounding to the same number of digits as in the input images
        self.assertTrue(np.array_equal(np.round(np.real(image_ft),1), self.test_unit_impulse_output_real3))
        # test imaginary part
        self.assertTrue(np.array_equal(np.round(np.imag(image_ft),3), self.test_unit_impulse_output_imag3))

    def test_linearity(self):
        """
        testing linearity of FT using 2x2 images of real- and complex-valued unit impulses
        """
        image1 = self.test_unit_impulse_input_real1 + 1j * self.test_unit_impulse_input_imag1
        image2 = self.test_unit_impulse_input_real2 + 1j * self.test_unit_impulse_input_imag2
        lhs = self.ft.ft2d_centered(2*image1 + 3*image2)
        rhs = 2*self.ft.ft2d_centered(image1) + 3*self.ft.ft2d_centered(image2)
        #
        # testing the left hand side and the right hand side are equal - rounding to the same number of digits as in the input images 1 and 2
        self.assertTrue(np.array_equal(np.round(lhs,1), np.round(rhs,1)))