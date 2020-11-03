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
        self.test_image = pd.DataFrame([[0, 0, 0, 0, 0],
                             [0, 4, 4, 4, 0],
                             [0, 4, 16, 4, 0],
                             [0, 4, 4, 4, 0],
                             [0, 0, 0, 0, 0]])
        #

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_energy_conservation(self):
        """
        test energy conservation of ft2d_centered and ift2d_centered
        :return:
        """
        #
        #compute Fourier transform of the input image
        image_ft = self.ft.ft2d_centered(np.sqrt(self.test_image))
        #
        # check if the energies in both domains are equal
        energy_kspace = int(round(np.sum(np.sum(np.abs(image_ft) ** 2))))
        energy_rspace = int(round(self.test_image.shape[0] * self.test_image.shape[1] * np.sum(np.sum(np.abs(self.test_image)))))
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

