import unittest

import numpy as np
import os
from skimage.util import pad
import shutil, tempfile

from ddt import ddt

from PhaseRetrieval.classes.rspaceimage import RSpaceImage

@ddt
class TestRSpaceImageClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        #import k-space image class
        self.rs = RSpaceImage()
        #create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        #remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_empty_arguments(self):
        """
        test the input arguments are existing and are all None
        :return:
        """
        for var in ['filename', 'delimiter', 'image', 'image_apodization_filter']:
            self.assertIn(var, self.rs.__dict__)
            self.assertEqual(self.rs.__dict__[var], None)

    def test_read_from_csv(self):
        """
        test missing positional arguments
        test non-existing path
        :return:
        """
        with self.assertRaises(TypeError):
            self.rs.read_from_csv()
        with self.assertRaises(ValueError):
            self.rs.read_from_csv(filename='some_none_existing_data.csv',
                             delimiter = ',')

    def test_read_from_tif(self):
        """
        testing to read some non-existing data
        :return:
        """
        with self.assertRaises(TypeError):
            self.rs.read_from_tif()
        with self.assertRaises(ValueError):
            self.rs.read_from_tif(filename='some_none_existing_data.tif')

    def test_plot_image(self):
        """
        testing the case when an input image object is None
        :return:
        """
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.plot_image()

    def test_rotate_image(self):
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.rotate_image()

    def test_flip_image(self):
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.flip_image()

    def test_centre_image_watershed(self):
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.centre_image_watershed(apodization=False)
        #
        n_signal = 3
        n_topad = 2
        self.rs.image = np.ones((n_signal ,n_signal ))
        self.rs.image = pad(self.rs.image,
                     ((n_topad, n_topad),
                      (n_topad, n_topad)),
                     mode='constant')
        with self.assertRaises(ValueError):
            self.rs.centre_image_watershed(npixels_pad = 1.0, apodization=False)
        with self.assertRaises(ValueError):
            self.rs.centre_image_watershed(npixels_pad = 1, apodization=False)
        #
        self.rs.image = np.ones((n_signal ,n_signal ))
        self.rs.image = pad(self.rs.image,
                     ((n_topad, n_topad),
                      (n_topad, n_topad)),
                     mode='constant')
        with self.assertRaises(ValueError):
            self.rs.centre_image_watershed(npixels_pad = 5.0, apodization=True)
        with self.assertRaises(ValueError):
            self.rs.centre_image_watershed(npixels_pad = 5, apodization=True)
        #
        self.rs.image = np.ones((n_signal ,n_signal ))
        self.rs.image = pad(self.rs.image,
                     ((n_topad, n_topad),
                      (n_topad, n_topad)),
                     mode='constant')
        self.rs.centre_image_watershed(linear_object_size=n_signal, npixels_pad=20, apodization=False)
        self.assertEqual(self.rs.image.shape[0], 20)

    def test_subtract_background(self):
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.subtract_background()

    def test_resample_image(self):
        self.rs.image = None
        self.rs.metadata['Image centred and padded?'] = None
        with self.assertRaises(ValueError):
            self.rs.resample_image()
        #
        self.rs.image = None
        self.rs.metadata['Image centred and padded?'] = 'yes'
        with self.assertRaises(ValueError):
            self.rs.resample_image()
        #
        self.rs.image = np.ones((3,3))
        self.rs.metadata['Image centred and padded?'] = 'no'
        with self.assertRaises(ValueError):
            self.rs.resample_image()
        #
        #self.rs.image = np.ones((3,3))
        #self.rs.metadata['Image centred and padded?'] = 'yes'
        #self.rs.metadata['Pixel size object domain, nm']=350
        #self.rs.metadata['Linear number of pixels in the zero-padded real-space image']=2000
        #with self.assertRaises(ValueError):
        #    self.rs.resample_image(fieldofview = 17, npixels_kspace = 500, pixelsize_dr0 = 100, lambd = 880)

    def test_save_as_tif(self):
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.save_as_tif()
        #
        self.rs.image = np.ones((3,3))
        with self.assertRaises(ValueError):
            self.rs.save_as_tif(outputfilename=None, pathtosave=self.test_dir)
        #
        self.rs.image = np.ones((3,3))
        with self.assertRaises(ValueError):
            self.rs.save_as_tif(outputfilename='valid_filename.tif', pathtosave='some_invalid_path')
        # test reading
        self.rs.image = np.ones((3,3))
        outputfilename = 'valid_filename.tif'
        pathtosave = self.test_dir
        self.rs.save_as_tif(pathtosave, outputfilename)
        self.rs.read_from_tif(filename=os.path.join(pathtosave, outputfilename))
        self.assertIsNotNone(self.rs.image)

if __name__ == '__main__':
    unittest.main()