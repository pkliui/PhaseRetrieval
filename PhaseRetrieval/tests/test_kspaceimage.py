import unittest

import numpy as np
import os
import shutil, tempfile
from skimage import io

from ddt import ddt

from PhaseRetrieval.classes.kspaceimage import KSpaceImage

@ddt
class TestKSpaceImageClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        #import k-space image class
        self.ks = KSpaceImage()
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
        for var in ['filename', 'delimiter', 'image',  'renorm_factor']:
            self.assertIn(var, self.ks.__dict__)
            self.assertEqual(self.ks.__dict__[var], None)

    def test_read_from_csv(self):
        """
        test missing positional arguments
        test to read some non-existing data
        :return:
        """
        with self.assertRaises(TypeError):
            self.ks.read_from_csv()
        with self.assertRaises(ValueError):
            self.ks.read_from_csv(filename='some_none_existing_data.csv',
                             delimiter = ',')

    def test_read_from_csv_metadata(self):
        """
        test metadata
        :return:
        """
        # save data in temp directory
        files_in_dir = os.listdir(self.test_dir)
        test_image = np.ones((3,3))
        np.savetxt(os.path.join(self.test_dir, 'test_image.csv'), test_image, delimiter='\t')
        #
        # read saved data and check its metadata
        self.ks.read_from_csv(filename=os.path.join(self.test_dir, 'test_image.csv'), delimiter = '\t')
        #
        self.assertEqual(self.ks.metadata['Image centred and padded?'], 'no')
        self.assertEqual(self.ks.metadata['Background subtracted?'], 'no')

    def test_read_from_tif(self):
        """
        test to read some non-existing data
        :return:
        """
        with self.assertRaises(TypeError):
            self.ks.read_from_tif()
        with self.assertRaises(ValueError):
            self.ks.read_from_tif(filename='some_none_existing_data.tif')

    def test_read_from_tif_metadata(self):
        """
        test metadata
        :return:
        """
        # save data in temp directory
        files_in_dir = os.listdir(self.test_dir)
        test_image = np.ones((3,3))
        io.imsave(os.path.join(self.test_dir, 'test_image.tif'), test_image.astype(np.float32))
        #
        # read saved data and check its metadata
        self.ks.read_from_tif(filename=os.path.join(self.test_dir, 'test_image.tif'))
        #
        self.assertEqual(self.ks.metadata['Image centred and padded?'], 'no')
        self.assertEqual(self.ks.metadata['Background subtracted?'], 'no')

    def test_plot_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.ks.image = None
        with self.assertRaises(ValueError):
            self.ks.plot_image()

    def test_rotate_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.ks.image = None
        with self.assertRaises(ValueError):
            self.ks.rotate_image()

    def test_flip_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.ks.image = None
        with self.assertRaises(ValueError):
            self.ks.flip_image()

    def test_centre_image_manually_empty(self):
        """
        test empty arguments
        :return:
        """
        self.ks.image = None
        with self.assertRaises(ValueError):
            self.ks.centre_image_manually()

    def test_centre_image_manually_too_high_npixels_final(self):
        """
        test the final linear number of pixels npixels_final is exceeding image's linear dimensions
        :return:
        """
        self.ks.image = np.ones((6,6))
        with self.assertRaises(ValueError):
            self.ks.centre_image_manually(npixels_final = 4.0)
        with self.assertRaises(ValueError):
            self.ks.centre_image_manually(npixels_final = 4)
        #

    def test_centre_image_manually(self):
        """
        test centering manually
        :return:
        """
        self.ks.image = np.ones((6,6))
        self.ks.image[0,0] = 0
        self.ks.centre_image_manually(manual_centroid = (4, 4), npixels_final=8)
        print(self.ks.image)
        #self.assertEqual(self.ks.image.shape[0],20)

    def test_centre_image(self):
        self.ks.image = None
        with self.assertRaises(ValueError):
            self.ks.centre_image()
        #
        self.ks.image = np.ones((10,10))
        with self.assertRaises(ValueError):
            self.ks.centre_image(npixels_pad = 5.0)
        with self.assertRaises(ValueError):
            self.ks.centre_image(npixels_pad = 5)
        #
        self.ks.image = np.ones((10,10))
        self.ks.centre_image(npixels_pad = 20, estimate_only=False)
        self.assertEqual(self.ks.image.shape[0],20)

    def test_subtract_background(self):
        self.ks.image = np.zeros((10,10))
        self.ks.metadata['Image centred and padded?'] = 'no'
        with self.assertRaises(ValueError):
            self.ks.subtract_background()
        #
        self.ks.image = None
        self.ks.metadata['Image centred and padded?'] = 'yes'
        with self.assertRaises(ValueError):
            self.ks.subtract_background()
        #
        self.ks.image = None
        self.ks.metadata['Image centred and padded?'] = 'no'
        with self.assertRaises(ValueError):
            self.ks.subtract_background()

    def test_renormalise_image(self):
        with self.assertRaises(ValueError):
            self.ks.renormalise_image(energy_rspace = None)
        #
        energy_rspace = 1
        image = np.ones((3, 3))
        renorm_factor = energy_rspace * image.shape[0] * image.shape[1] / np.sum(np.sum(np.array(image)))
        self.ks.image = np.ones((3, 3))
        self.ks.renormalise_image(energy_rspace=1)
        self.assertEqual(self.ks.renorm_factor, renorm_factor)

    def test_save_as_tif(self):
        self.ks.image = None
        with self.assertRaises(ValueError):
            self.ks.save_as_tif()
        #
        self.ks.image = np.ones((3,3))
        with self.assertRaises(ValueError):
            self.ks.save_as_tif(outputfilename=None, pathtosave=self.test_dir)
        #
        self.ks.image = np.ones((3,3))
        with self.assertRaises(ValueError):
            self.ks.save_as_tif(outputfilename='valid_filename.tif', pathtosave='some_invalid_path')
        # test reading
        self.ks.image = np.ones((3,3))
        outputfilename = 'valid_filename.tif'
        pathtosave = self.test_dir
        self.ks.save_as_tif(pathtosave, outputfilename)
        self.ks.read_from_tif(filename=os.path.join(pathtosave, outputfilename))
        self.assertIsNotNone(self.ks.image)

if __name__ == '__main__':
    unittest.main()