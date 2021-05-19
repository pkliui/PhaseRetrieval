import unittest

import numpy as np
import os
from skimage.util import pad
import shutil, tempfile
from skimage import io

from ddt import ddt

from PhaseRetrieval.classes.rspaceimage import RSpaceImage
from PhaseRetrieval.classes.rspacemetadata import RSpaceMetadata

@ddt
class TestRSpaceImageClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        #import r-space image class
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
        for var in ['filename', 'delimiter', 'image', 'image_binary',
                 'image_segmented', 'image_centred']:
            self.assertIn(var, self.rs.__dict__)
            self.assertEqual(self.rs.__dict__[var], None)

    def test_read_from_csv(self):
        """
        test missing positional arguments
        test to read some non-existing data
        :return:
        """
        with self.assertRaises(TypeError):
            self.rs.read_from_csv()
        with self.assertRaises(ValueError):
            self.rs.read_from_csv(filename='some_none_existing_data.csv',
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
        self.rs.read_from_csv(filename=os.path.join(self.test_dir, 'test_image.csv'), delimiter = '\t')
        #
        self.assertEqual(self.rs.metadata['Image centred and padded?'], 'no')
        self.assertEqual(self.rs.metadata['Background subtracted?'], 'no')


    def test_read_from_tif(self):
        """
        test to read some non-existing data
        :return:
        """
        with self.assertRaises(TypeError):
            self.rs.read_from_tif()
        with self.assertRaises(ValueError):
            self.rs.read_from_tif(filename='some_none_existing_data.tif')

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
        self.rs.read_from_tif(filename=os.path.join(self.test_dir, 'test_image.tif'))
        #
        self.assertEqual(self.rs.metadata['Image centred and padded?'], 'no')
        self.assertEqual(self.rs.metadata['Background subtracted?'], 'no')


    def test_plot_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.plot_image()

    def test_rotate_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.rotate_image()

    def test_flip_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.flip_image()

    def test_subtract_background_empty_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.subtract_background()

    def test_subtract_background(self):
        """
        test whether the right amount of counts was subtracted from input image
        :return:
        """
        self.rs.image = np.array([[0.0, 1.0, 2.0],[3.0, 4.5, 5.5],[6.0, 7.0, 8.0]])
        counts_to_subtract = 2.0
        image_after_bg_subtraction = np.array([[0.0, 0.0, 0.0],[1.0, 2.5, 3.5],[4.0, 5.0, 6.0]])
        im_bgfree = self.rs.subtract_background(counts = counts_to_subtract, estimate_only = False)
        print("im_bgfree", im_bgfree)
        #
        self.assertTrue(np.array_equal(im_bgfree, image_after_bg_subtraction))

    def test_subtract_background_metadata(self):
        """
        test metadata
        :return:
        """
        self.rs.image = np.array([[0.0, 1.0, 2.0],[3.0, 4.5, 5.5],[6.0, 7.0, 8.0]])
        counts_to_subtract = 2.0
        image_after_bg_subtraction = np.array([[0.0, 0.0, 0.0],[1.0, 2.5, 3.5],[4.0, 5.0, 6.0]])
        im_bgfree = self.rs.subtract_background(counts = counts_to_subtract, estimate_only = False)
        #
        self.assertEqual(self.rs.metadata['Background subtracted?'], 'yes')

    def test_subtract_background_mean_noise(self):
        """
        test subtracting mean noise value computed from the patch
        :return:
        """
        self.rs.image = np.array([[0.0, 1.0, 2.0],[3.0, 4.0, 5.5],[6.0, 7.0, 8.0]])
        image_after_bg_subtraction = np.array([[0.0, 0.0, 0.0],[1.0, 2.0, 3.5],[4.0, 5.0, 6.0]])
        im_bgfree = self.rs.subtract_background(noise_mean= True, patch_corner=(0,0), patch_size=(2,2), estimate_only = False)
        #counts_to_subtract obtained from the mean pixel value within the patch = 2.0
        #
        self.assertTrue(np.array_equal(im_bgfree, image_after_bg_subtraction))

    def test_segment_image_watershed_empty_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.segment_image_watershed()

    def test_segment_image_watershed_pixelsize(self):
        """
        test whether the pixel size  is computed correctly
        :return:
        """
        #
        # initialise input image sampled at n_signal x n_signal non-zero-valued pixels surrounded by a zero-padded region of width n_topad
        # gives 8x8 image
        n_signal = 4
        n_topad = 2
        #
        self.rs.image = 3.0 * np.ones((n_signal, n_signal))
        self.rs.image = pad(self.rs.image,
                            ((n_topad, n_topad),
                             (n_topad, n_topad)),
                            mode='constant') + 2.0
        #
        # binarize input image using estimate_only mode of the subtract_background method (background = 2.0 here)
        self.rs.subtract_background(counts=2.0,
                               estimate_only=True,
                               plot_progress=False)
        #
        # set the structuring element size to 1x1 pxls. This should yield a 4x4 non-zero-valued foreground after the opening (erosion by 1x1 str.element + dilation by 1x1 str.element)
        pixelsize_dr0 = self.rs.segment_image_watershed(linear_object_size=10, str_element_size = 1, plot_progress=False)
        # pixel size = sqrt(10*10/4x4)= 2,5
        self.assertEqual(pixelsize_dr0, 2.5)

    def test_segment_image_watershed_metadata(self):
        """
        test metadata
        :return:
        """
        #
        # initialise input image sampled at n_signal x n_signal non-zero-valued pixels surrounded by a zero-padded region of width n_topad
        # gives 6x6 image
        # test even n_signal
        n_signal = 4
        n_topad = 2
        #
        self.rs.image = 3.0 * np.ones((n_signal ,n_signal ))
        self.rs.image = pad(self.rs.image,
                     ((n_topad, n_topad),
                      (n_topad, n_topad)),
                     mode='constant') + 2.0
        #
        # binarize input image using estimate_only mode of the subtract_background method (background = 2.0 here)
        self.rs.subtract_background(counts=2.0,
                               estimate_only=True,
                               plot_progress=False)
        #
        pixelsize_dr0 = self.rs.segment_image_watershed(linear_object_size=10, str_element_size = 1, plot_progress=False)
        self.assertEqual(self.rs.metadata['Pixel size object domain, m'], pixelsize_dr0)
        self.assertEqual(self.rs.metadata['Linear size of the object, m'], 10)

    def test_segment_image_opening_empty_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.segment_image_opening()

    def test_segment_image_opening_pixelsize(self):
        """
        test whether the pixel size  is computed correctly
        :return:
        """
        #
        # initialise input image sampled at n_signal x n_signal non-zero-valued pixels surrounded by a zero-padded region of width n_topad
        # gives 8x8 image
        n_signal = 4
        n_topad = 2
        #
        self.rs.image = 3.0 * np.ones((n_signal, n_signal))
        self.rs.image = pad(self.rs.image,
                            ((n_topad, n_topad),
                             (n_topad, n_topad)),
                            mode='constant') + 2.0
        #
        # binarize input image using estimate_only mode of the subtract_background method (background = 2.0 here)
        self.rs.subtract_background(counts=2.0,
                               estimate_only=True,
                               plot_progress=False)
        #
        # set the structuring element size to 1x1 pxls. This should yield a 4x4 non-zero-valued foreground after the opening (erosion by 1x1 str.element + dilation by 1x1 str.element)
        pixelsize_dr0 = self.rs.segment_image_opening(linear_object_size=10, str_element_size = 1, plot_progress=False)
        self.assertEqual(pixelsize_dr0, 2500.0)

    def test_segment_image_opening_metadata(self):
        """
        test metadata
        :return:
        """
        #
        # initialise input image sampled at n_signal x n_signal non-zero-valued pixels surrounded by a zero-padded region of width n_topad
        # gives 6x6 image
        # test even n_signal
        n_signal = 4
        n_topad = 2
        #
        self.rs.image = 3.0 * np.ones((n_signal ,n_signal ))
        self.rs.image = pad(self.rs.image,
                     ((n_topad, n_topad),
                      (n_topad, n_topad)),
                     mode='constant') + 2.0
        #
        # binarize input image using estimate_only mode of the subtract_background method (background = 2.0 here)
        self.rs.subtract_background(counts=2.0,
                               estimate_only=True,
                               plot_progress=False)
        #
        pixelsize_dr0 = self.rs.segment_image_opening(linear_object_size=10, str_element_size = 1, plot_progress=False)
        self.assertEqual(self.rs.metadata['Pixel size object domain, m'], pixelsize_dr0)
        self.assertEqual(self.rs.metadata['Linear size of the object, m'], 10)

    def test_segment_image_kmeans_empty_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.segment_image_kmeans()

    def test_segment_image_kmeans_pixelsize(self):
        """
        test whether the pixel size  is computed correctly
        :return:
        """
        #
        # initialise input image sampled at n_signal x n_signal non-zero-valued pixels surrounded by a zero-padded region of width n_topad
        # gives 8x8 image
        n_signal = 4
        n_topad = 2
        #
        self.rs.image = 3.0 * np.ones((n_signal, n_signal))
        self.rs.image = pad(self.rs.image,
                            ((n_topad, n_topad),
                             (n_topad, n_topad)),
                            mode='constant') + 2.0
        #
        # set the structuring element size to 1x1 pxls. This should yield a 4x4 non-zero-valued foreground after the opening (erosion by 1x1 str.element + dilation by 1x1 str.element)
        pixelsize_dr0 = self.rs.segment_image_kmeans(n_clusters=2, init="k-means++", n_init = 10, max_iter = 300, tol = 1e-4,
                        algorithm='elkan', str_element_size = 1, linear_object_size=10, plot_progress = False)
        self.assertEqual(pixelsize_dr0, 2500.0)

    def test_segment_image_kmeans(self):
        """
        test metadata
        :return:
        """
        #
        # initialise input image sampled at n_signal x n_signal non-zero-valued pixels surrounded by a zero-padded region of width n_topad
        # gives 8x8 image
        n_signal = 4
        n_topad = 2
        #
        self.rs.image = 3.0 * np.ones((n_signal, n_signal))
        self.rs.image = pad(self.rs.image,
                            ((n_topad, n_topad),
                             (n_topad, n_topad)),
                            mode='constant') + 2.0
        #
        # set the structuring element size to 1x1 pxls. This should yield a 4x4 non-zero-valued foreground after the opening (erosion by 1x1 str.element + dilation by 1x1 str.element)
        pixelsize_dr0 = self.rs.segment_image_kmeans(n_clusters=2, init="k-means++", n_init = 10, max_iter = 300, tol = 1e-4,
                        algorithm='elkan', str_element_size = 1, linear_object_size=10, plot_progress = False)
        #
        self.assertEqual(self.rs.metadata['Pixel size object domain, m'], pixelsize_dr0)
        self.assertEqual(self.rs.metadata['Linear size of the object, m'], 10)

    def test_centre_image_empty_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.centre_image()

    def test_centre_image_empty_image_segmented(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.rs.image = np.ones((2 ,2))
        self.rs.image_segmented = None
        with self.assertRaises(ValueError):
            self.rs.centre_image()

    def test_centre_image_empty_npixels_pad(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.rs.image = np.ones((2 ,2))
        self.rs.image_segmented = np.ones((2 ,2))
        with self.assertRaises(ValueError):
            self.rs.centre_image(npixels_pad = None)


    def test_center_image_low_npixels_pad(self):
        """
        test zero-padding input image to a fewer final linear number of pixels than that in the input image
        :return:
        """
        #
        # initialise input image sampled at n_signal x n_signal non-zero-valued pixels surrounded by a zero-padded region of width n_topad
        # gives 8x8 image
        n_signal = 4
        n_topad = 2
        #
        self.rs.image = 3.0 * np.ones((n_signal, n_signal))
        self.rs.image = pad(self.rs.image,
                            ((n_topad, n_topad),
                             (n_topad, n_topad)),
                            mode='constant') + 2.0
        #
        # segment the image prior to centering
        self.rs.segment_image_kmeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300, tol=1e-4,
                                     algorithm='elkan', str_element_size=1, linear_object_size=10, plot_progress=False)
        #
        with self.assertRaises(ValueError):
            self.rs.centre_image(npixels_pad = 6, apodization=False)

    def test_centre_image_im_centroids_shift(self):
        """
        test whether the shift of the image's centroid w.r.t. the image's computational centre  is computed correctly
        :return:
        """
        #
        # initialise input image sampled at n_signal x n_signal non-zero-valued pixels surrounded by a zero-padded region of width n_topad
        # gives 8x8 image
        n_signal = 4
        n_topad = 2
        #
        self.rs.image = 3.0 * np.ones((n_signal, n_signal))
        self.rs.image = pad(self.rs.image,
                            ((n_topad, n_topad),
                             (n_topad, n_topad)),
                            mode='constant') + 2.0
        #
        # segment the image prior to centering
        self.rs.segment_image_kmeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300, tol=1e-4,
                                     algorithm='elkan', str_element_size=1, linear_object_size=10, plot_progress=False)
        print("segmented image: ", self.rs.image_segmented)
        #
        # pad this image to 12x12 pixels and check the centroid's shift of the segmented region w.r.t. the computational centre is (-2,-2)
        im_centroids_shift  = self.rs.centre_image(npixels_pad=12)
        print("im_centroids_shift = ", im_centroids_shift)
        print("centered original image: ", self.rs.image)
        self.assertEqual(im_centroids_shift, (-2,-2))



    def test_centre_image_padding_size(self):
        """
        test whether the output image is padded to a specified npixels_pad linear size
        :return:
        """
        #
        # initialise input image sampled at n_signal x n_signal non-zero-valued pixels surrounded by a zero-padded region of width n_topad
        # gives 6x6 image
        # test even n_signal
        n_signal = 2
        n_topad = 2
        #
        self.rs.image = np.ones((n_signal ,n_signal ))
        self.rs.image = pad(self.rs.image,
                     ((n_topad, n_topad),
                      (n_topad, n_topad)),
                     mode='constant')
        #
        # mimic image_segmentation
        self.rs.image_segmented = np.copy(self.rs.image)
        #
        # centre image
        self.rs.centre_image(npixels_pad=20, apodization=False)
        self.assertEqual(self.rs.image_centred.shape[0], 20)
        self.assertEqual(self.rs.image_centred.shape[1], 20)
        self.rs.centre_image(npixels_pad=20, apodization=True)
        self.assertEqual(self.rs.image_centred.shape[0], 20)
        self.assertEqual(self.rs.image_centred.shape[1], 20)
        #
        # initialise input image sampled at n_signal x n_signal non-zero-valued pixels surrounded by a zero-padded region of width n_topad
        # gives 7x7 image
        # test odd n_signal
        n_signal = 3
        n_topad = 2
        #
        self.rs.image = np.ones((n_signal ,n_signal ))
        self.rs.image = pad(self.rs.image,
                     ((n_topad, n_topad),
                      (n_topad, n_topad)),
                     mode='constant')
        #
        # mimic image_segmentation
        self.rs.image_segmented = np.copy(self.rs.image)
        #
        # centre image
        self.rs.centre_image(npixels_pad=20, apodization=False)
        self.assertEqual(self.rs.image_centred.shape[0], 20)
        self.assertEqual(self.rs.image_centred.shape[1], 20)
        self.rs.centre_image(npixels_pad=20, apodization=True)
        self.assertEqual(self.rs.image_centred.shape[0], 20)
        self.assertEqual(self.rs.image_centred.shape[1], 20)

    def test_centre_image_metadata(self):
        """
        test metadata
        :return:
        """
        #
        # initialise input image sampled at n_signal x n_signal non-zero-valued pixels surrounded by a zero-padded region of width n_topad
        # gives 6x6 image
        # test even n_signal
        n_signal = 2
        n_topad = 2
        #
        self.rs.image = np.ones((n_signal ,n_signal ))
        self.rs.image = pad(self.rs.image,
                     ((n_topad, n_topad),
                      (n_topad, n_topad)),
                     mode='constant')
        #
        # mimic image_segmentation
        self.rs.image_segmented = np.copy(self.rs.image)
        #
        # centre image
        _ = self.rs.centre_image(npixels_pad=20, apodization = False)
        self.assertEqual(self.rs.metadata['Linear number of pixels in the zero-padded real-space image'], 20)
        self.assertEqual(self.rs. metadata['Apodization filter applied?'], 'no' )

    def test_resample_image_empty_image(self):
        """
        test the case when an input image object is None
        :return:
        """
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.resample_image()

    def test_resample_image_pixelsize_dk(self):
        """
        test the size of the pixel in Fourier domain
        :return:
        """
        self.rs.image = np.zeros((6,6))
        self.rs.image[1:5,1:5] = 1
        print(self.rs.image)
        self.rs.metadata = RSpaceMetadata()
        self.rs.metadata['Image centred and padded?'] = 'yes'
        self.rs.metadata['Linear number of pixels in the zero-padded real-space image'] = 6
        pixelsize_dk, _, _, _ = self.rs.resample_image(fieldofview = 30, npixels_kspace = 6, lambd = np.pi/6, pixelsize_dr0 = 0.25, estimate_only = True)
        #
        self.assertEqual(pixelsize_dk, 1.0)

    def test_resample_image_downsampling(self):
        """
        test the downsampling ratio
        :return:
        """
        self.rs.image = np.zeros((6,6))
        self.rs.image[1:5,1:5] = 1
        print(self.rs.image)
        self.rs.metadata = RSpaceMetadata()
        self.rs.metadata['Image centred and padded?'] = 'yes'
        self.rs.metadata['Linear number of pixels in the zero-padded real-space image'] = 6
        _, _, downsampling, _ = self.rs.resample_image(fieldofview = 30, npixels_kspace = 6, lambd = np.pi/6, pixelsize_dr0 = 0.25, estimate_only = True)
        #
        self.assertEqual(downsampling, 4.0)

    def test_resample_image_npixels_final(self):
        """
        test the final linear number of pixels
        :return:
        """
        self.rs.image = np.zeros((6,6))
        self.rs.image[1:5,1:5] = 1
        print(self.rs.image)
        self.rs.metadata = RSpaceMetadata()
        self.rs.metadata['Image centred and padded?'] = 'yes'
        self.rs.metadata['Linear number of pixels in the zero-padded real-space image'] = 6
        _, _, _,npixels_final = self.rs.resample_image(fieldofview = 30, npixels_kspace = 6, lambd = np.pi/6, pixelsize_dr0 = 0.25, estimate_only = True)
        #
        self.assertEqual(npixels_final, 6.0)

    def test_resample_metadata(self):
        """
        test metadata
        :return:
        """
        self.rs.image = np.zeros((6,6))
        self.rs.image[1:5,1:5] = 1
        print(self.rs.image)
        self.rs.metadata = RSpaceMetadata()
        self.rs.metadata['Image centred and padded?'] = 'yes'
        self.rs.metadata['Linear number of pixels in the zero-padded real-space image'] = 6
        pixelsize_dk, pixelsize_dr_pad, _, _ = self.rs.resample_image(fieldofview = 30, npixels_kspace = 6, lambd = np.pi/6, pixelsize_dr0 = 0.25, estimate_only = True)
        #
        self.assertEqual(self.rs.metadata['Pixel size Fourier domain, 1/nm'], 1e9)
        self.assertEqual(self.rs.metadata['Pixel size object domain from padded Fourier data, 1/nm'], pixelsize_dr_pad * 1e9)
        self.assertEqual(self.rs.metadata['Wavelength of light, m'], np.pi/6)
        self.assertEqual(self.rs.metadata['Field of view, deg'], 30)
        self.assertEqual(self.rs.metadata['Number of pixels within the field of view'], 6)


    def test_save_as_tif_empty_image(self):
        """
        test empty input image
        :return:
        """
        self.rs.image = None
        with self.assertRaises(ValueError):
            self.rs.save_as_tif()

    def test_save_as_tif_empty_filename(self):
        """
        test empty filename
        :return:
        """
        self.rs.image = np.ones((3,3))
        with self.assertRaises(ValueError):
            self.rs.save_as_tif(outputfilename=None, pathtosave=self.test_dir)

    def test_save_as_tif_invalid_path(self):
        """
        test invalid path
        :return:
        """
        self.rs.image = np.ones((3,3))
        with self.assertRaises(ValueError):
            self.rs.save_as_tif(outputfilename='valid_filename.tif', pathtosave='some_invalid_path')

    def test_save_as_tif_save(self):
        """
        test saving
        :return:
        """
        # test saving
        self.rs.image = np.ones((3,3))
        outputfilename = 'valid_filename.tif'
        pathtosave = self.test_dir
        self.rs.save_as_tif(pathtosave, outputfilename)
        #
        # reset the rs.image
        self.rs.image = np.zeros((3,3))
        #
        # read saved image
        self.rs.read_from_tif(filename=os.path.join(pathtosave, outputfilename))
        #
        self.assertIsNotNone(self.rs.image)
        self.assertTrue(np.array_equal(self.rs.image, np.ones((3,3))))


if __name__ == '__main__':
    unittest.main()