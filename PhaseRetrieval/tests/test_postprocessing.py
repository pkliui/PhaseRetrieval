import unittest

from ddt import ddt

import csv, io, os, shutil, tempfile
import pandas as pd
import numpy as np

from PhaseRetrieval.modules import postprocessing

@ddt
class TestPostprocessing(unittest.TestCase):
    """
    Class to test the postprocessing module
    """
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        #import postprocessing module
        self.pp = postprocessing
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        #
        self.test_amplitude = pd.DataFrame([[0, 2, 7, 4, 8],
                             [1, 7, 6, 3, 7],
                             [1, 1, 1, 8, 9],
                             [0, 2, 3, 1, 6],
                             [0, 5, 1, 4, 9]])
        #
        self.test_phase = pd.DataFrame([[1.0, 1.0, 1.0, 1.0, 1.0],
                                            [1.0, 1.0, 1.0, 1.0, 1.0],
                                            [1.0, 1.0, 0.0, 1.0, 1.0],
                                            [1.0, 1.0, 1.0, 1.0, 1.0],
                                            [1.0, 1.0, 1.0, 1.0, 1.0]])
        #
        self.test_aligned_phase = pd.DataFrame([[3.0, 3.0, 3.0, 3.0, 3.0],
                                            [3.0, 3.0, 3.0, 3.0, 3.0],
                                            [3.0, 3.0, 0.0, 3.0, 3.0],
                                            [3.0, 3.0, 3.0, 3.0, 3.0],
                                            [3.0, 3.0, 3.0, 3.0, 3.0]])
        #
        self.test_aligned_phase_norm = pd.DataFrame([[1.0, 1.0, 1.0, 1.0, 1.0],
                                            [1.0, 1.0, 1.0, 1.0, 1.0],
                                            [1.0, 1.0, 0.0, 1.0, 1.0],
                                            [1.0, 1.0, 1.0, 1.0, 1.0],
                                            [1.0, 1.0, 1.0, 1.0, 1.0]])
        #
        # Write test amplitude data into a temporary file
        self.tmp_amplitude_file = os.path.join(self.test_dir, 'tmp_amplitude.csv')
        with open(self.tmp_amplitude_file, 'w') as amplitude_file:
            writer = csv.writer(amplitude_file)
            writer.writerows(self.test_amplitude.values)
        #
        # Write test phase data into a temporary file
        self.tmp_phase_file1 = os.path.join(self.test_dir, 'tmp_phase1.csv')
        with open(self.tmp_phase_file1, 'w') as phase_file:
            writer = csv.writer(phase_file)
            writer.writerows(self.test_phase.values)
        #
        self.tmp_phase_file2 = os.path.join(self.test_dir, 'tmp_phase2.csv')
        with open(self.tmp_phase_file2, 'w') as phase_file:
            writer = csv.writer(phase_file)
            writer.writerows(self.test_phase.values)
        #
        self.tmp_phase_file3 = os.path.join(self.test_dir, 'tmp_phase3.csv')
        with open(self.tmp_phase_file3, 'w') as phase_file:
            writer = csv.writer(phase_file)
            writer.writerows(self.test_phase.values)

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_amplitude_filename_phase_alignment_gerchberg_saxton(self):
        with self.assertRaises(ValueError):
            self.pp.phase_alignment_gerchberg_saxton(amplitude_filename='some_none_existing_data.csv',
                                                     delimiter = '\t')

    def test_phase_filename_phase_alignment_gerchberg_saxton(self):
        with self.assertRaises(ValueError):
            self.pp.phase_alignment_gerchberg_saxton(amplitude_filename='some_none_existing_data.csv',
                                                     delimiter = '\t',
                                                     phase_filenames = ['some_none_existing_data.csv',
                                                                        'some_none_existing_data.csv'])
    def test_ref_coordinates_phase_alignment_gerchberg_saxton(self):
        #
        #ref_coordinates are None
        read_amplitude, read_phase, ref_coordinates, _, _, _ = self.pp.phase_alignment_gerchberg_saxton(amplitude_filename=self.tmp_amplitude_file,
                                                 delimiter=',',
                                                 phase_filenames = [self.tmp_phase_file1, self.tmp_phase_file2, self.tmp_phase_file3],
                                                 ref_coordinates = None,
                                                 num_files_to_align = 3)
        #
        #check the shape of input data
        self.assertEqual(read_amplitude.shape, (5,5))
        self.assertEqual(read_phase.shape, (5,5))
        #
        #ref_coordinates must be [int(phase.shape[0]) // 2 + 1, int(phase.shape[1]) // 2 + 1]
        self.assertEqual(ref_coordinates, [2,2])
        #
        #ref_coordinates are set by user
        read_amplitude, read_phase, ref_coordinates, _, _, _ = self.pp.phase_alignment_gerchberg_saxton(amplitude_filename=self.tmp_amplitude_file,
                                                 delimiter=',',
                                                 phase_filenames = [self.tmp_phase_file1, self.tmp_phase_file2, self.tmp_phase_file3],
                                                 ref_coordinates = [2,2],
                                                 num_files_to_align = 3)

        self.assertEqual(read_amplitude.shape, (5,5))
        self.assertEqual(read_phase.shape, (5,5))
        #ref_coordinates must be [2,2] as set by user
        self.assertEqual(ref_coordinates, [2,2])

    def test_aligned_phases_phase_alignment_gerchberg_saxton(self):
        #
        read_amplitude, read_phase, ref_coordinates, len_phase_filenames,_,_  = self.pp.phase_alignment_gerchberg_saxton(amplitude_filename=self.tmp_amplitude_file,
                                                 delimiter=',',
                                                 phase_filenames = [self.tmp_phase_file1, self.tmp_phase_file2, self.tmp_phase_file3],
                                                 ref_coordinates = None,
                                                 num_files_to_align = 3,
                                                 symmetric_phase = False)
        #
        #test the length of filenames
        self.assertEqual(len_phase_filenames, 3)
        #
        read_amplitude, read_phase, ref_coordinates, len_phase_filenames, aligned_phase, aligned_phase_norm  = self.pp.phase_alignment_gerchberg_saxton(amplitude_filename=self.tmp_amplitude_file,
                                                 delimiter=',',
                                                 phase_filenames = [self.tmp_phase_file1, self.tmp_phase_file2, self.tmp_phase_file3],
                                                 ref_coordinates = None,
                                                 num_files_to_align = 3,
                                                 symmetric_phase = False)
        #
        #test if the phase summation goes well
        self.assertTrue(np.array_equal(aligned_phase, self.test_aligned_phase.values))
        #
        #test if the phase averaging goes well
        self.assertTrue(np.array_equal(aligned_phase_norm, self.test_aligned_phase_norm.values))

