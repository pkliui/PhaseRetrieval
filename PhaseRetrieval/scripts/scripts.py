"""
Scripts to run modules and functions for image pre-processing, phase retrieval and  post-processing of reconstructed images
"""

import glob, os, shutil, sys

from PhaseRetrieval.classes.rspaceimage import RSpaceImage
from PhaseRetrieval.classes.kspaceimage import KSpaceImage
from PhaseRetrieval.classes.phaseretrieval import PhaseRetrieval
from PhaseRetrieval.modules.postprocessing import phase_alignment_gerchberg_saxton

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def preprocessing_script(datapath = None,
                           rs_filename = None,
                           ks_filename = None,
                           delimiter = '\t',
                           lambd=880e-9,
                           rs_linear_object_size = 100e-6,
                           rs_pixelsize=340e-9,
                           ks_fieldofview = 17,
                           ks_npixels = 500,
                           ks_centre=(1, 1),
                           ks_gaussian_filter=True,
                           ks_sigma=1,
                           ks_roi=(400, 400, 550, 550),
                           ks_min_distance=5,
                           ks_threshold_abs=5000,
                           ks_num_peaks=5,
                           npixels_pad_iter = None,
                           ks_noise_iter = None,
                           rs_noise_iter=None,
                           suppress_print = False):
    """
    Script for image processing of object- and Fourier-domain images prior to application of phase retrieval algorithms.
    With options for various linear number of pixels and subtracted noise levels.
    Processed images are saved as tif in the same working directory.
    ---
    Parameters
    Parameters
    ---
    --- Initialization of RSpaceImage and KSpaceImage classes-related:---
    datapath: str
        Path used to load data.
        Default is None.
    rs_filename: str
        Name of the file containing object-domain image.
        Default is None.
    ks_filename: str
        Name of the file containing Fourier-domain image.
        Default is None.
    delimiter : str, optional
        Delimiter in csv files.
        Default is '\t'.
    ---
    --- RSpaceImage.centre_image_watershed method-related:---
    rs_linear_object_size: float, optional
        Physical linear size of the (non-zero-valued) input object distribution, in m.
        Default is 100 micrometer.
    --- RSpaceImage.resample_image method-related:---
    ks_fieldofview: int, optional
        One half of field of view in Fourier-space, in degrees
        Default is ±17°, i.e. fieldofview = 17
    ks_npixels: int, optional
        One half of linear number of non-zero-valued pixels in experimental Fourier-space image to be used together with the real-space image
        (= corresponds to the linear number of pixels within the 1/2 of field of view)
        Default is 500
    rs_pixelsize: int, optional
        Pixel size in the object-domain image, in m.
        If it is None, the value will be read from metadata (saved to metadata after centering and segmentation of the image).
        If the value in metadata is None, there will be an error message.
        Alternatively, one can set the pixel size manually.
        Default is None.
    lambd: float, optional
        Wavelength of light, in m
        Default is 880e-9
    ---
    --- KSpaceImage.centre_image method-related:---
    ks_centre: tuple, optional
        Centre of the image.
        Default is (1,1), which must be changed by user once the centre of the image (one of the local maxima) is found.
    ks_gaussian_filter : bool, optional
        Apply Gaussian filter to filter noise while centering the image with centre_image method
        Default is False
    ks_sigma : float, optional
        Standard deviation of a Gaussian filter used in centre_image method
        Default is 1.0
    ks_roi: tuple, optional
        Region of interest (ROI) used to search for local maxima.
        Default is (0,10,0,10).
    ks_min_distance, int, optional
        Minimal distance between the local maxima used in centre_image method
        Must be tuned by user to make the search most effective.
        Default is 10.
    ks_threshold_abs: float, optional
        Minimum intensity of peaks used in centre_image method.
        Default is 0.
    ks_num_peaks: int, optional
        Maximum number of peaks used in centre_image method.
        When the number of peaks exceeds num_peaks, return num_peaks peaks based on highest peak intensity.
        Default is 1.
    npixels_pad_iter : list of int
        List of integers denoting final linear number of pixels in the processed images.
        Default is None
    ---
    --- RSpaceImage.subtract_background method-related:---
    rs_noise_iter : list of int
        List of integers denoting the noise levels to be stubtracted from the object-domain image
        Default is None
    ---
    --- KSpaceImage.subtract_background method-related:---
    ks_noise_iter : list of int
        List of integers denoting the noise levels to be stubtracted from the Fourier-domain image
        Default is None
    ---
    suppress_print : bool, optional
        False will block print calls.
        True will let print calls to be displayed.
        Default is False.
    """

# generate images with different linear number of pixels
    for npixels_pad in npixels_pad_iter:
        #
        # subtract different noise levels in Fourier domain
        for ks_noise in ks_noise_iter:
#
            # subtract noise in object domain
            for rs_noise in rs_noise_iter:
                #
                # change to data directory
                os.chdir(datapath)
                #
                ##### image processing workflow #####
                #
                # read data, rotate and flip
                # object-domain data
                #
                if suppress_print is True:
                    with HiddenPrints():
                        rs = RSpaceImage(filename=os.path.join(datapath, rs_filename),
                                         delimiter=delimiter)
                        rs.rotate_image(estimate_only=False,
                                        plot_progress=False)
                        rs.flip_image(estimate_only=False,
                                      plot_progress=False)
                        ##Fourier-domain data
                        ks = KSpaceImage(filename=os.path.join(datapath, ks_filename),
                                         delimiter=delimiter)
                        ks.rotate_image(estimate_only=False,
                                        plot_progress=False)
                        ks.flip_image(estimate_only=False,
                                      plot_progress=False)
                        #
                        # subtract noise in object domain
                        rs.subtract_background(counts=rs_noise,
                                               estimate_only=False,
                                               log_scale=True,
                                               plot_progress=False)
                        #
                        # apply watershed segmentation in object domain and centre the image
                        rs.centre_image_watershed(linear_object_size=rs_linear_object_size,
                                                  npixels_pad=npixels_pad,
                                                  apodization=True,
                                                  plot_progress=False)
                        #
                        # resample in object domain to make sure the sizes of the pixels are congruent with the digital Fourier transform
                        npixels_final = rs.resample_image(fieldofview=ks_fieldofview,
                                                          npixels_kspace=ks_npixels,
                                                          pixelsize_dr0=rs_pixelsize,
                                                          lambd=lambd,
                                                          estimate_only=False)
                        #
                        # centre image in Fourier domain
                        ks.centre_image(estimate_only=False,
                                        centre=ks_centre,
                                        gaussian_filter=ks_gaussian_filter,
                                        sigma=ks_sigma,
                                        roi=ks_roi,
                                        min_distance=ks_min_distance,
                                        threshold_abs=ks_threshold_abs,
                                        num_peaks=ks_num_peaks,
                                        npixels_pad=npixels_final,
                                        plot_progress=False)
                        #
                        # subtract background in Fourier domain
                        ks.subtract_background(counts=ks_noise,
                                               estimate_only=False,
                                               plot_progress=False)
                        #
                        # fulfill Parseval's theorem
                        energy_rspace = sum(sum(rs.image))
                        ks.renormalise_image(energy_rspace)
                        #
                        # save images together with their metadata (same directory)
                        rs.save_as_tif(pathtosave=datapath,
                                       outputfilename = "Ntot" + str(npixels_pad) + "rsnoise" + str(
                                           rs_noise) + "ksnoise" + str(ks_noise) + "_" + rs_filename[:-4] + "_amplitude.tif")
                        ks.save_as_tif(pathtosave=datapath,
                                       outputfilename = "Ntot" + str(npixels_pad) + "rsnoise" + str(
                                           rs_noise) + "ksnoise" + str(ks_noise) + "_" + ks_filename[:-4] + "_amplitude.tif")
                else:
                    print('Processing images... Processing with linear number of pixels = ', npixels_pad,',Fourier-domain noise = ', ks_noise, ',Object-domain noise = ', rs_noise)
                    #
                    rs = RSpaceImage(filename=os.path.join(datapath, rs_filename),
                                     delimiter=delimiter)
                    rs.rotate_image(estimate_only=False,
                                    plot_progress=False)
                    rs.flip_image(estimate_only=False,
                                  plot_progress=False)
                    ##Fourier-domain data
                    ks = KSpaceImage(filename=os.path.join(datapath, ks_filename),
                                     delimiter=delimiter)
                    ks.rotate_image(estimate_only=False,
                                    plot_progress=False)
                    ks.flip_image(estimate_only=False,
                                  plot_progress=False)
                    #
                    # subtract noise in object domain
                    rs.subtract_background(counts=rs_noise,
                                           estimate_only=False,
                                           log_scale=True,
                                           plot_progress=False)
                    #
                    # apply watershed segmentation in object domain and centre the image
                    rs.centre_image_watershed(linear_object_size=rs_linear_object_size,
                                              npixels_pad=npixels_pad,
                                              apodization=True,
                                              plot_progress=False)
                    #
                    # resample in object domain to make sure the sizes of the pixels are congruent with the digital Fourier transform
                    npixels_final = rs.resample_image(fieldofview=ks_fieldofview,
                                                      npixels_kspace=ks_npixels,
                                                      pixelsize_dr0=rs_pixelsize,
                                                      lambd=lambd,
                                                      estimate_only=False)
                    #
                    # centre image in Fourier domain
                    ks.centre_image(estimate_only=False,
                                    centre=ks_centre,
                                    gaussian_filter=ks_gaussian_filter,
                                    sigma=ks_sigma,
                                    roi=ks_roi,
                                    min_distance=ks_min_distance,
                                    threshold_abs=ks_threshold_abs,
                                    num_peaks=ks_num_peaks,
                                    npixels_pad=npixels_final,
                                    plot_progress=False)
                    #
                    # subtract background in Fourier domain
                    ks.subtract_background(counts=ks_noise,
                                           estimate_only=False,
                                           plot_progress=False)
                    #
                    # fulfill Parseval's theorem
                    energy_rspace = sum(sum(rs.image))
                    ks.renormalise_image(energy_rspace)
                    #
                    # save images together with their metadata (same directory)
                    rs.save_as_tif(pathtosave=datapath,
                                   outputfilename = "Ntot" + str(npixels_pad) + "rsnoise" + str(
                                       rs_noise) + "ksnoise" + str(ks_noise) + "_" +rs_filename[:-4] + "_amplitude.tif")
                    ks.save_as_tif(pathtosave=datapath,
                                   outputfilename = "Ntot" + str(npixels_pad) + "rsnoise" + str(
                                       rs_noise) + "ksnoise" + str(ks_noise) + "_" + ks_filename[:-4] + "_amplitude.tif")

                    print('Image processing completed.')


def gerchberg_saxton_script(datapath = None,
                                   rs_prefix = None,
                                   ks_prefix = None,
                                   files_extension = "*.tif",
                                   gs_steps = 100,
                                   plot_progress = False,
                                   plot_every_kth_iteration = 1,
                                   zoom=1,
                                   Fourier_amplitude = True,
                                   Fourier_phase = True,
                                   object_amplitude = True,
                                   object_phase = True,
                                   filename = None,
                                   rec_number=1,
                                   suppress_print = False):
    """
    Script to launch GS algorithm and save reconstructed images.
    Customized for the case when image are saved in several folders (e.g. each containing images with different parameters)
    Reconstructed images are saved as csv (no metadata) in individual folders containing images of the same kind (e.g. equal number of pixels)
    ---
    Parameters
    ---
    files_extension: {"*.tif", "*.csv"}
        Extension of raw images.
        Must be chosen from the given set of extensions, i.e. either "*.tif" or "*.csv".
        Default is "*.tif".
    ---Initialisation of PhaseRetrieval class-related:---
    datapath: str
        Path used to load data (the same as used to load raw data).
        Default is None.
    rs_prefix: str
        Prefix common to names of the files containing object-domain images.
        Default is None.
    ks_filename: str
        Prefix common to names of the files containing Fourier-domain images.
        Default is None.
    ---
    ---PhaseRetrieval.gerchberg_saxton_extrapolation method-related:---
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
    ---PhaseRetrieval.save_as_tif method-related:---
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
    ---
    rec_number : int, optional
        Number of times the algorithm is run with random initial phases = number of reconstructions
        Default is 1.
    suppress_print : bool, optional
        False will block print calls.
        True will let print calls to be displayed.
        Default is False.
    """
    # read file names
    filenames_list = sorted(glob.glob(os.path.join(datapath, files_extension)), key=os.path.getmtime)
    print(filenames_list)
    # object-domain filenames
    rs_filenames = sorted([file for file in filenames_list if rs_prefix in file], key=os.path.getmtime)
    print(enumerate(sorted(rs_filenames)))
    # Fourier-domain filenames
    ks_filenames = sorted([file for file in filenames_list if ks_prefix in file], key=os.path.getmtime)
    print(enumerate(sorted(ks_filenames)))

    #first go through all object-domain files (intensity distributions)
    for rs_idx, rs_file in enumerate(rs_filenames):
        #create folder having the name of  the current  file
        folderName = rs_file[-len(str(rs_filenames))+4: -14]
        #
        #copy object-domain data (intensity distributions)
        if not os.path.exists(folderName):
            os.mkdir(folderName)
            shutil.copy(rs_file, folderName)
        else:
            shutil.copy(rs_file, folderName)
        #
        #now go through all Fourier-domain files (intensity distributions) and copy them to the respective folders
        for _, ks_file in enumerate(ks_filenames):
            ks_file = ks_filenames[rs_idx]
            if os.path.exists(folderName):
                shutil.copy(ks_file, folderName)
            else:
                raise ValueError("Invalid path! Object- and Fourier domain filenames must have the same endings!")
                    #initialise phase retrieval class
        #
        #initialise phase retrieval class
        pr = PhaseRetrieval(filename_rspace=rs_file, filename_kspace=ks_file)
        #
        #phase retrieval
        for ii in range(0, rec_number):
            #
            if suppress_print is True:
                with HiddenPrints():
                    pr.gerchberg_saxton(gs_steps = gs_steps,
                                                      plot_progress = plot_progress,
                                                      plot_every_kth_iteration = plot_every_kth_iteration,
                                                      zoom = zoom)##
                    pr.save_as_csv(filename = filename,
                                   pathtosave = os.path.join(datapath, folderName),
                                   Fourier_amplitude = Fourier_amplitude,
                                   Fourier_phase = Fourier_phase,
                                   object_amplitude = object_amplitude,
                                   object_phase = object_phase)
            else:
                pr.gerchberg_saxton(gs_steps=gs_steps,
                                                  plot_progress=plot_progress,
                                                  plot_every_kth_iteration=plot_every_kth_iteration,
                                                  zoom=zoom)  ##
                pr.save_as_csv(filename=filename,
                               pathtosave=os.path.join(datapath, folderName),
                               Fourier_amplitude=Fourier_amplitude,
                               Fourier_phase=Fourier_phase,
                               object_amplitude=object_amplitude,
                               object_phase=object_phase)

def gerchberg_saxton_extrapolation_script(datapath = None,
                                   rs_prefix = None,
                                   ks_prefix = None,
                                   files_extension = "*.tif",
                                   gs_steps = 100,
                                   plot_progress = True,
                                   plot_every_kth_iteration = 1,
                                   zoom=1,
                                   Fourier_amplitude = True,
                                   Fourier_phase = True,
                                   object_amplitude = True,
                                   object_phase = True,
                                   filename = None,
                                   rec_number=1,
                                   print_progress = True):
    """
    Script to launch GS algorithm with extrapolation and save reconstructed images.
    Customized for the case when image are saved in several folders (e.g. each containing images with different parameters)
    Reconstructed images are saved as csv (no metadata) in individual folders containing images of the same kind (e.g. equal number of pixels)
    ---
    Parameters
    ---
    files_extension: {"*.tif", "*.csv"}
        Extension of raw images.
        Must be chosen from the given set of extensions, i.e. either "*.tif" or "*.csv".
        Default is "*.tif".
    ---Initialisation of PhaseRetrieval class-related:---
    datapath: str
        Path used to load data (the same as used to load raw data).
        Default is None.
    rs_prefix: str
        Prefix common to names of the files containing object-domain images.
        Default is None.
    ks_filename: str
        Prefix common to names of the files containing Fourier-domain images.
        Default is None.
    ---
    ---PhaseRetrieval.gerchberg_saxton_extrapolation method-related:---
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
    ---PhaseRetrieval.save_as_tif method-related:---
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
    ---
    rec_number : int, optional
        Number of times the algorithm is run with random initial phases = number of reconstructions
        Default is 1.
    print_progress : bool, optional
        False will block print calls.
        True will let print calls to be displayed.
        Default is False.
    """
    if files_extension is ".*tif" or "*.csv":
        # read file names
        filenames_list = sorted(glob.glob(os.path.join(datapath, files_extension)), key=os.path.getmtime)
        # object-domain filenames
        rs_filenames = [file for file in filenames_list if rs_prefix in file]
        # Fourier-domain filenames
        ks_filenames = [file for file in filenames_list if ks_prefix in file]
        #
        #first go through all object-domain files
        for rs_idx, rs_file in enumerate(sorted(rs_filenames)):
            #
            #create folders with respective file names
            folderName = os.path.basename(rs_file)[0:-4]
            #
            #copy object-domain data
            if not os.path.exists(folderName):
                os.mkdir(folderName)
                shutil.copy(rs_file, folderName)
            else:
                shutil.copy(rs_file, folderName)
            #
            #now go through all Fourier-domain files and copy them to the respective folders
            for ks_file in sorted(ks_filenames):
                ks_file = ks_filenames[rs_idx]
                if os.path.exists(folderName):
                    shutil.copy(ks_file, folderName)
                else:
                    raise ValueError("Invalid path! Object- and Fourier domain filenames must have the same endings!")
                        #initialise phase retrieval class
            #
            #initialise phase retrieval class
            pr = PhaseRetrieval(filename_rspace=rs_file, filename_kspace=ks_file)
            #
            #phase retrieval
            for ii in range(0, rec_number):
                #
                if print_progress is True:
                    pr.gerchberg_saxton_extrapolation(gs_steps = gs_steps,
                                                      plot_progress = plot_progress,
                                                      plot_every_kth_iteration = plot_every_kth_iteration,
                                                      zoom = zoom)##
                    pr.save_as_csv(filename = filename,
                                   pathtosave = os.path.join(datapath, folderName),
                                   Fourier_amplitude = Fourier_amplitude,
                                   Fourier_phase = Fourier_phase,
                                   object_amplitude = object_amplitude,
                                   object_phase = object_phase)
                else:
                    with HiddenPrints():
                        pr.gerchberg_saxton_extrapolation(gs_steps=gs_steps,
                                                          plot_progress=plot_progress,
                                                          plot_every_kth_iteration=plot_every_kth_iteration,
                                                          zoom=zoom)  ##
                        pr.save_as_csv(filename=filename,
                                       pathtosave=os.path.join(datapath, folderName),
                                       Fourier_amplitude=Fourier_amplitude,
                                       Fourier_phase=Fourier_phase,
                                       object_amplitude=object_amplitude,
                                       object_phase=object_phase)
    else:
        raise ValueError("file_extension argument must be either 'tif' or 'csv'! ")

def phase_alignment_gerchberg_saxton_script(datapath=None,
                                     folder_id=None,
                                     phase_id=None,
                                     amplitude_id=None,
                                     amplitude_extension = None,
                                     delimiter = '\t',
                                     num_files_to_align=None,
                                     ref_coordinates=None,
                                     symmetric_phase = True,
                                     plot_progress=True,
                                     plot_every_kth_iteration=1,
                                     zoom=1,
                                     print_progress = True):
    """
    Script for alignment of phase images yielded by Gechrberg-Saxton algorithm (here it is assumed that images to align are in csv file format).
    It is assumed that the corresponding amplitude is known and the phases are all in spatial registry.
    That is, no registration is needed.
    ---
    Parameters
    ---
    datapath : str
        Path to a directory with folders containing reconstructed phase images.
        Default is None.
    folder_id : str
        String that must be contained in the name(s) of the folder that contains files to be read and aligned
        Default is None
    phase_id : str
        String that must be contained in the names of the files with phase distributions to be read and aligned
        Default is None
    amplitude_id : str, optional
        String that must be contained in the names of the files with the amplitude distribution associated with the phase images to be aligned.
        Default is None
    amplitude_extension : str, optional
        Extension of the amplitude distribution specified as either "tif" or "csv".
        Default is None
    delimiter : str, optional
        Delimiter in csv files
        Default is '\t'
    ref_coordinates: list [1x2], optional
        Coordinates of a reference phase value.
        If set to None, the ref_coordinates will be set the image's centre coordinates (i.e. [501,501] for 1000x1000 pixels image)
        If a list is provided, the values will be taken from there.
    symmetric_phase ; bool, optional
        If set to True, pixel values in the final phase distribution will be shifted symmetrically w.r.t zero
        If False, pixels values in the final phase distribution will be left as they are after alignment
        Default is True
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
    print_progress : bool, optional
        False will block print calls.
        True will let print calls to be displayed.
        Default is False.
    """
    if datapath and folder_id and phase_id is not None:
        #
        # get sub-directories
        subdir_list = [os.path.join(datapath, subdir) for subdir in os.listdir(datapath) if
                       os.path.isdir(os.path.join(datapath, subdir)) and folder_id in subdir]
        print(subdir_list)
        #
        # go to each sub-directory and align images there
        for ii in range(0, len(subdir_list)):
            #
            # read file names in the directory (sorted by time saved)
            filenames_list = sorted(glob.glob(os.path.join(subdir_list[ii], '*.csv')), key=os.path.getmtime)
            print(filenames_list)
            #
            # load amplitude image
            if amplitude_id is not None:
                # filter names of files containing amplitude image
                amplitude_filenames_list = sorted(glob.glob(os.path.join(subdir_list[ii], "*" + amplitude_id +  "." + amplitude_extension)), key=os.path.getmtime)
                print(amplitude_filenames_list)
                amplitude_filename = [file for file in amplitude_filenames_list]
                amplitude_filename = ''.join(amplitude_filename)
                print(amplitude_filename)
            else:
                raise ValueError("Filename of the amplitude data must be specified.")

            #
            # filter names of files containing phase images
            phase_filenames = [file for file in filenames_list if phase_id in file]
            print(phase_filenames)
            #
            if print_progress is True:
                phase_alignment_gerchberg_saxton(amplitude_filename=amplitude_filename,
                                                 phase_filenames=phase_filenames,
                                                 num_files_to_align=num_files_to_align,
                                                 delimiter=delimiter,
                                                 ref_coordinates=ref_coordinates,
                                                 symmetric_phase=symmetric_phase,
                                                 plot_progress=plot_progress,
                                                 plot_every_kth_iteration=plot_every_kth_iteration,
                                                 zoom=zoom)
            else:
                phase_alignment_gerchberg_saxton(amplitude_filename=amplitude_filename,
                                                 phase_filenames=phase_filenames,
                                                 num_files_to_align=num_files_to_align,
                                                 delimiter=delimiter,
                                                 ref_coordinates=ref_coordinates,
                                                 symmetric_phase=symmetric_phase,
                                                 plot_progress=plot_progress,
                                                 plot_every_kth_iteration=plot_every_kth_iteration,
                                                 zoom=zoom)
    else:
        raise ValueError('Invalid path! Provide path to the directory with data')