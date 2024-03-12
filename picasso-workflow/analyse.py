#!/usr/bin/env python
"""
Module Name: analyse.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: This is the picasso interface of picasso-workflow
"""
from picasso import io, localize, gausslq, postprocess
from picasso import __version__ as picassoversion
import os
import time
from concurrent import futures as _futures
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import logging

from picasso_workflow.util import AbstractModuleCollection, get_caller_name


logger = logging.getLogger(__name__)


class AutoPicasso(AbstractModuleCollection):
    """A class to automatically evaluate datasets.
    Each module that runs saves their results into a separate folder.
    """
    self.movie = None
    self.info = None
    self.identifications = None
    self.locs = None
    self.drift = None

    def __init__(self, results_folder, analysis_config):
        """
        Args:
            results_folder : str
                the folder all analysis modules save their respective results to
            analysis_config : dict
                the general configuration. necessary items:
                    camera_info : dict
                        as used by picasso
                    gpufit_installed : bool
                        whether the machine has gpufit installed
        """
        self.results_folder = results_folder
        self.analysis_config = analysis_config

    #################################################################################
    #### MODULES
    #################################################################################

    @module_decorator
    def load_dataset(self, i, parameters, results):
        """Loads a DNA-PAINT dataset in a format supported by picasso.
        The data is saved in
            self.movie
            self.info
        Args:
            parameters : dict
                necessary items:
                    filename : str
                        the (main) file name to load
                optional items:
                    sample_movie : dict, used for creating a subsampled movie
                        keywords as used in method create_sample_movie
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results
        """
        results['picasso version'] = picassoversion,
        t00 = time.time()
        self.movie, self.info = io.load_movie(parameters['filename'])
        results['movie.shape'] = self.movie.shape

        # create sample movie
        if (samplemov_pars := parameters.get('sample_movie')) is not None:
            samplemov_pars['filename'] = os.path.join(
                module_result_dir, samplemov_pars['filename'])
            res = self._create_sample_movie(**samplemov_pars)
            results['sample_movie'] = res

        dt = np.round(time.time() - t00, 2)
        results['duration'] = dt
        return parameters, results

    def _create_sample_movie(self, 
            saveas, n_sample=30, min_quantile=0, max_quantile=.9998, fps=1):
        """Create a subsampled movie of the movie loaded. The movie is saved
        to disk and referenced by filename.
        Args:
            saveas : str
                the file name to save the subsamled movie as (.mp4)
            rest: as in save_movie
        """
        results = {}
        # frame_numbers = np.random.choice(np.arange(len(self.movie)), n_sample)
        if len(self.movie) < n_sample:
            n_sample = len(self.movie)

        dn = int(len(self.movie) / (n_sample - 1))
        frame_numbers = np.arange(0, len(self.movie), dn)
        results['sample_frame_idx'] = frame_numbers

        subsampled_frames = np.array([self.movie[i] for i in frame_numbers])
        save_movie(
            saveas, subsampled_frames,
            min_quantile=min_quantile, max_quantile=max_quantile, fps=fps)
        results['filename'] = fn_movie
        return results

    def _auto_min_netgrad(self, plot_fn, box_size, frame_numbers, start_ng=-3000, zscore=5, bins=None):
        """Calculate histograms of the net gradient at local maxima of n frames.
        For the automatic calculation of a threshold net_gradient for localizations,
        assume the background (of random local maxima without a localization signal)
        to be Gaussian distributed. Assume the background peak in the histogram
        is the highest value. The threshold net_gradient will be determined as
        zscore bacground standard deviations above the peak. 

        Args:
            plot_fn : str
                the file name of the plot to be created
            box_size : int
                the box size for evaluation
            frame_numbers : list
                the frame indexes to analyze
            start_ng : float
                the minimum net gradient to accept for the histogram.
                this should be below zero, to capture all net gradient
                values that exist in the data
            zscore : float
                the number of sigmas above the background net gradient peak
                to set as the estimated min net gradient threshold
            bins : None, int or array
                specify the bins of the histogram
        Returns:
            filename : string
                filename of the generated plot
            estd_net_grad : float
                the estimated min net gradient
        """
        results = {}
        identifications = []

        for frame_number in frame_numbers:
            identifications.append(
                localize.identify_by_frame_number(
                    self.movie, start_ng, box_size, frame_number))
        # id_list = identifications
        identifications = np.hstack(identifications).view(np.recarray)
        identifications.sort(kind="mergesort", order="frame")

        # calculate histogram
        if bins is None:
            hi = np.quantile(identifications['net_gradient'], .9995)
            bins = np.linspace(start_ng, hi, num=200)
        hist, edges = np.histogram(
            identifications['net_gradient'], bins=bins, density=True)

        # find the background peak, assume Gaussian, find max and FWHM
        # FWHM as the most robust detection for peak width
        bkg_peak_height, bkg_peak_pos = np.max(hist), np.argmax(hist)
        bkg_halfclose = np.argsort(np.abs(hist - bkg_peak_height / 2))
        bkg_fwhm = np.abs(bkg_halfclose[1] - bkg_halfclose[0])
        bkg_sigma = bkg_fwhm / np.sqrt(4 * np.log(2))
        ng_est_idx = int(zscore * bkg_sigma) + bkg_peak_pos  # threshold at zscore * bkg_sigma
        if ng_est_idx >= len(edges):
            ng_est_idx = len(edges) - 1
        results['estd_net_grad'] = edges[ng_est_idx]

        # plot results
        fig, ax = plt.subplots()
        ax.plot(edges[:-1], hist, color='b', label='combined histogram')
        # for i, frame_number in enumerate(frame_numbers):
        #     hi, ed = np.histogram(id_list[i]['net_gradient'], bins=bins, density=True)
        #     ax.plot(ed[:-1], hi, color='gray')
        ylims = ax.get_ylim()
        ax.set_title('Net Gradient histogram of subsampled frames')
        ax.set_xlabel('net gradient')
        ax.set_yscale('log')
        ax.plot(
            [results['estd_net_grad'], results['estd_net_grad']], ylims, color='r',
            label='estimated min net gradient: {:.0f}'.format(results['estd_net_grad']))
        ax.plot([edges[bkg_peak_pos], edges[bkg_peak_pos]], ylims,
            color='gray', label='detected background peak')
        ax.legend()
        # plt.show()
        results['filename'] = plot_fn
        fig.savefig(results['filename'])
        return results

    @module_decorator
    def identify(self, i, parameters, results):
        """Identifies localizations in a loaded dataset.
        The data is saved in
            self.identifications
        Args:
            parameters : dict
                necessary items:
                    box_size : as always
                    min_gradient : only if not auto_netgrad
                optional items:
                    auto_netgrad : dict, in case the min net_gradient
                        shall be automatically detected.
                        Items correspond to arguments of _auto_min_netgrad
                    ids_vs_frame : dict
                        for plotting identifications vs time
                        items correspond to arguments of _plot_ids_vs_frame
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results
        """
        t00 = time.time()

        # auto-detect net grad if required:
        if (autograd_pars := parameters.get('auto_netgrad')) is not None:
            if 'filename' in autograd_pars.keys():
                autograd_pars['filename'] = os.path.join(
                    module_result_dir, autograd_pars['filename'])
            res = self._auto_min_netgrad(**autograd_pars)
            results['auto_netgrad'] = res
            parameters['min_gradient'] = res['estd_net_grad']

        curr, futures = localize.identify_async(
            self.movie,
            parameters['min_gradient'],
            parameters['box_size'],
            roi=None,
        )
        self.identifications = localize.identifications_from_futures(futures)
        dt = np.round(time.time() - t00, 2)
        results['duration'] = dt
        results['num_identifications'] = len(self.identifications)

        if (pars := parameters.get('ids_vs_frame')) is not None:
            if 'filename' in pars.keys():
                pars['filename'] = os.path.join(
                    module_result_dir, pars['filename'])
            results['ids_vs_frame'] = self._plot_ids_vs_frame(**pars)
        return parameters, results

    def _plot_ids_vs_frame(self, filename):
        """Plot identifications vs frame index
        """
        results = {}
        frames = np.arange(len(self.movie))
        bins = np.arange(len(self.movie) + 1) - .5
        locs, _ = np.histogram(self.identifications['frame'], bins=bins)
        fig, ax = plt.subplots()
        ax.plot(frames, locs)
        ax.set_xlabel('frame')
        ax.set_ylabel('number of identifications')
        results['filename'] = filename
        fig.savefig(results['filename'])
        plt.close(fig)
        return results

    @module_decorator
    def localize(self, i, parameters, results):
        """Localizes Spots previously identified.
        The data is saved in
            self.locs
        Args:
            parameters : dict
                necessary items:
                    box_size : as always
                    fit_parallel : bool
                        whether to fit on multiple cores
                optional items:
                    locs_vs_frame : dict
                        for plotting locs vs time
                        items correspond to arguments of _plot_locs_vs_frame
                    save_locs : dict
                        if saving localizations is requested.
                        Items correpsond to arguments of save_locs
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results
        """
        t00 = time.time()
        em = self.analysis_config['camera_info']["gain"] > 1
        spots = localize.get_spots(
            self.movie, self.identifications, parameters['box_size'],
            self.analysis_config['camera_info'])
        if self.analysis_config['gpufit_installed']:
            theta = gausslq.fit_spots_gpufit(spots)
            self.locs = gausslq.locs_from_fits_gpufit(
                self.identifications, theta, parameters['box_size'], em)
        else:
            if parameters['fit_parallel']:
                # theta = gausslq.fit_spots_parallel(spots, asynch=False)
                fs = gausslq.fit_spots_parallel(spots, asynch=True)
                n_tasks = len(fs)
                # N = len(identifications)
                # while lib.n_futures_done(fs) < n_tasks:
                #     print('progress made:', round(N * lib.n_futures_done(fs) / n_tasks), N)
                #     time.sleep(0.2)
                with tqdm(total=n_tasks, unit="task") as progress_bar:
                    for f in _futures.as_completed(fs):
                        progress_bar.update()
                theta = gausslq.fits_from_futures(fs)
                em = self.analysis_config['camera_info']["gain"] > 1
                self.locs = gausslq.locs_from_fits(
                    self.identifications, theta, parameters['box_size'], em
                )
            else:
                theta = np.empty((len(spots), 6), dtype=np.float32)
                theta.fill(np.nan)
                for i in tqdm(range(len(spots))):
                    theta[i] = gausslq.fit_spot(spots[i])

                self.locs = gausslq.locs_from_fits(
                    self.identifications, theta, parameters['box_size'], em
                )

        if (pars := parameters.get('locs_vs_frame')):
            if 'filename' in pars.keys():
                pars['filename'] = os.path.join(
                    module_result_dir, pars['filename'])
            results['locs_vs_frame'] = self._plot_locs_vs_frame(pars['filename'])

        # save locs
        if (pars := parameters.get('save_locs')):
            if 'filename' in pars.keys():
                pars['filename'] = os.path.join(
                    module_result_dir, pars['filename'])
            self.save_locs(pars['filename'])

        dt = np.round(time.time() - t00, 2)
        results['duration'] = dt
        results['locs_columns'] = self.locs.dtype.names
        return parameters, results

    def _plot_locs_vs_frame(self, filename):
        results = {}
        frames = np.arange(len(self.movie))
        bins = np.arange(len(self.movie) + 1) - .5

        df_locs = pd.DataFrame(self.locs)
        gbframe = df_locs.groupby('frame')
        photons_mean = gbframe['photons'].mean()
        photons_std = gbframe['photons'].std()
        sx_mean = gbframe['sx'].mean()
        sx_std = gbframe['sx'].std()
        sy_mean = gbframe['sy'].mean()
        sy_std = gbframe['sy'].std()

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(frames, photons_mean, color='b', label='mean photons')
        xhull = np.concatenate([frames, frames[::-1]])
        yhull = np.concatenate([photons_mean + photons_std, photons_mean[::-1] - photons_std[::-1]])
        ax[0].fill_between(xhull, yhull, color='b', alpha=.2, label='std photons')
        ax[0].set_xlabel('frame')
        ax[0].set_ylabel('photons')
        ax[0].legend()
        ax[1].plot(frames, sx_mean, color='c', label='mean sx')
        yhull = np.concatenate([sx_mean + sx_std, sx_mean[::-1] - sx_std[::-1]])
        ax[1].fill_between(xhull, yhull, color='c', alpha=.2, label='std sx')
        ax[1].plot(frames, sy_mean, color='m', label='mean sy')
        yhull = np.concatenate([sy_mean + sy_std, sy_mean[::-1] - sy_std[::-1]])
        ax[1].fill_between(xhull, yhull, color='m', alpha=.2, label='std sy')
        ax[1].set_xlabel('frame')
        ax[1].set_ylabel('width')
        ax[1].legend()
        results['filename'] = filename
        fig.savefig(results['filename'])
        plt.close(fig)
        return results

    @module_decorator
    def undrift_rcc(self, i, parameters, results):
        """Undrifts localized data using redundant cross correlation.
        """
        t00 = time.time()

        seg_init = parameters['segmentation']
        for i in range(parameters.get('max_iter_segmentations', 3)):
            # if the segmentation is too low, the process raises an error
            # adaptively increase the value.
            try:
                self.drift, self.locs = postprocess.undrift(
                    self.locs, self.info, segmentation=parameters['segmentation'],
                    display=False)
                results['success'] = True
                break
            except ValueError:
                parameters['segmentation'] = 2 * parameters['segmentation']
                logger.debug(f'RCC with segmentation {parameters['segmentation']} raised an error. Doubling.')
                results['message'] = f'Initial Segmentation of {seg_init} was too low.'
        else:  # did not work until the end
            logger.error(f'RCC failed up to segmentation {parameters['segmentation']}. Aborting.')
            max_segmentation = parameters['segmentation']
            # initial segmentation
            parameters['segmentation'] = int(
                parameters['segmentation'] / 2**parameters['max_iter_segmentations'])
            results['message'] = f'''
                    Undrifting did not work in {parameters['max_iter_segmentations']} iterations
                    up to a segmentation of {max_segmentation}.'''
            results['success'] = False

        parameters['dimensions'] = ['x', 'y']

        parameters['filename'] = os.path.join(
                    module_result_dir, parameters['filename'])
        np.savetxt(parameters['filename'], self.drift, delimiter=',')
        parameters['filename'] = os.path.splitext(parameters['filename'])[0] + '.png'
        self._plot_drift(parameters['drift_image'], parameters['dimensions'])

        # save locs
        if (pars := parameters.get('save_locs')):
            if 'filename' in pars.keys():
                pars['filename'] = os.path.join(
                    module_result_dir, pars['filename'])
            self.save_locs(pars['filename'])

        dt = np.round(time.time() - t00, 2)
        results['duration'] = dt

        return parameters, results

    def _plot_drift(self, filename, dimensions):
        fig, ax = plt.subplots()
        frames = np.arange(self.drift.shape[0])
        for i, dim in enumerate(dimensions):
            if isinstance(self.drift, np.recarray):
                ax.plot(frames, self.drift[dim], label=dim)
            else:
                ax.plot(frames, self.drift[:, i], label=dim)
        ax.set_xlabel('frame')
        ax.set_ylabel('drift [px]')
        ax.set_title('drift graph')
        ax.legend()
        fig.savefig(filename)
        plt.close(fig)

    @module_decorator
    def manual(self, i, parameters, results):
        """Handles a manual step: if the files required are not
        present, prompt the user to provide them. if they are, move
        to the next step.
        Args:
            i : int
                the index of the module 
            parameters: dict
                with required keys:
                    prompt : str
                        the user prompt
                    filename : str
                        the file the user should provide.
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        filepath = os.path.join(module_result_dir, parameters['filename'])
        if os.path.exists(filepath):
            results['filepath'] = filepath
            results['success'] = True
        else:
            msg = 'This is a manual step. Please provide input. '
            msg += parameters['prompt']
            msg += f' The resulting file should be {filepath}.'
            logger.debug(msg)
            print(msg)
            results['success'] = False
            raise ManualInputLackingError(f'{filepath} missing.')
        return parameters, results

    @module_decorator
    def describe(self, i, parameters, results):
        for meth, meth_pars in parameters['methods'].items():
            if meth.lower() == 'nena':
                res, best_vals = postprocess.nena(self.locs, self.info)
                results['nena'] = {'res': res, 'best_vals': best_vals}
            else:
                raise NotImplementedError(f'Description method {meth} not implemented.')
        return parameters, results

    def save_locs(self, filename):
        t00 = time.time()
        base_info = {
            "Frames": self.movie.shape[0],
            "Width": self.movie.shape[1],
            "Height": self.movie.shape[2],
        }
        info = {
            "Generated by": "AutoPicasso: Localize",
            "Pixelsize": self.analysis_config['camera_info']['pixelsize'],
        }
        info = [base_info] + [info]

        io.save_locs(filename, self.locs, self.info)
        # # when the paths get long, the hdf5 library throws an error, so chdir
        # # but apparently, the issue is the length of the filename itself
        # previous_dir = os.getcwd()
        # parent_dir, fn = os.path.split(filename)
        # os.chdir(parent_dir)
        # io.save_locs(fn, self.locs, self.info)
        # os.chdir(previous_dir)

        dt = np.round(time.time() - t00, 2)
        results_save = {'duration': dt}
        return results_save


class ManualInputLackingError(Exception):
    pass


def module_decorator(method):
    def module_wrapper(self, *args, **kwargs):
        # create the results direcotry
        method_name = get_caller_name(2)
        module_result_dir = os.path.join(self.results_folder, f'{i:02d}_' + method_name)
        try:
            os.mkdir(module_result_dir)
        except:
            pass

        results = {
            'folder': module_result_dir,
        }
        kwargs['results'] = results
        parameters, results = method(self, *args, **kwargs)
        return parameters, results
    return module_wrapper


def get_ap():
    fn = 'R1to6_3C-20nm-10nm_231208-1547\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_1\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_NDTiffStack_1'
    id_params = {
        'min_grad': 5000,  # minimum gradient
        'box_size': 7,  # box size in pixels
    }

    # camera info
    camera_info = {
        'gain': 1,
        'sensitivity': 0.45,
        'baseline': 100,
        'qe': 0.82,
        'pixelsize': 130,  # nm
    }
    #____________________________________________#
    gpufit_installed = False
    fit_parallel = True
    return AutoPicasso(fn, id_params, camera_info, gpufit_installed, fit_parallel)
