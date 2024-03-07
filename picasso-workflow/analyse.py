"""
analyse.py

This is the picasso interface of picasso-workflow
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

from picasso_workflow.util import AbstractPipeline


logger = logging.getLogger(__name__)


class AutoPicasso(AbstractPipeline):
    """A class to automatically evaluate datasets
    """

    def __init__(self, filename, id_params, camera_info, gpufit_installed, fit_parallel):
        self.filename = filename
        self.id_params = id_params
        self.camera_info = camera_info
        self.gpufit_installed = gpufit_installed
        self.fit_parallel = fit_parallel
        pass

    def load(self, load_pars):
        results = {}
        results['picasso version'] = picassoversion
        t00 = time.time()
        self.movie, self.info = io.load_movie(self.filename)
        results['movie.shape'] = self.movie.shape

        # create sample movie
        if (samplemov_pars := load_pars.get('sample_movie')) is not None:
            res = self.create_sample_movie(**samplemov_pars)
            results['sample_movie'] = res

        dt = np.round(time.time() - t00, 2)
        results['duration'] = dt
        self.results_load = results
        return results

    def create_sample_movie(self, filename, n_sample=30, max_quantile=.9998, fps=1):
        """Create a subsampled movie of the movie loaded
        """
        results = {}
        # frame_numbers = np.random.choice(np.arange(len(self.movie)), n_sample)
        if len(self.movie) < n_sample:
            n_sample = len(self.movie)

        dn = int(len(self.movie) / (n_sample - 1))
        frame_numbers = np.arange(0, len(self.movie), dn)
        results['sample_frame_idx'] = frame_numbers

        subsampled_frames = np.array([self.movie[i] for i in frame_numbers])
        fn_movie = filename
        save_movie(
            fn_movie, subsampled_frames,
            max_quantile=max_quantile, fps=fps)
        results['filename'] = fn_movie
        return results

    def get_netgrad_hist(self, filename, frame_numbers, start_ng=-3000, zscore=5, bins=None):
        """Get histograms of the net gradient at local maxima of n randomly
        chosen frames
        Call after loading
        Args:
            frame_numbers : list
                the frame indexess to analyze
            start_ng : float
                the minimum net gradient to accept for the histogram.
                this should be below zero, to capture all net gradient
                values that exist in the data
            zscore : float
                the number of sigmas above the bakground net gradient peak
                to set as the estimated min net gradient threshold
        Returns:
            fn : string
                filename of the generated plot
            ng_est : float
                the estimated min net gradient
        """
        results = {}
        identifications = []

        for frame_number in frame_numbers:
            identifications.append(
                localize.identify_by_frame_number(
                    self.movie, start_ng,
                    self.id_params['box_size'], frame_number))
        # id_list = identifications
        identifications = np.hstack(identifications).view(np.recarray)
        identifications.sort(kind="mergesort", order="frame")

        if bins is None:
            hi = np.quantile(identifications['net_gradient'], .9995)
            bins = np.linspace(start_ng, hi, num=200)
        hist, edges = np.histogram(
            identifications['net_gradient'], bins=bins, density=True)

        # # find the background peak
        # peaks, properties = find_peaks(hist, height=height, width=width)
        # print(peaks)
        # print(properties)
        # ng_est = 2 * int(edges[int(properties['right_ips'][0])])

        # alternatively, assume gaussian, find max and FWHM
        bkg_peak_height, bkg_peak_pos = np.max(hist), np.argmax(hist)
        bkg_halfclose = np.argsort(np.abs(hist - bkg_peak_height / 2))
        bkg_fwhm = np.abs(bkg_halfclose[1] - bkg_halfclose[0])
        bkg_sigma = bkg_fwhm / np.sqrt(4 * np.log(2))
        ng_est_idx = int(zscore * bkg_sigma) + bkg_peak_pos  # cut off at zscore * bkg_sigma
        if ng_est_idx >= len(edges):
            ng_est_idx = len(edges) - 1
        results['estd_net_grad'] = edges[ng_est_idx]

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
        results['filename'] = filename
        fig.savefig(results['filename'])
        return results

    def identify(self, pars_identify):
        t00 = time.time()
        results = {}

        # auto-detect net grad if required:
        if (autograd_pars := pars_identify.get('auto_netgrad')) is not None:
            res = self.get_netgrad_hist(**autograd_pars)
            results['auto_netgrad'] = res
            pars_identify['min_grad'] = res['estd_net_grad']

        curr, futures = localize.identify_async(
            self.movie,
            pars_identify['min_grad'],
            pars_identify['box_size'],
            roi=None,
        )
        self.identifications = localize.identifications_from_futures(futures)
        dt = np.round(time.time() - t00, 2)
        results['duration'] = dt
        results['num_identifications'] = len(self.identifications)

        if (pars := pars_identify.get('ids_vs_frame')) is not None:
            results['ids_vs_frame'] = self.plot_ids_vs_frame(**pars)
        return pars_identify, results

    def plot_ids_vs_frame(self, filename):
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

    def localize(self, pars_localize):
        results = {}
        t00 = time.time()
        em = self.camera_info["gain"] > 1
        spots = localize.get_spots(
            self.movie, self.identifications, pars_localize['box_size'],
            self.camera_info)
        if self.gpufit_installed:
            theta = gausslq.fit_spots_gpufit(spots)
            self.locs = gausslq.locs_from_fits_gpufit(
                self.identifications, theta, pars_localize['box_size'], em)
        else:
            if self.fit_parallel:
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
                em = self.camera_info["gain"] > 1
                self.locs = gausslq.locs_from_fits(
                    self.identifications, theta, pars_localize['box_size'], em
                )
            else:
                theta = np.empty((len(spots), 6), dtype=np.float32)
                theta.fill(np.nan)
                for i in tqdm(range(len(spots))):
                    theta[i] = gausslq.fit_spot(spots[i])

                self.locs = gausslq.locs_from_fits(
                    self.identifications, theta, pars_localize['box_size'], em
                )

        if (pars := pars_localize.get('locs_vs_frame')):
            results['locs_vs_frame'] = self.plot_locs_vs_frame(pars['filename'])

        # save locs
        if (pars := pars_localize.get('save_locs')):
            self.save_locs(pars['filename'])

        dt = np.round(time.time() - t00, 2)
        results['duration'] = dt
        results['locs_columns'] = self.locs.dtype.names
        return pars_localize, results

    def plot_locs_vs_frame(self, filename):
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

    def undrift_mutualnearestneighbors(self, pars_undrift):
        results = {}
        t00 = time.time()

        self.locs, self.drift = undrift(
            self.locs, dimensions=pars_undrift['dimensions'],
            method=pars_undrift['method'], max_dist=pars_undrift['max_dist'],
            use_multiprocessing=pars_undrift['use_multiprocessing'])

        np.savetxt(pars_undrift['drift_file'], self.drift, delimiter=',')
        pars_undrift['drift_image'] = os.path.splitext(pars_undrift['drift_file'])[0] + '.png'
        self.plot_drift(pars_undrift['drift_image'], pars_undrift['dimensions'])

        # save locs
        if (pars := pars_undrift.get('save_locs')):
            self.save_locs(pars['filename'])

        dt = np.round(time.time() - t00, 2)
        results['duration'] = dt

        return pars_undrift, results

    def undrift_rcc(self, pars_undrift):
        results = {}
        t00 = time.time()

        seg_init = pars_undrift['segmentation']
        for i in range(pars_undrift.get('max_iter_segmentations', 3)):
            # if the segmentation is too low, the process raises an error
            # adaptively increase the value.
            try:
                self.drift, self.locs = postprocess.undrift(
                    self.locs, self.info, segmentation=pars_undrift['segmentation'],
                    display=False)
                break
            except ValueError:
                pars_undrift['segmentation'] = 2 * pars_undrift['segmentation']
                results['message'] = f'Initial Segmentation of {seg_init} was too low.'
        else:  # did not work until the end
            raise UndriftError()

        pars_undrift['dimensions'] = ['x', 'y']

        np.savetxt(pars_undrift['drift_file'], self.drift, delimiter=',')
        pars_undrift['drift_image'] = os.path.splitext(pars_undrift['drift_file'])[0] + '.png'
        self.plot_drift(pars_undrift['drift_image'], pars_undrift['dimensions'])

        # save locs
        if (pars := pars_undrift.get('save_locs')):
            self.save_locs(pars['filename'])

        dt = np.round(time.time() - t00, 2)
        results['duration'] = dt

        return pars_undrift, results

    def plot_drift(self, filename, dimensions):
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

    def describe(self, pars_describe):
        results = {}
        for meth, meth_pars in pars_describe['methods'].items():
            if meth.lower() == 'nena':
                res, best_vals = postprocess.nena(self.locs, self.info)
                results['nena'] = {'res': res, 'best_vals': best_vals}
            else:
                raise NotImplementedError('Description method ' + meth + ' not implemented.')

        return pars_describe, results

    def save_locs(self, filename):
        t00 = time.time()
        base_info = {
            "Frames": self.movie.shape[0],
            "Width": self.movie.shape[1],
            "Height": self.movie.shape[2],
        }
        info = {
            "Generated by": "AutoPicasso: Localize",
            "Pixelsize": self.camera_info['pixelsize'],
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
        self.results_save = {'duration': dt}


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
