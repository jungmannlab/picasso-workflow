import sys
#sys.path.insert(1, 'Z:/users/grabmayr/picasso_heerpa/picasso')

from picasso import io, localize, gausslq, postprocess
from picasso import __version__ as picassoversion
import os
import re
import logging
import requests
import time
import yaml
from tqdm import tqdm
from concurrent import futures as _futures
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial


logger = logging.getLogger(__name__)


# Function to adjust contrast
def adjust_contrast(img, min_quantile, max_quantile):
    min_val = np.quantile(img, min_quantile)
    max_val = np.quantile(img, max_quantile)
    img = img.astype(np.float32) - min_val
    img = img * 255 / (max_val - min_val)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return np.rollaxis(np.array([img, img, img], dtype=np.uint8), 0, 3)


def save_movie(fname, movie, max_quantile=1, fps=1):
    # Assuming 'array_3d' is your 3D numpy array and 'contrast_factors' is a list of contrast factors for each frame
    adjusted_images = [adjust_contrast(frame, 0, max_quantile)[..., np.newaxis] for frame in movie]

    # Create movie file
    clip = ImageSequenceClip(adjusted_images, fps=fps)
    clip.write_videofile(fname, verbose=False)#, codec='mpeg4')


class ReportingAnalyzer:
    # identification parameters
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
    prefix = datetime.now().strftime('%y%m%d-%H%M_')
    #____________________________________________#
    gpufit_installed = False
    fit_parallel = True

    def __init__(self, filename, base_url, space_key, parent_page_title,
                 report_name, parameters=None):
        # ensure correct directory separator
        path_components = re.split(r'[\\/]', filename)
        filename = os.path.join(*path_components)

        report_name = report_name + '_' + self.prefix[:-1]
        # create analysis directory
        self.savedir = os.path.split(filename)[0]
        self.savedir = os.path.join(self.savedir, report_name)
        os.mkdir(self.savedir)
        self.filename = filename
        self.report_name = report_name

        self.autopicasso = AutoPicasso(
            filename, self.id_params, self.camera_info,
            self.gpufit_installed, self.fit_parallel)
        self.confluencereporter = ConfluenceReporter(
            base_url, space_key, parent_page_title, report_name)

    #     # to be implemented at a later stage
    #     if not parameters:
    #         self.parameters = self.get_default_parameters()

    # def get_default_parameters(self):
    #     self.parameters = {
    #         'load': {
    #             'filename': self.autopicasso.filename,
    #             'save_directory': self.savedir,
    #             'sample_movie': {
    #                 'filename': self.get_prefixed_filename(
    #                     'selected_frames.mp4'),
    #                 'n_sample': 40,
    #                 'max_quantile': .9998,
    #                 'fps': 2,
    #             }
    #         },
    #         'identify': {
    #             'auto_netgrad': {
    #                 'filename': self.get_prefixed_filename('ng_histogram.png'),
    #                 'frame_numbers': self.results_load[
    #                     'sample_movie']['sample_frame_idx'],
    #                 'start_ng': -3000,
    #                 'zscore': 5,
    #             },
    #             'ids_vs_frame': {
    #                 'filename': self.get_prefixed_filename('ids_vs_frame.png')
    #             },
    #             'box_size': self.autopicasso.id_params['box_size']
    #         },
    #         'localize': {
    #             'box_size': self.autopicasso.id_params['box_size'],
    #             'locs_vs_frame': {
    #                 'filename': self.get_prefixed_filename('locs_vs_frame.png')
    #             },
    #             'save_locs': {
    #                 'filename': self.get_prefixed_filename('locs.hdf5')
    #             }
    #         },
    #         'undrift' : {
    #             'segmentation': 1000, 
    #             'drift_file': self.get_prefixed_filename('drift.csv'),
    #             'save_locs': {'filename': self.get_prefixed_filename('locs_undrift.hdf5')}
    #         },
    #         'describe': {
    #             'methods': {
    #                 'nena': {},
    #             }
    #         }
    #     }

    def get_prefixed_filename(self, name):
        return os.path.join(self.savedir, self.prefix + name)

    def load(self, pars_load=None):
        if not pars_load:
            pars_load = {
            'filename': self.autopicasso.filename,
            'save_directory': self.savedir,
                'sample_movie' : {
                    'filename': self.get_prefixed_filename('selected_frames.mp4'),
                    'n_sample': 40,
                    'max_quantile': .9998,
                    'fps': 2,
                }
            }
        self.results_load = self.autopicasso.load(pars_load)
        self.confluencereporter.load(pars_load, self.results_load)
        print('Loaded image data.')

    def identify(self, pars_identify=None):
        if not pars_identify:
            pars_identify = {
                'auto_netgrad': {
                    'filename': os.path.join(self.savedir, self.prefix + 'ng_histogram.png'),
                    'frame_numbers': self.results_load['sample_movie']['sample_frame_idx'],
                    'start_ng': -3000,
                    'zscore': 5,
                },
                'ids_vs_frame': {
                    'filename': os.path.join(self.savedir, self.prefix + 'ids_vs_frame.png')
                },
                'box_size': self.autopicasso.id_params['box_size']
            }
        pars_identify, self.results_identify = self.autopicasso.identify(pars_identify)
        self.confluencereporter.identify(pars_identify, self.results_identify)
        print('Identified spots.')

    def localize(self, pars_localize=None):
        if not pars_localize:
            fn = os.path.split(self.filename)[1]
            base, ext = os.path.splitext(fn)
            save_locs_path = os.path.join(self.savedir, base + '_locs.hdf5')
            save_locs_path = os.path.join(self.savedir, self.prefix + 'locs.hdf5')
            # print('this location does not work: ')
            # print(save_locs_path)
            # save_locs_path = os.path.join('R1to6_3C-20nm-10nm_231208-1547', 'R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_1', base + '_locs.hdf5')
            # save_locs_path = os.path.join('R1to6_3C-20nm-10nm_231208-1547', 'R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_1', self.report_name, 'locs.hdf5')
            # print('saving to')
            # print(save_locs_path)

            pars_localize = {
                'box_size': self.autopicasso.id_params['box_size'],
                'locs_vs_frame': {
                    'filename': os.path.join(self.savedir, self.prefix + 'locs_vs_frame.png')
                },
                'save_locs': {
                    'filename': save_locs_path
                }
            }
        pars_localize, res_localize = self.autopicasso.localize(pars_localize)
        self.confluencereporter.localize(pars_localize, res_localize)
        print('Localized spots')

    def undrift(self, pars_undrift=None):
        # using RCC
        if not pars_undrift:
            pars_undrift = {
                'segmentation': 1000,
                'max_iter_segmentations': 4,
                'drift_file': os.path.join(self.savedir, self.prefix + 'drift.csv'),
                'save_locs': {'filename': os.path.join(self.savedir, self.prefix + 'locs_undrift.hdf5')}
            }
        try:
            pars_undrift, res_undrift = self.autopicasso.undrift_rcc(pars_undrift)
            self.confluencereporter.undrift_rcc(pars_undrift, res_undrift)
            print('undrifted dataset')
        except UndriftError:
            max_segmentation = pars_undrift['segmentation']
            # initial segmentation
            pars_undrift['segmentation'] = int(
                pars_undrift['segmentation'] / 2**pars_undrift['max_iter_segmentations'])
            res_undrift = {
                'message': f'''
                    Undrifting did not work in {pars_undrift['max_iter_segmentations']} iterations
                    up to a segmentation of {max_segmentation}.''',
            }
            self.confluencereporter.undrift_rcc(pars_undrift, res_undrift)
            print('Error in dataset undrifting')
        # # using mutual nearest neighbors (first test not very promising)
        # pars_undrift = {
        #     'dimensions': ['x', 'y'],
        #     'max_dist': 0.2,  # pixels
        #     'method': 'cdist',  # one of 'cdist' and 'kdtree'
        #     'use_multiprocessing': False,
        #     'drift_file': os.path.join(self.savedir, self.prefix + 'drift.csv'),
        #     'save_locs': {'filename': os.path.join(self.savedir, self.prefix + 'locs_undrift.hdf5')}
        # }
        # pars_undrift, res_undrift = self.autopicasso.undrift_mutualnearestneighbors(pars_undrift)
        # self.confluencereporter.undrift_mutualnearestneighbors(pars_undrift, res_undrift)
        # print('undrifted dataset')

    def describe(self, pars_describe=None):
        if not pars_describe:
            pars_describe = {
                'methods': {
                    'nena': {},
                }
            }
        pars_describe, res_describe = self.autopicasso.describe(pars_describe)
        self.confluencereporter.describe(pars_describe, res_describe)

    def save(self):
        self.autopicasso.save_locs()
        self.confluencereporter.save()


def get_ra():
    # fn = 'R1to6_3C-20nm-10nm_231208-1547\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_1\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_NDTiffStack_1.tif'
    fn = 'R1to6_3C-20nm-10nm_231208-1547\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_1\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_NDTiffStack.tif'
    base_url = 'https://mibwiki.biochem.mpg.de'
    space_key = "~hgrabmayr"
    parent_page_title = 'test page'
    report_name = 'analysis_report'
    return ReportingAnalyzer(fn, base_url, space_key, parent_page_title, report_name)


class AutoPicasso:
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


def undrift(
        data, dimensions=['x', 'y'], method='cdist', max_dist=.2,
        use_multiprocessing=True):
    # Assuming 'data' is your numpy recarray
    data.sort(order='frame')  # sort data by frame

    nframes = int(data['frame'].max())
    allframes = np.unique(data['frame'])

    shifts = np.zeros((nframes, len(dimensions)))

    if method == 'kdtree':
        get_frameshift = get_frameshift_kdtree
    elif method == 'cdist':
        get_frameshift = get_frameshift_cdist
    else:
        raise NotImplementedError(f'Method {method} not implemented for undrifting.')

    if not use_multiprocessing:
        for i in range(1, nframes):
            shifts[i, :] = get_frameshift(
                i, data, dimensions, max_dist)
    else:
        n_workers = min(
            60, max(1, int(0.75 * cpu_count()))
        ) # Python crashes when using >64 cores
        partial_fun = partial(get_frameshift, spots=data, dimensions=dimensions, max_dist=max_dist)
        with Pool(n_workers) as p:
            shifts = p.map(partial_fun, range(1, nframes + 1))
        shifts = np.array(shifts)

    drift = np.cumsum(shifts, axis=0)
    for i, dim in enumerate(dimensions):
        for frame in range(nframes):
            data[dim][data['frame'] == frame] -= drift[frame, i]

    return data, drift


def get_frameshift_kdtree(
        frame, spots, dimensions=['x', 'y'], max_dist=.2):
    """
    """
    from scipy.spatial import cKDTree, distance

    spots_current = spots[frame]
    spots_next = spots[frame + 1]

    tree_current = cKDTree(spots_current[['x', 'y']])
    tree_next = cKDTree(spots_next[['x', 'y']])

    dists_fwd, indices_fwd = tree_current.query(spots_next[['x', 'y']])
    dists_bwd, indices_bwd = tree_next.query(spots_current[['x', 'y']])

    nn_tuples_fwd = {
        (i, nxt) for i, nxt in enumerate(indices_fwd)
        if dists_fwd[i] <= max_dist}
    # no need to make distance comparison backwards because of intersection
    nn_tuples_bwd = {(prv, i) for i, prv in enumerate(indices_bwd)}

    # intersect the sets to get only the mutual neighbors
    mutual_nn = nn_tuples_fwd & nn_tuples_bwd
    mnn_current = [mnn[0] for mnn in mutual_nn]
    mnn_next = [mnn[1] for mnn in mutual_nn]

    shifts = np.array((len(mutual_nn), len(dimensions)))
    for i, d in enumerate(dimensions):
        shifts[:, i] = spots_next[mnn_next, d] - spots_current[mnn_current, d]
    # mean_shift = shifts.mean(axis=0)
    mean_shift = shifts.median(axis=0)

    return mean_shift


def get_frameshift_cdist(
        frame, spots, dimensions=['x', 'y'], max_dist=.2):
    from scipy.spatial.distance import cdist

    spots_previous = spots[spots['frame'] == frame - 1]
    spots_current = spots[spots['frame'] == frame]

    # Compute pairwise distances between spots in frame1 and frame2
    distances = cdist(
        rec2nparray(spots_previous, dimensions),
        rec2nparray(spots_current, dimensions))

    # Find the nearest neighbor in frame2 for each spot in frame1
    nn_fwd = np.argmin(distances, axis=1)
    nn_tuples_fwd = {
        (i, curr) for i, curr in enumerate(nn_fwd)
        if distances[i, curr] <= max_dist}

    # Find the nearest neighbor in frame1 for each spot in frame2
    nn_bwd = np.argmin(distances, axis=0)
    nn_tuples_bwd = {(prv, i) for i, prv in enumerate(nn_bwd)}

    # intersect the sets to get only the mutual neighbors
    mutual_nn = nn_tuples_fwd & nn_tuples_bwd
    mnn_previous = [mnn[0] for mnn in mutual_nn]
    mnn_current = [mnn[1] for mnn in mutual_nn]

    shifts = np.zeros((len(mutual_nn), len(dimensions)))
    for i, d in enumerate(dimensions):
        shifts[:, i] = spots_current[d][mnn_current] - spots_previous[d][mnn_previous]
    if len(mutual_nn) > 1:
        mean_shift = shifts.mean(axis=0)
    else:
        mean_shift = np.zeros(len(dimensions))

    return mean_shift


def rec2nparray(rec, cols, dtype=np.float32):
    return np.stack([rec[col].view(dtype) for col in cols]).T


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


class ConfluenceReporter:
    """A class to upload reports of automated picasso evaluations
    to confluence
    """

    def __init__(self, base_url, space_key, parent_page_title, report_name):
        self.ci = ConfluenceInterface(base_url, space_key, parent_page_title)

        # create page
        self.report_page_name = report_name
        for i in range(1, 30):
            try:
                self.report_page_id = self.ci.create_page(self.report_page_name, body_text='')
                print(f'Created page {self.report_page_name}')
                break
            except KeyError:
                self.report_page_name = report_name + '_{:02d}'.format(i)

    def load(self, pars_load, results_load):
        """Describes the loading
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
        """
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Load</strong></p>
        <ul>
        <li>Picasso Version: {results_load['picasso version']}</li>
        <li>Movie Location: {pars_load['filename']}</li>
        <li>Analysis Location: {pars_load['save_directory']}</li>
        <li>Movie Size: Frames: {results_load['movie.shape'][0]}, Width: {results_load['movie.shape'][1]}, Height: {results_load['movie.shape'][2]}</li>
        <li>Duration: {results_load['duration']} s</li>
        </ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(self.report_page_name, self.report_page_id, text)
        if (sample_mov_res := results_load.get('sample_movie')) is not None:
            text = f"""
            <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
            <p>Subsampled Frames</p>
            <ul>
            <li> {len(sample_mov_res['sample_frame_idx'])} frames:
             {str(sample_mov_res['sample_frame_idx'])}</li>
            </ul>
            </ac:layout-cell></ac:layout-section></ac:layout>
            """
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text)
            # print('uploading graph')
            self.ci.upload_attachment(
                self.report_page_id, sample_mov_res['filename'])
            self.ci.update_page_content_with_movie_attachment(
                self.report_page_name, self.report_page_id,
                os.path.split(sample_mov_res['filename'])[1])
        return  # document nothing

    def identify(self, pars_identify, results_identify):
        """Describes the identify step
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
            fn_movie : str
                the filename to the movie generated
            fn_hist : str
                the filename to the histogram plot generated
        """
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Identify</strong></p>
        <ul>
        <li>Min Net Gradient: {pars_identify['min_grad']:,.0f}</li>
        <li>Box Size: {pars_identify['box_size']} px</li>
        <li>Duration: {results_identify['duration']} s</li>
        <li>Identifications found: {results_identify['num_identifications']:,}</li>
        </ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(self.report_page_name, self.report_page_id, text)
        if (res_autonetgrad := results_identify.get('auto_netgrad')) is not None:
            # print('uploading graph')
            self.ci.upload_attachment(
                self.report_page_id, res_autonetgrad['filename'])
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name, self.report_page_id,
                os.path.split(res_autonetgrad['filename'])[1])
        if (res := results_identify.get('ids_vs_frame')) is not None:
            # print('uploading graph')
            self.ci.upload_attachment(
                self.report_page_id, res['filename'])
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name, self.report_page_id,
                os.path.split(res['filename'])[1])

    def localize(self, pars_localize, results_localize):
        """Describes the Localize section of picasso
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
        """
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Localize</strong></p>
        <ul><li>Duration: {results_localize['duration']}</li>
        <li>Locs Column names: {results_localize['locs_columns']}</li></ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        # text = "<p><strong>Localize</strong></p>"
        self.ci.update_page_content(self.report_page_name, self.report_page_id, text)

        if (res := results_localize.get('locs_vs_frame')) is not None:
            # print('uploading graph')
            self.ci.upload_attachment(
                self.report_page_id, res['filename'])
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name, self.report_page_id,
                os.path.split(res['filename'])[1])

    def undrift_mutualnearestneighbors(self, pars_undrift, res_undrift):
        """Describes the Localize section of picasso
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
        """
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Undrifting</strong></p>
        <ul><li>Dimensions: {pars_undrift['dimensions']}</li>
        <li>Method: {pars_undrift['method']}</li>
        <li>max_dist: {pars_undrift['max_dist']} pixels</li>
        <li>Multiprocessing used: {pars_undrift['use_multiprocessing']}</li>
        <li>Duration: {res_undrift['duration']} s</li></ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(self.report_page_name, self.report_page_id, text)

        self.ci.upload_attachment(
            self.report_page_id, pars_undrift['drift_image'])
        self.ci.update_page_content_with_image_attachment(
            self.report_page_name, self.report_page_id,
            os.path.split(pars_undrift['drift_image'])[1])

    def undrift_rcc(self, pars_undrift, res_undrift):
        """Describes the Localize section of picasso
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
        """
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Undrifting via RCC</strong></p>
        <ul><li>Dimensions: {pars_undrift.get('dimensions')}</li>
        <li>Segmentation: {pars_undrift.get('segmentation')}</li>
        """
        if (msg := res_undrift.get('message')):
            text += f"""<li>Note: {msg}</li>"""
        text += f"""
        <li>Duration: {res_undrift.get('duration')} s</li></ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text)

        if (driftimg_fn := pars_undrift.get('drift_image')):
            self.ci.upload_attachment(
                self.report_page_id, driftimg_fn)
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name, self.report_page_id,
                os.path.split(driftimg_fn)[1])

    def describe(self, pars_describe, res_describe):
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Descriptive Statistics</strong></p>"""
        for meth, meth_pars in pars_describe['methods'].items():
            if meth.lower() == 'nena':
                meth_res = res_describe['nena']
                text += f"""
                    <p>NeNa</p>
                    <ul><li>Segmentation: {meth_pars['segmentation']}</li>
                    <li>Best Values: {str(meth_res['best_vals'])}</li>
                    <li>Result: {str(meth_res['res'])}</li>
                    </ul>"""
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(self.report_page_name, self.report_page_id, text)


class UndriftError(Exception):
    pass


def get_cfrep():
    base_url = 'https://mibwiki.biochem.mpg.de'
    space_key = "~hgrabmayr"
    parent_page_title = 'test page'
    report_name = 'my report'
    cr = ConfluenceReporter(base_url, space_key, parent_page_title, report_name)
    return cr


class ConfluenceInterface():
    """A Interface class to access Confluence

    For access to the Confluence API, create an API token in confluence,
    and store it as an environment variable:
    $ setx CONFLUENCE_BEARER "your_confluence_api_token"
    """

    def __init__(self, base_url, space_key, parent_page_title):
        self.bearer_token = self.get_bearer_token()
        self.base_url = base_url
        self.space_key = space_key
        self.parent_page_id, _ = self.get_page_properties(parent_page_title)

    def get_bearer_token(self):
        '''Set this by setting the environment variable in the windows command
        line on the server:
        $ setx CONFLUENCE_BEARER <your_confluence_api_token>
        The confluence api token can be generated and copied in the personal
        details of confluence.
        '''
        return os.environ.get('CONFLUENCE_BEARER')

    def get_page_properties(self, page_title='', page_id=''):
        """
        Returns:
            id : str
                the page id
            title : str
                the page title
        """
        if page_title != '':
            url = self.base_url + "/rest/api/content"
            params = {
                "spaceKey": self.space_key,
                "title": page_title
            }
        elif page_id != '':
            url = self.base_url + f"/rest/api/content/{page_id}"
            params = {
                "spaceKey": self.space_key,
            }
        else:
            logger.error('One of page_title and page_id must be given.')
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            logger.warn('Failed to get page content.')
        results = response.json()['results'][0]
        return results['id'], results['title']

    def get_page_version(self, page_title='', page_id=''):
        """
        Returns:
            data : dict
                results
                    id, title, version
        """
        if page_title != '':
            url = self.base_url + "/rest/api/content"
            params = {
                "spaceKey": self.space_key,
                "title": page_title,
            }
        elif page_id != '':
            url = self.base_url + f"/rest/api/content/{page_id}"
            params = {}
        else:
            logger.error('One of page_title and page_id must be given.')
        params['expand'] = ['version']
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            logger.warn('Failed to get page content.')
        return response.json()['results'][0]['version']['number']

    def get_page_body(self, page_title='', page_id=''):
        """
        Returns:
            data : dict
                results
                    id, title, version
        """
        if page_title != '':
            url = self.base_url + "/rest/api/content"
            params = {
                "spaceKey": self.space_key,
                "title": page_title,
            }
        elif page_id != '':
            url = self.base_url + f"/rest/api/content/{page_id}"
            params = {}
        else:
            logger.error('One of page_title and page_id must be given.')
        params['expand'] = ['body.storage']
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            logger.warn('Failed to get page content.')
        return response.json()['results'][0]['body']['storage']['value']

    # def get_page_properties(self, page_id):
    #     url = self.base_url + f"/wiki/api/v2/pages/{page_id}/properties"
    #     headers = {
    #         "Authorization": "Bearer {:s}".format(self.bearer_token),
    #         "Accept": "application/json"
    #     }
    #     response = requests.get(url, headers=headers)
    #     return response
    #     if response.status_code == 200:
    #         data = response.json()
    #         if data.get("results"):
    #             page_id = data["results"][0]["id"]
    #             logger.debug(f"Page ID: {page_id}")
    #             return page_id
    #         else:
    #             logger.warn("Page not found.")
    #     else:
    #         logger.warn("Failed to retrieve page ID.")

    # def get_spaces(self):
    #     url = self.base_url + f"/wiki/api/v2/spaces"
    #     headers = {
    #         "Authorization": "Bearer {:s}".format(self.bearer_token),
    #         "Accept": "application/json"
    #     }
    #     # auth = requests.auth.HTTPBasicAuth("Heinrich Grabmayr", self.bearer_token)
    #     # response = requests.get(url, headers=headers, auth=auth)
    #     response = requests.get(url, headers=headers)
    #     return response
    #     if response.status_code != 200:
    #         logger.warn("Could not retrieve spaces.")
    #     data = response.json()
    #     return data

    def create_page(self, page_title, body_text, parent_id='rootparent'):
        """
        Args:
            page_title : str
                the title of the page to be created
            body_text : str
                the content of the page, with the confuence markdown / html
            parent_id : str
                the id of the parent page. If 'rootparent', the parent_page_id
                of this ConfluenceInterface is used
        Returns:
            page_id : str
                the id of the newly created page
        """
        if parent_id == 'rootparent':
            parent_id = self.parent_page_id
        url = self.base_url + "/rest/api/content"
        headers = {
            "Authorization": "Bearer {:s}".format(self.bearer_token),
            "Content-Type": "application/json"
        }
        data = {
            "type": "page",
            "title": page_title,
            "space": {"key": self.space_key},
            "ancestors": [{"id": parent_id}],
            "body": {
                "storage": {
                    "value": body_text,
                    "representation": "storage"
                }
            }
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            logger.warning(f"Failed to create page {page_title}.")
            raise KeyError()

        return response.json()["id"]

    def upload_attachment(self, page_id, filename):
        """Uploads an attachment to a page
        Args:
            page_id : str
                the page id the attachment should be saved to.
            filename : str
                the local filename of the file to attach
        Returns:
            attachment_id : str
                the id of the attachment
        """
        url = self.base_url + f"/rest/api/content/{page_id}/child/attachment"
        headers = {
            "Authorization": "Bearer {:s}".format(self.bearer_token),
            'X-Atlassian-Token': 'nocheck'
        }
        files = {
            "file": open(filename, "rb")
        }
        response = requests.post(url, headers=headers, files=files)
        if response.status_code != 200:
            logger.warning("Failed to upload attachment.")
            return

        attachment_id = response.json()["results"][0]["id"]
        return attachment_id

    def update_page_content(self, page_name, page_id, body_update):
        prev_version = self.get_page_version(page_name)
        prev_body = self.get_page_body(page_name)
        _, prev_title = self.get_page_properties(page_name)

        url = self.base_url + f"/rest/api/content/{page_id}"
        headers = {
            "Authorization": "Bearer {:s}".format(self.bearer_token),
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        data = {
            "version": {
                "number": prev_version + 1,
                "message": "version update"
            },
            "type": "page",
            "title": prev_title,
            "body": {
                "storage": {
                    "value": prev_body + body_update,
                    "representation": "storage"
                }
            }
        }
        response = requests.put(url, headers=headers, json=data)
        if response.status_code != 200:
            logger.warning("Failed to update page content.")


    def update_page_content_with_movie_attachment(self, page_name, page_id, filename):
        # body_update = f"""<ac:structured-macro ac:name="multimedia" ac:schema-version="1" ac:macro-id="12345678-90ab-cdef-1234-567890abcdef">
        body_update = f"""<ac:structured-macro ac:name="multimedia" ac:schema-version="1">
              <ac:parameter ac:name="autoplay">false</ac:parameter>
              <ac:parameter ac:name="name"><ri:attachment ri:filename=\"{filename}\" /></ac:parameter>
              <ac:parameter ac:name="loop">false</ac:parameter>
              <ac:parameter ac:name="width">30%</ac:parameter>
              <ac:parameter ac:name="height">30%</ac:parameter>
            </ac:structured-macro>
            """
        self.update_page_content(page_name, page_id, body_update)


    def update_page_content_with_image_attachment(self, page_name, page_id, filename):
        body_update = f"<ac:image><ri:attachment ri:filename=\"{filename}\" /></ac:image>"
        self.update_page_content(page_name, page_id, body_update)


def get_cfd():
    base_url = 'https://mibwiki.biochem.mpg.de'
    space_key = "~hgrabmayr"
    parent_page_title = 'test page'
    return ConfluenceInterface(base_url, space_key, parent_page_title)


def find_data_dirs(parent_dir):
    data_dirs = {}
    for dirpath, dirnames, filenames in os.walk(parent_dir):
        if 'NDTiff.index' in filenames:
            start_file = [fn for fn in filenames if 'NDTiffStack.tif' in fn]
            if len(start_file) == 1:
                data_dirs[dirpath] = start_file[0]
    # sort by directory name
    data_dirs = dict(sorted(data_dirs.items()))
    return data_dirs


def analyze_all_datasets():
    base_url = 'https://mibwiki.biochem.mpg.de'
    space_key = "~hgrabmayr"
    parent_page_title = 'Mock Experiment'
    report_name = 'analysis_report'

    data_dirs = find_data_dirs('.')
    for ddir, fn in data_dirs.items():
        filename = os.path.join(ddir, fn)
        ra = ReportingAnalyzer(
            filename, base_url, space_key, parent_page_title, report_name)
        ra.load()
        ra.identify()
        ra.localize()
        ra.undrift()


def main():
    ra = get_ra()
    ra.load()
    ra.identify()
    ra.localize()
    ra.undrift()
    return ra


if __name__ == '__main__':
    main()
