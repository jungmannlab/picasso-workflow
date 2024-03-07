"""
workflow.py

This module implements the class ReportingAnalyzer, which orchestrates
picasso-analysis and confluence reporting
"""
import os
import re
from datetime import datetime
import logging

from picasso_workflow.analyse import AutoPicasso
from picasso_workflow.confluence import ConfluenceReporter
from picasso_workflow.util import AbstractPipeline


logger = logging.getLogger(__name__)


class ReportingAnalyzer(AbstractPipeline):
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
