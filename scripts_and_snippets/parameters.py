reporter_config = {
    'ConfluenceReporter': {
        'base_url': '',
        'space_key': '',
        'parent_page_title': '',
        'report_name': ''}
}

analysis_config = {
    'result_location': '',
    'camera_info': {
        'gain': 1,
        'sensitivity': 0.45,
        'baseline': 100,
        'qe': 0.82,
        'pixelsize': 130,  # nm
    },
    'gpufit_installed': False,
}

# e.g. for single dataset evaluation
workflow_modules_sgl = [
    ('load', {
        'filename': 'resiround1/test1.ome.tif',
        'sample_movie' : {
            'filename': 'selected_frames.mp4',
            'n_sample': 40,
            'max_quantile': .9998,
            'fps': 2,
            },
        },
    ),
    ('identify', {
        'auto_netgrad': {
            'filename': 'ng_histogram.png',
            'frame_numbers': ('$get_prior_result', "results, load, sample_movie, sample_frame_idx"), # get from prior results
            'start_ng': -3000,
            'zscore': 5,
            },
        'ids_vs_frame': {
            'filename': 'ids_vs_frame.png'
            },
        'box_size': 7,
        },
    ),
    # ('identify', {
    #     'net_gradient': 5000,
    #     'ids_vs_frame': {
    #         'filename': 'ids_vs_frame.png'
    #     },
    #     'box_size': 7,
    #     },
    # ),
    ('localize', {
        'fit_method': 'lsq',
        'box_size': 7,
        'fit_parallel': True
        },
    ),
    ('undrift_rcc', {
        'segmentation': 1000,
        'max_iter_segmentations': 4,
        'filename': 'drift.csv',
        'save_locs': {'filename': 'locs_undrift.hdf5'}
        
        }
    ),
    ('manual', {
        'prompt': 'Please manually undrift.',
        'filename': 'locs_undrift.hdf5',
        }),
    # ('segmentation', {
    #     'method': 'brightfield',
    #     'parameters': {
    #         'filename': 'BF.png'}
    #     },
    # ),
    # ('cluster_smlm', {
    #     'min_locs': 10,
    #     'cluster_radius': 4
    #     },
    # ),
]

# for dataset aggregation, after they have been analyzed separately
workflow_modules_agg = [
    ('RESI', {
            'evaldirs': [
                'resiround1/eval',
                'resiround2/eval',
                ] 
        })
]


# e.g. for multi dataset evaluation and aggregation
workflow_modules_multi = {
    'single_dataset_tileparameters':{
        '$tags': ['A', 'B'],
        'filename': ['fn1', 'fn2'],},
    'single_dataset_modules': [
        ('load', {
            'filename': ('$map', 'filename'),
            'sample_movie' : {
                'filename': 'selected_frames.mp4',
                'n_sample': 40,
                'max_quantile': .9998,
                'fps': 2,
                },
            },
        ),...],
    'aggregation_modules': [
        ('RESI', {
            'evaldirs': ('$get_prior_results', 'all_results, $all, undrift_rcc, locs_undrift.hdf5')
            })
        ]
}
