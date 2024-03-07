picassso_workflow = {
    'load': {
        'filename': [
            'resiround1/test1.ome.tif',
            'resiround2/test1.ome.tif'],
    },
    'identify': {
        'auto_ng': True,
        'box_size': 7,
    },
    'localize': {
        'fit_method': 'lsq',
        'box_size': 7
    },
    'segmentation': {
        'method': 'brightfield',
        'parameters': {
            'filename': 'BF.png'}
    },
    'postproc': [
        'smlm': {
            'min_locs': 10,
            'cluster_radius': 4}
    ],
    'aggregation': {
        'RESI': {
            'evaldirs': [
                'resiround1/eval',
                'resiround2/eval',
                ] 
        }
    }
}