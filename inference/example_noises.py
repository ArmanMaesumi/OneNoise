# Some example noise configurations for testing the model during training
# For the names/aliases of all noise types, as well as their supported
# parameters, please see `OneNoise/config/noise_config.py`

def vertical_blends():
    cs = []

    c1 = {
        'cls': 'micro',
        'sbsparams': {
            'scale':0.25
        }
    }
    c2 = {
        'cls': 'rust',
        'sbsparams': {
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'bnw_spots',
        'sbsparams': {
        }
    }
    c2 = {
        'cls': 'damas',
        'sbsparams': {
            'distortion': 0.85,
            'divisions': 0.85,
            'waves': 0.85,
            'details': 0.85,
            'rotation_random': 0.8
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'perlin',
        'sbsparams': {
            'scale':0.4
        }
    }
    c2 = {
        'cls': 'grunge2',
        'sbsparams': {
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'grunge5',
        'sbsparams': {
        }
    }
    c2 = {
        'cls': 'grunge2',
        'sbsparams': {
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'fibers',
        'sbsparams': {
        }
    }
    c2 = {
        'cls': 'grunge2',
        'sbsparams': {
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'cells1',
        'sbsparams': {
            'scale':0.35
        }
    }
    c2 = {
        'cls': 'liquid',
        'sbsparams': {
            'scale': 0.5,
            'liquid_warp_intensity': 0.5
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'grunge5',
        'sbsparams': {
        }
    }
    c2 = {
        'cls': 'fibers',
        'sbsparams': {
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'damas',
        'sbsparams': {
            'distortion': 0.85,
            'divisions': 0.85,
            'waves': 0.85,
            'details': 0.85,
            'rotation_random': 0.8
        }
    }
    c2 = {
        'cls': 'liquid',
        'sbsparams': {
            'scale': 0.5,
            'liquid_warp_intensity': 0.5
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'voro',
        'sbsparams': {
            'scale': 0.5,
            'distortion_intensity': 0.5
        }
    }
    c2 = {
        'cls': 'grunge5',
        'sbsparams': {
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'micro',
        'sbsparams': {
            'scale': 0.5
        }
    }
    c2 = {
        'cls': 'galvanic',
        'sbsparams': {
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'gaussian',
        'sbsparams': {
            'scale': 0.5
        }
    }
    c2 = {
        'cls': 'cells_4',
        'sbsparams': {
            'scale': 0.5
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'micro',
        'sbsparams': {
            'scale': 0.5
        }
    }
    c2 = {
        'cls': 'cells_1',
        'sbsparams': {
            'scale': 0.5
        }
    }
    cs += [ (c1, c2) ]

    return cs

def horizontal_blends():
    cs = []

    c1 = {
        'cls': 'grunge5',
        'sbsparams': {
        }
    }
    c2 = {
        'cls': 'galvanic',
        'sbsparams': {}
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'cells1',
        'sbsparams': {
            'scale':0.35
        }
    }
    c2 = {
        'cls': 'damas',
        'sbsparams': {
            'distortion': 0.5,
            'divisions': 0.5,
            'waves': 0.7,
            'details': 0.7,
            'rotation_random': 0.0
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'bnw_spots',
        'sbsparams': {
        }
    }
    c2 = {
        'cls': 'clouds2',
        'sbsparams': {}
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'liquid',
        'sbsparams': {
            'scale': 0.5
        }
    }
    c2 = {
        'cls': 'galvanic',
        'sbsparams': {}
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'clouds2',
        'sbsparams': {}
    }
    c2 = {
        'cls': 'clouds3',
        'sbsparams': {}
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'cells4',
        'sbsparams': {
            'scale': 0.5
        }
    }
    c2 = {
        'cls': 'micro',
        'sbsparams': {
            'scale': 0.8
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'damas',
        'sbsparams': {
            'distortion': 0.6,
            'divisions': 0.6,
            'waves': 0.0,
            'details': 0.0,
            'rotation_random': 0.0
        }
    }
    c2 = {
        'cls': 'phasor',
        'sbsparams': {
            'phasor_freq': 0.0,
            'phasor_ncells': 0.2,
            'phasor_density': 1.0,
            'phasor_spread': 0.0
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'grunge2',
        'sbsparams': {}
    }
    c2 = {
        'cls': 'paint',
        'sbsparams': {}
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'galvanic',
        'sbsparams': {}
    }
    c2 = {
        'cls': 'rust',
        'sbsparams': {}
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'voro',
        'sbsparams': {
            'distortion_intensity': 0.5,
            'scale': 0.2
        }
    }
    c2 = {
        'cls': 'cells1',
        'sbsparams': {
            'scale': 0.5
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'fibers',
        'sbsparams': {
        }
    }
    c2 = {
        'cls': 'liquid',
        'sbsparams': {
            'scale': 0.5,
            'liquid_warp_intensity': 0.5
        }
    }
    cs += [ (c1, c2) ]

    c1 = {
        'cls': 'perlin',
        'sbsparams': {
            'scale': 0.9
        }
    }
    c2 = {
        'cls': 'gaussian',
        'sbsparams': {
            'scale': 0.9
        }
    }
    cs += [ (c1, c2) ]

    return cs