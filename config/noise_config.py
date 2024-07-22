# all noise types:
noise_types = [
    'cells_4', 
    'cells_1', 
    'voronoi', 
    'microscope_view', 
    'grunge_galvanic_small', 
    'liquid', 
    'clouds_3', 
    'bnw_spots_1', 
    'grunge_leaky_paint', 
    'grunge_rust_fine', 
    'grunge_map_002', 
    'grunge_damas', 
    'grunge_map_005', 
    'messy_fibers_3', 
    'perlin_noise', 
    'gaussian_noise', 
    'clouds_2', 
    'clouds_1', 
    'gabor', 
    'phasor'
]

# short aliases for noise types:
noise_aliases = {
    'cells_4':                  ['cells_4', 'cells4'],
    'cells_1':                  ['cells_1', 'cells1'],
    'voronoi':                  ['voronoi', 'voro'],
    'microscope_view':          ['microscope_view', 'microscope', 'micro'],
    'grunge_galvanic_small':    ['grunge_galvanic_small', 'grunge_galvanic', 'galvanic_small', 'galvanic'],
    'liquid':                   ['liquid'],
    'clouds_3':                 ['clouds_3', 'clouds3'],
    'bnw_spots_1':              ['bnw_spots_1', 'bnw_spots'],
    'grunge_leaky_paint':       ['grunge_leaky_paint', 'leaky_paint', 'paint'],
    'grunge_rust_fine':         ['grunge_rust_fine', 'rust_fine', 'rust'],
    'grunge_map_002':           ['grunge_map_002', 'grunge_map_2', 'grunge_2', 'grunge2'],
    'grunge_damas':             ['grunge_damas', 'damas'],
    'grunge_map_005':           ['grunge_map_005', 'grunge_map_5', 'grunge_5', 'grunge5'],
    'messy_fibers_3':           ['messy_fibers_3', 'messy_fibers', 'fibers'],
    'perlin_noise':             ['perlin_noise', 'perlin'],
    'gaussian_noise':           ['gaussian_noise', 'gaussian', 'gauss'],
    'clouds_2':                 ['clouds_2', 'clouds2'],
    'clouds_1':                 ['clouds_1', 'clouds1'],
    'gabor':                    ['gabor'],
    'phasor':                   ['phasor']
}
noise_aliases = {v: k for k, vs in noise_aliases.items() for v in vs} # invert alias dict

# All noise parameters (in order as they appear in the conditioning vector):
param_names = [
    'scale', 
    'distortion_intensity', 
    'distortion_scale_multiplier', 
    'warp_intensity', 
    'crispness', 
    'dirt', 
    'micro_distortion', 
    'liquid_warp_intensity', 
    'leak_intensity', 
    'leak_scale', 
    'leak_crispness', 
    'base_grunge_contrast', 
    'base_warp_intensity', 
    'distortion', 
    'divisions', 
    'waves', 
    'details', 
    'rotation_random', 
    'gabor_a', 
    'gabor_f', 
    'gabor_omega', 
    'phasor_freq', 
    'phasor_ncells',
    'phasor_density', 
    'phasor_spread'
]

# Which parameters belong to which noise type:
ntype_to_params = [
    ['scale'],
    ['scale'],
    ['scale', 'distortion_intensity', 'distortion_scale_multiplier'],
    ['scale', 'warp_intensity'],
    ['crispness', 'dirt', 'micro_distortion'],
    ['scale', 'liquid_warp_intensity'],
    ['scale'],
    ['scale'],
    ['leak_intensity', 'leak_scale', 'leak_crispness'],
    ['base_grunge_contrast', 'base_warp_intensity'],
    [],
    ['distortion', 'divisions', 'waves', 'details', 'rotation_random'],
    [],
    ['scale'],
    ['scale'],
    ['scale'],
    ['scale'],
    ['scale'],
    ['gabor_a', 'gabor_f', 'gabor_omega'],
    ['phasor_freq', 'phasor_ncells', 'phasor_density', 'phasor_spread']
]
ntype_to_params_map = {noise_types[i]: ntype_to_params[i] for i in range(len(noise_types))}
