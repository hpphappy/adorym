from adorym.ptychography import reconstruct_ptychography
import adorym
import h5py
import numpy as np
import dxchange
import datetime
import argparse
import os

timestr = str(datetime.datetime.today())
timestr = timestr[:timestr.find('.')]
for i in [':', '-', ' ']:
    if i == ' ':
        timestr = timestr.replace(i, '_')
    else:
        timestr = timestr.replace(i, '')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default='None')
parser.add_argument('--save_path', default='Apr_2015_Deng')
parser.add_argument('--output_folder', default='algae2_rot90_lr1e-3_size128_positive_coords_99_180') # Will create epoch folders under this
args = parser.parse_args()
epoch = args.epoch
if epoch == 'None':
    epoch = 0
    init = None
else:
    epoch = int(epoch)
    if epoch == 0:
        init = None
    else:
        init_delta = dxchange.read_tiff(os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init_beta = dxchange.read_tiff(os.path.join(args.save_path, args.output_folder, 'epoch_{}/beta_ds_1.tiff'.format(epoch - 1)))
        print(os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init = [np.array(init_delta[...]), np.array(init_beta[...])]

# output_folder = 'siemens_star_aps_2idd/test'
output_folder = 'Apr_2015_Deng/algae2_rot90_lr1e-3_size128_positive_coords_99_180'
distribution_mode = None
optimizer_obj = adorym.AdamOptimizer('obj', output_folder=output_folder, distribution_mode=distribution_mode,
                                     options_dict={'step_size': 1e-3})
optimizer_probe = adorym.AdamOptimizer('probe', output_folder=output_folder, distribution_mode=distribution_mode,
                                        options_dict={'step_size': 1e-3, 'eps': 1e-7})
optimizer_all_probe_pos = adorym.AdamOptimizer('probe_pos_correction', output_folder=output_folder, distribution_mode=distribution_mode,
                                               options_dict={'step_size': 1e-2})


with h5py.File("../demos/Apr_2015_Deng/adorym_fly0444_probe_initial_size128.h5", "r") as f:
    probe_mag = f["exchange/probe_mag"][...]
    probe_phase = f["exchange/probe_phase"][...]


params_21idd_bnp_gpu = {'fname': 'adorym_fly0444_rotate_90_size128_99_180.h5',
                    'theta_st': 0,
                    'theta_end': 0,
                    'n_epochs': 200,
                    'obj_size': (480, 436, 1),
                    'two_d_mode': True,
                    'energy_ev': 5200.0,
                    'psize_cm': 2.2742795912413350e-06, # 2*11.371397956206675 nm
                    'minibatch_size': 141,
                    'output_folder': 'algae2_rot90_lr1e-3_size128_positive_coords_99_180',
                    'cpu_only': True,
                    'save_path': '../demos/Apr_2015_Deng',
                    'use_checkpoint': False,
                    'n_epoch_final_pass': None,
                    'save_intermediate': True,
                    'full_intermediate': True,
                    'initial_guess': None,
                    'random_guess_means_sigmas': (1., 0., 0.001, 0.002), # the Gaussian parameters for initializing the object function. (magnitude_mean, magnitude_sigma, phase_mean, phase_sigma)
                    'n_dp_batch': 350,
                    # ===============================
                    'probe_type': 'supplied',
                    'probe_initial': [probe_mag, probe_phase],
#                     'sign_convention':-1,
                    'n_probe_modes': 8,
                    'probe_pos': [(y, x) for y in (np.arange(161) * 2.198498381314203) for x in (np.arange(141) * 2.198498381314203)],
                    # ===============================
                    'rescale_probe_intensity': True,
                    'free_prop_cm': 'inf',
                    'backend': 'pytorch',
                    'raw_data_type': 'intensity',
                    'beamstop': None,
                    'optimizer': optimizer_obj,
                    'optimize_probe': False,
                    'optimizer_probe': optimizer_probe,
                    'optimize_all_probe_pos': False,
                    'optimizer_all_probe_pos': optimizer_all_probe_pos,
                    'save_history': True,
                    'update_scheme': 'immediate',
                    'unknown_type': 'real_imag',
                    'save_stdout': True,
                    'loss_function_type': 'lsq',
                    'normalize_fft': False
                    }

params = params_21idd_bnp_gpu

reconstruct_ptychography(**params)
