import numpy as np

input_folder = './relaxed_SMBA2PBI4_16162_nc8_nv8/'
use_eqp = True
use_xct = True
#

nc = 8
nv = 8

nc_for_r = 40
nv_for_r = 60

nc_in_file = 40
nv_in_file = 60

hovb = 560
nxct = 2000

#W = np.linspace(3.08, 3.35, 8000)
# W = np.linspace(4, 8.8, 10000)
#W = np.linspace(3.5, 13, 10000)
W = np.linspace(2.25, 3.44, 10000)
eta = 0.05
energy_shift = 0.57
#eps1_correction = 2.72
eps1_correction = 0