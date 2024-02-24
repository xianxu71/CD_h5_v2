import numpy as np
import reader
import electromagnetic
import h5py as h5

class main_class:
    '''
    This is the main class where most parameters and data store
    '''

    def __init__(self, nc, nv, nc_for_r, nv_for_r, nc_in_file, nv_in_file ,hovb, nxct, input_folder, W, eta , use_eqp, energy_shift, eps1_correction, use_xct):
        """
        intialize main_class from input.py and all the input files
        """
        self.nc = nc #number of conduction bands in eigenvectors.h5
        self.nv = nv  #number of valence bands in eigenvectors.h5
        self.nc_for_r = nc_for_r #number of conduction bands for <\psi|r|\psi>
        self.nv_for_r = nv_for_r #number of valence bands for <\psi|r|\psi>
        self.hovb = hovb # index of the highest occupied band
        self.nxct = nxct # number of exciton states
        self.input_folder = input_folder #address of input folder
        self.W = W #energy range
        self.eta = eta #broadening coefficient
        self.use_eqp = use_eqp #use eqp correction or not
        self.energy_shift = energy_shift
        self.eps1_correction = eps1_correction
        self.nc_in_file = nc_in_file
        self.nv_in_file = nv_in_file
        self.use_xct = use_xct

        reader.reader(self)
        electromagnetic.electromagnetic(self)

