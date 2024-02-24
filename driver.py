from input import *
import main_class
import optical

if __name__ == '__main__':
    main_class = main_class.main_class(nc, nv, nc_for_r, nv_for_r, nc_in_file, nv_in_file, hovb, nxct, input_folder, W, eta , use_eqp, energy_shift, eps1_correction, use_xct)

    #optical.calculate_epsR_epsL_noeh(main_class)
    #optical.calculate_epsR_epsL_noeh_test(main_class)
    #optical.calculate_epsR_epsL_noeh_mole(main_class)
    optical.calculate_epsR_epsL_eh(main_class)
    #optical.calculate_epsR_epsL_eh_mole(main_class)
    #optical.calculate_absorption_noeh(main_class)
    #optical.calculate_absorption_test(main_class)
    #optical.calculate_absorption_eh_mole(main_class)
    #optical.calculate_absorption_eh(main_class)
    #optical.calculate_m_eh(main_class)
    #optical.calculate_m_noeh(main_class)