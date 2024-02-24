import numpy as np
from math_function import *
import matplotlib.pyplot as plt

def calculate_absorption_eh(main_class):
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction
    volume = main_class.volume
    nk = main_class.nk
    W = main_class.W
    EE = main_class.ME
    MM = main_class.MM
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr= main_class.eqp_corr
    eta = main_class.eta



    pref = 16.0 * np.pi ** 2 / volume / nk/main_class.spinor*4
    RYD = 13.6057039763  # W=W/RYD
    epsilon_r = 1
    Y1_eps2 = np.zeros_like(W)


    E1 = (EE[:, 0])



    for s in range(main_class.nxct):
        energyDif = main_class.excited_energy[s]+energy_shift
        Y1_eps2 += np.abs(E1[s]) ** 2 \
                   * delta_gauss(W / RYD, energyDif/RYD, eta / RYD)



    Y1_eps2 *= pref


    plt.figure()
    #plt.plot(W, -(Y2_eps2 - Y1_eps2-Y2_eps2_0+Y1_eps2_0)*20, 'r', label='L-R')
    plt.plot(1240/W, Y1_eps2, 'r', label='Eps2')
    #plt.plot(W, Y1_eps2, 'b', label='eps2')

    plt.legend()
    plt.show()

    data = np.array([W, Y1_eps2])
    np.savetxt(main_class.input_folder+'absp.dat', data.T)

    return 0

def calculate_absorption_eh_mole(main_class):
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction
    volume = main_class.volume
    nk = main_class.nk
    W = main_class.W
    EE = main_class.ME
    MM = main_class.MM
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr= main_class.eqp_corr
    eta = main_class.eta



    pref = 16.0 * np.pi ** 2 / volume / nk/main_class.spinor*4
    RYD = 13.6057039763  # W=W/RYD
    epsilon_r = 1
    Y1_eps2 = np.zeros_like(W)


    E1 = (EE[:, 0])
    E2 = (EE[:, 1])
    E3 = (EE[:, 2])

    for s in range(main_class.nxct):
        energyDif = main_class.excited_energy[s]+energy_shift
        Y1_eps2 += (np.abs(E1[s]) ** 2+np.abs(E2[s]) ** 2+np.abs(E3[s]) ** 2) \
                   * delta_gauss(W / RYD, energyDif/RYD, eta / RYD)/3



    Y1_eps2 *= pref


    plt.figure()
    #plt.plot(W, -(Y2_eps2 - Y1_eps2-Y2_eps2_0+Y1_eps2_0)*20, 'r', label='L-R')
    plt.plot(1240/W, Y1_eps2, 'b', label='Eps2')
    #plt.plot(W, Y1_eps2, 'b', label='eps2')

    plt.legend()
    plt.show()

    data = np.array([W, Y1_eps2])
    np.savetxt(main_class.input_folder+'absp.dat', data.T)

    return 0

# def calculate_epsR_epsL_noeh(main_class):
#     energy_shift=main_class.energy_shift
#     eps1_correction=main_class.eps1_correction
#     volume = main_class.volume
#     nk = main_class.nk
#     W = main_class.W
#     E_kvc = main_class.E_kvc
#     L_kvc = main_class.L_kvc
#     nv = main_class.nv
#     nc = main_class.nc
#     energy_dft = main_class.energy_dft
#     use_eqp = main_class.use_eqp
#     if use_eqp:
#         eqp_corr= main_class.eqp_corr
#     eta = main_class.eta
#
#
#
#     pref = 16.0 * np.pi ** 2 / volume / nk/main_class.spinor
#     RYD = 13.6057039763  # W=W/RYD
#     light_speed = 137
#     epsilon_r = 1
#     Y1_eps2 = np.zeros_like(W)
#     CD = np.zeros_like(W)
#
#
#     E1 = (E_kvc[:, :, :, 0] + 1j * E_kvc[:, :, :, 1])
#
#
#
#     for ik in range(nk):
#         for iv in range(nv): #range(nv)
#             for ic in range(nc): #range(nc)
#                 energyDif = energy_dft[ik, ic + nv] - energy_dft[ik, iv]+energy_shift/RYD
#                 if use_eqp:
#                     energyDif2 = energyDif + eqp_corr[ik, nv + ic] / RYD - eqp_corr[ik, iv] / RYD
#                 else:
#                     energyDif2 = energyDif
#                 Y1_eps2 += np.abs(E1[ik, iv, ic]) ** 2 \
#                            * delta_gauss(W / RYD, energyDif2, eta / RYD)/2
#                 CD += np.real(E_kvc[ik,iv,ic, 0]*np.conj(L_kvc[ik,iv,ic,0])+E_kvc[ik,iv,ic, 1]*np.conj(L_kvc[ik,iv,ic,1]))\
#                       * W / RYD / light_speed / epsilon_r*delta_gauss(W / RYD, energyDif2, eta / RYD)*2
#
#
#
#     Y1_eps2 *= pref
#
#     CD*=pref
#
#
#     plt.figure()
#     #plt.plot(W, -(Y2_eps2 - Y1_eps2-Y2_eps2_0+Y1_eps2_0)*20, 'r', label='L-R')
#     plt.plot(W, (CD), 'r', label='L-R')
#     #plt.plot(W, Y1_eps2, 'b', label='eps2')
#
#     plt.legend()
#     plt.show()
#
#     data = np.array([W, CD])
#     np.savetxt(main_class.input_folder+'CD0.dat', data.T)
#
#     return 0
def calculate_epsR_epsL_noeh_test(main_class):
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction
    volume = main_class.volume
    nk = main_class.nk
    W = main_class.W
    E_kvc = main_class.E_kvc
    L_kvc = main_class.L_kvc+main_class.S_kvc*2
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr= main_class.eqp_corr
    eta = main_class.eta



    pref = 16.0 * np.pi ** 2 / volume / nk/main_class.spinor*4
    RYD = 13.6057039763  # W=W/RYD
    light_speed = 274
    epsilon_r = 1
    Y1_eps2 = np.zeros_like(W)
    CD = np.zeros_like(W)


    E1 = (E_kvc[:, :, :, 0] + 1j * E_kvc[:, :, :, 1])/np.sqrt(2)



    for ik in range(nk):
        for iv in range(nv): #range(nv)
            for ic in range(nc): #range(nc)
                energyDif = energy_dft[ik, ic + nv] - energy_dft[ik, iv]+energy_shift/RYD
                if use_eqp:
                    energyDif2 = energyDif + eqp_corr[ik, nv + ic] / RYD - eqp_corr[ik, iv] / RYD
                else:
                    energyDif2 = energyDif
                Y1_eps2 += np.abs(E1[ik, iv, ic]) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD)
                CD += 2/3*np.real(E_kvc[ik,iv,ic, 0]*np.conj(L_kvc[ik,iv,ic,0])+E_kvc[ik,iv,ic, 1]*np.conj(L_kvc[ik,iv,ic,1]+
                                                                                                       E_kvc[ik,iv,ic, 2]*np.conj(L_kvc[ik,iv,ic,2])))\
                      * W / RYD / light_speed / epsilon_r*delta_gauss(W / RYD, energyDif2, eta / RYD)*(W*5.064*10**6/(207.56)*3298.2/100) #(5.064*10**6/100*20**3*0.86261/1000*W) #(W*5.064*10**6/100/207.56*3298.2)*np.sqrt(2)




    Y1_eps2 *= pref

    CD*=pref


    plt.figure()
    #plt.plot(W, -(Y2_eps2 - Y1_eps2-Y2_eps2_0+Y1_eps2_0)*20, 'r', label='L-R')
    plt.plot(1240/W, (CD), 'r', label='L-R') #*3298.2*100*400
    #plt.plot(W, Y1_eps2, 'b', label='eps2')

    plt.legend()
    plt.show()

    data = np.array([W, CD])
    np.savetxt(main_class.input_folder+'CD0.dat', data.T)

    return 0

def calculate_absorption_noeh(main_class):
    # noeh_dipole, nk, nv, nc, energy_dft, W, eta, volume, use_eqp=False, eqp_corr = None
    RYD = 13.6057039763
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction

    E_kvc = main_class.E_kvc
    nk = main_class.nk
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    W = main_class.W
    eta = main_class.eta
    volume = main_class.volume
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr = main_class.eqp_corr

    pref = 16.0 * np.pi**2/volume/nk/main_class.spinor*4 # 4 comes from v = p/m ~ 2p

    #Y = np.zeros_like(W)
    eps_2 = np.zeros_like(W)
    eps_1 = np.zeros_like(W)
    #Y2 = np.zeros_like(W)

    for ik in range(nk):
        for iv in range(nv):
            for ic in range(nc):
                energyDif = energy_dft[ik,ic+nv]-energy_dft[ik,iv]+energy_shift/RYD
                if use_eqp:
                    energyDif2 =energyDif + eqp_corr[ik,nv+ic]/RYD-eqp_corr[ik,iv]/RYD
                else:
                    energyDif2 =energyDif

                eps_2 += np.abs(E_kvc[ik,iv,ic,0])**2 * (delta_gauss(W/RYD, energyDif2, eta/RYD))
                eps_1 += np.abs(E_kvc[ik, iv, ic, 0]) ** 2 * (
                    delta_lorentzian(W / RYD, energyDif2, eta / RYD))*(energyDif2-W/RYD)/eta * RYD

    eps_2 *= pref
    eps_1 = pref*eps_1 + 1 + eps1_correction

    plt.figure()
    plt.plot(1240/W, eps_2, 'b')
    #plt.plot(W, eps_1, 'r')
    plt.show()

    data = np.array([W, eps_2, eps_1])
    np.savetxt(main_class.input_folder+'absp0.dat', data.T)

    return

def calculate_absorption_test(main_class):
    # noeh_dipole, nk, nv, nc, energy_dft, W, eta, volume, use_eqp=False, eqp_corr = None
    RYD = 13.6057039763
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction

    E_kvc = main_class.E_kvc
    nk = main_class.nk
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    W = main_class.W
    eta = main_class.eta
    volume = main_class.volume
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr = main_class.eqp_corr

    pref = 16.0 * np.pi**2/volume/nk/main_class.spinor*4 # 4 comes from v = p/m ~ 2p

    #Y = np.zeros_like(W)
    eps_2 = np.zeros_like(W)
    eps_1 = np.zeros_like(W)
    #Y2 = np.zeros_like(W)

    for ik in range(nk):
        for iv in range(nv):
            for ic in range(nc):
                energyDif = energy_dft[ik,ic+nv]-energy_dft[ik,iv]+energy_shift/RYD
                if use_eqp:
                    energyDif2 =energyDif + eqp_corr[ik,nv+ic]/RYD-eqp_corr[ik,iv]/RYD
                else:
                    energyDif2 =energyDif

                eps_2 += (np.abs(E_kvc[ik,iv,ic,0])**2+np.abs(E_kvc[ik,iv,ic,1])**2+np.abs(E_kvc[ik,iv,ic,2])**2) * (delta_gauss(W/RYD, energyDif2, eta/RYD))/3
                eps_1 += (np.abs(E_kvc[ik, iv, ic, 0]) ** 2+np.abs(E_kvc[ik, iv, ic, 1]) ** 2+np.abs(E_kvc[ik, iv, ic, 2]) ** 2) * (
                    delta_lorentzian(W / RYD, energyDif2, eta / RYD))*(energyDif2-W/RYD)/eta * RYD/3

    eps_2 *= pref
    eps_1 = pref*eps_1 + 1 + eps1_correction

    plt.figure()
    plt.plot(1240/W, eps_2, 'b')
    #plt.plot(W, eps_1, 'r')
    plt.show()

    data = np.array([W, eps_2, eps_1])
    np.savetxt(main_class.input_folder+'absp0.dat', data.T)

    return

def calculate_epsR_epsL_noeh_mole(main_class):
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction
    volume = main_class.volume
    nk = main_class.nk
    W = main_class.W
    E_kvc = main_class.E_kvc
    L_kvc = main_class.L_kvc+main_class.S_kvc*2
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr= main_class.eqp_corr
    eta = main_class.eta



    pref = 16.0 * np.pi ** 2 / volume / nk/main_class.spinor*4
    RYD = 13.6057039763  # W=W/RYD
    light_speed = 137
    epsilon_r = 1
    Y1_eps2 = np.zeros_like(W)
    CD = np.zeros_like(W)


    E1 = (E_kvc[:, :, :, 0] + 1j * E_kvc[:, :, :, 1])



    for ik in range(nk):
        for iv in range(nv): #range(nv)
            for ic in range(nc): #range(nc)
                energyDif = energy_dft[ik, ic + nv] - energy_dft[ik, iv]+energy_shift/RYD
                if use_eqp:
                    energyDif2 = energyDif + eqp_corr[ik, nv + ic] / RYD - eqp_corr[ik, iv] / RYD
                else:
                    energyDif2 = energyDif
                Y1_eps2 += np.abs(E1[ik, iv, ic]) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD)/2
                CD += np.real(E_kvc[ik,iv,ic, 0]*np.conj(L_kvc[ik,iv,ic,0])+E_kvc[ik,iv,ic, 1]*np.conj(L_kvc[ik,iv,ic,1]+
                                                                                                       E_kvc[ik,iv,ic, 2]*np.conj(L_kvc[ik,iv,ic,2])))\
                      * W / RYD / light_speed / epsilon_r*delta_gauss(W / RYD, energyDif2, eta / RYD)*2



    Y1_eps2 *= pref

    CD*=pref


    plt.figure()
    #plt.plot(W, -(Y2_eps2 - Y1_eps2-Y2_eps2_0+Y1_eps2_0)*20, 'r', label='L-R')
    plt.plot(W, (CD)*3298.2*100*400, 'r', label='L-R')
    #plt.plot(W, Y1_eps2, 'b', label='eps2')

    plt.legend()
    plt.show()

    data = np.array([W, CD])
    np.savetxt(main_class.input_folder+'CD0.dat', data.T)

    return 0
def calculate_epsR_epsL_eh_mole(main_class):
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction
    volume = main_class.volume
    nk = main_class.nk
    W = main_class.W
    EE = main_class.ME
    MM = main_class.MM+main_class.MS*2
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr= main_class.eqp_corr
    eta = main_class.eta



    pref = 16.0 * np.pi ** 2 / volume / nk/main_class.spinor*4
    RYD = 13.6057039763  # W=W/RYD
    light_speed = 274
    epsilon_r = 1
    Y1_eps2 = np.zeros_like(W)
    CD = np.zeros_like(W)


    E1 = (EE[:, 0])



    for s in range(main_class.nxct):
        energyDif = main_class.excited_energy[s]+energy_shift
        Y1_eps2 += np.abs(E1[s]) ** 2 \
                   * delta_gauss(W / RYD, energyDif/RYD, eta / RYD)
        CD += 2/3*np.real(
            EE[s, 0] * np.conj(MM[s, 0]) + EE[s, 1] * np.conj(MM[s, 1] +EE[s, 2] * np.conj(MM[s, 2]))) \
             * W / RYD / light_speed / epsilon_r * delta_gauss(W / RYD, energyDif/RYD, eta / RYD)



    Y1_eps2 *= pref

    CD*=pref


    plt.figure()
    #plt.plot(W, -(Y2_eps2 - Y1_eps2-Y2_eps2_0+Y1_eps2_0)*20, 'r', label='L-R')
    plt.plot(1240/W, (CD), 'r', label='L-R')
    #plt.plot(W, Y1_eps2, 'b', label='eps2')

    plt.legend()
    plt.show()

    data = np.array([W, CD])
    np.savetxt(main_class.input_folder+'CD.dat', data.T)

    return 0

def calculate_epsR_epsL_eh(main_class):
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction
    volume = main_class.volume
    nk = main_class.nk
    W = main_class.W
    EE = main_class.ME
    MM = main_class.MM
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr= main_class.eqp_corr
    eta = main_class.eta



    pref = 16.0 * np.pi ** 2 / volume / nk/main_class.spinor
    RYD = 13.6057039763  # W=W/RYD
    light_speed = 137
    epsilon_r = 1
    Y1_eps2 = np.zeros_like(W)
    CD = np.zeros_like(W)


    E1 = ((EE[:, 0])+1j*(EE[:, 1]))/np.sqrt(2)



    for s in range(main_class.nxct):
        energyDif = main_class.excited_energy[s]+energy_shift
        Y1_eps2 += np.abs(E1[s]) ** 2 \
                   * delta_gauss(W / RYD, energyDif/RYD, eta / RYD)
        CD += np.real(
            EE[s, 0] * np.conj(MM[s, 0]) + EE[s, 1] * np.conj(MM[s, 1])) \
             * W / RYD / light_speed / epsilon_r * delta_gauss(W / RYD, energyDif/RYD, eta / RYD) * 2



    Y1_eps2 *= pref

    CD*=pref


    plt.figure()
    #plt.plot(W, -(Y2_eps2 - Y1_eps2-Y2_eps2_0+Y1_eps2_0)*20, 'r', label='L-R')
    plt.plot(1240/W, CD, 'r', label='L-R')
    #plt.plot(W, Y1_eps2, 'b', label='eps2_R')

    plt.legend()
    plt.show()

    data = np.array([W, CD])
    np.savetxt(main_class.input_folder+'CD0.dat', data.T)

    return 0