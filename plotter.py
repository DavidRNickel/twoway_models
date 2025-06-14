import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from itertools import chain

if __name__=='__main__':
    mpl.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['legend.fontsize'] = 10

    #
    # Commented out results are from the old RX knowledge vector convention. 
    #

    #
    # ONEWAY vs TWOWAY
    ow_v_twoway_snr = [0,1,2.5,4,5]
    # ow_k4m4t8 = [2.25e-1, 1.2e-1, 1.48e-2, 2.8e-4, 2.3e-5]
    # tw_k4m2t8 = [1.1e-1, 5.2e-2, 1.18e-2, 1.48e-3, 2.4e-4]

    # ow_k6m6t9 = [7.18e-1, 5.0e-1, 2.0e-1, 2.2e-2, 4.6e-3]
    # tw_k6m3t9 = [2.3e-1, 1.18e-1, 3e-2, 4.2e-3, 7.4e-4]

    ow_k4m4t8 = [1.27e-1, 5.54e-2, 7.56e-3, 1.10e-4, 1.52e-5]
    tw_k4m2t8 = [1.09e-1, 5.19e-2, 1.18e-2, 1.48e-3, 2.34e-4]

    ow_k6m6t9 = [3.58e-1, 2.45e-1, 9.57e-2, 8.10e-2, 2.78e-3]
    tw_k6m3t9 = [2.3e-1, 1.18e-1, 3.0e-2, 4.24e-3, 2.12e-4]

    twlin_k6t18 = [1.39, 1.29, 1.12, 9.12e-1, 7.65e-1]
    cl_k6t9 = [1.97, 1.73, 1.18, 7.10e-1, 4.69e-1]

    twlin_k4t16 = [5.79e-1, 4.44e-1, 2.68e-1, 1.33e-1, 7.31e-2]
    cl_k4t8 = [1.11, 9.47e-1, 5.48e-1, 2.88e-1, 1.68e-1]

    #
    # TWOWAY MODEL COMPARISON 
    tw_comparison_snr2 = [1, 5, 10, 15, 20, 30]
    
    # TWLC
    # twlc_k6m3t9_snr1_1 = [1.18e-1, 1.7e-2, 1.17e-3, 1.5e-4, 1.8e-5, 4.5e-6]
    # twlc_k6m3t9_snr1_n1 = [2.5e-1, 1.3e-1, 4.9e-2, 3.3e-2, 2e-2, 1.4e-2]

    # twlc_k6m2t6_snr1_1 = [1.89e-1, 2.64e-2, 1.32e-3, 3.2e-5, 2.84e-5, 4.55e-6]
    # twlc_k6m2t6_snr1_n1 = [3.4e-1, 1.67e-1, 7.2e-2, 4.5e-2, 3.29e-2, 3.27e-2]

    # twlc_k4m2t5_snr1_1 = [2.01e-1, 5.95e-2, 2.34e-2, 9.08e-3, 4.91e-3, 2.53e-3]
    # twlc_k4m2t5_snr1_n1 = [3.30e-1, 1.96e-1, 1.33e-1, 1.11e-1, 1.04e-1, 9.89e-2]

    twlc_k6m2t6_snr1_1 = [1.89e-1, 2.66e-2, 2.45e-3, 1.07e-4, 2.92e-7, 2.88e-7]
    twlc_k6m2t6_snr1_n1 = [3.43e-1, 1.50e-1, 7.74e-2, 4.58e-2, 3.56e-2, 3.05e-2]

    twlc_k6m3t9_snr1_1 = [1.18e-1, 2.57e-2, 7.10e-4, 1.33e-5, 7.90e-6, 5.17e-7]
    twlc_k6m3t9_snr1_n1 = [2.52e-1, 1.38e-1, 4.58e-2, 2.73e-2, 2.81e-2, 1.10e-2]

    twlc_k4m2t5_snr1_1 = [2.01e-1, 7.21e-2, 1.61e-2, 7.69e-3, 4.90e-3, 1.48e-3]
    twlc_k4m2t5_snr1_n1 = [3.30e-1, 1.85e-1, 1.29e-1, 1.11e-1, 9.89e-2, 1.04e-1]
    
    # TWBAF
    twbaf_k6m2t6_snr1_1 = [1.89e-1, 1.97e-2, 1.89e-3, 1.28e-4, 4.04e-7, 9.60e-8]
    twbaf_k6m2t6_snr1_n1 = [3.4e-1, 1.46e-1, 6.52e-2, 4.86e-2, 2.52e-2, 2.22e-2]

    twbaf_k6m3t9_snr1_1 = [8.34e-2, 2.43e-2, 1.31e-4, 1.25e-5, 6.57e-6, 1.19e-7]
    twbaf_k6m3t9_snr1_n1 = [2.52e-1, 1.12e-1, 3.75e-2, 2.28e-2, 1.74e-2, 3.09e-3]

    twbaf_k4m2t5_snr1_1 = [1.61e-1, 5.27e-2, 1.49e-2, 7.09e-3, 3.87e-3, 2.56e-3]
    twbaf_k4m2t5_snr1_n1 = [2.90e-1, 1.89e-1, 1.30e-1, 1.14e-1, 9.45e-2, 9.25e-2]

    # twbaf_k6m2t6_snr1_1 = [1.89e-1, 2.41e-1, 1.48e-3, 1.16e-4, 1.28e-5, 3.95e-7]
    # twbaf_k6m2t6_snr1_n1 = [2.78e-1, 1.92e-1, 7.99e-2, 3.30e-2, 3.18e-2, 2.59e-2]

    # twbaf_k6m3t9_snr1_1 = [1.18e-1, 1.59e-2, 1.54e-3, 4.53e-6, 2.08e-7, 5.16e-6]
    # twbaf_k6m3t9_snr1_n1 = [2.52e-1, 1.14e-1, 5.12e-2, 5.62e-2, 9.37e-3, 2.22e-2]

    # twbaf_k4m2t5_snr1_1 = [2.01e-1, 5.38e-2, 2.09e-2, 4.61e-3, 4.11e-3, 2.18e-3]
    # twbaf_k4m2t5_snr1_n1 = [2.90e-1, 1.85e-1, 1.42e-1, 1.26e-1, 1.04e-1, 9.73e-2]

    # TWRNN
    # Results provided by J. Kim et al.
    bler1_proposed = [2.20E-02,2.21E-02,9.51E-04,6.42E-05,1.55E-05,2.64E-08]
    bler2_proposed = [2.21E-02,2.53E-04,3.57E-04,1.36E-05,2.76E-07,2.03E-08]
    rnn_m6_snr1_1 =  np.array(bler1_proposed)+ np.array(bler2_proposed)
    
    bler1_proposed = [6.84E-02,6.78E-02,7.60E-04,2.4e-6,1.79E-07,1.88E-08]
    bler2_proposed = [6.84E-02,8.94E-04,3.40E-04,2.8e-6,3.00E-09,1.51E-09]
    rnn_m3_snr1_1 =  np.array(bler1_proposed)+ np.array(bler2_proposed)

    bler1_proposed = [1.15E-01,1.16E-01,2.58E-02,1.15E-02,2.14E-03,9.19E-04]
    bler2_proposed = [2.19E-02,1.98E-04,3.09E-03,1.09E-03,2.60E-04,1.11E-05]
    rnn_m6_snr1_n1 = np.array(bler1_proposed)+ np.array(bler2_proposed)
    
    bler1_proposed = [2.22E-01,1.85E-01,3.78E-02,1.47E-02,6.31E-03,6.59E-03]
    bler2_proposed = [6.82E-02,8.44E-03,2.46E-03,7.16E-04,7.50E-05,2.32E-06]
    rnn_m3_snr1_n1 = np.array(bler1_proposed)+ np.array(bler2_proposed)

    twrnn_k4m2t5_snr1_1 = [1.01e-1, 5.19e-2, 8.62e-3, 2.71e-3, 2.22e-3, 7.60e-4]
    twrnn_k4m2t5_snr1_n1 = [1.65e-1, 1.08e-1, 6.20e-2, 6.31e-2, 4.80e-2, 4.70e-2]

    # linear two-way (Allerton) B=3
    bler1_linear = [6.38E-01,5.26E-01,3.64E-01,1.06E-01,1.01E-02, 3.66E-09]
    bler2_linear = [6.58E-01,5.49E-01,2.17E-01,2.93E-02,4.31E-05, 6.2E-11]
    twlin_k3_snr1_1 = np.array(bler1_linear) + np.array(bler2_linear)

    # linear two-way (Allerton) B=3
    bler1_linear = [7.30E-01,7.05E-01,5.20E-01,2.67E-01,7.42E-02,8.31E-03]
    bler2_linear = [6.62E-01,4.17E-01,1.67E-01,2.38E-02,2.86E-04,2.13E-07]
    twlin_k3_snr1_n1 = np.array(bler1_linear) + np.array(bler2_linear)

    # this naming convention is analogous to the one used above but adapted to 
    # the inputs of the linear scheme
    twlin_l4k2t16_snr1_1 = [7.14e-1, 4.52e-1, 1.72e-1, 3.30e-2, 6.27e-3, 1.48e-3]
    twlin_l4k2t16_snr1_n1 = [8.40e-1, 5.78e-1, 3.22e-1, 1.61e-1, 1.11e-1, 9.30e-2]

    # Polar Codes
    polar_k4m10_snr1_1 = [4.42e-2, 2.21e-2, 2.21e-2, 2.21e-2, 2.21e-2, 2.21e-2]
    polar_k4m10_snr1_n1 = [1.04e-1, 8.17e-2, 8.17e-2, 8.17e-2, 8.17e-2, 8.17e-2]

    polar_k6m18_snr1_1 = [3.14e-2, 1.57e-2, 1.57e-2, 1.57e-2, 1.57e-2, 1.57e-2]
    polar_k6m18_snr1_n1 = [1.08e-1, 9.18e-2, 9.18e-2, 9.18e-2, 9.18e-2, 9.18e-2]

    # SENSITIVITY ANALYSIS
    # twlc_fixsnr1_1_snr2_15 = [7.1e-1, 3.1e-1, 1.8e-2, 1.5e-4, 7.0e-5, 5.7e-5]
    twbaf_fixsnr1_1_snr2_15 = [7.8e-1, 3.2e-1, 9.8e-3, 3.18e-6, 6.75e-7, 5.45e-7] # m2

    twlc_fixsnr1_1_snr2_15 = [2.22e-1, 8.46e-2, 3.84e-3, 1.07e-4, 6.00e-5, 4.88e-5]
    # twbaf_fixsnr1_1_snr2_15 = [7.81e-1, 3.50e-1, 1.72e-2, 1.33e-5, 3.56e-6, 3.16e-6]
    twrnn_fix_snr1_1_snr2_15 = [4.3e-1, 1.3e-1, 1.5e-2, rnn_m3_snr1_1[3], 8.3e-7, 7.5e-7]
    
    fig, axs = plt.subplots(2,3)
    axs = list(chain.from_iterable(axs))
    ax1,ax2,ax3,ax4,ax5,ax6=axs
    ax1.semilogy(ow_v_twoway_snr, ow_k4m4t8, label='ALC (R=1/2)', color='blue', ls='-.', marker='o', markersize=8)
    ax1.semilogy(ow_v_twoway_snr, tw_k4m2t8, label='TWLC (R=1/4)', color='orange', ls='-.', marker='o', markersize=8)
    ax1.semilogy(ow_v_twoway_snr, ow_k6m6t9, label='ALC (R=2/3)', color='blue', marker='^', markersize=10)
    ax1.semilogy(ow_v_twoway_snr, tw_k6m3t9, label='TWLC (R=1/3)', color='orange', marker='^', markersize=10)
    ax1.semilogy(ow_v_twoway_snr, twlin_k4t16, label='TWLIN (R=1/4)', color='green', ls='-.', marker='o', markersize=8)
    ax1.semilogy(ow_v_twoway_snr, cl_k4t8, label='CL (R=1/2)', color='black', ls='-.', marker='o', markersize=8)
    ax1.semilogy(ow_v_twoway_snr, twlin_k6t18, label='TWLIN (R=1/3)', color='green', marker='^', markersize=10)
    ax1.semilogy(ow_v_twoway_snr, cl_k6t9, label='CL (R=2/3)', color='black', marker='^', markersize=10)
    ax1.set_xlabel(r'(a) OW vs TW: SNR$_1$=SNR$_2$', fontsize=12)

    ax4.semilogy(tw_comparison_snr2, twlc_fixsnr1_1_snr2_15, label='LC (Fixed)', color='blue', marker='o', markersize=8)
    ax4.semilogy(tw_comparison_snr2, twlc_k6m2t6_snr1_1, label='LC (Orig)', color='blue', ls='-.', marker='^', markersize=10)
    ax4.semilogy(tw_comparison_snr2, twbaf_fixsnr1_1_snr2_15, label='BAF (Fixed)', color='black', marker='o', markersize=8)
    ax4.semilogy(tw_comparison_snr2, twbaf_k6m2t6_snr1_1, label='BAF (Orig)', color='black',ls='-.', marker='^', markersize=10)
    ax4.semilogy(tw_comparison_snr2, twrnn_fix_snr1_1_snr2_15, label='RNN (Fixed)', color='orange', marker='o', markersize=8)
    ax4.semilogy(tw_comparison_snr2, rnn_m3_snr1_1, label='RNN (Orig)', color='orange',ls='-.', marker='^', markersize=10)
    ax4.semilogy(tw_comparison_snr2, twlin_k3_snr1_1, label='LIN (Orig)', color='green', marker='o', markersize=8)
    ax4.set_xlabel(r'(b) Sensitivity Analysis', fontsize=12)

    ax2.semilogy(tw_comparison_snr2, twlc_k6m3t9_snr1_1, label='LC M3', color='blue', marker='o', markersize=8)
    ax2.semilogy(tw_comparison_snr2, twlc_k6m2t6_snr1_1, label='LC M2', color='blue', ls='-.', marker='o', markersize=8)
    ax2.semilogy(tw_comparison_snr2, twbaf_k6m3t9_snr1_1, label='BAF M3', color='black', marker='o', markersize=8)
    ax2.semilogy(tw_comparison_snr2, twbaf_k6m2t6_snr1_1, label='BAF M2', color='black', ls='-.', marker='^', markersize=10)
    ax2.semilogy(tw_comparison_snr2, rnn_m3_snr1_1, label='RNN M3', color='orange', marker='o', markersize=8)
    ax2.semilogy(tw_comparison_snr2, rnn_m6_snr1_1, label='RNN M6', color='orange', ls='-.', marker='^', markersize=10)
    ax2.semilogy(tw_comparison_snr2, twlin_k3_snr1_1, label='LIN M3', color='green', marker='o', markersize=8)
    ax2.semilogy(tw_comparison_snr2, polar_k6m18_snr1_1, label='POL (OL)', color='red', marker='*', markersize=8)
    ax2.set_xlabel(r'(c) SNR$_1$=1, K=6, T=18', fontsize=12, loc='center')

    ax5.semilogy(tw_comparison_snr2, twlc_k6m3t9_snr1_n1, label='LC M3', color='blue', marker='o', markersize=8)
    ax5.semilogy(tw_comparison_snr2, twlc_k6m2t6_snr1_n1, label='LC M2', color='blue', ls='-.', marker='o', markersize=8)
    ax5.semilogy(tw_comparison_snr2, twbaf_k6m3t9_snr1_n1, label='BAF M3', color='black', marker='o', markersize=8)
    ax5.semilogy(tw_comparison_snr2, twbaf_k6m2t6_snr1_n1, label='BAF M2', color='black', ls='-.', marker='^', markersize=10)
    ax5.semilogy(tw_comparison_snr2, rnn_m6_snr1_n1, label='RNN M3', color='orange', marker='o', markersize=8)
    ax5.semilogy(tw_comparison_snr2, rnn_m3_snr1_n1, label='RNN M6', color='orange',ls='-.', marker='^', markersize=10)
    ax5.semilogy(tw_comparison_snr2, twlin_k3_snr1_n1, label='LIN M3', color='green', marker='o', markersize=8)
    ax5.semilogy(tw_comparison_snr2, polar_k6m18_snr1_n1, label='POL (OL)', color='red', marker='*', markersize=8)
    ax5.set_xlabel(r'(d) SNR$_1$=-1, K=6, T=18', fontsize=12, loc='center')

    ax3.semilogy(tw_comparison_snr2, twlc_k4m2t5_snr1_1, label='LC M2', color='blue', marker='o', markersize=8)
    ax3.semilogy(tw_comparison_snr2, twbaf_k4m2t5_snr1_1, label='BAF M2', color='black', marker='o', markersize=8)
    ax3.semilogy(tw_comparison_snr2, twrnn_k4m2t5_snr1_1, label='RNN M2', color='orange', marker='o', markersize=8)
    ax3.semilogy(tw_comparison_snr2, twlin_l4k2t16_snr1_1, label='LIN M2', color='green', marker='o', markersize=8)
    ax3.semilogy(tw_comparison_snr2, polar_k4m10_snr1_1, label='POL (OL)', color='red', marker='*', markersize=8)
    ax3.set_xlabel(r'(e) SNR$_1$=1, K=4, T=10', fontsize=12, loc='center')

    ax6.semilogy(tw_comparison_snr2, twlc_k4m2t5_snr1_n1, label='LC M2', color='blue', marker='o', markersize=8)
    ax6.semilogy(tw_comparison_snr2, twbaf_k4m2t5_snr1_n1, label='BAF M2', color='black', marker='o', markersize=8)
    ax6.semilogy(tw_comparison_snr2, twrnn_k4m2t5_snr1_n1, label='RNN M2', color='orange', marker='o', markersize=8)
    ax6.semilogy(tw_comparison_snr2, twlin_l4k2t16_snr1_n1, label='LIN M2', color='green', marker='o', markersize=8)
    ax6.semilogy(tw_comparison_snr2, polar_k4m10_snr1_n1, label='POL (OL)', color='red', marker='*', markersize=8)
    ax6.set_xlabel(r'(f) SNR$_1$=-1, K=4, T=10', fontsize=12, loc='center')

    fig.supylabel('Sum BLER (all vertical axes)')
    fig.supxlabel(r'SNR$_2$ [dB] (all horizontal axes)')
    
    q = ['a','b','c', 'd', 'e', 'f']
    for adx, ax in enumerate(axs):
        ax.tick_params(which='both')
        ax.grid(which='both')
        ax.legend()

    plt.show()