import glob
from os import P_ALL
from statistics import median
import sys
from tempfile import TemporaryFile
import pandas as pd
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as ss
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import least_squares, curve_fit
import statsmodels.api as sm
import numpy as np
import json

plt.style.use('~/Desktop/thesis_fullwidth.mplstyle')

potentials = ['-100mV']
camera_freq = 485.44 # Hz
nyquist = 1/(camera_freq/2)

for potential in potentials:
    print(potential)
    savepath = f'./{potential}/'
    files = glob.glob('../2021*/data_cleaning+tracking/*'+potential+'/SM+pore*/*.csv')

    # The following are all simultaneous events
    # For plotting pore closed state + 10 frames open state, synced on max smf
    signal_incl_open_state_synced_max_smf = []

    # For plotting pore closed state + 10 frames open state, synced on start of block
    signal_incl_open_state_synced_start_block = []

    # Only blocked states, synced on max smf
    signal_synced_max_smf = []

    # Only blocked states, synced on start of block
    signal_synced_start_block = []

    # Kinetics of unzipping quencher and fluorophore, and total
    BHQ2_kinetics = []
    Cy5_kinetics = []
    kinetics_SM = []

    # All points intensity closed state smf
    intensity_SM = []
    intensity_non_SM = []

    # Mean intensity
    mean_intensity_SM = []
    mean_intensity_non_SM = []


    # Kinetics of non simultaneous events
    kinetics_non_SM = []

    for f in files:
        data = pd.read_csv(f, index_col=0)
        
        median_SM_value = data['SM_mass_divided_median_filter'].median()
        std_median_SM_value = data['SM_mass_divided_median_filter'].std()

        # median_SM_value = 1
        # std_median_SM_value = 0.05

        # Find only the closed states that we can use for kinetics (preceded by open and followed by open)
        _closed_states = data.loc[(data['idealised'] == 'closed')]

        if len(_closed_states) == 0:
            continue

        # Split closed states into groups of subsequent indices
        for key, group in groupby(enumerate(_closed_states['frame'].values), key=lambda x: x[1]-x[0]):
            idx_list = list(map(itemgetter(1), group))
            
            try:
                _preceding_state = data.loc[data['frame'] == idx_list[0]-1]['idealised'].values[0]
                _subsequent_state = data.loc[data['frame']== idx_list[-1]+1]['idealised'].values[0]

                if _preceding_state == 'open' and _subsequent_state == 'open':
                    if len(idx_list) >= nyquist:
                        # Include 10 frames before and after closed state for plotting
                        idx_list_incl_open_state = [idx_list[0] - i for i in range(25, 0, -1)]+idx_list+[idx_list[-1]+i for i in range(1, 26)]
                        _simulmeas = data.query('frame in @idx_list')
                        _simulmeas_incl_open_state = data.query('frame in @idx_list_incl_open_state')
                        if max(_simulmeas['SM_mass_divided_median_filter']) >= median_SM_value+3*std_median_SM_value:
                            start_frame_blocked = _simulmeas['frame'].min()
                            end_frame_blocked = _simulmeas['frame'].max()
                            SM_frame = _simulmeas.loc[_simulmeas['SM_mass_divided_median_filter']>=median_SM_value+2*std_median_SM_value, 'frame'].min()

                            # Only closed state, synced on start smf frame above treshold
                            only_closed = _simulmeas.copy()
                            only_closed['frame'] = only_closed['frame']-SM_frame
                            signal_synced_max_smf.append(only_closed)

                            # Closed state +- 10 frames before and after, synced on max smf
                            with_open = _simulmeas_incl_open_state.copy()
                            with_open['frame'] = with_open['frame']-SM_frame
                            signal_incl_open_state_synced_max_smf.append(with_open)

                            # Only closed state, synced on start block
                            sync = _simulmeas.copy()
                            sync['frame'] = sync['frame']-start_frame_blocked
                            signal_synced_start_block.append(sync)

                            # Closed state +- 10 frames before and after, synced on start block
                            with_open = _simulmeas_incl_open_state.copy()
                            with_open['frame'] = with_open['frame'] - start_frame_blocked
                            signal_incl_open_state_synced_start_block.append(with_open)
                            
                            # Difference between start and singlemol frame is BHQ2 unzipping
                            BHQ2_kinetics.append(SM_frame-start_frame_blocked)
                            Cy5_kinetics.append(end_frame_blocked-SM_frame)
                            kinetics_SM.append(len(_simulmeas['frame'].values))

                            intensity_SM.extend(_simulmeas['rescaled_raw_mass'].values)
                            mean_intensity_SM.append(_simulmeas['rescaled_raw_mass'].mean())

                        else:
                            kinetics_non_SM.append(len(_simulmeas['frame'].values))
                            intensity_non_SM.extend(_simulmeas['rescaled_raw_mass'].values)
                            mean_intensity_non_SM.append(_simulmeas['rescaled_raw_mass'].mean())

                        
            except IndexError:
                continue
    
    
    #### PLOTTING INTENSITIES ####
    # Plot all-point histograms of intensity of SM and non-SM
    gs = GridSpec(2,2)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax1.hist(intensity_SM, bins=25, density=True, histtype='barstacked', facecolor='maroon', edgecolor='maroon', rwidth=0.8, alpha=0.5)
    ax1.hist(intensity_non_SM, bins=25, density=True, histtype='barstacked', facecolor='black', edgecolor='black', rwidth=0.8, alpha=0.5)
    ax1.set_xlabel('I (arb. u.)')
    ax1.set_ylabel('Density')
    ax1.set_xlim(-0.5, 1)

    ax2.plot(mean_intensity_SM, [x/camera_freq for x in kinetics_SM], ms=2, ls='none', marker='.', mec='maroon', mfc='maroon', alpha=0.5)
    ax2.plot(mean_intensity_non_SM, [x/camera_freq for x in kinetics_non_SM], ms=2, ls='none', marker='.', mec='black', mfc='black', alpha=0.5)
    ax2.set_xlabel('I (arb. u.)')
    ax2.set_ylabel('Time (s)')
    ax2.set_xlim(-0.5, 1)
    
    plt.savefig(f'{potential}/intensity_SM_vs_nonSM.png')
    plt.close(fig)

    # Plot simultaneous measurements only closed state
    # Synced max smf
    gs = GridSpec(3,1)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])

    for idx, i in enumerate(signal_synced_max_smf):
        ax1.plot(i['frame'].iloc[4:-4]/camera_freq, i['rescaled_raw_mass'].iloc[4:-4], ls='none', marker='s', ms=0.2, color='mediumblue', alpha=0.4)
        ax2.plot(i['frame'].iloc[4:-4]/camera_freq, i['SM_mass_divided_median_filter'].iloc[4:-4], ls='none', marker='s', ms=0.2, color='maroon', alpha=0.4)
    
    ax1.set_xlim(-0.1, 0.1)
    ax1.set_ylim(-0.5, 1)
    ax2.set_xlim(-0.1, 0.1)

    ax1.tick_params(labelbottom=False)
    ax2.set_xlabel('Time (s)')
    ax1.set_ylabel('I (arb. u.)')
    ax2.set_ylabel('I (arb. u.)')
    
    fig.savefig(f'{potential}/closed_state_synced_smf_dots_200ms.png')
    plt.close(fig)

    gs = GridSpec(2,2)
    fig = plt.figure()
    ax = fig.add_subplot(gs[0,0])
    for idx, i in enumerate(signal_synced_max_smf):
        ax.plot(i['rescaled_raw_mass'], i['SM_mass_divided_median_filter'], marker='s', ms=0.2, color='black', alpha=0.4, ls='none')
    ax.set_ylabel('I smf (arb. u.)')
    ax.set_xlabel('I SCCaFT (arb. u.)')
    fig.savefig(f'{potential}/pore_vs_smf_intensity.png')
    plt.close(fig)


    # Plot simultaneous measurements closed state + open state
    # Synced start block, zoomed in
    gs = GridSpec(3,5)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0,:4])
    ax2 = fig.add_subplot(gs[1,:4])

    for idx, i in enumerate(signal_incl_open_state_synced_start_block):
        _open = i.loc[i['idealised']=='open']
        _closed = i.loc[i['idealised']=='closed']
        ax1.plot(_open['frame']/camera_freq, _open['rescaled_raw_mass'], ls='none', marker='s', ms=0.2, color='orangered', alpha=0.6)
        ax1.plot(_closed['frame']/camera_freq, _closed['rescaled_raw_mass'], ls='none', marker='s', ms=0.2, color='mediumblue', alpha=0.6)
        ax2.plot(i['frame']/camera_freq, i['SM_mass_divided_median_filter'], ls='none', marker='s', ms=0.2, color='maroon', alpha=0.6)

    ax1.tick_params(labelbottom=False)
    ax2.set_xlabel('Time (s)')
    ax1.set_ylabel('I (arb. u.)')
    ax2.set_ylabel('I (arb. u.)')

    ax1.set_xlim(-0.025, 0.075)
    ax2.set_xlim(-0.025, 0.075)
    
    fig.savefig(f'{potential}/closed_state_incl_open_state_synced_closed_state_dots_zoom_200ms.png')
    plt.close(fig)

    # Plot simultaneous measurements closed state + open state
    # Synced start block, zoomed in
    gs = GridSpec(2,2)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])

    for idx, i in enumerate(signal_incl_open_state_synced_start_block):
        _open = i.loc[i['idealised']=='open']
        _closed = i.loc[i['idealised']=='closed']
        ax1.plot(_open['frame']/camera_freq, _open['rescaled_raw_mass'], ls='none', marker='s', ms=0.2, color='orangered', alpha=0.6)
        ax1.plot(_closed['frame']/camera_freq, _closed['rescaled_raw_mass'], ls='none', marker='s', ms=0.2, color='mediumblue', alpha=0.6)
        ax2.plot(i['frame']/camera_freq, i['SM_mass_divided_median_filter'], ls='none', marker='s', ms=0.2, color='maroon', alpha=0.6)

    ax1.tick_params(labelbottom=False)
    ax2.set_xlabel('Time (s)')
    ax1.set_ylabel('I (arb. u.)')
    ax2.set_ylabel('I (arb. u.)')

    ax1.set_xlim(-0.1, 0.6)
    ax2.set_xlim(-0.1, 0.6)
    
    fig.savefig(f'{potential}/closed_state_incl_open_state_synced_closed_state_dots_all.png')
    plt.close(fig)

    ##### KINETICS #####
    def survival_onestep(x, k1):
        return np.e**(-k1*x)

    def survival_onestep_residual(params, x, y):
        return y-np.log(survival_onestep(x, params[0]))

    def pdf_2step(x, k1, k2):
        return ((k1*k2)/(k2-k1))*(np.e**(-k1*x)-np.e**(-k2*x))
    
    def pdf_2step_residual(params, x, y):
        return y - pdf_2step(x, params[0], params[1])
    
    def survival_mixture_onestep(x, w1, k1, k2):
        return (w1*np.e**(-k1*x)+(1-w1)*np.e**(-k2*x))
    
    def survival_mixture_onestep_residual(params, x, y):
        return y-np.log(survival_mixture_onestep(x, params[0], params[1], params[2]))

    BHQ2_kinetics = [x/camera_freq for x in BHQ2_kinetics]
    Cy5_kinetics = [x/camera_freq for x in Cy5_kinetics]
    kinetics_SM = [x/camera_freq for x in kinetics_SM]
    kinetics_non_SM = [x/camera_freq for x in kinetics_non_SM]
    
    BHQ2_kinetics_sorted = np.sort(BHQ2_kinetics)
    p_BHQ2 = 1. * np.arange(len(BHQ2_kinetics_sorted))/(len(BHQ2_kinetics_sorted)-1)
    survival_BHQ2 = [1 - x for x in p_BHQ2]

    Cy5_kinetics_sorted = np.sort(Cy5_kinetics)
    p_Cy5 = 1. * np.arange(len(Cy5_kinetics_sorted))/(len(Cy5_kinetics_sorted)-1)
    survival_Cy5 = [1 - x for x in p_Cy5]

    kinetics_sm_sorted = np.sort(kinetics_SM)
    p_sm = 1. * np.arange(len(kinetics_sm_sorted))/(len(kinetics_sm_sorted)-1)
    survival_sm = [1 - x for x in p_sm]

    kinetics_non_sm_sorted = np.sort(kinetics_non_SM)
    p_non_sm = 1. * np.arange(len(kinetics_non_sm_sorted))/(len(kinetics_non_sm_sorted)-1)
    survival_non_sm = [1 - x for x in p_non_sm]

    plt.style.use('~/Desktop/thesis_fullwidth.mplstyle')
    gs = GridSpec(2,2)

    fig = plt.figure()
    ax = fig.add_subplot(gs[0,0])
    ax.plot(kinetics_sm_sorted[:-1], survival_sm[:-1], marker='s', ms=0.2, c='maroon', ls='none', alpha=0.5)
    ax.set_yscale('log')
    ax.set_ylabel("Survival Probability")
    ax.set_xlabel("Blocked time (s)")
    ax.set_xlim(right=1)

    fig.savefig(f'{potential}/survival_probability_sm.png')
    plt.close(fig)
    
    fig = plt.figure()
    ax = fig.add_subplot(gs[0,0])
    ax.plot(kinetics_non_sm_sorted[:-1], survival_non_sm[:-1], marker='s', ms=0.2, c='black', ls='none', alpha=0.5)
    ax.set_yscale('log')
    ax.set_ylabel("Survival Probability")
    ax.set_xlabel("Blocked time (s)")
    ax.set_xlim(right=1)

    fig.savefig(f'{potential}/survival_probability_non-sm.png')
    plt.close(fig)

    # Fit BHQ2 and Cy5 unzipping
    x0_BHQ2 = np.array([0.5, 80, 10])
    res_lsq_BHQ2 = least_squares(survival_mixture_onestep_residual, x0_BHQ2, bounds=([0, 1, 1], [1, np.inf, np.inf]), args=(BHQ2_kinetics_sorted[:-1], np.log(survival_BHQ2[:-1])))
    
    # popt_BHQ2, pcov_BHQ2 = curve_fit(survival_onestep, BHQ2_kinetics_sorted[1:-1], survival_BHQ2[1:-1])
    x_fit_BHQ2 = np.linspace(BHQ2_kinetics_sorted[0], BHQ2_kinetics_sorted[-1])
    y_fit_BHQ2 = survival_mixture_onestep(x_fit_BHQ2, *res_lsq_BHQ2.x)

    x0_Cy5 = np.array([0.8, 10, 5])
    res_lsq_Cy5 = least_squares(survival_mixture_onestep_residual, x0_Cy5, bounds=([0, 1, 1], [1, np.inf, np.inf]), args=(Cy5_kinetics_sorted[:-1], np.log(survival_Cy5[:-1])))

    x_fit_Cy5 = np.linspace(Cy5_kinetics_sorted[0], Cy5_kinetics_sorted[-1])
    y_fit_Cy5 = survival_mixture_onestep(x_fit_Cy5, *res_lsq_Cy5.x)

    print(res_lsq_BHQ2.x[1])
    print(res_lsq_Cy5.x[1])

    fig = plt.figure()
    ax = fig.add_subplot(gs[0,0])
    ax.plot(BHQ2_kinetics_sorted[:-1], survival_BHQ2[:-1], marker='s', ms=0.5, c='darkviolet', ls='none')
    ax.plot(Cy5_kinetics_sorted[:-1], survival_Cy5[:-1], marker='s', ms=0.5, c='orangered', ls='none')
    ax.plot(x_fit_BHQ2, y_fit_BHQ2, c='black', lw=0.5)
    ax.plot(x_fit_Cy5, y_fit_Cy5, c='black', lw=0.5)
    ax.set_yscale('log')
    ax.set_ylabel("Survival Probability")
    ax.set_xlabel("Blocked time (s)")
    # ax.set_xlim(0,0.2)

    fig.savefig(f'{potential}/survival_probability_BHQ2_Cy5.png')
    plt.close(fig)

    vals, bins = np.histogram([x for x in kinetics_sm_sorted if x < 0.4], bins=15, density=True)
    bin_centers = bins[:-1] + np.diff(bins) / 2
    print(len(vals), len(bin_centers))
    x0_pdf = np.array([100, 20])
    res_lsq_pdf = least_squares(pdf_2step_residual, x0_pdf, bounds = ([1, 1], [np.inf, np.inf]), args = (bin_centers, vals))

    x_fit = np.linspace(0, 0.4)
    y_fit = pdf_2step(x_fit, *res_lsq_pdf.x)

    print(*res_lsq_pdf.x)

    fig = plt.figure()
    ax = fig.add_subplot(gs[0,0])
    ax.hist([x for x in kinetics_sm_sorted if x < 0.4], bins=15, density=True, histtype='barstacked', facecolor='maroon', rwidth=0.8, alpha=0.5)
    ax.hist([x for x in kinetics_non_sm_sorted if x < 0.4], bins=15, density=True, histtype='barstacked', facecolor='black', rwidth=0.8, alpha=0.5)
    ax.plot(x_fit, y_fit, color='maroon', lw=0.5)
    ax.set_xlabel("Blocked time (s)")
    ax.set_ylabel("Density")

    fig.savefig(f'{potential}/sm_vs_non_sm_unzipping_histogram.png')
    plt.close(fig)

    # Test all times
    kinetics_all = kinetics_non_SM+kinetics_SM
    kinetics_all_sorted = np.sort(kinetics_all)
    p_all = 1. * np.arange(len(kinetics_all_sorted))/(len(kinetics_all_sorted)-1)
    survival_all = [1 - x for x in p_all]

    gs = GridSpec(2,2)

    fig = plt.figure()
    ax = fig.add_subplot(gs[0,0])
    ax.plot(kinetics_all_sorted[:-1], survival_all[:-1], marker='s', ms=0.2, c='maroon', ls='none', alpha=0.5)
    ax.set_yscale('log')
    ax.set_ylabel("Survival Probability")
    ax.set_xlabel("Blocked time (s)")
    ax.set_xlim(right=1)
    # plt.show()
    plt.close(fig)




