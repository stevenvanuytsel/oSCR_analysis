import sys
import os
import json
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import groupby
from operator import itemgetter
import scipy.stats as ss
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

plt.style.use('~/Desktop/thesis_halfwidth.mplstyle')


potentials = ['-75mV', '-100mV', '-115mV', '-125mV', '-135mV', '-150mV']

camera_freq = 485.44 # Hz
nyquist = 1/(camera_freq/2)

gs = GridSpec(2, 2)

for potential in potentials:
    print(potential)
    savepath = f'./{potential}/'

    # Construct the kinetics data so we can apply the filter
    files = glob.glob('../2021*/data_cleaning*/*'+potential+'/idealisation_normalised_0mV/[!X]*.csv')

    # Lists that hold values
    open_frames = []
    open_intensity = []
    
    closed_frames = []
    closed_intensity = []

    bg_intensity = []

    for f in files:
        data = pd.read_csv(f, index_col=0)
        idealised = data['idealised'].to_list()
        rescaled_mass = data['rescaled_raw_mass'].to_list()

        # We can only find out the duration of a state if there are at least three present (two of them can be the same)
        if (len(set(idealised)) < 2):
            continue

        # Find all open and closed states
        _open_states = np.where(np.array(idealised)=='open')[0]
        _closed_states = np.where(np.array(idealised)=='closed')[0]

        # If there are no open or closed states we can't extract kinetic information
        if len(_open_states) == 0 or len(_closed_states) == 0:
            continue

        # Lists that hold values
        _open_frames = []
        _open_intensity = []

        _closed_frames = []
        _closed_intensity = []

        _bg_intensity = []

        # Split _open_states into groups of subsequent indices
        for key, group in groupby(enumerate(_open_states), key=lambda x: x[1]-x[0]):
            # Get separate lists of open states
            _idx_list = list(map(itemgetter(1), group))
            try:
                # Try to see if index before and after list exists
                _preceding_state = idealised[_idx_list[0]-1]
                _subsequent_state = idealised[_idx_list[-1]+1]

                # Open state only counts if it is preceded by closed/control state AND followed by closed state
                if (_preceding_state=='closed' or _preceding_state=='reverse' or _preceding_state=='reference') and _subsequent_state=='closed':
                    _open_frames.append(len(_idx_list))
                    _open_intensity.append(np.mean([rescaled_mass[x] for x in _idx_list]))
            except IndexError:
                continue

        # Split _closed_states into groups of subsequent indices
        for key, group in groupby(enumerate(_closed_states), key=lambda x: x[1]-x[0]):
            # Get separate lists of open states
            _idx_list = list(map(itemgetter(1), group))
            try:
                # Try to see if index before and after list exists
                _preceding_state = idealised[_idx_list[0]-1]
                _subsequent_state = idealised[_idx_list[-1]+1]

                # Closed state only counts if it is preceded by open state AND followed by open state
                if _preceding_state=='open' and _subsequent_state=='open':
                    _closed_frames.append(len(_idx_list))
                    _closed_intensity.append(np.mean([rescaled_mass[x] for x in _idx_list]))
            except IndexError:
                continue

        open_frames.extend(_open_frames)
        open_intensity.extend(_open_intensity)
        closed_frames.extend(_closed_frames)
        closed_intensity.extend(_closed_intensity)
        bg_intensity.extend(_bg_intensity)

    # Convert frames to times
    open_times = [x/camera_freq for x in open_frames]
    closed_times = [x/camera_freq for x in closed_frames]

    # Construct dataframe before time filtering so we can discard the background intensities associated with spurious times
    filter_data = pd.DataFrame({'open times (s)':open_times})
    filter_data = pd.concat([filter_data, pd.Series(open_intensity, name='open intensity')], axis=1)
    filter_data = pd.concat([filter_data, pd.Series(closed_times, name='closed times (s)')], axis=1)
    filter_data = pd.concat([filter_data, pd.Series(closed_intensity, name='closed intensity')], axis=1)
    
    # Time filtering (need to do it in list because in place filtering removes unwanted rows from other state too)
    open_times = filter_data['open times (s)'].tolist()
    open_intensity = filter_data['open intensity'].tolist()

    # Filter to not violate Nyquist
    _open_times = []
    _open_intensity = []
    for idx, j in enumerate(open_times):
        if j >= nyquist:
            _open_times.append(j)
            _open_intensity.append(open_intensity[idx])
    
    closed_times = filter_data['closed times (s)'].tolist()
    closed_intensity = filter_data['closed intensity'].tolist()
    _closed_times = []
    _closed_intensity = []
    for idx, j in enumerate(closed_times):
        if j >= nyquist:
            _closed_times.append(j)
            _closed_intensity.append(closed_intensity[idx])

    filter_data = pd.DataFrame({'open times (s)':_open_times})
    filter_data = pd.concat([filter_data, pd.Series(_open_intensity, name='open intensity')], axis=1)
    filter_data = pd.concat([filter_data, pd.Series(_closed_times, name='closed times (s)')], axis=1)
    filter_data = pd.concat([filter_data, pd.Series(_closed_intensity, name='closed intensity')], axis=1)
    filter_data.to_csv(savepath+f'./{potential}_kinetics.csv')

    capture_times = filter_data['open times (s)'].dropna()
    unzipping_times = filter_data['closed times (s)'].dropna()

    # Capture
    #########
    print('Capture')

    tau_capture = capture_times.mean()
    tau_capture_sem = capture_times.dropna().sem()
    k_capture = 1/tau_capture
    k_capture_sem = k_capture*((tau_capture_sem/tau_capture)**2)**(1/2)

    with open(savepath+f'capture/{potential}_naive_mean_capture_kinetics.txt', 'w') as handle:
        handle.write(f'k_capture: {k_capture} +- {k_capture_sem}\n')
        handle.write(f'tau_capture: {tau_capture} +- {tau_capture_sem}\n')


    # Unzipping
    #######
    print('Unzipping')

    tau_unzipping = unzipping_times.mean()
    tau_unzipping_sem = unzipping_times.dropna().sem()
    k_unzipping = 1/tau_unzipping
    k_unzipping_sem = k_unzipping*((tau_unzipping_sem/tau_unzipping)**2)**(1/2)

    with open(savepath+f'unzipping/{potential}_naive_mean_unzipping_kinetics.txt', 'w') as handle:
        handle.write(f'k_unzipping: {k_unzipping} +- {k_unzipping_sem}\n')
        handle.write(f'tau_unzipping: {tau_unzipping} +- {tau_unzipping_sem}\n')

    