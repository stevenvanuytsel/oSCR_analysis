import glob
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import groupby
from operator import itemgetter
import json

plt.style.use('~/Desktop/thesis_fullwidth.mplstyle')

potential = os.path.basename(os.path.dirname(os.path.realpath(__file__))).split('_')[-1]

open_fit_path = f'../../../intensity_histograms_normalised_0mV/{potential}/{potential}_open_pore_fit.json'
with open(open_fit_path, 'r') as handle:
    open_fitting_params = json.load(handle)
open_fit_mean = open_fitting_params['Max']

closed_fit_path = f'../../../intensity_histograms_normalised_0mV/{potential}/{potential}_closed_pore_fit.json'
with open(closed_fit_path, 'r') as handle:
    closed_fitting_params = json.load(handle)
closed_fit_mean = closed_fitting_params['Max']

threshold = np.mean([open_fit_mean, closed_fit_mean])

files = glob.glob('track_intensities_normalised_0mV/*.csv')

framerate = 485.44
# If subfolder is empty, we should do nothing
if files:
    for f in files:
        savepath = os.path.splitext(os.path.basename(f))[0]
        savepath = f'idealisation_normalised_0mV/{savepath}'
        
        tracks = pd.read_csv(f, index_col=0)

        conditions = [(tracks['rescaled_raw_mass']<threshold), (tracks['rescaled_raw_mass']>threshold)]
        values = ['closed', 'open']
        tracks['idealised'] = np.select(conditions, values)
        # Make sure that the control voltages (0 mV, 50 mV and 100 mV) are clearly identifiable so assign 0.5 to them
        # Set voltage threshold quite close to applied potential so we need to nearly reach applied potential before we step out of control sequence
        # We filter out all the anomalously closed events in the kinetics step so don't need to worry about those here
        tracks.loc[(tracks['voltage (mV)']>=-2) & (tracks['voltage (mV)']<=2), 'idealised'] = 'reference'
        tracks.loc[(tracks['voltage (mV)']>=48) & (tracks['voltage (mV)']<=52), 'idealised'] = 'reverse'
        tracks.to_csv(savepath+'.csv', index=False)

        closed_state = tracks.loc[tracks['idealised']=='closed', ['frame', 'rescaled_raw_mass']]
        open_state = tracks.loc[tracks['idealised']=='open', ['frame', 'rescaled_raw_mass']]
        reference_state = tracks.loc[tracks['idealised']=='reference', ['frame', 'rescaled_raw_mass']]
        reverse_state = tracks.loc[tracks['idealised']=='reverse', ['frame', 'rescaled_raw_mass']]

        fig = plt.figure()
        gs = GridSpec(4, 1)
        ax1 = fig.add_subplot(gs[0,0])
        ax1.plot(tracks['frame']/framerate, tracks['rescaled_raw_mass'], color='black', linewidth=0.7)
        ax1.plot(closed_state['frame']/framerate, closed_state['rescaled_raw_mass'], c='mediumblue', marker='o', ms=0.4, ls='none')
        ax1.plot(open_state['frame']/framerate, open_state['rescaled_raw_mass'], c='orangered', marker='o', ms=0.4, ls='none')
        ax1.plot(reference_state['frame']/framerate, reference_state['rescaled_raw_mass'], c='darkcyan', marker='o', ms=0.4, ls='none')
        ax1.plot(reverse_state['frame']/framerate, reverse_state['rescaled_raw_mass'], c='darkviolet', marker='o', ms=0.4, ls='none')
        ax1.set_ylabel('I (arb. u.)')
        ax2 = fig.add_subplot(gs[1,0])
        ax2.plot(tracks['frame']/framerate, tracks['voltage (mV)'], lw=0.7, color='black')
        ax2.set_ylabel('Voltage (mV)')
        ax2.set_xlabel('Time (s)')
        
        plt.tight_layout()
        fig.savefig(savepath+'.png')
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(gs[0,0])
        ax.plot(tracks['frame']/framerate, tracks['rescaled_raw_mass'], color='black', linewidth=0.7)
        ax.plot(closed_state['frame']/framerate, closed_state['rescaled_raw_mass'], c='mediumblue', marker='o', ms=0.4, ls='none')
        ax.plot(open_state['frame']/framerate, open_state['rescaled_raw_mass'], c='orangered', marker='o', ms=0.4, ls='none')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('I (arb. u.)')
        plt.tight_layout()
        fig.savefig(savepath+'_minimal.png')
        plt.close(fig)
        




