import OSCR_analysis as OA
import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.style.use('~/Desktop/thesis_fullwidth.mplstyle')

f = glob.glob('tracking/*linking*.csv')[0]
savename = os.path.splitext(os.path.basename(f))[0]
savepath_normalised = 'track_intensities_normalised_0mV/'+savename

tracks = pd.read_csv(f, index_col=0).reset_index().drop('frame.1', axis=1)

framerate = 485.44

########################################################################################
# Normalise intensity using internal reference level of 0 mV and background fluorescence
########################################################################################

new_tracking_dfs = []
particles = tracks.particle.unique()

for particle in particles:
        _track = tracks.loc[tracks['particle']==particle]

        # Check if 0 mv exist in dataframe and that there are a sufficient number of points to grab a decent average
        if (len(_track.loc[(_track['voltage (mV)']>-2) & (_track['voltage (mV)']<2)])>20):
                _neutral_intensity = _track.loc[(_track['voltage (mV)']>-2) & (_track['voltage (mV)']<2), 'raw_mass'].mean()
                _bg_intensity = _track['background_raw_mass'].mean()
                _track['rescaled_raw_mass'] = _track['raw_mass'].apply(lambda x: (x-_bg_intensity)/(_neutral_intensity-_bg_intensity))
                if not (_track['rescaled_raw_mass']<-2).any():
                        new_tracking_dfs.append(_track)
       
new_tracking_df = pd.concat(new_tracking_dfs).reset_index(drop=True)

# Split dataframe into separate particles for plotting
particles = new_tracking_df.particle.unique()

for particle in particles:
        _track = new_tracking_df.loc[new_tracking_df['particle']==particle].reset_index(drop=True)
        _track.to_csv(savepath_normalised+'normalised_intensity_0mV_particle'+str(particle)+'.csv')
        gs = GridSpec(4, 1)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[0,0])
        ax1.plot(_track['frame']/framerate, _track['rescaled_raw_mass'], lw=1, color='black')
        ax2 = fig.add_subplot(gs[1,0])
        ax2.plot(_track['frame']/framerate, _track['voltage (mV)'], lw=0.7, color='black')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Voltage (mV)')
        ax1.set_ylabel('I (arb. u.)')
        plt.tight_layout()
        fig.savefig(savepath_normalised+'normalised_intensity_0mV_particle'+str(particle)+'.png')
        plt.close(fig)

