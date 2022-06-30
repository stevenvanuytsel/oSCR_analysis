import OSCR_analysis as OA
import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
import scipy.ndimage
import trackpy as tp

files = glob.glob('cleaned_data/*.tif')

for f in files:
        savename = os.path.splitext(os.path.basename(f))[0]
        stack = OA.load_stack(f)
        ephys = pd.read_csv(glob.glob('cleaned_data/*.csv')[0], index_col=0)
        # Protect against negative infinity potentials due failed conversion from WinEDR to ABF
        if np.isinf(ephys.loc[:, 'voltage (mV)']).any():
                continue
        
        # Trackpy parameters for localising
        diameter = 15
        minmass = None
        separation = 9

        # Localise particles
        location_df = tp.batch(stack, diameter, minmass=minmass, separation=separation, preprocess=True, processes='auto', engine='python')
        # Add applied potentials from electrical recording to localisation dataframe
        new_df = []
        for _frame in location_df.frame.unique():
                print(_frame)
                _loc = location_df.loc[location_df['frame']==_frame].reset_index(drop=True)
                _voltage = list((ephys.loc[ephys['frame'].isin(_loc['frame'])].groupby('frame').mean().reset_index(drop=True)['voltage (mV)'].values))*len(_loc)
                _loc['voltage (mV)'] = _voltage
                new_df.append(_loc)
        location_df = pd.concat(new_df, ignore_index=True)
        location_df.to_csv('tracking/'+savename+f'localisation_df_diameter_{diameter}_separation_{separation}.csv')
        
        # Trackpy parameters for linking tracks
        distance = 5
        memory = 0
        threshold = 250 
        
        # Link particles
        linking_df = tp.link(location_df, distance, memory=memory)
        # Filter short tracks
        linking_df = tp.filter_stubs(linking_df, threshold=threshold)
        
        linking_df.to_csv('tracking/'+savename+f'linking_df_distance_{distance}_memory_{memory}.csv')
        OA.save_track_overlay(stack[50], linking_df, 'tracking/track_overlay.png', legend=False)


        
