import glob
import OSCR_analysis as OA
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.filters import median_filter
import numpy as np
import sys

plt.style.use('~/Desktop/thesis_fullwidth.mplstyle')

files = glob.glob('idealisation_normalised_0mV/[!X]*.csv')
red_stack_path = glob.glob('cleaned_data/*_red.tif')[0]
green_stack_path = glob.glob('cleaned_data/*_green.tif')[0]

framerate=485.44
SM_diameter = 3
binary_mask = OA.binary_mask(SM_diameter)
filter_window = 50

red_stack = OA.load_stack(red_stack_path)
green_stack = OA.load_stack(green_stack_path)

for f in files:
    savepath = f'SM+pore_intensities/{os.path.splitext(os.path.basename(f))[0]}'
    data = pd.read_csv(f, index_col=0)
    data.reset_index(inplace=True)

    SM_mass = []

    kymo_red = np.zeros((len(data['frame']), 5))
    kymo_green = np.zeros((len(data['frame']), 7))

    # Loop over frames and find the SM integrated brightness (sum over pixels, using the circular mask)
    for idx, frame in data['frame'].iteritems():
        x = round(data['x'].iloc[idx])
        y = round(data['y'].iloc[idx])

        rect_environ = red_stack[frame, y-SM_diameter//2:y+SM_diameter//2+1, x-SM_diameter//2:x+SM_diameter//2+1]
        circ_environ = rect_environ*binary_mask
        _SM_mass = circ_environ.sum()
        SM_mass.append(_SM_mass)

        _kymo_red = red_stack[frame, y, x-2:x+3].reshape(1, 5)
        kymo_red[idx] = _kymo_red

        _kymo_green = green_stack[frame, y, x-3:x+4].reshape(1, 7)
        kymo_green[idx] = _kymo_green
    
    gs = GridSpec(4, 1, hspace = 0.1)

    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])

    colors_red = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
    colors_green = [(0, 0, 0), (0, 1, 0)] 
    cm_red = LinearSegmentedColormap.from_list("Custom", colors_red, N=100)
    cm_green = LinearSegmentedColormap.from_list("Custom", colors_green, N=100)
    ax1.imshow(kymo_green.T, aspect= 'auto', cmap=cm_green)
    ax2.imshow(kymo_red.T, aspect= 'auto', cmap=cm_red)

    ax1.yaxis.set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.set_visible(False)

    ax2.yaxis.set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    label_format = '{:.3f}'
    ticks_loc = ax2.get_xticks().tolist()
    ticks_loc = [x/framerate for x in ticks_loc]
    ax2.set_xticklabels([label_format.format(x) for x in ticks_loc])
    ax2.set_xlabel('Time (s)')
    fig.savefig(savepath+'_raw_kymograph.png')
    plt.close(fig)
        
    
    data['SM_mass'] = SM_mass
    data['SM_mass_median_filtered'] = median_filter(data['SM_mass'], filter_window)
    data['SM_mass_divided_median_filter'] = data['SM_mass']/data['SM_mass_median_filtered']
    data.to_csv(savepath+'.csv')

    closed_state = data.loc[data['idealised']=='closed', ['frame', 'rescaled_raw_mass']]
    open_state = data.loc[data['idealised']=='open', ['frame', 'rescaled_raw_mass']]
    reference_state = data.loc[data['idealised']=='reference', ['frame', 'rescaled_raw_mass']]
    reverse_state = data.loc[data['idealised']=='reverse', ['frame', 'rescaled_raw_mass']]

    gs = GridSpec(4, 1)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(data['frame']/framerate, data['rescaled_raw_mass'], color='black', linewidth=0.7)
    ax1.plot(closed_state['frame']/framerate, closed_state['rescaled_raw_mass'], c='mediumblue', marker='o', ms=0.4, ls='none')
    ax1.plot(open_state['frame']/framerate, open_state['rescaled_raw_mass'], c='orangered', marker='o', ms=0.4, ls='none')
    ax1.plot(reference_state['frame']/framerate, reference_state['rescaled_raw_mass'], c='darkcyan', marker='o', ms=0.4, ls='none')
    ax1.plot(reverse_state['frame']/framerate, reverse_state['rescaled_raw_mass'], c='darkviolet', marker='o', ms=0.4, ls='none')
    ax1.set_ylabel('I (arb. u.)')

    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(data['frame']/framerate, data['SM_mass'], color='maroon', linewidth=0.7)
    ax2.set_ylabel('I (arb. u.)')
    ax2.set_xlabel('Time (s)')

    plt.tight_layout()
    fig.savefig(savepath+'_raw_SM_signal.png')
    plt.close(fig)

    gs = GridSpec(4, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(data['frame']/framerate, data['rescaled_raw_mass'], color='black', linewidth=0.7)
    ax1.plot(closed_state['frame']/framerate, closed_state['rescaled_raw_mass'], c='mediumblue', marker='o', ms=0.4, ls='none')
    ax1.plot(open_state['frame']/framerate, open_state['rescaled_raw_mass'], c='orangered', marker='o', ms=0.4, ls='none')
    ax1.plot(reference_state['frame']/framerate, reference_state['rescaled_raw_mass'], c='darkcyan', marker='o', ms=0.4, ls='none')
    ax1.plot(reverse_state['frame']/framerate, reverse_state['rescaled_raw_mass'], c='darkviolet', marker='o', ms=0.4, ls='none')
    ax1.set_ylabel('I (arb. u.)')

    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(data['frame']/framerate, data['SM_mass_divided_median_filter'], color='maroon', linewidth=0.7)
    ax2.set_ylabel('I (arb. u.)')
    ax2.set_xlabel('Time (s)')

    plt.tight_layout()
    fig.savefig(savepath+'_filtered_SM_signal.png')
    plt.close(fig)