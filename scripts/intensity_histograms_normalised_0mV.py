import glob
import pandas as pd
import OSCR_analysis as OA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import sys

from sklearn.neighbors import KernelDensity

plt.style.use('~/Desktop/thesis_fullwidth.mplstyle')


potentials = ['-150mV', '-135mV', '-125mV', '-115mV', '-100mV', '-75mV']


for potential in potentials:
    print(potential)
    files = glob.glob('../2021*/data_cleaning+tracking/*'+potential+'/track_intensities_normalised_0mV*/*.csv')
    
    normalised_intensities = []

    for f in files:
        data = pd.read_csv(f, index_col=0)
        # Only take negative potentials, all of them are more negative than -95 so that's a good threshold
        normalised_intensities.extend(data.loc[data['voltage (mV)']<-70, 'rescaled_raw_mass'])

    normalised_intensities = np.array(normalised_intensities)

    savepath = f'./{potential}/{potential}_'
    
    # Fit open pore intensities with gaussian
    print('Fit open pore')
    _max, _binnumber, _bandwidth, _threshold_1, _threshold_2 = OA.fit_density_kde(normalised_intensities, savepath+'open_pore_fit.png')

    fit_dict = {'Max': _max, 'Binnumber': _binnumber, 'Bandwidth': _bandwidth, 'Threshold_1': _threshold_1, 'Threshold_2': _threshold_2}

    with open(savepath+'open_pore_fit.json', 'w') as handle:
        json.dump(fit_dict, handle)

    # Fit open pore intensities with gaussian
    print('Fit closed pore')
    _max, _binnumber, _bandwidth, _threshold_1, _threshold_2 = OA.fit_density_kde(normalised_intensities, savepath+'closed_pore_fit.png')

    fit_dict = {'Max': _max, 'Binnumber': _binnumber, 'Bandwidth': _bandwidth, 'Threshold_1': _threshold_1, 'Threshold_2': _threshold_2}

    with open(savepath+'closed_pore_fit.json', 'w') as handle:
        json.dump(fit_dict, handle)