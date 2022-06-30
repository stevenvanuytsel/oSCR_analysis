import datetime

import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def _nbr2odd(number):
    if number %2 == 0:
        number += 1
    return number

def locate_frame(frame, diameter, minmass, **kwargs):
    '''
    Locate features in give frame using trackpy.locate and display the result. 
    Even diameter values are  automatically converted into the subsequent odd number.

    Parameters
    ----------
    frame:  2D numpy array
             Frame used to locate features in.
    diameter:  int
                Minimum diameter for features to be identified
    minmass:  int
               Minimum intensity of features to be identified
    
    Return
    ------
    diameter, minmass
    '''
    _plot_style = dict(markersize=diameter, markeredgewidth=2,
                       markerfacecolor='none', markeredgecolor='r',
                       marker='o', linestyle='none', alpha=0.5)

    vmin = kwargs.pop('vmin', 0)
    vmax = kwargs.pop('vmax', np.iinfo(frame.dtype).max)
    cmap = kwargs.pop('cmap', 'gray')

    confirm = False
    while confirm == False:
        diameter = _nbr2odd(diameter)
        _plot_style['markersize'] = diameter
        print(diameter)
        dataframe = tp.locate(frame, diameter, minmass=minmass, **kwargs)
    
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.plot(dataframe['x'], dataframe['y'], **_plot_style)
        plt.show()

        choice = input("[c]Confirm, [r]Retry or [x]Cancel: ")
        if choice == 'c':
            print('Confirmed')
            return int(diameter), minmass
        elif choice == 'r':
            print('Retry')
            diameter, minmass = map(float, input('Diameter Minmass?').split(' '))
        elif choice == 'x':
            print('Cancelled')
            break
        else:
            print('That is not a valid option')


def locate_stack(array, diameter, minmass, **kwargs):

    trackpy_version = tp.__version__
    timestamp = datetime.datetime.now().strftime("%Y%m%d %X")
    diameter = _nbr2odd(diameter)    
    
    if 'quiet' in kwargs:
        quiet = kwargs.pop('quiet')

        if quiet:
            tp.quiet()
            dataframe = tp.batch(array, diameter, minmass=minmass, **kwargs)
        else:
            dataframe = tp.batch(array, diameter, minmass=minmass, **kwargs)
    
    else:
        dataframe = tp.batch(array, diameter, minmass=minmass, **kwargs)

    return dataframe, diameter, minmass, trackpy_version, timestamp

def link_features(localisation_dataframe, distance, memory, **kwargs):

    trackpy_version = tp.__version__
    timestamp = datetime.datetime.now().strftime("%Y%m%d %X")

    if 'quiet' in kwargs:
        quiet = kwargs.pop('quiet')
        
        if quiet:
            tp.quiet()
            dataframe = tp.link(localisation_dataframe, distance, memory=memory, **kwargs)
        else:
            dataframe = tp.link(localisation_dataframe, distance, memory=memory, **kwargs)

    else:
        dataframe = tp.link(localisation_dataframe, search_range = distance, memory=memory, **kwargs)
    
    return dataframe, distance, memory, trackpy_version, timestamp
