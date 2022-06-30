import pyabf
import pandas as pd
import numpy as np


def load_ephys(path, **kwargs):
    '''
    Load in electrical recording from patchclamp amplifier. Works
    for both .txt and .abf files. In case of the latter, the pyabf module
    is used.

    Parameter
    ---------
    path:   str
              filepath
    kwargs: keyword arguments
              Used to identify current, voltage and shutter columns.
              Works only when the filepath directs to a .abf file
    
    Returns
    -------
    dataframe:  pandas dataframe
                  Dataframe holding all the values
    datarate:   int
                  Temporal resolution of the recording in Hz
    '''
    # Kwargs are passed on to extract abf function to allow the channels for voltage
    # current and shutter to be custom-set
    if path.endswith('.abf'):
        abf = pyabf.ABF(path)
        # recording rate
        datarate = abf.dataRate
        if abf.channelCount == 1:
            abf.setSweep(0)
            time = abf.sweepX
            current = None
            voltage = abf.sweepY
            shutter = None

            dataframe = pd.DataFrame(np.vstack((time, voltage))).transpose()
            dataframe.rename(columns={0:"time (s)", 1: "voltage (mV)"}, inplace=True)

        elif abf.channelCount == 2:
            abf.setSweep(sweepNumber=0, channel=kwargs.get('current', 0))
            time = abf.sweepX
            current = abf.sweepY

            abf.setSweep(sweepNumber=0, channel=kwargs.get('voltage', 1))
            voltage = abf.sweepY
            shutter = None

            dataframe = pd.DataFrame(np.vstack((time, current, voltage))).transpose()
            dataframe.rename(columns={0:"time (s)",
                                    1:"current (pA)",
                                    2: "voltage (mV)"}, inplace=True)

        elif abf.channelCount ==3:
            abf.setSweep(sweepNumber=0, channel = kwargs.get('current', 0))
            time = abf.sweepX
            current = abf.sweepY

            abf.setSweep(sweepNumber=0, channel=kwargs.get('voltage', 1))
            voltage = abf.sweepY

            abf.setSweep(sweepNumber=0, channel=kwargs.get('shutter', 2))
            shutter = abf.sweepY

            dataframe = pd.DataFrame(np.vstack((time, current, voltage, 
                                    shutter))).transpose()
            dataframe.rename(columns={0:"time (s)",
                                    1:"current (pA)",
                                    2: "voltage (mV)",
                                    3: "shutter (mV)"}, inplace=True)

        else:
            print('Error, channelcount needs to be between 1 and 3')
            time = None
            current = None
            voltage = None
            shutter = None

    elif path.endswith('.txt'):
        dataframe = pd.read_csv(path, sep="\t", header=None, names=['time (s)', 'current (pA)', 'voltage (mV)', 'shutter (mV)'])
        datarate = int(1/np.nanmean(dataframe.loc[:,'time (s)'].diff()))

    return dataframe, datarate

    
    

    
