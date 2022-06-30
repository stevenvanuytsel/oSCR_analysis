import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, hann, find_peaks
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
from OSCR_analysis.image_analysis.stack_viewer import _group_particles
from OSCR_analysis.ephys_analysis.ephys_processing import _find_idx_recording_end, _add_frames_to_electrical

def add_frames_to_electrical(dataframe, electrical_rate, optical_rate):
    '''
    Utility function to add a column with frames to the electrical data to 
    make plotting and analysis more convenient.

    Parameters
    ----------
    dataframe:  Pandas dataframe
                    Dataframe holding the electrical data
    electrical_rate:    Int
                          Temporal resolution in Hz of electrical recording.
    optical_rate:       Int
                          Temporal resolution in Hz of optical recording
    
    Returns
    -------
    dataframe:  Pandas dataframe
                    Original dataframe with an additional 'frame' column
    '''

    # Find last index where camera is on
    _idx_recording_end = _find_idx_recording_end(dataframe)

    # Add frame column to dataframe
    dataframe = _add_frames_to_electrical(dataframe, electrical_rate, optical_rate, _idx_recording_end)
    return dataframe

def fit_density_kde(data, savepath):
    '''
    Fit density histogram with *gaussian* kernel density estimate, using sklearn KernelDensity.

    Parameters
    ----------
    data:       data to be fitteed
                    array-like
    savepath:   path to save kde-fitted histogram to
                    string
    
    Returns
    -------
    _max:           max value of kde
                        float
    _binnumber:     amount of bins used to construct histogram
                        int
    _bandwidth:     bandwidth used for kde
                        float
    _threshold_1:   lower threshold used to filter data
                        float
    _threshold_2:   upper threshold used to filter data
                        float
    '''

    data=np.array(data)

    # Thresholding
    confirm = False
    while confirm == False:
        # Plot histogram
        fig, ax = plt.subplots()
        ax.hist(data, bins=50, density=True)
        plt.show()

        # Recreate histogram
        _threshold_1 =  float(input("Threshold_1: "))
        _threshold_2 =  float(input("Threshold_2: "))
        if _threshold_1 < min(data):
            _threshold_1 = min(data)
        if _threshold_2 > max(data):
            _threshold_2 = max(data)
 
        _data = data[(data>_threshold_1) & (data<_threshold_2)].reshape(-1, 1)

        fig, ax = plt.subplots()
        ax.hist(_data, bins=50, density=True)
        plt.show()

        choice = input("[c]Confirm or [r]Retry: ")
        if choice == 'c':
            print('Confirmed')
            confirm=True
        else:
            print('Retry')

    # Binning
    confirm = False
    _binnumber = 50
    while confirm == False:
        print(f"Number of bins: {_binnumber}")
        fig, ax = plt.subplots()
        _vals, _bins, _ = ax.hist(_data, bins=_binnumber, density=True)
        plt.show()

        choice = input("[c]Confirm or [r]Retry: ")
        if choice == 'c':
            print('Confirmed')
            confirm=True

        else:
            print('Retry')
            _binnumber = int(input("# of bins: "))

    # Bandwidth selection
    x_fit = np.linspace(_threshold_1, _threshold_2, 1000).reshape(-1, 1)

    confirm = False
    _bandwidth = 1
    while confirm == False:
        print(f"Bandwidth is {_bandwidth}")
        kde = KernelDensity(bandwidth=_bandwidth).fit(_data)
        log_dens = kde.score_samples(x_fit)
        _max = x_fit[np.argmax(np.exp(log_dens))][0]

        fig, ax = plt.subplots()
        ax.hist(_data, bins=_binnumber, density=True, color='orangered', edgecolor='black')
        ax.plot(x_fit, np.exp(log_dens), color='black', lw=1.5)
        ax.axvline(_max, ls='dashed', color='black', lw=1)
        ax.set_xlabel('I (arb. u.)')
        ax.set_ylabel('Density')
        plt.show()

        choice = input("[c]Confirm or [r]Retry: ")
        if choice == 'c':
            print('Confirmed')
            confirm=True
            fig.savefig(savepath, dpi=300)
            plt.close(fig)
            return _max, _binnumber, _bandwidth, _threshold_1, _threshold_2
        
        else:
            print('Retry')
            _bandwidth = float(input("Bandwidth: "))

    


def fit_intensity_histogram_gaussian(data):

    data=np.array(data)

    def gaussian(x, A, mean, sigma):
        return A*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-1/2*(((x-mean)/sigma)**2))

    confirm1 = False
    while confirm1 == False:
        # Plot histogram
        fig, ax = plt.subplots()
        ax.hist(data, bins=50, edgecolor='black')
        plt.show()

        # Recreate histogram
        _threshold_1, _threshold_2 = map(float, input("Threshold_1 Threshold_2 (int): ").split(' '))
        if _threshold_1 < min(data):
            _threshold_1 = min(data)
        if _threshold_2 > max(data):
            _threshold_2 = max(data)
        
        data = data[(data>_threshold_1) & (data<_threshold_2)]

        fig, ax = plt.subplots()
        ax.hist(data, bins=50, edgecolor='black')
        plt.show()

        choice = input("[c]Confirm or [r]Retry: ")
        if choice == 'c':
            print('Confirmed')
            confirm1=True
        else:
            print('Retry')

    # Choose number of bins
    confirm2 = False
    _binnumber = 50
    while confirm2 == False:
        print(f"Number of bins: {_binnumber}")
        fig, ax = plt.subplots()
        _vals, _bins, _ = ax.hist(data, bins=_binnumber, edgecolor='black')
        plt.show()

        choice = input("[c]Confirm or [r]Retry: ")
        if choice == 'c':
            print('Confirmed')
            confirm2=True

        else:
            print('Retry')
            _binnumber = int(input("# of bins: "))

    # Iteratively do the fitting
    confirm3 = False
    while confirm3 == False:
        _bin_centers = _bins[:-1]+np.diff(_bins)/2
        xfit = np.linspace(_bins[0], _bins[-1], 1000)
        
        A, mean, sigma  = map(float, input('Amplitude Mean Sigma ').split(" "))
        popt, _ = curve_fit(gaussian, _bin_centers, _vals, p0=[A, mean, sigma])
        yfit = gaussian(xfit, *popt)
        fig, ax = plt.subplots()
        ax.hist(_bins[:-1], bins=_bins, weights=_vals, edgecolor='black')
        ax.plot(xfit, yfit)
        plt.show()
        choice = input("[c]Confirm or [r]Retry: ")
        if choice == 'c':
            print('Confirmed')
            fitting_params = dict({"Guess_amplitude": A, "Guess_mean": mean, "Guess_sigma": sigma,
                            "Fit_amplitude": popt[0], "Fit_mean": popt[1], "Fit_sigma": popt[2]})
            return popt, fitting_params, _binnumber, _threshold_1, _threshold_2
        else:
            print('Retry')
        

def convolve_hann(signal, kernelpoints=42):
    # Convolve input signal with derivative of Hann kernel
    kernel = hann(kernelpoints)
    dkernel = np.gradient(kernel)
    dsignal = convolve(signal, dkernel)

    return dsignal

def find_transition_idx(convolved_voltage, height=0):
    return find_peaks(convolved_voltage, height)

def find_transition_times(transition_idx, ephys_data):
    times = []
    for i in transition_idx:
        time = ephys_data.loc[i, 'time (s)']
        frames.append(time)
    return times

def find_transition_frames(transition_idx, ephys_data):
    frames = []
    for i in transition_idx:
        frame = ephys_data.loc[i, 'frame']
        frames.append(frame)
    return frames


def slice_voltage_per_frames(dataframe, transition_frames, feature, slice_names=None):
    data_slice = {}
    if 'frame' not in feature:
        feature.append('frame')

    for idx, frame in enumerate(transition_frames):
        if idx == 0:
            data_slice[idx] = dataframe.loc[dataframe['frame']<frame, feature]
        elif idx > 0:
            data_slice[idx] = dataframe.loc[(dataframe['frame']>=transition_frames[idx-1]) & (dataframe['frame']<transition_frames[idx]), feature]
    
    if slice_names != None:
        for idx, name in enumerate(slice_names):
            data_slice[name] = data_slice.pop(idx)
    return data_slice
    
def slice_voltage_per_idx(dataframe, transition_idx):
    data_slice={}

    for idx, index in enumerate(transition_idx):
        
        if idx == 0:
            data_slice[idx] = dataframe.iloc[:index]
        elif idx > 0:
            data_slice[idx] = dataframe.iloc[transition_idx[idx-1]:transition_idx[idx]]
    
    return data_slice


def binary_mask(diameter):
    ''' 
    Make a isotropic 2D elliptical mask with radius diam//2 in a rectangle. Adapted from trackpy function.
    '''
    diameter = (diameter,) * 2
    radius = tuple([x//2 for x in diameter])

    points = [np.arange(-rad, rad+1) for rad in radius]

    coords = np.array(np.meshgrid(*points, indexing='ij'))

    r = [(coord/rad)**2 for (coord, rad) in zip(coords, radius)]

    return sum(r) <= 1





