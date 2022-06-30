import pyabf
import pandas as pd
import numpy as np


def _find_idx_recording_end(dataframe):
    if not ('time (s)' and 'shutter (mV)') in dataframe:
        raise KeyError("No 'time(s)' or 'voltage (mV)' column was found in the dataframe")
    
    idx_recording_end = dataframe.loc[dataframe['shutter (mV)']> 100].index[-1]
    return idx_recording_end

def _add_frames_to_electrical(dataframe, electrical_rate, optical_rate, idx_recording_end):
    
    # Timepoints per frame
    tpf = int(electrical_rate/optical_rate)
    
    # Calculate number of frames
    frames = round(idx_recording_end/tpf)
    
    # Construct framelist and repeat for the number of timepoints per frame
    framelist = list(range(frames))
    framelist = np.repeat(framelist, tpf)
    
    dataframe['frame'] = pd.Series(framelist)
    
    return dataframe


    # # Add frame column to synchronised dataframe to show which frame the data is
    # # Find timepoints per frame
    # time_points_per_frame = np.round(exposure_time/(synchronised_dataframe['time (s)'].iloc[1]-synchronised_dataframe['time (s)'].iloc[0]),0).astype('int')
    
    # # Original frame is written pythonically so it's +1 when comparing to ImageJ
    # original_frame = startframes_to_delete
    # original_frame_list = [original_frame]
    # frame = 0
    # frame_list = [frame]
    # for idx, i in enumerate(synchronised_dataframe['time (s)'].iloc[1:]):
    #     if (idx+1)%time_points_per_frame == 0:
    #         frame+=1
    #         frame_list.append(frame)

    #         original_frame+=1
    #         original_frame_list.append(original_frame)
    #     else:
    #         frame_list.append(frame)
    #         original_frame_list.append(original_frame)
    
    # synchronised_dataframe.loc[:,'frame']=frame_list
    # synchronised_dataframe.loc[:,'original_frame'] = original_frame_list

    # return startframes_to_delete, synchronised_dataframe

def calculate_voltage_offset(dataframe, track_points):
    off_set = dataframe['voltage (mV)'].tail(track_points).mean()
    off_set_std = dataframe['voltage (mV)'].tail(track_points).std()
    return off_set, off_set_std




