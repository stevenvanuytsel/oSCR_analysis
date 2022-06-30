import os
import OSCR_analysis as OA
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load oSCR recording and dark backgroudn
droplet = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
fpath = '../../'+droplet+'.tif'
blankpath = '../../blank_50em.tif'

savepath = '20211014_R9.4.1_0.75MCaCl2_1.5MKCl_64bp_Cy5-BHQ2_1uM_kinetics_bay4_2ms_'+droplet

# Use potential to synchronise electrical and optical recording
first_applied_potential = -95
last_applied_potential = 95 #mV (the last applied potential is +100 mV so 95 mV is the threshold)

# Load in stack, grab max value so we can rescale back to the original values
stack = OA.load_stack(fpath)
stack_max = np.max(stack)
framerate = 485.44

# Load in electrical data
epath = '../../'+droplet+'.abf'
ephys, datarate = OA.load_ephys(epath)

# Rescale ephys so t=0 is start of stack
ephys = ephys.loc[ephys['shutter (mV)']>1000].reset_index(drop=True)
ephys['time (s)'] = ephys['time (s)']-ephys.iloc[0]['time (s)']

# Make framelist to assign frames to electrical recording
framelist = []
for time in ephys['time (s)'].items():
    framelist.append(int(time[1]/(1/framerate)))
ephys['frame'] = framelist 
# Clean up (probably inaccuracy with the division or shutter closure)
ephys = ephys.loc[ephys['frame']<stack.shape[0]]

# Subtract black bg from stack and filter with 1 px Gaussian
blank = OA.load_stack(blankpath)
blank = OA.project_median_Z(blank)
stack = OA.subtract_stacks(stack, blank)
stack = OA.gaussian_filter(stack, 1)

# Divide each frame by its median pixel due to intensity fluctuations
for idx, _frame in enumerate(stack):
    median = np.median(_frame)
    stack[idx] = _frame/median

#########################################################################
# Sync stack with applied potential until no applied potential is applied
#########################################################################
first_potential_frame = ephys.loc[ephys['voltage (mV)'] < first_applied_potential, 'frame'].unique()
# Add one to the first_potential_frame because the frame is probably a mixture of 0 mV and -100 mV
first_potential_frame_first = int(first_potential_frame[0]+1)

last_potential_frame = ephys.loc[ephys['voltage (mV)'] > last_applied_potential, 'frame'].unique()
# Subtract one from the last frame because that is also a mixture of + 100 mV and 0 mV
last_potential_frame_last = int(last_potential_frame[-1]-1)

# Sync ephys, subtract the startframe from the frame column so we synchronise with the stack
sync_ephys = ephys.loc[(ephys['frame']>=first_potential_frame_first) & (ephys['frame']<=last_potential_frame_last)].reset_index(drop=True)

sync_ephys['frame'] = sync_ephys['frame'] - first_potential_frame_first
# Add one to the last_potential frame to include it
sync_stack = stack[first_potential_frame_first:last_potential_frame_last+1]

# Save synchronised electrical_data and stack
sync_ephys.to_csv('cleaned_data/'+savepath+'.csv')

px1 = 10
# Divide away emission profile, 10 px gaussian blur and rescale to int16
stack_positive = OA.project_median_Z(stack[int(last_potential_frame[0]):last_potential_frame_last])
stack_positive = OA.gaussian_filter(stack_positive, px1)

# Divide dark frame away and rescale back to original stack values
sync_stack = sync_stack.astype(stack_positive.dtype) # Cast to same type as stack_positive
sync_stack = OA.divide_stacks(sync_stack, stack_positive)
sync_stack = OA.rescale_float_to_int(sync_stack, 16, max=stack_max)[0]

# Sync with electrical and save
OA.save_stack(sync_stack, 'cleaned_data/'+savepath+f'blanked_spatially_normalised_{px1}px_rescaled16bint.tif')