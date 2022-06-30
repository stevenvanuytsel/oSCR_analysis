from sys import argv

import numpy as np
import scipy.ndimage


def duplicate_stack(stack, *frame_range):
    if len(stack.shape)==2:
        return np.copy(stack)
    elif len(stack.shape)==3:
        if frame_range and len(frame_range)==2:
            return np.copy(stack[frame_range[0]:frame_range[1], :, :])
        elif frame_range and len(frame_range)==1:
            return np.copy(stack[frame_range[0]:, :, :])
        elif not frame_range:
            return np.copy(stack)

def project_median_Z(stack):
    median_stack = np.median(stack, axis=0)
    return median_stack

def subtract_stacks(stack_1, stack_2):
    subtracted_stack = np.clip(np.subtract(stack_1, stack_2), 0., None)
    return subtracted_stack

def divide_stacks(stack_1, stack_2):
    divided_stack = np.divide(stack_1, stack_2, where=stack_2!=0)
    return divided_stack

def gaussian_filter(stack, sigma):
    return scipy.ndimage.gaussian_filter(stack, sigma)

def boxcar_filter(stack, width):
    return scipy.ndimage.uniform_filter(stack, width)

def moving_average(stack, window):
    # If stack length isn't evenly divisibly by window, we have to remove some frames from the stack
    if type((stack.shape[0]/window)) != np.int:
        new_stack_length = np.int(stack.shape[0]/window)*window
        stack = stack[:new_stack_length, :, :]

    new_stack = np.ones((int(stack.shape[0]/window), stack.shape[1], stack.shape[2]))
    print(new_stack.shape)

    # for i in range(int(stack.shape[0]/window)):
        
    
def rescale_float_to_int(stack, bit_depth, **kwargs):
    '''
    This function performs a linear rescaling of an input image to a new unsigned bit-depth.
    '''
    if bit_depth not in [8, 16]:
        raise Exception('The selected bit depth is not supported. Please select 8 or 16.')
    
    data_types = {8:'uint8', 16:'uint16'}

    bit_max = kwargs.get('max', np.iinfo(data_types[bit_depth]).max)
    stack_max = stack.max()
    scale_factor = bit_max/stack_max

    rescaled_array = np.copy(stack)
    if len(stack.shape) == 2:
        rescaled_array = (scale_factor*stack)
    
    elif len(stack.shape) == 3:
        for idx, frame in enumerate(rescaled_array):
            rescaled_array[idx] = scale_factor*stack[idx]
    
    rescaled_array = rescaled_array.astype(data_types[bit_depth])
    return rescaled_array, scale_factor


