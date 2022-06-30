import os
import numpy as np
from PIL import Image, ImageSequence
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from OSCR_analysis.image_analysis.stack_viewer import _group_particles

def load_stack(path):
    # Check if it is a file
    if os.path.isfile(path):
        sequence = Image.open(path)
        if 'n_frames' in dir(sequence):

        # Extract all frames
            stack = []
            for frame in ImageSequence.Iterator(sequence):
                stack.append( np.copy(np.array(frame)) )

            imageArray = np.array(stack)

    # Convert simple image type
        else:
            imageArray = np.array(sequence)

             # Format the shape of all image arrays
            imageArray = np.reshape( imageArray, (1, *imageArray.shape) )

        return imageArray

    # Abort if the file is not recognized
    else:
        raise Exception('The input path is neither a file nor a directory.')

def save_stack(stack, path):
    '''
    Save a stack to a file. Literal use of skimage.io and is lazy in the
    sense that it doesn't check for bit-depths or extension. Should be used to
    save .tif stacks 
    
    PARAMETERS
    ----------
    stack:  Numpy array
                Stack to save
    path:   Str
                Location to save stack, including extension
    '''
    io.imsave(path, stack)


def save_track_overlay(frame, tracking_dataframe, savepath, legend=True, **kwargs):
    _plot_style = dict(linewidth=0.7, alpha=0.7)

    vmin = kwargs.get('vmin', frame.min())
    vmax = kwargs.get('vmax', frame.max())
    cmap = kwargs.get('cmap', 'gray')

    fontP = FontProperties()
    fontP.set_size('xx-small')

    fig, ax = plt.subplots()
    ax.imshow(frame, cmap='gray', vmin = vmin, vmax=vmax)

    particle_dict = _group_particles(tracking_dataframe, ['x', 'y'])
    for particle in particle_dict:
        ax.plot(particle_dict[particle]['x'], particle_dict[particle]['y'], **_plot_style, label=particle)
    if 'legend' == True:
        ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1), prop=fontP)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.savefig(savepath)

