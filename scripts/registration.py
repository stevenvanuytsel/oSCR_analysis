import OSCR_analysis as OA
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
import scipy.ndimage
import numpy as np
import pandas as pd
import sys

regpath = '../../registry.tif'

regstack = OA.load_stack(regpath)

# # Project median Z as there are 200 frames in the stack
regframe = OA.project_median_Z(regstack)

rows=62

# # Separate the two channels. We'll take the top 62 rows and the bottom 62 rows, effectively removing 4 rows of pixels from the data
green = OA.rescale_float_to_int(regframe[:rows, :], 16)[0]
red = OA.rescale_float_to_int(regframe[-rows:, :], 16)[0]

# Find shift between both images, green is reference, accuracy 0.01 px
shift, error, diffphase = phase_cross_correlation(green, red, upsample_factor=100, overlap_ratio=2)
registration = pd.DataFrame({'shift_y': shift[0], 'shift_x':shift[1], 'error': error, 'diffphase':diffphase, 'upsample_factor':100, 'rows':rows},index=[0])
registration.to_csv('./registration_params.csv')

red_backtransform = scipy.ndimage.shift(red, shift, cval=0)

OA.save_stack(green, './green_registered.tif')
OA.save_stack(red_backtransform, './red_registered.tif')

# # Plot the overlay of the graticule and the ratio of the two images
fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(green, cmap='Greens', alpha=1)
ax1.imshow(red, cmap='Reds', alpha=0.5)
ax1.set_title('Unregistered', fontsize=18, fontname='Arial', fontweight='black')

ax2.imshow(green, cmap='Greens', alpha=1)
ax2.imshow(red_backtransform, cmap='Reds', alpha=0.5)
ax2.set_title('Registered', fontsize=18, fontname='Arial', fontweight='black')

fig.savefig('graticule_after_registration_0.01px_acc.png')

