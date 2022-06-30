import sys
import os
import json
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.stats import expon

plt.style.use('~/Desktop/thesis_halfwidth.mplstyle')

potential=-125

gs = GridSpec(2,2)

f = f'../{potential}mV_kinetics.csv'
data = pd.read_csv(f, index_col=0)


# Capture
#########
open_times = data['open times (s)'].dropna()

# Test log of times
log_times = np.log10(open_times)
fig = plt.figure()
ax = fig.add_subplot(gs[0,0])
ax.hist(log_times, bins=15, density=False, color='orangered', edgecolor='black')
ax.set_xlabel("log$_{10}$ (Interevent Time (s))")
ax.set_ylabel("Counts")
# plt.show()
# sys.exit()
fig.savefig(f"{potential}mV_log10_interevent_times.png")
plt.close(fig)


def fit(x, a):
    return -a*x

def exp(x, a):
    return np.e**(-a*x)

# Own cdf
open_times_sorted = np.sort(open_times)
p = 1. * np.arange(len(open_times))/(len(open_times)-1)
survival_function = [1- x for x in p]

popt, pcov = curve_fit(fit, open_times_sorted[1:-5], np.log(survival_function[1:-5]))
x_fit = np.linspace(open_times_sorted[1], open_times_sorted[-2], 10000)
y_fit = exp(x_fit, *popt)

fig = plt.figure()
ax = fig.add_subplot(gs[0,0])
ax.plot(open_times_sorted[1:-1], survival_function[1:-1], marker='s', ms=0.5, color='black', ls='none')
ax.plot(x_fit, y_fit, c='orangered', lw=0.5)
ax.set_yscale('log')
ax.set_ylabel("Survival Probability")
ax.set_xlabel("Interevent Time (s)")
# plt.show()
# sys.exit()
fig.savefig(f"{potential}mV_Survival_probability.png")
plt.close(fig)

with open(f'{potential}mV_capturing_kinetics.txt', 'w') as handle:
    handle.write(f'k_capture: {popt[0]} +- {np.sqrt(pcov[0,0])}\n')
    handle.write(f"tau_capture = {1/popt[0]} +- {(np.sqrt(pcov[0,0])/popt[0])*1/popt[0]}")

def exp_pdf(x, a):
    return a*np.e**(-a*x)

# Test distribution of times
vals, bins = np.histogram(open_times_sorted[:-5], bins=15, density=True)
expon_vals = exp_pdf(bins, popt[0])

fig = plt.figure()
ax = fig.add_subplot(gs[0,0])
ax.hist(open_times_sorted[:-5], bins=20, density=True, histtype='barstacked', facecolor='black', rwidth=0.8)
ax.plot(bins, expon_vals, lw=0.5, color='orangered')
ax.set_xlabel("Interevent Time (s)")
ax.set_ylabel("Density")
# plt.show()
# sys.exit()
fig.savefig(f"{potential}mV_interevent_pdf.png")
plt.close(fig)  
