import sys
import os
import json
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit, least_squares

plt.style.use('~/Desktop/thesis_halfwidth.mplstyle')

potential=-125

gs = GridSpec(2,2)

f = f'../{potential}mV_kinetics.csv'
data = pd.read_csv(f, index_col=0)


# Capture
#########
closed_times = data['closed times (s)'].dropna()
# closed_times = [x for x in closed_times if x<1]

# Test log of times
log_times = np.log10(closed_times)
fig = plt.figure()
ax = fig.add_subplot(gs[0,0])
ax.hist(log_times, bins=25, density=False, color='orangered', edgecolor='black')
ax.set_xlabel("log$_{10}$ (Blocked Time (s))")
ax.set_ylabel("Counts")
# plt.show()
# sys.exit()
fig.savefig(f"{potential}mV_log10_blocked_times.png")
plt.close(fig)


def fit(x, a):
    return -a*x

def exp(x, a):
    return np.e**(-a*x)

# Own cdf
closed_times_sorted = np.sort(closed_times)
p = 1. * np.arange(len(closed_times))/(len(closed_times)-1)
survival_function = [1- x for x in p]


## ONE STEP ##
# Fit one step survival probability via linear fit
popt, pcov = curve_fit(fit, closed_times_sorted[:-1], np.log(survival_function[:-1]))
x_fit = np.linspace(closed_times_sorted[0], closed_times_sorted[-2], 10000)
y_fit = exp(x_fit, *popt)

fig = plt.figure()
ax = fig.add_subplot(gs[0,0])
ax.plot(closed_times_sorted, survival_function, marker='s', ms=0.5, color='black', ls='none')
ax.plot(x_fit, y_fit, c='orangered', lw=0.5)
ax.set_ylabel("Survival Probability")
ax.set_xlabel("Interevent Time (s)")
ax.set_yscale('log')
# plt.show()
# sys.exit()
fig.savefig(f"{potential}mV_survival_probability_onestep.png")
plt.close(fig)

with open(f'{potential}mV_onestep_unzipping_kinetics.txt', 'w') as handle:
    handle.write(f'k_unzipping: {popt[0]} +- {np.sqrt(pcov[0,0])}\n')
    handle.write(f"tau_unzipping = {1/popt[0]} +- {(np.sqrt(pcov[0,0])/popt[0])*1/popt[0]}")

# Test distribution of times
bins=50
vals, bins = np.histogram(closed_times_sorted, bins=bins, density=True)

def exp_pdf(x, a):
    return a*np.e**(-a*x)
expon_vals = exp_pdf(x_fit, *popt)

fig = plt.figure()
ax = fig.add_subplot(gs[0,0])
ax.hist(closed_times_sorted, bins=bins, density=True, histtype='barstacked', facecolor='black', rwidth=0.8)
ax.plot(x_fit, expon_vals, lw=0.5, color='orangered')
ax.set_xlabel("Interevent Time (s)")
ax.set_ylabel("Density")
# plt.show()
# sys.exit()
fig.savefig(f"{potential}mV_interevent_pdf_onestep.png")
plt.close(fig)  


## MIXTURE 1step ##
def pdf_mixture_onestep(x, w1, k1, k2):
    return w1*(k1*np.e**(-k1*x))+(1-w1)*(k2*np.e**(-k2*x))

def cdf_mixture_onestep(x, w1, k1, k2):
    return w1*(1-np.e**(-k1*x))+(1-w1)*(1-np.e**(-k2*x))

def survival_mixture_onestep(x, w1, k1, k2):
    return (w1*np.e**(-k1*x)+(1-w1)*np.e**(-k2*x))

def survival_mixture_onestep_residual(params, x, y):
    return y-np.log(survival_mixture_onestep(x, params[0], params[1], params[2]))

x0 = np.array([0.9, 10, 5])
res_lsq = least_squares(survival_mixture_onestep_residual, x0, bounds=([0, 1, 1], [1,np.inf, np.inf]), args=(closed_times_sorted[:-1], np.log(survival_function[:-1])))

SS_res = np.sum(res_lsq.fun**2)
SS_tot = np.sum([(np.mean(survival_function[:-1])-x)**2 for x in survival_function[:-1]])
R2 = 1-(SS_res/SS_tot)

y_fit = survival_mixture_onestep(x_fit, *res_lsq.x)

fig = plt.figure()
ax = fig.add_subplot(gs[0,0])
ax.plot(closed_times_sorted[:-1], survival_function[:-1], marker='s', ms=0.5, color='black', ls='none')
ax.plot(x_fit, y_fit, c='orangered', lw=0.5)
ax.set_ylabel("Survival Probability")
ax.set_xlabel("Interevent Time (s)")
ax.set_yscale('log')
# plt.show()
# sys.exit()
fig.savefig(f"{potential}mV_survival_probability_mixture_onestep_onestep.png")
plt.close(fig)  

with open(f'{potential}mV_onestep_onestep_unzipping_kinetics.txt', 'w') as handle:
    handle.write(f'k_1: {res_lsq.x[1]}\n')
    handle.write(f'k_2: {res_lsq.x[2]}\n')
    handle.write(f'w_1: {res_lsq.x[0]}\n')
    handle.write(f'w_2: {1-res_lsq.x[0]}\n')
    handle.write(f"R2 = {R2}")

expon_vals = pdf_mixture_onestep(x_fit, *res_lsq.x)

plt.style.use('~/Desktop/thesis_fullwidth.mplstyle')
gs = GridSpec(2,1)
fig = plt.figure()
ax = fig.add_subplot(gs[0,0])
ax.hist([x for x in closed_times_sorted if x <=0.18], bins=25, density=True, histtype='barstacked', facecolor='black', rwidth=0.8)
# ax.plot(x_fit, expon_vals, lw=0.5, color='orangered')
ax.set_xlim(0, 0.18)
ax.set_xlabel("Interevent Time (s)")
ax.set_ylabel("Density")
# fig.savefig(f"{potential}mV_interevent_pdf_mixture_onestep_onestep.png")
fig.savefig(f"{potential}mV_fast_interevent_pdf.png")
plt.close(fig)  