#!/usr/bin/env python3

"""
Plotting a 2D contoured histogram
"""

import numpy as np
import matplotlib.pyplot as plt
import string
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def contoured_2d_hist(wl,delta_perf,outputfolder):
    """
    Creating a contoured 2D histogram and saving the plot.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    delta_perf : numpy.ndarray
        Change in transmission/reflection from nominal case [%].
    outputfolder : str
        Name of output folder for plots.

    Returns
    -------
    None.

    """

    bin_size = 0.1 # binsize
    n_bins = np.arange(delta_perf.min(), delta_perf.max() + bin_size, bin_size) # number of bins
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10,6))
    cm_used = plt.cm.get_cmap('nipy_spectral')

    # Iterate over wl values
    for i in range(len(wl)):
        print(i,'/',len(wl)-1)        
        delta_perf_sample = delta_perf[:, i]
        # remove any non-finite values 
        delta_perf_sample = delta_perf_sample[np.isfinite(delta_perf_sample)] 
        # the histogram and bin edges 
        hist, bin_edges = np.histogram(delta_perf_sample, bins=n_bins)
        # Calculate the density as a percentage of the total points 
        density = hist/len(delta_perf) * 100
        density_value = np.array(density.tolist())
        # Mask for non-zero frequencies
        nonzero_mask = np.where(density_value > 0)[0]
        x_nonzero = np.full((len(bin_edges[:-1][nonzero_mask])), wl[i])
        y_nonzero = bin_edges[:-1][nonzero_mask]
        density_values = density_value[nonzero_mask]
        
        # color normalization with the minimum and maximum density values.
        norm = mcolors.Normalize(vmin=min(density_values), vmax=max(density_values))  # Normalize the density values
        # create scatter plot 
        sc = ax.scatter(x_nonzero, y_nonzero, c=density_values, cmap=cm_used, norm=norm, marker='s')
    
    # add a colorbar 
    cb = plt.colorbar(sc, format='%1.0f', ax=ax)
    cb.set_label('Density[%]',rotation=90)
    
    ax.set_ylabel(r'$ \Delta T $ [%]')
    ax.set_xlabel('$\lambda$ [nm]')
    ax.set_xlim((min(wl),max(wl)))
    ax.axhline(y=0, color='white', linestyle='-',lw=3)
    ax.axhline(y=0, color='black', linestyle='-.',lw=1.5)
    ax.axvline(x=1950, color='white', linestyle='-',lw=3)
    ax.axvline(x=1950, color='deeppink', linestyle='-',lw=1.5)
    ax.set(xlim=(min(wl),max(wl)))
    fig.savefig(f"{outputfolder}/rand_dep_error_delta_t.png",dpi=600)


def double_contoured_2d_hist(wl,delta_perf1,delta_perf2,outputfolder):
    """
    Creating a set of two contoured 2D histogram and saving the plot.

    Parameters
    ----------
    wl : TYPE
        Wavelengths.
    delta_perf1 : numpy.ndarray
        Change in transmission/reflection from nominal case [%] for first histogram .
    delta_perf2 : numpy.ndarray
        Change in transmission/reflection from nominal case [%] for second histogram.
    outputfolder : str
        Name of output folder for plots.

    Returns
    -------
    None.

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(2,1,figsize=(10,6))
    for delta_perf_ind,delta_perf in enumerate([delta_perf1,delta_perf2]):
    
        bin_size = 0.1 # binsize
        n_bins = np.arange(delta_perf.min(), delta_perf.max() + bin_size, bin_size) # number of bins
        
        # Create a figure and axes
        cm_used = plt.cm.get_cmap('nipy_spectral')
    
        # Iterate over wl values
        for i in range(len(wl)):
            print(i,'/',len(wl)-1)
            delta_perf_sample = delta_perf[:, i]
            # remove any non-finite values 
            delta_perf_sample = delta_perf_sample[np.isfinite(delta_perf_sample)]
            # the histogram and bin edges 
            hist, bin_edges = np.histogram(delta_perf_sample, bins=n_bins)
            # calculate the density as a percentage of the total points 
            density = hist/len(delta_perf) * 100# Calculate the density as a percentage of the total points
            density_value = np.array(density.tolist())  # Density values for current wl value
            # Mask for non-zero frequencies            
            nonzero_mask = np.where(density_value > 0)[0]  # Mask for non-zero frequencies
            x_nonzero = np.full((len(bin_edges[:-1][nonzero_mask])), wl[i])
            y_nonzero = bin_edges[:-1][nonzero_mask]
            density_values = density_value[nonzero_mask]
            
            # color normalization with the minimum and maximum density values.
            norm = mcolors.Normalize(vmin=min(density_values), vmax=max(density_values))  # Normalize the density values
            # create scatter plot 
            sc = ax[delta_perf_ind].scatter(x_nonzero, y_nonzero, c=density_values, cmap=cm_used, norm=norm, marker='s')
        
        # add a colorbar 
        cb = plt.colorbar(sc, format='%1.0f', ax=ax[delta_perf_ind])
        cb.set_label('Density[%]',rotation=90)
        
        ax[delta_perf_ind].set_ylabel(r'$ \Delta T $ [%]')
        ax[delta_perf_ind].set_xlabel('$\lambda$ [nm]')
        ax[delta_perf_ind].set_xlim((min(wl),max(wl)))
        ax[delta_perf_ind].axhline(y=0, color='white', linestyle='-',lw=3)
        ax[delta_perf_ind].axhline(y=0, color='black', linestyle='-.',lw=1.5)
        ax[delta_perf_ind].axvline(x=1950, color='white', linestyle='-',lw=3)
        ax[delta_perf_ind].axvline(x=1950, color='deeppink', linestyle='-',lw=1.5)
        ax[delta_perf_ind].set(xlim=(min(wl),max(wl)))
    
    for n, axes in enumerate(ax):
        axes.text(0.05, 0.9, '('+string.ascii_uppercase[n]+')', transform=axes.transAxes, 
                size=12, weight='bold')
    
    plt.tight_layout()
    fig.savefig(f"{outputfolder}/rand_sys_dep_error_delta_t.png",dpi=600)





