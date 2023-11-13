#!/usr/bin/env python3

"""
Code used to create various plots in run_dichroic_model.py for visualisation
for visualisation of the random generated errors

Graphs here include:
    Rainbow scatter plot showing distributions of generated random errors
    Scatter plots with histrograms showing the sensitivity of each layer to the generated errors
    The spectrum with grey errorbars showing the extent of the generated random errors
    Red/blue bars showing the errors generated as a function of wavelength
"""

import numpy as np
import matplotlib.pyplot as plt
import string
import matplotlib.cm as cm
import matplotlib.colors as mcolors

#%% Source for code: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_zoom_effect.html#sphx-glr-gallery-subplots-axes-and-figures-axes-zoom-effect-py
    
from matplotlib.transforms import (
    Bbox, TransformedBbox)
from mpl_toolkits.axes_grid1.inset_locator import (
    BboxPatch, BboxConnector, BboxConnectorPatch)

def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
            "clip_on": False,
        }

    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           # loc1a=3, loc2a=2, loc1b=4, loc2b=1,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           clip_on=False,
                           **prop_patches)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect01(ax1, ax2, xmin, xmax, **kwargs):
    """
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both axes will
    be marked.

    Parameters
    ----------
    ax1
        The main axes.
    ax2
        The zoomed axes.
    xmin, xmax
        The limits of the colored area in both plot axes.
    **kwargs
        Arguments passed to the patch constructor.
    """

    bbox = Bbox.from_extents(xmin, 0, xmax, 1)

    mybbox1 = TransformedBbox(bbox, ax1.get_xaxis_transform())
    mybbox2 = TransformedBbox(bbox, ax2.get_xaxis_transform())

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.1}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def contoured_2d_hist(wl,residual,outputfolder):

    bin_size = 0.1
    #  Here, the bin_size is defined, which determines the width of each bin in the histogram. 
    #  n_bins is calculated using np.arange to create an array of bin edges based on the minimum and maximum values of y and the specified bin_size.
    n_bins = np.arange(residual.min(), residual.max() + bin_size, bin_size)
    
    # Create a figure and axes
    fig, ax = plt.subplot_mosaic([["zoom1"],["main"],],constrained_layout=True,figsize=(10,6))
    cm_used = plt.cm.get_cmap('nipy_spectral')


    # Iterate over wl values
    for i in range(len(wl)):
        print(i,'/',len(wl)-1)
        #  y_sample is extracted for the current x value, 
        #  and any non-finite values are removed using np.isfinite(y_sample).
        residual_sample = residual[:, i]
        residual_sample = residual_sample[np.isfinite(residual_sample)]
        #  The histogram hist and bin edges bin_edges are computed for the filtered y_sample using np.histogram.
        hist, bin_edges = np.histogram(residual_sample, bins=n_bins)
        #  The density is calculated by dividing the histogram by the total number of data points in y 
        #  and multiplying by 100 to obtain the density as a percentage.
        density = hist/len(residual) * 100# Calculate the density as a percentage of the total points
        #  If you want the same colorbar to apply for all x values, you should use density = hist/len(y) * 100.
        #  By using density = hist/len(y) * 100, you are calculating the density values based on the entire dataset y, 
        #     regardless of the specific x value. 
        #  This ensures that the density values are normalized consistently across all x values, 
        #     allowing for a consistent color mapping and colorbar across the entire plot.
        
        #  density_values is converted to a list.
        density_value = np.array(density.tolist())  # Density values for current wl value
        
        nonzero_mask = np.where(density_value > 0)[0]  # Mask for non-zero frequencies
        x_nonzero = np.full((len(bin_edges[:-1][nonzero_mask])), wl[i])
        y_nonzero = bin_edges[:-1][nonzero_mask]
        density_values = density_value[nonzero_mask]
        
        #  A color normalization object norm is created using mcolors.Normalize with the minimum and maximum density values.
        norm = mcolors.Normalize(vmin=min(density_values), vmax=max(density_values))  # Normalize the density values
        #  that maps the density values to the range [0, 1]. 
        #  The vmin and vmax arguments specify the minimum and maximum values to which the density values will be mapped
        
        #  The scatter plot is created using ax.scatter with the x values set as a full array of the current x value, 
        #  the bin edges as the y values, the density values as the color values, 
        #  the colormap 'inferno', the normalization norm, and a square marker 's'.
        for plot_squares in ["main","zoom1"]:
            sc = ax[plot_squares].scatter(x_nonzero, y_nonzero, c=density_values, cmap=cm_used, norm=norm, marker='s')
    
    # Add a colorbar with percentage values
    # These lines create a colorbar for the scatter plot and format it to display the density values as a percentage. 
    # The label 'Density' is assigned to the colorbar.
    cb = plt.colorbar(sc, format='%1.0f', ax=ax["main"])
    cb.set_label('Density[%]',rotation=90)
    
    ax["main"].set_ylabel(r'$ \Delta T $ [%]')
    ax["main"].set_xlabel('$\lambda$ [nm]')
    ax["main"].set_xlim((min(wl),max(wl)))
    ax["main"].axhline(y=0, color='white', linestyle='-',lw=3)
    ax["main"].axhline(y=0, color='black', linestyle='-.',lw=1.5)
    ax["main"].axvline(x=1950, color='white', linestyle='-',lw=3)
    ax["main"].axvline(x=1950, color='deeppink', linestyle='-',lw=1.5)
    ax["zoom1"].axhline(y=0, color='white', linestyle='-',lw=3)
    ax["zoom1"].axhline(y=0, color='black', linestyle='-.',lw=1.5)
    ax["zoom1"].axvline(x=1950, color='white', linestyle='-',lw=3)
    ax["zoom1"].axvline(x=1950, color='deeppink', linestyle='-',lw=1.5)
    ax["zoom1"].set_ylabel(r'$ \Delta T $ [%]')
    ax["main"].set(xlim=(min(wl),max(wl)))
    ax["zoom1"].set(xlim=(1800, 2300))
    # ax["main"].set(ylim=(-9,9))
    ax["zoom1"].set(ylim=(-9,9))
    zoom_effect01(ax["zoom1"], ax["main"], 1800, 2300)
    # plt.tight_layout()
    fig.savefig(f"{outputfolder}/rand_dep_error_delta_t.png",dpi=600)


def double_contoured_2d_hist(wl,residual1,residual2,outputfolder):

    
    fig, ax = plt.subplots(2,1,figsize=(10,6))
    for residual_ind,residual in enumerate([residual1,residual2]):
    
        bin_size = 0.1
        #  Here, the bin_size is defined, which determines the width of each bin in the histogram. 
        #  n_bins is calculated using np.arange to create an array of bin edges based on the minimum and maximum values of y and the specified bin_size.
        n_bins = np.arange(residual.min(), residual.max() + bin_size, bin_size)
        
        # Create a figure and axes
        
        cm_used = plt.cm.get_cmap('nipy_spectral')
    
        # Iterate over wl values
        for i in range(len(wl)):
            print(i,'/',len(wl)-1)
            #  y_sample is extracted for the current x value, 
            #  and any non-finite values are removed using np.isfinite(y_sample).
            residual_sample = residual[:, i]
            residual_sample = residual_sample[np.isfinite(residual_sample)]
            #  The histogram hist and bin edges bin_edges are computed for the filtered y_sample using np.histogram.
            hist, bin_edges = np.histogram(residual_sample, bins=n_bins)
            #  The density is calculated by dividing the histogram by the total number of data points in y 
            #  and multiplying by 100 to obtain the density as a percentage.
            density = hist/len(residual) * 100# Calculate the density as a percentage of the total points
            #  If you want the same colorbar to apply for all x values, you should use density = hist/len(y) * 100.
            #  By using density = hist/len(y) * 100, you are calculating the density values based on the entire dataset y, 
            #     regardless of the specific x value. 
            #  This ensures that the density values are normalized consistently across all x values, 
            #     allowing for a consistent color mapping and colorbar across the entire plot.
            
            #  density_values is converted to a list.
            density_value = np.array(density.tolist())  # Density values for current wl value
            
            nonzero_mask = np.where(density_value > 0)[0]  # Mask for non-zero frequencies
            x_nonzero = np.full((len(bin_edges[:-1][nonzero_mask])), wl[i])
            y_nonzero = bin_edges[:-1][nonzero_mask]
            density_values = density_value[nonzero_mask]
            
            #  A color normalization object norm is created using mcolors.Normalize with the minimum and maximum density values.
            norm = mcolors.Normalize(vmin=min(density_values), vmax=max(density_values))  # Normalize the density values
            #  that maps the density values to the range [0, 1]. 
            #  The vmin and vmax arguments specify the minimum and maximum values to which the density values will be mapped
            
            #  The scatter plot is created using ax.scatter with the x values set as a full array of the current x value, 
            #  the bin edges as the y values, the density values as the color values, 
            #  the colormap 'inferno', the normalization norm, and a square marker 's'.
            sc = ax[residual_ind].scatter(x_nonzero, y_nonzero, c=density_values, cmap=cm_used, norm=norm, marker='s')
        
        # Add a colorbar with percentage values
        # These lines create a colorbar for the scatter plot and format it to display the density values as a percentage. 
        # The label 'Density' is assigned to the colorbar.
        cb = plt.colorbar(sc, format='%1.0f', ax=ax[residual_ind])
        cb.set_label('Density[%]',rotation=90)
        
        # ax["main"].grid()
        ax[residual_ind].set_ylabel(r'$ \Delta T $ [%]')
        ax[residual_ind].set_xlabel('$\lambda$ [nm]')
        ax[residual_ind].set_xlim((min(wl),max(wl)))
        ax[residual_ind].axhline(y=0, color='white', linestyle='-',lw=3)
        ax[residual_ind].axhline(y=0, color='black', linestyle='-.',lw=1.5)
        ax[residual_ind].axvline(x=1950, color='white', linestyle='-',lw=3)
        ax[residual_ind].axvline(x=1950, color='deeppink', linestyle='-',lw=1.5)
        ax[residual_ind].set(xlim=(min(wl),max(wl)))
    
    for n, axes in enumerate(ax):
        axes.text(0.05, 0.9, '('+string.ascii_uppercase[n]+')', transform=axes.transAxes, 
                size=12, weight='bold')
    
    plt.tight_layout()
    fig.savefig(f"{outputfolder}/rand_sys_dep_error_delta_t.png",dpi=600)





