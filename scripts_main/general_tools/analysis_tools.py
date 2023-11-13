#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Studying change in R and T, the shift in transition wavelengths due to systematics
"""


# Importing Libraries
import numpy as np

def change_spectral(nominal_refl,nominal_transm,error_refl,error_transm):
    """
    Calculate the change in spectral performance from the nominal case 
    due to uncertainties.

    Parameters
    ----------
    nominal_refl : numpy.ndarray
        Nominal reflectance of dichroic [%].
    nominal_transm : numpy.ndarray
        Nominal transmittance of dichroic [%].
    error_refl : numpy.ndarray
        Reflectance of dichroic once error has been applied [%].
    error_transm : numpy.ndarray
        Transmittance of dichroic once error has been applied [%].

    Returns
    -------
    d_refl : numpy.ndarray
        Change in reflectance [%].
    d_transm : numpy.ndarray
        Change in transmittance [%].

    """
    
    d_refl = error_refl - nominal_refl
    d_transm = error_transm - nominal_transm
    
    return d_refl,d_transm
    


def trans_wl_shift(wl, transition_wl, m, spect):
    """
    Shift in transition wavelength from nominal wavelength position

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    spect : numpy.ndarray
        Spectral perfomance (reflectance / transmittance [%]).
    m : float
        Gradient of transition slope.

    Returns
    -------
    d_lambda : float
        New transition wavelength position.

    """
        
    # Find transition wavelength 
    transition_y = 50 # reflection / transmission at point of transition wavelength
    # transition_wl = np.interp(transition_y,spect[np.where((spect<=51)&(spect>=49))],
    #                           wl[np.where((spect<=51)&(spect>=49))])
        
    # Find shift in transition wavelength
    d_T = transition_y - np.interp(transition_wl,wl,spect)
    d_lambda = d_T/m
    
    return d_lambda



def weighted_lsr(wl,model_perf,bc_wl,bc_perf):
    """
    Calculate weighted least-square residual for Monte Carlo.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    model_perf : numpy.ndarray
        Simulated performance [%].
    bc_wl : numpy.ndarray
        Wavelength sampling of boundary conditions.
    bc_perf : numpy.ndarray
        Boundary conditions of performance [%].

    Returns
    -------
    lsr : float
        Weighted least-square residual between simulated and boundary conditions.

    """
    # find simulated performance at boundary condition wavelength points
    model_perf_interp = np.interp(bc_wl, wl, model_perf)
    # residual between simulated performance and boundary condition performance
    delta = abs(model_perf_interp - bc_perf)/100 # converting from percentage to decimal
    k = 10 # 'steepness' of weighting slope
    w = 1/((1-delta)**k) # weighting
    lsr = np.sum(w*(delta**2)) # weighted least square residual
    return lsr
