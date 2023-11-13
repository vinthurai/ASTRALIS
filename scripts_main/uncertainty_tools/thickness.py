#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate random and systematic errors of coating thicknesses
"""

import numpy as np
from tlm_model.analysis import spectral_analysis


def random_thickness_error(phys_l_front,phys_l_bbar,n_sample,error_val,error_layers):
    """
    Generate random thicknesses within a given percentage error from a normal distribution.

    Parameters
    ----------
    phys_l_front : numpy.ndarray
        Physical thicknesses of front coating layers.
    phys_l_bbar : numpy.ndarray
        Physical thicknesses of BBAR coating layers.
    n_sample : int
        Number of randomly generated thickness errors.
    error_val : float
        Relative deposition uncertainty of machine [%].
    error_layers : str
        Layers which are affected by a deposition error, 
        if 'all', apply error to all layers

    Returns
    -------
    phys_l_front_rand : numpy.ndarray
        Thicknesses of front coating layers after applying random errors.
    phys_l_rear_rand : numpy.ndarray
        Thicknesses of BBAR coating layers after applying random errors.

    """

    # if all layers are considered to have errors
    if error_layers == 'all':
        error_layers_front = np.arange(len(phys_l_front))
        error_layers_rear = np.arange(len(phys_l_bbar))
    print('Generating',n_sample,'random errors in thicknesses of layers')

    depo_uncert_front = np.zeros_like(phys_l_front)
    depo_uncert_front[error_layers_front] = phys_l_front[error_layers_front]*error_val # deposition uncertainty of sputtering machine is 0.5%
    depo_uncert_rear = np.zeros_like(phys_l_bbar)
    depo_uncert_rear[error_layers_rear] = phys_l_bbar[error_layers_rear]*error_val # deposition uncertainty of sputtering machine is 0.5%
    
    np.random.seed(n_sample) # seed for reproducibility
    # between 0 to 1 sigma from a normal distribution 
    rand_sigma_front = np.random.normal(0, 1, (n_sample, len(error_layers_front)))
    rand_sigma_rear = np.random.normal(0, 1, (n_sample, len(error_layers_rear)))

    # thicknesses of coating layers after applying errors
    phys_l_front_rand = phys_l_front +(rand_sigma_front*depo_uncert_front)
    phys_l_rear_rand = phys_l_bbar + (rand_sigma_rear*depo_uncert_rear)
    
    return phys_l_front_rand,phys_l_rear_rand



    
def random_thickness_errors_spectrum(wl,n_front,n_substrate,n_bbar,phys_l_error_front,
                                     phys_l_substrate,phys_l_error_rear,theta,polar):
    """
    Calculate the spectral performance for randomly altered thicknesses.


    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    n_front : numpy.ndarray
        Complex refractive indices of front coating and its bounding media.
    n_substrate : numpy.ndarray
        Complex refractive indices of substrate and its bounding media.
    n_bbar : numpy.ndarray
        Complex refractive indices of BBAR coating.
    phys_l_error_front : numpy.ndarray
        Thicknesses of front coating layers after applying random errors.
    phys_l_substrate : numpy.ndarray
        Physical thickness of substrate.
    phys_l_error_rear : numpy.ndarray
        Thicknesses of BBAR coating layers after applying random errors.
    theta : numpy.ndarray
        Angle of incidence [degrees].
    polar : str
        Linear polarisation type – 'TE', 'TM' or 'unpolarised'.

    Returns
    -------
    rand_dep_r : numpy.ndarray
        Reflectance using randomly altered thicknesses [%].
    rand_dep_t : numpy.ndarray
        Transmittance using randomly altered thicknesses [%].

    """
    
    n_sample=len(phys_l_error_front)
    rand_dep_r = np.zeros((n_sample,len(wl))) 
    rand_dep_t = np.zeros((n_sample,len(wl)))
    
    # calculate the spectral performance after random thickness errors have been applied
    for m in np.arange(0,n_sample):
      print(m,'/',n_sample-1)
      rand_dep_r[m,:],rand_dep_t[m,:],_ = spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_error_front[m,:],phys_l_substrate,phys_l_error_rear[m,:],theta,polar)

    print('Generating',n_sample,'random errors in thicknesses complete!')
    return rand_dep_r,rand_dep_t



def systematic_thickness_error(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar,sys_dep_error,systematic_type):
    """
    Find the spectral performance due to a systematic error in the 
    coating thicknesses of all layers.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    n_front : numpy.ndarray
        Complex refractive indices of front coating and its bounding media.
    n_substrate : numpy.ndarray
        Complex refractive indices of substrate and its bounding media.
    n_bbar : numpy.ndarray
        Complex refractive indices of BBAR coating.
    phys_l_front : numpy.ndarray
        Physical thicknesses of front coating layers.
    phys_l_substrate : numpy.ndarray
        Physical thickness of substrate.
    phys_l_bbar : numpy.ndarray
        Physical thicknesses of BBAR coating layers.
    theta : numpy.ndarray
        Angle of incidence [degrees].
    polar : str
        Linear polarisation type – 'TE', 'TM' or 'unpolarised'.
    sys_dep_error : float
        Systematic error on thickness [%].
    systematic_type : str
        Either 'surplus' or 'deficiency' indicating whether the 
        systematic error is added added or substrated, respectively.

    Returns
    -------
    sys_dep_r : numpy.ndarray
        Reflectance of altered coating thicknesses [%].
    sys_dep_t : numpy.ndarray
        Transmittance of altered coating thicknesses [%].
    sys_dep_ghost : numpy.ndarray
        Amplitude of ghosting reflections of altered coating thicknesses [%].
    phys_l_error_front : numpy.ndarray
        Thicknesses of front coating layers with systematic error.
    phys_l_error_rear : numpy.ndarray
        Thicknesses of BBAR coating layers with systematic error.
    
    """

    if systematic_type == 'deficiency':
        phys_l_error_front = phys_l_front - phys_l_front*sys_dep_error 
        phys_l_error_rear = phys_l_bbar - phys_l_bbar*sys_dep_error

    elif systematic_type == 'surplus':
        phys_l_error_front = phys_l_front + phys_l_front*sys_dep_error 
        phys_l_error_rear = phys_l_bbar + phys_l_bbar*sys_dep_error
    
    sys_dep_r,sys_dep_t,sys_dep_ghost = spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_error_front,phys_l_substrate,phys_l_error_rear,theta,polar)

    return sys_dep_r,sys_dep_t,sys_dep_ghost,phys_l_error_front,phys_l_error_rear


