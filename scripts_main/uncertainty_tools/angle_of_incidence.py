#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an ofset in the angle of incidence of the collimated beam.
"""

from tlm_model.analysis import spectral_analysis


def angle_error(wl,sys_theta_error,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar):
    """
    Find the spectral performance due to a systematic error in the 
    angle of incidence of the collimated beam incident on the dichroic.    

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    sys_theta_error : float
        Value of systematic error in angle of incidence [degrees].
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

    Returns
    -------
    sys_theta_r : numpy.ndarray
        Reflectance after altered angle of incidence [%].
    sys_theta_t : numpy.ndarray
        Transmittance after altered angle of incidence [%].
    sys_theta_ghost : numpy.ndarray
        Amplitude of ghosting reflections after altered angle of incidence [%].

    """
    
    # Apply error
    theta_error = theta + sys_theta_error
    
    #Calculating the nominal spectral performance 
    sys_theta_r,sys_theta_t,sys_theta_ghost = spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta_error,polar)
    
    return sys_theta_r,sys_theta_t,sys_theta_ghost

