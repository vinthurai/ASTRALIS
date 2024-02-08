#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to model the varied thicknesses of a coating for different deposition models,
for unlocalised points across the dichroic surface
"""


from uncertainty_tools.thickness import systematic_thickness_error


def lsf_errors(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar,sys_dep_error):
    """
    Generate a low-spatial frequency error in the form of systematic error in 
    the thicknesses of all coatings.

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

    Returns
    -------
    sys_dep_r : numpy.ndarray
        Reflectance of altered coating thicknesses [%].
    sys_dep_t : numpy.ndarray
        Transmittance of altered coating thicknesses [%].
    sys_dep_r_phase : numpy.ndarray
        Phase of reflected beam using altered thicknesses [degrees].
    sys_dep_t_phase : numpy.ndarray
        Phase of transmitted beam using altered thicknesses [degrees].
 
    """
 
     
    # Generate a systematic error of +error% in all layers & resulting change in performance
    systematic_type = 'surplus'
    # Find spectral performance
    sys_dep_r,sys_dep_t,_,sys_dep_r_phase,sys_dep_t_phase,_,_ = systematic_thickness_error(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar,sys_dep_error,systematic_type)

    return sys_dep_r,sys_dep_t,sys_dep_r_phase,sys_dep_t_phase




