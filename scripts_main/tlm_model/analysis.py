#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysing the spectral performance of a dichroic.

"""


from tlm_model.spectrum import linear_polar_transmline
from tlm_model.spectrum import combine_spectrum
from tlm_model.substrate_absorption import substrate_absorp
from tlm_model.ghosts import predicted_ghost

def spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar):
    """
    Analysing the spectral performance of a dichroic.

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

    Returns
    -------
    combined_spectrum_refl : numpy.ndarray
        Overall reflectance of dichroic [%].
    combined_spectrum_transm : numpy.ndarray
        Overall transmittance of dichroic [%].
    ghosting_refl : numpy.ndarray
        Amplitude of ghosting reflections [%].

    """
    
    # Spectral performance of the front coating
    refl_front,transm_front,output_theta = linear_polar_transmline(wl,n_front,phys_l_front,theta,polar)
    
    # Spectral performance of the BBAR coating
    refl_bbar,transm_bbar,_ = linear_polar_transmline(wl,n_bbar,phys_l_bbar,output_theta,polar)
    
    # Exponential decay of substrate absorption
    substrate_attentuation = substrate_absorp(wl,n_substrate,phys_l_substrate,output_theta)
    
    # Overall spectral performance of dichroic
    combined_spectrum_refl,combined_spectrum_transm = combine_spectrum(refl_front,transm_front,transm_bbar,substrate_attentuation)
    
    # Amplitude of ghosting reflections 
    ghosting_refl = predicted_ghost(transm_front,refl_bbar,substrate_attentuation)
    
    return combined_spectrum_refl,combined_spectrum_transm,ghosting_refl


