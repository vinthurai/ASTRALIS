#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysing the spectral performance of a dichroic.

"""

import numpy as np
from tlm_model.spectrum import linear_polar_transmline
from tlm_model.spectrum import combine_spectrum
from tlm_model.substrate import substrate_absorp, substrate_phase
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
    combined_refl_phase : numpy.ndarray
        Overall phase of reflected light [degrees].
    combined_transm_phase : numpy.ndarray
        Overall phase of transmitted light [degrees].
    """
    
    # Spectral performance and phase – front coating
    refl_front,transm_front,output_theta,relf_phase_front,transm_phase_front = linear_polar_transmline(wl,n_front,phys_l_front,theta,polar)
    # Spectral performance and phase – BBAR coating
    refl_bbar,transm_bbar,_,_,transm_phase_bbar = linear_polar_transmline(wl,n_bbar,phys_l_bbar,output_theta,polar)
    
    # Exponential decay of substrate absorption
    substrate_attentuation = substrate_absorp(wl,n_substrate,phys_l_substrate,output_theta)
    
    # Phase change of light transmitted through substrate
    transm_phase_substrate = substrate_phase(wl,n_substrate,phys_l_substrate,output_theta)
    
    # Overall spectral performance and phase    
    combined_spectrum_refl,combined_spectrum_transm,combined_refl_phase,combined_transm_phase = combine_spectrum(refl_front,relf_phase_front,transm_front,transm_phase_front,transm_bbar,transm_phase_bbar,substrate_attentuation,transm_phase_substrate)

    # Amplitude of ghosting reflections 
    ghosting_refl = predicted_ghost(transm_front,refl_bbar,substrate_attentuation)
    
    return combined_spectrum_refl,combined_spectrum_transm,ghosting_refl,combined_refl_phase,combined_transm_phase
    
    
    
def av_intensity(spectrum,n_rays):
    """
    Average intensities of rays to find intensity of overall beam 
    by combining rays incoherently.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Intensity of ray [%].
    n_rays : int
        Number of rays to be combined.

    Returns
    -------
    av_spectrum : TYPE
        Averaged intensity [%].

    """
    
    av_spectrum = np.sum(spectrum,axis=0)/n_rays
    return av_spectrum
    
