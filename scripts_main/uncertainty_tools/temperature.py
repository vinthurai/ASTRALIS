#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate systematic error in operational temperature of dichroic.
"""

from tlm_model.analysis import spectral_analysis
from general_tools.multilayer_structure import multi_layout
from general_tools.refractive_index import refr_index


def temp_error(wl,temp_sys_error,config_multilayout,phys_l_front,phys_l_substrate,phys_l_bbar,temp,theta,polar):
    """
    Find the spectral performance due to a systematic error in the 
    operation temperature of the dichroic. 


    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    temp_sys_error : float
        Value of systematic error in temperature [K].
    config_multilayout : dict
        Configuration of multilayer coating recipe (including index and thickness).
    phys_l_front : numpy.ndarray
        Physical thicknesses of front coating layers.
    phys_l_substrate : numpy.ndarray
        Physical thickness of substrate.
    phys_l_bbar : numpy.ndarray
        Physical thicknesses of BBAR coating layers.
    temp : float
        Operational temperature of dichroic [K].
    theta : numpy.ndarray
        Angle of incidence [degrees].
    polar : str
        Linear polarisation type – 'TE', 'TM' or 'unpolarised'.

    Returns
    -------
    sys_temp_r : numpy.ndarray
        Reflectance after altered operational temperature [%].
    sys_temp_t : numpy.ndarray
        Transmittance after altered operational temperature [%].
    sys_temp_ghost : numpy.ndarray
        Amplitude of ghosting reflections after altered operational temperature [%].

    """
    
    # Apply error
    temp_error = temp+temp_sys_error
    
    # Multilayer structure
    material_type = 'component'
    n_H = refr_index(config_multilayout['refr_index_h'],material_type,wl,temp_error)
    n_L = refr_index(config_multilayout['refr_index_l'],material_type,wl,temp_error)

    # Front Coating Refractive indices 
    n_incident = refr_index(config_multilayout['refr_index_incident'],material_type,wl,temp_error)
    n_substrate = refr_index(config_multilayout['refr_index_substrate'],material_type,wl,temp_error)
    layout_sequence_front = config_multilayout['refr_index_layout_front_coating_only']['sequence']
    n_front = multi_layout(wl,n_H,n_L,n_incident,n_substrate,layout_sequence_front)

    # BBAR Coating Refractive indices
    n_incident_rear = refr_index(config_multilayout['refr_index_substrate'],material_type,wl,temp_error)
    n_substrate_rear = refr_index(config_multilayout['refr_index_incident'],material_type,wl,temp_error)
    layout_sequence_rear = config_multilayout['refr_index_layout_rear_coating_only']['sequence']
    n_bbar = multi_layout(wl,n_H,n_L,n_incident_rear,n_substrate_rear,layout_sequence_rear)

    #Calculating the nominal spectral performance 
    sys_temp_r,sys_temp_t,sys_temp_ghost = spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
    
    return sys_temp_r,sys_temp_t,sys_temp_ghost


