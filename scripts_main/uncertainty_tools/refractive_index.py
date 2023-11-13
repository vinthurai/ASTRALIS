#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate systematic error in refractive index of material
"""

from tlm_model.analysis import spectral_analysis
from general_tools.multilayer_structure import multi_layout
from general_tools.refractive_index import refr_index


def refr_ind_error(wl,n_sys_error,error_material,config_multilayout,phys_l_front,
                   phys_l_substrate,phys_l_bbar,temp,theta,polar):
    """
    Find the spectral performance due to a systematic error in the 
    refractive index in one material of the coating layers.    

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    n_sys_error : float
        Value of systematic error in index [%].
    error_material : str
        Material of which index has systematic error.
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
    sys_index_r : numpy.ndarray
        Reflectance after altered refractive indices [%].
    sys_index_t : numpy.ndarray
        Transmittance after altered refractive indices [%].
    sys_index_ghost : numpy.ndarray
        Amplitude of ghosting reflections after altered refractive indices [%].

    """
   
    # Multilayer structure
    material_type = 'component'

    m_dict = {
        'n_H': {'name':'refr_index_h'},
        'n_L': {'name':'refr_index_l'},
        'n_incident': {'name':'refr_index_incident'},
        'n_substrate': {'name':'refr_index_substrate'},
        'n_incident_rear': {'name':'refr_index_substrate'},
        'n_substrate_rear': {'name':'refr_index_incident'}
    }    
    
    for m_kind,m_info in m_dict.items():
        # name of material
        material = config_multilayout[m_info['name']]
        # refractive index of material
        refr_index_n = refr_index(material,material_type,wl,temp)

        # if material is the chosen material with uncertainty, apply error
        if material['name'] == error_material:
            # systematic errors are negative since indices of thin films tend to be of lower index than their bulk equivalent
            m_dict[m_kind]['value'] = refr_index_n + n_sys_error*refr_index_n
        else: 
            m_dict[m_kind]['value'] = refr_index_n
    
    # Front Coating Refractive indices 
    layout_sequence_front = config_multilayout['refr_index_layout_front_coating_only']['sequence']
    n_front_error = multi_layout(wl,m_dict['n_H']['value'],m_dict['n_L']['value'],m_dict['n_incident']['value'],m_dict['n_substrate']['value'],layout_sequence_front)
    # BBAR Coating Refractive indices
    layout_sequence_rear = config_multilayout['refr_index_layout_rear_coating_only']['sequence']
    n_bbar_error = multi_layout(wl,m_dict['n_H']['value'],m_dict['n_L']['value'],m_dict['n_incident_rear']['value'],m_dict['n_substrate_rear']['value'],layout_sequence_rear)
    # Substrate Refractuve indices
    n_substrate_error = m_dict['n_substrate']['value']
    # Spectral performance
    sys_index_r,sys_index_t,sys_index_ghost = spectral_analysis(wl,n_front_error,n_substrate_error,n_bbar_error,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)

    return sys_index_r,sys_index_t,sys_index_ghost
    

