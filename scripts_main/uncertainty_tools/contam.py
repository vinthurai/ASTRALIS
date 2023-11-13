#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelling the performance of dichroics with molecular contamination 
"""

# Local imports
import numpy as np
from general_tools.refractive_index import refr_index
from tlm_model.analysis import spectral_analysis


def mol_contam_analysis(wl,n_incident,n_front,n_substrate,n_bbar,phys_l_front,
                        phys_l_substrate,phys_l_bbar,temp,theta,polar,contam_config):
    """

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    n_incident : TYPE
        DESCRIPTION.
    n_front : numpy.ndarray
        Complex refractive indices of front coating.
    n_substrate : numpy.ndarray
        Complex refractive indices of substrate.
    n_bbar : numpy.ndarray
        Complex refractive indices of BBAR coating.
    phys_l_front : numpy.ndarray
        Physical thicknesses of front coating layers.
    phys_l_substrate : numpy.ndarray
        Physical thickness of substrate.
    phys_l_bbar : numpy.ndarray
        Physical thicknesses of BBAR coating layers.
    temp : TYPE
        DESCRIPTION.
    theta : numpy.ndarray
        Angle of incidence [degrees].
    polar : str
        Linear polarisation type – 'TE', 'TM' or 'unpolarised'.
    contam_config : dict
        Congifuration information for contamination recipes from config file.

    Returns
    -------
    refl : numpy.ndarray
        Overall reflectance of dichroic with molecular contamination[%].
    transm : numpy.ndarray
        Overall transmittance of dichroic with molecular contamination[%].

    """

    # Refractive indices material type
    material_type = 'contamination'
    n_incident_rear = n_substrate
    n_substrate_rear = n_incident
    
    n_c_list_f,n_c_list_r = [],[]
    phys_l_c_list_f,phys_l_c_list_r = [],[]
    
    # Loop through different layers of contamination for the given scenario 
    for c_material_f , c_material_r in zip(contam_config['front'],contam_config['bbar']):
        
        c_thickn_f = c_material_f['thickness']
        c_thickn_r = c_material_r['thickness']

        # Index of contamination layer on front coating
        n_c_list_f.append(refr_index(c_material_f,material_type,wl,temp))
        # Thickness of contamination layer on front coating
        phys_l_c_list_f.append(c_thickn_f)
    
        # Index of contamination layer on bbar coating
        n_c_list_r.append(refr_index(c_material_r,material_type,wl,temp))
        # Thickness of contamination layer on bbar coating
        phys_l_c_list_r.append(c_thickn_r)
    
    # Array of indices of the front coating including contamination layers
    n_c_list_f = np.array(n_c_list_f).T
    n_front_contam = np.hstack((n_incident[:,None],n_c_list_f,n_front,n_substrate[:,None]))
    
    # Array of indices of the bbar coating including contamination layers
    n_c_list_r = np.array(n_c_list_r).T
    n_bbar_contam = np.hstack((n_incident_rear[:,None],n_bbar,n_c_list_r,n_substrate_rear[:,None]))
        
    # Array of thicknesses of the front coating including contamination layers
    phys_l_front_contam = np.concatenate((phys_l_c_list_f, phys_l_front))
    # Array of thicknesses of the bbar coating including contamination layers
    phys_l_bbar_contam = np.concatenate((phys_l_bbar,phys_l_c_list_r))
             
    # Calculate the spectral performance of the contaminated dichroic
    refl,transm, _ = spectral_analysis(wl,n_front_contam,n_substrate,n_bbar_contam,phys_l_front_contam,
                                       phys_l_substrate,phys_l_bbar_contam,theta,polar)
    
    return refl,transm
