#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimating the magnitude of the expected ghost of the dichroic

"""
import numpy as np

def predicted_ghost(transm_front,refl_bbar,substrate_attentuation):
    """
    Calculating the expected amplitude of ghosting reflections

    Parameters
    ----------
    transm_front : numpy.ndarray
        Transmittance of front coating.
    refl_bbar : numpy.ndarray
        Reflectance of BBAR coating.
    substrate_attentuation : numpy.ndarray
        Exponential decay of substrate.

    Returns
    -------
    ghosting_refl : numpy.ndarray
        Amplitude of ghosting reflections [%].

    """
    
    ghosting_refl = np.empty_like(transm_front)
        
    for ray_i in range(np.shape(transm_front)[0]):
        ghosting_refl[ray_i,:] = ( ((transm_front[ray_i,:]/100)**2) * (refl_bbar[ray_i,:]/100) * (substrate_attentuation[ray_i,:])**2) *100
    
    return ghosting_refl

