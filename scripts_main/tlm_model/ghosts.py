#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimating the magnitude of the expected ghost of the dichroic

"""


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
    
    ghosting_refl = ( ((transm_front/100)**2) * (refl_bbar/100) * (substrate_attentuation)**2) *100
    
    return ghosting_refl

