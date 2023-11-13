#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building the array of indices of a given coating on the dichroic substrate

"""

import numpy as np

def multi_layout(wl,n_H,n_L,n_incident,n_exit,layout_sequence):
    """
    Creating a 2D array of the wavelength-dependent refractive indices of
    a coating and its bouding media.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    n_H : numpy.ndarray
        Wavelength-dependent refractive index of high-index material.
    n_L : numpy.ndarray
        Wavelength-dependent refractive index of low-index material.
    n_incident : numpy.ndarray
        Wavelength-dependent refractive index of incident medium.
    n_exit : numpy.ndarray
        Wavelength-dependent refractive index of exit medium.
    layout_sequence : list
        List of L and H to represent order of coating layers from incident 
        to exiting medium.

    Returns
    -------
    n_stack : numpy.ndarray
        Stacked 2d array of wavelength-dependent refractive indices in the 
        order of n_incident - n_coating - n_exit.

    """
    
    n_coating_list = []
    for eta_type in layout_sequence:
        if eta_type == 'H':
            n_coating_list.append(n_H[:,None])
        elif eta_type == 'L':
            n_coating_list.append(n_L[:,None])
    n_coating = np.hstack(n_coating_list)
    n_stack = np.hstack((n_incident[:,None],n_coating,n_exit[:,None]))
    
    return n_stack