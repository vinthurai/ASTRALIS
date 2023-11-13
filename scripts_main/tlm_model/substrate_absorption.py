#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculating the absorption due to the substrate of the dichroic.

"""


# Importing Libraries
import numpy as np

def substrate_absorp(wl,n_substrate,thickness,angle):
    """
    Calculates the absorption attenuation due to the substrate 
    
    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    n_complex : numpy.ndarray
        Complex refractive index of substrate material.
    thickness : numpy.ndarray
        Thickness of substrate.
    angle : numpy.ndarray
        Angle of incidence to substrate.

    Returns
    -------
    absorption_val : numpy.ndarray
        Exponential decay of substrate absorption given by exp(- alpha x).

    """
        
    alpha = 4* np.pi * abs(np.imag(n_substrate))/wl # absorption coeff #abs() used to make sure we use k, not -k
    absorption_val = np.exp(-1*alpha*thickness) # exponential decay function
        
    return absorption_val
