#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelling the behaviour of the substrate of the dichroic.

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
    n_substrate : numpy.ndarray
        Complex refractive index of substrate material.
    thickness : numpy.ndarray
        Thickness of substrate.
    angle : numpy.ndarray
        Angle of incidence to substrate [degrees].

    Returns
    -------
    absorption_val : numpy.ndarray
        Exponential decay of substrate absorption given by exp(- alpha x).

    """
    
    absorption_subs = []
    
    alpha = 4* np.pi * abs(np.imag(n_substrate))/wl # absorption coeff #abs() used to make sure we use k, not -k
    
    for theta_i in angle:
        eff_opl = thickness / np.cos(np.deg2rad(theta_i))# effective optical path length
        absorption_val = np.exp(-1*alpha*eff_opl) # exponential decay function
        absorption_subs.append(absorption_val)
    
    absorption_subs = np.array(absorption_subs)
    
    return absorption_subs



def substrate_phase(wl,n_substrate,thickness,angle):
    """
    Calculates the phase change of the light while travelling through 
    the substrate.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    n_substrate : numpy.ndarray
        Complex refractive index of substrate material.
    thickness : numpy.ndarray
        Thickness of substrate.
    angle : numpy.ndarray
        Angle of incidence to substrate [degrees].

    Returns
    -------
    phase_subs : numpy.ndarray
        One-pass phase change of light transmitted through substrate [degrees].

    """

    phase_subs = []
    # one-pass phase change
    for theta_i in angle:
        phase_subs_val = (2*np.pi/wl) * np.real(n_substrate) * thickness / np.cos(np.deg2rad(theta_i))
        phase_subs.append(phase_subs_val)
        
    return phase_subs

