#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions that carry out combinations of spectra to obtain the overall 
reflectance and transmittance as a function wavelength of a dichroic

"""


# Importing Libraries
import numpy as np
from tlm_model.transmission_line import transmline
    
def linear_polar_transmline(wl,n,phys_l,theta,polar):
        """
        In the case of unpolarised light, the p- and s-polarizations are 
        averaged using the transmline() funtion.        

        Parameters
        ----------
        wl : numpy.ndarray
            Wavelengths.
        n : numpy.ndarray
            Complex refractive indices.
        phys_l : numpy.ndarray
            Physical thicknesses of coating layers.
        theta : numpy.ndarray
            Angle of incidence [degrees].
        polar : str
            Linear polarisation type – 'TE', 'TM' or 'unpolarised'.

        Returns
        -------
        refl : numpy.ndarray
            Reflectance of coating[%].
        transm : numpy.ndarray
            Transmittance of coating[%].
        output_theta : numpy.ndarray
            Angle of outgoing ray [degrees].

        """
               
        if polar == 'unpolarised':
            
            # find spectral performance for 'TE' and 'TM' with absorption implemented
            te_polar_refl,te_polar_transm,output_theta = transmline(wl,n,phys_l,theta,'TE')
            tm_polar_refl,tm_polar_transm,_ = transmline(wl,n,phys_l,theta,'TM')

            # average results
            refl = (te_polar_refl+tm_polar_refl)/2
            transm = (te_polar_transm+tm_polar_transm)/2
            
        else:
            
            refl,transm,output_theta = transmline(wl,n,phys_l,theta,polar)
            
            refl = refl
            transm = transm
            
        return refl,transm,output_theta
    

def combine_spectrum(refl_front,transm_front,transm_bbar,substrate_attentuation):
        """
        Combining the spectrum of the front and rear coatings to find the 
        overall transmission and reflection.

        Parameters
        ----------
        refl_front : numpy.ndarray
            Reflectance of front coating.
        transm_front : numpy.ndarray
            Transmittance of front coating.
        transm_bbar : numpy.ndarray
            Transmittance of BBAR coating.
        substrate_attentuation : numpy.ndarray
            Exponential decay of substrate absorption.

        Returns
        -------
        combined_spectrum_refl : numpy.ndarray
            Overall reflectance of dichroic [%].
        combined_spectrum_transm : numpy.ndarray
            Overall transmittance of dichroic [%].

        """

        combined_transm = (((transm_front)/100) * ((transm_bbar)/100)) * (substrate_attentuation)
        combined_transm = np.real(np.squeeze(combined_transm)) * 100

        combined_refl = np.real(refl_front)
        
        return combined_refl,combined_transm

