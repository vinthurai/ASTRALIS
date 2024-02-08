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
        phase_r : numpy.ndarray
            Phase of reflected light as a function of wavelength [degrees].
        phase_t : numpy.ndarray
            Phase of transmitted light as a function of wavelength [degrees].

        """
               
        if polar == 'unpolarised':
            
            # find spectral performance for 'TE' and 'TM' with absorption implemented
            te_polar_refl,te_polar_transm,output_theta,te_phase_r,te_phase_t = transmline(wl,n,phys_l,theta,'TE')
            tm_polar_refl,tm_polar_transm,_,_,_ = transmline(wl,n,phys_l,theta,'TM')

            # average results
            refl = (te_polar_refl+tm_polar_refl)/2
            transm = (te_polar_transm+tm_polar_transm)/2

            # ultimately the phase change is required, so either of the polarisations will do
            phase_r = te_phase_r
            phase_t = te_phase_t
            
        else:
            
            refl,transm,output_theta,phase_r,phase_t = transmline(wl,n,phys_l,theta,polar)
            
            refl = refl
            transm = transm
            
        return refl,transm,output_theta,phase_r,phase_t




def combine_spectrum(refl_front,relf_phase_front,transm_front,transm_phase_front,
                     transm_bbar,transm_phase_bbar,substrate_attentuation,transm_phase_substrate):
        """
        Combining the spectrum of the front and rear coatings to find the 
        overall transmission and reflection as well as the phases of the 
        transmitted and reflected light.

        Parameters
        ----------
        refl_front : numpy.ndarray
            Reflectance of front coating.
        relf_phase_front : numpy.ndarray
            Phase change of light reflected by front coating [degrees].
        transm_front : numpy.ndarray
            Transmittance of front coating.
        transm_phase_front : numpy.ndarray
            Phase change of light transmitted through front coating [degrees].
        transm_bbar : numpy.ndarray
            Transmittance of BBAR coating.
        transm_phase_bbar : numpy.ndarray
            Phase change of light transmitted through BBAR coating [degrees].
        substrate_attentuation : numpy.ndarray
            Exponential decay of substrate absorption.
        transm_phase_substrate : numpy.ndarray
            Phase change of light transmitted through substrate [degrees].

        Returns
        -------
        combined_spectrum_refl : numpy.ndarray
            Overall reflectance of dichroic [%].
        combined_spectrum_transm : numpy.ndarray
            Overall transmittance of dichroic [%].
        combined_refl_phase : numpy.ndarray
            Overall phase change of reflected light [degrees].
        combined_transm_phase : numpy.ndarray
            Overall phase change of transmitted light [degrees].
    
        """

        # ampltitude of overall transmitted beam 
        combined_transm = (((transm_front)/100) * ((transm_bbar)/100)) * (substrate_attentuation)
        combined_transm = np.real(combined_transm) * 100
        # amplitude of overall reflected beam 
        combined_refl = np.real(refl_front)
        
        # phase of transmitted beam
        # summing the phase changes of light transmitted through the front coating, the substrate and the bbar coating
        total_transm_phase = transm_phase_front + transm_phase_bbar
        # performing the modulo opteration to ensure value remains within 0 to 360 degrees
        combined_transm_phase = np.mod(total_transm_phase[0,:],360)
        
        # phase of reflected beam
        combined_refl_phase = relf_phase_front[0,:]
                        
        return combined_refl,combined_transm,combined_refl_phase,combined_transm_phase

