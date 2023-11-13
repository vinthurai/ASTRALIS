#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions that carry out transmission line calculations and returns the 
reflectance and transmittance as a function wavelength

"""


# Importing Libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def transmline(wl,n,phys_l,theta,polar):
    """
    Transmission line function taking array refractive index of materials 
    and thickness arrays of materials relevant wavelength range, angle of 
    incidence in degrees and polarisation to find resulting 
    reflection/transmission% as a function of wavelength. 
    This uses the matrix method.
    
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
    refl_perc : numpy.ndarray
        Reflectance [%].
    transm_perc : numpy.ndarray
        Transmittance [%].
    output_theta : numpy.ndarray
        Angle of outgoing ray [degrees].

    """
        
    k_0 = 2*np.pi/wl # Wavenumber
    theta = np.array([(theta)*np.pi/180])[0] # Convert AOI from degrees to radians
    M = np.shape(phys_l)[0] # Number of coating layers 
    S = len(wl) # Sampling size of wavelength array
            
    refl_perc_val = np.empty((np.size(theta),S)) # Nominal reflectance for each AOI
    transm_perc_val = np.empty((np.size(theta),S)) # Nominal reflectance for each AOI
    output_theta = np.empty(np.size(theta)) # Outgoing ray angle
    
    # Loop to calculate function outputs for each AOI
    for theta_ind in range(len(theta)):
                    
        kz = np.zeros_like(n, dtype='complex')         
        rho_t = np.zeros((S,M+1),dtype = 'complex') # Transverse Fresnel reflection coeff 
        tau_t = np.zeros((S,M+1),dtype = 'complex') # Transverse transmission coeff 
        cos_theta_i = np.zeros((np.shape(n)),dtype = 'complex') # cos(theta),theta = angle in radians
        matrix_M = np.zeros((2, 2, S, M+1),dtype = 'complex') # Matching matrix across interface i and i+1 in the absence of roughness
                    
        for nth in np.arange(np.shape(n)[1]):                      
            cos_theta_i[:,nth] = np.sqrt(1 - ((n[:,0]*np.sin(theta[theta_ind]))**2/ n[:,nth]**2)) # cos(theta_i),theta_i = angle in radians                      
            kz[:,nth] = k_0 * n[:,nth] * cos_theta_i[:,nth]                    
        output_theta[theta_ind] = np.arccos(cos_theta_i[0,-1]) *180/np.pi # Finding outgoing ray angle in degrees
    
            # calculating transverse refractive indices, determined by polarisation type
        if polar == 'TE':
            n_t = n * cos_theta_i
        elif polar == 'TM':
            n_t = n / cos_theta_i
            
        # calculating the Matching matrix of each interface in the system
        for nth in range(0,M+1):
            rho_t[:,nth] = (n_t[:,nth]-n_t[:,nth+1])/(n_t[:,nth]+n_t[:,nth+1]) # Transverse Fresnel reflection coeff 
            tau_t[:,nth] = 1+rho_t[:,nth] # Transverse Fresnel transmission coeff
            # Matching matrix
            matrix_M[:,:,:,nth] = 1./tau_t[:,nth] * np.array([[np.full((S),1),rho_t[:,nth]],[rho_t[:,nth],np.full((S),1)]],dtype = 'complex')
                   
        matrix_S = matrix_M[:,:,:,0] # intialising the Transfer matrix
    
        # calculating the Propagation matrix of coating layer in the system and the final Transfer matrix
        for nth in range(M):
            phi_i = kz[:,nth+1] * phys_l[nth] # Phase thickness of layer i
            # Propagation matrix
            matrix_P = np.array([[np.exp(1.j*phi_i),np.zeros(S)],[np.zeros(S),np.exp(-1j*phi_i)]],dtype = 'complex') 
               
            # Transfer matrix for each wavelength
            for lam in np.arange(0,len(wl),1): 
                matrix_S[:,:,lam] = np.dot(np.dot(matrix_S[:,:,lam],matrix_P[:,:,lam]),matrix_M[:,:,lam,nth+1])
                   
        # Calculating reflectance and transmittance
        s1 = matrix_S[0,0,:] 
        s2 = matrix_S[1,0,:]     
        refl_perc_val[theta_ind,:] = abs(s2/s1)**2 *100 # Reflectance in %                                           
        transm_perc_val[theta_ind,:] = abs(1./s1)**2*n[:,-1]/n[:,0] *100 # Transmittance in %  
            
    # For the case of many rays with different AOI, finding the overall reflectance/transmittance
    refl_perc = np.sum(refl_perc_val,axis=0)/np.size(theta)                       
    transm_perc = np.sum(transm_perc_val,axis=0)/np.size(theta)
    
    return refl_perc,transm_perc,output_theta
    

