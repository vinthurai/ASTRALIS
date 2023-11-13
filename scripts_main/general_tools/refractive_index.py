#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions to calculate the refractive indices as a function of wavelength 
"""

import numpy as np
import os
import sys
from scipy import interpolate


def refr_index(material_dict,material_type,wl,temp):
    """
    Calculating the complex refractive index as a function of wavelength

    Parameters
    ----------
    material_dict : dict
        Dictionary of material name and temperature dependence info.
    material_type : str
        Type of material – 'component' or 'contamination'.
    wl : numpy.ndarray
        Wavelengths.
    temp : float
        Operational temperature of dichroic [K].

    Returns
    -------
    refr_index_val : numpy.ndarray
        Complex refractive index as a function of wavelength.

    """
        
    main_directory = os.path.abspath(os.path.dirname(sys.argv[0]))
    InputFilePath = os.path.join(main_directory, 'data/indices_data')
    
    material_name = material_dict['name']
    temp_dependence = material_dict['temp_dependence']

    if material_type == 'component':   
        
        if material_name == 'vacuum':

            if temp_dependence == True:
                print('No sources available for this refractive index option of ',material_name)

            elif temp_dependence == False:
                n_vac= np.full((np.shape(wl)),1) #index of indicence medium - vacuum
                refr_index_val = n_vac
    
    
        elif material_name == 'YbF3':

            if temp_dependence == True:
                print('No sources available for this refractive index option of ',material_name)
                
            elif temp_dependence == False:
                #source: https://refractiveindex.info/tmp/database/data-nk/main/YbF3/Amotchkina.txt
                #paper: T. Amotchkina, M. Trubetskov, D. Hahner, V. Pervak, Characterization of e-beam evaporated Ge, YbF3, ZnS, and LaF3 thin films for laser-oriented coatings, Appl. Opt. 59, A40-A47 (2020) 
                
                FILE = 'YbF3_nk_Amotchkina2020.txt'
        
                file_name = os.path.join(InputFilePath, FILE) # Read parameters from the configuration file
        
                index_data = np.genfromtxt(os.path.expanduser(file_name), delimiter='\t')
                        
                wl_index_data = index_data[:,0] #in nm
                n_index_data = index_data[:,1]
                k_index_data = index_data[:,2]
        
                #extrapolate n_real to all wavelengths needed
                f_n_index_data = interpolate.interp1d(wl_index_data,n_index_data)
                n_index_data_interp = f_n_index_data(wl)
        
                #interpolate k to all wavelengths needed
                f_k_index_data = interpolate.interp1d(wl_index_data,k_index_data)
                k_index_data_interp = f_k_index_data(wl)
                
                refr_index_val = n_index_data_interp - k_index_data_interp*1j 
        
        
        elif material_name == 'ZnSe':
            
            if temp_dependence == True:
                # ZnSe model with temperature depedence but with no absorption data
                # Using a temperature-dependent Sellmeier model applicable to temperatures of 93.2≤T≤473.2 K.
                #source: https://www.spiedigitallibrary.org/journals/optical-engineering/volume-34/issue-5/0000/Temperature-dependent-refractive-index-models-for-BaF2-CaF2-MgF2-SrF2/10.1117/12.201666.full
                #paper: Tropf, William J. "Temperature-dependent refractive index models for BaF2, CaF2, MgF2, SrF2, LiF, NaF, KCl, ZnS, and ZnSe." Optical Engineering 3
                    
                S1 = 4.4795137 + 9.60711e-5*temp - 8.03819e-7*(temp**2) + 1.31557e-9*(temp**3) - 1.03854e-12*(temp**4)
                S2 = 0.3724243 + 4.44903e-5*temp + 1.47850e-6*(temp**2) - 2.66133e-9*(temp**3) + 2.09294e-12*(temp**4)
                S3 = 2.8702146 + 2.14134e-5*temp + 1.66440e-8*(temp**2) + 4.39910e-10*(temp**3) - 8.36072e-13*(temp**4)  
                lambda1 = 0.20107634
                lambda2 = 0.39210520
                lambda3 = 47.047590
    
                n_znse = np.sqrt( 1 + ((S1*((wl*0.001)**2))/(((wl*0.001)**2)-((lambda1)**2))) +
                                      ((S2*((wl*0.001)**2))/(((wl*0.001)**2)-((lambda2)**2))) +
                                      ((S3*((wl*0.001)**2))/(((wl*0.001)**2)-((lambda3)**2))))
    
                refr_index_val = n_znse  
    
            elif temp_dependence == False:
                #ZnSe model with no temperature depedence but with absorption data
                #source: https://refractiveindex.info/tmp/database/data-nk/main/ZnSe/Amotchkina.txt
                #paper: T. Amotchkina, M. Trubetskov, D. Hahner, V. Pervak, Characterization of e-beam evaporated Ge, YbF3, ZnS, and LaF3 thin films for laser-oriented coatings, Appl. Opt. 59, A40-A47 (2020) 
                    
                FILE = 'ZnSe_nk_Amotchkina2020.txt'
        
                file_name = os.path.join(InputFilePath, FILE) # Read parameters from the configuration file
        
                index_data = np.genfromtxt(os.path.expanduser(file_name), delimiter='\t')
                        
                wl_index_data = index_data[:,0] #in nm
                n_index_data = index_data[:,1]
                k_index_data = index_data[:,2]
        
                #extrapolate n_real to all wavelengths needed
                f_n_index_data = interpolate.interp1d(wl_index_data,n_index_data)
                n_index_data_interp = f_n_index_data(wl)
        
                #interpolate k to all wavelengths needed
                f_k_index_data = interpolate.interp1d(wl_index_data,k_index_data)
                k_index_data_interp = f_k_index_data(wl)
                
                refr_index_val = n_index_data_interp - k_index_data_interp*1j 
        
    
        
    if material_type == 'contamination':    

        if material_name == 'ice_water': 
            
            if temp_dependence == True:
                print('No sources available for this refractive index option of ',material_name)
            
            elif temp_dependence == False:
                #source: https://refractiveindex.info/database/data/main/H2O/Warren-1984.yml
                #paper: S. G. Warren. Optical constants of ice from the ultraviolet to the microwave, Appl. Opt. 23, 1206-1225 (1984)
                FILE = 'index_ice_water_warren1984.txt'        
                
                file_name = os.path.join(InputFilePath, FILE) # Read parameters from the configuration file
                index_data = np.genfromtxt(os.path.expanduser(file_name))
        
                wl_index_data = index_data[:,0]*1000
                n_index_data = index_data[:,1]
                k_index_data = index_data[:,2]
                
                #extrapolate n_real to all wavelengths needed
                f_n_index_data = interpolate.interp1d(wl_index_data,n_index_data,bounds_error=False)
                n_index_data_interp = f_n_index_data(wl)
        
                #interpolate k to all wavelengths needed
                f_k_index_data = interpolate.interp1d(wl_index_data,k_index_data,bounds_error=False)
                k_index_data_interp = f_k_index_data(wl)
                
                refr_index_val = n_index_data_interp - k_index_data_interp*1j 

    
        elif material_name == 'DC-704':
            
            if temp_dependence == True:
                print('No sources available for this refractive index option of ',material_name)
            
            elif temp_dependence == False:
                #source: https://livchem-logistics.com/wp-content/uploads/2017/03/SIT7757.0-msds.pdf
                #paper: Gelest, Material safety data sheet - safety data sheet sit7757.0
    
                n_data_sheet = 1.551 #index of indicence medium - vacuum
                k_data_sheet = 0 #index of indicence medium - vacuum
        
                n_index_data_interp = np.full(np.shape(wl),n_data_sheet)
                k_index_data_interp = np.full(np.shape(wl),k_data_sheet)
        
                refr_index_val = n_index_data_interp - k_index_data_interp*1j                 
                       
                
    return refr_index_val
