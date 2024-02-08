#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run model for dichroic performance
"""

#%% Importing Libraries 

# Third-party imports
import os
import sys
import yaml
import numpy as np
from configparser import ConfigParser
import matplotlib.pyplot as plt

# Local imports
from general_tools.refractive_index import refr_index
from general_tools.multilayer_structure import multi_layout
from tlm_model.analysis import spectral_analysis,av_intensity


#%% CONFIGURATION: Reading configuration files

#Local folder imports
main_directory = os.path.abspath(os.path.dirname(sys.argv[0]))

ConfigInputFilePath = os.path.join(main_directory, 'config')
CONFIGFILES = ['config.ini','monte_carlo.ini','contamination.ini','systematics.ini']

param={}
mc_param={}
syst_param={}
contam_param = {}

for config_params,param_dict in zip(CONFIGFILES,[param,mc_param,contam_param,syst_param]):
    # Read parameters from configuration files
    config_file = os.path.join(ConfigInputFilePath, config_params) # Read parameters from the configuration file
    config_parameters = ConfigParser()
    file_read = config_parameters.read(os.path.expanduser(config_file))
    
    for name, value in config_parameters.items():
        param_dict[name] = {}
        for key, item in value.items(): 
            data = yaml.safe_load(item)
            param_dict[name][key] = data
            if isinstance(data, dict) and 'value' in data.keys():
                param_dict[name][key]['value'] = data['value']

config_multilayout = param['multilayer_layout']

# make output folder, if it doesn't exist
OutputFilePath = f"{param['DEFAULT']['outputfilepath']}/run_dichroic"
if not os.path.exists(OutputFilePath):
    os.makedirs(OutputFilePath)
# make folder for plots and data files
if not os.path.exists(f"{OutputFilePath}/plots"):
    os.makedirs(f"{OutputFilePath}/plots")
if not os.path.exists(f"{OutputFilePath}/data"):
    os.makedirs(f"{OutputFilePath}/data")


#%% NOMINAL PERFORMANCE: Dichroic Parameters Setup

# General modelling parameters
theta = np.array([param['environmental']['angle_of_incidence']['value']]) #AOI in degrees
n_rays = len(theta) # number of rays
wl = np.arange(param['environmental']['wavelength_min']['value'],param['environmental']['wavelength_max']['value']+param['environmental']['delta_wavelength']['value'],param['environmental']['delta_wavelength']['value']) #wl_data#wavelength array in nm
temp = param['environmental']['temperature']['value'] #temperature in kelvin
polar = param['environmental']['polarisation']

# Thicknesses of coating layers and substrate in nm
output_file = os.path.join(main_directory,param['DEFAULT']['inputfilepath'], config_multilayout['thickness_front_coating_only']) 
phys_l_front = np.loadtxt(output_file, dtype=float)
outputfile_rear = os.path.join(main_directory,param['DEFAULT']['inputfilepath'], config_multilayout['thickness_rear_coating_only']) 
phys_l_bbar = np.loadtxt(outputfile_rear, dtype=float)
phys_l_substrate = config_multilayout['thickness_substrate']['value']

# Multilayer structure
material_type = 'component'

n_H = refr_index(config_multilayout['refr_index_h'],material_type,wl,temp)
n_L = refr_index(config_multilayout['refr_index_l'],material_type,wl,temp)

# Front Coating Refractive indices 
n_incident = refr_index(config_multilayout['refr_index_incident'],material_type,wl,temp)
n_substrate = refr_index(config_multilayout['refr_index_substrate'],material_type,wl,temp)
layout_sequence_front = config_multilayout['refr_index_layout_front_coating_only']['sequence']
n_front = multi_layout(wl,n_H,n_L,n_incident,n_substrate,layout_sequence_front)

# BBAR Coating Refractive indices
n_incident_rear = refr_index(config_multilayout['refr_index_substrate'],material_type,wl,temp)
n_substrate_rear = refr_index(config_multilayout['refr_index_incident'],material_type,wl,temp)
layout_sequence_rear = config_multilayout['refr_index_layout_rear_coating_only']['sequence']
n_bbar = multi_layout(wl,n_H,n_L,n_incident_rear,n_substrate_rear,layout_sequence_rear)

#Calculating the nominal spectral performance 
nominal_r,nominal_t, nominal_ghost, _ ,_ = spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
nominal_r = av_intensity(nominal_r,n_rays)
nominal_t = av_intensity(nominal_t,n_rays)
nominal_ghost = av_intensity(nominal_ghost,n_rays)

# Plotting the nominal spectral performance
fig,ax = plt.subplots(figsize=(10,6))
ax.plot(wl,nominal_r,color='b',lw=2,label='Simulated R')
ax.plot(wl,nominal_t,color='r',lw=2,label='Simulated T')
ax.set_xlabel('$\lambda$ [nm]')
ax.set_ylabel(r'$R$ / $T$ [%]')
ax.set_ylim((0,100))
ax.set_xlim((min(wl),max(wl)))
ax.legend()
ax.grid()
fig.savefig(f"{OutputFilePath}/plots/nominal_spectrum.png",dpi=600)

