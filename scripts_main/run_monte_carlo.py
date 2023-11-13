#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo code seperated from other dichroics modelling code used to vary 
thicknesses of coatings from measurements of thicknesses and reflection % 
from graphs until reflection % of dichroics match initial conditions

These thicknesses are then used in further simulations to test tolerance of the 
dichroic to uncertainties from various sources, including refractive index and 
thickness of layers
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run Monte Carlo for dichroic recipe
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
from tlm_model.analysis import spectral_analysis
from general_tools.analysis_tools import weighted_lsr
from general_tools.monte_carlo import mc_recipe


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
OutputFilePath = param['DEFAULT']['outputfilepath']


#%% NOMINAL PERFORMANCE: Dichroic Parameters Setup

# General modelling parameters
theta = np.array([param['environmental']['angle_of_incidence']['value']]) #AOI in degrees
wl = np.arange(param['environmental']['wavelength_min']['value'],param['environmental']['wavelength_max']['value']+param['environmental']['delta_wavelength']['value'],param['environmental']['delta_wavelength']['value']) #wl_data#wavelength array in nm
temp = param['environmental']['temperature']['value'] #temperature in kelvin
polar = param['environmental']['polarisation']

# Thicknesses of coating layers and substrate in nm
input_file = os.path.join(main_directory,mc_param['DEFAULT']['filepath_recipe'], mc_param['recipe_data']['inputfile_front']) 
phys_l_front = np.loadtxt(input_file, dtype=float)
inputfile_rear = os.path.join(main_directory,mc_param['DEFAULT']['filepath_recipe'], mc_param['recipe_data']['inputfile_rear']) 
phys_l_bbar = np.loadtxt(inputfile_rear, dtype=float)
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
initial_r,initial_t, _ = spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)


#%% Monte Carlo Parameters
step_size_max = mc_param['recipe_data']['monte_carlo_step']

# Define wavelength regions before and after the transition region
min_wl = wl[wl < 1830]
max_wl = wl[wl > 2070]

# Boundary conditions of performance before and after the transition region
bc_wl_min = np.linspace(min(min_wl), max(min_wl), len(min_wl) + 100)
bc_r_min = np.full_like(bc_wl_min, 100)
bc_wl_max = np.linspace(min(max_wl), max(max_wl), len(max_wl) + 100)
bc_r_max = np.zeros_like(bc_wl_max)

# Obtaining boundary condition points for the transition slope
extent_wl = [bc_wl_min[-1], bc_wl_max[0]]
extent_r = [bc_r_min[-1], bc_r_max[0]]
# Transition wavelength boundary conditions
bc_wl_transit = np.linspace(bc_wl_min[-1], bc_wl_max[0], 50)
bc_r_transit = np.interp(bc_wl_transit, extent_wl, extent_r) 

# Concatenate the boundary conditions for the three wavelength regions
bc_wl = np.concatenate((bc_wl_min, bc_wl_transit, bc_wl_max))
bc_r = np.concatenate((bc_r_min, bc_r_transit, bc_r_max)) # in reflectance
bc_t = 100-bc_r # in transmittance
# Weighted least-square residual between intial guess and boundary conditions
w_lsr_input_r = weighted_lsr(wl,initial_r,bc_wl,bc_r)
w_lsr_input_t = weighted_lsr(wl,initial_t,bc_wl,bc_t)
w_lsr_input = w_lsr_input_r + w_lsr_input_t

# Restrictions on coating thicknesses
min_phys_l = mc_param['thickness_rules']['min_thickness'] # minimum thickness per coating layer in nm
max_phys_l =  mc_param['thickness_rules']['max_thickness'] # maximum thickness per coating layer in nm
max_tot_phys_l = mc_param['thickness_rules']['max_total_thickness']# maximum thickness of total coating stack in nm

# Output files of improved recipe from MC
MC_output_front = os.path.join(main_directory,mc_param['DEFAULT']['filepath_recipe'],mc_param['recipe_data']['outputfile_front'])
MC_output_bbar = os.path.join(main_directory,mc_param['DEFAULT']['filepath_recipe'],mc_param['recipe_data']['outputfile_rear'])

#% Plotting Monte Carlo boundary conditions
fig,ax = plt.subplots(figsize=(10,6))
ax.plot(wl,initial_r,color='b',lw=2,label='Simulated R')
ax.plot(wl,initial_t,color='r',lw=2,label='Simulated T')
ax.plot(bc_wl,bc_r,'g.-',label='Boundary Conditions for R')
ax.plot(bc_wl,bc_t,'k.-',label='Boundary Conditions for T')
ax.set_xlabel('$\lambda$ [nm]')
ax.set_ylabel(r'$R$ / $T$ [%]')
ax.set_ylim((0,100))
ax.set_xlim((min(wl),max(wl)))
ax.legend()
ax.grid()
fig.savefig(f"{OutputFilePath}/MC_initial_conditions.png",dpi=600)


#%% Run Monte Carlo
diff_lsr = 100 # difference between previous and current weighted LSR, initiated with large value
while abs(diff_lsr) > 1e-10:
    phys_l_front_error,phys_l_bbar_error,w_lsr_output = phys_l_front_error,phys_l_bbar_error,diff_lsr = mc_recipe(wl,step_size_max,bc_wl,bc_r,bc_t,w_lsr_input,min_phys_l,max_phys_l,max_tot_phys_l,MC_output_front,MC_output_bbar,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
    phys_l_front = phys_l_front_error
    phys_l_bbar = phys_l_bbar_error
    w_lsr_input = w_lsr_output


#%% Performance after MC run
# Thicknesses of coating layers and substrate in nm
phys_l_front = np.loadtxt(MC_output_front, dtype=float)
phys_l_bbar = np.loadtxt(MC_output_bbar, dtype=float)
#Calculating the nominal spectral performance 
output_r,output_t,_ = spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)


#%% Plotting Monte Carlo ouput
fig,ax = plt.subplots(figsize=(10,6))
ax.plot(wl,output_r,color='b',lw=2,label='Simulated R')
ax.plot(wl,output_t,color='r',lw=2,label='Simulated T')
ax.plot(wl,initial_r,'b:',lw=2)
ax.plot(wl,initial_t,'r:',lw=2)
ax.plot(bc_wl,bc_r,'go-',label='Boundary Conditions for R')
ax.set_xlabel('$\lambda$ [nm]')
ax.set_ylabel(r'$R$ / $T$ [%]')
ax.set_ylim((0,100))
ax.set_xlim((min(wl),max(wl)))
ax.legend()
ax.grid()
fig.savefig(f"{OutputFilePath}/perf_after_MC.png",dpi=600)
  