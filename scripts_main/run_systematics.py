#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run model for dichroic performance as a result of systematics
"""

#%% Importing Libraries 

# Third-party imports
import os
import sys
import yaml
import numpy as np
from configparser import ConfigParser
import matplotlib.pyplot as plt
import pandas as pd
import string
import copy


# Local imports
from general_tools.refractive_index import refr_index
from general_tools.multilayer_structure import multi_layout
from tlm_model.analysis import spectral_analysis,av_intensity
from general_tools.analysis_tools import change_spectral,trans_wl_shift
from uncertainty_tools.contam import mol_contam_analysis
from uncertainty_tools.thickness import random_thickness_error,random_thickness_errors_spectrum,systematic_thickness_error
from uncertainty_tools.refractive_index import refr_ind_error
from uncertainty_tools.temperature import temp_error
from uncertainty_tools.angle_of_incidence import angle_error
from plot_tools.histogram_2D import contoured_2d_hist,double_contoured_2d_hist


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
OutputFilePath = f"{param['DEFAULT']['outputfilepath']}/run_systematics"
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
nominal_r,nominal_t,nominal_ghost,_,_ = spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
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


#%% MOLECULAR CONTAMINATION: Modelling contamination on both sides of the dichroic surface

# Plotting the contamination model
fig,ax = plt.subplots(nrows=1, ncols=2,figsize=(12,3.5))
# Plot no contamination model
ax[0].plot(wl,nominal_r,'k',lw=2,label='clean')
ax[1].plot(wl,nominal_t,'k',lw=2,label='clean')

# Loop through different contamination scenarios
for c_name,c_layout in contam_param['multilayer_layout']['layouts'].items():
        
    c_refl, c_transm = mol_contam_analysis( wl, n_incident, n_front[:,1:-1], n_substrate, n_bbar[:,1:-1], phys_l_front, phys_l_substrate, phys_l_bbar, 
    temp, theta, polar, c_layout)

    c_refl = av_intensity(c_refl,n_rays)
    c_transm = av_intensity(c_transm,n_rays)

    # Plot reflectance and transmittance in two plots
    for ax_n,spec_n in zip(ax,[c_refl,c_transm]):
        ax_n.plot(wl,spec_n,label=c_name)
        ax_n.set_xlabel('$\lambda$ [nm]')
        ax_n.set_ylim((0,100))
        ax_n.set_xlim((min(wl),max(wl)))
        ax_n.legend(fontsize=10)
        ax_n.grid()

ax[0].set_ylabel('R [%]')
ax[1].set_ylabel('T [%]')
fig.savefig(f"{OutputFilePath}/plots/contamination_spectrum.png",dpi=600)


#%% RANDOM THICKNESS ERRORS: Random errors in coating layer thicknesses of error% 

# number of randomly generated thickness errors
n_sample = syst_param['general']['n_sample']['value'] 
# deposition uncertainty of machine in percent
rand_dep_error = syst_param['manufacturing']['random_dep_error']['value'] 
# layers which are affected by a deposition error
rand_dep_error_layers = 'all'

# Generate random errors in thickness
phys_l_front_rand, phys_l_rear_rand = random_thickness_error(phys_l_front,phys_l_bbar,n_sample,rand_dep_error,rand_dep_error_layers)
# Calculate the spectral performance for randomly altered thicknesses
rand_dep_r, rand_dep_t = random_thickness_errors_spectrum(wl,n_front,n_substrate,n_bbar,phys_l_front_rand,phys_l_substrate,phys_l_rear_rand,theta,polar)
# Find change in spectral performance due to random thickness errors
rand_dep_delta_r,rand_dep_delta_t = change_spectral(nominal_r,nominal_t,rand_dep_r,rand_dep_t)


# Save data from front coating thicknesses with random errors 
df_front = pd.DataFrame(data=phys_l_front_rand)
# Saving data as CSV file
df_front.to_csv(f"{OutputFilePath}/data/rand_error_thicknesses_nsample_"+str(n_sample)+".csv",sep=',',index=False)
df = pd.DataFrame(data=rand_dep_delta_t)
df.to_csv(f"{OutputFilePath}/data/rand_error_delta_t_nsample_"+str(n_sample)+".csv",sep=',',index=False)

# Read data from random thickness error files
n_sample = syst_param['general']['n_sample']['value'] 
filename = f"{OutputFilePath}/data/rand_error_delta_t_nsample_"+str(n_sample)+".csv"
df_read = pd.read_csv(filename,delimiter=',')
rand_dep_delta_t = df_read.values

# Plot a contoured 2D histogram of the change in transmittance
contoured_2d_hist(wl,rand_dep_delta_t,f"{OutputFilePath}/plots")


#%% SYSTEMATIC THICKNESS ERRORS: Systematic thickness errors of +/-error% in all layers 

sys_dep_error = syst_param['manufacturing']['systematic_dep_error']['value'] # deposition uncertainty of sputtering machine is error%

# Generate a systematic error of -error% in all layers & resulting change in performance
systematic_type = 'deficiency'
# Find spectral performance
sys_dep_minus_r,sys_dep_minus_t,sys_dep_minus_ghost,_,_,phys_l_front_sys_minus,phys_l_rear_sys_minus = systematic_thickness_error(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar,sys_dep_error,systematic_type)
# Combining rays of beam to find overall intensity
sys_dep_minus_r = av_intensity(sys_dep_minus_r,n_rays)
sys_dep_minus_t = av_intensity(sys_dep_minus_t,n_rays)
sys_dep_minus_ghost = av_intensity(sys_dep_minus_ghost,n_rays)

# Generate a systematic error of +error% in all layers & resulting change in performance
systematic_type = 'surplus'
# Find spectral performance
sys_dep_plus_r,sys_dep_plus_t,sys_dep_plus_ghost,_,_,phys_l_front_sys_plus,phys_l_rear_sys_plus = systematic_thickness_error(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar,sys_dep_error,systematic_type)
# Combining rays of beam to find overall intensity
sys_dep_plus_r = av_intensity(sys_dep_plus_r,n_rays)
sys_dep_plus_t = av_intensity(sys_dep_plus_t,n_rays)
sys_dep_plus_ghost = av_intensity(sys_dep_plus_ghost,n_rays)


#%% RANDOM & SYSTEMATIC THICKNESS ERRORS: Random errors of error% combined with Systematic thickness errors of +/-error% in all layers 

##### Case A: Systematic deposition deficiency & random errors #####

# Generate random errors in thickness
phys_l_front_sys_minus_rand, phys_l_rear_sys_minus_rand = random_thickness_error(phys_l_front_sys_minus,phys_l_rear_sys_minus,n_sample,rand_dep_error,rand_dep_error_layers)
# Calculate the spectral performance for randomly altered thicknesses
sys_dep_minus_rand_r,sys_dep_minus_rand_t = random_thickness_errors_spectrum(wl,n_front,n_substrate,n_bbar,phys_l_front_sys_minus_rand,phys_l_substrate,phys_l_rear_sys_minus_rand,theta,polar)

# Resulting change in spectral performance
sys_dep_minus_rand_delta_r,sys_dep_minus_rand_delta_t = change_spectral(nominal_r,nominal_t,sys_dep_minus_rand_r,sys_dep_minus_rand_t)


##### Case B: Systematic deposition surplus & random errors #####

# Generate random errors in thickness
phys_l_front_sys_plus_rand, phys_l_rear_sys_plus_rand = random_thickness_error(phys_l_front_sys_plus,phys_l_rear_sys_plus,n_sample,rand_dep_error,rand_dep_error_layers)
# Calculate the spectral performance for randomly altered thicknesses
sys_dep_plus_rand_r,sys_dep_plus_rand_t = random_thickness_errors_spectrum(wl,n_front,n_substrate,n_bbar,phys_l_front_sys_plus_rand,phys_l_substrate,phys_l_rear_sys_plus_rand,theta,polar)

# Resulting change in spectral performance
sys_dep_plus_rand_delta_r,sys_dep_plus_rand_delta_t = change_spectral(nominal_r,nominal_t,sys_dep_plus_rand_r,sys_dep_plus_rand_t)


# Save data
df = pd.DataFrame(data=sys_dep_minus_rand_delta_t)
df.to_csv(f"{OutputFilePath}/data/sys_minus_rand_error_delta_t_nsample_"+str(n_sample)+".csv",sep=',',index=False)
df = pd.DataFrame(data=sys_dep_plus_rand_delta_t)
df.to_csv(f"{OutputFilePath}/data/sys_plus_rand_error_delta_t_nsample_"+str(n_sample)+".csv",sep=',',index=False)

# Read data
filename = f"{OutputFilePath}/data/sys_minus_rand_error_delta_t_nsample_"+str(n_sample)+".csv"
df_read = pd.read_csv(filename,delimiter=',')
sys_dep_minus_rand_delta_t = df_read.values
filename = f"{OutputFilePath}/data/sys_plus_rand_error_delta_t_nsample_"+str(n_sample)+".csv"
df_read = pd.read_csv(filename,delimiter=',')
sys_dep_plus_rand_delta_t = df_read.values

# Plot a contoured 2D histogram of the change in transmittance for Case A & B
double_contoured_2d_hist(wl,sys_dep_minus_rand_delta_t,sys_dep_plus_rand_delta_t,f"{OutputFilePath}/plots")


#%% SYSTEMATIC INDEX ERROR: Uncertainty in refractive index as a linear decrease in a refrative index

n_sys_error = syst_param['manufacturing']['index_error']['value'] # error% in refractive index

error_material = syst_param['manufacturing']['index_error']['material'] # material with uncertainy

# Find resulting spectral performance
sys_index_r,sys_index_t,sys_index_ghost = refr_ind_error(wl,n_sys_error,error_material,config_multilayout,phys_l_front,phys_l_substrate,phys_l_bbar,temp,theta,polar) 
# Combining rays of beam to find overall intensity
sys_index_r = av_intensity(sys_index_r,n_rays)
sys_index_t = av_intensity(sys_index_t,n_rays)
sys_index_ghost = av_intensity(sys_index_ghost,n_rays)


#%% SYSTEMATIC TEMPERATURE ERROR: Uncertainty in operational temperature

# Setting refractive index of ZnSe to be temperature dependent (with no absorption available for this model)
config_multilayout_temp = copy.deepcopy(config_multilayout)  # make a copy of the configuration of indices for the temp-dependent model
config_multilayout_temp['refr_index_h']['temp_dependence']=True
config_multilayout_temp['refr_index_substrate']['temp_dependence']=True

# Alternative nominal performance using temperature-dependent index of ZnSe
n_H_temp = refr_index(config_multilayout_temp['refr_index_h'],material_type,wl,temp)
# Front Coating Refractive indices 
n_substrate_temp = refr_index(config_multilayout_temp['refr_index_substrate'],material_type,wl,temp)
n_front_temp = multi_layout(wl,n_H_temp,n_L,n_incident,n_substrate_temp,layout_sequence_front)
# BBAR Coating Refractive indices
n_incident_rear_temp = refr_index(config_multilayout_temp['refr_index_substrate'],material_type,wl,temp)
n_bbar_temp = multi_layout(wl,n_H_temp,n_L,n_incident_rear_temp,n_substrate_rear,layout_sequence_rear)

#Calculating the nominal spectral performance 
nominal_r_temp,nominal_t_temp,nominal_ghost_temp,_,_ = spectral_analysis(wl,n_front_temp,n_substrate_temp,n_bbar_temp,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
# Combining rays of beam to find overall intensity
nominal_r_temp = av_intensity(nominal_r_temp,n_rays)
nominal_t_temp = av_intensity(nominal_t_temp,n_rays)
nominal_ghost_temp = av_intensity(nominal_ghost_temp,n_rays)

# Plotting the spectral performance
fig,ax = plt.subplots(figsize=(10,6))
ax.plot(wl,nominal_r,color='b',lw=2,label='Simulated R')
ax.plot(wl,nominal_t,color='r',lw=2,label='Simulated T')
ax.plot(wl,nominal_r_temp,'b:',lw=2,label='Simulated R temp')
ax.plot(wl,nominal_t_temp,'r:',lw=2,label='Simulated T temp')
ax.set_xlabel('$\lambda$ [nm]')
ax.set_ylabel(r'$R$ / $T$ [%]')
ax.set_ylim((0,100))
ax.set_xlim((min(wl),max(wl)))
ax.legend()
ax.grid()


#%% Uncertainty on operating temperature
temp_sys_error = syst_param['environmental']['temperature_error']['value'] # error in temperature (Kelvin)

# Find resulting spectral performance
sys_temp_plus_r,sys_temp_plus_t,sys_temp_plus_ghost = temp_error(wl,temp_sys_error,config_multilayout_temp,phys_l_front,phys_l_substrate,phys_l_bbar,temp,theta,polar)
# Combining rays of beam to find overall intensity
sys_temp_plus_r = av_intensity(sys_temp_plus_r,n_rays)
sys_temp_plus_t = av_intensity(sys_temp_plus_t,n_rays)
sys_temp_plus_ghost = av_intensity(sys_temp_plus_ghost,n_rays)

temp_sys_error = -1*temp_sys_error
# Find resulting spectral performance
sys_temp_minus_r,sys_temp_minus_t,sys_temp_minus_ghost = temp_error(wl,temp_sys_error,config_multilayout_temp,phys_l_front,phys_l_substrate,phys_l_bbar,temp,theta,polar)
# Combining rays of beam to find overall intensity
sys_temp_minus_r = av_intensity(sys_temp_minus_r,n_rays)
sys_temp_minus_t = av_intensity(sys_temp_minus_t,n_rays)
sys_temp_minus_ghost = av_intensity(sys_temp_minus_ghost,n_rays)

#%% Plotting the nominal spectral performance
fig,ax = plt.subplots(figsize=(10,6))
ax.plot(wl,sys_temp_plus_ghost,color='r',lw=2,label='Simulated T')
ax.plot(wl,sys_temp_minus_ghost,'g--',lw=2,label='Simulated T temp')
ax.plot(wl,nominal_ghost,'b:',lw=2,label='Simulated T temp')
ax.set_xlabel('$\lambda$ [nm]')
ax.set_ylabel(r'$R$ / $T$ [%]')
ax.set_xlim((min(wl),max(wl)))
ax.legend()
ax.grid()


#%% SYSTEMATIC INCIDENT ANGLE ERROR: Uncertainty in angle of incidence of a collimated beam

sys_theta_error = syst_param['environmental']['theta_error']['value'] # error in angle of incidence (degrees)

# Find resulting spectral performance
sys_theta_r,sys_theta_t,sys_theta_ghost = angle_error(wl,sys_theta_error,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
# Combining rays of beam to find overall intensity
sys_theta_r = av_intensity(sys_theta_r,n_rays)
sys_theta_t = av_intensity(sys_theta_t,n_rays)
sys_theta_ghost = av_intensity(sys_theta_ghost,n_rays)


#%% STORING THE PERFORMANCES: Dictionary storing the spectral performance for the scenarios

spectral_perf = {
    r'Systematic $d_i$ deficiency': {
        'R':sys_dep_minus_r,'T':sys_dep_minus_t,'ghost': sys_dep_minus_ghost
    },
    r'Systematic $d_i$ surplus': {
        'R':sys_dep_plus_r,'T':sys_dep_plus_t,'ghost': sys_dep_plus_ghost
    },
    'Uncertainty in $n_{L}$':{
        'R':sys_index_r,'T':sys_index_t,'ghost':sys_index_ghost
    },
    r'Shift of $ \theta$':{
        'R':sys_theta_r,'T':sys_theta_t,'ghost':sys_theta_ghost
    },
    r'Systematic $t$ increase':{
        'R':sys_temp_plus_r,'T':sys_temp_plus_t,'ghost':sys_temp_plus_ghost
    },
    r'Systematic $t$ decrease':{
        'R':sys_temp_minus_r,'T':sys_temp_minus_t,'ghost':sys_temp_minus_ghost
    }
}


#%% PLOTTING ALL PERFORMANCES

fig,(ax,ax1) = plt.subplots(nrows=2, ncols=1,figsize=(9,5.5))
colour_list = ['#0000FF','green','#FFA500','#FF0000']#,'purple','y']

for col, (perf_name,perf_val) in zip(colour_list,spectral_perf.items()):
    ax.plot(wl,perf_val['T'],color=col,label=perf_name,linewidth=2,linestyle='solid')
    delta_t = perf_val['T'] - nominal_t
    ax1.plot(wl,delta_t,color=col,label=perf_name,linewidth=2,linestyle='solid')
ax.plot(wl,sys_temp_plus_t,'purple',label=r'Systematic $t$ increase',linewidth=2)
ax1.plot(wl,sys_temp_plus_t-nominal_t_temp,'purple',label='Nominal performance',linewidth=2)
ax.plot(wl,sys_temp_minus_t,'y',label=r'Systematic $t$ dencrease',linewidth=2)
ax1.plot(wl,sys_temp_minus_t-nominal_t_temp,'y',label='Nominal performance',linewidth=2)
ax.plot(wl,nominal_t,'k--',label='Nominal performance',linewidth=1)
ax1.plot(wl,nominal_t-nominal_t,'k--',label='Nominal performance',linewidth=1)

ax.legend(loc='lower right',prop={'size': 10})
ax.set_ylabel(r'$ T $ [%]')
ax.set_xlabel('$\lambda$ [nm]')
ax.set_ylim((0,100))
ax.set_xlim((min(wl),max(wl)))
ax1.set_ylabel(r'$ \Delta T  $ [%]')
ax1.set_xlabel('$\lambda$ [nm]')
ax1.set_xlim((min(wl),max(wl)))

fig.savefig(f"{OutputFilePath}/plots/sys_performances.png",dpi=600)


#%% PLOTTING ALL GHOST REFLECTIONS

fig,(ax) = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
colour_list = ['#0000FF','green','#FFA500','#FF0000','purple','y']

for col, (perf_name,perf_val) in zip(colour_list,spectral_perf.items()):
    ax.plot(wl,perf_val['ghost'],color=col,label=perf_name,linewidth=2,linestyle='solid')

ax.set_ylabel(r'$R$ [%]')
ax.set_ylim((0,4.3))
ax.set_xlabel('$\lambda$ [nm]')
ax.legend(fontsize=10,loc='lower right')
ax.set_xlim((min(wl),max(wl)))

ax.plot(wl,nominal_ghost,label='Nominal performance',color='k',linestyle='--',lw=2)

fig.savefig(f"{OutputFilePath}/plots/ghosts.png",dpi=600)


#%% TRANSITION WAVELENGTH SHIFTS: Calculating the shifts in the transition wavelength for the scenarios

# Find nominal transition wavelength 
transition_y = 50 # reflection / transmission at point of transition wavelength
nominal_transwl = np.interp(transition_y,nominal_t[np.where((nominal_t<=51)&(nominal_t>=49))],wl[np.where((nominal_t<=51)&(nominal_t>=49))])
# Find gradient of nominal transition slope
fit_range = 50 # in nm, wavelength range used to fit gradient – edit as required
slope_range = np.where((wl>nominal_transwl-fit_range)&(wl<nominal_transwl+fit_range))
wl_fit = wl[slope_range]
nominal_t_fit = nominal_t[slope_range]
A = np.stack([wl_fit, np.ones_like(wl_fit)]).T
m, c = np.linalg.lstsq(A, nominal_t_fit, rcond=None)[0]


# Find nominal transition wavelength, again but this time for temp-dependent model
nominal_transwl_temp = np.interp(transition_y,nominal_t_temp[np.where((nominal_t_temp<=51)&(nominal_t_temp>=49))],wl[np.where((nominal_t_temp<=51)&(nominal_t_temp>=49))])
# Find gradient of nominal transition slope
fit_range = 50 # in nm, wavelength range used to fit gradient – edit as required
slope_range = np.where((wl>nominal_transwl_temp-fit_range)&(wl<nominal_transwl_temp+fit_range))
wl_fit = wl[slope_range]
nominal_t_fit_temp = nominal_t_temp[slope_range]
A = np.stack([wl_fit, np.ones_like(wl_fit)]).T
m_temp, c_temp = np.linalg.lstsq(A, nominal_t_fit_temp, rcond=None)[0]


# Systematic thickness errors
systematic_type = 'surplus'
sys_dep_error_list = np.arange(-0.006,0.006+0.001,0.001) # deposition uncertainty of sputtering machine 
d_lambda_sys_dep_plus = []
for sys_dep_error_val in sys_dep_error_list:
    # Find spectral performance
    sys_dep_plus_r,sys_dep_plus_t,sys_dep_plus_ghost,_,_,phys_l_front_sys_plus,phys_l_rear_sys_plus = systematic_thickness_error(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar,sys_dep_error_val,systematic_type)
    # Combining rays of beam to find overall intensity
    sys_dep_plus_r = av_intensity(sys_dep_plus_r,n_rays)
    sys_dep_plus_t = av_intensity(sys_dep_plus_t,n_rays)
    sys_dep_plus_ghost = av_intensity(sys_dep_plus_ghost,n_rays)

    # Find shift in transition wavelength 
    shifted_trans_wl = trans_wl_shift(wl, nominal_transwl, m, sys_dep_plus_t)
    # Add to array
    d_lambda_sys_dep_plus.append(shifted_trans_wl)
   
    
# Systematic index errors
n_sys_error_list = -1*np.arange(0,0.004+0.0005,0.0005) #np.linspace(0,0.004,step_error)
d_lambda_sys_index = []
for n_sys_error in n_sys_error_list:
    # Find spectral performance
    sys_index_r,sys_index_t,sys_index_ghost = refr_ind_error(wl,n_sys_error,error_material,config_multilayout,phys_l_front,phys_l_substrate,phys_l_bbar,temp,theta,polar)
    # Combining rays of beam to find overall intensity
    sys_index_r = av_intensity(sys_index_r,n_rays)
    sys_index_t = av_intensity(sys_index_t,n_rays)
    sys_index_ghost = av_intensity(sys_index_ghost,n_rays)

    # Find shift in transition wavelength 
    shifted_trans_wl = trans_wl_shift(wl, nominal_transwl, m, sys_index_t)
    # Add to array
    d_lambda_sys_index.append(shifted_trans_wl)
    
       
# Systematic temperature errors
d_lambda_sys_temp_plus = []
temp_sys_error_list = np.arange(-12.5,12.5+2.5,2.5) #np.linspace(-12.5,12.5,step_error)
for temp_sys_error in temp_sys_error_list:
    # Find spectral performance
    sys_temp_plus_r,sys_temp_plus_t,sys_temp_plus_ghost = temp_error(wl,temp_sys_error,config_multilayout_temp,phys_l_front,phys_l_substrate,phys_l_bbar,temp,theta,polar)
    # Combining rays of beam to find overall intensity
    sys_temp_plus_r = av_intensity(sys_temp_plus_r,n_rays)
    sys_temp_plus_t = av_intensity(sys_temp_plus_t,n_rays)
    sys_temp_plus_ghost = av_intensity(sys_temp_plus_ghost,n_rays)

    # Find shift in transition wavelength 
    shifted_trans_wl = trans_wl_shift(wl, nominal_transwl_temp, m_temp, sys_temp_plus_t)
    # Add to array
    d_lambda_sys_temp_plus.append(shifted_trans_wl)


# Systematic angle of incidence errors
sys_theta_error_list = np.arange(-0.1,0.1+0.025,0.025) #np.linspace(-0.1,0.1,step_error)
d_lambda_sys_theta = []
for sys_theta_error in sys_theta_error_list:
    # Find spectral performance
    sys_theta_r,sys_theta_t,sys_theta_ghost = angle_error(wl,sys_theta_error,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
    # Combining rays of beam to find overall intensity
    sys_theta_r = av_intensity(sys_theta_r,n_rays)
    sys_theta_t = av_intensity(sys_theta_t,n_rays)
    sys_theta_ghost = av_intensity(sys_theta_ghost,n_rays)

    # Find shift in transition wavelength 
    shifted_trans_wl = trans_wl_shift(wl, nominal_transwl, m, sys_theta_t)
    # Add to array
    d_lambda_sys_theta.append(shifted_trans_wl)


#%% PLOTTING TRANSITION WAVELENGTH SHIFTS

fig, (ax,ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=4,figsize=(14,4))

# Systematic thickness errors
ax.scatter(sys_dep_error_list*100, d_lambda_sys_dep_plus, s=50, facecolors='b', edgecolors='b')
ax.plot(sys_dep_error_list*100,d_lambda_sys_dep_plus,'b--',lw=2)
ax.set_xlim((min(sys_dep_error_list*100),max(sys_dep_error_list*100)))
xticks=[-0.6,-0.3,0.0,0.3,0.6]
ax.xaxis.set_ticks(xticks)
ax.xaxis.set_ticklabels(['%.1f'  % i for i in xticks])
ax.set_xlabel(r'Systematic change in $d_i$',fontsize=16)
ax.set_ylabel(r'$\Delta \lambda$ [nm]',fontsize=16)

# Systematic index errors
ax1.scatter(n_sys_error_list*100, d_lambda_sys_index, s=50, facecolors='r', edgecolors='r')#,label='Systematic deposition error')
ax1.plot(n_sys_error_list*100,d_lambda_sys_index,'r--',lw=2)
ax1.set_xlabel(r'Uncertainty in $n_{L}$',fontsize=16)
ax1.set_ylabel(r'$\Delta \lambda$ [nm]',fontsize=16)
ax1.set_xlim((min(100*n_sys_error_list),max(100*n_sys_error_list)))
 
# Systematic temperature errors
ax2.set_xlabel(r'Systematic change in $t$',fontsize=16)
ax2.set_ylabel(r'$\Delta \lambda$ [nm]',fontsize=16)
ax2.scatter(temp_sys_error_list, d_lambda_sys_temp_plus, s=50, facecolors='g', edgecolors='g')
xticks=[-10,-5,0,5,10]
ax2.xaxis.set_ticks(xticks)
ax2.xaxis.set_ticklabels(['%d'  % i for i in xticks])
yticks=[-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4]
ax2.yaxis.set_ticks(yticks)
ax2.yaxis.set_ticklabels(['%.1f'  % i for i in yticks])
ax2.plot(temp_sys_error_list,d_lambda_sys_temp_plus,'--',color='g', lw=2)
ax2.set_xlim((temp_sys_error_list.min(),temp_sys_error_list.max()))

# Systematic angle of incidence errors
ax3.scatter(sys_theta_error_list, d_lambda_sys_theta, s=50, facecolors='purple', edgecolors='purple')
ax3.plot(sys_theta_error_list,d_lambda_sys_theta,'--', color='purple', lw=2)
ax3.set_xlim((min(sys_theta_error_list),max(sys_theta_error_list)))
ax3.set_xlabel(r'Shift of $ \theta$',fontsize=16)
ax3.set_ylabel(r'$\Delta \lambda$ [nm]',fontsize=16)


for n, ax_chosen in enumerate([ax, ax1, ax2, ax3]):
    ax_chosen.text(0.03, 0.93, '('+string.ascii_uppercase[n]+')', transform=ax_chosen.transAxes, 
            size=16, weight='bold')
    ax_chosen.tick_params(axis='x', labelsize=16)
    ax_chosen.tick_params(axis='y', labelsize=16)
fig.tight_layout()
fig.savefig(f"{OutputFilePath}/plots/transwl_shifts.png",dpi=600)

