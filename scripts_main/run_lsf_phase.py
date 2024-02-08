#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run model for low spatial frequency (LSF) thickness errors for the phase study
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
from tlm_model.analysis import spectral_analysis, av_intensity
from phase_tools.wfe import phase_to_wfe
from phase_tools.unlocal_deposition import lsf_errors
from phase_tools.local_deposition import disk_points, spatial_deposition, phase_error_calc

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
OutputFilePath = f"{param['DEFAULT']['outputfilepath']}/run_lsf_phase"
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
component_types = config_multilayout['component_type']
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

# Nominal phase and spectral performance
nominal_r, nominal_t, _, nominal_r_phase, nominal_t_phase = spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
nominal_r = av_intensity(nominal_r,n_rays)
nominal_t = av_intensity(nominal_t,n_rays)

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


#%% UNLOCALISED LSF ERROR: a systematic thickness error
    
# Testing different tolerances
rand_dep_error_list = np.array([0.001,0.002,0.003,0.005,0.01,0.02])[::-1] # parametrically varying random errors
n_tolerances = len(rand_dep_error_list)

wl_point = np.array([1100,5000]) #nm #chosen wavelength
ind_wl = np.where(np.isin(wl, wl_point))[0] # finding indices of wavelength values chosen

# Finding corresponding refractive index at the chosen wavelengths
n_point_front = n_front[ind_wl,:]
n_point_substrate = n_substrate[ind_wl]
n_point_bbar = n_bbar[ind_wl,:] 

# Corresponding phase at wl_point
# nominal_r_phase_point = nominal_r_phase[ind_wl]
# nominal_t_phase_point = nominal_t_phase[ind_wl]

nominal_r_point, nominal_t_point, _, nominal_r_phase_point, nominal_t_phase_point = spectral_analysis(wl_point,n_point_front,n_point_substrate,n_point_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)

#%%# Deposition uncertainty of machine in percent
sys_dep_error = syst_param['manufacturing']['systematic_dep_error']['value']

# Setting up empty arrays: 
# Spectral performance
sys_dep_r = np.empty((n_tolerances,len(wl_point)))
sys_dep_t = np.empty((n_tolerances,len(wl_point)))
# Phase changes
sys_dep_r_phase = np.empty_like(sys_dep_r)
sys_dep_t_phase = np.empty_like(sys_dep_t)
# Phase difference from nominal case
sys_dep_r_phase_delta = np.empty(n_tolerances)
sys_dep_t_phase_delta = np.empty(n_tolerances)

# Calculating the phase difference for different systematic errors
for i,rand_dep_error in enumerate(rand_dep_error_list):
    # Modelling the systematic error
    sys_dep_r_val,sys_dep_t_val,sys_dep_r_phase[i,:],sys_dep_t_phase[i,:] = lsf_errors(wl_point,n_point_front,n_point_substrate,n_point_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar,rand_dep_error)
    # Combining rays of beam to find overall intensity
    sys_dep_r[i,:] = av_intensity(sys_dep_r_val, n_rays)
    sys_dep_t[i,:] = av_intensity(sys_dep_t_val, n_rays)
    # Difference in phase from nominal case, for R and T at different wavelength points
    sys_dep_r_phase_delta[i] = np.real(nominal_r_phase_point[0]-sys_dep_r_phase[i,0])       
    sys_dep_t_phase_delta[i] = np.real(nominal_t_phase_point[1]-sys_dep_t_phase[i,1]) 
        
# Plotting resulting phase change as a function of t_error
fig1, (ax0,ax1) = plt.subplots(2,1, figsize=(6,5))
for i,rand_dep_error in zip(range(n_tolerances),rand_dep_error_list):
    ax0.plot(100*rand_dep_error,sys_dep_t_phase_delta[i],'o',color='purple')
ax0.set_ylabel('Phase difference in R \nat $\lambda=1.1\mu m$ ($\degree$)')
ax0.set_xlabel('$\Delta t$ (%)')    
ax0.grid(visible=True) 
for i,rand_dep_error in zip(range(n_tolerances),rand_dep_error_list):
    ax1.plot(100*rand_dep_error,sys_dep_t_phase_delta[i],'o',color='purple')
ax1.set_ylabel('Phase difference in T \nat $\lambda=5\mu m$ ($\degree$)')
ax1.set_xlabel('$\Delta t$ (%)')    
ax1.grid(visible=True) 
fig1.set_tight_layout(True)
fig1.savefig(f"{OutputFilePath}/plots/Case2_vs_t_error_D1_front.png",dpi=600)


#%% WAVEFRONT ERROR: Calculating respective wavefront error over a range of wavelengths

# Testing a few tolerances
rand_dep_error_list_short = np.array([0.001,0.005,0.01])[::-1] # parametrically varying random errors
# Selecting range of wavelengths
wl_sample_r = wl[np.where((wl>=1550)&(wl<=1750))]
wl_sample_t = wl[np.where((wl>=5900)&(wl<=6100))]
# finding indices of wavelength values chosen
ind_wl_r = np.where(np.isin(wl, wl_sample_r))[0]
ind_wl_t = np.where(np.isin(wl, wl_sample_t))[0]
# corresponding nominal phase 
nominal_r_phase_sample = nominal_r_phase[ind_wl_r]
nominal_t_phase_sample = nominal_t_phase[ind_wl_t]
# Setting up empty arrays
sys_dep_wfe_r = np.empty((len(rand_dep_error_list_short),len(wl)))
sys_dep_wfe_t = np.empty((len(rand_dep_error_list_short),len(wl)))

# Finding corresponding refractive index at the chosen wavelengths,
# chosen based flat parts of the spectrum in reflection and transmission
n_point_front_r = n_front[ind_wl_r,:] 
n_point_front_t = n_front[ind_wl_t,:] 
n_point_substrate_r = n_substrate[ind_wl_r]
n_point_substrate_t = n_substrate[ind_wl_t]
n_point_bbar_r = n_bbar[ind_wl_r,:] 
n_point_bbar_t = n_bbar[ind_wl_t,:] 

# Reflection, focusing on flat part of spectrum
xlim_lo, xlim_hi = min(wl_sample_r), max(wl_sample_r)
xlimits_cut_r = np.where((wl_sample_r>=xlim_lo) & (wl_sample_r<=xlim_hi))[0]
wl_cut_wfe1_r = wl_sample_r[xlimits_cut_r]
# Transmission, focusing on flat part of spectrum
xlim_lo, xlim_hi = min(wl_sample_t), max(wl_sample_t)
xlimits_cut_t = np.where((wl_sample_t>=xlim_lo) & (wl_sample_t<=xlim_hi))[0]
wl_cut_wfe1_t = wl_sample_t[xlimits_cut_t]

for i,rand_dep_error in enumerate(rand_dep_error_list_short):
    # Modelling the systematic error
    _,_,sys_dep_r_phase,_ = lsf_errors(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar,rand_dep_error)
    _,_,_,sys_dep_t_phase = lsf_errors(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar,rand_dep_error)
    # Difference in phase from nominal case, for R and T at different wl points
    sys_dep_r_phase_delta = np.real(nominal_r_phase-sys_dep_r_phase)        
    sys_dep_t_phase_delta = np.real(nominal_t_phase-sys_dep_t_phase)        
    # WFE conversion
    sys_dep_wfe_r[i,:] = phase_to_wfe(wl,sys_dep_r_phase_delta)
    sys_dep_wfe_t[i,:] = phase_to_wfe(wl,sys_dep_t_phase_delta)


#%% WAVEFRONT ERROR: Plotting wavefront error for a range of wavelengths

fig2,ax2 =plt.subplots(2,len(rand_dep_error_list_short),figsize=(12,5))
for i,rand_dep_error in enumerate(rand_dep_error_list_short):
    # plotting subplots in reverse
    n_ax = len(rand_dep_error_list_short)-(i+1)

    # Reflection    
    ax2[0,n_ax].plot(wl,sys_dep_wfe_r[i,:],'-',color='purple')
    ax2[0,n_ax].set_title('$\Delta d$ = '+str(100*rand_dep_error)+'%',fontsize=12)
    xlim_lo, xlim_hi = np.min(wl_sample_r), np.max(wl_sample_r) 
    xlimits_cut_r = np.where((wl>=xlim_lo)&(wl<=xlim_hi))
    ylim_lo = np.nanmin(sys_dep_wfe_r[i,:][xlimits_cut_r]) 
    ylim_hi = np.nanmax(sys_dep_wfe_r[i,:][xlimits_cut_r])
    ax2[0,n_ax].set_xlim(xlim_lo,xlim_hi)
    ax2[0,n_ax].set_ylim(ylim_lo,ylim_hi)
    ax2[0,n_ax].grid(visible=True) 
    ax2[0,n_ax].tick_params(axis='both', which='major', labelsize=12) 
    ax2[0,n_ax].set_ylabel('WFE in R at \n$\lambda=1.1\mu m$ [nm]',fontsize=12)
    ax2[0,n_ax].set_xlabel('$\lambda$ [nm]',fontsize=12)

    # Transmission
    ax2[1,n_ax].plot(wl,sys_dep_wfe_t[i,:],'-',color='purple')
    ax2[1,n_ax].set_title('$\Delta d$ = '+str(100*rand_dep_error)+'%',fontsize=12)
    xlim_lo, xlim_hi = np.min(wl_sample_t), np.max(wl_sample_t) # focus on flat part of spectrum
    xlimits_cut_t = np.where((wl>=xlim_lo)&(wl<=xlim_hi))
    ylim_lo = np.nanmin(sys_dep_wfe_t[i,:][xlimits_cut_t]) 
    ylim_hi = np.nanmax(sys_dep_wfe_t[i,:][xlimits_cut_t])
    ax2[1,n_ax].set_xlim(xlim_lo,xlim_hi)
    ax2[1,n_ax].set_ylim(ylim_lo,ylim_hi)
    ax2[1,n_ax].grid(visible=True) 
    ax2[1,n_ax].tick_params(axis='both', which='major', labelsize=12)
    ax2[1,n_ax].set_ylabel('WFE in T at \n$\lambda=5\mu m$ [nm]',fontsize=12)
    ax2[1,n_ax].set_xlabel('$\lambda$ [nm]',fontsize=12)
fig2.set_tight_layout(True)
fig2.savefig(f"{OutputFilePath}/Plots/Case2WFE_D1_front.pdf",dpi=600) # save figure


#%% MODELLING THE DICHOIRC SURFACE POINTS: simulating points on a disk for a defined sampling space

thick = np.concatenate((phys_l_front,phys_l_bbar)) # np.loadtxt(output_file, dtype=float)
nt = len(thick)
# error on thicknesses
t_err = 0.005
# radius of D1 substrate (mm)
rd = 16.5 # diameter is 33mm
# grid spacing step (or size of pixels, on one side) (mm) across surface plane of dichroic
dd = 1#0.1
# modelling the localised disk points
nside, xx, yy, x_size, y_size, mask = disk_points(thick,nt,t_err,rd,dd)

wl = np.array([860,1100,2400,2600,4000,5000,6700,7400]) # wavelength array of chosen points


#%% FIRST-ORDER GRADIENT

model = '1st_order_gradient'
orientation = 'random'

wl = np.array([860,1100,2400,2600,4000,5000,6700,7400]) # wavelength array of chosen points
# modelling a randomly oriented 1st order gradient
d_thick_xy, thick_xy_altered_1storder = spatial_deposition(wl,thick,nt,nside,xx,yy,mask,t_err,rd,dd,model,orientation,OutputFilePath)




# n_H = refr_index(config_multilayout['refr_index_h'],material_type,wl,temp)
# n_L = refr_index(config_multilayout['refr_index_l'],material_type,wl,temp)

# # Front Coating Refractive indices 
# n_incident = refr_index(config_multilayout['refr_index_incident'],material_type,wl,temp)
# n_substrate = refr_index(config_multilayout['refr_index_substrate'],material_type,wl,temp)
# n_front = multi_layout(wl,n_H,n_L,n_incident,n_substrate,layout_sequence_front)

# # BBAR Coating Refractive indices
# n_incident_rear = refr_index(config_multilayout['refr_index_substrate'],material_type,wl,temp)
# n_substrate_rear = refr_index(config_multilayout['refr_index_incident'],material_type,wl,temp)
# n_bbar = multi_layout(wl,n_H,n_L,n_incident_rear,n_substrate_rear,layout_sequence_rear)

# Nominal phase and spectral performance
nominal_r, nominal_t, _, nominal_r_phase, nominal_t_phase = spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)

# calculating the point-by-point phase error
phase_error_xy_1storder_r,phase_error_xy_1storder_t = phase_error_calc(wl,nside,x_size,y_size,len(phys_l_front),thick_xy_altered_1storder,phys_l_substrate,n_front,n_substrate,n_bbar,theta,polar,nominal_r_phase,nominal_t_phase)






