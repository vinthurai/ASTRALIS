#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run model for low spatial frequency (HSF) thickness errors for the phase study
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
import time

from scipy.stats import norm, skew, kurtosis
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
from scipy.stats import binned_statistic
from matplotlib.patches import Rectangle

# Local imports
from general_tools.refractive_index import refr_index
from general_tools.multilayer_structure import multi_layout
from tlm_model.analysis import phase_analysis
from phase_tools.wfe import phase_to_wfe
from uncertainty_tools.thickness import random_thickness_error,random_thickness_errors_spectrum,systematic_thickness_error
from tlm_model.analysis_scatter import scatter_analysis

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def annotation_line( ax, xmin, xmax, y, text, ytext=0, linecolor='black', linewidth=1.5, fontsize=12 ):

    ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '|-|', 'color':linecolor, 'linewidth':linewidth})
    ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '<->', 'color':linecolor, 'linewidth':linewidth})

    xcenter = xmin + (xmax-xmin)/2
    if ytext==0:
        ytext = y + ( ax.get_ylim()[1] - ax.get_ylim()[0] ) / 20

    ax.annotate( text, xy=(xcenter,ytext), ha='center', va='center', fontsize=fontsize)


def average_rays(spectrum,n_rays):
    av_spectrum = np.sum(spectrum,axis=0)/n_rays
    return av_spectrum

#%%###########################################################################################################################
####################################### Part 1: Nominal spectrum and import parameters #######################################
##############################################################################################################################
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
OutputFilePath = f"{param['DEFAULT']['outputfilepath']}/run_hsf_phase"
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
nominal_r, nominal_t, nominal_r_phase, nominal_t_phase = phase_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
nominal_r = average_rays(nominal_r,n_rays)
nominal_t = average_rays(nominal_t,n_rays)

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
fig.savefig(f"{OutputFilePath}/Plots/nominal_spectrum.png",dpi=600)

#%%####################################################################################################################################
####################################### Part 2: Unlocalised phase simulations (Case 1 & Case 2) #######################################
#######################################################################################################################################
#%% Deposition Cases definition

# Case 1
def depo_case1(wl,rand_dep_error,n_sample,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar):
    # generating random error in thicknesses of ARC layers
    # inputs: wavelengths,thickness array,refractive index array,AOI,
            # template/ideal reflection spectrum,% error,array of indices of layers with errors
    # outputs: residual between template and generated spectrums, 
             # generated reflection spectrum resulting from inclusion of erros, 
             # array of generated errors, array of % errors for each layer
    
    # layers which are affected by a deposition error
    rand_dep_error_layers = 'all'
    # Generate random errors in thickness
    phys_l_front_rand, phys_l_rear_rand  = random_thickness_error(phys_l_front,phys_l_bbar,n_sample,rand_dep_error,rand_dep_error_layers)
    # Calculate the spectral performance for randomly altered thicknesses
    rand_dep_r, rand_dep_t, rand_dep_r_phase, rand_dep_t_phase = random_thickness_errors_spectrum(wl,n_front,n_substrate,n_bbar,phys_l_front_rand,phys_l_substrate,phys_l_rear_rand,theta,polar)
    
    if len(wl)==1:
        rand_dep_r = rand_dep_r[:,0]
        rand_dep_t = rand_dep_t[:,0]
        rand_dep_r_phase = rand_dep_r_phase[:,0]
        rand_dep_t_phase = rand_dep_t_phase[:,0]
    
    return rand_dep_r,rand_dep_t,rand_dep_r_phase,rand_dep_t_phase,phys_l_front_rand,phys_l_rear_rand


# Case 2
def depo_case2(wl,sys_dep_error,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar):
 
    # generating random error in thicknesses of ARC layers
    # inputs: wavelengths,thickness array,refractive index array,AOI,
            # template/ideal reflection spectrum,% error,array of indices of layers with errors
    # outputs: residual between template and generated spectrums, 
              # generated reflection spectrum resulting from inclusion of erros, 
              # array of generated errors, array of % errors for each layer   
    
    # Generate a systematic error of +error% in all layers & resulting change in performance
    systematic_type = 'surplus'
    # Find spectral performance
    sys_dep_r,sys_dep_t,sys_dep_r_phase,sys_dep_t_phase,phys_l_front_sys,phys_l_rear_sys = systematic_thickness_error(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar,sys_dep_error,systematic_type)

    return sys_dep_r,sys_dep_t,sys_dep_r_phase,sys_dep_t_phase,phys_l_front_sys,phys_l_rear_sys


#%% Setting up MC

# testing different tolerances
rand_dep_error_list = np.array([0.001,0.002,0.003,0.005,0.01,0.02])[::-1] # parametrically varying random errors
n_tolerances = len(rand_dep_error_list)
max_thick = np.max([np.max(phys_l_front),np.max(phys_l_bbar)])
ind_thick = np.where(phys_l_front == max_thick)[0][0] # find index of thickest layer #!!!! change this to adapt if it's the front or bbar coating 

wl_point = np.array([1100,5000]) #nm #chosen wavelength
ind_wl = np.where(np.isin(wl, wl_point))[0] # finding indices of wavelength values chosen

n_point_front = n_front[ind_wl,:] #!!!! change this to adapt if it's the front or bbar coating 
n_point_substrate = n_substrate[ind_wl]
n_point_bbar = n_bbar[ind_wl,:] #!!!! change this to adapt if it's the front or bbar coating 

# Resulting phase and spectral performance at wl_point
nominal_r_point, nominal_t_point, nominal_r_phase_point, nominal_t_phase_point = phase_analysis(wl_point,n_point_front,n_point_substrate,n_point_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)


# Case 1 error
# number of randomly generated thickness errors
n_sample = syst_param['general']['n_sample']['value'] 

# Case 2 error
# deposition uncertainty of machine in percent
sys_dep_error = syst_param['manufacturing']['systematic_dep_error']['value']


#%% Setting up empty arrays for running MC – used to plot figures

# Case 1
# Spectral performance
rand_dep_r = np.empty((n_tolerances,n_sample,len(wl_point)))
rand_dep_t = np.empty((n_tolerances,n_sample,len(wl_point)))
# Changed recipes
phys_l_front_rand = np.empty((n_tolerances,n_sample,len(phys_l_front)))
phys_l_rear_rand = np.empty((n_tolerances,n_sample,len(phys_l_bbar)))
# Phase changes
rand_dep_r_phase = np.empty_like(rand_dep_r)
rand_dep_t_phase = np.empty_like(rand_dep_t)
# Phase difference from nominal case
rand_dep_r_phase_delta = np.empty((n_tolerances,n_sample))
rand_dep_t_phase_delta = np.empty((n_tolerances,n_sample))

# Case 2
# Spectral performance
sys_dep_r = np.empty((n_tolerances,len(wl_point)))
sys_dep_t = np.empty((n_tolerances,len(wl_point)))
# Changed recipes
phys_l_front_sys = np.empty((n_tolerances,len(phys_l_front)))
phys_l_rear_sys = np.empty((n_tolerances,len(phys_l_bbar)))
# Phase changes
sys_dep_r_phase = np.empty_like(sys_dep_r)
sys_dep_t_phase = np.empty_like(sys_dep_t)
# Phase difference from nominal case
sys_dep_r_phase_delta = np.empty(n_tolerances)
sys_dep_t_phase_delta = np.empty(n_tolerances)


#%% Running MC & write to file
for i,rand_dep_error in enumerate(rand_dep_error_list):
        
        # Depo Case 1
        # rand_dep_r_val,rand_dep_t_val,rand_dep_r_phase[i,:,:],rand_dep_t_phase[i,:,:],phys_l_front_rand[i,:,:],phys_l_rear_rand[i,:,:] = depo_case1(wl_point,rand_dep_error,n_sample,n_point_front,n_point_substrate,n_point_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
        # Depo Case 2 
        sys_dep_r_val,sys_dep_t_val,sys_dep_r_phase[i,:],sys_dep_t_phase[i,:],phys_l_front_sys[i,:],phys_l_rear_sys[i,:] = depo_case2(wl_point,rand_dep_error,n_point_front,n_point_substrate,n_point_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
        
        # rand_dep_r[i,:,:] = average_rays(rand_dep_r_val, n_rays)
        # rand_dep_t[i,:,:] = average_rays(rand_dep_t_val, n_rays)
        sys_dep_r[i,:] = average_rays(sys_dep_r_val, n_rays)
        sys_dep_t[i,:] = average_rays(sys_dep_t_val, n_rays)
        
        # Difference in phase from nominal case, for R and T at different wl points
        # rand_dep_r_phase_delta[i,:] = np.real(nominal_r_phase_point[0]-rand_dep_r_phase[i,:,0])       
        # rand_dep_t_phase_delta[i,:] = np.real(nominal_t_phase_point[1]-rand_dep_t_phase[i,:,1]) 
        sys_dep_r_phase_delta[i] = np.real(nominal_r_phase_point[0]-sys_dep_r_phase[i,0])       
        sys_dep_t_phase_delta[i] = np.real(nominal_t_phase_point[1]-sys_dep_t_phase[i,1]) 
        
        # ################################### write to file ###################################
        
        # rand_dep_error_write = np.full((n_sample,1),100*rand_dep_error)
        
        # # one new file for each MC run of Case 1
        # column_names = ['Thickness error (%)']
        # for ind_phys,phys in enumerate(range(len(phys_l_front))):
        #     column_names.append('Front Coating no.'+str(ind_phys+1)+' thickness (nm)') 
        # for ind_phys,phys in enumerate(range(len(phys_l_bbar))):
        #     column_names.append('BBAR Coating no.'+str(ind_phys+1)+' thickness (nm)') 
        # column_names.append('Phase difference ($\degree$)')
        # column_names.append('Performance [R/T] (%)')
        
        # phys_l_front_error_write = phys_l_front_rand[i,:,:]
        # phys_l_bbar_error_write = phys_l_rear_rand[i,:,:] 
        
        # ################# ----- Reflection ----- #################
        
        # phase_error_write_r = rand_dep_r_phase_delta[i,:] #Case 1
        # perf_error_write_r = rand_dep_r[i,:,0]
        # # create data cube to write to file
        # data_write_r = np.hstack((rand_dep_error_write,phys_l_front_error_write,phys_l_bbar_error_write,phase_error_write_r[:,None],perf_error_write_r[:,None]))
        
        # # write file 
        # filename_r = 'D1_'+str(100*rand_dep_error)+'_lambda'+str(wl_point[0])+'_%t_error_'+str(n_sample)+'MC_runs_Case1_R.csv'
        # df_r = pd.DataFrame(data_write_r, columns = column_names)    
        # df_r.to_csv(f"{OutputFilePath}/Phase Error/{filename_r}",index=False)

        # ################# ----- Transmission ----- #################

        # phase_error_write_t = rand_dep_t_phase_delta[i,:] #Case 1
        # perf_error_write_t = rand_dep_t[i,:,1]
        # # create data cube to write to file
        # data_write_t = np.hstack((rand_dep_error_write,phys_l_front_error_write,phys_l_bbar_error_write,phase_error_write_t[:,None],perf_error_write_t[:,None]))
        
        # # write file 
        # filename_t = 'D1_'+str(100*rand_dep_error)+'_lambda'+str(wl_point[1])+'_%t_error_'+str(n_sample)+'MC_runs_Case1_T.csv'
        # df_t = pd.DataFrame(data_write_t, columns = column_names)    
        # df_t.to_csv(f"{OutputFilePath}/Phase Error/{filename_t}",index=False)
                  
        # ########################### end of file writing ##########################


#%% Read data from files

# Case 1
# Spectral performance
rand_dep_r = np.empty((n_tolerances,n_sample,len(wl_point)))
rand_dep_t = np.empty((n_tolerances,n_sample,len(wl_point)))
# Changed recipes
phys_l_front_rand = np.empty((n_tolerances,n_sample,len(phys_l_front)))
phys_l_rear_rand = np.empty((n_tolerances,n_sample,len(phys_l_bbar)))
# Phase changes
rand_dep_r_phase = np.empty_like(rand_dep_r)
rand_dep_t_phase = np.empty_like(rand_dep_t)
# Phase difference from nominal case
rand_dep_r_phase_delta = np.empty((n_tolerances,n_sample))
rand_dep_t_phase_delta = np.empty((n_tolerances,n_sample))

for i,rand_dep_error in enumerate(rand_dep_error_list):

    filename_r = 'D1_'+str(100*rand_dep_error)+'_lambda'+str(wl_point[0])+'_%t_error_'+str(n_sample)+'MC_runs_Case1_R.csv'
    df_r = pd.read_csv(f"{OutputFilePath}/Phase Error/{filename_r}",delimiter=',')
    filename_t = 'D1_'+str(100*rand_dep_error)+'_lambda'+str(wl_point[1])+'_%t_error_'+str(n_sample)+'MC_runs_Case1_T.csv'
    df_t = pd.read_csv(f"{OutputFilePath}/Phase Error/{filename_t}",delimiter=',')

    rand_dep_r_phase_delta[i,:] = df_r['Phase difference ($\degree$)'].values
    rand_dep_t_phase_delta[i,:] = df_t['Phase difference ($\degree$)'].values
    rand_dep_r[i,:,0] = df_r['Performance [R/T] (%)'].values
    rand_dep_t[i,:,1] = df_t['Performance [R/T] (%)'].values


    # one new file for each MC run of Case 1
    column_names = ['Thickness error (%)']
    for ind_phys,phys in enumerate(range(len(phys_l_front))):
        column_name_thick = 'Front Coating no.'+str(ind_phys+1)+' thickness (nm)'
        phys_l_front_rand[i,:,ind_phys] = df_r[column_name_thick].values
    for ind_phys,phys in enumerate(range(len(phys_l_bbar))):
        column_name_thick = 'BBAR Coating no.'+str(ind_phys+1)+' thickness (nm)'
        phys_l_rear_rand[i,:,ind_phys] = df_t[column_name_thick].values


#%% Figures:

rand_dep_error_list_short = np.array([0.001,0.005,0.01])[::-1] # parametrically varying random errors
ind_rand_dep = np.where(np.isin(rand_dep_error_list, rand_dep_error_list_short))[0] 

# Phase differences due to error on thickness scatter histograms
fig1, ax1 = plt.subplots(2,6,figsize=(15,5))
ax1[0,0].set_ylabel('Frequency',fontsize=12)
ax1[1,0].set_ylabel(r'Change in $\Sigma d_i$ [nm]',fontsize=12)

ax1[0,3].set_ylabel('Frequency',fontsize=12)
ax1[1,3].set_ylabel(r'Change in $\Sigma d_i$ [nm]',fontsize=12)


ax1[1,1].set_xlabel(r'$\Delta \phi_R$ [$\degree$]',fontsize=12)

ax1[1,4].set_xlabel(r'$\Delta \phi_T$ [$\degree$]',fontsize=12)


pl_hold = 2
pl_hold_t = 5
for i,rand_dep_error in zip(ind_rand_dep,rand_dep_error_list_short): 
    
    
    # Reflection
    
    # pl_hold = pl_hold-(i+1) # plotting subplots in reverse
    ax1[0,pl_hold].hist(rand_dep_r_phase_delta[i,:],bins=17,density=True,edgecolor="black",alpha=0.5,facecolor='lightgreen')
    # Fit a normal distribution to the data:
    xmin, xmax = ax1[0,pl_hold].get_xlim()
    ymin_ax, ymax_ax = ax1[0,pl_hold].get_ylim()
    mu, std = norm.fit(rand_dep_r_phase_delta[i,:])
    x = np.linspace(np.min(rand_dep_r_phase_delta[i,:]),np.max(rand_dep_r_phase_delta[i,:]),1000)#np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    ax1[0,pl_hold].plot(x, p, linewidth=1.5,alpha=0.5)
        
    ax1[0,pl_hold].set_title('$\Delta d$ = '+str(100*rand_dep_error)+'%',fontsize=12)
    # ax1[0,pl_hold].set_xlim([xmin,xmax])
    ax1[0,pl_hold].set_xlim([-sys_dep_r_phase_delta[i]*1.08,sys_dep_r_phase_delta[i]*1.08])

    ax1[0,pl_hold].vlines(x=mu,ymin = 0,ymax =np.max(p),color='k',linewidth=1.5)
    ax1[0,pl_hold].vlines(x=mu-std,ymin = 0,ymax =p[find_nearest(x, mu-std)],color='g',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold].vlines(x=mu+std,ymin = 0,ymax =p[find_nearest(x, mu+std)],color='g',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold].vlines(x=mu-2*std,ymin = 0,ymax =p[find_nearest(x, mu-2*std)],color='g',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold].vlines(x=mu+2*std,ymin = 0,ymax =p[find_nearest(x, mu+2*std)],color='g',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold].fill_between(x[np.where((x>mu-std)&(x<mu+std))],p[np.where((x>mu-std)&(x<mu+std))], facecolor='darkgreen', alpha=.3)
    ax1[0,pl_hold].fill_between(x[np.where((x>mu-2*std)&(x<mu+2*std))],p[np.where((x>mu-2*std)&(x<mu+2*std))], facecolor='green', alpha=.2)
    
    ax1[0,pl_hold].vlines(x=sys_dep_r_phase_delta[i],ymin = 0,ymax = np.max(p)*1.06,color='magenta',linestyle='-.',linewidth=1.5)
    ax1[0,pl_hold].set_ylim(0,np.max(p)*1.07)
    
    # # Phase differences due to error on thickness scatter
    ax1[1,pl_hold].scatter(rand_dep_r_phase_delta[i,:], (np.sum(phys_l_front_rand[i,:,:],axis=1)+np.sum(phys_l_rear_rand[i,:,:],axis=1)) - (np.sum(phys_l_front)+np.sum(phys_l_bbar)), alpha=0.8,color='green',label='Case 1')
    ax1[1,pl_hold].scatter(sys_dep_r_phase_delta[i],(np.sum(phys_l_front_sys[i,:])+np.sum(phys_l_rear_sys[i,:])) - (np.sum(phys_l_front)+np.sum(phys_l_bbar)),marker='D',s = 35,color='magenta',label='Case 2')
    ax1[1,pl_hold].set_xlim([-sys_dep_r_phase_delta[i]*1.08,sys_dep_r_phase_delta[i]*1.08])

    pl_hold = pl_hold - 1
    
    
    # Transmission
    # pl_hold_t = pl_hold_t-(i+1) # plotting subplots in reverse
    
    ax1[0,pl_hold_t].hist(rand_dep_t_phase_delta[i,:],bins=17,density=True,edgecolor="black",alpha=0.5,facecolor='lightblue')
    # Fit a normal distribution to the data:
    xmin, xmax = ax1[0,pl_hold_t].get_xlim()
    ymin_ax, ymax_ax = ax1[0,pl_hold_t].get_ylim()
    mu, std = norm.fit(rand_dep_t_phase_delta[i,:])
    x = np.linspace(np.min(rand_dep_t_phase_delta[i,:]),np.max(rand_dep_t_phase_delta[i,:]),1000)#np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    ax1[0,pl_hold_t].plot(x, p, color='hotpink', linewidth=1.5,alpha=0.5)
    
    ax1[0,pl_hold_t].set_title('$\Delta d$ = '+str(100*rand_dep_error)+'%',fontsize=12)
    # ax1[0,pl_hold_t].set_xlim([xmin,xmax])
    ax1[0,pl_hold_t].set_xlim([-sys_dep_t_phase_delta[i]*1.08,sys_dep_t_phase_delta[i]*1.08])

    ax1[0,pl_hold_t].vlines(x=mu,ymin = 0,ymax =np.max(p),color='k',linewidth=1.5)
    ax1[0,pl_hold_t].vlines(x=mu-std,ymin = 0,ymax =p[find_nearest(x, mu-std)],color='b',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold_t].vlines(x=mu+std,ymin = 0,ymax =p[find_nearest(x, mu+std)],color='b',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold_t].vlines(x=mu-2*std,ymin = 0,ymax =p[find_nearest(x, mu-2*std)],color='b',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold_t].vlines(x=mu+2*std,ymin = 0,ymax =p[find_nearest(x, mu+2*std)],color='b',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold_t].fill_between(x[np.where((x>mu-std)&(x<mu+std))],p[np.where((x>mu-std)&(x<mu+std))], facecolor='darkblue', alpha=.3)
    ax1[0,pl_hold_t].fill_between(x[np.where((x>mu-2*std)&(x<mu+2*std))],p[np.where((x>mu-2*std)&(x<mu+2*std))], facecolor='blue', alpha=.2)
    
    ax1[0,pl_hold_t].vlines(x=sys_dep_t_phase_delta[i],ymin = 0,ymax = np.max(p)*1.06,color='red',linestyle='-.',linewidth=1.5)
    ax1[0,pl_hold_t].set_ylim(0,np.max(p)*1.07)
    
    # # Phase differences due to error on thickness scatter
    ax1[1,pl_hold_t].scatter(rand_dep_t_phase_delta[i,:], (np.sum(phys_l_front_rand[i,:,:],axis=1)+np.sum(phys_l_rear_rand[i,:,:],axis=1)) - (np.sum(phys_l_front)+np.sum(phys_l_bbar)), alpha=0.8,color='darkblue',label='Case 1')
    ax1[1,pl_hold_t].scatter(sys_dep_t_phase_delta[i],(np.sum(phys_l_front_sys[i,:])+np.sum(phys_l_rear_sys[i,:])) - (np.sum(phys_l_front)+np.sum(phys_l_bbar)),marker='D',s = 35,color='red',label='Case 2')
    ax1[1,pl_hold_t].set_xlim([-sys_dep_t_phase_delta[i]*1.08,sys_dep_t_phase_delta[i]*1.08])
    pl_hold_t = pl_hold_t - 1    

    # Increase tick label sizes
    ax1[1,pl_hold_t].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the first subplot
    ax1[0,pl_hold_t].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the second subplot
    ax1[1,pl_hold].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the first subplot
    ax1[0,pl_hold].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the second subplot



# fig1.supxlabel('Phase difference [$\degree$]',fontsize=12)
fig1.set_tight_layout(True)
plt.savefig(f"{OutputFilePath}/Plots/Case1_Scatter&Hist_D1_front.png",dpi=400)


#%% Figures: Thesis

# Phase differences due to error on thickness scatter histograms
fig1, ax1 = plt.subplots(2,n_tolerances,figsize=(15,7))

fig4, ax4 = plt.subplots(4,sharex=True,figsize=(6,5))

ax4[0].set_ylabel('Mean [$\degree$]')
ax4[1].set_ylabel('Standard \n deviation [$\degree$]')
ax4[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax4[0].grid(visible=True)
ax4[1].grid(visible=True)

ax1[0,0].set_ylabel('Frequency')
ax1[1,0].set_ylabel('Change in Total Thickness [nm]')

for i,rand_dep_error in enumerate(rand_dep_error_list):
    pl_hold = n_tolerances-(i+1) # plotting subplots in reverse
    ax1[0,pl_hold].hist(rand_dep_r_phase_delta[i,:],bins=17,density=True,edgecolor="black",alpha=0.5,facecolor='lightgreen')
    # Fit a normal distribution to the data:
    xmin, xmax = ax1[0,pl_hold].get_xlim()
    ymin_ax, ymax_ax = ax1[0,pl_hold].get_ylim()
    mu, std = norm.fit(rand_dep_r_phase_delta[i,:])
    x = np.linspace(np.min(rand_dep_r_phase_delta[i,:]),np.max(rand_dep_r_phase_delta[i,:]),1000)#np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    ax1[0,pl_hold].plot(x, p, linewidth=1.5,alpha=0.5)
        
    ax1[0,pl_hold].set_title('$\Delta d$ = '+str(100*rand_dep_error)+'%',fontsize=12)
    ax1[0,pl_hold].set_xlim([xmin,xmax])

    ax1[0,pl_hold].vlines(x=mu,ymin = 0,ymax =np.max(p),color='k',linewidth=1.5)
    ax1[0,pl_hold].vlines(x=mu-std,ymin = 0,ymax =p[find_nearest(x, mu-std)],color='g',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold].vlines(x=mu+std,ymin = 0,ymax =p[find_nearest(x, mu+std)],color='g',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold].vlines(x=mu-2*std,ymin = 0,ymax =p[find_nearest(x, mu-2*std)],color='g',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold].vlines(x=mu+2*std,ymin = 0,ymax =p[find_nearest(x, mu+2*std)],color='g',linestyle='--',linewidth=1.5)
    ax1[0,pl_hold].fill_between(x[np.where((x>mu-std)&(x<mu+std))],p[np.where((x>mu-std)&(x<mu+std))], facecolor='darkgreen', alpha=.3)
    ax1[0,pl_hold].fill_between(x[np.where((x>mu-2*std)&(x<mu+2*std))],p[np.where((x>mu-2*std)&(x<mu+2*std))], facecolor='green', alpha=.2)
    
    ax1[0,pl_hold].vlines(x=sys_dep_r_phase_delta[i],ymin = 0,ymax = np.max(p)*1.06,color='magenta',linestyle='-.',linewidth=1.5)
    ax1[0,pl_hold].set_ylim(0,np.max(p)*1.07)
    
    # # Phase differences due to error on thickness scatter
    ax1[1,pl_hold].scatter(rand_dep_r_phase_delta[i,:], (np.sum(phys_l_front_rand[i,:,:],axis=1)+np.sum(phys_l_rear_rand[i,:,:],axis=1)) - (np.sum(phys_l_front)+np.sum(phys_l_bbar)), alpha=0.8,color='green',label='Case 1')
    ax1[1,pl_hold].scatter(sys_dep_r_phase_delta[i],(np.sum(phys_l_front_sys[i,:])+np.sum(phys_l_rear_sys[i,:])) - (np.sum(phys_l_front)+np.sum(phys_l_bbar)),marker='D',s = 35,color='magenta',label='Case 2')

    ax4[0].plot(100*rand_dep_error,mu,'bo')
    ax4[1].plot(100*rand_dep_error,std,'bo')

# More Gaussian fit parameters Skewness and Kurtosis of phase
for i,rand_dep_error in enumerate(rand_dep_error_list):
    ax4[2].plot(100*rand_dep_error,skew(rand_dep_r_phase_delta[i,:]),'bo')    
    ax4[3].plot(100*rand_dep_error,kurtosis(rand_dep_r_phase_delta[i,:]),'bo')

ax4[2].set_ylabel('Skewness')
ax4[3].set_ylabel('Kurtosis')
ax4[3].set_xlabel('$\Delta d$ [%]')     
ax4[2].grid(visible=True) 
ax4[3].grid(visible=True) 
ax4[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax4[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig4.set_tight_layout(True)
# plt.savefig('Unlocalised_phase_and_scatter_output_files/Case1_Gaussian_params_D1_front.png',dpi=300,transparent=True)

fig1.supxlabel('Phase difference [$\degree$]',fontsize=12)
fig1.set_tight_layout(True)
plt.savefig(f"{OutputFilePath}/Case1_Gaussian_params_D1_front.pdf",dpi=600)

# plt.savefig('Unlocalised_phase_and_scatter_output_files/Case1_Scatter&Hist_D1_front.pdf',dpi=300,transparent=True)



#%% Effect of maximum thickness on Case 1

color_list = ['brown','gold','darkgreen','darkblue','plum','purple']

fig6, ax6 = plt.subplots(2,figsize=(8,5))
for (i,rand_dep_error),col in zip(enumerate(rand_dep_error_list),color_list):
    ax6[0].scatter(phys_l_front_rand[i,:,ind_thick] - max_thick,rand_dep_r_phase_delta[i,:], color=col, marker='o', label='$\Delta d$ = '+str(100*rand_dep_error)+'%', alpha=0.8)
    ax6[1].scatter(phys_l_front_rand[i,:,ind_thick] - max_thick,rand_dep_t_phase_delta[i,:], color=col, marker='o', label='$\Delta d$ = '+str(100*rand_dep_error)+'%', alpha=0.8)

ax6[0].set_ylabel(r'$\Delta \phi_R$ [$\degree$]',fontsize=12)
ax6[1].set_ylabel(r'$\Delta \phi_T$ [$\degree$]',fontsize=12)
ax6[1].set_xlabel(r'Change in largest $d_i$ [nm]',fontsize=12)
#reversing order of legend
handles, labels = ax6[0].get_legend_handles_labels()
leg=ax6[0].legend(handles[::-1], labels[::-1],fontsize=11)
leg.get_frame().set_alpha(0.9)  # Adjust the alpha value (0.0 to 1.0) for transparency
ax6[1].vlines(0,-55,55,'k',linestyle='--',linewidth=1.5)
ax6[0].vlines(0,-55,55,'k',linestyle='--',linewidth=1.5)

xlim_lo, xlim_hi = np.min(phys_l_front_rand[:,:,ind_thick]-max_thick), np.max(phys_l_front_rand[:,:,ind_thick]-max_thick)
ylim_lo = np.min(rand_dep_r_phase_delta) 
ylim_hi = np.max(rand_dep_r_phase_delta)
ax6[0].set_xlim(xlim_lo-2,xlim_hi+2)
ax6[0].set_ylim(ylim_lo-2,ylim_hi+2)
ax6[0].axhline(y=0, color='black', linestyle='-.') # thin black dashed line to help higlight y=0 axis
# Increase tick label sizes
plt.xticks(fontsize=12)  # Adjust the font size for the x-axis ticks
plt.yticks(fontsize=12)  # Adjust the font size for the y-axis ticks

xlim_lo, xlim_hi = np.min(phys_l_front_rand[:,:,ind_thick]-max_thick), np.max(phys_l_front_rand[:,:,ind_thick]-max_thick)
ylim_lo = np.min(rand_dep_t_phase_delta) 
ylim_hi = np.max(rand_dep_t_phase_delta)
ax6[1].set_xlim(xlim_lo-2,xlim_hi+2)
ax6[1].set_ylim(ylim_lo-2,ylim_hi+2)
ax6[1].axhline(y=0, color='black', linestyle='-.') # thin black dashed line to help higlight y=0 axis

# Increase tick label sizes
plt.xticks(fontsize=12)  # Adjust the font size for the x-axis ticks
plt.yticks(fontsize=12)  # Adjust the font size for the y-axis ticks

ax6[0].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the first subplot
ax6[1].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the second subplot


fig6.set_tight_layout(True)
# fig6.text(0.06, 0.5, 'Phase difference [$\degree$]', va='center', rotation='vertical')
plt.savefig(f"{OutputFilePath}/Plots/Case1_Max_thickness_D1_front.png",dpi=400)


#%%####################################################################################################################################
####################################### Part 2.5: Unlocalised phase simulations (Case 1 & Case 2) #######################################
#######################################################################################################################################
#%% WFE Analysis Case 1

wl_sample_r = wl[np.where((wl>=1550)&(wl<=1750))]
wl_sample_t = wl[np.where((wl>=5900)&(wl<=6100))]

ind_wl_r = np.where(np.isin(wl, wl_sample_r))[0] # finding indices of wavelength values chosen
ind_wl_t = np.where(np.isin(wl, wl_sample_t))[0] # finding indices of wavelength values chosen
n_point_front_r = n_front[ind_wl_r,:] #!!!! change this to adapt if it's the front or bbar coating 
n_point_front_t = n_front[ind_wl_t,:] #!!!! change this to adapt if it's the front or bbar coating 
n_point_substrate_r = n_substrate[ind_wl_r]
n_point_substrate_t = n_substrate[ind_wl_t]
n_point_bbar_r = n_bbar[ind_wl_r,:] #!!!! change this to adapt if it's the front or bbar coating 
n_point_bbar_t = n_bbar[ind_wl_t,:] #!!!! change this to adapt if it's the front or bbar coating 

# Resulting phase and spectral performance at wl_point
_, _, nominal_r_phase_rsample, _ = phase_analysis(wl_sample_r,n_point_front_r,n_point_substrate_r,n_point_bbar_r,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
_, _, _, nominal_t_phase_tsample = phase_analysis(wl_sample_t,n_point_front_t,n_point_substrate_t,n_point_bbar_t,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)



#%% Setting up empty arrays for running MC – used to plot figures – WFE

# Case 1

iteration = int(50)

rand_dep_error_list_short = np.array([0.001,0.005,0.01])[::-1] # parametrically varying random errors

rand_dep_error_list_short = np.array([0.005])

for i,rand_dep_error in enumerate(rand_dep_error_list_short):

    filename_wfe_r = 'WFE_Case1_error'+str(rand_dep_error)+'R.csv'
    filename_wfe_t = 'WFE_Case1_error'+str(rand_dep_error)+'R.csv'
    file_path_r = f"{OutputFilePath}/WFE/{filename_wfe_r}"
    file_path_t = f"{OutputFilePath}/WFE/{filename_wfe_t}"

    try:
        rand_dep_wfe_data_r_df = pd.read_csv(file_path_r,delimiter=',')
        runs_completed = rand_dep_wfe_data_r_df.shape[1]-1
        print(runs_completed)
    except FileNotFoundError:
        runs_completed = 0

        
    while runs_completed <= iteration:

        try:
            rand_dep_wfe_data_r_df = pd.read_csv(file_path_r,delimiter=',')
            runs_completed = rand_dep_wfe_data_r_df.shape[1]-1
            print(runs_completed)
        except FileNotFoundError:
            runs_completed = 0
        
        remaining_runs = n_sample - runs_completed
        runs_single = int(n_sample/10000)
        current_runs = min(runs_single, remaining_runs)

        # Depo Case 1
        _,_,rand_dep_r_phase,_,_,_ = depo_case1(wl_sample_r,rand_dep_error,current_runs,n_point_front_r,n_point_substrate_r,n_point_bbar_r,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
        _,_,_,rand_dep_t_phase,_,_ = depo_case1(wl_sample_t,rand_dep_error,current_runs,n_point_front_t,n_point_substrate_t,n_point_bbar_t,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
        
        # Difference in phase from nominal case, for R and T at different wl points
        rand_dep_r_phase_delta = np.real(nominal_r_phase_rsample-rand_dep_r_phase)       
        rand_dep_t_phase_delta = np.real(nominal_t_phase_tsample-rand_dep_t_phase)          
    
        # WFE conversion
        rand_dep_wfe_r = phase_to_wfe(wl_sample_r,rand_dep_r_phase_delta)
        rand_dep_wfe_t = phase_to_wfe(wl_sample_t,rand_dep_t_phase_delta)

        #Calculate WFE and write to file
        column_names = ['wavelength[nm]']
        for col_val in range(runs_single):
            column_names.append('WFE[nm]') 
        
        # Save data to file
        rand_dep_wfe_r_data = np.vstack([wl_sample_r[None,:],rand_dep_wfe_r]).T
        rand_dep_wfe_t_data = np.vstack([wl_sample_t[None,:],rand_dep_wfe_t]).T
        df_wfe_r = pd.DataFrame(rand_dep_wfe_r_data, columns = column_names)
        df_wfe_t = pd.DataFrame(rand_dep_wfe_t_data, columns = column_names)



        if os.path.exists(filename_wfe_r):
            existing_df = pd.read_csv(filename_wfe_r)
            
            combined_df = pd.concat([existing_df, df_wfe_r], axis=1)
            combined_df.to_csv(file_path_r,index=False)
        else:
            df_wfe_r.to_csv(file_path_r,index=False)


        if os.path.exists(filename_wfe_t):
            existing_df = pd.read_csv(filename_wfe_t)
            combined_df = pd.concat([existing_df, df_wfe_t], axis=1)
            combined_df.to_csv(f"{OutputFilePath}/WFE/{filename_wfe_t}",index=False)
        else:
            df_wfe_t.to_csv(f"{OutputFilePath}/WFE/{filename_wfe_t}",index=False)



        runs_completed += current_runs
        
        if runs_completed >= n_sample:
            break




#%% !!! Setting up empty arrays for running MC – used to plot figures – WFE

start = time.time()

# Case 1
rand_dep_wfe_r = np.empty((n_tolerances,n_sample,len(wl_sample_r)))
rand_dep_wfe_t = np.empty((n_tolerances,n_sample,len(wl_sample_t)))

#Calculate WFE and write to file
column_names_r = []
for col_val in range(len(wl_sample_r)):
    column_names_r.append("wavelength="+str(wl_sample_r[col_val])+"nm") 
column_names_t = []
for col_val in range(len(wl_sample_r)):
    column_names_t.append("wavelength="+str(wl_sample_t[col_val])+"nm") 


rand_dep_error_list_short = np.array([0.001,0.005,0.01])[::-1] # parametrically varying random errors

for i,rand_dep_error in enumerate(rand_dep_error_list_short):

    filename_wfe_r = 'WFE_Case1_error'+str(rand_dep_error)+'R.csv'
    filename_wfe_t = 'WFE_Case1_error'+str(rand_dep_error)+'T.csv'
    file_path_r = f"{OutputFilePath}/WFE/{filename_wfe_r}"
    file_path_t = f"{OutputFilePath}/WFE/{filename_wfe_t}"


    # Depo Case 1
    _,_,rand_dep_r_phase,_,_,_ = depo_case1(wl_sample_r,rand_dep_error,n_sample,n_point_front_r,n_point_substrate_r,n_point_bbar_r,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
    # Difference in phase from nominal case, for R and T at different wl points
    rand_dep_r_phase_delta = np.real(nominal_r_phase_rsample-rand_dep_r_phase)       
    # WFE conversion
    rand_dep_wfe_r[i,:,:] = phase_to_wfe(wl_sample_r,rand_dep_r_phase_delta)
    rand_dep_wfe_r_data = rand_dep_wfe_r[i,:,:]

    if os.path.exists(file_path_r):
        existing_df = pd.read_csv(file_path_r,delimiter=',').values
        rand_dep_wfe_r_data = np.concatenate([existing_df, rand_dep_wfe_r_data], axis=0)
    else:
        # Save data to file
        rand_dep_wfe_r_data = rand_dep_wfe_r[i,:,:]

    df_wfe_r = pd.DataFrame(rand_dep_wfe_r_data, columns = column_names_r)
    df_wfe_r.to_csv(file_path_r,index=False)


    
    # Depo Case 1
    _,_,_,rand_dep_t_phase,_,_ = depo_case1(wl_sample_t,rand_dep_error,n_sample,n_point_front_t,n_point_substrate_t,n_point_bbar_t,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
    # Difference in phase from nominal case, for R and T at different wl points
    rand_dep_t_phase_delta = np.real(nominal_t_phase_tsample-rand_dep_t_phase)  
    # WFE conversion
    rand_dep_wfe_t[i,:,:] = phase_to_wfe(wl_sample_t,rand_dep_t_phase_delta)
    rand_dep_wfe_t_data = rand_dep_wfe_t[i,:,:]

    if os.path.exists(file_path_t):
        existing_df = pd.read_csv(file_path_t,delimiter=',').values
        rand_dep_wfe_t_data = np.concatenate([existing_df, rand_dep_wfe_t_data], axis=0)
    else:
        # Save data to file
        rand_dep_wfe_t_data = rand_dep_wfe_t[i,:,:]
    
    rand_dep_wfe_t_data = rand_dep_wfe_t[i,:,:]
    df_wfe_t = pd.DataFrame(rand_dep_wfe_t_data, columns = column_names_t)
    df_wfe_t.to_csv(file_path_t,index=False)

stop = time.time()

print(f"Execution time: {stop-start}")

#%% Read data for WFE Case 1
rand_dep_error_list_short = np.array([0.001,0.005,0.01])[::-1] # parametrically varying random errors
rand_dep_wfe_r = np.empty((n_tolerances,n_sample,len(wl_sample_r)))
rand_dep_wfe_t = np.empty((n_tolerances,n_sample,len(wl_sample_t)))

# Read data from file
for i,rand_dep_error in enumerate(rand_dep_error_list_short):
    
    filename_wfe_r = 'WFE_Case1_error'+str(rand_dep_error)+'R.csv'
    filename_wfe_t = 'WFE_Case1_error'+str(rand_dep_error)+'T.csv'

    rand_dep_wfe_data_r_df = pd.read_csv(f"{OutputFilePath}/WFE/{filename_wfe_r}",delimiter=',')
    rand_dep_wfe_data_t_df = pd.read_csv(f"{OutputFilePath}/WFE/{filename_wfe_t}",delimiter=',')

    rand_dep_wfe_r[i,:,:] = rand_dep_wfe_data_r_df.values
    rand_dep_wfe_t[i,:,:] = rand_dep_wfe_data_t_df.values


#%% Write data for rainbow plots

# Write Rainbow plot in R

# xlim_lo, xlim_hi = np.min(wl), np.max(wl) # whole plot    
# xlim_lo, xlim_hi = 1080, 1120 # focus on flat part of spectrum
xlim_lo, xlim_hi = np.min(wl_sample_r), np.max(wl_sample_r) # focus on flat part of spectrum
# xlim_lo, xlim_hi = 1025, 1125 # focus on flat part of spectrum
xlimits_cut = np.where((wl_sample_r>=xlim_lo) & (wl_sample_r<=xlim_hi))[0]
wl_cut_wfe1 = wl_sample_r[xlimits_cut]
wfe_value_cut = rand_dep_wfe_r[:,:,xlimits_cut]
bin_size = 0.3
column_names = ['x_nonzero','y_nonzero','density_values']

for error_ind,rand_dep_error in enumerate(rand_dep_error_list_short): # looping over different errors in thickness
    
    residual = wfe_value_cut[error_ind,:,:] # using wfe_value_cut for given wl and t_error
    #  Here, the bin_size is defined, which determines the width of each bin in the histogram. 
    #  n_bins is calculated using np.arange to create an array of bin edges based on the minimum and maximum values of y and the specified bin_size.
    #n_bins = np.arange(np.nanmin(residual), np.nanmax(residual) + bin_size, bin_size)
    n_bins = np.arange(np.nanmin(residual), np.nanmax(residual) + bin_size, bin_size)
        
    # Iterate over wl values
    for i in np.arange(0,len(wl_cut_wfe1)): # looping over different wavelength points
        
        #  y_sample is extracted for the current x value, 
        #  and any non-finite values are removed using np.isfinite(y_sample).
        residual_sample = residual[:,i] 
        residual_sample = residual_sample[np.isfinite(residual_sample)] # chosing values that are not nan or inf
        
        #  The histogram hist and bin edges bin_edges are computed for the filtered y_sample using np.histogram.
        hist, bin_edges = np.histogram(residual_sample, bins=n_bins)
        #  The density is calculated by dividing the histogram by the total number of data points in y 
        #  and multiplying by 100 to obtain the density as a percentage.
        density = hist/len(residual) * 100# Calculate the density as a percentage of the total points
        #  If you want the same colorbar to apply for all x values, you should use density = hist/len(y) * 100.
        #  By using density = hist/len(y) * 100, you are calculating the density values based on the entire dataset y, 
        #     regardless of the specific x value. 
        #  This ensures that the density values are normalized consistently across all x values, 
        #     allowing for a consistent color mapping and colorbar across the entire plot.
        
        #  density_values is converted to a list.
        density_value = np.array(density.tolist())  # Density values for current wl value
        nonzero_mask = np.where(density_value > 0)[0]  # Mask for non-zero frequencies
        x_nonzero = np.full((len(bin_edges[:-1][nonzero_mask])), wl_cut_wfe1[i])
        y_nonzero= np.array(bin_edges[:-1][nonzero_mask])
        density_values = density_value[nonzero_mask]

        write_filename = f"{OutputFilePath}/WFE/WFE_Case1_rainbow_error_{rand_dep_error}wl_{wl_cut_wfe1[i]}R.csv"    
        wfe_data_case1_rainbow = np.vstack((x_nonzero,y_nonzero,density_values)).T
        df_wfe_case1_rainbow = pd.DataFrame(wfe_data_case1_rainbow, columns = column_names)
        df_wfe_case1_rainbow.to_csv(write_filename,index=False)


# Write Rainbow plot in T

xlim_lo, xlim_hi = np.min(wl_sample_t), np.max(wl_sample_t) # focus on flat part of spectrum
xlimits_cut = np.where((wl_sample_t>=xlim_lo) & (wl_sample_t<=xlim_hi))[0]
wl_cut_wfe1 = wl_sample_t[xlimits_cut]
wfe_value_cut = rand_dep_wfe_t[:,:,xlimits_cut]
bin_size = 0.3
column_names = ['x_nonzero','y_nonzero','density_values']

for error_ind,rand_dep_error in enumerate(rand_dep_error_list_short): # looping over different errors in thickness
    
    residual = wfe_value_cut[error_ind,:,:] # using wfe_value_cut for given wl and t_error
    #  Here, the bin_size is defined, which determines the width of each bin in the histogram. 
    #  n_bins is calculated using np.arange to create an array of bin edges based on the minimum and maximum values of y and the specified bin_size.
    #n_bins = np.arange(np.nanmin(residual), np.nanmax(residual) + bin_size, bin_size)
    n_bins = np.arange(np.nanmin(residual), np.nanmax(residual) + bin_size, bin_size)
        
    # Iterate over wl values
    for i in np.arange(0,len(wl_cut_wfe1)): # looping over different wavelength points
        
        #  y_sample is extracted for the current x value, 
        #  and any non-finite values are removed using np.isfinite(y_sample).
        residual_sample = residual[:,i] 
        residual_sample = residual_sample[np.isfinite(residual_sample)] # chosing values that are not nan or inf
        
        #  The histogram hist and bin edges bin_edges are computed for the filtered y_sample using np.histogram.
        hist, bin_edges = np.histogram(residual_sample, bins=n_bins)
        #  The density is calculated by dividing the histogram by the total number of data points in y 
        #  and multiplying by 100 to obtain the density as a percentage.
        density = hist/len(residual) * 100# Calculate the density as a percentage of the total points
        #  If you want the same colorbar to apply for all x values, you should use density = hist/len(y) * 100.
        #  By using density = hist/len(y) * 100, you are calculating the density values based on the entire dataset y, 
        #     regardless of the specific x value. 
        #  This ensures that the density values are normalized consistently across all x values, 
        #     allowing for a consistent color mapping and colorbar across the entire plot.
        
        #  density_values is converted to a list.
        density_value = np.array(density.tolist())  # Density values for current wl value
        nonzero_mask = np.where(density_value > 0)[0]  # Mask for non-zero frequencies
        x_nonzero = np.full((len(bin_edges[:-1][nonzero_mask])), wl_cut_wfe1[i])
        y_nonzero= np.array(bin_edges[:-1][nonzero_mask])
        density_values = density_value[nonzero_mask]

        write_filename = f"{OutputFilePath}/WFE/WFE_Case1_rainbow_error_{rand_dep_error}wl_{wl_cut_wfe1[i]}T.csv"    
        wfe_data_case1_rainbow = np.vstack((x_nonzero,y_nonzero,density_values)).T
        df_wfe_case1_rainbow = pd.DataFrame(wfe_data_case1_rainbow, columns = column_names)
        df_wfe_case1_rainbow.to_csv(write_filename,index=False)

#%% Masking extreme values
rand_dep_wfe_r[abs(rand_dep_wfe_r)>=500] = np.nan #masking values greater than 200 to avoid extra distractions caused by spiky spectrum
rand_dep_wfe_t[abs(rand_dep_wfe_t)>=500] = np.nan #masking values greater than 200 to avoid extra distractions caused by spiky spectrum

#%% Read files in R and T & create rainbow plots 

bin_size = 0.3

# Reflection
xlim_lo, xlim_hi = np.min(wl_sample_r), np.max(wl_sample_r) # focus on flat part of spectrum
xlimits_cut_r = np.where((wl_sample_r>=xlim_lo) & (wl_sample_r<=xlim_hi))[0]
wl_cut_wfe1_r = wl_sample_r[xlimits_cut_r]
wfe_value_cut_r = rand_dep_wfe_r[:,:,xlimits_cut_r]

# Transmission
xlim_lo, xlim_hi = np.min(wl_sample_t), np.max(wl_sample_t) # focus on flat part of spectrum
xlimits_cut_t = np.where((wl_sample_t>=xlim_lo) & (wl_sample_t<=xlim_hi))[0]
wl_cut_wfe1_t = wl_sample_t[xlimits_cut_t]
wfe_value_cut_t = rand_dep_wfe_t[:,:,xlimits_cut_t]

# Create a figure and axes
fig,axes = plt.subplots(2,3, figsize=(20,20))
cm_used = plt.cm.get_cmap('nipy_spectral')


# for error_ind,ax_no in zip(range(len(rand_dep_error_list_short)),axes[::-1,::-1].flatten()): # looping over different errors in thickness
for error_ind,ax_no in zip(range(len(rand_dep_error_list_short)),range(np.shape(axes)[1])): # looping over different errors in thickness

    # Reflection

    # Iterate over wl values
    for i in np.arange(0,len(wl_cut_wfe1_r)): # looping over different wavelength points
        
        write_filename = f"{OutputFilePath}/WFE/WFE_Case1_rainbow_error_{rand_dep_error_list_short[error_ind]}wl_{wl_cut_wfe1_r[i]}R.csv"
        df_wfe_case1_read_rainbow = pd.read_csv(write_filename,delimiter=',')
        
        x_nonzero = df_wfe_case1_read_rainbow['x_nonzero'].values
        y_nonzero = df_wfe_case1_read_rainbow['y_nonzero'].values
        density_values = df_wfe_case1_read_rainbow['density_values'].values
        
        #  A color normalization object norm is created using mcolors.Normalize with the minimum and maximum density values.
        norm = mcolors.Normalize(vmin=min(density_values), vmax=max(density_values))  # Normalize the density values
        #  that maps the density values to the range [0, 1]. 
        #  The vmin and vmax arguments specify the minimum and maximum values to which the density values will be mapped
        
        #  The scatter plot is created using ax.scatter with the x values set as a full array of the current x value, 
        #  the bin edges as the y values, the density values as the color values, 
        #  the colormap 'inferno', the normalization norm, and a square marker 's'.
        sc = axes[0,2-ax_no].scatter(x_nonzero, y_nonzero, c=density_values, cmap=cm_used, norm=norm, marker='s')

    # Add a colorbar with percentage values
    # These lines create a colorbar for the scatter plot and format it to display the density values as a percentage. 
    # The label 'Density' is assigned to the colorbar.
    cb = plt.colorbar(sc, format='%.2f', ax=ax)
    cb.set_label('Density [%]',rotation=90)
    
    axes[0,2-ax_no].set_ylabel(r'WFE$_R$ [nm]') #y-label
    axes[0,2-ax_no].set_xlabel('$\lambda$ [nm]') #x-label
    axes[0,2-ax_no].set_xlim(min(wl_sample_r),max(wl_sample_r)) #set x axis range
    axes[0,2-ax_no].axhline(y=0, color='white', linestyle='-',lw=3) # white line to mark y=0 axis
    axes[0,2-ax_no].axhline(y=0, color='black', linestyle='-.',lw=1.5) # thin black dashed line to help higlight y=0 axis
    axes[0,2-ax_no].set_title('$\Delta d$ = '+str(100*rand_dep_error_list_short[error_ind])+'%',fontsize=12)
    axes[0,2-ax_no].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the first subplot

    # Add a colorbar with percentage values
    # These lines create a colorbar for the scatter plot and format it to display the density values as a percentage. 
    # The label 'Density' is assigned to the colorbar.
    cb = plt.colorbar(sc, format='%.2f', ax=axes[0,2-ax_no])
    cb.set_label('Density [%]',rotation=90)
       
    # Transmission

    # Iterate over wl values
    for i in np.arange(0,len(wl_cut_wfe1_t)): # looping over different wavelength points
        write_filename = f"{OutputFilePath}/WFE/WFE_Case1_rainbow_error_{rand_dep_error_list_short[error_ind]}wl_{wl_cut_wfe1_t[i]}T.csv"    
        df_wfe_case1_read_rainbow = pd.read_csv(write_filename,delimiter=',')
        
        x_nonzero = df_wfe_case1_read_rainbow['x_nonzero'].values
        y_nonzero = df_wfe_case1_read_rainbow['y_nonzero'].values
        density_values = df_wfe_case1_read_rainbow['density_values'].values
        
        #  A color normalization object norm is created using mcolors.Normalize with the minimum and maximum density values.
        norm = mcolors.Normalize(vmin=min(density_values), vmax=max(density_values))  # Normalize the density values
        #  that maps the density values to the range [0, 1]. 
        #  The vmin and vmax arguments specify the minimum and maximum values to which the density values will be mapped
        
        #  The scatter plot is created using ax.scatter with the x values set as a full array of the current x value, 
        #  the bin edges as the y values, the density values as the color values, 
        #  the colormap 'inferno', the normalization norm, and a square marker 's'.
        sc = axes[1,2-ax_no].scatter(x_nonzero, y_nonzero, c=density_values, cmap=cm_used, norm=norm, marker='s')
    
    # Add a colorbar with percentage values
    # These lines create a colorbar for the scatter plot and format it to display the density values as a percentage. 
    # The label 'Density' is assigned to the colorbar.
    cb = plt.colorbar(sc, format='%.2f', ax=axes[1,2-ax_no])
    cb.set_label('Density[%]',rotation=90)
    
    axes[1,2-ax_no].set_ylabel(r'WFE$_T$ [nm]') #y-label
    axes[1,2-ax_no].set_xlabel('$\lambda$ [nm]') #x-label
    axes[1,2-ax_no].set_xlim(min(wl_sample_t),max(wl_sample_t)) #set x axis range
    axes[1,2-ax_no].axhline(y=0, color='white', linestyle='-',lw=3) # white line to mark y=0 axis
    axes[1,2-ax_no].axhline(y=0, color='black', linestyle='-.',lw=1.5) # thin black dashed line to help higlight y=0 axis
    axes[1,2-ax_no].set_title('$\Delta d$ = '+str(100*rand_dep_error_list_short[error_ind])+'%',fontsize=12)
    axes[1,2-ax_no].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the first subplot
fig.set_tight_layout(True) # set plot layout to tight
plt.savefig(f"{OutputFilePath}/Plots/Case1WFE_D1_front_t_error_{rand_dep_error_list_short[error_ind]*100}%col.png",dpi=300) # save figure


#%% WFE Analysis Case 2

# WFE from nominal case
sys_dep_wfe_r = np.empty((n_tolerances,len(wl)))
sys_dep_wfe_t = np.empty((n_tolerances,len(wl)))

wl_sample_r = wl[np.where((wl>=1550)&(wl<=1750))]
wl_sample_t = wl[np.where((wl>=5900)&(wl<=6100))]

ind_wl_r = np.where(np.isin(wl, wl_sample_r))[0] # finding indices of wavelength values chosen
ind_wl_t = np.where(np.isin(wl, wl_sample_t))[0] # finding indices of wavelength values chosen
n_point_front_r = n_front[ind_wl_r,:] #!!!! change this to adapt if it's the front or bbar coating 
n_point_front_t = n_front[ind_wl_t,:] #!!!! change this to adapt if it's the front or bbar coating 
n_point_substrate_r = n_substrate[ind_wl_r]
n_point_substrate_t = n_substrate[ind_wl_t]
n_point_bbar_r = n_bbar[ind_wl_r,:] #!!!! change this to adapt if it's the front or bbar coating 
n_point_bbar_t = n_bbar[ind_wl_t,:] #!!!! change this to adapt if it's the front or bbar coating 

# Resulting phase and spectral performance at wl_point
_, _, nominal_r_phase_rsample, _ = phase_analysis(wl_sample_r,n_point_front_r,n_point_substrate_r,n_point_bbar_r,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
_, _, _, nominal_t_phase_tsample = phase_analysis(wl_sample_t,n_point_front_t,n_point_substrate_t,n_point_bbar_t,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)

# Reflection
xlim_lo, xlim_hi = min(wl_sample_r), max(wl_sample_r) # focus on flat part of spectrum
xlimits_cut_r = np.where((wl_sample_r>=xlim_lo) & (wl_sample_r<=xlim_hi))[0]
wl_cut_wfe1_r = wl_sample_r[xlimits_cut_r]

# Transmission
xlim_lo, xlim_hi = min(wl_sample_t), max(wl_sample_t) # focus on flat part of spectrum
xlimits_cut_t = np.where((wl_sample_t>=xlim_lo) & (wl_sample_t<=xlim_hi))[0]
wl_cut_wfe1_t = wl_sample_t[xlimits_cut_t]

rand_dep_error_list_short = np.array([0.001,0.005,0.01])[::-1] # parametrically varying random errors
for i,rand_dep_error in enumerate(rand_dep_error_list_short):
        
    # Depo Case 2 
    _,_,sys_dep_r_phase,_,_,_ = depo_case2(wl,rand_dep_error,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)
    _,_,_,sys_dep_t_phase,_,_ = depo_case2(wl,rand_dep_error,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar)

    # Difference in phase from nominal case, for R and T at different wl points
    sys_dep_r_phase_delta = np.real(nominal_r_phase-sys_dep_r_phase)        
    sys_dep_t_phase_delta = np.real(nominal_t_phase-sys_dep_t_phase)        
        
    # WFE conversion
    sys_dep_wfe_r[i,:] = phase_to_wfe(wl,sys_dep_r_phase_delta)
    sys_dep_wfe_t[i,:] = phase_to_wfe(wl,sys_dep_t_phase_delta)


#%% Plot Case 2 of wavefront error –  all error values

fig_wfe2,ax_wfe2 =plt.subplots(2,len(rand_dep_error_list_short),figsize=(12,5))

for i,rand_dep_error in enumerate(rand_dep_error_list_short): # looping over different errors in thickness
    pl_hold = len(rand_dep_error_list_short)-(i+1) # plotting subplots in reverse
    ax_wfe2[0,pl_hold].plot(wl,sys_dep_wfe_r[i,:],'-',color='purple')
    ax_wfe2[0,pl_hold].set_title('$\Delta d$ = '+str(100*rand_dep_error)+'%',fontsize=12)
    xlim_lo, xlim_hi = np.min(wl_sample_r), np.max(wl_sample_r) # whole plot
    # xlim_lo, xlim_hi = 1044, 1244 # focus on flat part of spectrum #larger part
    # xlim_lo, xlim_hi = 1065, 1165 # focus on flat part of spectrum
    
    xlimits_cut_r = np.where((wl>=xlim_lo)&(wl<=xlim_hi))
    ylim_lo = np.nanmin(sys_dep_wfe_r[i,:][xlimits_cut_r]) 
    ylim_hi = np.nanmax(sys_dep_wfe_r[i,:][xlimits_cut_r])
    ax_wfe2[0,pl_hold].set_xlim(xlim_lo,xlim_hi)
    ax_wfe2[0,pl_hold].set_ylim(ylim_lo,ylim_hi)
    ax_wfe2[0,pl_hold].grid(visible=True) 

    ax_wfe2[1,pl_hold].plot(wl,sys_dep_wfe_t[i,:],'-',color='purple')
    ax_wfe2[1,pl_hold].set_title('$\Delta d$ = '+str(100*rand_dep_error)+'%',fontsize=12)
    xlim_lo, xlim_hi = np.min(wl_sample_t), np.max(wl_sample_t) # focus on flat part of spectrum
    # xlim_lo, xlim_hi = 5000, 5200 # focus on flat part of spectrum
    xlimits_cut_t = np.where((wl>=xlim_lo)&(wl<=xlim_hi))

    ylim_lo = np.nanmin(sys_dep_wfe_t[i,:][xlimits_cut_t]) 
    ylim_hi = np.nanmax(sys_dep_wfe_t[i,:][xlimits_cut_t])
    ax_wfe2[1,pl_hold].set_xlim(xlim_lo,xlim_hi)
    ax_wfe2[1,pl_hold].set_ylim(ylim_lo,ylim_hi)
    ax_wfe2[1,pl_hold].grid(visible=True) 
    ax_wfe2[0,pl_hold].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the first subplot
    ax_wfe2[1,pl_hold].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the first subplot


fig_wfe2.supylabel('WFE [nm]',fontsize=12)
fig_wfe2.supxlabel('$\lambda$ [nm]',fontsize=12)
fig_wfe2.set_tight_layout(True)
plt.savefig(f"{OutputFilePath}/Plots/Case2WFE_D1_front.pdf",dpi=600) # save figure


#%% Plot Case 2 of wavefront error –  plot 0.5% error with median bins

# Masking extreme values
sys_dep_wfe_r[abs(sys_dep_wfe_r)>=100] = np.nan #masking values greater than 200 to avoid extra distractions caused by spiky spectrum
sys_dep_wfe_t[abs(sys_dep_wfe_t)>=100] = np.nan #masking values greater than 200 to avoid extra distractions caused by spiky spectrum


fig_wfe2,ax_wfe2 = plt.subplots(2,figsize=(10,6))


wfe_value2_halfperc = sys_dep_wfe_r[2,:] #selecting array of case 2 for 0.5% error

ax_wfe2[0].plot(wl,wfe_value2_halfperc,'-',color='lightblue',linewidth=2,alpha=1,label='Unsmoothed')
xlim_lo, xlim_hi = np.min(wl), np.max(wl) # whole plot
# xlim_lo, xlim_hi = 1320, 1680 # focus on flat part of spectrum
# xlim_lo, xlim_hi = 1210, 1230 # focus on flat part of spectrum
ylim_lo = np.nanmin(wfe_value2_halfperc[np.where((wl>=xlim_lo) & (wl<=xlim_hi))]) 
ylim_hi = np.nanmax(wfe_value2_halfperc[np.where((wl>=xlim_lo) & (wl<=xlim_hi))])
ax_wfe2[0].set_xlim(xlim_lo,xlim_hi)
ax_wfe2[0].set_ylim(ylim_lo,ylim_hi)
ax_wfe2[0].grid(visible=False) 

bin_width = 200 #10nm bin widths
s,edges,binnumber = binned_statistic(wl,wfe_value2_halfperc, statistic=np.nanmedian, bins=len(wfe_value2_halfperc)/bin_width)

# Define the rectangle parameters (position and size)
rect_x = min(wl_sample_r)  # x-coordinate of the bottom-left corner of the rectangle
rect_y = -10    # y-coordinate of the bottom-left corner of the rectangle
rect_width = 200  # Width of the rectangle
rect_height = 30  # Height of the rectangle
# Create the Rectangle patch
rectangle = Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, color='red', linewidth=2)
# Add the rectangle to the plot
ax_wfe2[0].add_patch(rectangle)


ys = np.repeat(s,2)
xs = np.repeat(edges,2)[1:-1]
ax_wfe2[0].hlines(s,edges[:-1],edges[1:], color="maroon",label='Smoothed')
for e in edges:
    ax_wfe2[0].axvline(e, color="grey", linestyle="--",linewidth=0.5,alpha=1)


wfe_value2_halfperc = sys_dep_wfe_t[2,:] #selecting array of case 2 for 0.5% error

ax_wfe2[1].plot(wl,wfe_value2_halfperc,'-',color='lightblue',linewidth=2,alpha=1,label='Unsmoothed')
xlim_lo, xlim_hi = np.min(wl), np.max(wl) # whole plot
# xlim_lo, xlim_hi = 1320, 1680 # focus on flat part of spectrum
# xlim_lo, xlim_hi = 1210, 1230 # focus on flat part of spectrum
ylim_lo = np.nanmin(wfe_value2_halfperc[np.where((wl>=xlim_lo) & (wl<=xlim_hi))]) 
ylim_hi = np.nanmax(wfe_value2_halfperc[np.where((wl>=xlim_lo) & (wl<=xlim_hi))])
ax_wfe2[1].set_xlim(xlim_lo,xlim_hi)
ax_wfe2[1].set_ylim(ylim_lo,ylim_hi)
ax_wfe2[1].grid(visible=False) 

bin_width = 200 #10nm bin widths
s,edges,binnumber = binned_statistic(wl,wfe_value2_halfperc, statistic=np.nanmedian, bins=len(wfe_value2_halfperc)/bin_width)

# Define the rectangle parameters (position and size)
rect_x = min(wl_sample_t)  # x-coordinate of the bottom-left corner of the rectangle
rect_y = 6    # y-coordinate of the bottom-left corner of the rectangle
rect_width = 200  # Width of the rectangle
rect_height = 20  # Height of the rectangle
# Create the Rectangle patch
rectangle = Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, color='red', linewidth=2)
# Add the rectangle to the plot
ax_wfe2[1].add_patch(rectangle)


ax_wfe2[0].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the first subplot
ax_wfe2[1].tick_params(axis='both', which='major', labelsize=12)  # Set the font size for the first subplot

ys = np.repeat(s,2)
xs = np.repeat(edges,2)[1:-1]
ax_wfe2[1].hlines(s,edges[:-1],edges[1:], color="maroon",label='Smoothed')
for e in edges:
    ax_wfe2[1].axvline(e, color="grey", linestyle="--",linewidth=0.5,alpha=1)


# ax_wfe2.legend()
fig_wfe2.supylabel('WFE [nm]',fontsize=12)
fig_wfe2.supxlabel('$\lambda$ [nm]',fontsize=12)
fig_wfe2.set_tight_layout(True)
plt.savefig(f"{OutputFilePath}/Plots/Case2WFE_D1_front_0.5case.pdf",dpi=600) # save figure



#%%#####################################################################################################################################################
####################################### Part 3: Scattering models for high spatial frequency variations (Case 1) #######################################
########################################################################################################################################################
#%% Scattering parameters

rms_subs = 1 # rms roughness of substrate in nm
# RMS of YbF3 and ZnSe from papers
rms_layer_front = np.zeros_like(phys_l_front)
rms_layer_bbar = np.zeros_like(phys_l_bbar)

# RMS of Multilayer structure
for eta_type_ind,eta_type in enumerate(param['multilayer_layout']['refr_index_layout_front_coating_only']['sequence']):
    if eta_type == 'H':
        rms_layer_front[eta_type_ind] = phys_l_front[eta_type_ind] * 0.005
    elif eta_type == 'L':
        rms_layer_front[eta_type_ind] = phys_l_front[eta_type_ind] * 0.007
for eta_type_ind,eta_type in enumerate(param['multilayer_layout']['refr_index_layout_rear_coating_only']['sequence']):
    if eta_type == 'H':
        rms_layer_bbar[eta_type_ind] = phys_l_bbar[eta_type_ind] * 0.005
    elif eta_type == 'L':
        rms_layer_bbar[eta_type_ind] = phys_l_bbar[eta_type_ind] * 0.007
sigma_front = np.append(rms_layer_front,rms_subs) # array of rms of each layer of front coating
sigma_bbar = np.append(rms_subs,rms_layer_bbar) # array of rms of each layer of bbar coating


#%% Generating a set of random thicknesses with errors to input the same MC runs into the models

start = time.time()
rand_dep_error_single = syst_param['manufacturing']['random_dep_error']['value'] # error in percentage # deposition uncertainty of sputtering machine is 0.5%
# find index of case 1 where error=rand_dep_error_single
mc_index = np.where(rand_dep_error_list == rand_dep_error_single)[0][0]
# selecting the random thickness errors applied to the right thickness
phys_l_front_rand_single = (phys_l_front_rand[mc_index,:,:]) # to ensure we use the same recipes as for the phase calculations, making sure array is in required shape
phys_l_rear_rand_single = (phys_l_rear_rand[mc_index,:,:]) # to ensure we use the same recipes as for the phase calculations, making sure array is in required shape

# Running both scattering models for different rms models and save to files
# Adjusting size but this should be changed later when publishing code
theta_scatter = np.full(n_sample,theta[0]) # assume a fully collimated beam

for scatter_model in ['optimistic','worst']:
    for surface_type in ['correlated','uncorrelated','additive']:
        start_time = time.time()
        print(scatter_model,' and ',surface_type)
        
        # Calculating spectral performance after scatter loss
        scatter_r_specular, scatter_t_specular, _, _ = scatter_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta_scatter,polar,sigma_front,sigma_bbar,surface_type,scatter_model,component_types)
        scatter_r_specular = np.sum(scatter_r_specular,axis=0)/n_sample
        scatter_t_specular = np.sum(scatter_t_specular,axis=0)/n_sample
        scatter_r_diffuse = nominal_r - scatter_r_specular
        scatter_t_diffuse = nominal_t - scatter_t_specular
        
        # Saving data as CSV file
        scatter_data = np.array([scatter_r_specular,scatter_t_specular,scatter_r_diffuse,scatter_t_diffuse]).T
        df = pd.DataFrame(data=scatter_data)
        df.columns = ['R_specular','T_specular','R_diffuse','T_diffuse']
        filename = "D1_Ariel_scatter_n_sample"+str(n_sample)+"_"+str(scatter_model)+"case_"+str(surface_type)+"interface"+".csv"
        df.to_csv(f"{OutputFilePath}/Scattering/{filename}",sep=',',index=False,columns=['R_specular','T_specular','R_diffuse','T_diffuse'])
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
end = time.time()

print(f"time={end-start}")

#%% Read data from files

scatter_data = {}
for scatter_model in ['optimistic','worst']:
    surface_types = {}
    for surface_type in ['correlated','uncorrelated','additive']:
        gen_spectr_random = {}
        filename = "D1_Ariel_scatter_n_sample"+str(n_sample)+"_"+str(scatter_model)+"case_"+str(surface_type)+"interface"+".csv"
        df = pd.read_csv(f"{OutputFilePath}/Scattering/{filename}",delimiter=',')

        # Iterate over the columns in the DataFrame
        for column in df.columns:
           # Extract the column values as a NumPy array
           values = df[column].values
           # Store the column values in the dictionary
           gen_spectr_random[column] = values
        
        surface_types[str(surface_type)] = gen_spectr_random
    scatter_data[str(scatter_model)] = surface_types
    

#%% Plot results of scattering models for different rms models

fig,ax = plt.subplots(2,2, figsize=(12,6))

for scatter_model_type_ind,scatter_model_type in enumerate(['optimistic','worst']):

    for surface_type in ['correlated','additive','uncorrelated']:
        
        plot_array = scatter_data[str(scatter_model_type)][str(surface_type)]
        ax[scatter_model_type_ind,0].plot(wl,plot_array['R_specular'],'--',lw=1.5,label='Surface Correlation: '+str(surface_type))
        ax[scatter_model_type_ind,1].plot(wl,plot_array['T_specular'],'--',lw=1.5,label='Surface Correlation: '+str(surface_type))
    
    ax[scatter_model_type_ind,0].set_ylabel('Reflectance [%]')
    ax[scatter_model_type_ind,1].set_ylabel('Transmittance [%]')

    for axes in [ax[0,0],ax[1,0],ax[0,1],ax[1,1]]:
        axes.set_xlabel('$\lambda$ [nm]')
        axes.set_xlim((min(wl),max(wl)))
        axes.grid(visible=True)
        axes.set_ylim(0,100)
    for axes in [ax[0,1],ax[0,0]]:
        axes.legend(loc='center')

    ax[scatter_model_type_ind,0].plot(wl,nominal_r,lw=1,color='k',label='Nominal',linestyle=':')
    ax[scatter_model_type_ind,1].plot(wl,nominal_t,lw=1,color='k',label='Nominal',linestyle=':')

    
for n, axes in enumerate([ax[0,0],ax[0,1],ax[1,0],ax[1,1]]):
    axes.text(0.93, 0.9, '('+string.ascii_lowercase[n]+')', transform=axes.transAxes, size=12, weight='bold')

fig.set_tight_layout(True)
plt.tight_layout()
fig.savefig(f"{OutputFilePath}/Scattering_{n_sample}_both_cases.pdf",dpi=600)



# for scatter_model_type in ['worst','optimistic']:
#     fig,ax = plt.subplots(2,2, figsize=(12,6))
#     ax[0,0].plot(wl,nominal_r,lw=1,color='k',label='No roughness')
#     ax[0,1].plot(wl,nominal_t,lw=1,color='k',label='No roughness')
#     for surface_type in ['correlated','uncorrelated','additive']:
        
#         plot_array = scatter_data[str(scatter_model_type)][str(surface_type)]
#         ax[0,0].plot(wl,plot_array['R_specular'],'--',lw=1.5,label='Surface Correlation: '+str(surface_type))
#         ax[0,1].plot(wl,plot_array['T_specular'],'--',lw=1.5,label='Surface Correlation: '+str(surface_type))
#         ax[1,0].plot(wl,plot_array['R_diffuse'],'--',lw=1.5,label='Surface Correlation: '+str(surface_type))
#         ax[1,1].plot(wl,plot_array['T_diffuse'],'--',lw=1.5,label='Surface Correlation: '+str(surface_type))
    
#     ax[0,0].set_ylabel('Specular Reflectance [%]')
#     ax[0,1].set_ylabel('Specular Transmittance [%]')
#     ax[1,0].set_ylabel('Scatter Losses for Reflectance [%]')
#     ax[1,1].set_ylabel('Scatter Losses for Transmittance [%]')

#     for axes in [ax[0,0],ax[1,0],ax[0,1],ax[1,1]]:
#         axes.set_xlabel('$\lambda$ [nm]')
#         axes.set_xlim((min(wl),max(wl)))
#         axes.grid(visible=True)
#         # axes.legend(loc='best')
#         # axes.set_xlim(0,100)
#         axes.set_ylim(0,100)
#     # fig.suptitle('Model: ' +str(scatter_model_type) +' case')
#     for axes in [ax[0,1],ax[0,0]]:
#         axes.legend(loc='best')
    
#     fig.set_tight_layout(True)
#     plt.tight_layout()


# fig,ax = plt.subplots(2,2, figsize=(12,6))

# for scatter_model_type_ind,scatter_model_type in enumerate(['optimistic','worst']):
#     for surface_type in ['correlated','uncorrelated','additive']:

#         plot_array = scatter_data[str(scatter_model_type)][str(surface_type)]
#         ax[scatter_model_type_ind,0].plot(wl,plot_array['R_diffuse'],'--',lw=1.5,label='Surface Correlation: '+str(surface_type))
#         ax[scatter_model_type_ind,1].plot(wl,plot_array['T_diffuse'],'--',lw=1.5,label='Surface Correlation: '+str(surface_type))
    
#     ax[scatter_model_type_ind,0].set_ylabel('Scatter Losses for Reflectance [%]')
#     ax[scatter_model_type_ind,1].set_ylabel('Scatter Losses for Transmittance [%]')

#     for axes in [ax[0,0],ax[1,0],ax[0,1],ax[1,1]]:
#         axes.set_xlabel('$\lambda$ [nm]')
#         axes.set_xlim((min(wl),max(wl)))
#         axes.grid(visible=True)
#         # axes.legend(loc='best')
#         # axes.set_xlim(0,100)
#         # axes.set_ylim(0,100)
#     # fig.suptitle('Model: ' +str(scatter_model_type) +' case')
#     for axes in [ax[0,1],ax[0,0]]:
#         axes.legend(loc='best')
        
# for n, axes in enumerate([ax[0,0],ax[0,1],ax[1,0],ax[1,1]]):
#     axes.text(0.93, 0.9, '('+string.ascii_lowercase[n]+')', transform=axes.transAxes, size=12, weight='bold')

# fig.set_tight_layout(True)
# plt.tight_layout()
# fig.savefig(f"{OutputFilePath}/Scattering_{n_sample}_both_cases.pdf",dpi=600)




#%% Plotting surface roughness models

sigma_list = [sigma_bbar,sigma_front]
component_type_list =['bbar','front']

fig,ax = plt.subplots(figsize=(12,5))
for component_ind,component_type in enumerate(component_type_list):
    sigma = sigma_list[component_ind]
    M = np.shape(sigma)[0]-1
    
    for surface_type in ['correlated','uncorrelated','additive']:
        rms_measurable = np.zeros((M+1),dtype = 'complex') #M+1 since there are M+1 interfaces                  # Distribution of coating layers' surface roughnesses
    
        if surface_type == 'uncorrelated':
            if component_type == 'bbar':            
                rms_measurable = sigma # or just layer rms?
                x_plot = np.linspace(0,np.size(sigma)-1,np.size(sigma)).astype('int')
                ax.plot(-1*x_plot,rms_measurable,'--o', markersize=5, color='blue')
            
            else:              
                rms_measurable = sigma # or just layer rms?
                x_plot = np.linspace(0,np.size(sigma)-1,np.size(sigma)).astype('int')
                ax.plot(x_plot,rms_measurable[::-1],'--o', markersize=5, label=str(surface_type),color='blue')
            
        elif surface_type == 'correlated':
            if component_type == 'bbar':
                print('bbar')
                for n_layer in range(M+1):
                    rms_measurable[n_layer] = np.sum(sigma[:n_layer+1])# or just layer rms?
                x_plot = np.linspace(0,np.size(sigma)-1,np.size(sigma)).astype('int')
                ax.plot(-1*x_plot,rms_measurable,'--s', markersize=5, color='green')

            else:
                print('k')
                for n_layer in range(M+1):
                    rms_measurable[n_layer] = np.sum(sigma[n_layer:])# or just layer rms?
                x_plot = np.linspace(0,np.size(sigma)-1,np.size(sigma)).astype('int')
                ax.plot(x_plot,rms_measurable[::-1],'--s', markersize=5, label=str(surface_type),color='green')
                        
        elif surface_type == 'additive':
            print('k')
            if component_type == 'bbar':
                for n_layer in range(M+1):    
                    rms_measurable[n_layer] = np.sqrt(np.sum(sigma[:n_layer+1]**2))# or just layer rms?
                x_plot = np.linspace(0,np.size(sigma)-1,np.size(sigma)).astype('int')
                ax.plot(-1*x_plot,rms_measurable,'--v', label=str(surface_type),color='red')

            else:
                for n_layer in range(M+1):
                    rms_measurable[n_layer] = np.sqrt(np.sum(sigma[n_layer:]**2))# or just layer rms?   
                x_plot = np.linspace(0,np.size(sigma)-1,np.size(sigma)).astype('int')
                ax.plot(x_plot,rms_measurable[::-1],'--v', color='red')      
        
# Set the reversed tick locations and labels
annotation_line( ax=ax, text='Rear Coating', xmin=-11, xmax=0, y=50, ytext=52, linewidth=1, linecolor='black', fontsize=11 )
annotation_line( ax=ax, text='Front Coating', xmin=0, xmax=55, y=50, ytext=52, linewidth=1, linecolor='black', fontsize=11 )
ax.vlines(0,ymin=0,ymax=52.5,colors='k',linestyle='--')
ax.text(-1.2,17,'Location of Substrate',rotation=90, fontsize=11)
ax.grid()
ax.set_xlabel('Interface number, i', fontsize=12)
ax.set_ylabel(r'$\sigma_{eff,i}$ [nm]', fontsize=12)
ax.set_xlim(-11.5,55.5)
ax.legend(loc='center')
fig.set_tight_layout(True)
plt.savefig(f"{OutputFilePath}/Plots/rms_models.pdf",dpi=600)


