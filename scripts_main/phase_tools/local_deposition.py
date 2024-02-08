#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to model the varied thicknesses of a coating for different deposition models, 
for localised points across the dichroic surface
"""

import numpy as np
import matplotlib.pyplot as plt
from tlm_model.analysis import spectral_analysis


def disk_points(thick,nt,t_err,rd,dd):
    

# thick = np.concatenate((phys_l_front,phys_l_bbar)) # np.loadtxt(output_file, dtype=float)
# nt = len(thick)
# # error on thicknesses
# t_err = 0.005
# # radius of D1 substrate (mm)
# rd = 16.5 # diameter is 33mm
# # grid spacing step (or size of pixels, on one side) (mm) across surface plane of dichroic
# dd = 1#0.1
    

# number of grid elements (or pixels) along one side (given by 2*radius*step_size)
    nside = 2*rd/dd
    # number of points (or pixels) across the entire square plane with side 'nside' (given by nside**2)
    print('Number of points simulated: ',nside**2)
    
    # creating a cylinder mask over the square plane to represent dichroic stack
    # dist() creates an array where each element is the distance in pixels to the nearest corner (useful for 2D)
    # shift by half of the number of pixels on each side, so 0 is now at the centre of the array
    # this gives a cone from the centre of the array, useful when tou neeed an array of radial distances. 
    # mask = shift(dist(nside),nside/2,nside/2) lt (rd/0.1)
    # roll the rows and columns around and get an array with distances from the centre point. 
    # using pythagoras to calc the distance from the centre to each element in an array of size (xsize,ysize)
    # the centre point has a value of zero
    x_size, y_size = nside, nside
    # the X and Y coordinates of the grid, respectively. These arrays will have values ranging from 0 to x_size-1 for x_arr and from 0 to y_size-1 for y_arr.
    x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
    # This line calculates the coordinates of the center of the grid (assuming 0-based indexing). It uses (nside-1)/2 for both X and Y coordinates to find the center point.
    cell = ((nside-1)/2, (nside-1)/2)
    # This line calculates the Euclidean distance between each pixel in the grid and the center point cell. It uses the Pythagorean theorem to compute the distance. The resulting dists array will contain the distances of each pixel from the center.
    dists = np.sqrt((x_arr - cell[0])**2 + (y_arr - cell[1])**2)
    # gives cylinder of 1s where condition is met (values below (rd/0.1)), otherwise to zero. 
    # This means that pixels within a certain distance from the center (defined by (rd/dd)) will have a value of 1, and pixels outside that distance will have a value of 0 in the mask array.
    # i.e. if a point is within the number of pixels equaling the radius of the circle, the condition is met
    mask = np.where(dists < (rd/dd),1,0)
    
    # a 2D array of integers from 0 to nside-1, diving this array by nside ???????
    # These lines create two 1D arrays x and y, which represent the X and Y coordinates, respectively, for the pixels in the grid. The np.arange function is used to generate sequences of numbers from 0 to nside-1.
    x = np.arange(0, nside)
    y = np.arange(0, nside)
    # full coordinate arrays
    # This line uses the np.meshgrid function to create two 2D arrays, xind and yind, which represent the X and Y coordinates of the pixels in the grid. These arrays will have the same values as the x and y arrays, respectively, but they are organized in grid format, where each element corresponds to a specific pixel's coordinates.
    xind, yind = np.meshgrid(x, y)
    # creating a grid of xx and yy values, centred on centre of circle
    # this effectively shifts the grid coordinates by rd/dd units in both the X and Y directions.
    xx=xind-rd/dd
    yy=yind-rd/dd
    
    return nside, xx, yy, x_size, y_size, mask


# Model of deposition & plot of cumulative deposition
def spatial_deposition(wl,thick,nt,nside,xx,yy,mask,t_err,rd,dd,model_type,orientation,OutputFilePath):

    # Change in thickness across plane: 3D array of zeroes, for each layer and each grid element
    d_thick_xy = np.zeros((int(nt),int(nside),int(nside)))
    thick_xy_altered = np.zeros((int(nt),int(nside),int(nside)))
    disk = 0
    
    # generating random array of angles from uniform distriubtion, for each layer, given by random*2*pi
    # I think... generating random angles from 0 to 2*pi, from a uniform distribution 
    if orientation == 'random':
        orient_values = nt
    elif orientation == 'constant':
        orient_values = 1
        
    if model_type == '1st_order_gradient':
        # In this line, a random angle is generated. np.random.uniform(size=1) generates a random number between 0 and 1. Multiplying this random number by 2 * np.pi scales it to cover the entire range of angles in radians, which is from 0 to 2π (360 degrees). So, angle represents a random angle in radians.
        # angle = np.random.uniform(size=orient_values)*2*np.pi
        np.random.seed(nt)
        angle = np.random.uniform(-np.pi,np.pi,size=orient_values)

    elif model_type == '2nd_order_gradient':
        np.random.seed(nt)
        # generates random numbers from a uniform distribution between 0 (inclusive) and 1 (exclusive). 
        # the code scales and shifts them to be in the range of [-1, 1). Multiplying by 2 scales the values to [0, 2), and then subtracting 1 shifts them to the desired range of [-1, 1).
        coeff_rnd_2 = np.random.uniform(size=(6,orient_values))#*2-1  # this was initially by Giorgio random constants between -1 and 1, but I changed it to between 0 and 1 to create more of a 'focus'-type term #x^2,y^2,xy,x,y,c

        
    fig,axes = plt.subplots(5,8, figsize=(20,20),sharex=True, sharey=True)
    
    for klay,ax in zip(range (nt),axes.flatten()): # for kth layer from 0 to nt-1
        
        if model_type == '1st_order_gradient':
            
            rr = np.sqrt(xx**2+yy**2)
            ang = np.arctan2(yy,xx)
            rmax = nside/2
            
            # test = x_centre*cos(angle_of_kth_layer) + y_centre*sin(angle_of_kth_layer) ???????
            if orientation == 'random':
                # computes a linear combination of xx and yy based on the random angle. This combination creates a gradient where the direction and magnitude of the gradient depend on the angle.
                # In other words, the test array will have values that represent the projection of the grid coordinates onto a line with the angle defined by angle. The result is a gradient that changes direction and magnitude as the random angle changes.
                # test = xx*np.cos(angle[klay])+yy*np.sin(angle[klay])#/rr
                test = rr/rmax*np.cos(ang-angle[klay])*mask
            elif orientation == 'constant':
                # computes a linear combination of xx and yy based on the random angle. This combination creates a gradient where the direction and magnitude of the gradient depend on the angle.
                # In other words, the test array will have values that represent the projection of the grid coordinates onto a line with the angle defined by angle. The result is a gradient that changes direction and magnitude as the random angle changes.
                # test = xx*np.cos(angle)+yy*np.sin(angle)#/rr
                test = rr/rmax*np.cos(ang-angle)*mask

            # apply mask to test and scaled by the deposition error thickness added to the regular thickness. This step makes the gradient extend along the third dimension (thickness) of the disk.
            # 2*rd/dd is the number of pixels within the diameter. In other words, the diameter normalised by pixel size.
            # The gradient is then scaled by the normalised diameter, making sure that the gradient values are scaled appropriately based on the size of the disk in terms of pixel units.
            # ensuring that the gradient's intensity is distributed evenly across the disk or adjusting the gradient's values to match the size of the region of interest (the disk)
            # i.e. it scales the gradient across the disk, while the thickness error scales it in the third dimension. 
            add2disk = test*mask*(thick[klay]*(t_err/2))#/(2*rd/dd)
  
        elif model_type == '2nd_order_gradient':
            
            # c_1 * (xx**2): This term represents a quadratic function along the x-axis (horizontal). It's scaled by c_1, which is a random constant.
            # c_2 * (yy**2): This term represents a quadratic function along the y-axis (vertical). It's scaled by c_2, which is another random constant.
            # c_3 * (xx * yy): This term represents the cross-product term between xx and yy. It's scaled by c_3, which is yet another random constant.
            # c_4 * xx: This term represents a linear function along the x-axis. It's scaled by c_4, which is a random constant.
            # c_5 * yy: This term represents a linear function along the y-axis. It's scaled by c_5, which is a random constant.
            
            np.random.seed(nt*5)
            # defining the centre of the second order gradient, no longer 0,0 but a random point x_c,y_c
            centre_rnd_2 = np.random.uniform(0,5.5,size=(2,orient_values))
            x_c = centre_rnd_2[0,:] # x centre
            y_c = centre_rnd_2[1,:] # y centre
            
            if orientation == 'random':
                
                test = coeff_rnd_2[0,klay]*((xx-x_c[klay])**2) + coeff_rnd_2[1,klay]*((yy-y_c[klay])**2) + coeff_rnd_2[2,klay]*((xx-x_c[klay])*(yy-y_c[klay]))
                test += coeff_rnd_2[3,klay]*(xx-x_c[klay]) + coeff_rnd_2[4,klay]*(yy-y_c[klay])
                test = test/(np.max(test)-np.min(test))
            elif orientation == 'constant':
                test = coeff_rnd_2[0]*((xx-x_c[klay])**2) + coeff_rnd_2[1]*((yy-y_c[klay])**2) + coeff_rnd_2[2]*((xx-x_c[klay])*(yy-y_c[klay]))
                test += coeff_rnd_2[3]*(xx-x_c[klay]) + coeff_rnd_2[4]*(yy-y_c[klay])
                test = test/(np.max(test)-np.min(test))
            # apply mask to test and multiply by (thickness_of_kth_array*error/2*radius*grid_spacing) ???????
            # ensuring that the gradient's intensity is distributed evenly across the disk or adjusting the gradient's values to match the size of the region of interest (the disk)
            # apply mask to test and scaled by the deposition error thickness added to the regular thickness. This step makes the gradient extend along the third dimension (thickness) of the disk.
            # 2*rd/dd is the number of pixels within the diameter. In other words, the diameter normalised by pixel size.
            # The gradient is then scaled by the normalised diameter, making sure that the gradient values are scaled appropriately based on the size of the disk in terms of pixel units.
            # ensuring that the gradient's intensity is distributed evenly across the disk or adjusting the gradient's values to match the size of the region of interest (the disk)
            # i.e. it scales the gradient across the disk, while the thickness error scales it in the third dimension. 
            add2disk = test*mask*(thick[klay]*(t_err))#/(2*rd/dd))
                
        add2disk[mask == 0] = np.nan
        # cumulatively add each layer's disk array to all the previous ones
        # this is a cumulative, incremental total of additions to each layer
        disk += add2disk
        # scaling the intensity values of Image into the range of the image ???????
        # showing image of disk for each layer
        ax.set_title('Layers ≤'+str(klay+1))
        ax.imshow(add2disk) 
            
        # set kth layer value of d_thick_xy to disk 
        d_thick_xy[klay,:,:]=add2disk
        thick_xy_altered[klay,:,:]=d_thick_xy[klay,:,:]+thick[klay]
    
    fig.set_tight_layout(True)
    fig.savefig(f"{OutputFilePath}/plots/Phase_error_layers_{model_type}.pdf",dpi=600)

    return d_thick_xy, thick_xy_altered


# Phase Error Calculations
def phase_error_calc(wl,nside,x_size,y_size,front_phys_l_length, thick_xy_altered,phys_l_substrate,n_front,n_substrate,n_bbar,theta,polar,ideal_phase_r,ideal_phase_t):
    
    phase_xy_r = np.zeros((len(wl),int(nside),int(nside)))
    phase_xy_t = np.zeros((len(wl),int(nside),int(nside)))

    for i_x in range(int(x_size)):
        for j_y in range(int(y_size)):
            # print(f"The element in the {i}th row, {j}th col is {thick_xy_altered[:][i][j]}")
            recipe_altered_front = thick_xy_altered[:front_phys_l_length,i_x,j_y]
            recipe_altered_bbar = thick_xy_altered[front_phys_l_length:,i_x,j_y]

            _, _, _, r_phase_altered, t_phase_altered = spectral_analysis(wl,n_front,n_substrate,n_bbar,recipe_altered_front,phys_l_substrate,recipe_altered_bbar,theta,polar)
            # _,_,_,phi_phase_tracking_refl_front,phi_phase_tracking_transm_front,_ = p_s_polar_combo_transmline(n_single,recipe_altered,wl,theta,polar,convolution_std,component_type)    
            phase_xy_r[:,i_x,j_y] = r_phase_altered
            phase_xy_t[:,i_x,j_y] = t_phase_altered
    
    phase_error_xy_output_r = np.real(phase_xy_r - ideal_phase_r[:,None,None]) # calculating the phase error by removing the mean phase
    phase_error_xy_output_t = np.real(phase_xy_t - ideal_phase_t[:,None,None]) # calculating the phase error by removing the mean phase

    return phase_error_xy_output_r,phase_error_xy_output_t


