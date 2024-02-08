
import numpy as np
from tlm_model.analysis import spectral_analysis
from general_tools.analysis_tools import weighted_lsr


def mc_recipe(wl,step_size_max,bc_wl,bc_r,bc_t,w_lsr_input,min_phys_l,max_phys_l,max_tot_phys_l,MC_output_front,MC_output_bbar,n_front,n_substrate,n_bbar,phys_l_front,phys_l_substrate,phys_l_bbar,theta,polar): 
    """
    Monte Carlo to randomly vary the thicknesses of a given recipe to move closer to the desired performance. 

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths.
    step_size : float
        Initial maxmimum variation in thicknesses [%].
    bc_wl : numpy.ndarray
        Wavelength sampling of boundary conditions.
    bc_r : numpy.ndarray
        Boundary conditions of reflectance [%].
    bc_r : numpy.ndarray
        Boundary conditions of transmittance [%].
    w_lsr_input : float
        Weighted least-square residual between intial guess and boundary conditions.
    min_phys_l : float
        Minimum thickness per coating layer.
    max_phys_l : float
        Maximum thickness per coating layer.
    max_tot_phys_l : float
        Maximum thickness of total coating stack.
    MC_output_front : str
        Name of output file for improved recipe of thicknesses of front coating.
    MC_output_bbar : str
        Name of output file for improved recipe of thicknesses of BBAR coating.
    n_front : numpy.ndarray
        Complex refractive indices of front coating and its bounding media.
    n_substrate : numpy.ndarray
        Complex refractive indices of substrate and its bounding media.
    n_bbar : numpy.ndarray
        Complex refractive indices of BBAR coating.
    phys_l_front : numpy.ndarray
        Physical thicknesses of front coating layers.
    phys_l_substrate : numpy.ndarray
        Physical thickness of substrate.
    phys_l_bbar : numpy.ndarray
        Physical thicknesses of BBAR coating layers.
    theta : numpy.ndarray
        Angle of incidence [degrees].
    polar : str
        Linear polarisation type – 'TE', 'TM' or 'unpolarised'.

    Returns
    -------
    phys_l_front_error : numpy.ndarray
        Improved recipe of front coating thicknesses.
    phys_l_bbar_error : numpy.ndarray
        Improved recipe of BBAR coating thicknesses.
    w_lsr_output : float
        Weighted least-square residuals of output design performance.
    diff_lsr : float
        Change in weighted least-square residuals of output design performance and previous design performance.

    """
    
    phys_l = np.concatenate((phys_l_front, phys_l_bbar))    
    count = 0
    step_size_max = step_size_max/100 # maximum step size
    
    step_size = step_size_max # starting step size set to maximum
    # when count reach maximum, exit loop as solution may be a local minima
    while step_size > 1e-5 or count < 1e8: # stop MC if set difference between LSR and previous LSR is below number
        # generate random vector between -1 and 1 from a uniform distribution 
        phys_l = np.concatenate((phys_l_front, phys_l_bbar))
        random_variation = 2* np.random.rand(len(phys_l)) -1
        error = random_variation * (phys_l * step_size)
        # generate a different random vector between 0 and 1 from a uniform distribution 
        # implement boolean condition to randomly select some coating layers
        random_number = np.random.rand(len(phys_l))
        bool_cond = np.where(random_number < 0.5, 1, 0) * error
        # apply error to randomly selected coating layers
        phys_l_error = phys_l + bool_cond
        # seperate array into front and BBAR coating thicknesses
        phys_l_front_error = phys_l_error[:len(phys_l_front)]
        phys_l_bbar_error = phys_l_error[len(phys_l_front):]
        # performance of dichroic after applying random thickness variations 
        varied_r,varied_t,_ = spectral_analysis(wl,n_front,n_substrate,n_bbar,phys_l_front_error,phys_l_substrate,phys_l_bbar_error,theta,polar)
        # integrated difference between altered performance and boundary condition performance
        w_lsr_varied_r = weighted_lsr(wl,varied_r,bc_wl,bc_r)
        w_lsr_varied_t = weighted_lsr(wl,varied_t,bc_wl,bc_t)
        w_lsr_varied = w_lsr_varied_r + w_lsr_varied_t
        # difference in initial design performance and varied design performance weighted LSR values
        diff_lsr = w_lsr_input - w_lsr_varied
        
        # if weighted LSR of varied recipe is less and coating thicknesses follow the design restrictions
        if w_lsr_varied < w_lsr_input and np.min(phys_l) >=min_phys_l and np.max(phys_l) <=max_phys_l and np.sum(phys_l)<=max_tot_phys_l: #saving new model values if they are a better fit # added condition to hve all layers larger than 10nm
            # update weighted LSR of input recipe
            w_lsr_input = w_lsr_varied
            phys_l = phys_l_error
            np.savetxt(MC_output_front, phys_l_front_error, fmt='%1.9f') #save new thickness estimates to textfile
            np.savetxt(MC_output_bbar, phys_l_bbar_error, fmt='%1.9f') #save new thickness estimates to textfile
            print('\n Weighted LSR=',w_lsr_input,'\n Improvement=',diff_lsr)
            # minor decrease to variation to find minima in MC
            step_size = step_size*0.5
            print('\n Variation set to: ',step_size)
            print('-----------------')
            count = 0 # reset count    
        else:
            count += 1
            # reset the step size to its maximum to look for other minima
            step_size = step_size_max
            print('\n Variation reset to: ',step_size)
            
    w_lsr_output = w_lsr_input
    return phys_l_front_error,phys_l_bbar_error,w_lsr_output,diff_lsr


