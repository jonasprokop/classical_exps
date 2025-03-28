############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
import math
## Utils
from classical_exps.functions.utils import *
from classical_exps.functions.experiments import get_GSF_surround_AMRF
## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.functions.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy


########################################
##### PART I : Filtering functions #####
########################################


def filter_fitting_error(
    h5_file, 
    neuron_ids,
    fit_err_thresh = 0.2,
    print_results = False
):
    ''' The goal of this function is to return only the neurons with a center that fits well to a Gaussian model.
        The function goes in the h5_file, searches for the '/preferred_pos' group and selects the neurons with an error below 
        the threshold. It will also exclude the neurons with error = np.nan

        Prerequisite :
            
            - function 'get_preferred_position' executed for the required neurons    

        Arguments : 

            - fit_err_thresh : The threshold of fitting error, a value above this will lead to the neuron to be excluded
            - print_results   : (Optional) If set to 'True', it will print the amount of neurons excluded in the terminal

        Outputs :

            - filtered_neuron_ids : An array containing the neurons that are well fitted
    '''
    
    group_path = '/preferred_pos'

    ## Check if the 'get_preferred_position' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=group_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[group_path], neuron_ids=neuron_ids)

    filtered_neuron_ids = []

    with h5py.File(h5_file,  'r') as file :

        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"
            error = file[group_path][neuron][:][2]

            ## keep the neuron if its error is below the threshold
            if error < fit_err_thresh :
                filtered_neuron_ids.append(neuron_id)

    if print_results :

        n = len(neuron_ids)
        new_n = len(filtered_neuron_ids)

        ## Show the results
        print("--------------------------------------")
        print("Filtering the Neurons poorly fitted :")
        print(f"    > There were initially {n} neurons")
        print(f"    > {n-new_n} neurons were removed ({round(100*((n-new_n)/n),1)}%)")
        print(f"    > There is {new_n} neurons left")
        print("--------------------------------------")
        print()
        
    return np.array(filtered_neuron_ids)
    

def filter_low_supp_neurons(
    h5_file, 
    neuron_ids,
    supp_thresh = 0.1,
    print_results = False
    ):

    ''' The goal of this function is to select only the neurons with a decent surround suppression.
        The function goes in the h5_file, searches for the '/size_tuning/results' group and selects the neurons with 
        a suppression index above the threshold.

        Prerequisite :
            
            - function 'size_tuning_experiment_all_phases' executed for the required neurons

        Arguments :

            - supp_thresh      : Threshold, neurons with a suppression index below that value will be excluded
            - print_results    : (Optional) If set to 'True', it will print the amount of neurons excluded in the terminal

        Outputs :

            - filtered_neuron_ids : An array containing the neurons that have a decent suppression index

    '''

    group_path = '/size_tuning/results'

    ## Check if the 'size_tuning_experiment_all_phases' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=group_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[group_path], neuron_ids=neuron_ids)

    filtered_neuron_ids = []

    with h5py.File(h5_file,  'r') as file :

        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"
            SI = file[group_path][neuron][:][4]

            ## keep the neuron if its error is below the threshold
            if SI > supp_thresh :
                filtered_neuron_ids.append(neuron_id)

    if print_results :

        n = len(neuron_ids)
        new_n = len(filtered_neuron_ids)

        ## Show the results
        print("--------------------------------------")
        print("Filtering the Neurons with low suppression :")
        print(f"    > There were initially {n} neurons")
        print(f"    > {n-new_n} neurons were removed ({round(100*((n-new_n)/n),1)}%)")
        print(f"    > There is {new_n} neurons left")
        print("--------------------------------------")
        print()
        
    return np.array(filtered_neuron_ids)


def filter_no_supp_neurons(
    h5_file, 
    neuron_ids,
    print_results = False
    ):
    ''' This function excludes unwanted neurons from our analyses.
        The unwanted neurons are neurons whith no surround suppression nor response saturation. (basically those are the neurons for which increasing the size of the stimulus will always lead to a greater response)
        The function goes in the h5_file, searches for the '/size_tuning/results' group and selects the neurons as so : 
            - It excludes neurons that have a negative Suppression Index
            - Having negative (or null) suppression index means that their is neither response saturation nor suppression

        Prerequisite :
            
            - function 'size_tuning_experiment_all_phases' executed for the required neurons

        Arguments :

            - see filter_low_supp_neurons     

        Outputs :

            - filtered_neuron_ids : An array containing the selected neurons
    '''
    ## Exclude neurons with a negative suppression index
    filtered_neuron_ids = filter_low_supp_neurons(h5_file=h5_file, neuron_ids=neuron_ids, supp_thresh=0.0001, print_results = False)

    filtered_neuron_ids = np.array(filtered_neuron_ids)
    
    if print_results :

        n = len(neuron_ids)
        new_n = len(filtered_neuron_ids)

        ## Show the results
        print("--------------------------------------")
        print("Filtering the Neurons with no suppression nor response saturation:")
        print(f"    > There were initially {n} neurons")
        print(f"    > {n-new_n} neurons were removed ({round(100*((n-new_n)/n),1)}%)")
        print(f"    > There is {new_n} neurons left")
        print("--------------------------------------")
        print()
        
    return np.array(filtered_neuron_ids)


def filter_SNR(
    h5_file, 
    neuron_ids,
    SNR_thresh = 2,
    print_results = False
):
    ''' This function is used in the third article only
        The goal of this function is to return only the neurons with a significant response to black or white dot stimulus
        The function goes in the h5_file, searches for the '/black_white_preference' group and selects the neurons with either SNRb or SNRw > SNR_thresh 

        Prerequisite :
            
            - function 'black_white_preference_experiment' executed for the required neurons    

        Arguments : 

            - SNR_thresh      : The threshold of SNR value, a value below this for both SNRb and SNRw will lead to the neuron to be excluded
            - print_results   : (Optional) If set to 'True', it will print the amount of neurons excluded in the terminal

        Outputs :

            - filtered_neuron_ids : An array containing the neurons that were kept

    '''

    group_path    = '/black_white_preference'
    subgroup_path = group_path + '/results'

    ## Check if the 'get_preferred_position' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_path], neuron_ids=neuron_ids)

    filtered_neuron_ids = []

    with h5py.File(h5_file,  'r') as file :
        
        subgroup = file[subgroup_path]
        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"

            ## Get the SNR
            SNRb = subgroup[neuron][:][0]
            SNRw = subgroup[neuron][:][1]

            ## keep the neuron if its error is below the threshold
            if SNRb > SNR_thresh or SNRw > SNR_thresh :
                filtered_neuron_ids.append(neuron_id)

    if print_results :

        n = len(neuron_ids)
        new_n = len(filtered_neuron_ids)

        ## Show the results
        print("--------------------------------------")
        print("Filtering the Neurons with low SNR :")
        print(f"    > There were initially {n} neurons")
        print(f"    > {n-new_n} neurons were removed ({round(100*((n-new_n)/n),1)}%)")
        print(f"    > There is {new_n} neurons left")
        print("--------------------------------------")
        print()
        
    return np.array(filtered_neuron_ids)


