############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
import math
## Utils
from classical_exps.functions.analysis.a_filtering_fucntions import *
from classical_exps.functions.utils import *
from classical_exps.functions.article_1.common.metrics_size import get_GSF_surround_AMRF
## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.functions.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy


##################################################################################
#####  PART III : Experiments for the second article, Cavanaugh et al., 2002 #####
#####   ------------------------------------------------------------------   #####
#####        Selectivity and Spatial Distribution of Signals From the        #####
#####            Receptive Field   Surround in Macaque V1 Neurons            #####
#####   ------------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00693.2001               #####
##################################################################################

def plot_orientation_tuning_curve(
    h5_file,
    neuron_id
):
    
    ''' This function goes into a HDF5 file, get the orientation tuning curves (center and surround with fixed center) for one neuron and plot them.

        Prerequisite :

            - function 'orientation_tuning_experiment_all_phases' executed for the required neuron    

        About the plot :

            - The dashed horizontal line correspond to the center max response at the preferred orientation
            - The x axis displays the orientation in degrees for better clarity
    '''

    group_path = '/orientation_tuning'
    subgroup_path_center = group_path + '/curves_center'
    subgroup_path_surround = group_path + '/curves_surround_fixed_center'

    ## Check if the orientation tuning experiment has been performed
    check_group_exists_error(h5_file, subgroup_path_surround)

    ## Check if the neuron is present in the data
    check_neurons_presence_error(h5_file, [subgroup_path_surround], [neuron_id])

    with h5py.File(h5_file, 'r') as f :

        ## Get the curves for the neuron
        neuron = f"neuron_{neuron_id}"
        center_tuning_curve = f[subgroup_path_center][neuron][:]
        surround_tuning_curve = f[subgroup_path_surround][neuron][:][1] ## Select the curve for a center orientation fixed at +0° relative to the preffered orientation

        ## Get the parameters used in the orientation tuning experiment
        group = f[group_path]
        group_args_str = group.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        ori_shifts = group_dico_args["ori_shifts"]
        ori_shifts = get_list_from_str(ori_shifts)

    ## Make the plot
    max_val_center = max(center_tuning_curve)
    plt.plot(np.degrees(ori_shifts), center_tuning_curve, label='Center only', color = 'darkslateblue', linewidth= 0.7)
    plt.plot(np.degrees(ori_shifts), surround_tuning_curve, label='Surround with fixed center', color = 'darkslateblue', linewidth=2.5,)
    plt.plot([min(np.degrees(ori_shifts)), max(np.degrees(ori_shifts))], [max_val_center, max_val_center], linestyle='--',color = 'k', linewidth=1)

    ## Change the ticks for better clarity
    custom_ticks = [-180,-135,-90,-45,0,45,90,135,180]
    plt.xticks(custom_ticks)

    ## Unzoom a little
    minresp = min(min(center_tuning_curve),min(surround_tuning_curve))
    maxresp = max(max(center_tuning_curve),max(surround_tuning_curve))
    plt.ylim(minresp-0.3, maxresp+0.3)

    plt.title(f"Orientation Tuning Curve for the neuron {neuron_id}")
    plt.xlabel("Orientation shift compared to the preferred orientation (degrees)")
    plt.ylabel("Response")
    plt.legend()

    directory = f"/project/results/selectivity_and_spatial_distribution/orientation_tunning_curve"  + "/" + neuron + "/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + f"orietation_tunning_curve_{neuron_id}.png")
    plt.close

def orientation_tuning_results_1(
    h5_file, 
    neuron_ids, 
    fit_err_thresh=0.2
):
    ''' This function aims to visualise the response of neurons to different center and surround orientations :
            
            - 1) It performs some filtering.
            - 2) It prints some informations in the terminal. 
            - 3) It plots the center orientation tuning curve and the surround orientation tuning curve with fixed center

        Prerequisite :
        
            - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'orientation_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - fit_err_thresh : Fitting error threshold, exclude every neuron with an error above that value
    '''

    group_path    = '/orientation_tuning'
    subgroup_path = group_path + '/results'

    ## Check if the 'get_preferred_position' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_path], neuron_ids=neuron_ids)

    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    n = len(neuron_ids)
    n_new = len(filtered_neuron_ids)

    ## Initialise the arrays to fill
    with h5py.File(h5_file, 'r') as f :
        ## Get the parameters used in the experiment
        group_args_str = f[subgroup_path].attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        ori_shifts = group_dico_args["ori_shifts"]
        ## Convert str to a list
        ori_shifts = get_list_from_str(ori_shifts)

        curves_center      = np.zeros((n_new,len(ori_shifts)))
        curves_surround    = np.zeros((n_new,len(ori_shifts)))

    ## Get the curves for every neuron
    with h5py.File(h5_file, 'r') as f :
        for i, neuron_id in enumerate(filtered_neuron_ids) :
            neuron = f"neuron_{neuron_id}"

            curve_center    = f[group_path + '/curves_center'][neuron][:]
            curve_surround  = f[group_path + '/curves_surround_fixed_center'][neuron][:][1]

            curves_center[i] = curve_center
            curves_surround[i] = curve_surround

    mean_curve_center = np.mean(curves_center, axis=0)
    mean_curve_surround = np.mean(curves_surround, axis=0)

    print("--------------------------------------")
    print("Visualisation of random orientation tuning curves :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print()
    print(f"    > The mean curves accross every neuron :")

    ## Make the plot
    max_val_center = max(mean_curve_center)
    plt.plot(np.degrees(ori_shifts), mean_curve_center, label='Center only', color = 'darkgoldenrod', linewidth= 0.7)
    plt.plot(np.degrees(ori_shifts), mean_curve_surround, label='Surround with fixed center', color = 'darkgoldenrod', linewidth=2.5,)
    plt.plot([min(np.degrees(ori_shifts)), max(np.degrees(ori_shifts))], [max_val_center, max_val_center], linestyle='--',color = 'k', linewidth=1)

    ## Change the ticks for better clarity
    custom_ticks = [-180,-135,-90,-45,0,45,90,135,180]
    plt.xticks(custom_ticks)

    ## Unzoom a little
    minresp = min(min(mean_curve_center),min(mean_curve_surround))
    maxresp = max(max(mean_curve_center),max(mean_curve_surround))
    plt.ylim(minresp-0.3, maxresp+0.3)

    plt.title(f"Mean orientation tuning curves accross every neuron")
    plt.xlabel("Orientation shift compared to the preferred orientation (degrees)")
    plt.ylabel("Response")
    plt.legend()

    directory = f"/project/results/selectivity_and_spatial_distribution/mean_orientation_tunning_curves"  + "/" + neuron + "/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + f"orientation_tuning_results_1.png")
    plt.close


    print()
    print(f"    > Random neurons :")

    for i in range(3) : 

        ## Select a random neuron
        r = np.random.randint(0,n_new)
        neuron_id = filtered_neuron_ids[r]

        plot_orientation_tuning_curve(h5_file=h5_file, neuron_id=neuron_id)

    print("--------------------------------------")
    print()

def orientation_tuning_results_2(
    h5_file, 
    neuron_ids, 
    fit_err_thresh=0.2
):
    ''' This function to show the most suppressive surround orientations for different center orientations
            
            - 1) It performs some filtering.
            - 2) It prints some informations in the terminal. 
            - 3) It plot 3 histograms, one for each center orientation (-45°, 0 and +45°), the histograms shows the distribution of the most suppressive surround orientation
        NB : The orientation is the orientation compared to the preferred orientation

        Prerequisite :
        
            - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'orientation_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

    '''
    subgroup_path = '/orientation_tuning/results'

    ## Check if the 'get_preferred_position' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_path], neuron_ids=neuron_ids)

    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    ## Get the neurons results
    all_result_0 = []   #center orientation  : -45
    all_result_1 = []   #center orientation  :   0
    all_result_2 = []   #center orientation  :  45

    with h5py.File(h5_file, 'r') as f : 
        
        for neuron_id in filtered_neuron_ids :

            neuron = f"neuron_{neuron_id}"

            neuron_results = f[subgroup_path][neuron][:]
            result_0 = neuron_results[0]
            result_1 = neuron_results[1]
            result_2 = neuron_results[2]

            all_result_0.append(result_0)
            all_result_1.append(result_1)
            all_result_2.append(result_2)

            ## Get the parameters used in the orientation tuning experiment
            group = f['/orientation_tuning']
            group_args_str = group.attrs["arguments"]
            group_dico_args = get_arguments_from_str(group_args_str)
            ori_shifts = group_dico_args["ori_shifts"]
            ori_shifts = get_list_from_str(ori_shifts)

    ## Select the orientation shift >= -90 and <= 90 degrees
    sub_array_id = np.where(np.abs(ori_shifts) < np.pi/2 + 0.01 )[0] # In radiant for now
    sub_ori = ori_shifts[sub_array_id]
    sub_ori = np.degrees(sub_ori) #Convert in degrees

    n = len(neuron_ids)
    n_new = len(filtered_neuron_ids)

    ## Print in the terminal
    print("--------------------------------------")
    print("Visualisation of the most suppressive surround orientations :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > Plots :")

    all_results = [all_result_0, all_result_1, all_result_2]
    for i, orientation in enumerate([-45,0,45]):
        
        print()
        print(f"    > For center orientation of {orientation} deg :")
        print()

        results = np.degrees(all_results[i])

        ## Create the good amount of bins
        nbins = len(sub_ori)
        step  = np.abs(sub_ori[-1] - sub_ori[-2])
        edge  = sub_ori[-1] + (step/2)
        bins = np.linspace(- edge, edge,nbins+1) # The values should not be below -90 and above 90 deg
        weights = np.ones(results.shape) / len(results)
        plt.hist(results, bins = bins,edgecolor='black', density=False, weights=weights )

        ## Annotate the plot
        plt.xticks(sub_ori)
        plt.title(f"Distribution of the most suppressive surround orientation for a center at {orientation} deg")
        plt.ylabel("Distribution")
        plt.xlabel("Most suppressive surround direction\nrelative to the preferred one (deg)")
        plt.show()


        directory = f"/project/results/selectivity_and_spatial_distribution/orientation_tunning_curve"  + "/" + neuron + "/"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + f"orientation_tuning_results_2.png")
        plt.close
