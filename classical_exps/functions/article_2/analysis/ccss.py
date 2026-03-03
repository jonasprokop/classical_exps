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
from classical_exps.functions.article_1.analysis.shared import plot_scatter_hist, neuron_key, ensure_dir
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

def plot_ccss_curves(
    h5_file,
    neuron_id
):  
    ''' This function goes into a HDF5 file, get every values for the required neuron and plot three curves :

            - One curve for the responses of the neuron for a surround at the preffered orientation
            - One curve for the responses of the neuron for a surround at the orthogonal orientation
            - One curve for the responses of the neuron for the center only

        NB1 : ccss stands for 'center contrast surround suppression'
        NB2 : yaxis  = Response of the single neuron modele 
              xaxis  = Center contrast

        Prerequisite :

            - function 'center_contrast_surround_suppression_experiment' executed for the required neuron
    '''

    subgroup_path = "/center_contrast_surround_suppression/results"

    ## Check if the experiment has been performed
    check_group_exists_error(h5_file, subgroup_path)

    ## Check if the neuron is present in the data
    check_neurons_presence_error(h5_file, [subgroup_path], [neuron_id])

    with h5py.File(h5_file, 'r') as f :
        
        neuron = f"neuron_{neuron_id}"
        neuron_curves = f[subgroup_path][neuron][:]

        curve_iso    = neuron_curves[:,0]
        curve_ortho  = neuron_curves[:,1]
        curve_center = neuron_curves[:,2]

        ## Get the parameters used in the experiment
        group_args_str = f[subgroup_path].attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        center_contrasts_ccss = group_dico_args["center_contrasts_ccss"]
        ## Convert str to a list
        center_contrasts_ccss = get_list_from_str(center_contrasts_ccss)

        plt.plot(center_contrasts_ccss,curve_center, label = "Center alone" )
        plt.plot(center_contrasts_ccss,curve_ortho, label = "Surround ortho-oriented" )
        plt.plot(center_contrasts_ccss,curve_iso, label = "Surround iso-oriented" )
        
        plt.xscale("log")
        plt.xticks(center_contrasts_ccss)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        
        plt.title(f"CCSS Curves for the neuron {neuron_id}")
        plt.xlabel("Contrast of the center")
        plt.ylabel("Response")
        plt.legend()

        plt.show()

        
        directory = f"/project/results/selectivity_and_spatial_distribution/css_curves"  + "/" + neuron + "/"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + f"plot_ccss_curves.png")

def ccss_results_1(
    h5_file,
    neuron_ids,
    fit_err_thresh = 0.2    
):
    '''
    This function aims to visualise the relation between the center contrast and the surround suppression (with a surround at different orientations)
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. 
            - 3) It plots 3 curves, one for the surround at the preffered orientation, one for the surround at the orthogonal orientation and one for the center alone
            - 4) It does 3) for 3 random neurons in the neuron ids left after filtering

        Prerequisite :
            
            - function 'center_contrast_surround_suppression_experiment' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'
    '''
    
    group_path = "/center_contrast_surround_suppression/results"

    ## Check if the 'get_preferred_position' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=group_path)
    
    # ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[group_path], neuron_ids=neuron_ids)

    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)
    
    n = len(neuron_ids)
    n_new = len(filtered_neuron_ids)

    ## Initialise the arrays to fill
    with h5py.File(h5_file, 'r') as f :
        ## Get the parameters used in the experiment
        group_args_str = f[group_path].attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        center_contrasts_ccss = group_dico_args["center_contrasts_ccss"]
        ## Convert str to a list
        center_contrasts_ccss = get_list_from_str(center_contrasts_ccss)

        curves_iso      = np.zeros((n_new,len(center_contrasts_ccss)))
        curves_ortho    = np.zeros((n_new,len(center_contrasts_ccss)))
        curves_center   = np.zeros((n_new,len(center_contrasts_ccss)))

    ## Get the curves for every neuron
    with h5py.File(h5_file, 'r') as f :
        for i, neuron_id in enumerate(filtered_neuron_ids) :
            neuron = f"neuron_{neuron_id}"

            results_neuron = f[group_path][neuron][:]

            curve_iso    = results_neuron[:,0]
            curve_ortho  = results_neuron[:,1]
            curve_center = results_neuron[:,2]

            curves_iso[i]    = curve_iso
            curves_ortho[i]  = curve_ortho
            curves_center[i] = curve_center

    mean_curve_iso    = np.mean(curves_iso, axis=0)
    mean_curve_ortho  = np.mean(curves_ortho, axis=0)
    mean_curve_center = np.mean(curves_center, axis=0)

    print("--------------------------------------")
    print("Visualisation of ccss curves for random neurons :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print()
    print(f"    > The mean curves accross every neuron :")

    plt.plot(center_contrasts_ccss,mean_curve_center, label = "Center alone", color = 'gold')
    plt.plot(center_contrasts_ccss,mean_curve_ortho, label = "Surround ortho-oriented", color = 'goldenrod')
    plt.plot(center_contrasts_ccss,mean_curve_iso, label = "Surround iso-oriented", color = 'darkgoldenrod' )
    
    plt.xscale("log")
    plt.xticks(center_contrasts_ccss)
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    
    plt.title(f"Mean CCSS Curves")
    plt.xlabel("Contrast of the center")
    plt.ylabel("Response")
    plt.legend()

    plt.show()


    directory = f"/project/results/selectivity_and_spatial_distribution/css_result_1"  + "/" + neuron + "/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + f"ccss_results_1.png")

    
    print()
    print(f"    > Random neurons :")

    for i in range(3) : 

        ## Select a random neuron
        r = np.random.randint(0,n_new)
        neuron_id = filtered_neuron_ids[r]

        plot_ccss_curves(h5_file=h5_file, neuron_id=neuron_id)

    print("--------------------------------------")
    print()

def ccss_results_2(
    h5_file,
    neuron_ids,
    contrast_id,
    norm_center_contrast_id,
    fit_err_thresh = 0.2    
):
    '''
    This function aims to visualise the distribution of response to surround iso and ortho-oriented to the center (the center is at the preferred orientation)
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. 
            - 3) It plots a scatter plot to compare the normalised response for both conditions

        NB : The normalised response is done by dividing the response by the response to the center alone
        Prerequisite :
            
            - function 'center_contrast_surround_suppression_experiment' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - contrast_id             : The id of the contrast (corresponding to the id in the array center_contrasts_ccss used in the ccss experiment) to visualise
            - norm_center_contrast_id : The id of the center contrast that will serve for the normalisation  (corresponding to the id in the array center_contrasts_ccss used in the ccss experiment) 
    '''
    
    subgroup_path = '/center_contrast_surround_suppression/results'

    ## Check if the experiment has been run
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_path)
    
    # ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_path], neuron_ids=neuron_ids)

    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)
    
    all_iso   = []
    all_ortho = []
    ## Get the data 
    with h5py.File(h5_file, 'r') as f :

        for neuron_id in filtered_neuron_ids :
            
            neuron = f"neuron_{neuron_id}"

            results_neuron = f[subgroup_path][neuron][:]

            resp_iso    = results_neuron[contrast_id,0]
            resp_ortho  = results_neuron[contrast_id,1]
            resp_center = results_neuron[norm_center_contrast_id,2]
            norm_resp_iso   = resp_iso / resp_center
            norm_resp_ortho = resp_ortho / resp_center
            all_iso.append(norm_resp_iso)
            all_ortho.append(norm_resp_ortho)
        all_iso   = np.array(all_iso)
        all_ortho = np.array(all_ortho)

        ## Get the center contrast that correspond to contrast_id
        group = f[subgroup_path]
        group_args_str = group.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        center_contrasts_ccss = group_dico_args["center_contrasts_ccss"]
        center_contrasts_ccss = get_list_from_str(center_contrasts_ccss)
        center_contrast = str(round(center_contrasts_ccss[contrast_id], 2))

    n = len(neuron_ids)
    n_new = len(filtered_neuron_ids)

    print("--------------------------------------")
    print(f"Visualisation of the responses to ortho and iso-oriented surround (center contrast = {center_contrast}):")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean response to the ortho-oriented surround is {str(round(np.mean(all_ortho),3))}")
    print(f"    > The mean response to the iso-oriented surround is {str(round(np.mean(all_iso),3))}")
    print(f"    > Scatter Plot :")

    plot_scatter_hist(
    x=all_ortho,
    y=all_iso,
    title=f'Distribution of the response depending on the surround orientation\nContrast of the center : {center_contrast}',
    x_label='Normalised response for the \n orthogonal surround',
    y_label='Normalised response for the \n parallel surround',
    log_axes=False)
    print("--------------------------------------")
