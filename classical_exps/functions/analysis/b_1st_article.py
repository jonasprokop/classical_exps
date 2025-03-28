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
from classical_exps.functions.experiments import get_GSF_surround_AMRF
## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.functions.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy


##############################################################################
#####  PART II : Analyses for the first article, Cavanaugh et al., 2002  #####
#####   --------------------------------------------------------------   #####
#####        Nature and Interaction of Signals From the Receptive        #####
#####          Field Center and Surround in Macaque V1 Neurons           #####
#####   --------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00692.2001           #####
##################################################################################
                        

def plot_scatter_hist(
    x,
    y,
    title,
    x_label,
    y_label,
    log_axes = True
    ):

    ''' This function aims to create a scatter plot with two histograms on its sides
        It is used to compare the GSF diameter and the surround diameter

        Arguments :

            - x, y              : The arrays to compare
            - title             : The title on the plot
            - x_label, y_label  : The labels on the plot
            - log_axes          : If set to True, will plot with logarithmic axes
    '''

    ## Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))

    ## Name the fig
    plt.suptitle(title)

    ## Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    ## the size of the marginal axes and the main axes in both directions.
    ## Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
    
    
    ## Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ## Draw the scatter plot and the marginals
    ## The scatter plot:
    ax.scatter(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_axes :
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Set formatters
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Set limits
    
    xymin = min(np.min(np.abs(x)), np.min(np.abs(y)))
    minedge = max(0.06,xymin - 0.5) 
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    maxedge = xymax +1
    ax.set_xlim(minedge, maxedge)
    ax.set_ylim(minedge, maxedge)

    # Plot diagonal line
    ax.plot([minedge, maxedge], [minedge, maxedge], 'k--')

    ## The histograms
    nbins = 15

    ## To show proportions 
    weights_x = np.ones(x.shape) / len(x)
    weights_y = np.ones(y.shape) / len(y)

    if log_axes :
        ax_histx.hist(x, density = False, weights = weights_x, bins = np.logspace(np.log10(minedge), np.log10(maxedge),nbins))
        ax_histy.hist(y, density = False, weights = weights_y, orientation='horizontal', bins = np.logspace(np.log10(minedge), np.log10(maxedge),nbins))
    else :
        ax_histx.hist(x, density = False, weights = weights_x, bins = np.linspace(minedge, maxedge,nbins))
        ax_histy.hist(y, density = False, weights = weights_y, orientation='horizontal', bins = np.linspace(minedge, maxedge,nbins))
    
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    
    plt.show()


def plot_size_tuning_curve(
    h5_file,
    neuron_id
):
    ''' This function goes into a HDF5 file, get the size tuning curves for one neuron and plot them.

        Prerequisite :

            - function 'size_tuning_experiment_all_phases' executed for the required neuron

        NB  : The plot x axis shows diameters, not radii    
     
    '''
    group_size_tuning_path = "/size_tuning"
    subgroup_tuning_curves_path = group_size_tuning_path + "/curves"

    ## Check if the size tuning experiment has been performed
    check_group_exists_error(h5_file, group_size_tuning_path)

    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file, [subgroup_tuning_curves_path], [neuron_id])

    with h5py.File(h5_file, 'r') as file :

        ## Get the parameters used in the size tuning experiment
        group = file[group_size_tuning_path]
        group_args_str = group.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        radii = group_dico_args["radii"]
        ## Convert str to a list
        radii = get_list_from_str(radii)

        ## Get the tuning curves
        neuron = f"neuron_{neuron_id}"
        subgroup = file[subgroup_tuning_curves_path]
        both_curves = subgroup[neuron][:]
        
        circular_tuning_curve = both_curves[0]
        annular_tuning_curve  = both_curves[1]

    
    ## Multiply the radii to get the diameters
    plt.plot(radii*2,circular_tuning_curve, label = "center")
    plt.plot(radii*2, annular_tuning_curve, label = "surround")
    plt.xlabel("Diameter (deg)")
    plt.ylabel('Response')
    plt.legend()
    plt.title(f"Size tuning curve of the Neuron {neuron_id}")
    plt.show()


def plot_contrast_response(
    h5_file,
    neuron_id,
    title = None
):

    ''' This function goes into a HDF5 file, get the contrast response curves for one neuron and plot them.
        
        NB :    yaxis  = Response of the single neuron modele , 
                xaxis  = Contrast of the center
                curves = One curve for each surround contrast

        Prerequisite :

            - function 'contrast_response_experiment' executed for the required neuron    
        
        Arguments :

            - title : (Optional) str : The customized title for the plot. If set to 'None', it will set a default title
    '''

    group_path_cr = "/contrast_response"
    group_path_cr_curves = group_path_cr + "/curves"

    ## Check if the contrast response experiment has been performed
    check_group_exists_error(h5_file, group_path_cr_curves)

    ## Check if the neuron is present in the data
    check_neurons_presence_error(h5_file, [group_path_cr_curves], [neuron_id])

    with h5py.File(h5_file, 'r') as file :

        ## Get the parameters used in the size tuning experiment
        group = file[group_path_cr]
        group_args_str = group.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        center_contrasts = group_dico_args["center_contrasts"]
        surround_contrasts = group_dico_args["surround_contrasts"]
        ## Convert str to a list
        center_contrasts = get_list_from_str(center_contrasts)
        surround_contrasts = get_list_from_str(surround_contrasts)
    
        ## Get the curves
        neuron = f"neuron_{neuron_id}"
        subgroup = file[group_path_cr_curves]
        contrast_response_curves = subgroup[neuron][:]


    ## plot each curve
    for i in range(len(contrast_response_curves)) :
        label = round(surround_contrasts[i],2)
        plt.plot(center_contrasts,contrast_response_curves[i], label = str(label))

    ## Set the title if no title was given
    if title is None :
        title = f"Response contrast of the neuron {neuron_id}"

    plt.legend(title='Surround contrast', title_fontsize='large')
    plt.title(title)
    plt.xlabel("Center contrast")
    plt.ylabel("Response")
    plt.xscale('log')

    plt.show()


def plot_contrast_size_tuning_curve(
    h5_file,
    neuron_id,
    title = None
    ):

    ''' This function goes into a HDF5 file, get the contrast size tuning response curves for one neuron and plot them.

        Prerequisite :

            - function 'contrast_size_tuning_experiment_all_phases' executed for the required neuron    
    
        NB : yaxis  = Response of the single neuron modele , 
             xaxis  = diameter (radius * 2)
             curves = One curve for each contrast

    '''

    group_path_cst = "/contrast_size_tuning"
    group_path_cst_curves = group_path_cst + "/curves"

    ## Check if the contrast response experiment has been performed
    check_group_exists_error(h5_file, group_path_cst_curves)

    ## Check if the neuron is present in the data
    check_neurons_presence_error(h5_file, [group_path_cst_curves], [neuron_id])

    with h5py.File(h5_file, 'r') as file :

        ## Get the parameters used in the size tuning experiment
        group = file[group_path_cst]
        group_args_str = group.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        contrasts = group_dico_args["contrasts"]
        radii = group_dico_args["radii"]
        ## Convert str to a list
        contrasts = get_list_from_str(contrasts)
        radii = get_list_from_str(radii)
    
        ## Get the curves
        neuron = f"neuron_{neuron_id}"
        subgroup = file[group_path_cst_curves]
        cst_curves = subgroup[neuron][:]

    ## Reverse the order for better visualisation
    for i, contrast in enumerate(contrasts[::-1]) :
        label = round(contrast,2)
        plt.plot(radii*2,cst_curves[- (i+1)], label = str(label) )
        plt.xlabel("diameter (deg)")
        plt.ylabel('response')

    ## Set the title if no title was given
    if title is None :
        title = f"Contrast Size Tuning Curves of the neuron {neuron_id}"


    plt.legend(title='Contrast', title_fontsize='large')
    plt.title(title)
    plt.show()
        

def size_tuning_results_1(  
    h5_file,
    neuron_ids,
    fit_err_thresh = 0.2,
    supp_thresh = 0.1
    ):

    ''' This function aims to visualise the relation between the GSF diameter and the surround extent diameter :
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. (number of neurons filtered, mean GSF ...)
            - 3) It plots a scatter plot of the values, with histograms showing the distribution,

        Prerequisite :
            
            - function 'size_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'
            - Exclude the neurons that have a suppression index too low. function 'filter_low_supp_neurons'

        Arguments :

            - fit_err_thresh : Fitting error threshold, exclude every neuron with an error above that value
            - supp_thresh    : Suppression index threshold, exclude every neuron with a SI below that value
    '''
    
    ## These functions verify if the prerequisite functions are performed for the requested neurons
    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    ## Filter the neurons with a surround suppression too low
    filtered_neuron_ids = filter_low_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, supp_thresh=supp_thresh, print_results = False)

    group_path = "/size_tuning/results"

    with h5py.File(h5_file, 'r') as f :

        ## For each neuron, get the GSF and the surround extent
        all_GSF = []
        all_surr_ext = []

        for neuron_id in filtered_neuron_ids :

            neuron = f"neuron_{neuron_id}"

            GSF      = f[group_path][neuron][:][1]
            surr_ext = f[group_path][neuron][:][2]

            all_GSF.append(GSF)
            all_surr_ext.append(surr_ext)

    all_GSF = np.array(all_GSF)
    all_surr_ext = np.array(all_surr_ext)
    
    ## Multiply by 2 to get diameters
    all_GSF *= 2
    all_surr_ext *= 2
    
    ## Informations to print
    n = len(neuron_ids)
    n_new = len(all_GSF)
    mean_GSF = round(np.mean(all_GSF),3)
    mean_surr_ext = round(np.mean(all_surr_ext),2)
    mean_ratio    = round(np.mean(all_surr_ext/all_GSF),2)

    ## Parameters for the plot
    title = f"GSF diameter vs surround extent diameter for {n_new} neurons"
    x_label = "GSF diameter (deg)"
    y_label = "surround diameter (deg)"

    ## Show the results
    print("--------------------------------------")
    print("Comparison of the GSF and surround extent diameter :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean diameter of the GSF is {mean_GSF}")
    print(f"    > The mean diameter of the surround_extent is {mean_surr_ext}")
    print(f"    > The mean ratio between the surround and the GSF is {mean_ratio}")
    print(f"    > Plot :")
    plot_scatter_hist(x=all_GSF, y=all_surr_ext,title=title, x_label=x_label, y_label=y_label)
    print("--------------------------------------")
    print()


    return
    

def size_tuning_results_2(
    h5_file,
    neuron_ids,
    fit_err_thresh = 0.2
    ):
    ''' This function aims to visualize the distribution of the suppression index in the requested neuron set :
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. (number of neurons filtered, mean SI ...)
            - 3) It plots an histogram of the neuron SI values

        Prerequisite :
            
            - function 'size_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - see 'size_tuning_results_1'
    '''
    
    ## These functions verify if the prerequisite functions are performed for the requested neurons
    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    group_path = "/size_tuning/results"

    with h5py.File(h5_file, 'r') as f :

        ## For each neuron, get the GSF and the surround extent
        all_SI = []

        for neuron_id in filtered_neuron_ids :

            neuron = f"neuron_{neuron_id}"

            SI = f[group_path][neuron][:][-1]

            all_SI.append(SI)
    
    all_SI = np.array(all_SI)
    n = len(neuron_ids)
    n_new = len(all_SI)
    mean_SI = round(np.mean(all_SI), 2)
    max_SI = round(max(all_SI),1)
    bins = np.linspace(0,max(max_SI,1),6)

    ## Show the results
    print("--------------------------------------")
    print("Distribution of the Suppression Index :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean value of the SI is {mean_SI}")
    print(f"    > Plot :")

    weights = np.ones(all_SI.shape) / len(all_SI)
    plt.hist(all_SI, bins = bins,edgecolor='black', density=False, weights=weights)
    plt.xlabel("Suppression Index (SI)" )
    plt.ylabel('Distribution')
    plt.xticks(bins)
    plt.title(f"Distribution of the SI for the {len(all_SI)} neurons")

    plt.show()

    print("--------------------------------------")
    print()


def sort_by_spread(
    h5_file,
    neuron_ids,
    sort_by_std = False
):
    ''' This function sorts the neurons from the ones with the lowest spread values to the ones with the highest.  
        Spread is assessed thanks to the mean_std or maxmin_ratio values computed in the 'contrast_response_experiment' function

        Prerequisite :
            
            - function 'contrast_response_experiment' executed for the required neurons

        Arguments :

            - sort_by_std   : If set to true, sort with the mean_std value, if set to false, sort with max/min value.

        Outputs :

            - sorted_neuron_ids    : The sorted neuron ids
            - sorted_spread        : The sorted spread values (either std or maxmin)
    '''
    group_path = "/contrast_response/results"

    ## Check if the 'contrast_response_experiment' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=group_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[group_path], neuron_ids=neuron_ids)

    sorted_neuron_ids = np.copy(neuron_ids)

    with h5py.File(h5_file, 'r') as f :

        if sort_by_std :
            
            all_std = []

            for neuron_id in sorted_neuron_ids : 

                neuron = f"neuron_{neuron_id}"

                ## Get the neurons mean_std
                mean_std = f[group_path][neuron][:][1]
                all_std.append(mean_std)

            ## Sort accross mean_std
            order = np.argsort(all_std)

            ## Order the spread values
            sorted_spread = np.array(all_std)[order]

        else :

            all_maxmin = []

            for neuron_id in sorted_neuron_ids : 

                neuron = f"neuron_{neuron_id}"

                ## Get the neurons mean_std
                maxmin = f[group_path][neuron][:][0]
                all_maxmin.append(maxmin)

            ## Sort accross max/min
            order = np.argsort(all_maxmin)

            ## Order the spread values
            sorted_spread = np.array(all_maxmin)[order]

    ## Order the neuron_ids
    sorted_neuron_ids = sorted_neuron_ids[order]

    return sorted_neuron_ids, sorted_spread


def contrast_response_results_1(
    h5_file,
    neuron_ids,
    fit_err_thresh = 0.2,
    sort_by_std = False,
    spread_to_plot = [15, 50, 85]
    ):
    ''' This function aims to visualize some representative contrast response curves for the requested neuron set :
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. (number of neurons filtered, mean spread ...)
            - 3) It sorts the array of neurons according to the how spread their curves are
            - 4) It plots the contrast response curves for neurons representing different contrasts

        To estimate how spread the curves are for a neuron, there are two possibilities :

            - Either take the last points of the curves and calculate the ratio : max_response/min_response
            - Either compute the mean standard deviation for every points

        Prerequisite :
        
            - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'contrast_response_experiment' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - sort_by_std    : Bool, decides which method to use to estimate how the curves are spread. If set to 'True', it will use the second method and sort with the mean_std value
            - spread_to_plot : An array containing the position of the neuron in the sorted spread array (in percentage). 100 means that it's the neuron with the highest spread, 50 means that 50% of the neurons have a smaller spread.
            - other          : see 'size_tuning_results_1'
    '''
     
    ## These functions verify if the prerequisite functions are performed for the requested neurons
    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    ## Sort the neurons
    sorted_neuron_ids, sorted_spread = sort_by_spread(h5_file=h5_file, neuron_ids=filtered_neuron_ids, sort_by_std=sort_by_std)

    n = len(neuron_ids)
    n_new = len(sorted_neuron_ids)

    ## Show the results
    mean_spread = round(np.mean(np.array(sorted_spread)),2)
    median_spread = round(np.median(np.array(sorted_spread)),2)

    print("--------------------------------------")
    print("Visualisation of representative contrast response curves :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    if sort_by_std :
        print(f"    > The median of mean_std is {median_spread}")
    else :
        print(f"    > The median of max/min is {median_spread}")

    print(f"    > Plots :")

    for spread_percent in spread_to_plot :

        neuron_pos = int((spread_percent/100) * (n_new-1))
        neuron_id  = sorted_neuron_ids[neuron_pos]
        
        print(f"    > {spread_percent}% of spread, neuron {neuron_id} :")

        title = f"Response contrast for a spread above {spread_percent}% of the neurons (neuron {neuron_id})"

        plot_contrast_response(h5_file=h5_file, neuron_id=neuron_id, title=title)

    print("--------------------------------------")
    print()


def sort_by_shift(
    h5_file,
    neuron_ids,
    low_contrast_id = None,
    high_contrast_id = None
):
    ''' This function sorts the neurons from the ones with the lowest shift value to the ones with the highest.  
        shift is assessed thanks to the GSF at low contrast divided by the GSF at high contrast. Computed in the 'contrast_size_tuning_experiment_all_phases' function

        Prerequisite :
            
            - function 'contrast_size_tuning_experiment_all_phases' executed for the required neurons

        Arguments :

            - low_contrast_id  : Int (Optional) If None, it will take the low contrast to be the lowest contrast computed in the experiment. If not None, the low contrast will be the corresponding id in the 'contrasts' array (the array containing every contrasts tested)
            - high_contrast_id : Same for high contast
            
        Outputs :

            - sorted_neuron_ids    : The sorted neuron ids
            - sorted_shift         : The sorted shift values
    '''
    group_path = "/contrast_size_tuning/results"
    curves_path= "/contrast_size_tuning/curves"

    ## Check if the 'contrast_size_tuning_experiment_all_phases' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=group_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[group_path], neuron_ids=neuron_ids)

    sorted_neuron_ids = np.copy(neuron_ids)

    with h5py.File(h5_file, 'r') as f :
        
        ## Access the groups
        group_results = f[group_path]
        group_curves  = f[curves_path]
        
        ## Get the radii (useful if low_contrast_id or high_contrast_id is not None)
        group_args_str = group_curves.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        radii = group_dico_args["radii"]
        ## Convert str to a list
        radii = get_list_from_str(radii)

        all_shift = []

        for neuron_id in sorted_neuron_ids : 

            neuron = f"neuron_{neuron_id}"

            ## Get the neurons shift
            shift = group_results[neuron][:][0]

            ## Get low_contrast_id or high_contrast_id are not None
            if low_contrast_id is not None or high_contrast_id is not None :

                ## Assert none of them are None
                if low_contrast_id is None :
                    low_contrast_id = 0
                if high_contrast_id is None :
                    high_contrast_id = -1
                
                ## Get the corresponding curves
                low_contrast_curve = group_curves[neuron][:][low_contrast_id]
                high_contrast_curve = group_curves[neuron][:][high_contrast_id]
                GSF_low,_,_,_,_  = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=low_contrast_curve, annular_tuning_curve = None)
                GSF_high,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=high_contrast_curve, annular_tuning_curve = None)

                shift = np.float64(GSF_low/GSF_high)

            
            all_shift.append(shift)

        ## Sort accross mean_std
        order = np.argsort(all_shift)
    
    ## Order the neuron ids and the shift values
    sorted_neuron_ids = sorted_neuron_ids[order]
    sorted_shift = np.array(all_shift)[order]

    return sorted_neuron_ids, sorted_shift


def contrast_size_tuning_results_1(
    h5_file,
    neuron_ids,
    fit_err_thresh = 0.2,
    shift_to_plot = [15,50,85],
    low_contrast_id = None,
    high_contrast_id= None
): 
    ''' This function aims to visualize what happens when a size tuning experiment is performed at different contrasts :

            - 1) It performs some filtering.
            - 2) It gets the ratio of the GSF at the lowest contrast divided by the GSF at the highest contrast (GSFlow/GSFhigh)
            - 3) It prints the mean value of the GSFlow/GSFhigh, which basically represents how the receptive field radius change when lowering the contrast
            - 4) It plots the curves for neurons representing different shifts
        NB : 'shift' refers to the 'GSFlow/GSFhigh ratio'

        Prerequisite :
        
            - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'contrast_size_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - shift_to_plot    : An array containing the position of the neurons to plot in the sorted shift array (in percentage). 100 means that it's the neuron with the highest shift, 50 means that 50% of the neurons have a smaller shift.
            - low_contrast_id  : Int (Optional) If None, it will take the low contrast to be the lowest contrast computed in the experiment. If not None, the low contrast will be the corresponding id in the 'contrasts' array (the array containing every contrasts tested)
            - high_contrast_id : Same for high contast
            - other            : see 'size_tuning_results_1'
    '''
    
    ## These functions verify if the prerequisite functions are performed for the requested neurons
    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    ## Sort the neurons
    sorted_neuron_ids, sorted_shift = sort_by_shift(h5_file=h5_file, neuron_ids=filtered_neuron_ids, low_contrast_id=low_contrast_id, high_contrast_id=high_contrast_id)    

    ## Show the results
    mean_shift = round(np.mean(np.array(sorted_shift)),2)
    n = len(neuron_ids)
    n_new = len(sorted_shift)
    print("--------------------------------------")
    print("Visualisation of representative contrast size tuning curves :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean shift value is {mean_shift}")

    print(f"    > Plots :")

    for shift_percent in shift_to_plot :

        neuron_pos = int((shift_percent/100) * (n_new-1))
        neuron_id  = sorted_neuron_ids[neuron_pos]
        
        print(f"    > {shift_percent}% of shift, neuron {neuron_id}, GSFlow/high = {round(sorted_shift[neuron_pos],2)} :")

        title = f"Response contrast for a shift above {shift_percent}% of the neurons (neuron {neuron_id})"

        plot_contrast_size_tuning_curve(h5_file=h5_file, neuron_id=neuron_id, title=title)

    print("--------------------------------------")
    print()
