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
from classical_exps.functions.article_1.common.metrics_size import get_GSF_surround_AMRF## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.functions.utils import plot_img
## Data storage
import h5py
from collections import Counter
import scipy



###################################################################################
#####  PART IV : Experiments for the third article, Chun-I Yeh et al., 2009  #####
#####   -------------------------------------------------------------------   #####
#####        “Black” Responses Dominate Macaque Primary Visual Cortex V1      #####
#####   -------------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1523/JNEUROSCI.1991-09.2009       #####
###################################################################################


def plot_logSNRwb(
    logSNR_OFF,
    logSNR_ON,
    neuron_depths_OFF = None,
    neuron_depths_ON = None
    ):
    ''' This functions plots the Signal Noise Ratios 

        Arguments : 

            - logSNR_OFF : the values of log(SNRwhite / SNRblack) that are < 0
            - logSNR_ON  :  ""  ""  ""  ""  ""  ""  ""  ""  ""  ""  ""  "" > 0

        NB : make sure the depths are discrete values and not continuous to plot the correct mean
    '''

    if neuron_depths_OFF is not None :
        y_OFF = neuron_depths_OFF
        y_ON  = neuron_depths_ON

    else :
        ## Create random y values to faciliate the reading of the plot
        y_OFF = np.random.rand(len(logSNR_OFF))
        y_ON  = np.random.rand(len(logSNR_ON))

    ## MAKE THE PLOT

    ## Start with a square Figure.
    fig = plt.figure(figsize=(8, 6))

    ## Name the fig
    plt.suptitle('Visualisation of the neurons color preference')

    ## Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    ## the size of the marginal axes and the main axes in both directions.
    ## Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 1,  height_ratios=(1, 4),
                    left=0.1, right=0.9, bottom=0.1, top=0.9,
                    wspace=0.05, hspace=0.05)
    
    ## Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[0, 0], sharex=ax)
    
    ## Scatter plot
    ax.scatter(logSNR_OFF, y_OFF, color='0', label = 'black-dominant')
    ax.scatter(logSNR_ON, y_ON, color='1', edgecolor ='k', label = 'white-dominant')
    

    # ## Set nice y axis
    max_y = max(max(y_OFF), max(y_ON))
    ax.set_ylim(max_y + 0.1 * max_y, - 0.10)

    ## Add vertical dashed line at zero
    ax.plot([0,0],[max_y + 0.1 * max_y, - 0.10], linestyle='--', color = 'k')
    
    # Calculate the bin edges (make an edge be 0)
    min_edge = np.min(logSNR_OFF) - (np.min(logSNR_OFF)/20)
    bin_edges = np.linspace(min_edge,0,11)
    gap = abs(bin_edges[1]-bin_edges[0])
    max_val = np.max(logSNR_ON)
    n_gap = round(max_val/gap)
    max_edge =  n_gap * gap
    bin_edges = np.concatenate((bin_edges, np.linspace(0,max_edge,n_gap+1))) 

    
    all_logSNR = logSNR_OFF+logSNR_ON
    weights = np.ones(len(all_logSNR)) / len(all_logSNR)
    ax_hist.hist(all_logSNR, bins=bin_edges, density = False, edgecolor='k', weights=weights)
    ax_hist.tick_params(axis="x", labelbottom=False)

    ax.set_xlabel('log(SNR_white / SNR_black)')

    if neuron_depths_OFF is None :
        ax.set_yticks([])
    else :

        depth_unique = np.union1d(np.unique(neuron_depths_OFF),np.unique(neuron_depths_ON))
        
        ## Key of the dict is the depth, value is the count or the sum of SNR
        dict_count = {}
        dict_sum   = {}

        ## Initialise the dict values
        for depth in depth_unique :
            dict_count[depth] = 0
            dict_sum[depth]   = 0

        ## For every off neuron :
        for i, depth in enumerate(neuron_depths_OFF) :

            logSNR = logSNR_OFF[i] 

            dict_count[depth] += 1
            dict_sum[depth]   += logSNR

        ## For every on neuron :
        for i, depth in enumerate(neuron_depths_ON) :

            logSNR = logSNR_ON[i] 

            dict_count[depth] += 1
            dict_sum[depth]   += logSNR

        mean_val = np.zeros(len(depth_unique))

        for i, depth in enumerate(depth_unique) :
            
            mean_val[i] = dict_sum[depth] / dict_count[depth]

        ## Plot the mean curve
        ax.plot(mean_val, depth_unique, label = 'Mean curve')

        ## Name the y axis
        ax.set_ylabel('Depth')
    
    ax_hist.set_ylabel('Distribution')
    
    ax.legend()
    directory = f"/project/results/black_reponses_dominate" + "/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + f"plot_logSNRwb.png")
    plt.close


def black_white_results_1(
    h5_file,
    neuron_ids,
    neuron_depths = None,
    SNR_thresh = 2
):  
    ''' This function aims to visualise whether there is a black or white preference for the selected neurons

        Prerequisite :
            
            - function 'black_white_preference_experiment' executed for the required neurons

        Filtering :

            - Exclude the neurons with neither SNRw nor SNRb that are > SNR_thresh. function 'filter_SNR'

        Arguments :

            - neuron_depths : If not None, will plot the results including depth
                              IMPORTANT : neuron_depths should contain ALL NEURONS DEPTHS, so that the neuron_depths[neuron_id] correspond to the depth of that specific neuron

    '''

    ## This function verify if the prerequisite function has been performed for the requested neurons
    filtered_neuron_ids = filter_SNR(h5_file=h5_file, neuron_ids=neuron_ids, SNR_thresh=SNR_thresh, print_results=False)

    group_path = '/black_white_preference'
    

    ## Get the results
    with h5py.File(h5_file, 'r') as f:
        
        subgroup_results = f[group_path + '/results']
        
        ## Get the log_SNRwb value of every neuron, and sort them whether this value is negative (black preference = OFF) or positive (white preference = ON)
        logSNR_OFF = []
        logSNR_ON  = []

        if neuron_depths is not None :
            neuron_depths_OFF = []
            neuron_depths_ON = []

        all_SNRb   = []
        all_SNRw   = []

        for neuron_id in filtered_neuron_ids :

            neuron = f"neuron_{neuron_id}"
            
            results_neuron = subgroup_results[neuron][:]

            SNR_b    = results_neuron[0]
            SNR_w    = results_neuron[1]
            logSNRwb = results_neuron[2]

            if logSNRwb < 0 : 
                logSNR_OFF.append(logSNRwb)
                if neuron_depths is not None :
                    neuron_depths_OFF.append(neuron_depths[neuron_id])

            else :
                logSNR_ON.append(logSNRwb)
                if neuron_depths is not None :
                    neuron_depths_ON.append(neuron_depths[neuron_id])

            all_SNRb.append(SNR_b)
            all_SNRw.append(SNR_w)
    
    ## Informations to print
    n = len(neuron_ids)
    n_new = len(filtered_neuron_ids)
    mean_SNRb   = round(np.mean(all_SNRb),3)
    mean_SNRw   = round(np.mean(all_SNRw),3)
    mean_logSNR = round(np.mean(logSNR_OFF + logSNR_ON),3)

    ## Show the results
    print("--------------------------------------")
    print("Visualisation of the neurons Black or white preference :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean SNR for black stimuli is {mean_SNRb}")
    print(f"    > The mean SNR for white stimuli is {mean_SNRw}")
    print(f"    > The mean log SNR white over black is {mean_logSNR}")
    print(f"    > Plot :")
    if neuron_depths is not None :
        plot_logSNRwb(logSNR_OFF=logSNR_OFF,logSNR_ON=logSNR_ON, neuron_depths_OFF=neuron_depths_OFF, neuron_depths_ON=neuron_depths_ON)
    else :
        plot_logSNRwb(logSNR_OFF=logSNR_OFF,logSNR_ON=logSNR_ON)

    print("--------------------------------------")
    print()
        
    return

