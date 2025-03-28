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
    plt.show()


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


def texture_noise_response_results_1(
    h5_file, 
    neuron_ids,
    wanted_fam_order = None
):  
    ''' This function aims to visualise if the neurons product the same response to texture or noise images
    
        Prerequisite :
            
            - function 'texture_noise_response_experiment' executed for the required neurons

        Filtering :
            
            - None
        
        Arguments :

            - dict_fam_order : (Optional) An array containing the families name in the desired order (make sure the names of the families match). If set to False, random order, 

    '''

    group_path    = '/texture_noise_response'
    subgroup_tex_path   = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    ## Check if the 'texture_noise_response' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_noise_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_noise_path,subgroup_tex_path], neuron_ids=neuron_ids)

    ## Show the results
    print("--------------------------------------")
    print("Visualisation of neurons response to texture and noise images :")
    print(f"    > Analysis made on {len(neuron_ids)} neurons")
    print(f"    > Plot :")

    ## Get the results for every neuron, shape = (n_family, n_sample, n_neurons)
    all_tex_resp   = []
    all_noise_resp = []

    with h5py.File(h5_file,  'r') as file :
        
        subgroup_tex = file[subgroup_tex_path]
        subgroup_noise = file[subgroup_noise_path]

        ## Get the families id (for the plot)
        description = subgroup_tex.attrs["description"]
        family_ids  = description.split('-')
        family_ids  = np.asarray(family_ids)

        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"

            ## Get the responses to texture and noise for this neuron
            neuron_results_tex    = subgroup_tex[neuron][:]
            neuron_results_noise  = subgroup_noise[neuron][:]

            all_tex_resp.append(neuron_results_tex)
            all_noise_resp.append(neuron_results_noise)

    all_tex_resp   = torch.Tensor(np.stack(all_tex_resp))
    all_noise_resp = torch.Tensor(np.stack(all_noise_resp))
    

    ## Computes the mean responses accross every neuron
    mean_all_tex_resp   = torch.mean(all_tex_resp, dim=0)
    mean_all_noise_resp = torch.mean(all_noise_resp, dim=0)
    
    ## For every family, computes the mean response and the standard deviation to textures and the mean response to noise
    mean_tex_resp   = torch.zeros(len(all_tex_resp[0]))
    mean_noise_resp = torch.zeros(mean_tex_resp.shape)
    std_tex_resp    = torch.zeros(mean_tex_resp.shape)
    std_noise_resp  = torch.zeros(mean_tex_resp.shape)

    n_sample = len(all_tex_resp[0])

    ## Plot for the mean accross neurons
    for i in range(len(mean_all_tex_resp)) :
            
        ## For the texture
        mean_tex_resp[i]   = torch.mean(mean_all_tex_resp[i,:])
        std_tex_resp[i]    = torch.std(mean_all_tex_resp[i,:]) 
        ## For the noise
        mean_noise_resp[i] = torch.mean(mean_all_noise_resp[i,:])
        std_noise_resp[i]  = torch.std(mean_all_noise_resp[i,:])
    
    ## Computes the standart deviation of the mean
    sem_tex_resp = std_tex_resp / math.sqrt(n_sample)
    sem_noise_resp = std_noise_resp / math.sqrt(n_sample)

    ## Change the order if required
    if wanted_fam_order is not None :
                
        dict_old_order = {}

        ## Get a dictionnary dict[name_family] = position in array
        pos = 0
        for i in family_ids :
            dict_old_order[i] = pos
            pos += 1

        wanted_fam_order = np.asarray(wanted_fam_order)
        
        ## Get the new order
        new_order = []
        for fam in wanted_fam_order :
            new_order.append(dict_old_order[fam])
        new_order = np.array(new_order)


    print('Mean accross every neuron')
    plt.title('Mean responses to texture and noise images')


    x_ticks = np.copy(family_ids)

    if wanted_fam_order is not None : 
        x_ticks = x_ticks[new_order]
        mean_tex_resp = mean_tex_resp[new_order]
        mean_noise_resp = mean_noise_resp[new_order]
        
    max_val = max(max(mean_tex_resp),max(mean_noise_resp))
    max_error = max(max(sem_tex_resp), max(sem_noise_resp))
    max_plot  = max_val + max_error
    max_plot  = max_plot + 0.05* max_plot
    y_ticks = np.arange(0,max_plot,0.5)


    plt.errorbar(x_ticks, mean_tex_resp, yerr=sem_tex_resp, fmt='o', capsize=7, capthick=2, elinewidth=2, markersize=12, color='darkgoldenrod', label='Texture')    
    plt.errorbar(x_ticks, mean_noise_resp, yerr=sem_noise_resp, fmt='o', capsize=7, capthick=2, elinewidth=2, markersize=12, color='orange', label='Noise', alpha=0.65)    

    plt.xlabel('Texture family')
    plt.ylabel('Mean response')
    
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.legend()

    plt.show()
    
    ## Plot for 3 random neurons
    for iter in range(3) :

        ## For every family, computes the mean response and the standard deviation to textures and the mean response to noise
        mean_tex_resp   = torch.zeros(len(all_tex_resp[0]))
        mean_noise_resp = torch.zeros(mean_tex_resp.shape)
        std_tex_resp    = torch.zeros(mean_tex_resp.shape)
        std_noise_resp  = torch.zeros(mean_tex_resp.shape)

        r = np.random.randint(0,len(neuron_ids))
        r = [0,1,2][iter]
        for i in range(len(all_tex_resp[0])) :
            
            ## For the texture
            mean_tex_resp[i]   = torch.mean(all_tex_resp[r,i,:])
            std_tex_resp[i]    = torch.std(all_tex_resp[r,i,:])
            ## For the noise
            mean_noise_resp[i] = torch.mean(all_noise_resp[r,i,:])
            std_noise_resp[i]  = torch.std(all_noise_resp[r,i,:])

        ## Computes the standart deviation of the mean
        sem_tex_resp = std_tex_resp / math.sqrt(n_sample)
        sem_noise_resp = std_noise_resp / math.sqrt(n_sample)

        rand_neuron_id = neuron_ids[r]

        print(f'Random neuron : {rand_neuron_id}')
        plt.title(f'Responses to texture and noise images for the neuron {rand_neuron_id}')

        x_ticks = family_ids 

        if wanted_fam_order is not None : 
            x_ticks = x_ticks[new_order]
            mean_tex_resp = mean_tex_resp[new_order]
            mean_noise_resp = mean_noise_resp[new_order]
            
        max_val = max(max(mean_tex_resp),max(mean_noise_resp))
        max_error = max(max(sem_tex_resp), max(sem_noise_resp))
        max_plot  = max_val + max_error
        max_plot  = max_plot + 0.05* max_plot
        y_ticks = np.arange(0,max_plot,0.5)

        plt.errorbar(x_ticks, mean_tex_resp, yerr=sem_tex_resp, fmt='o', capsize=7, capthick=2, elinewidth=2, markersize=12, color='darkgreen', label='Texture')    
        plt.errorbar(x_ticks, mean_noise_resp, yerr=sem_noise_resp, fmt='o', capsize=7, capthick=2, elinewidth=2, markersize=12, color='limegreen', label='Noise', alpha=0.7)    

        plt.xlabel('Texture family')
        plt.ylabel('Response')

        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.legend()
    
        plt.show()

    print("--------------------------------------")


def texture_noise_response_results_2(
    h5_file, 
    neuron_ids,
    wanted_fam_order = None
):  
    ''' This function aims to visualise the average modulation index accross every neuron for each texture family.
        The modulation index is defined as so : (response_texture - response_noise) / (response_texture + response_noise)

        Prerequisite :
            
            - function 'texture_noise_response_experiment' executed for the required neurons

        Filtering :
            
            - None 
    '''

    group_path    = '/texture_noise_response'
    subgroup_tex_path   = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    ## Check if the 'texture_noise_response' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_noise_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_noise_path,subgroup_tex_path], neuron_ids=neuron_ids)

    ## Show the results
    print("--------------------------------------")
    print("Visualisation of the average modulation index for each texture family:")
    print(f"    > Analysis made on {len(neuron_ids)} neurons")
    print(f"    > Plot :")

    ## Get the results for every neuron shape = (n_family, n_sample, n_neurons)
    all_tex_resp   = []
    all_noise_resp = []

    with h5py.File(h5_file,  'r') as file :
        
        subgroup_tex = file[subgroup_tex_path]
        subgroup_noise = file[subgroup_noise_path]

        ## Get the families id (for the plot)
        description = subgroup_tex.attrs["description"]
        family_ids  = description.split('-')

        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"

            ## Get the responses to texture and noise for this neuron
            neuron_results_tex    = subgroup_tex[neuron][:]
            neuron_results_noise  = subgroup_noise[neuron][:]

            all_tex_resp.append(neuron_results_tex)
            all_noise_resp.append(neuron_results_noise)

    all_tex_resp   = torch.Tensor(np.stack(all_tex_resp))
    all_noise_resp = torch.Tensor(np.stack(all_noise_resp))
    
    ## Change the order if required
    if wanted_fam_order is not None :
                
        dict_old_order = {}
        ## Get a dictionnary dict[name_family] = position in array
        pos = 0
        for i in family_ids :
            dict_old_order[i] = pos
            pos += 1

        wanted_fam_order = np.asarray(wanted_fam_order)
        
        ## Get the new order
        new_order = []
        for fam in wanted_fam_order :
            new_order.append(dict_old_order[fam])
        new_order = np.array(new_order)

    ## Compute every modulation index
    all_modulation_index = (all_tex_resp - all_noise_resp) / (all_tex_resp + all_noise_resp)

    ## Average the modulation accross every neuron
    all_modulation_index = torch.mean(all_modulation_index, dim=0)

    ## Average the modulation index accross every sample
    all_modulation_index = torch.mean(all_modulation_index, dim=1)
    
    if wanted_fam_order is not None : 

        family_ids = np.array(family_ids)[new_order]
        all_modulation_index = all_modulation_index[new_order]

    plt.bar(family_ids, all_modulation_index, width=0.8, color = 'limegreen', edgecolor='black')

    minval = min(all_modulation_index)
    maxval = max(all_modulation_index)
    minlim = min(-0.15, minval - abs(0.1 * minval))
    maxlim = max(0.3, maxval + abs(0.1 * maxval))

    plt.ylim([minlim, maxlim])
    plt.xlabel('Texture family')
    plt.ylabel('Modulation index')
    plt.title('Modulation index of every texture family\n averaged accross every neuron')

    plt.show()

    print("--------------------------------------")


def texture_noise_response_results_3(
    h5_file, 
    neuron_ids
):  
    ''' This function aims to visualise the distribution of the modulation index averaged accross every texture family and every sample
        The modulation index is defined as so : (response_texture - response_noise) / (response_texture + response_noise)

        Prerequisite :
            
            - function 'texture_noise_response_experiment' executed for the required neurons

        Filtering :
            
            - None
    '''

    group_path    = '/texture_noise_response'
    subgroup_tex_path   = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    ## Check if the 'texture_noise_response' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_noise_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_noise_path,subgroup_tex_path], neuron_ids=neuron_ids)

    ## Show the results
    print("--------------------------------------")
    print("Visualisation of distribution of the mean Modulation index for every neuron :")
    print(f"    > Analysis made on {len(neuron_ids)} neurons")
    print(f"    > Plot :")

    ## Get the results for every neuron shape = (n_family, n_sample, n_neurons)
    all_tex_resp   = []
    all_noise_resp = []

    with h5py.File(h5_file,  'r') as file :
        
        subgroup_tex = file[subgroup_tex_path]
        subgroup_noise = file[subgroup_noise_path]

        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"

            ## Get the responses to texture and noise for this neuron
            neuron_results_tex    = subgroup_tex[neuron][:]
            neuron_results_noise  = subgroup_noise[neuron][:]

            all_tex_resp.append(neuron_results_tex)
            all_noise_resp.append(neuron_results_noise)

    all_tex_resp   = torch.Tensor(np.stack(all_tex_resp))
    all_noise_resp = torch.Tensor(np.stack(all_noise_resp))
    
    ## Compute every modulation index
    all_modulation_index = (all_tex_resp - all_noise_resp) / (all_tex_resp + all_noise_resp)

    ## Average the modulation accross every neuron
    all_modulation_index = torch.mean(all_modulation_index, dim=2)

    ## Average the modulation index accross every sample
    all_modulation_index = torch.mean(all_modulation_index, dim=1)
    
    meanval = torch.mean(all_modulation_index).item()

    weights = np.ones(all_modulation_index.shape) / len(all_modulation_index)
    values, bins, _ = plt.hist(all_modulation_index, edgecolor='black', bins=10, density=False, weights=weights, color= 'limegreen')
    max_hist = max(values)

    plt.plot([0,0], [0,max_hist + 0.1 * max_hist], 'k--', label='zero')
    plt.plot([meanval,meanval], [0, max_hist + 0.1 * max_hist], color = 'darkgoldenrod', label = 'mean')

    minval = min(bins)
    maxval = max(bins)
    minlim = min(-0.5, minval - abs(0.1 * minval))
    maxlim = max( 0.5, maxval + abs(0.1 * maxval))

    plt.xlim([minlim,maxlim])
    plt.xlabel('Modulation Index')
    plt.ylabel('Distribution')
    plt.title('Distribution of the Modulation index in the neurons')
    plt.legend()
    plt.show()

    print("--------------------------------------")
