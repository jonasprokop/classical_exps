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

    directory = f"/project/results/texture_noise_response" + "/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + f"mean_response_to_noice_and_images.png")
    plt.close

    
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
    
        directory = f"/project/results/texture_noise_response" + "/"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + f"responses_np_random_neuron{iter}.png")
        plt.close


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


    directory = f"/project/results/texture_noise_response" + "/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + f"average_modulation_index.png")
    plt.close


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


    directory = f"/project/results/texture_noise_response" + "/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + f"distribution_of_modulation_index.png")
    plt.close

    print("--------------------------------------")
