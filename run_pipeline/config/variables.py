import numpy as np

# Functions
from classical_exps.core.tools.utils import plot_img, pickleread

## Models
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble

#######################
##### Directories #####
#######################

## Main directory
main_dir = "/project"


## Sub directory for containing the elements to run the pipeline (the main.py, config.py and file.h5)
run_dir = main_dir + "/run_pipeline"

## Sub directory containing elements/data  useful for the analyses
objects_dir = run_dir + "/objects"


###############################
##### PART I : Parameters #####
###############################

#####    ####   #####    ####   ##   ##   ####
#    #  #    #  #    #  #    #  # # # #  #
#####   ######  #####   ######  #  #  #   ### 
#       #    #  #   #   #    #  #     #      #
#       #    #  #    #  #    #  #     #  #### 



## Name of the HDF5 file 
h5_file = run_dir + "//results_convnext_model.h5"

## Select the model with all neurons
all_neurons_model = v1_convnext_ensemble

## Chose the indices of the neurons to work with
n = 458
neuron_ids = np.arange(n) ## Because 458 outputs in our model
# neuron_ids = neuron_ids[neuron_ids<20] ## Just to test the pipeline with a smaller number of neurons, can be removed to run with all neurons
corrs = pickleread(run_dir + '/objects/avg_corr.pkl') ## The correlation score of the neurons
# neuron_ids = neuron_ids[corrs>0.75] 

## Grating parameters to test
orientations = np.linspace(0, np.pi, 37)[:-1] 
spatial_frequencies = np.linspace(1, 7, 25) 
phases = np.linspace(0, 2*np.pi, 37)[:-1]

## Parameters for the dot stimulation
dot_size_in_pixels_gauss = 4
num_dots=200000
bs = 40
seed = 0

## Fixed image parameters
contrast = 1         
img_res = [93,93] 
pixel_min = -1.7876 # (the lowest value encountered in the data that served to train the model, serves as the black reference)
pixel_max =  2.1919 # (the highest [...] serves at white reference )
size = 2.67 

## Allow negative responses (because the response we show is the difference between the actual response and the response to a gray screen)
neg_val = True ## Keep True

## EXPERIMENTS 1 arguments
## Because the model is cut at the boundary of 2.67 but in experiment the monitor goes beoyond that, we should continue the annulus into infinity (eg. max radii)
radii = np.logspace(np.log10(0.15),np.log10(1.89),8) # log10(0.15)
## For the contrast response experiment
center_contrasts = np.logspace(np.log10(0.06),np.log10(1),18) 
surround_contrasts = np.logspace(np.log10(0.06),np.log10(1),6)
# contrasts_article_1 = [3, 6, 9, 13, 25, 37, 50, 75, 100] ## same as in the article, just added some more resolution
contrasts_article_1 = np.array([0.06, 0.13, 0.25, 0.5, 1.0])


## EXPERIMENTS 2 arguments
## For the orientation tuning experiment
ori_shifts = np.linspace(-np.pi/2,np.pi/2,4, endpoint=False)
experiment_2_contrast = 0.5
## For the ccss experiment
center_contrasts_ccss = np.array([0.0, 0.03, 0.06, 0.12, 0.25, 0.5])
surround_contrasts_ccss = np.array([0.0, 0.03, 0.06, 0.12, 0.25, 0.5])

## EXPERIMENTS 3 arguments
contrasts_article_3 = np.logspace(np.log10(0.06),np.log10(1),5)  ## Same as in the article
dot_size_in_pixels = 5  

## EXPERIMENTS 4 arguments
directory_imgs = objects_dir + '/shareStim_NN13' #TODO fix
target_res = img_res
num_samples = 15

### ANALYSIS ARGUMENTS

## Filtering parameters
fit_err_thresh = 0.2
supp_thresh = 0.1
SNR_thresh = 2

##size_tuning_results_2

sort_by_std = False
spread_to_plot = [0,15,50,85,100]

## contrast_size_tuning_results_1
shift_to_plot = spread_to_plot
low_contrast_id  =  0
high_contrast_id = -1

## Center Contrast Surround Suppression
high_center_contrast_id      = -3 ## The contrast of the center corresponding to the 'high' contrast
high_norm_center_contrast_id = high_center_contrast_id
low_center_contrast_id       = 1  ## The contrast of the center corresponding to the 'low' contrast
low_norm_center_contrast_id  = low_center_contrast_id  

## black_white_results_1
neuron_depths = pickleread(objects_dir + "/depth_info.pickle") 

## texture_noise_response_results
wanted_fam_order = ['60', '56', '13', '48', '71', '18', '327', '336', '402', '38', '23', '52', '99', '393', '30'] ## The order of the textures

## Example Pipeline
neuron_id = 0
