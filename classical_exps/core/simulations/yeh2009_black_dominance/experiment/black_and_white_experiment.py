import numpy as np
import torch
import h5py
import scipy.ndimage
from tqdm import tqdm
import os
## Utils
from classical_exps.core.tools.utils import *
from classical_exps.core.tools.utils import SingleCellModel
from classical_exps.core.simulations.yeh2009_black_dominance.experiment.tools import _map_variance, save_generated_stimuli_panel_png



##################################################################################
#####  PART IV : Experiments for the third article, Chun-I Yeh et al., 2009  #####
#####   -------------------------------------------------------------------  #####
#####        “Black” Responses Dominate Macaque Primary Visual Cortex V1     #####
#####   -------------------------------------------------------------------  #####
#####              DOI : https://doi.org/10.1523/JNEUROSCI.1991-09.2009      #####
##################################################################################


# File contains different implementations of the same experiment, with different metrics and indices. The first one is the most general one, and the two others are more strict implementations of the experiment as it was done in the article (one with variance as a metric, the other one with energy as a metric).
# The finally used one is the energy variance, as it best aproximates the paper


def black_white_preference_experiment(
    h5_file,
    all_neurons_model,
    neuron_ids,   
    overwrite = False, 
    dot_size_in_pixels=5,
    contrast = 1, 
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    seed = 42
    ):  
    ''' This function aims to find the Signal Noise Ratio (SNR) for black and white dot stimuli for the selected neurons :

            - 1) It creates the stimulations for every possible dot position
            - 2) It gets the response of the neuron of the model to the stimulations
            - 3) It creates a "position-response" image (matrix) where every pixel represents the summed response of the neuron to a dot at this position
            - 4) It creates a noise image similarly to 3) but the response is being shuffled
            - 5) Performs 1-4 for black and white stimuli and computes the Signal Noise Ratio as the variance of the position-response matrix divided by the variance of the noise
            - 6) Saves the position-response images, the noise images and the SNR in three subgroups in the HDF5 file

    
    It saves the data with this architecture : 


                                                _________ SubGroup ../position_response_img   --> neuron datasets
                                                |
        Group /black_white_preference   ________|________ SubGroup ../noise_img               --> neuron datasets
                                                |
                                                |________ SubGroup ../results                 --> neuron datasets
    Prerequisite :

        - None

    Arguments :

        - dot_size_in_pixels : Size of the pixels, corresponding to the size of the sides of the square
        - seed               : Random seed for reproducibility
        - others             : explained in other functions

    Outputs :

        - datasets in ../position_response_img  : A tensor containing two images (matrices). The first one is for black dots stimulation, the second one for white dot stimulation 
                                                  every pixel of the images correspond to the summed response of the neuron to this pixel

        - datasets in ../noise_img              :  A tensor containing two images (matrices). The first one is for black dots stimulation, the second one for white dot stimulation 
                                                   every pixel of the images correspond to the summed shuffled response of the neuron to this pixel (a shuffled response means that the response assigned to every dot can now be the response to another dot)


        - datasets in ../results                : An array containing the Signal noise ratios values and the log10(SNRw / SNRb)
                                         format : [SNR_b, SNR_w, logSNRwb]

    '''

    np.random.seed(seed)

    print(' > Black or white preference experiment')

    ## Groups to fill
    group_path = "/black_white_preference"
    subgroup_path_resp_img  = group_path + "/position_response_img"
    subgroup_path_noise_img = group_path + "/noise_img"
    subgroup_path_results   = group_path + "/results"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path)

    ## Initialize the Group and the subgroups
    args_str = f"dot_size_in_pixels={dot_size_in_pixels}/contrast={contrast}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}"
    group_init(h5_file=h5_file, group_path=group_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_resp_img, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_noise_img, group_args_str=args_str)


    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_neurons_model.to(device)
    all_neurons_model.eval()
    

    ## Generate the objects to fill
    sum_dot_pos   = torch.zeros((img_res[1], img_res[0]), dtype=int).to(device)
    black_stimuli = []
    white_stimuli = []
    resp_b = []
    resp_w = []

    # if you are out of memory, you can decrease the batching of images into the GPU
    def forward_in_batches(single_model, model_input, batch_size=32, device="cuda"):
        outputs = []

        with torch.no_grad():
            for start in range(0, model_input.shape[0], batch_size):
                end = min(start + batch_size, model_input.shape[0])

                batch = model_input[start:end].to(device)
                out = single_model(batch).detach().cpu().reshape(-1)
                outputs.append(out)

                del batch, out

        return torch.cat(outputs, dim=0)

    with torch.no_grad():
 
        ## Perform the dot stimulation
        for y in range(img_res[0] - dot_size_in_pixels + 1):
            
            for x in range(img_res[1] - dot_size_in_pixels + 1) :
            
                ## Create an image with values = 0
                black_stim = np.zeros(img_res, dtype=int)
                white_stim = np.zeros(img_res, dtype=int)

                ## Save the dot position
                sum_dot_pos[y:y+dot_size_in_pixels, x:x+dot_size_in_pixels] += 1

                ## Do one black and one white stimulus
                black_value = -1
                white_value = 1

                ## Create the dots
                black_stim[y:y+dot_size_in_pixels, x:x+dot_size_in_pixels] = black_value*contrast
                white_stim[y:y+dot_size_in_pixels, x:x+dot_size_in_pixels] = white_value*contrast

                ## Save the dots
                black_stimuli.append(torch.Tensor(black_stim))
                white_stimuli.append(torch.Tensor(white_stim))

        ## Convert the list of tensors to a unique tensor
        black_stimuli = torch.stack(black_stimuli)
        white_stimuli = torch.stack(white_stimuli)

        num_dots = len(black_stimuli)

        n_y = img_res[0] - dot_size_in_pixels + 1
        n_x = img_res[1] - dot_size_in_pixels + 1

        sample_positions = [
            (0, 0),
            (0, n_x - 1),
            ((n_y - 1) // 2, (n_x - 1) // 2),
            (n_y - 1, n_x - 1),
        ]
        sample_indices = [y * n_x + x for (y, x) in sample_positions]

        save_generated_stimuli_panel_png(
            black_stimuli=black_stimuli,
            white_stimuli=white_stimuli,
            sample_indices=sample_indices,
            sample_positions=sample_positions,
            save_path="/project/results/black_and_white_experiment/generated_stimuli_panel.png",
        )

        ## For every input get the responses of the all_neurons_model
        ## Rescale the image to pixel_min and pixel_max
        model_input_b = rescale(black_stimuli, -1, 1, pixel_min, pixel_max).reshape(num_dots, 1, img_res[0], img_res[1]).cpu()
        model_input_w = rescale(white_stimuli, -1, 1, pixel_min, pixel_max).reshape(num_dots, 1, img_res[0], img_res[1]).cpu()


        for neuron_id in tqdm(neuron_ids):

            single_model = SingleCellModel(all_neurons_model, neuron_id)
            single_model.to(device)
            single_model.eval()

            resp_b.append(forward_in_batches(single_model, model_input_b, batch_size=32, device=device))
            resp_w.append(forward_in_batches(single_model, model_input_w, batch_size=32, device=device))


        resp_b = torch.stack(resp_b, dim=1)   # shape: (num_dots, n_neurons)
        resp_w = torch.stack(resp_w, dim=1)   # shape: (num_dots, n_neurons)

        
        ## Tensor 1 is a tensor containing every dot position of the images (1 if the dot is here, else 0). shape = (num_dots * img_res) 
        ## Tensor 2 is a tensor containing the responses of every neuron to the dot images,  shape = (num_dots * nNeuron) #nNeuron = len(output_of_all_neurons_model)
        ## This function creates for every dot (first dimension = 'b') the following tensor :
        ## 'nxy' Basically, for each neuron, resp_to_the_image * dot_position_matrix_in_image
        ## It can be visualised as an array of matrices. Each matrix contains zeros where there isn't a dot and the value of the response where the dot is
        ## Then it sums this Tensor for every dot image that was presented
        ## The result image is what is called 'position response image' and baically contains the summed response of the dots at every position (pixel)
        all_position_resp_b = torch.einsum(
                            'bxy,bn->nxy', 
                            torch.abs(black_stimuli).to(device),  ## torch.abs because black values is -1
                            (resp_b).to(device)
                            )
        
        all_position_resp_w = torch.einsum(
                            'bxy,bn->nxy', 
                            white_stimuli.to(device),
                            (resp_w).to(device)
                            )
        
        ## Shuffle the responses for each neuron to create noise
        shuffle_resp_b = np.copy(resp_b.cpu())
        shuffle_resp_w = np.copy(resp_w.cpu())
        np.apply_along_axis(np.random.shuffle, axis=0, arr=shuffle_resp_b)
        np.apply_along_axis(np.random.shuffle, axis=0, arr=shuffle_resp_w)


        ## Put everything back in a tensor on the correct device
        shuffle_resp_b = torch.Tensor(shuffle_resp_b).to(device)
        shuffle_resp_w = torch.Tensor(shuffle_resp_w).to(device)
        
        all_noise_b = torch.einsum(
                            'bxy,bn->nxy', 
                            torch.abs(black_stimuli).to(device),  ## torch.abs because black values is -1
                            (shuffle_resp_b).to(device)
                            )
        
        all_noise_w = torch.einsum(
                            'bxy,bn->nxy', 
                            white_stimuli.to(device),  
                            (shuffle_resp_w).to(device)
                            )
        
    ## Fill the HDF5 file 
    with h5py.File(h5_file, 'a') as f :

        ## Save everything in a dictionnary where every key is a neuron index
         ## Save everything in a dictionnary where every key is a neuron index
        for j, neuron_id in enumerate(tqdm(neuron_ids)):

            neuron = f"neuron_{neuron_id}"

            if neuron not in f[subgroup_path_results] :
                
                ## Create datasets
                pos_resp_imgs = torch.stack([all_position_resp_b[j], all_position_resp_w[j]]).to('cpu')
                noises        = torch.stack([all_noise_b[j], all_noise_w[j]]).to('cpu')
        
                true_b = pos_resp_imgs[0].numpy()
                true_w = pos_resp_imgs[1].numpy()

                n_shuffles = 2000
                sigma = 1.0
                rng = np.random.default_rng(seed + int(neuron_id))

                ## Signal = variance of the smoothed true map
                true_b_s = scipy.ndimage.gaussian_filter(true_b, sigma=sigma)
                true_w_s = scipy.ndimage.gaussian_filter(true_w, sigma=sigma)

                signal_b = np.var(true_b_s)
                signal_w = np.var(true_w_s)

                ## Noise = mean variance of smoothed shuffled maps
                noise_vars_b = []
                noise_vars_w = []

                flat_b = true_b.ravel()
                flat_w = true_w.ravel()

                for _ in range(n_shuffles):
                    shuf_b = rng.permutation(flat_b).reshape(true_b.shape)
                    shuf_w = rng.permutation(flat_w).reshape(true_w.shape)

                    shuf_b_s = scipy.ndimage.gaussian_filter(shuf_b, sigma=sigma)
                    shuf_w_s = scipy.ndimage.gaussian_filter(shuf_w, sigma=sigma)

                    noise_vars_b.append(np.var(shuf_b_s))
                    noise_vars_w.append(np.var(shuf_w_s))

                noise_b = float(np.mean(noise_vars_b))
                noise_w = float(np.mean(noise_vars_w))

                eps = 1e-12
                SNR_b = signal_b / max(noise_b, eps)
                SNR_w = signal_w / max(noise_w, eps)
                logSNRwb = np.log10(max(SNR_w, eps) / max(SNR_b, eps))
                
                f[subgroup_path_resp_img].create_dataset(name=neuron, data=pos_resp_imgs)
                f[subgroup_path_noise_img].create_dataset(name=neuron, data=noises)
                f[subgroup_path_results].create_dataset(name=neuron, data=[SNR_b, SNR_w, logSNRwb])




# ============================================================
# Main strict, paper-aligned static approximation
# ============================================================

def black_white_preference_experiment_strict_paper_version_with_variance(
    h5_file,
    all_neurons_model,
    neuron_ids,
    *,
    overwrite=False,
    contrast=1,
    img_res=[93, 93],
    pixel_min=-1.7876,
    pixel_max=2.1919,
    device=None,
    seed=42,
    n_shuffles=2000,
    apply_smoothing=True,
    sigma=1.0,
):
    """
    Paper-aligned sparse black/white preference experiment for a static model.

    Geometry:
    - shared image-centered 12x12 grid
    - non-overlapping 7x7 px squares
    - responses stored directly in 12x12 stimulus-grid coordinates

    Signal:
    - spatial variance of the true response map
    - raw variance by default (closer to the paper)
    - optional Gaussian smoothing as robustness check

    Null baseline:
    - expected spatial variance under broken stimulus-location association
    - implemented by permuting the 12x12 response values across locations
      within the same sparse-noise ensemble

    Notes:
    - static-model proxy: no temporal reverse correlation available
    - denominator is therefore a spatial null, not a true zero-lag temporal baseline
    """

    np.random.seed(seed)

    print(" > Black or white preference experiment (strict paper-aligned static proxy)")

    group_path = "/black_white_preference"
    subgroup_path_resp_img  = group_path + "/position_response_img"
    subgroup_path_noise_img = group_path + "/noise_img"
    subgroup_path_results   = group_path + "/results"

    if overwrite:
        clear_group(h5_file, group_path)

    # ---------------------------------------------------------
    # Shared centered geometry:
    # 12x12 grid, 7x7 px squares => 84x84 sampled region
    # ---------------------------------------------------------
    H, W = int(img_res[0]), int(img_res[1])
    grid_n = 12
    dot_size_in_pixels = 7
    sampled_px = grid_n * dot_size_in_pixels  # 84

    if sampled_px > H or sampled_px > W:
        raise ValueError(
            f"Grid does not fit into img_res={img_res}: sampled_px={sampled_px}"
        )

    y_start = (H - sampled_px) // 2
    x_start = (W - sampled_px) // 2

    deg_per_px = 2.67 / W
    sampled_deg = sampled_px * deg_per_px
    dot_deg = dot_size_in_pixels * deg_per_px

    print(f"   sampled region:  {sampled_px}px = {sampled_deg:.4f} deg")
    print(f"   square size:     {dot_size_in_pixels}px = {dot_deg:.4f} deg")
    print(f"   grid start:      y={y_start}, x={x_start}")
    print(f"   variance mode:   {'smoothed' if apply_smoothing else 'raw'}")
    if apply_smoothing:
        print(f"   smoothing sigma: {sigma}")
    print(f"   null shuffles:   {n_shuffles}")

    args_str = (
        f"grid_n={grid_n}/dot_size_in_pixels={dot_size_in_pixels}/"
        f"sampled_px={sampled_px}/sampled_deg={sampled_deg:.6f}/"
        f"contrast={contrast}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}/"
        f"n_shuffles={n_shuffles}/apply_smoothing={apply_smoothing}/sigma={sigma}"
    )

    group_init(h5_file=h5_file, group_path=group_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_resp_img, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_noise_img, group_args_str=args_str)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    all_neurons_model.to(device)
    all_neurons_model.eval()

    black_stimuli = []
    white_stimuli = []
    resp_b = []
    resp_w = []

    def forward_in_batches(single_model, model_input, batch_size=32, device="cuda"):
        outputs = []
        with torch.no_grad():
            for start in range(0, model_input.shape[0], batch_size):
                end = min(start + batch_size, model_input.shape[0])
                batch = model_input[start:end].to(device)
                out = single_model(batch).detach().cpu().reshape(-1)
                outputs.append(out)
                del batch, out
        return torch.cat(outputs, dim=0)

    with torch.no_grad():
        # Generate one black and one white square per grid position
        for gy in range(grid_n):
            for gx in range(grid_n):
                y = y_start + gy * dot_size_in_pixels
                x = x_start + gx * dot_size_in_pixels

                black_stim = np.zeros((H, W), dtype=np.float32)
                white_stim = np.zeros((H, W), dtype=np.float32)

                black_stim[y:y + dot_size_in_pixels, x:x + dot_size_in_pixels] = -1.0 * contrast
                white_stim[y:y + dot_size_in_pixels, x:x + dot_size_in_pixels] =  1.0 * contrast

                black_stimuli.append(torch.tensor(black_stim, dtype=torch.float32))
                white_stimuli.append(torch.tensor(white_stim, dtype=torch.float32))

        black_stimuli = torch.stack(black_stimuli)  # (144, H, W)
        white_stimuli = torch.stack(white_stimuli)  # (144, H, W)
        num_dots = len(black_stimuli)

        # Save sanity panel from exact generated stimuli
        sample_grid_positions = [
            (0, 0),
            (0, grid_n - 1),
            (grid_n // 2, grid_n // 2),
            (grid_n - 1, grid_n - 1),
        ]
        sample_indices = [gy * grid_n + gx for (gy, gx) in sample_grid_positions]
        sample_positions = [
            (y_start + gy * dot_size_in_pixels, x_start + gx * dot_size_in_pixels)
            for (gy, gx) in sample_grid_positions
        ]

        save_generated_stimuli_panel_png(
            black_stimuli=black_stimuli,
            white_stimuli=white_stimuli,
            sample_indices=sample_indices,
            sample_positions=sample_positions,
            save_path="/project/results/black_and_white_experiment/paper_strict/generated_stimuli_panel.png",
        )

        # Rescale once, keep on CPU, batch to GPU later
        model_input_b = rescale(black_stimuli, -1, 1, pixel_min, pixel_max).reshape(num_dots, 1, H, W).cpu()
        model_input_w = rescale(white_stimuli, -1, 1, pixel_min, pixel_max).reshape(num_dots, 1, H, W).cpu()

        # Per-neuron forward
        for neuron_id in tqdm(neuron_ids, desc="Forward per neuron"):
            single_model = SingleCellModel(all_neurons_model, neuron_id)
            single_model.to(device)
            single_model.eval()

            resp_b.append(forward_in_batches(single_model, model_input_b, batch_size=32, device=device))
            resp_w.append(forward_in_batches(single_model, model_input_w, batch_size=32, device=device))

        # shape: (num_dots, n_neurons)
        resp_b = torch.stack(resp_b, dim=1)
        resp_w = torch.stack(resp_w, dim=1)

        # Response maps directly in stimulus-grid coordinates
        all_position_resp_b = resp_b.T.reshape(len(neuron_ids), grid_n, grid_n).to(device)
        all_position_resp_w = resp_w.T.reshape(len(neuron_ids), grid_n, grid_n).to(device)

        # One stored null map for visualization only:
        # single broken-association realization
        shuffle_resp_b = resp_b.cpu().numpy().copy()
        shuffle_resp_w = resp_w.cpu().numpy().copy()
        np.apply_along_axis(np.random.shuffle, axis=0, arr=shuffle_resp_b)
        np.apply_along_axis(np.random.shuffle, axis=0, arr=shuffle_resp_w)

        shuffle_resp_b = torch.tensor(shuffle_resp_b, dtype=torch.float32, device=device)
        shuffle_resp_w = torch.tensor(shuffle_resp_w, dtype=torch.float32, device=device)

        all_noise_b = shuffle_resp_b.T.reshape(len(neuron_ids), grid_n, grid_n)
        all_noise_w = shuffle_resp_w.T.reshape(len(neuron_ids), grid_n, grid_n)

    # Save results
    with h5py.File(h5_file, "a") as f:
        for j, neuron_id in enumerate(tqdm(neuron_ids, desc="Saving results")):
            neuron = f"neuron_{neuron_id}"

            if neuron not in f[subgroup_path_results]:
                pos_resp_imgs = torch.stack([all_position_resp_b[j], all_position_resp_w[j]]).cpu()
                noises = torch.stack([all_noise_b[j], all_noise_w[j]]).cpu()

                true_b = pos_resp_imgs[0].numpy()
                true_w = pos_resp_imgs[1].numpy()

                rng = np.random.default_rng(seed + int(neuron_id))

                # Signal = spatial variance of the true map
                signal_b = _map_variance(true_b, apply_smoothing=apply_smoothing, sigma=sigma)
                signal_w = _map_variance(true_w, apply_smoothing=apply_smoothing, sigma=sigma)

                # Null = expected variance under broken location-response association
                noise_vars_b = []
                noise_vars_w = []

                flat_b = true_b.ravel()
                flat_w = true_w.ravel()

                for _ in range(n_shuffles):
                    shuf_b = rng.permutation(flat_b).reshape(true_b.shape)
                    shuf_w = rng.permutation(flat_w).reshape(true_w.shape)

                    noise_vars_b.append(
                        _map_variance(shuf_b, apply_smoothing=apply_smoothing, sigma=sigma)
                    )
                    noise_vars_w.append(
                        _map_variance(shuf_w, apply_smoothing=apply_smoothing, sigma=sigma)
                    )

                noise_b = float(np.mean(noise_vars_b))
                noise_w = float(np.mean(noise_vars_w))

                eps = 1e-12
                SNR_b = signal_b / max(noise_b, eps)
                SNR_w = signal_w / max(noise_w, eps)
                logSNRwb = np.log10(max(SNR_w, eps) / max(SNR_b, eps))

                f[subgroup_path_resp_img].create_dataset(name=neuron, data=pos_resp_imgs)
                f[subgroup_path_noise_img].create_dataset(name=neuron, data=noises)
                f[subgroup_path_results].create_dataset(name=neuron, data=[SNR_b, SNR_w, logSNRwb])

                
def black_white_preference_experiment_paper_energy_version(
    h5_file,
    all_neurons_model,
    neuron_ids,
    *,
    overwrite=False,
    contrast=1,
    img_res=[93, 93],
    pixel_min=-1.7876,
    pixel_max=2.1919,
    device=None,
    seed=42,
):
    """
    Paper-aligned sparse black/white preference experiment for a static model.

    Geometry:
    - shared image-centered 12x12 grid
    - non-overlapping 7x7 px squares
    - responses stored directly in 12x12 stimulus-grid coordinates

    Signal:
    - raw whole-map spatial energy of the black and white response maps
    - energy = mean squared response across all stimulus positions

    Index:
    - log10(E_white / E_black)

    Notes:
    - Yeh et al. used spatial energy/variance of time-resolved reverse-correlation maps.
    - Here responses are static scalar responses per stimulus position, so we use the
      underlying spatial-energy quantity directly.
    - No temporal SNR denominator is used because no zero-lag temporal baseline exists.
    """

    np.random.seed(seed)

    print(" > Black or white preference experiment (paper-aligned spatial energy version)")

    group_path = "/black_white_preference"
    subgroup_path_resp_img = group_path + "/position_response_img"
    subgroup_path_results = group_path + "/results"

    if overwrite:
        clear_group(h5_file, group_path)

    # ---------------------------------------------------------
    # Shared centered geometry:
    # 12x12 grid, 7x7 px squares => 84x84 sampled region
    # ---------------------------------------------------------
    H, W = int(img_res[0]), int(img_res[1])
    grid_n = 12
    dot_size_in_pixels = 7
    sampled_px = grid_n * dot_size_in_pixels  # 84

    if sampled_px > H or sampled_px > W:
        raise ValueError(
            f"Grid does not fit into img_res={img_res}: sampled_px={sampled_px}"
        )

    y_start = (H - sampled_px) // 2
    x_start = (W - sampled_px) // 2

    deg_per_px = 2.67 / W
    sampled_deg = sampled_px * deg_per_px
    dot_deg = dot_size_in_pixels * deg_per_px

    print(f"   sampled region:  {sampled_px}px = {sampled_deg:.4f} deg")
    print(f"   square size:     {dot_size_in_pixels}px = {dot_deg:.4f} deg")
    print(f"   grid start:      y={y_start}, x={x_start}")
    print("   metric:          raw whole-map spatial energy")
    print("   index:           log10(E_white / E_black)")

    args_str = (
        f"grid_n={grid_n}/dot_size_in_pixels={dot_size_in_pixels}/"
        f"sampled_px={sampled_px}/sampled_deg={sampled_deg:.6f}/"
        f"contrast={contrast}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}/"
        f"metric=raw_spatial_energy/index=log10_Ewhite_over_Eblack"
    )

    group_init(h5_file=h5_file, group_path=group_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_resp_img, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_results, group_args_str=args_str)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    all_neurons_model.to(device)
    all_neurons_model.eval()

    black_stimuli = []
    white_stimuli = []
    resp_b = []
    resp_w = []

    def forward_in_batches(single_model, model_input, batch_size=32, device="cuda"):
        outputs = []
        with torch.no_grad():
            for start in range(0, model_input.shape[0], batch_size):
                end = min(start + batch_size, model_input.shape[0])
                batch = model_input[start:end].to(device)
                out = single_model(batch).detach().cpu().reshape(-1)
                outputs.append(out)
                del batch, out
        return torch.cat(outputs, dim=0)

    def spatial_energy(response_map):
        """
        Raw whole-map spatial energy.

        Uses mean squared response, not sum, so the value is independent
        of the number of sampled positions.
        """
        response_map = np.asarray(response_map, dtype=np.float64)
        return float(np.mean(response_map ** 2))

    with torch.no_grad():
        # Generate one black and one white square per grid position
        for gy in range(grid_n):
            for gx in range(grid_n):
                y = y_start + gy * dot_size_in_pixels
                x = x_start + gx * dot_size_in_pixels

                black_stim = np.zeros((H, W), dtype=np.float32)
                white_stim = np.zeros((H, W), dtype=np.float32)

                black_stim[
                    y:y + dot_size_in_pixels,
                    x:x + dot_size_in_pixels
                ] = -1.0 * contrast

                white_stim[
                    y:y + dot_size_in_pixels,
                    x:x + dot_size_in_pixels
                ] = 1.0 * contrast

                black_stimuli.append(torch.tensor(black_stim, dtype=torch.float32))
                white_stimuli.append(torch.tensor(white_stim, dtype=torch.float32))

        black_stimuli = torch.stack(black_stimuli)  # (144, H, W)
        white_stimuli = torch.stack(white_stimuli)  # (144, H, W)
        num_dots = len(black_stimuli)

        # Save sanity panel from exact generated stimuli
        sample_grid_positions = [
            (0, 0),
            (0, grid_n - 1),
            (grid_n // 2, grid_n // 2),
            (grid_n - 1, grid_n - 1),
        ]
        sample_indices = [gy * grid_n + gx for (gy, gx) in sample_grid_positions]
        sample_positions = [
            (y_start + gy * dot_size_in_pixels, x_start + gx * dot_size_in_pixels)
            for (gy, gx) in sample_grid_positions
        ]

        save_generated_stimuli_panel_png(
            black_stimuli=black_stimuli,
            white_stimuli=white_stimuli,
            sample_indices=sample_indices,
            sample_positions=sample_positions,
            save_path="/project/results/black_and_white_experiment/paper_energy/generated_stimuli_panel.png",
        )

        # Rescale once, keep on CPU, batch to GPU later
        model_input_b = (
            rescale(black_stimuli, -1, 1, pixel_min, pixel_max)
            .reshape(num_dots, 1, H, W)
            .cpu()
        )

        model_input_w = (
            rescale(white_stimuli, -1, 1, pixel_min, pixel_max)
            .reshape(num_dots, 1, H, W)
            .cpu()
        )

        # Per-neuron forward
        for neuron_id in tqdm(neuron_ids, desc="Forward per neuron"):
            single_model = SingleCellModel(all_neurons_model, neuron_id)
            single_model.to(device)
            single_model.eval()

            resp_b.append(
                forward_in_batches(
                    single_model,
                    model_input_b,
                    batch_size=32,
                    device=device,
                )
            )

            resp_w.append(
                forward_in_batches(
                    single_model,
                    model_input_w,
                    batch_size=32,
                    device=device,
                )
            )

        # shape: (num_dots, n_neurons)
        resp_b = torch.stack(resp_b, dim=1)
        resp_w = torch.stack(resp_w, dim=1)

        # Response maps directly in stimulus-grid coordinates
        all_position_resp_b = resp_b.T.reshape(len(neuron_ids), grid_n, grid_n)
        all_position_resp_w = resp_w.T.reshape(len(neuron_ids), grid_n, grid_n)

    # Save results
    with h5py.File(h5_file, "a") as f:
        for j, neuron_id in enumerate(tqdm(neuron_ids, desc="Saving results")):
            neuron = f"neuron_{neuron_id}"

            if neuron in f[subgroup_path_results]:
                continue

            pos_resp_imgs = torch.stack(
                [
                    all_position_resp_b[j],
                    all_position_resp_w[j],
                ]
            ).cpu()

            true_b = pos_resp_imgs[0].numpy()
            true_w = pos_resp_imgs[1].numpy()

            # Raw whole-map spatial energy
            energy_b = spatial_energy(true_b)
            energy_w = spatial_energy(true_w)

            eps = 1e-12
            logEwb = float(
                np.log10(max(energy_w, eps) / max(energy_b, eps))
            )

            # Store response maps:
            # [0] = black response map
            # [1] = white response map
            f[subgroup_path_resp_img].create_dataset(
                name=neuron,
                data=pos_resp_imgs.numpy(),
            )

            # Store:
            # [0] = E_black
            # [1] = E_white
            # [2] = log10(E_white / E_black)
            f[subgroup_path_results].create_dataset(
                name=neuron,
                data=np.array([energy_b, energy_w, logEwb], dtype=np.float64),
            )


