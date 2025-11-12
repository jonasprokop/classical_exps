############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
from tqdm import tqdm
import os
## Utils
from classical_exps.functions.utils import *

## Image generation
import imagen
from imagen.image import BoundingBox
import matplotlib.image as mpimg
## Import models 
from classical_exps.functions.utils import SingleCellModel
## Data storage
import h5py
from scipy.optimize import curve_fit
import time
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from classical_exps.functions.experiments.b_1st_article import get_size_tuning_curves, get_GSF_surround_AMRF


###############################################################################
#####  PART VII :   #####
#####   ---------------------------------------------------------------   #####
#####          #####
#####                                     #####
#####   ---------------------------------------------------------------   #####
#####      DOI :           #####
###############################################################################


def get_surround_contrast_facilitation(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite=False,
    img_res=[93, 93],
    pixel_min=-1.7876,
    pixel_max=2.1919,
    device=None,
    size=2.67,
    sigma=2.0,
    contrasts = np.arange(0.02, 1.02, 0.02),
    radii_list_annular_tunning_curve= 40,
    radii_list_contrast_tunning_curve = np.logspace(-2,np.log10(2.67),40, endpoint=True),
    save_minimal_receptive_fields=True,
    save_contrast_values=True,
):
    '''
    '''
    print(' > Get preferred orientation contrast stimulation')

    # Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"

    # Groups to fill
    group_facilitation = "/surround_contrast_group_facilitation" 

    subgroup_facilitation_st_curves_size_tunning_path  = group_facilitation + "/curves/size_tunning"
    subgroup_facilitation_st_results_size_tunning_path = group_facilitation + "/results/size_tunning"

    subgroup_facilitation_st_results_high_path = group_facilitation + "/results/high"
    subgroup_facilitation_st_results_low_path  = group_facilitation + "/results/low"
    subgroup_facilitation_st_curves_high_path  = group_facilitation + "/curves/high"
    subgroup_facilitation_st_curves_low_path  = group_facilitation +"/curves/low"

    subgroup_facilitation_st_curves_HH_path  = group_facilitation +"/curves/HH"
    subgroup_facilitation_st_curves_LH_path  = group_facilitation +"/curves/LH"
    subgroup_facilitation_st_curves_LL_path  = group_facilitation +"/curves/LL"

    subgroup_facilitation_st_results_HH_path  = group_facilitation +"/results/HH"
    subgroup_facilitation_st_results_LH_path  = group_facilitation +"/results/LH"
    subgroup_facilitation_st_results_LL_path  = group_facilitation +"/results/LL"

    minimal_receptive_fields_path = group_facilitation + "/minimal_receptive_fields"

    contrast_values_path = group_facilitation + "/contrast_values"
    
    ## Clear the group if requested
    #if overwrite : 
    clear_group(h5_file,group_facilitation)

    args_str = f"sigma={sigma}/contrasts={contrasts}/radii={radii_list_contrast_tunning_curve}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}"
    
    group_init(h5_file=h5_file, group_path=group_facilitation, group_args_str=args_str)

    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_curves_size_tunning_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_results_size_tunning_path, group_args_str=args_str)

    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_results_high_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_results_low_path, group_args_str=args_str)

    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_curves_high_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_curves_low_path, group_args_str=args_str)

    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_curves_HH_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_curves_LH_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_curves_LL_path, group_args_str=args_str)

    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_results_HH_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_results_LH_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_facilitation_st_results_LL_path, group_args_str=args_str)

    group_init(h5_file=h5_file, group_path=minimal_receptive_fields_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=contrast_values_path, group_args_str=args_str)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ratios = {}
    ratios_size_tunning = {}
    ratios_GSF = {}
    bad_fits = {}
    r2_dict = {}

    with h5py.File(h5_file, "a") as file:

        ## Access the groups 
        subgroup_facilitation_st_curves_size_tunning = file[subgroup_facilitation_st_curves_size_tunning_path]
        subgroup_facilitation_st_results_size_tunning = file[subgroup_facilitation_st_results_size_tunning_path]

        subgroup_facilitation_st_results_high = file[subgroup_facilitation_st_results_high_path]
        subgroup_facilitation_st_results_low   = file[subgroup_facilitation_st_results_low_path]

        subgroup_facilitation_st_curves_high = file[subgroup_facilitation_st_curves_high_path]
        subgroup_facilitation_st_curves_low = file[subgroup_facilitation_st_curves_low_path]

        subgroup_facilitation_st_curves_HH = file[subgroup_facilitation_st_curves_HH_path]
        subgroup_facilitation_st_curves_LH = file[subgroup_facilitation_st_curves_LH_path]
        subgroup_facilitation_st_curves_LL = file[subgroup_facilitation_st_curves_LL_path]

        subgroup_facilitation_st_results_HH = file[subgroup_facilitation_st_results_HH_path]
        subgroup_facilitation_st_results_LH = file[subgroup_facilitation_st_results_LH_path]
        subgroup_facilitation_st_results_LL = file[subgroup_facilitation_st_results_LL_path]

        # Save the receptive fields
        if save_minimal_receptive_fields:
            minimal_receptive_fields = file[minimal_receptive_fields_path]

        if save_contrast_values:
            contrast_values = file[contrast_values_path]

        neuron_data = []
        nxs = []
        nys = []

        nxs_dict = {}
        nys_dict = {}
            

        for neuron_id in tqdm(neuron_ids, desc="Applying Orientation Contrasts"):

            neuron = f"neuron_{neuron_id}"

            # Get the single_model of the neuron
            single_model = SingleCellModel(all_neurons_model, neuron_id)
            single_model.to(device)
            single_model.eval()

            ## Create Gray stimulus to substract to the non gray stimuli
            gray_stim = torch.ones((1, *img_res)).to(device) * ((pixel_min + pixel_max)/2)
            gray_resp = single_model(gray_stim)

            try:

                # Get preffered params
                preferred_ori = file[group_path_ff_params][neuron][:][0]
                preferred_sf  = file[group_path_ff_params][neuron][:][1]
                preferred_phase = file[group_path_ff_params][neuron][:][2]
                max_response = file[group_path_ff_params][neuron][:][3]

            except:
                print(f"Loading values for preffered params for {neuron} has failed")
                continue

            try:
                # get center coordinates
                x_pix         = file[group_path_pos][neuron][:][0]
                y_pix         = file[group_path_pos][neuron][:][1]

                # get the sigmas for coordinates
                sigma_x_opt   = file[group_path_pos][neuron][:][2]
                sigma_y_opt   = file[group_path_pos][neuron][:][3]
                rho_opt = file[group_path_pos][neuron][:][4]
                mse = file[group_path_pos][neuron][:][5] 
                r2 = file[group_path_pos][neuron][:][6]

                print(x_pix, y_pix, sigma_x_opt, rho_opt, sigma_y_opt, r2)

                r2_dict[neuron] = r2

                if r2 < 0.2:
                    print(f"Fitting for {neuron} might be bad with r2 = {r2}")
                    bad_fits[neuron] = r2
                    continue

                print(f"Center params for {neuron} : x_pix = {x_pix}, y_pix = {y_pix}, sigma_x_opt = {sigma_x_opt}, sigma_y_opt = {sigma_y_opt}")
            except:
                print(f"Loading values for center params for {neuron} has failed")
                continue

            try:
                percentages_of_gaussian_energies_2D = {
                    0.50: 0.1175030974,
                    0.75: 0.2431167344,
                    1.00: 0.3934693403,
                    1.25: 0.5220457768,
                    1.50: 0.6766764162,
                    1.75: 0.7843774931,
                    2.00: 0.8646647168,
                    2.25: 0.9231163464,
                    2.50: 0.9569971130,
                    2.75: 0.9780316240,
                    3.00: 0.9888910035
                }

                deg_per_pix = size / img_res[0]  
                nx = sigma_x_opt * deg_per_pix * preferred_sf
                ny = sigma_y_opt * deg_per_pix * preferred_sf
                nxs.append(nx)
                nys.append(ny)

                nxs_dict[neuron] = nx   
                nys_dict[neuron] = ny   



                continue


                percentage_of_guassian_energy = percentages_of_gaussian_energies_2D[sigma]

                mRF_radius = np.sqrt(-2 * np.log(1 - percentage_of_guassian_energy)) * np.sqrt(sigma_x_opt * sigma_y_opt)

                deg_per_pix = size / img_res[0]  
                mRF_radius = mRF_radius * deg_per_pix

                print(f"mRF_radius for {neuron} : {mRF_radius}")

                if np.isnan(mRF_radius):
                    print(f"For {neuron} the mRF_radius was not calculated")
                    continue
            except:
                    print(f"For {neuron} the mRF_radius was not calculated")
                    continue
            
            try:
                sf_pref = preferred_sf  

                expected_rf = 1.0 / (2 * sf_pref) 

                ratio = mRF_radius / expected_rf

                print(f"{neuron}: ratio = {ratio:.3f}")
                ratios[neuron] = ratio

            except Exception as e:
                print(f"For {neuron}, comparison failed: {e}")
                continue
                    

            if save_minimal_receptive_fields:
                minimal_receptive_fields.create_dataset(name=neuron, data=[mRF_radius])
            
            plot = True
            if plot:
                grating = torch.Tensor(imagen.SineGrating(
                    orientation=preferred_ori,
                    frequency=preferred_sf,
                    phase=preferred_phase,
                    bounds=BoundingBox(points=((-size/2, -size/2), (size/2, size/2))),
                    offset=0,
                    scale=1,
                    xdensity=img_res[1]/size,
                    ydensity=img_res[0]/size,
                )())
                grating_np = grating.squeeze().detach().cpu().numpy()
                carrier_min_val, carrier_max_val = grating_np.min(), grating_np.max()

                directory = f"/project/results/facilitation/mrf_visualization/{neuron}/"
                os.makedirs(directory, exist_ok=True)

                deg_per_pix = size / img_res[0]
                x_deg = (x_pix - img_res[0]/2) * deg_per_pix
                y_deg = (y_pix - img_res[1]/2) * deg_per_pix
                sigma_x_deg = sigma_x_opt * deg_per_pix
                sigma_y_deg = sigma_y_opt * deg_per_pix

                # --- Precompute stuff ---
                # Gaussian coordinates
                x_coords = np.linspace(-size/2, size/2, img_res[1])
                y_coords = np.linspace(-size/2, size/2, img_res[0])
                X, Y = np.meshgrid(x_coords, y_coords)

                Z_gauss = np.exp(-1 / (2 * (1 - rho_opt**2)) * (
                    ((X - x_deg)**2) / sigma_x_deg**2 +
                    ((Y - y_deg)**2) / sigma_y_deg**2 -
                    (2 * rho_opt * (X - x_deg) * (Y - y_deg)) / (sigma_x_deg * sigma_y_deg)
                ))

                # Ellipse parameters in degrees
                cov = np.array([[sigma_x_deg**2, rho_opt*sigma_x_deg*sigma_y_deg],
                                [rho_opt*sigma_x_deg*sigma_y_deg, sigma_y_deg**2]])
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = eigvals.argsort()[::-1]
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]
                width, height = 2 * 2 * np.sqrt(eigvals)  # 2σ ellipse
                theta_deg = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

                # Circle coordinates
                theta_vals = np.linspace(0, 2*np.pi, 400)
                circle_x = x_deg + mRF_radius * np.cos(theta_vals)
                circle_y = y_deg + mRF_radius * np.sin(theta_vals)

                # --- Create figure ---
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))

                # 1. Contours
                im0 = axs[0].imshow(grating_np, cmap='gray', vmin=carrier_min_val, vmax=carrier_max_val,
                                    origin="lower", extent=[-size/2, size/2, -size/2, size/2])
                levels = np.exp(-0.5 * np.array([3, 2, 1])**2)
                axs[0].contour(X, Y, Z_gauss, levels=levels, colors="#FF00AA", linewidths=1.5, alpha=0.8)
                axs[0].set_aspect("equal")
                axs[0].set_title(f"Gaussian contours (1σ, 2σ, 3σ)")
                fig.colorbar(im0, ax=axs[0], shrink=0.7, label="Grating intensity")

                                # 2. Ellipse
                # --- Draw 2σ contour as ellipse approximation ---
                levels = np.exp(-0.5 * np.array([3, 2, 1])**2)

                # Same as before for grating
                im1 = axs[1].imshow(
                    grating_np, cmap='gray',
                    vmin=carrier_min_val, vmax=carrier_max_val,
                    origin="lower", extent=[-size/2, size/2, -size/2, size/2]
                )

                # Get contours (returns list of contour sets)
                contours = axs[1].contour(
                    X, Y, Z_gauss,
                    levels=[np.exp(-0.5 * 2**2)],  # the 2σ contour only
                    colors='#FF00AA', linewidths=2.0, alpha=0.9
                )

                axs[1].set_aspect("equal")
                axs[1].set_title(f"2σ elliptical fit")

                fig.colorbar(im1, ax=axs[1], shrink=0.7, label="Grating intensity")
                # 3. mRF circle
                im2 = axs[2].imshow(grating_np, cmap='gray', vmin=carrier_min_val, vmax=carrier_max_val,
                                    origin="lower", extent=[-size/2, size/2, -size/2, size/2])
                axs[2].plot(circle_x, circle_y, color="#FF00AA", lw=1.5)

                axs[2].set_xlim(-size/2, size/2)
                axs[2].set_ylim(-size/2, size/2)


                axs[2].set_aspect("equal")
                axs[2].legend(loc="upper right")
                axs[2].set_title(f"mRF, radius: {mRF_radius:.2f}°, srf ratio: {ratio:.2f}")
                fig.colorbar(im2, ax=axs[2], shrink=0.7, label="Grating intensity")

                plt.tight_layout()
                plt.savefig(os.path.join(directory, f"mRF_panel_{neuron_id}.png"), dpi=300, bbox_inches="tight")
                plt.close(fig)


            responses_size_tunning_curve = get_size_tunning_curve(
                        single_model, 
                        preferred_ori = preferred_ori, 
                        preferred_sf = preferred_sf, 
                        preferred_phase = preferred_phase,
                        img_res = img_res, 
                        pixel_min = pixel_min, #e.g:  (the lowest value encountered in the data we worked with)
                        pixel_max =  pixel_max, #e.g:  (the highest value encountered in the data we worked with)
                        device = device,
                        size = size,   
                        x_pix = x_pix,
                        y_pix = y_pix,
                        radii_list_size_tunning_curve= np.logspace(np.log10(1e-3),np.log10(2.67),40, endpoint=True),
                        neg_val=True,
                        neuron=neuron,
                        contrast=1,
                        )
        
            
            mrf_radius_size_tunning = get_mrf_from_size_tuning(np.logspace(np.log10(1e-3),np.log10(2.67),40, endpoint=True), responses_size_tunning_curve, method="rpeak")


            try:
                sf_pref = preferred_sf  

                expected_rf = 1.0 / (2 * sf_pref) 

                ratio_size_tunning = mrf_radius_size_tunning / expected_rf

                print(f"{neuron}: ratio = {ratio_size_tunning:.3f}")

                ratios_size_tunning[neuron] = ratio_size_tunning

            except Exception as e:
                continue

            subgroup_facilitation_st_curves_size_tunning.create_dataset(name=neuron, data=[np.logspace(np.log10(1e-3),np.log10(2.67),40, endpoint=True), responses_size_tunning_curve.cpu()])
            subgroup_facilitation_st_results_size_tunning.create_dataset(name=neuron, data=[mrf_radius_size_tunning, ratio_size_tunning, expected_rf])


            # plot_size_tunning_curve(radii=np.logspace(np.log10(1e-3),np.log10(2.67),80, endpoint=True),
            #         responses=responses_size_tunning_curve,
            #         mrf_radius_size_tunning=mrf_radius_size_tunning,
            #         ratio_size_tunning=ratio_size_tunning,
            #         expected_rf=expected_rf,
            #         neuron=neuron, 
            #         )

            print(f"mrf for {neuron} : {mrf_radius_size_tunning}") 

            neuron_data.append({
                "neuron": neuron,
                "x_pix": x_pix,
                "y_pix": y_pix,
                "sigma_x_opt": sigma_x_opt,
                "sigma_y_opt": sigma_y_opt,
                "r2": r2,
                "mase": mse,
                "preferred_ori": preferred_ori,
                "preferred_sf": preferred_sf,
                "preferred_phase": preferred_phase,
                "max_response": max_response,
                "mRF_radius_gauss": mRF_radius,
                "mRF_radius_tuning": mrf_radius_size_tunning,
                "ratio_gauss": ratios.get(neuron, np.nan),
                "ratio_size_tuning": ratios_size_tunning.get(neuron, np.nan),
                "expected_rf": expected_rf})
            
            print(f"Tuning data for {neuron} appended")

            mrf_radius = mRF_radius
            # mRF_radius = float(mrf_radius_size_tunning)

            # get low and high contrast values for the neuron
            low_contrast, high_contrast = get_low_and_high_contrast_contrast(
                                            single_model, 
                                            preferred_ori = preferred_ori, 
                                            preferred_sf = preferred_sf, 
                                            preferred_phase = preferred_phase,
                                            contrasts = contrasts,         
                                            img_res = img_res, 
                                            pixel_min = pixel_min, 
                                            pixel_max =  pixel_max,
                                            device = device,
                                            size = size,   
                                            x_pix = x_pix,
                                            y_pix = y_pix,  
                                            mRF_radius = mRF_radius, 
                                            neuron_name=neuron,   
                                        )

            if save_contrast_values:
                contrast_values.create_dataset(name=neuron, data=[low_contrast, high_contrast])

            # get patch-size tunning curve 
            low_contrast_circular_size_tunning_curve = get_size_tuning_curves(
                                                single_model,
                                                x_pix,
                                                y_pix,
                                                preferred_ori,
                                                preferred_sf, 
                                                preferred_phase,
                                                radii = radii_list_contrast_tunning_curve,
                                                contrast = low_contrast,
                                                pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
                                                pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
                                                device = None,
                                                size = 2.67,
                                                img_res = [93,93],
                                                neg_val = True,     #TODO never used
                                                compute_annular = False,
                                                neuron=neuron,
                                                contrast_keyword="low",
                                            )
            high_contrast_circular_size_tunning_curve = get_size_tuning_curves(
                                                single_model,
                                                x_pix,
                                                y_pix,
                                                preferred_ori,
                                                preferred_sf, 
                                                preferred_phase,
                                                radii = radii_list_contrast_tunning_curve,
                                                contrast = high_contrast,
                                                pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
                                                pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
                                                device = None,
                                                size = 2.67,
                                                img_res = [93,93],
                                                neg_val = True,     #TODO never used
                                                compute_annular = False,
                                                neuron=neuron,
                                                contrast_keyword="high",
                                            )
                    
            ## Save the tuning curve in the subgroup '/size_tuning/curves
            subgroup_facilitation_st_curves_low.create_dataset(name=neuron, data=[radii_list_contrast_tunning_curve, low_contrast_circular_size_tunning_curve.cpu()])
            subgroup_facilitation_st_curves_high.create_dataset(name=neuron, data=[radii_list_contrast_tunning_curve, high_contrast_circular_size_tunning_curve.cpu()])

            subgroup_facilitation_st_results_low.create_dataset(name=neuron, data=low_contrast)
            subgroup_facilitation_st_results_high.create_dataset(name=neuron, data=high_contrast)

            ## Perform some analysis
            GSF_low, surround_extent_low, SI_low, Ropt_low, _ = get_GSF_surround_AMRF(
                radii = radii_list_contrast_tunning_curve,
                circular_tuning_curve = low_contrast_circular_size_tunning_curve,
                annular_tuning_curve = None
                )
            
            GSF_high, surround_extent_high, SI_high, Ropt_high, _ = get_GSF_surround_AMRF(
                radii = radii_list_contrast_tunning_curve,
                circular_tuning_curve = high_contrast_circular_size_tunning_curve,
                annular_tuning_curve = None
                )
            
            print(f"For {neuron} : GSF_low = {GSF_low}, GSF_high = {GSF_high}, mrf_radius: {mrf_radius, mRF_radius}")            
            if GSF_low <= 1.e-9 or GSF_high <= 1.e-9 or GSF_high >= GSF_low:
                continue

            try:
                sf_pref = preferred_sf  

                expected_rf = 1.0 / (2 * sf_pref) 

                ratio = GSF_low / expected_rf

                print(f"{neuron}: ratio = {ratio:.3f}")
                ratios_GSF[neuron] = ratio

            except Exception as e:
                continue

            outer_annulus_radii_HH, size_tuning_surround_annulus_curve_HH = get_size_tuning_surround_annulus_curve(
                    single_model,
                    x_pix,
                    y_pix,
                    preferred_ori,
                    preferred_sf,
                    preferred_phase,
                    pixel_min = pixel_min,
                    pixel_max =  pixel_max,
                    device = device,
                    size = size,
                    img_res = img_res,
                    neg_val = True,     
                    GSF_low=GSF_low,
                    GSF_high=GSF_high,
                    contrast_center=high_contrast,
                    contrast_far_surround=high_contrast,
                    gray_resp=gray_resp,
                    radii_list_annular_tunning_curve=radii_list_annular_tunning_curve,
                    neuron=neuron,
                    curve_type='HH'
                    )
            
            outer_annulus_radii_LH, size_tuning_surround_annulus_curve_LH = get_size_tuning_surround_annulus_curve(
                    single_model,
                    x_pix,
                    y_pix,
                    preferred_ori,
                    preferred_sf,
                    preferred_phase,
                    pixel_min = pixel_min,
                    pixel_max =  pixel_max,
                    device = device,
                    size = size,
                    img_res = img_res,
                    neg_val = True,     
                    GSF_low=GSF_low,
                    GSF_high=GSF_high,
                    contrast_center=low_contrast,
                    contrast_far_surround=high_contrast,
                    gray_resp=gray_resp,
                    radii_list_annular_tunning_curve=radii_list_annular_tunning_curve,
                    neuron=neuron,
                    curve_type='LH',
                    )
            
            outer_annulus_radii_LL, size_tuning_surround_annulus_curve_LL = get_size_tuning_surround_annulus_curve(
                    single_model,
                    x_pix,
                    y_pix,
                    preferred_ori,
                    preferred_sf,
                    preferred_phase,
                    pixel_min = pixel_min,
                    pixel_max =  pixel_max,
                    device = device,
                    size = size,
                    img_res = img_res,
                    neg_val = True,     
                    GSF_low=GSF_low,
                    GSF_high=GSF_high,
                    contrast_center=low_contrast,
                    contrast_far_surround=low_contrast,
                    gray_resp=gray_resp,
                    radii_list_annular_tunning_curve=radii_list_annular_tunning_curve,
                    neuron=neuron,
                    curve_type='LL',
                    )

            def is_empty(x):
                if x is None:
                    return True
                if isinstance(x, torch.Tensor):
                    return x.numel() == 0
                if isinstance(x, (list, np.ndarray)):
                    return len(x) == 0
                return True
            
            if is_empty(size_tuning_surround_annulus_curve_HH) or \
                is_empty(size_tuning_surround_annulus_curve_LH) or \
                is_empty(size_tuning_surround_annulus_curve_LL):
                    print(f"For neuron {neuron}: size tuning surround annulus curves were not properly obtained.")
                    continue
                
            
            response_center_alone_high_contrast = get_center_alone_stimulus(
                    single_model=single_model,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    preferred_ori=preferred_ori,
                    preferred_sf=preferred_sf,
                    preferred_phase=preferred_phase,
                    size = size,
                    pixel_min = pixel_min, #e.g:  (the lowest value encountered in the data we worked with)
                    pixel_max =  pixel_max, #e.g:  (the highest value encountered in the data we worked with)
                    device = device,
                    img_res = img_res,
                    GSF_high=GSF_high,
                    contrast_center=high_contrast,
                    neuron=neuron,
                    gray_resp=gray_resp
                    )

            response_center_alone_low_contrast = get_center_alone_stimulus(
                    single_model=single_model,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    preferred_ori=preferred_ori,
                    preferred_sf=preferred_sf,
                    preferred_phase=preferred_phase,
                    size = size,
                    pixel_min = pixel_min, #e.g:  (the lowest value encountered in the data we worked with)
                    pixel_max =  pixel_max, #e.g:  (the highest value encountered in the data we worked with)
                    device = device,
                    img_res = img_res,
                    GSF_high=GSF_high,
                    contrast_center=low_contrast,
                    neuron=neuron,
                    gray_resp=gray_resp
                    )

            response_full_grating_high_contrast = get_full_grating_stimulus(
                    single_model=single_model,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    preferred_ori=preferred_ori,
                    preferred_sf=preferred_sf,
                    preferred_phase=preferred_phase,
                    size = size,
                    pixel_min = pixel_min, #e.g:  (the lowest value encountered in the data we worked with)
                    pixel_max =  pixel_max, #e.g:  (the highest value encountered in the data we worked with)
                    device = device,
                    img_res = img_res,
                    GSF_high=GSF_high,
                    contrast_center=high_contrast,
                    neuron=neuron,
                    gray_resp=gray_resp
                    )

            
            response_full_grating_low_contrast = get_full_grating_stimulus(
                    single_model=single_model,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    preferred_ori=preferred_ori,
                    preferred_sf=preferred_sf,
                    preferred_phase=preferred_phase,
                    size = size,
                    pixel_min = pixel_min, #e.g:  (the lowest value encountered in the data we worked with)
                    pixel_max =  pixel_max, #e.g:  (the highest value encountered in the data we worked with)
                    device = device,
                    img_res = img_res,
                    GSF_high=GSF_high,
                    contrast_center=low_contrast,
                    neuron=neuron,
                    gray_resp=gray_resp
                    )
            
            print(f"{neuron}: tunning curves and responses obtained")

             

            gray_resp = gray_resp.item()
            response_center_alone_high_contrast = response_center_alone_high_contrast.item()
            response_center_alone_low_contrast = response_center_alone_low_contrast.item()
            response_full_grating_high_contrast = response_full_grating_high_contrast.item()
            response_full_grating_low_contrast = response_full_grating_low_contrast.item()


            subgroup_facilitation_st_curves_HH.create_dataset(name=neuron, data=[outer_annulus_radii_HH, size_tuning_surround_annulus_curve_HH.cpu()])
            subgroup_facilitation_st_curves_LH.create_dataset(name=neuron, data=[outer_annulus_radii_LH, size_tuning_surround_annulus_curve_LH.cpu()])
            subgroup_facilitation_st_curves_LL.create_dataset(name=neuron, data=[outer_annulus_radii_LL, size_tuning_surround_annulus_curve_LL.cpu()])

            subgroup_facilitation_st_results_HH.create_dataset(name=neuron, data=[gray_resp, GSF_low, GSF_high, response_center_alone_high_contrast, response_full_grating_high_contrast])
            subgroup_facilitation_st_results_LH.create_dataset(name=neuron, data=[gray_resp, GSF_low, GSF_high,  response_center_alone_low_contrast, response_full_grating_low_contrast])
            subgroup_facilitation_st_results_LL.create_dataset(name=neuron, data=[gray_resp, GSF_low, GSF_high, response_center_alone_low_contrast, response_full_grating_low_contrast])
                                            
            print(f"for neuron {neuron} saving results was a success")     


    from statsmodels.nonparametric.smoothers_lowess import lowess

    # # convert lists to arrays
    # nxs_arr = np.array(nxs)
    # nys_arr = np.array(nys)

    # # figure setup
    # plt.figure(figsize=(6,6))
    # plt.scatter(nxs_arr, nys_arr, s=20, alpha=0.7, edgecolor='none', label='neurons')
    # plt.xlabel(r"$n_x = \sigma_x \cdot f$")
    # plt.ylabel(r"$n_y = \sigma_y \cdot f$")
    # plt.title("Population RF structure (σ·f)")

    # # axes limits
    # max_val = max(max(nxs_arr), max(nys_arr)) * 1.05  # a little padding
    # plt.xlim(0, max_val)
    # plt.ylim(0, max_val)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True, ls="--", alpha=0.4)

    # # compute LOWESS smoothing (Cleveland & Devlin 1988)
    # # frac controls the amount of smoothing (0.3 is typical, adjust if needed)
    # smoothed = lowess(nys_arr, nxs_arr, frac=0.3)
    # x_smooth = smoothed[:,0]
    # y_smooth = smoothed[:,1]

    # # plot the smoothed path
    # plt.plot(x_smooth, y_smooth, color='red', lw=2, label='LOWESS path')
    # plt.legend()
    # os.makedirs("/project/results/facilitation/neuron_summary/", exist_ok=True)
    # plt.savefig("/project/results/facilitation/neuron_summary/rf_structure_nx_ny.png", dpi=300, bbox_inches="tight")
    # plt.close()


    # thresholds
    threshold = 1.5

    # extract all neurons
    neurons = list(nxs_dict.keys())
    nxs_arr = np.array([nxs_dict[n] for n in neurons])
    nys_arr = np.array([nys_dict[n] for n in neurons])
    r2_arr  = np.array([r2_dict[n]  for n in neurons])

    # threshold for r2
    r2_threshold = 0.8

    # create mask for neurons that pass r2 threshold
    mask = r2_arr >= r2_threshold

    # filter arrays
    nxs_arr = nxs_arr[mask]
    nys_arr = nys_arr[mask]
    r2_arr  = r2_arr[mask]
    neurons = np.array(neurons)[mask]

    # identify highlighted neurons
    highlight_mask = (nxs_arr > threshold) & (nys_arr > threshold)
    highlight_nxs = nxs_arr[highlight_mask]
    highlight_nys = nys_arr[highlight_mask]

    # figure setup
    plt.figure(figsize=(6,6))
    plt.scatter(nxs_arr, nys_arr, s=20, alpha=0.7, edgecolor='none', label='neurons')
    plt.scatter(highlight_nxs, highlight_nys, color='orange', s=50, label='nx & ny > 1.5')
    plt.xlabel(r"$n_x = \sigma_x \cdot f$")
    plt.ylabel(r"$n_y = \sigma_y \cdot f$")
    plt.title("Population RF structure (σ·f)")

    # axes limits
    max_val = max(max(nxs_arr), max(nys_arr)) * 1.05
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, ls="--", alpha=0.4)

    # LOWESS smoothing on all neurons
    smoothed = lowess(nys_arr, nxs_arr, frac=0.3)
    x_smooth = smoothed[:,0]
    y_smooth = smoothed[:,1]
    plt.plot(x_smooth, y_smooth, color='red', lw=2, label='LOWESS path')

    for n, x, y, r2 in zip(neurons, nxs_arr, nys_arr, r2_arr):
        if x > threshold and y > threshold:
            plt.text(x, y, f"{r2:.2f}", color='blue', fontsize=8,
                    ha='left', va='bottom') 

    os.makedirs("/project/results/facilitation/neuron_summary/", exist_ok=True)
    plt.savefig("/project/results/facilitation/neuron_summary/rf_structure_nx_ny_filtered.png", dpi=300, bbox_inches="tight")
    plt.close()



    import pandas as pd
    # --- Prepare histogram ---
    r2_values = list(r2_dict.values())
    bins = np.concatenate(([-0.05, 0], np.arange(0, 1.05, 0.05)))

    # --- Prepare scatter ---
    data = {"neuron": list(r2_dict.keys())}
    df = pd.DataFrame(data)
    df["r2"] = df["neuron"].map(r2_dict)
    df["ratio"] = df["neuron"].map(ratios)

    # --- Create panel ---
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # 1) Histogram
    # Histogram in percentages
    n_neurons = len(r2_values)
    weights = np.ones_like(r2_values) / n_neurons * 100  # each neuron counts as its percentage

    axs[0].hist(r2_values, bins=bins, color="#0072B2", alpha=0.7, edgecolor="black", label="Percentage of neurons (%)", weights=weights)
    axs[0].axvline(0.2, color="red", linestyle="--", linewidth=1.5, label="threshold r2 = 0.2")
    axs[0].axvline(0.8, color="green", linestyle="--", linewidth=1.5, label="threshold r2 = 0.8")

    tick_positions = np.concatenate([[-0.025], bins[2:-1]])
    tick_labels = ["<0"] + [f"{b:.2f}" for b in bins[2:-1]]

    axs[0].set_xticks(tick_positions)
    axs[0].set_xticklabels(tick_labels, rotation=45)


    axs[0].set_xlabel("r2")
    axs[0].set_ylabel("Percentage of neurons (%)")
    axs[0].set_title("Distribution of fitted r2 values")
    axs[0].legend()

    # 2) Scatter
    axs[1].scatter(df["r2"], df["ratio"], color="#0072B2", alpha=0.7, edgecolors="k")
    axs[1].axvline(0.2, color="red", linestyle="--", linewidth=1.5, label="threshold r2 = 0.2")
    axs[1].axvline(0.8, color="green", linestyle="--", linewidth=1.5, label="threshold r2 = 0.8")
    axs[1].set_ylim(0, 10)

    axs[1].set_xlabel("r2")
    axs[1].set_ylabel("mRF / expected RF ratio")
    axs[1].set_title("Scatter plot of fitted r2 and neurons mRF/expected RF ratio")
    axs[1].legend()

    plt.tight_layout()

    # --- Save ---
    save_dir = "/project/results/facilitation/neuron_summary/"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "neuron_r2_panel.png"), dpi=500, bbox_inches="tight")
    plt.close()

    
    print("\nNeurons with potentially bad fits (r2 < 0.8):")
    for neuron, r2 in bad_fits.items():
        print(f"{neuron}: r2 = {r2}")
    print("Compiling neuron summary table...")

    df = pd.DataFrame(neuron_data)

    save_dir = save_dir="/project/results/facilitation/neuron_summary/"
    excel_path = save_dir + "neuron_summary.xlsx"
    os.makedirs(save_dir, exist_ok=True)


    # round numerical columns a bit for readability
    num_cols = df.select_dtypes(include=[float, int]).columns
    df[num_cols] = df[num_cols].round(4)

    # save to Excel
    df.to_excel(excel_path, index=False)

    print(df.head(n=10))

    print(f"\nNeuron summary table saved at: {excel_path}")

    plot_ratio_histogram(ratios, sigma = sigma, percentage_of_guassian_energy = percentage_of_guassian_energy, save_dir="/project/results/facilitation/srf_check_gauss/")

    plot_ratio_histogram(ratios_size_tunning, save_dir="/project/results/facilitation/srf_check_size_tunning/")

    plot_ratio_histogram(ratios_GSF, save_dir="/project/results/facilitation/srf_check_gsf/")


def plot_size_tunning_curve(radii, responses, mrf_radius_size_tunning, ratio_size_tunning, expected_rf, neuron):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.scatter(radii, responses, s=25, color='b')

    ax.axvline(x=mrf_radius_size_tunning, color='red', linestyle='--',
               label=f"peak radius (mRF radius)= {mrf_radius_size_tunning:.3f}°")

    ax.text(
        0.95, 0.05,  
        f"Expected mRF from SF = {expected_rf:.3f}",
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,
        fontsize=9,
        color='gray'
    )

    ax.text(
        0.90, 0.05,  
        f"Ratio (mRF / (1/2SF)) = {ratio_size_tunning:.3f}",
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,
        fontsize=9,
        color='gray'
    )

    y_max = max(responses)
    y_range = max(responses) - min(responses)
    ax.set_ylim(min(responses) - 0.05*y_range, y_max + 0.2*y_range) 

    ax.set_xlabel("Stimulus diameter (°)")
    ax.set_ylabel("Neuronal response")
    ax.set_title(f"Size Tuning Curve – {neuron}")
    ax.legend()
    ax.grid(True)

    # save figure
    directory = f"/project/results/facilitation/MRF_size_tunning_curves/{neuron}/"
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"MRF_size_tunning_curve_{neuron}.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
            

def detect_outliers_mad(x, thresh=3.5):
    x = np.asarray(x)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0:
        q1, q3 = np.nanpercentile(x, [25,75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        mask_out = (x < lower) | (x > upper)
    else:
        modified_z = 0.6745 * (x - med) / mad   
        mask_out = np.abs(modified_z) > thresh
    return mask_out


def plot_ratio_histogram(ratios, sigma=None, percentage_of_guassian_energy=None, save_dir="/project/results/facilitation/srf_check/"):
    """
    Plot histogram of RF/(1/2SF) ratios with MAD-based outlier reporting.
    Displays MAD bounds and robust mean.
    
    MAD (Median Absolute Deviation) is a robust statistic:
        MAD = median(|x_i - median(x)|)
    Outliers are defined as values outside median ± 3.5*MAD.
    """
    vals = np.array(list(ratios.values()))

    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        print("No valid values to plot.")
        return
    total_n = len(vals)

    # Compute median and MAD
    median_val = np.median(vals)
    mad_val = np.median(np.abs(vals - median_val))
    
    # Robust bounds: median ± 3.5*MAD
    lower_bound = median_val - 3.5 * mad_val
    upper_bound = median_val + 3.5 * mad_val
    
    # Mask for values inside bounds
    mask_in = (vals >= lower_bound) & (vals <= upper_bound)
    vals_in = vals[mask_in]
    vals_out = vals[~mask_in]

    # Robust statistics
    mean_all = vals.mean()
    robust_mean = vals_in.mean() if len(vals_in) > 0 else np.nan
    out_n = len(vals_out)
    out_pct = 100 * out_n / total_n

    bin_width = 0.25
    max_bin = 10

    # bins: go from 0 → 10.25 (so last bin is 10–10.25)
    bins = np.arange(0, max_bin + bin_width*2, bin_width)

    # clip values above 10 into last bin
    vals_clipped = np.copy(vals)
    vals_clipped[vals_clipped > max_bin] = max_bin + bin_width - 1e-6

    weights = np.ones_like(vals_clipped) * 100.0 / len(vals_clipped)


    plt.figure(figsize=(14, 8))
    plt.hist(vals_clipped, bins=bins, weights=weights,
         edgecolor="k", alpha=0.7)

    # xticks: keep them as usual (0, 0.5, …, 10.0)
    xticks = np.arange(0, max_bin + 0.5, 0.5)
    xtick_labels = [f"{t:.1f}" for t in xticks]
    xtick_labels[-1] = f">{max_bin}"   # replace "10.0" with ">10"

    plt.xticks(xticks, xtick_labels, rotation=45)

    plt.ylim(0, 38)
    plt.yticks(np.arange(0, 36, 5))

    lower_bound = max(lower_bound, 0)

    # vertical lines
    plt.axvline(mean_all, color="red", linestyle="--", lw=2, label=f"Mean all ({mean_all:.3f})")
    plt.axvline(robust_mean, color="blue", linestyle="-.", lw=2, label=f"MAD mean ({robust_mean:.3f})")
    plt.axvline(lower_bound, color="green", linestyle="--", lw=2, label="MAD bounds")
    plt.axvline(upper_bound, color="green", linestyle="--", lw=2)
    

    # Expected ratio ranges
    strict_lo, strict_hi = 1.0, 2.0
    loose_lo, loose_hi   = 0.5, 2.0

    in_strict = (vals >= strict_lo) & (vals <= strict_hi)
    in_loose  = (vals >= loose_lo) & (vals <= loose_hi)

    n_strict = in_strict.sum()
    pct_strict = 100 * n_strict / total_n

    n_loose = in_loose.sum()
    pct_loose = 100 * n_loose / total_n

    plt.xlabel("mRF_radius / expected_RF")
    plt.ylabel("Count (%)")
    if sigma and percentage_of_guassian_energy:
        plt.title(f"Distribution of RF/(1/2SF) Ratios - {sigma:.1f} σ, {percentage_of_guassian_energy*100:.1f}% Gaussian Energy")
    else:
        plt.title(f"Distribution of RF/(1/2SF) Ratios - Gaussian Energy")


    stats_text = (
        f"Total N = {total_n}\n"
        f"Mean population = {mean_all:.3f}\n"
        f"In-range (1–2): {n_strict} ({pct_strict:.1f}%)\n"
        f"In-range (0.5–2): {n_loose} ({pct_loose:.1f}%)\n\n"
        "MAD = median(|x_i - median(x)|)\n"
        "Outliers = median ± 3.5×MAD\n"
        f"MAD mean = {robust_mean:.3f}\n"
        f"Outliers: {out_n} ({out_pct:.1f}%)\n"
        f"MAD lower = {lower_bound:.3f}\n"
        f"MAD upper = {upper_bound:.3f}\n"
    )

    plt.gca().text(
        0.98, 0.65, stats_text, transform=plt.gca().transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    plt.legend(loc="upper right")

    os.makedirs(save_dir, exist_ok=True)
    plot_name = os.path.join(save_dir, "srf_ratio_histogram_mad.png")
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()


def get_mrf_from_size_tuning(radii_list, size_tuning_curve, method="rpeak"):
    radii = np.array(radii_list)
    responses = np.array(size_tuning_curve)

    if len(radii) != len(responses):
        raise ValueError("radii_list and size_tuning_curve must have same length")

    # Peak response and index
    peak_idx = np.argmax(responses)
    peak_val = responses[peak_idx]
    r_peak = radii[peak_idx]

    if method == "rpeak":
        return r_peak

    # Rising curve up to peak
    r_rise = radii[:peak_idx+1]
    y_rise = responses[:peak_idx+1]

    # Target fraction of peak
    if method == "r95":
        target = 0.95 * peak_val
    elif method == "r50":
        target = 0.50 * peak_val

    else:
        raise ValueError("method must be 'r95', 'r50', or 'rpeak'")

    # Find first index crossing the target
    above = np.where(y_rise >= target)[0]
    if len(above) == 0:
        return np.nan
    j = above[0]
    if j == 0:
        return r_rise[0]
    
    # Linear interpolate between points j-1 and j
    x0, x1 = r_rise[j-1], r_rise[j]
    y0, y1 = y_rise[j-1], y_rise[j]
    if y1 == y0:
        return x1
    t = (target - y0) / (y1 - y0)
    return x0 + t * (x1 - x0)

def get_size_tunning_curve(
    single_model, 
    preferred_ori = 0, 
    preferred_sf = 0, 
    preferred_phase = 0,
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,   
    x_pix = 93/2,
    y_pix = 93/2,
    radii_list_size_tunning_curve= np.logspace(-2,np.log10(2),40) ,
    neg_val=False,
    neuron=1,
    contrast=1,
    ):

    gray_stim = torch.ones((1, *img_res)).to(device) * ((pixel_min + pixel_max)/2)
    single_model.to(device)
    single_model.eval()
    gray_resp = single_model(gray_stim)

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2

    ## To save the responses
    size_tuning_curve   = torch.zeros(len(radii_list_size_tunning_curve))

    grating_images_for_panel = []  
    grating_radii_for_panel = []  

    with torch.no_grad():

        for i, radii_tunning_curve in enumerate(radii_list_size_tunning_curve):

            ## Make the circular stimulus
            grating_center = torch.Tensor(imagen.SineGrating(
                                orientation=preferred_ori,
                                frequency=preferred_sf,
                                phase=preferred_phase,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                offset = -1,
                                scale=2,  
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
            
            ## Mask the circular stimulus 
            center_mask = torch.Tensor(imagen.Disk(smoothing=0.0,
                                size=radii_tunning_curve*2.0, 
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
            
            
            ## Add the gray background to the circular stimulus
            grating = grating_center * center_mask

            ## Change the contrast for the circle
            grating *= contrast


            # print(f"GSF_low={GSF_low}, GSF_high={GSF_high}, Contrasts={contrast_center, contrast_far_surround}, - radius={radius} - Annular_thickness={annular_thickness} ")
            
            directory = "/project/results/facilitation/classical_size_tunning" + "/" + neuron + "/center_masks/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_center_mask{radii_tunning_curve}.png", 
                grating_center.squeeze(), cmap='gray', format='png')
            
            directory = "/project/results/facilitation/classical_size_tunning" + "/" + neuron + "/surround_masks/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_{radii_tunning_curve}.png", 
                center_mask.squeeze(), cmap='gray', format='png')
        
            
            grating = torch.tensor(grating, dtype=torch.float32)
            grating = grating.reshape(1,*img_res).to(device)

            grating = rescale(grating, -1, 1, pixel_min, pixel_max)

            grating_images_for_panel.append(grating.cpu().numpy().squeeze())
            grating_radii_for_panel.append(radii_tunning_curve)
            
            directory = "/project/results/facilitation/classical_size_tunning" + "/" + neuron + "/gratings/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_{radii_tunning_curve}.png", 
                grating.cpu().squeeze(), cmap='gray', format='png')

            ## Save responses for this radius, substract the gray response
            if neg_val == False :
                ## Avoid negative values
                size_tuning_curve[i] = torch.maximum(single_model(grating) - gray_resp, torch.tensor(0))

            else : 
                ## Allow negative values
                size_tuning_curve[i] = single_model(grating) - gray_resp


        radii = np.array(grating_radii_for_panel)
        responses = np.array(size_tuning_curve.cpu())

        # Define your "informative" range
        min_radius = 0.1 
        max_radius = 2.0

        # Filter
        valid_indices = np.where((radii >= min_radius) & (radii <= max_radius))[0]

        # Sample n points evenly from the filtered indices
        n_samples = 12
        if len(valid_indices) > n_samples:
            sampled_indices = valid_indices[np.linspace(0, len(valid_indices)-1, n_samples, dtype=int)]
        else:
            sampled_indices = valid_indices

        # Pick radii and images
        radii_sampled = radii[sampled_indices]
        images_sampled = [grating_images_for_panel[i] for i in sampled_indices]

        plot_classical_size_tuning_gratings(
            neuron=f"{neuron}", 
            grating_images=images_sampled,
            radii=radii_sampled,
            n_samples=n_samples,
        )

    return size_tuning_curve


def get_low_and_high_contrast_contrast(
    single_model, 
    preferred_ori=0, 
    preferred_sf=0, 
    preferred_phase=0,
    contrasts=np.linspace(0.02, 0.82, 40, endpoint=True),         
    img_res=[93,93], 
    pixel_min=-1.7876,
    pixel_max=2.1919,
    device=None,
    size=2.67,
    x_pix=93/2,
    y_pix=93/2,
    mRF_radius=93/4,
    neuron_name="0",
):

    gray_stim = torch.ones((1, *img_res)).to(device) * ((pixel_min + pixel_max)/2)
    single_model.to(device)
    single_model.eval()
    gray_resp = single_model(gray_stim)

    if isinstance(size, (int, float)):
        size = [size]*2

    responses = {}
    gratings = []                     

    # --- generate all gratings and responses first ---
    for contrast in contrasts:
        grating = torch.Tensor(imagen.SineGrating(
            orientation=preferred_ori,
            frequency=preferred_sf,
            phase=preferred_phase,
            bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
            offset=-1,
            scale=2,  
            xdensity=img_res[1]/size[1],
            ydensity=img_res[0]/size[0],
            x=get_offset_in_degr(x_pix, img_res[1], size[1]), 
            y=-get_offset_in_degr(y_pix, img_res[0], size[0])
        )())

        mRF_mask = torch.Tensor(imagen.Disk(
            smoothing=0.0,
            size=mRF_radius * 2.0,  
            bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
            xdensity=img_res[1]/size[1],
            ydensity=img_res[0]/size[0],
            x=get_offset_in_degr(x_pix, img_res[1], size[1]), 
            y=-get_offset_in_degr(y_pix, img_res[0], size[0])
        )())

        masked_grating = grating * mRF_mask
        masked_grating = masked_grating*contrast
        masked_grating = rescale(masked_grating, -1, 1, pixel_min, pixel_max)
        masked_grating = masked_grating.reshape(1, *img_res).to(device)

        response = single_model(masked_grating).detach().cpu().numpy().item()
        responses[contrast] = response - gray_resp.detach().cpu().numpy().item()
        gratings.append(masked_grating.cpu().numpy().squeeze())

    responses_array = np.array([responses[c] for c in contrasts])
    contrasts_array = np.array(contrasts)


    low_contrast, high_contrast = pick_contrasts(contrasts_array, responses_array)

    print(f"Low contrast: {low_contrast}, High contrast: {high_contrast}")

    plot_contrast_tuning_with_gratings(
        neuron=neuron_name,
        contrasts=contrasts,
        responses=[responses[c] for c in contrasts],
        grating_images=gratings,
        low_contrast=low_contrast,
        high_contrast=high_contrast
    )

    return low_contrast, high_contrast

def pick_contrasts(contrasts, responses, low_min=0.04):
    contrasts = np.array(contrasts)
    responses = np.array(responses)

    peak_idx = np.argmax(responses)
    peak_resp = responses[peak_idx]
    low_target = 0.5 * peak_resp
    high_target = 0.9 * peak_resp

    def safe_interp(x0, x1, y0, y1, target):
        if y0 == y1:
            return x1
        if (target - y0) * (target - y1) > 0:  # target not between y0 and y1
            return x1
        return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

    # --- Low contrast: first crossing to >=50% of peak ---
    above_half = np.where(responses >= low_target)[0]
    if len(above_half) > 0:
        i = above_half[0]
        if i == 0:
            low_contrast = contrasts[0]
        else:
            low_contrast = safe_interp(contrasts[i-1], contrasts[i],
                                       responses[i-1], responses[i], low_target)
    else:
        low_contrast = low_min
    low_contrast = max(low_contrast, low_min)

    # --- High contrast: first crossing to >=90% of peak (before or near peak) ---
    above_90 = np.where(responses >= high_target)[0]
    if len(above_90) > 0:
        i = above_90[0]
        if i == 0:
            high_contrast = contrasts[0]
        else:
            high_contrast = safe_interp(contrasts[i-1], contrasts[i],
                                        responses[i-1], responses[i], high_target)
    else:
        # if never reaches 90%, use last point
        high_contrast = contrasts[-1]

    high_contrast = min(high_contrast, 1.0)

    return low_contrast, high_contrast


def plot_contrast_tuning_with_gratings(
    neuron,
    contrasts,
    responses,
    grating_images,
    n_samples=9,
    low_contrast=None,
    high_contrast=None,
    save_dir="/project/results/facilitation/contrast_tunning_panels/",
):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # --- Sample gratings evenly ---
    idxs = np.linspace(0, len(grating_images)-1, n_samples, dtype=int)
    sampled_gratings = [grating_images[i] for i in idxs]
    sampled_contrasts = [contrasts[i] for i in idxs]

    # --- Determine global vmin/vmax (shared for all images and colorbar) ---
    all_min = np.min([img.min() for img in grating_images])
    all_max = np.max([img.max() for img in grating_images])

    # --- Layout: large curve on left, 3x3 grid on right ---
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(3, 4, width_ratios=[3, 1, 1, 1], height_ratios=[1, 1, 1], wspace=0.4, hspace=0.3)

    # --- 1) Contrast tuning curve ---
    ax_curve = fig.add_subplot(gs[:, 0])
    ax_curve.plot(contrasts, responses, "-o", color="black", label="Response Magnitude")

    if low_contrast is not None:
        ax_curve.axvline(low_contrast, color="blue", linestyle="--", label=f"Low = {low_contrast:.2f}")
    if high_contrast is not None:
        ax_curve.axvline(high_contrast, color="red", linestyle="--", label=f"High = {high_contrast:.2f}")

    ax_curve.set_xlabel("Contrast")
    ax_curve.set_ylabel("Response Magnitude")
    ax_curve.set_title(f"{neuron} – Contrast tuning", fontsize=12)
    ax_curve.legend(fontsize=8)
    ax_curve.grid(True)

    # --- 2) 3x3 grating grid with shared scale ---
    for i, (img, contrast) in enumerate(zip(sampled_gratings, sampled_contrasts)):
        row = i // 3
        col = i % 3 + 1
        ax_img = fig.add_subplot(gs[row, col])
        im = ax_img.imshow(img, cmap="gray", origin="lower", vmin=all_min, vmax=all_max)
        ax_img.set_title(f"{contrast*100:.0f}%", fontsize=9)
        ax_img.axis("off")

    # --- Add single shared colorbar for all ---
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label("Pixel intensity", fontsize=10)

    # --- Final layout ---
    fig.suptitle(f"{neuron} – Contrast tuning and sampled stimuli", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])

    # --- Save ---
    save_path = os.path.join(save_dir, f"{neuron}_contrast_tuning_with_gratings.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved panel to {save_path}")


def get_size_tuning_surround_annulus_curve(
    single_model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    preferred_phase,
    size = 2.67,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    img_res = [93,93],
    neg_val = False,     #TODO never used
    GSF_low=1,
    GSF_high=0.5,
    contrast_center=0.8,
    contrast_far_surround=0.4,
    gray_resp=1,
    radii_list_annular_tunning_curve=40,
    neuron=1,
    curve_type="HH",
    ):
    
    '''
    '''

    def make_annulus_radii_and_widths(
        size,                
        inner_r,             
        n_steps=40,
        min_gap=0.05,        
        spacing='log'        
    ):
        """
        Generate outer_annulus_radii (outer circle radius) and corresponding
        edge_widths (distance from image border to outer circle).
        The outer circle moves from the edge inward but never touches the inner circle.
        """

        image_radius = size / 2.0

        # Outer circle moves inward from the image edge toward (inner_r + min_gap)
        max_outer = image_radius - min_gap
        min_outer = inner_r + min_gap

        if min_outer >= max_outer:
            return None, None

        # Generate the outer radii (largest first)
        if spacing == 'log':
            outer_annulus_radii = np.logspace(np.log10(min_outer), np.log10(max_outer), n_steps, endpoint=True)[::-1]
        else:
            outer_annulus_radii = np.linspace(max_outer, min_outer, n_steps)

        # Compute the edge width (distance from image edge to the outer circle)
        edge_widths = image_radius - outer_annulus_radii

        return outer_annulus_radii, edge_widths


    outer_annulus_radii, edge_widths = make_annulus_radii_and_widths(
        size=size,
        inner_r=GSF_low,
        n_steps=radii_list_annular_tunning_curve,
        min_gap=0.05,
        spacing='log'
    )

    if outer_annulus_radii is None or edge_widths is None:
        print(f"[Warning] For neuron {neuron}: inner radius ({GSF_low}) too large for stimulus size ({size}). Skipping annulus generation.")
        return [], torch.tensor([])


    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    single_model.to(device)

    ## Evaluation mode
    single_model.eval()

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2

    ## To save the responses
    circular_tuning_curve   = torch.zeros(len(outer_annulus_radii))

    all_gratings = []
    all_radii = []

    with torch.no_grad():

        for i, outer_annulus_radius in enumerate(outer_annulus_radii):

            ## Make the circular stimulus - the inner circle grating
            grating_center = torch.Tensor(imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=GSF_high*2.0),
                                orientation=preferred_ori,
                                frequency=preferred_sf,
                                phase=preferred_phase,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                offset = -1,
                                scale=2,  
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
            
            ## Mask the circular stimulus - the inner circle mask
            center_mask = torch.Tensor(imagen.Disk(smoothing=0.0,
                                size=GSF_high*2.0, 
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
            
            # surround grating, created for separete control of surround contrast
            surround = torch.Tensor(imagen.SineGrating(
                                orientation=preferred_ori,
                                frequency=preferred_sf,
                                phase=preferred_phase,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                offset = -1,
                                scale=2,  
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())

            surround_mask = (torch.Tensor(imagen.Disk(
                                smoothing=0.0, 
                                size=outer_annulus_radius * 2.0,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))()) * -1) + 1
            

            # print(f"GSF_low={GSF_low}, GSF_high={GSF_high}, Contrasts={contrast_center, contrast_far_surround}, - radius={radius} - Annular_thickness={annular_thickness} ")

            directory = "/project/results/facilitation/grating_centers" + "/" + neuron + "/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_grating_center{edge_widths[i]}.png", 
                grating_center.squeeze(), cmap='gray', format='png')
            
            directory = "/project/results/facilitation/center_masks" + "/" + neuron + "/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_center_mask{edge_widths[i]}.png", 
                center_mask.squeeze(), cmap='gray', format='png')
            
            directory = "/project/results/facilitation/surround_masks" + "/" + neuron + "/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_surround_mask{edge_widths[i]}.png", 
                surround_mask.squeeze(), cmap='gray', format='png')
            
            directory = "/project/results/facilitation/surround" + "/" + neuron + "/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_surround{edge_widths[i]}.png", 
                surround.squeeze(), cmap='gray', format='png')

            # Combine grating_center with background 
            grating = grating_center * center_mask * contrast_center + surround * surround_mask * contrast_far_surround
            
            directory = "/project/results/facilitation/gratings"  + "/" + neuron + "/"
            os.makedirs(directory, exist_ok=True)
            plt.imsave(directory + f"{neuron}_grating_array_{edge_widths[i]}.png", 
                grating.squeeze(), cmap='gray', format='png')
            
            
            grating = torch.tensor(grating, dtype=torch.float32)
            grating = grating.reshape(1,*img_res).to(device)

            grating = rescale(grating, -1, 1, pixel_min, pixel_max)

            all_gratings.append(grating.squeeze().cpu().numpy())
            all_radii.append(edge_widths[i])

            ## Save responses for this radius, substract the gray response
            if neg_val == False :
                ## Avoid negative values
                circular_tuning_curve[i] = torch.maximum(single_model(grating) - gray_resp, torch.tensor(0))

            else : 
                ## Allow negative values
                circular_tuning_curve[i] = single_model(grating) - gray_resp


        plot_sampled_annular_gratings(
            neuron=f"{neuron}",
            curve_type=curve_type,
            radii=np.array(all_radii),
            grating_images=all_gratings,
            n_samples=12
        )
    
    return edge_widths, circular_tuning_curve  



def plot_sampled_annular_gratings(
    neuron,
    curve_type,
    radii,
    grating_images,
    n_samples=12,
    save_dir="/project/results/facilitation/annular_curves_gratings_panel/",
):
    """
    Show sampled annular gratings (each with its own colorbar).
    Shared grayscale range so brightness is directly comparable.
    Always includes smallest and largest radii.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    os.makedirs(save_dir, exist_ok=True)

    # --- Determine which stimuli to show ---
    n_total = len(grating_images)
    n_samples = min(n_samples, n_total)

    if n_samples <= 2:
        sampled_idxs = [0, n_total - 1][:n_samples]
    else:
        middle_idxs = np.linspace(1, n_total - 2, n_samples - 2, dtype=int)
        sampled_idxs = np.concatenate(([0], middle_idxs, [n_total - 1]))

    sampled_gratings = [grating_images[i] for i in sampled_idxs]
    sampled_radii = [radii[i] for i in sampled_idxs]

    # --- Global intensity range for all images ---
    all_min = np.min([img.min() for img in grating_images])
    all_max = np.max([img.max() for img in grating_images])

    # --- Layout ---
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    axes = axes.ravel()

    for i, (ax, img, w) in enumerate(zip(axes, sampled_gratings, sampled_radii)):
        im = ax.imshow(img, cmap="gray", origin="lower", vmin=all_min, vmax=all_max)
        ax.set_title(f"width = {w:.2f}°", fontsize=8)
        ax.axis("off")

        # Individual colorbar, consistent scale
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Pixel intensity", fontsize=7)

    # --- Cleanup ---
    fig.suptitle(f"Sampled annular gratings – {neuron}, {curve_type}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # --- Save ---
    save_path = os.path.join(save_dir, f"{neuron}_{curve_type}_sampled_annular_gratings.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved panel to {save_path}")


def get_center_alone_stimulus(
    single_model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    preferred_phase,
    size = 2.67,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    img_res = [93,93],
    GSF_high=1,
    contrast_center=0.8,
    neuron=1,
    gray_resp=1
    ):

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2


    ## Make the circular stimulus - the inner circle grating
    grating_center = torch.Tensor(imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=GSF_high*2.0),
                        orientation=preferred_ori,
                        frequency=preferred_sf,
                        phase=preferred_phase,
                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                        offset = -1,
                        scale=2,  
                        xdensity=img_res[1]/size[1],
                        ydensity=img_res[0]/size[0], 
                        x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                        y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
    
    ## Mask the circular stimulus - the inner circle mask
    center_mask = torch.Tensor(imagen.Disk(smoothing=0.0,
                        size=GSF_high*2.0, 
                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                        xdensity=img_res[1]/size[1],
                        ydensity=img_res[0]/size[0], 
                        x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                        y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())

                
    grating = grating_center * center_mask * contrast_center


    directory = "/project/results/facilitation/center_only"  + "/" + neuron + "/"
    os.makedirs(directory, exist_ok=True)
    plt.imsave(directory + "center_only.png", 
        grating.squeeze(), cmap='gray', format='png')
    
    grating = torch.tensor(grating, dtype=torch.float32)
    grating = grating.reshape(1,*img_res).to(device)

    grating = rescale(grating, 0, 1, -1, 1)
    grating = rescale(grating, -1, 1, pixel_min, pixel_max)

    response = single_model(grating)

    return response - gray_resp

def get_full_grating_stimulus(
    single_model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    preferred_phase,
    size = 2.67,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    img_res = [93,93],
    GSF_high=1,
    contrast_center=0.8,
    gray_resp=1,
    neuron=1
    ):

            ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2


    ## Make the circular stimulus - the inner circle grating
    grating_center = torch.Tensor(imagen.SineGrating(
                        orientation=preferred_ori,
                        frequency=preferred_sf,
                        phase=preferred_phase,
                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                        offset = -1,
                        scale=2,  
                        xdensity=img_res[1]/size[1],
                        ydensity=img_res[0]/size[0], 
                        x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                        y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
    
    
    grating = grating_center * contrast_center

    
    grating = torch.tensor(grating, dtype=torch.float32)
    grating = grating.reshape(1,*img_res).to(device)

    grating = rescale(grating, -1, 1, pixel_min, pixel_max)

    response = single_model(grating)

    return response - gray_resp



def ellipse_from_fit(x0, y0, sigma_x, sigma_y, rho,
                     img_h, img_w,
                     n_sigma=2,
                     origin='lower',
                     force_major_point_right=True):
    """
    Return (center_x_plot, center_y_plot, width, height, angle_deg) suitable
    for matplotlib.patches.Ellipse so that it overlays correctly on imshow(..., origin=origin).

    - x0,y0 : fit outputs in array coordinates (x = col, y = row, origin top-left)
    - sigma_x, sigma_y, rho : Gaussian fit params (stddevs and correlation)
    - img_h, img_w : image shape (height, width)
    - n_sigma : contour level (e.g. 2 for 2σ)
    - origin : 'lower' (default) if you plot imshow(..., origin='lower') OR 'upper' if you use origin='upper'
    - force_major_point_right : ensures consistent eigenvector direction (avoids ±180° flips)
    """
    # 1) Covariance matrix in (x,y) coords where x is horizontal, y vertical (array coords)
    cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                    [rho*sigma_x*sigma_y, sigma_y**2]])

    # 2) Eigen-decomposition, sort descending so index 0 is major axis
    eigvals, eigvecs = np.linalg.eigh(cov)                # ascending
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]                          # columns are eigenvectors

    # 3) Semi-axis lengths (sigma along eigenvectors): sqrt(eigvals)
    #    Matplotlib Ellipse expects full axis lengths (width,height) = 2 * semi-axis
    #    For n_sigma contour: semi-axis = n_sigma * sqrt(eigval) -> full length = 2 * n_sigma * sqrt(eigval)
    full_lengths = 2.0 * n_sigma * np.sqrt(eigvals)      # [major_full, minor_full]

    width = float(full_lengths[0])
    height = float(full_lengths[1])

    # 4) Eigenvector for major axis
    vx, vy = eigvecs[0, 0], eigvecs[1, 0]  # major eigenvector components in array coords

    # 5) Convert fit center (array coords) --> plotting coords depending on origin
    if origin == 'lower':
        # imshow(..., origin='lower') shows row 0 at bottom => convert y
        cx = float(x0)
        cy = float(img_h - 1 - y0)
        # when flipping y, the eigenvector's y component must be flipped to compute angle in plot coords
        vy_plot = -vy
        vx_plot = vx
    elif origin == 'upper':
        # imshow(..., origin='upper') displays array coords directly (row 0 top)
        cx = float(x0)
        cy = float(y0)
        vy_plot = vy
        vx_plot = vx
    else:
        raise ValueError("origin must be 'lower' or 'upper'")

    # 6) Angle in degrees measured CCW from +x axis for matplotlib
    angle_deg = np.degrees(np.arctan2(vy_plot, vx_plot))

    # 7) Stabilize orientation: make angle in [0,180)
    angle_deg = angle_deg % 180.0

    # 8) Force major axis point-right (optional): if the x component of plot eigenvector is negative, flip 180°.
    #    This removes the arbitrary eigenvector sign ambiguity across neurons.
    if force_major_point_right:
        # compute unit vector of (vx_plot, vy_plot)
        norm = np.hypot(vx_plot, vy_plot)
        if norm == 0:
            pass
        else:
            ux, uy = vx_plot / norm, vy_plot / norm
            # if major axis's x-component is < 0, rotate by 180 deg (flip direction)
            if ux < 0:
                angle_deg = (angle_deg + 180.0) % 360.0
                # keep angle in [0,180)
                angle_deg = angle_deg % 180.0

    # 9) Guarantee width is major axis (defensive)
    if width < height:
        width, height = height, width
        angle_deg = (angle_deg + 90.0) % 180.0

    # Return ellipse parameters in plotting coordinates
    return cx, cy, width, height, angle_deg


def verify_ellipse_vs_gaussian(cx, cy, width, height, angle_deg,
                               xs, ys, mu_x_plot, mu_y_plot,
                               sigma_x, sigma_y, rho, n_sigma=2,
                               atol=0.05):
    """
    Numeric sanity check: sample points on ellipse boundary and check they are close to
    the Gaussian level set value for n_sigma (i.e. exp(-0.5*n_sigma^2)).
    Returns (max_abs_error, passed_bool). Useful as a unit test.
    - xs, ys : meshgrid used for Gaussian (plot coords)
    - mu_x_plot, mu_y_plot : center in plot coords (must match cx,cy)
    """
    # Gaussian analytic level
    level = np.exp(-0.5 * (n_sigma**2))

    # param paramization of ellipse boundary in plot coords
    # semi-axes:
    a = width / 2.0
    b = height / 2.0
    th = np.radians(angle_deg)

    # generate points around ellipse boundary
    thetas = np.linspace(0, 2*np.pi, 180)  # dense
    x_ell = a * np.cos(thetas)
    y_ell = b * np.sin(thetas)
    # rotate by angle and translate to cx,cy
    xr = cx + (x_ell * np.cos(th) - y_ell * np.sin(th))
    yr = cy + (x_ell * np.sin(th) + y_ell * np.cos(th))

    # Evaluate gaussian (in plot coords we assume gaussian_2d uses plot coords)
    X_rel = xr - mu_x_plot
    Y_rel = yr - mu_y_plot
    denom = 2.0 * (1.0 - rho**2)
    Zr = np.exp(-(X_rel**2 / sigma_x**2 + Y_rel**2 / sigma_y**2 - 2*rho*X_rel*Y_rel/(sigma_x*sigma_y)) / denom)

    # compute error relative to analytic level
    err = np.abs(Zr - level)
    return float(err.max()), (err.max() <= atol)


def plot_classical_size_tuning_gratings(
    neuron,
    grating_images,
    radii=None,           # optional, used for titles
    n_samples=12,         # number of images to show
    save_dir="/project/results/facilitation/classical_size_tunning_grating_panels/",
):
    """
    Show a grid of sampled gratings only, no curves.
    All gratings share the same color scale (vmin/vmax) for consistent background tone.
    Each grating gets its own colorbar.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    os.makedirs(save_dir, exist_ok=True)

    # --- Sample evenly across available gratings ---
    n_total = len(grating_images)
    n_samples = min(n_samples, n_total)
    idxs = np.linspace(0, n_total - 1, n_samples, dtype=int)
    sampled_gratings = [grating_images[i] for i in idxs]
    sampled_radii = [radii[i] for i in idxs] if radii is not None else [None] * n_samples

    # --- Determine shared color scale ---
    all_min = min(img.min() for img in sampled_gratings)
    all_max = max(img.max() for img in sampled_gratings)

    # --- Grid layout ---
    n_rows = 3
    n_cols = int(np.ceil(n_samples / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 9))
    axes = axes.ravel()

    for i, (ax, img, r) in enumerate(zip(axes, sampled_gratings, sampled_radii)):
        im = ax.imshow(img, cmap='gray', origin='lower', vmin=all_min, vmax=all_max)   
        title = f"R={r:.2f}" if r is not None else ""
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        

        # --- individual colorbar ---
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, fraction=0.046, pad=0.04)
        cbar.set_label("Pixel intensity", fontsize=8)  



    # Hide any empty axes if n_samples < n_rows * n_cols
    for ax in axes[len(sampled_gratings):]:
        ax.axis("off")

    # --- Title and save ---
    fig.suptitle(f"Sampled size tuning at full contrast – {neuron}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, f"{neuron}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved panel to {save_path}")