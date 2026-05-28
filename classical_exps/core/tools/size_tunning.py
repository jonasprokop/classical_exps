from __future__ import annotations

# Useful
import os
import time

import numpy as np
import torch
from tqdm import tqdm

# Utils (project)
from classical_exps.core.tools.utils import *  # noqa: F401,F403
from classical_exps.core.tools.utils import SingleCellModel  # noqa: F401
from classical_exps.core.simulations.cavanaugh2002_center_surround_gain.common.metrics_size import get_GSF_surround_AMRF


# Image generation
import imagen
from imagen.image import BoundingBox

# Data / optimization / plotting
import h5py
from scipy.optimize import curve_fit
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from classical_exps.core.tools.utils import ensure_dir

def get_size_tuning_curves(
    single_model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    preferred_phase,
    radii = np.logspace(-2,np.log10(2),40) ,
    contrast = 1,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True,     #TODO never used
    compute_annular = True,
    neuron = "0",
    contrast_keyword="",
    return_stimuli=True,
    n_stimuli_samples=12,
    stim_radius_range=(0.1, 2.0),
    save_stimuli_panel=True,
    stimuli_panel_dir="/project/results/nature_and_interactions/stimuli/size_tuning_panels/",
    stimuli_panel_tag="",
    ):
    
    ''' This function is a size tuning function that makes the size of circular and annular grating images vary 
        with their prefered parameters fixed. The difference between the response of the model for each image and the response for a gray image 
        is calculated and saved into an array of the same size of the argument "radii". You can afterwards plot these arrays
        to visualise the effect of the radii on the model's response. If neg_val is set to 'False', the negative values
        are set to 0.
        

        NB : We substract the response to gray image to the response of stimuli because it is necessary for the rest of our analysis, it normalises the values.
        NB2: What is called here 'annular' is actually not a ring since there is no outer boundary, the 'ring' fills the outer space of the image. 

        Arguments : 

            - single_model    : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
            - x_pix           : The x coordinates of the center of the neuron's receptive field (in pixel, can be a float)
            - y_pix           : The y coordinates   ""        ""        ""
            - preferred_ori   : The preferred orientation (the one that elicited the greatest response)
            - preferred_sf    : The preferred spatial frequency
            - preferred_phase : The preferred phase
            - radii           : An array containing every radius size to test, NB : here the radius is from the center to the inner edge of the ring
            - Thickness       : The thickness of the annular item (half of the full thickness)
            - contrast        : The value that will multiply the image's values 
            - pixel_min       : Value of the minimal pixel that will serve as the black reference
            - pixel_max       : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
            - device          : The device on which to execute the code, if set to "None" it will take the available one
        ?   - size            : The size of the image in terms of visual field degrees
            - img_res         : Resolution of the image in term of pixels shape : [pix_y, pix_x]
            - neg_val         : If set to 'False', the outputs won't include negative values but will replace them by 0.
            - compute_annular : If set to 'False', the function will only perform size tuning on a circular grating image

        Outputs : 

        - circular_tuning_curve  : An array containing the responses for every center stimulus minus the response to gray image
        
            - If compute_annular == True :
                
                - annular_tuning_curve   : An array containing the responses for every surround stimulus minus the response to gray image
    '''

    center_images_for_panel = []
    annular_images_for_panel = []
    radii_for_panel = []
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
    circular_tuning_curve   = torch.zeros(len(radii))
    annular_tuning_curve = torch.zeros(len(radii))

    with torch.no_grad():

        ## Create Gray stimulus to substract to the non gray stimuli
        gray_stim = torch.ones((1, *img_res)).to(device) * ((pixel_min + pixel_max)/2)
        gray_resp = single_model(gray_stim)

        for i, radius in enumerate(radii) :
            
            ## Make the circular stimulus
            grating_center = torch.Tensor(imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=radius*2.0),
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
                                size=radius*2.0, 
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
            
            
            ## Add the gray background to the circular stimulus
            grating_center = grating_center * center_mask

            ## Change the contrast for the circle
            grating_center *= contrast

            
            center_images_for_panel.append(grating_center.detach().cpu().numpy().squeeze())
            radii_for_panel.append(float(radius))

            ## Convert to the right shape for the model
            # fixed version
            grating_center = rescale(grating_center,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)
            
            # ## Convert to the right shape for the model
            # grating_center = rescale(grating_center,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)


            ## Save responses for this radius, substract the gray response
            if neg_val == False :
                ## Avoid negative values
                circular_tuning_curve[i] = torch.maximum(single_model(grating_center) - gray_resp, torch.tensor(0))

            else : 
                ## Allow negative values
                circular_tuning_curve[i] = single_model(grating_center) - gray_resp

            ## Same for the annular stimulus
            if compute_annular : 
                ## Make the annular stimulus
                ## Create a grating background
                grating_background = torch.Tensor(imagen.SineGrating(
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
                

                ## Create the inner edge of the ring
                inner_edge = (torch.Tensor(imagen.Disk(
                                    smoothing=0.0, 
                                    size=(radius)*2.0,
                                    bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
                                    xdensity=img_res[1]/size[1],
                                    ydensity=img_res[0]/size[0], 
                                    x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                    y = -get_offset_in_degr(y_pix, img_res[0], size[0]))()) * -1) + 1
                grating_ring = grating_background * inner_edge

                ## Change the contrast for the ring
                grating_ring   *= contrast

                annular_images_for_panel.append(grating_ring.detach().cpu().numpy().squeeze())

                
                ## Convert to the right shape for the model 
                grating_ring = rescale(grating_ring,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)

                
                ## Save responses for this radius, substract the gray response
                if neg_val == False :
                    ## Avoid negative values
                    annular_tuning_curve[i]  = torch.maximum(single_model(grating_ring) - gray_resp,torch.tensor(0))   

                else : 
                    ## Allow negative values
                    annular_tuning_curve[i]  = single_model(grating_ring) - gray_resp


    # radii and responses arrays
    radii = np.array(radii)
    responses = np.array(circular_tuning_curve.cpu())

    # Define your "informative" range
    min_radius = 0
    max_radius = 2.67

    # Filter
    valid_indices = np.where((radii >= min_radius) & (radii <= max_radius))[0]

    # radii and responses arrays
    radii = np.array(radii)

    radii_np = np.asarray(radii_for_panel, dtype=float)

    stim_payload = None
    if return_stimuli:
        lo, hi = stim_radius_range
        valid = np.where((radii_np >= lo) & (radii_np <= hi))[0]
        if len(valid) == 0:
            valid = np.arange(len(radii_np))

        n_samples = int(min(max(1, n_stimuli_samples), len(valid)))
        sampled = valid[np.linspace(0, len(valid) - 1, n_samples, dtype=int)]

        stim_payload = {
            "radii": radii_np,  # full
            "radii_sampled": radii_np[sampled],
            "center_sampled": [center_images_for_panel[i] for i in sampled],
            "annular_sampled": [annular_images_for_panel[i] for i in sampled] if compute_annular else None,
            # optionally keep full arrays too if you ever need them
            # "center_full": center_images_for_panel,
            # "annular_full": annular_images_for_panel if compute_annular else None,
        }


    if return_stimuli or save_stimuli_panel:
        radii_np = np.asarray(radii_for_panel, dtype=float)

        lo, hi = stim_radius_range
        valid = np.where((radii_np >= lo) & (radii_np <= hi))[0]
        if len(valid) == 0:
            valid = np.arange(len(radii_np))

        n_samples = int(min(max(1, n_stimuli_samples), len(valid)))
        sampled = valid[np.linspace(0, len(valid)-1, n_samples, dtype=int)]

        radii_sampled = radii_np[sampled]
        center_sampled = [center_images_for_panel[i] for i in sampled]
        annular_sampled = [annular_images_for_panel[i] for i in sampled] if compute_annular else None

    if save_stimuli_panel and compute_annular:
        neuron_name = neuron if neuron else "0"
        tag = f"_{stimuli_panel_tag}" if stimuli_panel_tag else ""
        plot_size_tuning_stimuli_panel(
            neuron=neuron,
            radii_sampled=radii_sampled,
            center_imgs_sampled=center_sampled,
            annular_imgs_sampled=annular_sampled,
            save_dir=stimuli_panel_dir,
            filename=f"{neuron_name}_size_tuning_stimuli{tag}.png",
            title_extra=(f" {tag}" if tag else ""),
        )


    if compute_annular :
        return circular_tuning_curve, annular_tuning_curve
    
    else :
        return circular_tuning_curve
    

def plot_size_tuning_stimuli_panel(
    neuron: str,
    radii_sampled,
    center_imgs_sampled,
    annular_imgs_sampled,
    save_dir,
    filename="stimuli.png",
    title_extra="",
    cmap="gray",
):
    os.makedirs(save_dir, exist_ok=True)

    radii_sampled = np.asarray(radii_sampled, dtype=float)
    K = len(radii_sampled)

    assert len(center_imgs_sampled) == K, "center_imgs_sampled length mismatch"
    assert len(annular_imgs_sampled) == K, "annular_imgs_sampled length mismatch"

    all_imgs = list(center_imgs_sampled) + list(annular_imgs_sampled)
    vmin = float(min(im.min() for im in all_imgs))
    vmax = float(max(im.max() for im in all_imgs))

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, axes = plt.subplots(2, K, figsize=(1.5*K, 3.6), dpi=220, sharex=True, sharey=True)
    if K == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j in range(K):
        axes[0, j].imshow(center_imgs_sampled[j], cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
        axes[1, j].imshow(annular_imgs_sampled[j], cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
        axes[0, j].axis("off")
        axes[1, j].axis("off")
        axes[0, j].set_title(f"{2*radii_sampled[j]:.2f}°", fontsize=8)  # diameter labels

    axes[0, 0].set_ylabel("Center", fontsize=10)
    axes[1, 0].set_ylabel("Annular", fontsize=10)

    fig.suptitle(f"{neuron} – size tuning stimuli{title_extra}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.94, 0.90])

    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.05, pad=0.02)
    cbar.set_label("pixel value", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    out = os.path.join(save_dir, filename)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out
