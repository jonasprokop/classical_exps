import torch
import numpy as np



def rog_asymptote_from_params(kc, ks, wc, ws):
    return (kc * (wc**2)) / (1.0 + ks * (ws**2))


def get_GSF_surround_AMRF(
    radii,
    circular_tuning_curve,
    annular_tuning_curve = None
    ): 

    ''' This function takes the circular and annular tuning curves outputs of the function "get_size_tuning_curves".
        For the circular stimuli : 
            -It finds the Grating Summation Field (GSF) defined as the "diameter of the smallest stimulus that elicited at least 95% of the
            neuron’s maximum response".
            -It finds the surround extent radius defined as the "inhibitory surround extent as the diameter of the smallest stimulus 
            for which the neuron’s response was reduced to within 5% of its asymptotic value for the largest gratings".
            -It computes the supression index defined as "(Ropt - Rsupp) / Ropt", where Ropt is the response at the GSF and Rsupp is the asymptotic response.
        For the annular stimuli :
            -It finds the Annular Minimum Response Field (AMRF) defined as "the point at which the response to the annular stimulus reached
            a value of at most 5% of the neuron’s maximum response to a circular patch of grating."
        
        Optionally, if no annular_tuning_curve is given, this function will not returns the AMRF

        Arguments : 

            - radii                  : The array that contains the radii that were used to obtain circular_tuning_curve and annular_tuning_curve
            - circular_tuning_curve  : An array, the center stimulus size tuning curve
            - annular_tuning_curve   : An array, the surround stimulus size tuning curve

        Outputs : 

            - GSF             : Optimal circular radius that elicits approximately the maximal response
            - surround_extent : Optimal circular radius that elicits approximately the asymptotic response (suppression)
            - AMRF            : Optimal annular radius that elicits approximately the lowest response
            - Ropt            : The response of the model at the GSF for the circular stimuli
            - Rsupp           : The asymptotic response of the model for the circular stimuli

            NB : outputs are radii, not diameters
    '''
    
    ## Make sure the elements are tensors
    circular_tuning_curve = torch.as_tensor(circular_tuning_curve)
    if annular_tuning_curve is not None :
        annular_tuning_curve = torch.as_tensor(annular_tuning_curve)

    ## Avoid zeros
    GSF             = 1.e-9
    surround_extent = 1.e-9
    AMRF            = 1.e-9
    Ropt            = 0
    Rsupp           = circular_tuning_curve[-1] #The last response = asymptotic value

    ## Select the thresholds 
    GSF_thresh   = (95 * torch.max(circular_tuning_curve)) / 100 # correct
    a = Rsupp - ((5 * Rsupp) /100)
    b = Rsupp + ((5 * Rsupp) /100)
    surround_thresh_min = min(a,b)
    surround_thresh_max = max(a,b)
    AMRF_thresh  = 5 * torch.max(circular_tuning_curve) / 100 # correct
    
    ## For the GSF
    for i, resp in enumerate(circular_tuning_curve):
        if resp > GSF_thresh :
            GSF = radii[i]
            Ropt = resp
            break

    ## For the surround extent
    for i, resp in enumerate(circular_tuning_curve):
        ## Avoid to take values with smaller radius
        if radii[i] > GSF :
            ## Assert that the response is decreasing
            if resp < Ropt :
                if resp >= surround_thresh_min and resp <= surround_thresh_max:
                    surround_extent = radii[i]
                    break
    
    if annular_tuning_curve != None :
        ## For the AMRF
        for i, resp in enumerate(annular_tuning_curve):
            if resp < AMRF_thresh:
                AMRF = radii[i]
                break
    
    
    SI = ((Ropt - Rsupp) / Ropt).item()


    if annular_tuning_curve is not None :
        return GSF, surround_extent, AMRF, SI, Ropt, Rsupp
    
    else :
        
        return GSF, surround_extent, SI, Ropt, Rsupp
    
def get_GSF_surround_AMRF_from_fit(
    *,
    radii,
    circular_fit,
    annular_fit=None,
    fit_params=None,     # accepted for compatibility; unused here
    tail_k=1,
    gsf_frac=0.95,
    recover_frac=0.05,   # ±5% of Rsupp band
    annular_frac=0.05,   # 5% of max(circular)
    min_valid_radius=1e-12,
):
    """
    Forced article-style markers sampled on `radii` (RADIUS outputs, not diameters).

    Always returns:
      GSF, surround_extent, AMRF, SI, Ropt, Rsupp
    with fallbacks so nothing is NaN and surround never ends at radius 0.
    """
    r0 = np.asarray(radii, float)
    yc0 = np.asarray(circular_fit, float)

    # basic align + finite mask
    n = min(r0.size, yc0.size)
    r0 = r0[:n]
    yc0 = yc0[:n]
    m = np.isfinite(r0) & np.isfinite(yc0)
    r = r0[m]
    yc = yc0[m]

    if r.size < 2:
        # absolute last-resort
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # ensure increasing radii (just in case)
    # (if your grid isn't sorted, nothing makes sense anyway)
    if np.any(np.diff(r) < 0):
        idx = np.argsort(r)
        r = r[idx]
        yc = yc[idx]

    # Rsupp: "largest gratings"
    k = int(min(max(1, tail_k), yc.size))
    Rsupp = float(np.mean(yc[-k:]))

    ymax = float(np.max(yc))
    if not np.isfinite(ymax):
        ymax = 0.0

    # ---------- GSF ----------
    # threshold at gsf_frac * max
    gsf_thr = float(gsf_frac) * ymax
    gsf_candidates = np.where(yc >= gsf_thr)[0]
    if gsf_candidates.size:
        gsf_i = int(gsf_candidates[0])
    else:
        gsf_i = int(np.argmax(yc))

    # avoid radius=0 as GSF if possible
    if r[gsf_i] <= min_valid_radius:
        pos = np.where(r > min_valid_radius)[0]
        if pos.size:
            # pick earliest positive radius that is still reasonably high, else just first positive
            # try to keep the "first reaching threshold" spirit
            pos_thr = pos[yc[pos] >= gsf_thr] if ymax > 0 else np.array([], int)
            gsf_i = int(pos_thr[0]) if pos_thr.size else int(pos[0])

    GSF = float(r[gsf_i])
    Ropt = float(yc[gsf_i])

    # ---------- SI ----------
    # force SI to be finite even if Ropt ~ 0
    denom = max(abs(Ropt), 1e-12)
    SI = float((Ropt - Rsupp) / denom)

    # ---------- Surround extent ----------
    # forced: if we cannot find "recovery", set to max radius
    # also: never allow surround extent at radius 0
    tol = float(recover_frac) * max(abs(Rsupp), 1e-12)
    lo, hi = Rsupp - tol, Rsupp + tol

    # search must start AFTER GSF, and at least from index 1 to avoid 0-radius nonsense
    start_i = max(gsf_i + 1, 1)

    surr_i = None
    for i in range(start_i, yc.size):
        yi = float(yc[i])
        if lo <= yi <= hi:
            surr_i = i
            break

    if surr_i is None:
        surround_extent = float(r[-1])
    else:
        surround_extent = float(r[surr_i])

    # final guard: no 0 radius surround
    if surround_extent <= min_valid_radius and r.size > 1:
        surround_extent = float(r[-1])

    # ---------- AMRF ----------
    # article-ish: first radius where annular <= annular_frac * max(circular)
    # forced: if never crosses, set to max radius
    if annular_fit is None:
        AMRF = float(r[-1])
    else:
        ya0 = np.asarray(annular_fit, float)[:n]
        ya = ya0[m]
        if np.any(np.diff(r) < 0):
            ya = ya[idx]  # keep same sort if we sorted

        thr_a = float(annular_frac) * ymax
        amrf_candidates = np.where(ya <= thr_a)[0]
        AMRF = float(r[int(amrf_candidates[0])]) if amrf_candidates.size else float(r[-1])

    return GSF, surround_extent, AMRF, SI, Ropt, Rsupp