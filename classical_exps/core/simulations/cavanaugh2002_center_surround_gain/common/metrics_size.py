import torch
import numpy as np



def rog_asymptote_from_params(kc, ks, wc, ws):
    return (kc * (wc**2)) / (1.0 + ks * (ws**2))


def _extract_size_tuning_markers_core(
    radii,
    circular_curve,
    annular_curve=None,
    *,
    min_valid_radius=1e-9,
    gsf_frac=0.95,
    surround_recover_frac=0.05,
    annular_frac=0.05,
    require_article_exclusion=True,
):
    """
    Cavanaugh-style marker extraction on sampled curves.

    Inputs and outputs are in RADII, not diameters.

    Notes
    -----
    This keeps the logic as close as possible to the article text:
    - GSF: first sampled radius reaching >= 95% of max circular response
    - Rsupp: asymptotic large-stimulus response approximated by last sampled point
    - surround_extent: first sampled point after the peak/GSF whose response is
      within 5% of Rsupp; otherwise largest sampled radius
    - exclude only when there is neither suppression nor response saturation
    - AMRF: first annular radius with response <= 5% of circular max
    """
    r0 = np.asarray(radii, dtype=float)
    yc0 = np.asarray(circular_curve, dtype=float)

    n = min(r0.size, yc0.size)
    r0 = r0[:n]
    yc0 = yc0[:n]

    m = np.isfinite(r0) & np.isfinite(yc0) & (r0 > float(min_valid_radius))
    r = r0[m]
    yc = yc0[m]

    result = {
        "GSF": np.nan,
        "surround_extent": np.nan,
        "AMRF": np.nan,
        "SI": np.nan,
        "Ropt": np.nan,
        "Rsupp": np.nan,
        "suppression_present": False,
        "saturation_present": False,
        "suppression_asymptote_reached": False,
        "valid": False,
    }

    if r.size < 2:
        return result

    if np.any(np.diff(r) < 0):
        order = np.argsort(r)
        r = r[order]
        yc = yc[order]
    else:
        order = None

    ymax = float(np.max(yc))
    i_opt = int(np.argmax(yc))
    Ropt = float(yc[i_opt])

    # Article-faithful discrete proxy for "asymptotic value for the largest gratings"
    Rsupp = float(yc[-1])

    denom = max(abs(Ropt), 1e-12)
    SI = float((Ropt - Rsupp) / denom)

    # GSF: smallest sampled radius reaching >= 95% of maximum
    thr_gsf = float(gsf_frac) * ymax
    gsf_idx = np.where(yc >= thr_gsf)[0]
    if gsf_idx.size == 0:
        return result
    i_gsf = int(gsf_idx[0])
    GSF = float(r[i_gsf])

    # Suppression present?
    eps = 1e-12 + 1e-6 * max(1.0, abs(ymax))
    suppression_present = bool((Ropt - Rsupp) > eps)

    # Response saturation:
    # minimal sampled-curve interpretation, not custom tail flattening.
    #
    # We call saturation present if, after the response has risen, the curve reaches
    # the final large-stimulus level within the 5% band.
    band = max(float(surround_recover_frac) * max(abs(Rsupp), 1e-12), 1e-12)
    lo, hi = Rsupp - band, Rsupp + band

    start = max(i_opt, i_gsf) + 1
    candidate_idx = np.where((np.arange(yc.size) >= start) & (yc >= lo) & (yc <= hi))[0]

    if candidate_idx.size > 0:
        i_surr = int(candidate_idx[0])
        saturation_present = True
        suppression_asymptote_reached = suppression_present
        surround_extent = float(r[i_surr])
    else:
        i_surr = None
        # If the tail itself is already near Rsupp, we at least have response saturation
        saturation_present = bool(abs(float(yc[-1]) - Rsupp) <= band)
        suppression_asymptote_reached = False
        surround_extent = float(r[-1])

    result.update({
        "GSF": GSF,
        "surround_extent": surround_extent,
        "SI": SI,
        "Ropt": Ropt,
        "Rsupp": Rsupp,
        "suppression_present": suppression_present,
        "saturation_present": saturation_present,
        "suppression_asymptote_reached": suppression_asymptote_reached,
    })

    # Paper exclusion: neither suppression nor response saturation
    if require_article_exclusion and (not suppression_present) and (not saturation_present):
        return result

    # AMRF: first annular radius where annular response <= 5% of circular max
    if annular_curve is None:
        AMRF = np.nan
    else:
        ya0 = np.asarray(annular_curve, dtype=float)[:n]
        ya = ya0[m]
        if order is not None:
            ya = ya[order]

        thr_a = float(annular_frac) * ymax
        amrf_idx = np.where(ya <= thr_a)[0]
        AMRF = float(r[int(amrf_idx[0])]) if amrf_idx.size else float(r[-1])

    result.update({
        "AMRF": AMRF,
        "valid": True,
    })
    return result

def get_GSF_surround_AMRF_from_fit(
    *,
    radii,
    circular_fit,
    annular_fit=None,
    fit_params=None,   # kept for pipeline compatibility; unused
    tail_k=1,
    min_valid_radius=1e-9,
    require_article_exclusion=False,
):
    """
    Marker extraction on sampled fit curves using the same core logic as raw data.

    Returns always:
        GSF, surround_extent, AMRF, SI, Ropt, Rsupp

    Notes
    -----
    - Inputs/outputs are in RADII.
    - By default, article exclusion is disabled for fit-based diagnostic use.
      Set require_article_exclusion=True for strict apples-to-apples comparison.
    """
    d = _extract_size_tuning_markers_core(
        radii,
        circular_fit,
        annular_fit,
        min_valid_radius=min_valid_radius,
        require_article_exclusion=require_article_exclusion,
    )

    if not d["valid"]:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return d["GSF"], d["surround_extent"], d["AMRF"], d["SI"], d["Ropt"], d["Rsupp"]



def get_GSF_surround_AMRF(
    radii,
    circular_tuning_curve,
    annular_tuning_curve=None,
    *,
    tail_k=1,
    min_valid_radius=1e-9,
):
    """
    Paper-faithful empirical markers (Cavanaugh et al. 2002).

    Returns:
      with annular:
        GSF, surround_extent, AMRF, SI, Ropt, Rsupp
      without annular:
        GSF, surround_extent, SI, Ropt, Rsupp
    """
    d = _extract_size_tuning_markers_core(
        radii,
        circular_tuning_curve,
        annular_tuning_curve,
        min_valid_radius=min_valid_radius,
        require_article_exclusion=True,
    )

    if not d["valid"]:
        if annular_tuning_curve is not None:
            return None, None, None, None, None, None
        return None, None, None, None, None

    if annular_tuning_curve is not None:
        return d["GSF"], d["surround_extent"], d["AMRF"], d["SI"], d["Ropt"], d["Rsupp"]

    return d["GSF"], d["surround_extent"], d["SI"], d["Ropt"], d["Rsupp"]