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
from classical_exps.functions.article_1.common.metrics_size import get_GSF_surround_AMRF
## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from classical_exps.functions.utils import plot_img
from .shared import neuron_key, ensure_dir  
## Data storage
import h5py
from collections import Counter
import scipy
from datetime import datetime
import json
import pandas as pd
import os
import openpyxl
from matplotlib.ticker import ScalarFormatter

def export_nature_and_interactions_excel(
    excel_path: str,
    h5_file: str,
    neuron_ids,
    run: dict,
    params: dict,
    loaded: dict,
):
    """
    Writes:
      - neurons: per-neuron GSF/surr/SI
      - run: run dict + params
      - loads: summary counts per loader
    """
    neuron_ids = np.asarray(neuron_ids, dtype=int)

    # Build per-neuron table (always possible even if some loads disabled)
    rows = []
    size_res = loaded.get("size_results", None)
    size_by_id = size_res["by_id"] if size_res is not None else {}

    for nid in neuron_ids:
        d = size_by_id.get(int(nid), None)
        rows.append({
            "neuron_id": int(nid),
            "has_size_results": d is not None,
            "GSF_radius_deg": (d["GSF"] if d else np.nan),
            "surround_extent_radius_deg": (d["surround_extent"] if d else np.nan),
            "SI": (d["SI"] if d else np.nan),
            "GSF_diameter_deg": (2.0 * d["GSF"] if d else np.nan),
            "surround_extent_diameter_deg": (2.0 * d["surround_extent"] if d else np.nan),
        })

    df_neurons = pd.DataFrame(rows).sort_values("neuron_id")

    # Loads summary
    load_rows = []
    for k, v in loaded.items():
        if not isinstance(v, dict):
            continue
        present = len(v.get("present_ids", [])) if "present_ids" in v else None
        missing = len(v.get("missing_ids", [])) if "missing_ids" in v else None
        load_rows.append({
            "name": k,
            "present": present,
            "missing": missing,
            "path": v.get("group_path", v.get("curves_path", "")),
        })
    df_loads = pd.DataFrame(load_rows)

    # Run/config sheet
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "h5_file": h5_file,
        "n_neuron_ids": int(len(neuron_ids)),
        "run_json": json.dumps(run, indent=2, sort_keys=True),
        "params_json": json.dumps(params, indent=2, sort_keys=True),
    }
    df_meta = pd.DataFrame([meta])

    ensure_dir(os.path.dirname(excel_path) or ".")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_neurons.to_excel(writer, sheet_name="neurons", index=False)
        df_loads.to_excel(writer, sheet_name="loads", index=False)
        df_meta.to_excel(writer, sheet_name="run", index=False)

    print(f">> Excel saved: {excel_path}")
