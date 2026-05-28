import h5py
import numpy as np

def save_experiment_config(
    h5_file,
    group_path,
    *,
    args_str=None,
    array_fields=None,
    attr_fields=None,
    overwrite=False,
):
    """
    Save structured experiment config alongside a legacy arguments string.
    """
    array_fields = {} if array_fields is None else array_fields
    attr_fields = {} if attr_fields is None else attr_fields

    config_path = group_path + "/config"

    with h5py.File(h5_file, "a") as f:
        grp = f[group_path]
        if args_str is not None:
            grp.attrs["arguments"] = args_str

        for k, v in attr_fields.items():
            grp.attrs[k] = v

        if config_path in f and overwrite:
            del f[config_path]
        if config_path not in f:
            cfg = f.create_group(config_path)
        else:
            cfg = f[config_path]

        for k, v in array_fields.items():
            arr = np.asarray(v)
            if k in cfg:
                del cfg[k]
            cfg.create_dataset(k, data=arr)