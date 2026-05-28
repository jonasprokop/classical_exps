import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.analysis.tools   import compute_freeman_modulation_index
from classical_exps.core.simulations.freeman2013_naturalistic_texture_modulation.analysis.loaders import load_texture_noise_responses, load_selected_lowlevel_family_stats, load_texture_noise_image_statistics





def _safe_pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if np.sum(mask) < 3:
        return np.nan

    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan

    return float(np.corrcoef(x[mask], y[mask])[0, 1])



def texture_noise_lowlevel_family_backtest(
    h5_file,
    neuron_ids,
    wanted_fam_order=None,
    save_dir="/project/results/texture_noise_response/validation_plots/lowlevel_backtest/",
):
    """
    Analysis branch only.

    Reads:
        - model texture/noise responses from HDF5
        - five low-level family image stats from HDF5

    Plots:
        - one barplot of correlations
        - one 2x3 panel showing each low-level stat vs family-level model MI
    """
    os.makedirs(save_dir, exist_ok=True)

    all_tex_resp, all_noise_resp, response_family_ids = load_texture_noise_responses(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
    )

    mi_neuron_family = compute_freeman_modulation_index(
        all_tex_resp,
        all_noise_resp,
    )

    mean_mi_family = np.mean(mi_neuron_family, axis=0)

    stat_family_ids, feature_names, family_features = load_selected_lowlevel_family_stats(
        h5_file=h5_file,
    )

    response_family_ids_int = np.asarray(response_family_ids, dtype=int)

    if not np.array_equal(response_family_ids_int, stat_family_ids):
        raise ValueError(
            "Family order mismatch between response results and low-level image stats:\n"
            f"responses: {response_family_ids_int}\n"
            f"stats:     {stat_family_ids}"
        )

    family_ids = response_family_ids_int.copy()

    pretty_names = {
        "delta_rms_contrast": "Δ RMS contrast",
        "delta_mean_luminance": "Δ mean luminance",
        "texture_noise_rms_difference": "Texture-noise RMS difference",
        "delta_spectral_centroid": "Δ spectral centroid",
        "delta_high_freq_fraction": "Δ high-frequency fraction",
    }

    if wanted_fam_order is not None:
        dict_old_order = {
            str(fam): pos for pos, fam in enumerate(family_ids.astype(str))
        }

        wanted_fam_order = np.asarray(wanted_fam_order).astype(str)
        new_order = np.array([dict_old_order[fam] for fam in wanted_fam_order])

        family_ids = family_ids[new_order]
        mean_mi_family = mean_mi_family[new_order]
        family_features = family_features[new_order]

    rs = np.asarray(
        [
            _safe_pearsonr(family_features[:, i], mean_mi_family)
            for i in range(len(feature_names))
        ],
        dtype=float,
    )

    # ------------------------------------------------------------
    # Joint variance explained by all five low-level predictors
    # ------------------------------------------------------------
    X_lowlevel = family_features
    y_mi = mean_mi_family

    joint_r2, joint_adj_r2, y_hat_insample, valid_mask = _multiple_regression_r2(
        X_lowlevel,
        y_mi,
    )

    joint_loocv_r2, y_hat_loocv, valid_mask_cv = _loocv_regression_r2(
        X_lowlevel,
        y_mi,
    )

    # ------------------------------------------------------------
    # CSV table
    # ------------------------------------------------------------
    csv_path = os.path.join(save_dir, "lowlevel_5feature_family_table.csv")

    with open(csv_path, "w") as f:
        f.write(
            "family,model_MI,"
            + ",".join(feature_names)
            + "\n"
        )

        for fam_i, fam in enumerate(family_ids):
            vals = family_features[fam_i]
            f.write(
                f"{fam},{mean_mi_family[fam_i]:.8f},"
                + ",".join(f"{v:.8f}" for v in vals)
                + "\n"
            )

    corr_csv_path = os.path.join(save_dir, "lowlevel_5feature_correlations.csv")

    with open(corr_csv_path, "w") as f:
        f.write("feature,pretty_name,pearson_r,abs_pearson_r\n")

        for name, r in sorted(
            zip(feature_names, rs),
            key=lambda t: -abs(t[1]) if np.isfinite(t[1]) else -np.inf,
        ):
            f.write(
                f"{name},"
                f"{pretty_names.get(name, name)},"
                f"{r:.8f},"
                f"{abs(r):.8f}\n"
            )

    # ------------------------------------------------------------
    # Correlation barplot
    # ------------------------------------------------------------
    order = np.argsort(-np.abs(rs))

    names_sorted = [pretty_names.get(feature_names[i], feature_names[i]) for i in order]
    rs_sorted = rs[order]

    fig, ax = plt.subplots(figsize=(6.2, 3.0))

    y = np.arange(len(names_sorted))

    ax.barh(
        y,
        rs_sorted,
        color="0.45",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel("Pearson r with family model MI", fontsize=11)
    ax.set_title("Low-level predictors of model modulation", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "lowlevel_5feature_correlation_barplot.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Scatter panel
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 6.2),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for i, name in enumerate(feature_names):
        ax = axes[i]

        x = family_features[:, i]
        y = mean_mi_family
        r = rs[i]

        ax.scatter(
            x,
            y,
            s=42,
            facecolor="0.45",
            edgecolor="black",
            linewidth=0.7,
        )

        for j, fam in enumerate(family_ids):
            ax.text(
                x[j],
                y[j],
                str(fam),
                fontsize=7,
                ha="center",
                va="bottom",
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlabel(pretty_names.get(name, name), fontsize=9)
        ax.set_ylabel("Family model MI", fontsize=9)
        ax.set_title(f"r = {r:.2f}", fontsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].axis("off")

    plt.savefig(
        os.path.join(save_dir, "lowlevel_5feature_vs_family_MI_panel.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Joint model: observed vs predicted MI
    # ------------------------------------------------------------
    valid_family_ids = family_ids[valid_mask]
    valid_y = mean_mi_family[valid_mask]

    fig, ax = plt.subplots(figsize=(4.8, 4.4))

    ax.scatter(
        valid_y,
        y_hat_insample,
        s=46,
        facecolor="0.45",
        edgecolor="black",
        linewidth=0.8,
        label="in-sample fit",
    )

    if y_hat_loocv is not None:
        ax.scatter(
            valid_y,
            y_hat_loocv,
            s=46,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            label="LOOCV prediction",
        )

    for i, fam in enumerate(valid_family_ids):
        ax.text(
            valid_y[i],
            y_hat_insample[i],
            str(fam),
            fontsize=7,
            ha="center",
            va="bottom",
        )

    lim_min = min(
        np.min(valid_y),
        np.min(y_hat_insample),
        np.min(y_hat_loocv) if y_hat_loocv is not None else np.min(y_hat_insample),
    )
    lim_max = max(
        np.max(valid_y),
        np.max(y_hat_insample),
        np.max(y_hat_loocv) if y_hat_loocv is not None else np.max(y_hat_insample),
    )

    pad = 0.08 * max(abs(lim_min), abs(lim_max), 1e-6)

    ax.plot(
        [lim_min - pad, lim_max + pad],
        [lim_min - pad, lim_max + pad],
        linestyle="--",
        color="black",
        linewidth=1,
    )

    ax.set_xlim(lim_min - pad, lim_max + pad)
    ax.set_ylim(lim_min - pad, lim_max + pad)

    ax.set_xlabel("Observed family model MI", fontsize=10)
    ax.set_ylabel("Predicted family model MI", fontsize=10)
    ax.set_title(
        f"All 5 low-level predictors\n"
        f"R²={joint_r2:.2f}, adj. R²={joint_adj_r2:.2f}, LOOCV R²={joint_loocv_r2:.2f}",
        fontsize=10,
    )

    ax.legend(frameon=False, fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "lowlevel_5feature_joint_prediction.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("--------------------------------------")
    print("Five-feature low-level backtest saved:")
    print(f"    > {save_dir}")
    print(f"    > family table: {csv_path}")
    print(f"    > correlation table: {corr_csv_path}")
    print("    > correlations:")

    for name, r in sorted(
        zip(feature_names, rs),
        key=lambda t: -abs(t[1]) if np.isfinite(t[1]) else -np.inf,
    ):
        print(f"      {pretty_names.get(name, name)}: r = {r:.3f}")

    
    print("    > joint low-level model:")
    print(f"      raw R²      = {joint_r2:.3f} ({100 * joint_r2:.1f}% variance)")
    print(f"      adjusted R² = {joint_adj_r2:.3f}")
    print(f"      LOOCV R²    = {joint_loocv_r2:.3f}")


    print("--------------------------------------")


def _standardize_columns(X, eps=1e-12):
    """
    Z-score columns. Constant columns become zeros.
    """
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)

    sd_safe = np.where(sd < eps, 1.0, sd)

    return (X - mu) / sd_safe, mu, sd_safe


def _multiple_regression_r2(X, y):
    """
    Ordinary least squares R² with intercept.

    X: [n_observation, n_feature]
    y: [n_observation]
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    n, p = X.shape

    if n <= p + 1:
        return np.nan, np.nan, np.full_like(y, np.nan), mask

    Xz, _, _ = _standardize_columns(X)
    yz = y - np.mean(y)

    X_design = np.column_stack([np.ones(n), Xz])

    beta, *_ = np.linalg.lstsq(X_design, yz, rcond=None)
    y_hat = X_design @ beta

    ss_res = np.sum((yz - y_hat) ** 2)
    ss_tot = np.sum((yz - np.mean(yz)) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

    return float(r2), float(adjusted_r2), y_hat, mask


def _loocv_regression_r2(X, y):
    """
    Leave-one-out cross-validated R².

    This is the useful one with n=15.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    n, p = X.shape

    if n <= p + 1:
        return np.nan, None, mask

    y_pred = np.full(n, np.nan, dtype=float)

    for held_out in range(n):
        train = np.ones(n, dtype=bool)
        train[held_out] = False

        X_train = X[train]
        y_train = y[train]

        X_test = X[held_out:held_out + 1]

        X_train_z, mu, sd = _standardize_columns(X_train)
        X_test_z = (X_test - mu) / sd

        y_mu = np.mean(y_train)
        y_train_centered = y_train - y_mu

        X_design_train = np.column_stack([np.ones(np.sum(train)), X_train_z])
        X_design_test = np.column_stack([np.ones(1), X_test_z])

        beta, *_ = np.linalg.lstsq(X_design_train, y_train_centered, rcond=None)

        y_pred[held_out] = float(X_design_test @ beta + y_mu)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    loocv_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(loocv_r2), y_pred, mask


def texture_noise_image_statistics_vs_model_mi(
    h5_file,
    neuron_ids,
    wanted_fam_order=None,
    save_dir="/project/results/texture_noise_response/validation_plots/image_stats_vs_mi/",
):
    """
    Backtest whether family-level model MI correlates with low-level image statistics.

    Makes:
        1. correlation bar plot: image statistic vs family MI
        2. scatter plots for top correlated statistics
        3. CSV table of all correlations
    """
    os.makedirs(save_dir, exist_ok=True)

    all_tex_resp, all_noise_resp, response_family_ids = load_texture_noise_responses(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
    )

    mi_neuron_family = compute_freeman_modulation_index(
        all_tex_resp,
        all_noise_resp,
    )

    mean_mi_family = np.mean(mi_neuron_family, axis=0)

    stat_family_ids, feature_names, family_features = (
        load_texture_noise_image_statistics(h5_file)
    )

    response_family_ids_int = np.asarray(response_family_ids, dtype=int)

    if not np.array_equal(response_family_ids_int, stat_family_ids):
        raise ValueError(
            "Family order mismatch between responses and image statistics:\n"
            f"responses: {response_family_ids_int}\n"
            f"stats:     {stat_family_ids}"
        )

    family_ids = response_family_ids_int.copy()

    if wanted_fam_order is not None:
        dict_old_order = {
            str(fam): pos for pos, fam in enumerate(family_ids.astype(str))
        }
        wanted_fam_order = np.asarray(wanted_fam_order).astype(str)
        new_order = np.array([dict_old_order[fam] for fam in wanted_fam_order])

        family_ids = family_ids[new_order]
        mean_mi_family = mean_mi_family[new_order]
        family_features = family_features[new_order]

    # ------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------
    rows = []

    for feat_i, feat_name in enumerate(feature_names):
        x = family_features[:, feat_i]
        y = mean_mi_family

        mask = np.isfinite(x) & np.isfinite(y)

        if np.sum(mask) >= 3 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
            r = float(np.corrcoef(x[mask], y[mask])[0, 1])
        else:
            r = np.nan

        rows.append((feat_name, r))

    rows_sorted = sorted(
        rows,
        key=lambda t: -abs(t[1]) if np.isfinite(t[1]) else -np.inf,
    )

    csv_path = os.path.join(save_dir, "image_stat_correlations_with_model_MI.csv")
    with open(csv_path, "w") as f:
        f.write("feature,pearson_r,abs_pearson_r\n")
        for feat_name, r in rows_sorted:
            abs_r = abs(r) if np.isfinite(r) else np.nan
            f.write(f"{feat_name},{r:.8f},{abs_r:.8f}\n")

    # ------------------------------------------------------------
    # 1. Correlation bar plot
    # ------------------------------------------------------------
    plot_names = [r[0] for r in rows_sorted]
    plot_rs = np.asarray([r[1] for r in rows_sorted], dtype=float)

    fig_height = max(5.0, 0.28 * len(plot_names))

    fig, ax = plt.subplots(figsize=(7.2, fig_height))

    y_pos = np.arange(len(plot_names))

    ax.barh(
        y_pos,
        plot_rs,
        color="0.45",
        edgecolor="black",
        linewidth=0.7,
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_names, fontsize=8)
    ax.invert_yaxis()

    ax.set_xlabel("Pearson r with family-level model MI", fontsize=11)
    ax.set_title("Low-level image statistics vs model modulation", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "image_stat_correlations_with_model_MI.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # 2. Scatter plots for top correlated features
    # ------------------------------------------------------------
    top_rows = [
        r for r in rows_sorted
        if np.isfinite(r[1])
    ][:6]

    for feat_name, r in top_rows:
        feat_i = feature_names.index(feat_name)
        x = family_features[:, feat_i]
        y = mean_mi_family

        fig, ax = plt.subplots(figsize=(4.6, 3.8))

        ax.scatter(
            x,
            y,
            s=42,
            facecolor="0.45",
            edgecolor="black",
            linewidth=0.7,
        )

        for i, fam in enumerate(family_ids):
            ax.text(
                x[i],
                y[i],
                str(fam),
                fontsize=8,
                ha="center",
                va="bottom",
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlabel(feat_name, fontsize=10)
        ax.set_ylabel("Family-level model MI", fontsize=10)
        ax.set_title(f"{feat_name} vs MI (r = {r:.2f})", fontsize=11)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        safe_name = feat_name.replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"scatter_{safe_name}_vs_MI.png"),
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

    print("--------------------------------------")
    print("Image-statistics vs model-MI backtest saved:")
    print(f"    > {save_dir}")
    print(f"    > correlation table: {csv_path}")

    if len(rows_sorted) > 0:
        print("    > top correlations:")
        for feat_name, r in rows_sorted[:8]:
            print(f"      {feat_name}: r = {r:.3f}")

    print("--------------------------------------")


def texture_noise_response_validation_plots(
    h5_file,
    neuron_ids,
    wanted_fam_order=None,
    save_dir="/project/results/texture_noise_response/validation_plots/",
    eps=1e-8,
):
    """
    Validation plots for static Freeman texture/noise experiment.

    Makes:
        1. texture-vs-noise response scatter
        2. MI vs response magnitude
        3. sample-wise MI consistency by family
        4. neuron-level MI distribution, cleaner than results_3
    """
    os.makedirs(save_dir, exist_ok=True)

    all_tex_resp, all_noise_resp, family_ids = load_texture_noise_responses(
        h5_file=h5_file,
        neuron_ids=neuron_ids,
    )

    # [n_neuron, n_family]
    tex_mean_nf = np.mean(all_tex_resp, axis=2)
    noise_mean_nf = np.mean(all_noise_resp, axis=2)

    mi_neuron_family = compute_freeman_modulation_index(
        all_tex_resp,
        all_noise_resp,
        eps=eps,
    )

    mean_mi_family = np.mean(mi_neuron_family, axis=0)
    mean_resp_family = np.mean((tex_mean_nf + noise_mean_nf) / 2.0, axis=0)
    mean_mi_neuron = np.mean(mi_neuron_family, axis=1)

    # Diagnostic only: sample-wise MI, then averaged over neurons.
    sample_mi = (all_tex_resp - all_noise_resp) / (all_tex_resp + all_noise_resp + eps)
    sample_mi_family = np.mean(sample_mi, axis=0)  # [n_family, n_sample]

    if wanted_fam_order is not None:
        dict_old_order = {fam: pos for pos, fam in enumerate(family_ids)}
        wanted_fam_order = np.asarray(wanted_fam_order).astype(str)
        new_order = np.array([dict_old_order[fam] for fam in wanted_fam_order])

        family_ids = family_ids[new_order]
        tex_mean_nf = tex_mean_nf[:, new_order]
        noise_mean_nf = noise_mean_nf[:, new_order]
        mi_neuron_family = mi_neuron_family[:, new_order]
        mean_mi_family = mean_mi_family[new_order]
        mean_resp_family = mean_resp_family[new_order]
        sample_mi_family = sample_mi_family[new_order]

    x = np.arange(len(family_ids))

    # ------------------------------------------------------------
    # 1. Texture vs noise response scatter
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    ax.scatter(
        noise_mean_nf.ravel(),
        tex_mean_nf.ravel(),
        s=13,
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.55,
    )

    lim_max = max(
        float(np.max(noise_mean_nf)),
        float(np.max(tex_mean_nf)),
        1e-6,
    )

    ax.plot(
        [0, lim_max * 1.03],
        [0, lim_max * 1.03],
        linestyle="--",
        color="0.4",
        linewidth=1.0,
    )

    ax.set_xlim(0, lim_max * 1.03)
    ax.set_ylim(0, lim_max * 1.03)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean noise response", fontsize=11)
    ax.set_ylabel("Mean texture response", fontsize=11)
    ax.set_title("Texture vs noise responses", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "texture_vs_noise_response_scatter.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # 2. Family MI vs response magnitude
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    ax.scatter(
        mean_resp_family,
        mean_mi_family,
        s=42,
        facecolor="0.35",
        edgecolor="black",
        linewidth=0.7,
    )

    for i, fam in enumerate(family_ids):
        ax.text(
            mean_resp_family[i],
            mean_mi_family[i],
            str(fam),
            fontsize=8,
            ha="center",
            va="bottom",
        )

    r = np.corrcoef(mean_resp_family, mean_mi_family)[0, 1] if len(mean_resp_family) >= 2 else np.nan

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean response magnitude", fontsize=11)
    ax.set_ylabel("Family modulation index", fontsize=11)
    ax.set_title(f"MI vs response magnitude (r = {r:.2f})", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "family_MI_vs_response_magnitude.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # 3. Sample-wise MI consistency by family
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 2.9))

    ax.boxplot(
        [sample_mi_family[i] for i in range(len(family_ids))],
        positions=x,
        widths=0.55,
        patch_artist=True,
        boxprops=dict(facecolor="0.85", edgecolor="black", linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        flierprops=dict(
            marker="o",
            markersize=2.5,
            markerfacecolor="white",
            markeredgecolor="black",
            alpha=0.7,
        ),
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(family_ids, fontsize=9)
    ax.set_xlabel("Texture family", fontsize=11)
    ax.set_ylabel("Sample-wise MI", fontsize=11)
    ax.set_title("Sample consistency of modulation", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "samplewise_MI_by_family.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # 4. Neuron-level MI distribution
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.1, 3.5))

    weights = np.ones_like(mean_mi_neuron, dtype=float) / len(mean_mi_neuron)

    ax.hist(
        mean_mi_neuron,
        bins=12,
        weights=weights,
        color="0.55",
        edgecolor="black",
        linewidth=0.8,
    )

    meanval = float(np.mean(mean_mi_neuron))

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, label="zero")
    ax.axvline(meanval, color="black", linewidth=1.4, label=f"mean = {meanval:.3f}")

    ax.set_xlabel("Mean modulation index per neuron", fontsize=11)
    ax.set_ylabel("Fraction of neurons", fontsize=11)
    ax.set_title("Neuron-level modulation distribution", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "neuron_level_MI_distribution.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("--------------------------------------")
    print("Texture/noise validation plots saved:")
    print(f"    > {save_dir}")
    print(f"    > Mean family MI: {np.mean(mean_mi_family):.4f}")
    print(f"    > Mean neuron MI: {np.mean(mean_mi_neuron):.4f}")
    print(f"    > MI-response magnitude correlation: {r:.4f}")
    print("--------------------------------------")





    






