#!/usr/bin/env python3
"""
Pedagogical inductive-bias illustration (Figure 1 in the Perspective).

Three models are fit to noisy observations from sin(x) on x in [0, 10]:
  - ReLU MLP (flexible, piecewise-linear extrapolation)
  - Gaussian process with a periodic kernel
  - Linear regression on sin/cos features (oracle periodic basis)

Top row: N=8 training points. Bottom row: N=200.
The fourth column applies the same periodic Gaussian process under both data
regimes but changes the target function at deployment, showing that a model
can fit the development relationship well and still be systematically wrong
when the oracle function changes.

Writes figures/bias_vs_data_viz.png by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "mathtext.fontset": "dejavusans",
    }
)

np.random.seed(42)


def true_fun(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def get_weak_bias_model() -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="lbfgs",
        alpha=1e-5,
        random_state=1,
        max_iter=5000,
    )


def get_middle_bias_model() -> GaussianProcessRegressor:
    periodic = ExpSineSquared(
        length_scale=1.0,
        periodicity=2.0 * np.pi,
        length_scale_bounds="fixed",
        periodicity_bounds="fixed",
    )
    kernel = (
        C(1.0, constant_value_bounds="fixed") * periodic
        + WhiteKernel(noise_level=0.05, noise_level_bounds="fixed")
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        optimizer=None,
        random_state=1,
    )


def periodic_features(x: np.ndarray) -> np.ndarray:
    return np.hstack([np.sin(x), np.cos(x)])


def get_strong_bias_model():
    return make_pipeline(FunctionTransformer(periodic_features), LinearRegression())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    x_plot = np.linspace(0, 15, 500)[:, np.newaxis]
    train_mask = x_plot.flatten() <= 10
    ood_mask = x_plot.flatten() > 10

    # Model names and equations stay separate from the panel letters.
    title_a = "Flexible ReLU Model\n" + r"$f(x) \approx \sum w_i\,\mathrm{ReLU}(v_i x + b_i)$"
    title_b = "Periodic Prior\n" + r"$f \sim \mathcal{GP}(0, k_{\mathrm{periodic}})$"
    title_c = "Correctly Specified Features\n" + r"$f(x) = w_1\sin(x) + w_2\cos(x) + b$"
    title_d = "Target Function Shift\n" + r"$f \sim \mathcal{GP}(0, k_{\mathrm{periodic}})\quad\text{[Changed Oracle]}$"

    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5))
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.09,
        top=0.86,
        wspace=0.25,
        hspace=0.26,
    )
    # The 21-inch canvas is reduced to about 7 inches in the manuscript.
    # Scale the letters to match the 11-point labels in the other figures.
    for ax, letter in zip(axes[0], "ABCD"):
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=33, color="#222222",
                fontfamily=["Arial", "Liberation Sans", "DejaVu Sans"])
    data_scenarios = [("Sparse training data ($N=8$)", 8), ("Dense training data ($N=200$)", 200)]

    for row, (scenario_name, n_samples) in enumerate(data_scenarios):
        X_train = np.sort(np.random.uniform(0, 10, n_samples))[:, np.newaxis]
        y_train = true_fun(X_train).ravel() + np.random.normal(0, 0.1, n_samples)

        weak_model = get_weak_bias_model()
        weak_model.fit(X_train, y_train)
        y_weak = weak_model.predict(x_plot)

        gp_model = get_middle_bias_model()
        gp_model.fit(X_train, y_train)
        y_gp, y_gp_std = gp_model.predict(x_plot, return_std=True)

        strong_model = get_strong_bias_model()
        strong_model.fit(X_train, y_train)
        y_strong = strong_model.predict(x_plot)

        # Col A
        ax_weak = axes[row, 0]
        ax_weak.plot(x_plot[train_mask], y_weak[train_mask], color="#D32F2F", lw=2.5)
        ax_weak.plot(
            x_plot[ood_mask], y_weak[ood_mask], color="#D32F2F", lw=2.5, ls="--", alpha=0.85
        )
        ax_weak.plot(x_plot, true_fun(x_plot), color="green", lw=1.8, ls=":")
        ax_weak.scatter(X_train, y_train, color="black", s=32 if n_samples == 8 else 18, zorder=5)
        ax_weak.axvline(10, color="gray", ls="-", lw=1.5)
        ax_weak.set_ylim(-1.6, 2.05)
        if row == 0:
            t = ax_weak.set_title(title_a, fontsize=13.5, pad=14)
            t.set_bbox(dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE", edgecolor="#EF9A9A", alpha=0.9))

        # Col B
        ax_gp = axes[row, 1]
        ax_gp.plot(x_plot[train_mask], y_gp[train_mask], color="#6A1B9A", lw=2.5)
        ax_gp.plot(x_plot[ood_mask], y_gp[ood_mask], color="#6A1B9A", lw=2.5, ls="--", alpha=0.85)
        ax_gp.fill_between(
            x_plot.flatten(),
            y_gp - 2.0 * y_gp_std,
            y_gp + 2.0 * y_gp_std,
            color="#CE93D8",
            alpha=0.25,
            linewidth=0,
        )
        ax_gp.plot(x_plot, true_fun(x_plot), color="green", lw=1.8, ls=":")
        ax_gp.scatter(X_train, y_train, color="black", s=32 if n_samples == 8 else 18, zorder=5)
        ax_gp.axvline(10, color="gray", ls="-", lw=1.5)
        ax_gp.set_ylim(-1.6, 2.05)
        if row == 0:
            t = ax_gp.set_title(title_b, fontsize=13.5, pad=14)
            t.set_bbox(dict(boxstyle="round,pad=0.4", facecolor="#F3E5F5", edgecolor="#CE93D8", alpha=0.9))

        # Col C
        ax_strong = axes[row, 2]
        ax_strong.plot(x_plot[train_mask], y_strong[train_mask], color="#1976D2", lw=2.5)
        ax_strong.plot(
            x_plot[ood_mask], y_strong[ood_mask], color="#1976D2", lw=2.5, ls="--", alpha=0.85
        )
        ax_strong.plot(x_plot, true_fun(x_plot), color="green", lw=1.8, ls=":")
        ax_strong.scatter(X_train, y_train, color="black", s=32 if n_samples == 8 else 18, zorder=5)
        ax_strong.axvline(10, color="gray", ls="-", lw=1.5)
        ax_strong.set_ylim(-1.6, 2.05)
        if row == 0:
            t = ax_strong.set_title(title_c, fontsize=13.5, pad=14)
            t.set_bbox(dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", edgecolor="#90CAF9", alpha=0.9))

        # Col D
        ax_shift = axes[row, 3]
        shifted_target = true_fun(x_plot).ravel()
        shifted_target[ood_mask] += 0.75
        ax_shift.plot(x_plot[train_mask], y_gp[train_mask], color="#6A1B9A", lw=2.5)
        ax_shift.plot(
            x_plot[ood_mask],
            y_gp[ood_mask],
            color="#6A1B9A",
            lw=2.5,
            ls="--",
            alpha=0.85,
        )
        ax_shift.fill_between(
            x_plot.flatten(),
            y_gp - 2.0 * y_gp_std,
            y_gp + 2.0 * y_gp_std,
            color="#CE93D8",
            alpha=0.25,
            linewidth=0,
        )
        ax_shift.plot(
            x_plot[train_mask],
            true_fun(x_plot[train_mask]),
            color="green",
            lw=1.8,
            ls=":",
        )
        ax_shift.plot(
            x_plot[ood_mask],
            shifted_target[ood_mask],
            color="#EF6C00",
            lw=2.5,
            ls=":",
        )
        ax_shift.scatter(X_train, y_train, color="black", s=32 if n_samples == 8 else 18, zorder=5)
        ax_shift.axvline(10, color="gray", ls="-", lw=1.5)
        ax_shift.set_ylim(-1.6, 2.05)
        if row == 0:
            title_d_obj = ax_shift.set_title(title_d, fontsize=13.5, pad=14)
            title_d_obj.set_bbox(dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#FFCC80", alpha=0.9))

        # Axis labeling and ticks
        for col_idx in range(4):
            ax = axes[row, col_idx]
            ax.set_xlim(0, 15)
            ax.tick_params(axis="both", labelsize=11)
            if row == 1:
                ax.set_xlabel("$x$", fontsize=13)
            if col_idx == 0:
                ax.set_ylabel("$f(x)$", fontsize=13, labelpad=4)

        # Row label on the left
        axes[row, 0].annotate(
            scenario_name,
            xy=(-0.28, 0.5),
            xycoords="axes fraction",
            fontsize=13,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / "bias_vs_data_viz.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
