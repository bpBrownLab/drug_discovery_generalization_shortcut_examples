#!/usr/bin/env python3
"""
Plot the DUD-E ligand-only control figure (Figure 4 in the Perspective).

Reads fold-level EF means from dude_ligand_only_control.py (or bundled
data/dude_fold_means.csv), aggregates to mean +/- sample standard deviation
across the 15 repeat/fold means, and overlays DrugCLIP, Glide-SP,
and Vina reference values from Jia et al.

Writes figures/toy_dude_fig2c_replot_ef.png by default.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# Jia et al. Table S6 (training ligand exclusions on DUD-E): EF1% values.
# DrugCLIP retains both pocket and ligand inputs in all four conditions.
DRUGCLIP_LIGAND_EF = {
    "ecfp4<0.9": 24.08,
    "ecfp4<0.6": 25.27,
    "ecfp4<0.3": 19.10,
    "scaffold0": 19.97,
}

DEFAULT_GLIDE_EF = 16.18  # Jia et al. Table S4
DEFAULT_VINA_EF = 7.32


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    xs2 = [x for x in xs if x == x]
    if not xs2:
        return float("nan"), float("nan")
    m = float(sum(xs2) / len(xs2))
    v = float(sum((x - m) ** 2 for x in xs2) / max(1, len(xs2) - 1))
    return m, v**0.5


def _parse_condition_order(conditions: List[str]) -> List[str]:
    out: List[str] = []
    if "baseline" in conditions:
        out.append("baseline")

    thrs = []
    for c in conditions:
        if c.startswith("ecfp4<"):
            s = c.split("<", 1)[1]
            if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s) is None:
                continue
            thrs.append((float(s), c))
    for _, c in sorted(thrs, key=lambda x: x[0], reverse=True):
        out.append(c)

    if "scaffold0" in conditions:
        out.append("scaffold0")

    for c in sorted(set(conditions) - set(out)):
        out.append(c)
    return out


def load_fold_means(path: Path) -> Dict[str, List[float]]:
    uniq: Dict[Tuple[int, int, str], float] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rep = int(row["rep"])
            fold = int(row["fold"])
            cond = row["condition"]
            ef = float(row["ef_mean_targets"])
            uniq[(rep, fold, cond)] = ef

    conditions = sorted({k[2] for k in uniq})
    order = _parse_condition_order(conditions)
    ef_by_cond: Dict[str, List[float]] = {c: [] for c in order}
    for (_rep, _fold, cond), ef in uniq.items():
        if cond in ef_by_cond:
            ef_by_cond[cond].append(ef)
    return ef_by_cond


def plot_figure(
    ef_by_cond: Dict[str, List[float]],
    *,
    glide_ef: float,
    vina_ef: float,
    out_png: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "black",
            "figure.dpi": 300,
        }
    )

    lig_order = [
        c for c in ["ecfp4<0.9", "ecfp4<0.6", "ecfp4<0.3", "scaffold0"] if c in ef_by_cond
    ]
    if not lig_order:
        lig_order = list(ef_by_cond.keys())

    xlabels = []
    for c in lig_order:
        if c.startswith("ecfp4<"):
            xlabels.append("ECFP4\n" + c.split("<", 1)[1])
        elif c == "scaffold0":
            xlabels.append("No shared\nMurcko scaffold")
        else:
            xlabels.append(c)

    xs = list(range(len(lig_order)))
    lig_means, lig_stds = [], []
    for c in lig_order:
        m, s = _mean_std(ef_by_cond[c])
        lig_means.append(m)
        lig_stds.append(s)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    eb_kwargs = {
        "capsize": 2.5,
        "capthick": 0.8,
        "elinewidth": 0.8,
        "linewidth": 1.2,
        "markersize": 5,
        "markeredgecolor": "black",
        "markeredgewidth": 0.6,
    }

    ys_dc = [DRUGCLIP_LIGAND_EF.get(c, float("nan")) for c in lig_order]
    # Scaffold exclusion is a separate condition, not zero ECFP4 similarity.
    # Connect only the fingerprint cutoffs; plot scaffold results separately.
    groups = [
        [i for i, c in enumerate(lig_order) if c != "scaffold0"],
        [i for i, c in enumerate(lig_order) if c == "scaffold0"],
    ]
    labeled = False
    for indices in groups:
        if not indices:
            continue
        ax.errorbar(
            [xs[i] for i in indices],
            [lig_means[i] for i in indices],
            yerr=[lig_stds[i] for i in indices],
            fmt="-o" if len(indices) > 1 else "o",
            color="#666666",
            label="Ligand-only control (this work)" if not labeled else None,
            **eb_kwargs,
        )
        ax.plot(
            [xs[i] for i in indices],
            [ys_dc[i] for i in indices],
            "-s" if len(indices) > 1 else "s",
            color="#5ab4ac",
            linewidth=1.2,
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="DrugCLIP (pocket + ligand)" if not labeled else None,
        )
        labeled = True

    if all(groups):
        boundary = (xs[groups[0][-1]] + xs[groups[1][0]]) / 2
        ax.axvline(boundary, ymax=0.84, color="#b5b5b5", linestyle=":", linewidth=0.8, zorder=0)

    ax.axhline(glide_ef, color="#cc79a7", linestyle="--", linewidth=1.5, alpha=0.9)
    ax.text(xs[0], glide_ef + 0.8, "Glide-SP", color="#cc79a7", fontsize=6, ha="left", va="bottom", fontweight="bold")

    ax.axhline(vina_ef, color="#8da0cb", linestyle="--", linewidth=1.5, alpha=0.9)
    ax.text(xs[0], vina_ef + 0.8, "Vina", color="#8da0cb", fontsize=6, ha="left", va="bottom", fontweight="bold")

    ax.set_ylim(0, 50)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Training ligand filter")
    ax.set_ylabel("EF1%")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1))
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fold-means",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data/dude_fold_means.csv",
        help="Fold-level means CSV from dude_ligand_only_control.py.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--glide-ef", type=float, default=DEFAULT_GLIDE_EF)
    ap.add_argument("--vina-ef", type=float, default=DEFAULT_VINA_EF)
    args = ap.parse_args()

    if not args.fold_means.exists():
        raise FileNotFoundError(
            f"Missing {args.fold_means}. Run dude_ligand_only_control.py first "
            "or use data/dude_fold_means.csv."
        )

    ef_by_cond = load_fold_means(args.fold_means)
    out_png = args.out_dir / "toy_dude_fig2c_replot_ef.png"
    plot_figure(ef_by_cond, glide_ef=args.glide_ef, vina_ef=args.vina_ef, out_png=out_png)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
