#!/usr/bin/env python3
"""Draw the split audit (Figure 2) and scaffold sensitivity (Figure S1).

All summaries give each of the ten CATH folds equal weight. Figure 2A shows
quartile boxes and all fold observations. Paired model differences are also
exported for the same fold, threshold, and partition. Figure 3 and Figure S2
are drawn separately by plot_full_data.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

from source_data import HERE, REVISIONS, INPUTS, sha256, source_data
from paths import DATA, SCRIPTS

FIGURE_DIR = HERE / 'figures'
SPLITS = [
    ('Random validation', '#355F85', 'o', '-'),
    ('Composite validation', '#B87636', 's', '--'),
    ('CATH-LSO test', '#79516F', '^', '-.'),
]
CDFS = HERE / 'figure2_chemistry_source_cdfs.csv'


def set_style() -> None:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 8.0, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'xtick.major.size': 3,
        'ytick.major.size': 3, 'axes.edgecolor': '#333333',
        'text.color': '#222222', 'axes.labelcolor': '#222222',
        'xtick.color': '#333333', 'ytick.color': '#333333',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'jmc-two-main-figures-2026-09-11',
        'axes.unicode_minus': False, 'savefig.facecolor': 'white',
    })


def axes_style(ax: plt.Axes, letter: str, title: str, grid: str = 'x') -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color='#E4E4E4', linewidth=0.4)
    ax.tick_params(pad=3)
    ax.set_title(title, loc='left', pad=12, fontweight='normal')
    ax.text(-0.12, 1.055, letter, transform=ax.transAxes,
            ha='right', va='bottom', fontweight='bold', fontsize=11)


def save_figure(fig: plt.Figure, stem: str, title: str) -> dict:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / stem
    fig.savefig(output.with_suffix('.pdf'), metadata={
        'Title': title, 'Author': 'B. P. Brown', 'CreationDate': None, 'ModDate': None})
    fig.savefig(output.with_suffix('.svg'), metadata={'Date': None})
    fig.savefig(output.with_suffix('.png'), dpi=600)
    size = fig.get_size_inches().tolist()
    # Check text extents at the actual publication dimensions, not a tight crop
    # that could silently change figure size to accommodate overflowing labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
            outside.append(text.get_text())
    if outside:
        raise ValueError(f'Text outside {stem}: {outside}')
    plt.close(fig)
    return {'size_inches': size, 'png_dpi': 600, 'text_inside_canvas': True,
            'outputs': {ext: {'path': str(output.with_suffix(ext).relative_to(REVISIONS)),
                              'sha256': sha256(output.with_suffix(ext))}
                        for ext in ['.pdf', '.svg', '.png']}}


def draw_split_audit(reuse: pd.DataFrame, neighbor: pd.DataFrame) -> dict:
    set_style()
    fig = plt.figure(figsize=(7.0, 4.2))
    ax = fig.add_axes([0.14, 0.235, 0.36, 0.665])
    bx = fig.add_axes([0.62, 0.235, 0.36, 0.665])
    for panel, letter, title, grid in [
        (ax, 'A', 'Pocket and chemical overlap', 'x'),
        (bx, 'B', 'Similarity to the nearest ligand', 'y'),
    ]:
        panel.spines[['top', 'right']].set_visible(False)
        panel.set_axisbelow(True)
        panel.grid(axis=grid, color='#DDDDDD', linewidth=0.45, zorder=0)
        panel.tick_params(pad=3, labelsize=9)
        panel.set_title(title, fontsize=10, fontweight='bold', pad=12)
        panel.text(-0.12, 1.04, letter, transform=panel.transAxes,
                   ha='right', va='bottom', fontsize=11)
    identity_keys = ['pocket', 'canonical_isomeric', 'murcko_typed']
    labels = ['Same pocket\nsequence', 'Same ligand\nstructure', 'Same Murcko\nscaffold']
    centers = [2.0, 1.0, 0.0]
    offsets = [0.26, 0.0, -0.26]
    means = []
    box_summaries = []
    jitter = np.array([-0.030, 0.014, -0.004, 0.030, -0.018,
                        0.004, -0.025, 0.022, -0.011, 0.009]) * 2
    for identity, center in zip(identity_keys, centers):
        for condition, (_, color, marker, _) in enumerate(SPLITS):
            values = reuse.loc[reuse.identity.eq(identity) & reuse.condition.eq(condition)].value.to_numpy()
            assert len(values) == 10 and np.isfinite(values).all()
            y = center + offsets[condition]
            ax.boxplot([values], positions=[y], widths=0.22, vert=False,
                       whis=1.5, patch_artist=True, showfliers=False,
                       manage_ticks=False,
                       boxprops={'facecolor': to_rgba(color, 0.55),
                                 'edgecolor': 'black', 'linewidth': 0.95},
                       medianprops={'color': 'black', 'linewidth': 0.9},
                       whiskerprops={'color': 'black', 'linewidth': 0.85},
                       capprops={'color': 'black', 'linewidth': 0.85})
            ax.scatter(values, y + jitter, s=18, marker=marker,
                       facecolors=to_rgba(color, 0.95), edgecolors='black',
                       linewidths=0.45, zorder=4)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            iqr = q3 - q1
            within = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
            box_summaries.append({'identity': identity, 'condition': condition,
                'n_folds': len(values), 'q1_percent': float(q1),
                'median_percent': float(median), 'q3_percent': float(q3),
                'whisker_low_percent': float(within.min()),
                'whisker_high_percent': float(within.max())})
            means.append({'identity': identity, 'condition': condition,
                          'mean_percent': float(values.mean()), 'sd_percent': float(values.std(ddof=1))})
    ax.set_yticks(centers, labels)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.set_ylim(-0.60, 2.5)
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Protein–ligand examples\nwith a match (%)', labelpad=6, fontsize=9.5)

    cdfs = pd.read_csv(CDFS)
    assert len(cdfs) == 33033
    for condition, (label, color, _, linestyle) in enumerate(SPLITS):
        rows = cdfs.loc[cdfs.condition.eq(condition)]
        individual = rows.loc[rows.cath.ne('macro_mean')]
        assert individual.cath.nunique() == 10
        curves = []
        for _, frame in individual.groupby('cath'):
            frame = frame.sort_values('tanimoto')
            assert len(frame) == 1001
            assert (np.diff(frame.cumulative_fraction) >= -1e-12).all()
            curves.append(frame.cumulative_fraction.to_numpy())
            bx.step(frame.tanimoto, 100 * frame.cumulative_fraction,
                    where='post', color=color, alpha=0.20, linewidth=0.5, zorder=1)
        macro = rows.loc[rows.cath.eq('macro_mean')].sort_values('tanimoto')
        assert np.allclose(np.mean(curves, axis=0), macro.cumulative_fraction, atol=1e-11)
        bx.step(macro.tanimoto, 100 * macro.cumulative_fraction, where='post',
                color=color, linestyle=linestyle, linewidth=1.5, label=label, zorder=3)
    bx.set_xlim(0, 1.015)
    bx.set_ylim(0, 102.5)
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.50', '0.75', '1.00'])
    bx.set_yticks([0, 25, 50, 75, 100])
    bx.set_xlabel('Morgan Tanimoto similarity', labelpad=6, fontsize=9.5)
    bx.set_ylabel('Ligands at or below\nthis similarity (%)', labelpad=5, fontsize=9.5)
    handles = [Line2D([], [], color=color, marker=marker, linestyle=line,
                      linewidth=1.0, markersize=5, markeredgecolor='black',
                      markeredgewidth=0.3, label=label)
               for label, color, marker, line in SPLITS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.52, 0.012),
               frameon=False, borderaxespad=0, borderpad=0, ncol=3,
               handlelength=2.0, columnspacing=1.6,
               handletextpad=0.5, fontsize=9)
    result = save_figure(fig, 'figure2_cath_split_audit', 'Protein and chemical overlap under CATH-LSO')
    result.update({'overlap_means': means,
                   'overlap_box_summaries': box_summaries,
                   'boxplot_definition': 'Median line, first-to-third quartile box, whiskers to extreme observations within 1.5 IQR; all ten fold values shown. Black outlines.',
                   'nearest_similarity_means': neighbor.groupby('condition').value.mean().to_dict(),
                   'reference_sets': 'Validation versus training; CATH-LSO versus training plus validation.',
                   'cdf_summary': 'Equal-weight mean of ten empirical CDFs on a 0.001 grid; individual folds shown. Displayed as percentages, without changing the source fractions.'})
    return result


def model_differences(auc: pd.DataFrame) -> pd.DataFrame:
    paired = auc.loc[auc.model.isin(['deepdta', 'transformerdta'])].pivot(
        index=['cath', 'development', 'evaluation', 'threshold'], columns='model', values='value')
    assert len(paired) == 80 and not paired.isna().any().any()
    paired['difference'] = paired.transformerdta - paired.deepdta
    return paired.reset_index()






def draw_scaffold_sensitivity() -> dict:
    set_style()
    chem = pd.read_csv(INPUTS['chemistry_overlap'])
    selected = chem.loc[chem.development_split.eq('random')
        & chem.evaluation.eq('cath_lso_test') & chem.reference.eq('development')
        & chem.identity.isin(['murcko_typed', 'murcko_generic'])
        & chem.weighting.isin(['rows', 'unique_keys'])].copy()
    assert len(selected) == 40
    selected['novelty_percent'] = 100 * (1 - selected.fraction_seen)
    source = HERE / 'figure2_scaffold_sensitivity.csv'
    selected.to_csv(source, index=False, float_format='%.12g')
    caths = sorted(selected.cath.unique(), key=lambda v: tuple(map(int, v.split('.'))))
    fig = plt.figure(figsize=(7, 3.6))
    axes = [fig.add_axes([0.15, 0.23, 0.345, 0.63]), fig.add_axes([0.635, 0.23, 0.345, 0.63])]
    for ax, identity, letter, title in zip(axes, ['murcko_typed', 'murcko_generic'], 'AB',
                                         ['Typed Murcko scaffolds', 'Generic Murcko scaffolds']):
        axes_style(ax, letter, title)
        subsets = [selected.loc[selected.identity.eq(identity) & selected.weighting.eq(w)]
                   .set_index('cath').loc[caths].novelty_percent.to_numpy()
                   for w in ['unique_keys', 'rows']]
        for y, x1, x2 in zip(range(10), *subsets):
            ax.plot([x1, x2], [y, y], color='#B8B8B8', linewidth=0.8, zorder=1)
        ax.scatter(subsets[0], range(10), marker='o', s=22, facecolors='white',
                   edgecolors='#333333', linewidths=0.8, zorder=3)
        ax.scatter(subsets[1], range(10), marker='s', s=16, color=SPLITS[2][1], zorder=3)
        ax.set_yticks(range(10), caths)
        ax.tick_params(axis='y', length=0, pad=5, labelsize=8)
        ax.set_ylim(9.55, -0.55)
        ax.set_xlim(-2, 103)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Scaffolds absent from development (%)', labelpad=6)
    fig.legend(handles=[
        Line2D([], [], marker='o', color='#333333', markerfacecolor='white', linestyle='none', markersize=4.5, label='Distinct scaffolds'),
        Line2D([], [], marker='s', color=SPLITS[2][1], linestyle='none', markersize=4, label='Test pairs')],
        loc='lower center', bbox_to_anchor=(0.55, 0.015), frameon=False, ncol=2,
        columnspacing=2, handletextpad=0.5, fontsize=8)
    result = {**save_figure(fig, 'figure2_cath_chemistry_v3', 'Scaffold novelty and scaffold-definition sensitivity'),
              'source': {'path': str(source.relative_to(REVISIONS)), 'sha256': sha256(source)},
              'inputs': {'chemistry_overlap': {
                  'path': str(INPUTS['chemistry_overlap'].relative_to(REVISIONS)),
                  'sha256': sha256(INPUTS['chemistry_overlap'])}},
              'script': {'path': str(Path(__file__).relative_to(REVISIONS)),
                         'sha256': sha256(Path(__file__))},
              'denominators': 'Acyclic scaffolds excluded; generic conversion failures additionally excluded from generic only.'}
    (HERE / 'figure2_chemistry_plot_metadata.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    from similarity_cdfs import main as make_cdfs
    make_cdfs()
    auc, reuse, novelty, neighbor = source_data()
    paired = model_differences(auc)
    differences = HERE / 'figure3_source_model_differences.csv'
    paired.to_csv(differences, index=False, float_format='%.12g')
    result = {
        'figure2': draw_split_audit(reuse, neighbor),
        'figureS1': draw_scaffold_sensitivity(),
        'script': {'path': str(Path(__file__).relative_to(REVISIONS)), 'sha256': sha256(Path(__file__))},
        'inputs': {name: {'path': str(path.relative_to(REVISIONS)), 'sha256': sha256(path)}
                   for name, path in {**INPUTS, 'cdfs': CDFS, 'source_loader': SCRIPTS / 'source_data.py'}.items()},
        'paired_difference_source': {'path': str(differences.relative_to(REVISIONS)), 'sha256': sha256(differences)},
        'versions': {'matplotlib': matplotlib.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
    }
    (HERE / 'main_figures_metadata.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v['outputs'] for k, v in result.items() if k.startswith('figure')}, indent=2))


if __name__ == '__main__':
    main()
