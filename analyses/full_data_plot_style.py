"""Boxplot and layout helpers for Figure 3 and Figure S2."""
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

W, H = 7.0, 5.0
LEFTS = np.array([.085, .395, .705]) * W
PANEL_W, PANEL_H = .275 * W, .345 * 4.6
BOTTOMS = [3.105, .79]
COLORS = {"deepdta": "#EE7992", "transformerdta": "#00A7A7",
          "ligand_lookup": "#A58DD1", "heavy_atom_count": "#D5A349"}

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

def original_box(ax, values, position, color, validation):
    result = ax.boxplot([values], positions=[position], widths=0.26,
                       whis=1.5, patch_artist=True, showfliers=False,
                       manage_ticks=False, medianprops={'color': 'black', 'linewidth': 0.9})
    edge = 'black'
    line = '--' if validation else '-'
    for patch in result['boxes']:
        patch.set(facecolor=to_rgba('#E2E2E2' if validation else color,
                                   0.70 if validation else 0.80),
                  edgecolor=edge, linewidth=0.85, linestyle=line)
    for artist in [*result['whiskers'], *result['caps']]:
        artist.set(color=edge, linewidth=0.7, linestyle=line)
    # Deterministic horizontal offsets distinguish folds without moving the
    # AUROC values. Every observation, including boxplot outliers, is shown.
    offsets = np.array([-0.043, 0.020, -0.005, 0.043, -0.026,
                         0.006, -0.036, 0.032, -0.016, 0.013])
    ax.scatter(position + offsets, values, s=7.5,
               facecolors=to_rgba('#8D8D8D' if validation else color,
                                  0.75 if validation else 0.95),
               edgecolors='black', linewidths=0.3, zorder=4)

def cohort_order(frame):
    ordered=sorted(frame.cath,key=lambda v:tuple(map(int,v.split('.'))))
    out=frame.set_index('cath').loc[ordered].reset_index()
    assert len(out)==10 and out.cath.nunique()==10 and np.isfinite(out.value).all()
    return out

def probe_box(ax,values,position,color,validation,intervened):
    artists=ax.boxplot([values],positions=[position],widths=.145,whis=1.5,
        patch_artist=True,showfliers=False,manage_ticks=False,
        medianprops={'color':'black','linewidth':.8})
    line='--' if validation else '-'
    for patch in artists['boxes']:
        patch.set(facecolor=to_rgba('#E2E2E2' if validation else color,.7 if validation else .8),
            edgecolor='black',linewidth=.8,linestyle=line,hatch='///' if intervened else None)
    for artist in [*artists['whiskers'],*artists['caps']]:
        artist.set(color='black',linewidth=.65,linestyle=line)
    jitter=np.array([-.043,.020,-.005,.043,-.026,.006,-.036,.032,-.016,.013])*.55
    ax.scatter(position+jitter,values,s=5,
        facecolors=to_rgba('#8D8D8D' if validation else color,.75 if validation else .95),
        edgecolors='black',linewidths=.25,zorder=4)

def make_axis(fig, col, row, letter):
    bottom = BOTTOMS[row]
    ax = fig.add_axes([LEFTS[col] / W, bottom / H, PANEL_W / W, PANEL_H / H])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8, pad=3)
    ax.text(-.09, 1.01, letter, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10)
    ax.axhline(.5, color="#888888", linewidth=.6, linestyle=(0, (3, 2)), zorder=0)
    ax.set_ylim(.40, 1.015)
    ax.set_yticks(np.arange(.4, 1.01, .1))
    ax.set_xlim(-.52, 1.52)
    ax.set_xticks([0, 1], ["Weak+ (≥5)", "High (≥7)"])
    if col == 0:
        ax.set_ylabel("ROC AUC", labelpad=5)
    else:
        ax.tick_params(labelleft=False)
    return ax
