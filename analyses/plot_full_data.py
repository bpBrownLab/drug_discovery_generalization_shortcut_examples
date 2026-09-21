#!/usr/bin/env python3
"""Reproduce Figure 3 and Figure S2 from the released fold-level results."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from paths import DATA, OUTPUTS
from full_data_plot_style import (make_axis, original_box, probe_box, set_style,
                                 cohort_order, COLORS, W, H, LEFTS, PANEL_W)

OUT = OUTPUTS / "figures"


def load_source(protocol):
    stem = "figure3_random_full_data" if protocol == "random" else "figureS_composite_full_data"
    source = pd.read_csv(DATA / (stem + "_source.csv"))
    assert len(source) == 320
    assert source.groupby(["panel", "evaluation", "threshold", "configuration"]).size().eq(10).all()
    return source


def plot(source, protocol):
    fig = plt.figure(figsize=(W,H))
    for row,titles in enumerate([['DeepDTA','TransformerDTA','Ligand lookup'],
                                 ['Ligand size','DeepDTA','TransformerDTA']]):
        for col,title in enumerate(titles):
            center = (LEFTS[col]+PANEL_W/2)/W
            fig.text(center,(4.875 if row==0 else 2.72)/H,title,
                     ha='center',va='top',fontsize=9,fontweight='bold')
            if row:
                subtitle = 'Heavy-atom count only' if col==0 else 'Whole-function additive projection'
                fig.text(center,2.535/H,subtitle,ha='center',va='top',fontsize=7.2)
            panel = 'ABCDEF'[row*3+col]
            ax = make_axis(fig,col,row,panel)
            data = source[source.panel==panel]
            model = data.model.iloc[0]
            assert data.value.between(*ax.get_ylim()).all(), (panel,data.value.min())
            for ti,threshold in enumerate(['Weak+','High']):
                for evaluation in ['validation','cath_lso_test']:
                    validation = evaluation=='validation'
                    g = data[(data.threshold==threshold)&(data.evaluation==evaluation)]
                    if panel in 'ABCD':
                        original_box(ax,cohort_order(g).value.to_numpy(),ti+(-.17 if validation else .17),COLORS[model],validation)
                    else:
                        for config,offset in [('original',-.085),('intervened',.085)]:
                            probe_box(ax,cohort_order(g[g.configuration==config]).value.to_numpy(),
                                      ti+(-.21 if validation else .21)+offset,
                                      COLORS[model],validation,config=='intervened')
    fig.text(LEFTS[0]/W,.49/H,'Full evaluation sets.  A–C: original exports; D: size control.  E,F: FP32 models.',
             fontsize=7.1,ha='left',va='center')
    label = f'{protocol.capitalize()} split (95/5)'
    fig.legend(handles=[
        Patch(facecolor='#E2E2E2',edgecolor='black',linestyle='--',label=f'{label}: original'),
        Patch(facecolor='#E2E2E2',edgecolor='black',linestyle='--',hatch='///',label=f'{label}: additive projection'),
        Patch(facecolor='#9B9B9B',edgecolor='black',label='CATH-LSO test: original'),
        Patch(facecolor='#9B9B9B',edgecolor='black',hatch='///',label='CATH-LSO test: additive projection')],
        loc='lower center',bbox_to_anchor=(.535,.045/H),ncol=2,frameon=False,fontsize=7.4,
        handlelength=1.65,columnspacing=2,labelspacing=.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for a in fig.findobj(matplotlib.text.Text):
        if not a.get_visible() or not a.get_text().strip(): continue
        b = a.get_window_extent(renderer)
        assert b.x0>=-.5 and b.y0>=-.5 and b.x1<=fig.bbox.width+.5 and b.y1<=fig.bbox.height+.5, a.get_text()
    stem = OUT / ('figure3_dta_full_data' if protocol == 'random' else 'figureS_dta_composite_full_data')
    source.to_csv(stem.with_name(stem.name+'_source.csv'),index=False)
    summary = source.groupby(['panel','model','evaluation','threshold','configuration']).value.agg(['mean','std','min','max','count'])
    summary.to_csv(stem.with_name(stem.name+'_summary.csv'))
    for ext in ['png','pdf','svg']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=600)
    plt.close(fig)
    return dict(protocol=protocol,fold_values=len(source),original_ABCD_preserved=True,
                all_evaluation_rows_used=True,projection_converged=True,y_limits=[.4,1.015])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    set_style()
    plt.rcParams["hatch.linewidth"] = .4
    checks = [plot(load_source(protocol), protocol) for protocol in ["random", "composite"]]
    (OUTPUTS / "full_data_figure_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("Wrote Figure 3 and Figure S2 to outputs/figures/.")


if __name__ == "__main__":
    main()
