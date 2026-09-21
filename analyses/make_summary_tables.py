"""Generate Tables S1–S7 from the saved analysis results."""
from pathlib import Path
import json
import re
import pandas as pd
from paths import ROOT, OUTPUTS as HERE, DATA
AUDIT=DATA
cohorts=json.loads((AUDIT/'cohort_integrity.json').read_text())['manifests']
counts={(r['cath'],r['method']):r['counts'] for r in cohorts}
caths=sorted({c for c,m in counts},key=lambda s:tuple(map(int,s.split('.'))))
parts=[r'''\begin{table}[htbp]
\centering\small
\caption{\textbf{Sample counts after filtering in each CATH-LSO fold.} We report counts after removing invalid inputs and filtering exact ligand--pocket pairs. Random and composite validation sets contain the same number of examples within a fold, but different examples. Both networks and both split methods use the same CATH test examples.}
\label{tab:figure2counts}
\begin{tabular}{lrrrr}
\hline
CATH fold & Random train & Composite train & Validation & CATH test \\
\hline
''']
for cath in caths:
 r=counts[cath,'random'];c=counts[cath,'composite_scaffold']
 assert r['validation']==c['validation'] and r['cath_lso_test']==c['cath_lso_test']
 parts.append(cath+' & '+' & '.join(f'{v:,}' for v in [r['train'],c['train'],r['validation'],r['cath_lso_test']])+r' \\'+'\n')
parts.append(r'\hline\end{tabular}\end{table}'+'\n\n')
m=pd.read_csv(AUDIT/'fold_metrics.csv');m=m[['model','development','evaluation','threshold','auc']]
l=pd.read_csv(AUDIT/'lookup_per_fold.csv');l=l[l.identity.eq('smiles')].rename(columns={'method':'development','partition':'evaluation'})
l['model']='ligand_lookup';l['threshold']=l.threshold.map({5:'Weak+',7:'High'});l.development=l.development.replace({'composite_scaffold':'composite'})
allmetrics=pd.concat([m,l[m.columns]],ignore_index=True)
parts.append(r'''\begin{table}[htbp]
\centering\small
\caption{\textbf{Ranking performance of the neural networks and ligand lookup.} We report mean AUROC $\pm$ sample standard deviation across ten equally weighted CATH folds for panels A--C of Figure~3 and Supplementary Figure~\ref{fig:GeneralizationGapComposite}. Standard deviations describe variation between held-out families, not uncertainty across independent training runs. The lookup assigns each exact SMILES its mean training label and uses training prevalence for unseen strings.}
\label{tab:figure2aucs}
\begin{tabular}{lllcc}
\hline
Model & Split & Threshold & Validation & CATH test \\
\hline
''')
for model,label in [('deepdta','DeepDTA'),('transformerdta','TransformerDTA'),('ligand_lookup','Ligand lookup')]:
 for dev in ['random','composite']:
  for threshold in ['Weak+','High']:
   r=allmetrics[allmetrics.model.eq(model)&allmetrics.development.eq(dev)&allmetrics.threshold.eq(threshold)]
   vals=[]
   for split in ['validation','cath_lso_test']:
    a=r.loc[r.evaluation.eq(split),'auc'];assert len(a)==10
    vals.append(f'${a.mean():.3f} \\pm {a.std(ddof=1):.3f}$')
   parts.append(' & '.join([label,dev.capitalize(),threshold,*vals])+r' \\'+'\n')
parts.append(r'\hline\end{tabular}\end{table}'+'\n\n')
chem=pd.read_csv(DATA/'chemistry_overlap.csv')
c=chem[chem.development_split.eq('random')&chem.evaluation.eq('cath_lso_test')&chem.reference.eq('development')]
protein=pd.read_csv(DATA/'protein_overlap_per_fold.csv');p=protein[protein.method.eq('random')&protein.query_partition.eq('cath_lso_test')&protein.reference_partition.eq('development')].set_index('cath_id')
parts.append(r'''\begin{table}[htbp]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{\textbf{Ligand, scaffold, and pocket reuse in the CATH test sets.} Percentages describe overlap with training and validation combined. Ligand and pocket percentages give each test example equal weight; ligand identity uses canonical isomeric SMILES. For typed and generic Murcko scaffolds, we count either examples with a scaffold (R) or distinct nonempty scaffolds (S). Acyclic ligands are excluded from scaffold calculations. Ligands for which generic scaffold conversion fails are excluded only from the generic columns. Each fold contributes equally to the mean.}
\label{tab:figure2overlap}
\begin{tabular}{lrrrrrr}
\hline
CATH fold & Ligand & Typed R & Typed S & Generic R & Generic S & Pocket \\
\hline
''')
values=[]
for cath in caths:
 d=c[c.cath.eq(cath)];v=[]
 for identity,weighting in [('canonical_isomeric','rows'),('murcko_typed','rows'),('murcko_typed','unique_keys'),('murcko_generic','rows'),('murcko_generic','unique_keys')]:
  a=d.loc[d.identity.eq(identity)&d.weighting.eq(weighting),'fraction_seen'];assert len(a)==1
  v.append(100*a.iloc[0])
 v.append(100*p.loc[cath,'seen_query_row_fraction']);values.append(v)
 parts.append(cath+' & '+' & '.join([*(f'{x:.2f}' for x in v[:-1]),f'{v[-1]:.4f}'])+r' \\'+'\n')
mean=pd.DataFrame(values).mean().tolist()
parts.append(r'\hline'+'\nMean & '+' & '.join([*(f'{x:.2f}' for x in mean[:-1]),f'{mean[-1]:.4f}'])+r' \\'+'\n')
parts.append(r'\hline\end{tabular}\end{table}'+'\n')
from make_supplementary_tables import supplementary_tables
parts.append('\n\n' + supplementary_tables())
table_order = [
    ('sample_counts', 'figure2counts'),
    ('overlap', 'figure2overlap'),
    ('similarity_ranges', 'similaritybands'),
    ('similarity_by_family', 'similaritycath'),
    ('auroc', 'figure2aucs'),
    ('metrics_random', 'additionalmetricsrandom'),
    ('metrics_composite', 'additionalmetricscomposite'),
]
blocks = re.findall(r'\\begin\{table\}.*?\\end\{table\}', ''.join(parts), flags=re.S)
tables = {re.search(r'\\label\{tab:([^}]+)\}', block).group(1): block for block in blocks}
assert len(blocks) == len(tables) == len(table_order)
table_dir = HERE / 'si_tables'
table_dir.mkdir(exist_ok=True)
ordered = []
for name, label in table_order:
    table = tables[label] + '\n'
    (table_dir / f'{name}.tex').write_text(table)
    ordered.append(table)
(HERE / 'figure2_summary_tables.tex').write_text('\n'.join(ordered))
print('Wrote Tables S1–S7 to si_tables/ and figure2_summary_tables.tex.')
