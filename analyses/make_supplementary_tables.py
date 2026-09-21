"""Summarize prediction metrics and ligand similarity for the SI.

Called by make_summary_tables.py; reads the saved per-fold results.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from paths import ROOT, OUTPUTS as HERE, DATA
from similarity_records import load_distribution
AUDIT = DATA
KEYS = ['model', 'development', 'evaluation', 'threshold', 'cath']
METRICS = ['positive_fraction', 'average_precision', 'brier', 'ece']
MODELS = [('deepdta', 'DeepDTA'), ('transformerdta', 'TransformerDTA'),
          ('ligand_lookup', 'Ligand lookup')]
BANDS = ['lt_0.4', '0.4_to_0.6', '0.6_to_0.8', '0.8_to_1.0', 'eq_1.0']
CONDITIONS = [('random', 'validation', 'Random validation'),
              ('composite_scaffold', 'validation', 'Composite validation'),
              ('both', 'cath_lso_test', 'CATH-LSO test')]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_sources():
    columns = KEYS + ['n', 'positives', 'auc'] + METRICS
    neural = pd.read_csv(AUDIT / 'fold_metrics.csv')[columns]
    lookup = pd.read_csv(DATA / 'lookup_additional_metrics.csv')[columns]
    assert len(neural) == 160 and len(lookup) == 80
    source = pd.concat([neural, lookup], ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    assert len(source) == 240 and not source.duplicated(KEYS).any()
    assert source.groupby(KEYS[:-1]).cath.nunique().eq(10).all()
    assert np.isfinite(source[METRICS + ['auc']]).all().all()
    assert source[METRICS + ['auc']].ge(0).all().all() and source[METRICS + ['auc']].le(1).all().all()
    assert np.allclose(source.positive_fraction, source.positives / source.n, atol=1e-14, rtol=0)
    # All three models must have exactly the same evaluation denominators and labels.
    cohorts = source.groupby(['development', 'evaluation', 'threshold', 'cath'])
    assert cohorts['n'].nunique().eq(1).all() and cohorts.positives.nunique().eq(1).all()
    original = pd.read_csv(AUDIT / 'lookup_per_fold.csv')
    original = original.loc[original.identity.eq('smiles')].rename(columns={
        'cath_id': 'cath', 'method': 'development', 'partition': 'evaluation'})
    original['development'] = original.development.replace({'composite_scaffold': 'composite'})
    original['threshold'] = original.threshold.map({5: 'Weak+', 7: 'High'})
    original['model'] = 'ligand_lookup'
    check = lookup.merge(original[KEYS + ['auc']], on=KEYS, validate='one_to_one', suffixes=('', '_original'))
    assert len(check) == 80 and np.allclose(check.auc, check.auc_original, atol=1e-12, rtol=0)
    rows = []
    for key, group in source.groupby(KEYS[:-1], sort=True):
        record = dict(zip(KEYS[:-1], key), n_folds=len(group))
        for metric in METRICS:
            record[metric + '_mean'] = float(group[metric].mean())
            record[metric + '_sd'] = float(group[metric].std(ddof=1))
        rows.append(record)
    summary = pd.DataFrame(rows)
    source.to_csv(HERE / 'si_additional_metrics_per_fold.csv', index=False, float_format='%.15g')
    summary.to_csv(HERE / 'si_additional_metrics_summary.csv', index=False, float_format='%.15g')
    return summary


def similarity_sources():
    nearest = load_distribution(DATA / 'chemistry_similarity_distribution.csv.gz')
    original = pd.read_csv(DATA / 'chemistry_similarity_summary.csv')
    group_keys = ['cath', 'development_split', 'evaluation', 'reference']
    assert nearest.groupby(group_keys).ngroups == 30
    assert np.isfinite(nearest.max_tanimoto).all() and nearest.max_tanimoto.between(0, 1).all()
    records, cath_rows = [], []
    for key, group in nearest.groupby(group_keys, sort=True):
        meta = dict(zip(group_keys, key))
        prior = original
        for field, value in meta.items():
            prior = prior.loc[prior[field].eq(value)]
        unique = prior.loc[prior.weighting.eq('unique_canonical_ligands')].iloc[0]
        values = group.max_tanimoto.to_numpy()
        assert len(group) == unique.denominator
        assert np.isclose(values.mean(), unique.mean_max_tanimoto, atol=1e-12, rtol=0)
        assert group.reference_ligands.nunique() == 1
        assert group.eligible_query_ligands.nunique() == 1
        sampled = bool(group.is_sampled.iloc[0])
        assert not sampled and not group.is_sampled.any()
        assert len(group) == group.eligible_query_ligands.iloc[0]
        if meta['evaluation'] == 'validation':
            assert meta['reference'] == 'train'
        else:
            assert meta['reference'] == 'development'
        # The original census allows 1e-12 tolerance at similarity cutoffs.
        ge = {x: values >= x - 1e-12 for x in [0.4, 0.6, 0.8, 1.0]}
        masks = [~ge[0.4], ge[0.4] & ~ge[0.6], ge[0.6] & ~ge[0.8],
                 ge[0.8] & ~ge[1.0], ge[1.0]]
        assert np.stack(masks).sum(axis=0).tolist() == [1] * len(values)
        record = dict(meta, n_queries=len(group), eligible_queries=int(group.eligible_query_ligands.iloc[0]),
                      reference_ligands=int(group.reference_ligands.iloc[0]), sampled=sampled,
                      mean_max_tanimoto=float(values.mean()))
        for band, mask in zip(BANDS, masks):
            record['count_' + band] = int(mask.sum())
            record['percent_' + band] = float(100 * mask.mean())
        for cutoff, mask in ge.items():
            assert np.isclose(mask.mean(), unique[f'fraction_ge_{cutoff:.1f}'], atol=1e-12, rtol=0)
        records.append(record)
        if meta['evaluation'] == 'cath_lso_test':
            row = prior.loc[prior.weighting.eq('rows')].iloc[0]
            assert int(group.query_rows.sum()) == row.denominator
            weighted = float(np.average(values, weights=group.query_rows))
            assert np.isclose(weighted, row.mean_max_tanimoto, atol=1e-12, rtol=0)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            assert np.allclose([q1, median, q3], [unique.q25, unique['median'], unique.q75], atol=1e-12, rtol=0)
            cath_rows.append(dict(cath=meta['cath'], query_ligands=len(group),
                reference_ligands=record['reference_ligands'], mean_max_tanimoto=record['mean_max_tanimoto'],
                q25=float(q1), median=float(median), q75=float(q3),
                percent_ge_0_8=float(100 * ge[0.8].mean()), percent_eq_1=float(100 * ge[1.0].mean()),
                query_rows=int(group.query_rows.sum()), row_weighted_mean=weighted))
    source = pd.DataFrame(records)
    assert len(source) == 30
    macros = []
    for development, evaluation, label in CONDITIONS:
        group = source.loc[source.development_split.eq(development) & source.evaluation.eq(evaluation)]
        assert len(group) == 10 and group.cath.nunique() == 10
        record = dict(development=development, evaluation=evaluation, condition=label, n_folds=10,
                      query_instances=int(group.n_queries.sum()), mean_max_tanimoto=float(group.mean_max_tanimoto.mean()),
                      sd_mean_max_tanimoto=float(group.mean_max_tanimoto.std(ddof=1)))
        for band in BANDS:
            record['percent_' + band] = float(group['percent_' + band].mean())
        assert np.isclose(sum(record['percent_' + band] for band in BANDS), 100)
        macros.append(record)
    macro = pd.DataFrame(macros)
    cath = pd.DataFrame(cath_rows)
    order = sorted(cath.cath, key=lambda value: tuple(map(int, value.split('.'))))
    cath = cath.set_index('cath').loc[order].reset_index()
    source.to_csv(HERE / 'si_ligand_similarity_bands_per_fold.csv', index=False, float_format='%.15g')
    macro.to_csv(HERE / 'si_ligand_similarity_bands_summary.csv', index=False, float_format='%.15g')
    cath.to_csv(HERE / 'si_ligand_similarity_cath.csv', index=False, float_format='%.15g')
    return macro, cath


def supplementary_tables():
    metrics = metric_sources()
    bands, cath = similarity_sources()
    parts = []
    for development in ['random', 'composite']:
        parts.append(r'''\begin{table}[htbp]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{\textbf{Precision--recall and calibration under ''' + development + r''' splitting.} These metrics complement AUROC by describing precision--recall performance and the accuracy of predicted probabilities. Values are means $\pm$ sample standard deviations across ten equally weighted folds. Positive fractions give the percentage of evaluation examples labeled active at each threshold. AP is noninterpolated average precision; Brier scores and ten-bin expected calibration errors (ECE) use the original probabilities. Higher AP and lower Brier/ECE are preferable. Standard deviations describe variation across folds and are not confidence intervals.}
\label{tab:additionalmetrics''' + development + r'''}
\begin{tabular}{lllrrrr}
\hline
Model & Threshold & Evaluation & Positive (\%) & AP & Brier & ECE \\
\hline
''')
        for model, label in MODELS:
            for threshold in ['Weak+', 'High']:
                for evaluation, cohort in [('validation', 'Validation'), ('cath_lso_test', 'CATH test')]:
                    row = metrics.loc[metrics.model.eq(model) & metrics.development.eq(development) &
                                      metrics.threshold.eq(threshold) & metrics.evaluation.eq(evaluation)]
                    assert len(row) == 1
                    row = row.iloc[0]
                    values = []
                    for metric in METRICS:
                        scale, digits = ((100, 2) if metric == 'positive_fraction'
                                         else (1, 4) if metric == 'average_precision' else (1, 3))
                        values.append(f'${scale * row[metric + "_mean"]:.{digits}f} \\pm {scale * row[metric + "_sd"]:.{digits}f}$')
                    parts.append(' & '.join([label, threshold, cohort, *values]) + r' \\' + '\n')
        parts.append(r'\hline\end{tabular}\end{table}' + '\n\n')
    parts.append(r'''\begin{table}[htbp]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{\textbf{Chemical similarity under random, composite, and CATH-LSO evaluation.} We summarize the distributions in Figure~2B by similarity range. Here $T$ is each ligand's maximum Morgan Tanimoto similarity to the reference set. We include every unique validation and CATH test ligand within each fold, comparing validation with training and CATH test with training plus validation. Range percentages give each fold equal weight and sum to 100\% before rounding. Mean $T$ gives the mean $\pm$ sample SD of the ten fold means. $T=1$ indicates identical fingerprints, which need not represent identical molecules.}
\label{tab:similaritybands}
\begin{tabular}{lrrrrrr}
\hline
Evaluation & Mean $T$ & $T<0.4$ & $[0.4,0.6)$ & $[0.6,0.8)$ & $[0.8,1)$ & $T=1$ \\
 & & (\%) & (\%) & (\%) & (\%) & (\%) \\
\hline
''')
    for _, row in bands.iterrows():
        mean = f'${row.mean_max_tanimoto:.3f} \\pm {row.sd_mean_max_tanimoto:.3f}$'
        parts.append(' & '.join([row.condition, mean, *(f'{row["percent_" + band]:.2f}' for band in BANDS)]) + r' \\' + '\n')
    parts.append(r'\hline\end{tabular}\end{table}' + '\n\n')
    parts.append(r'''\begin{table}[htbp]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{\textbf{Variation in chemical similarity across held-out superfamilies.} Query and reference counts give the numbers of unique canonical isomeric ligands within each fold. We give each test ligand equal weight, except in the final column, which weights protein--ligand examples equally to show the effect of repeated ligand observations. Median [Q1,Q3] describes the distribution within a fold. Both training/validation split methods use the same combined reference set and CATH test ligands. The final row averages fold statistics equally and sums counts across folds; a molecule may appear in more than one fold. $T=1$ indicates fingerprint identity, not necessarily molecular identity.}
\label{tab:similaritycath}
\begin{tabular}{lrrrrrrr}
\hline
CATH fold & Queries & References & Mean $T$ & Median [Q1,Q3] & $T\geq0.8$ & $T=1$ & Row mean \\
 & & & & & (\%) & (\%) & $T$ \\
\hline
''')
    for _, row in cath.iterrows():
        parts.append(' & '.join([row.cath, f'{row.query_ligands:,}', f'{row.reference_ligands:,}',
            f'{row.mean_max_tanimoto:.3f}', f'{row["median"]:.3f} [{row.q25:.3f},{row.q75:.3f}]',
            f'{row.percent_ge_0_8:.2f}', f'{row.percent_eq_1:.2f}', f'{row.row_weighted_mean:.3f}']) + r' \\' + '\n')
    parts.append(r'\hline' + '\n' + ' & '.join(['Mean / total', f'{cath.query_ligands.sum():,}',
        f'{cath.reference_ligands.sum():,}', f'{cath.mean_max_tanimoto.mean():.3f}', '--',
        f'{cath.percent_ge_0_8.mean():.2f}', f'{cath.percent_eq_1.mean():.2f}',
        f'{cath.row_weighted_mean.mean():.3f}']) + r' \\' + '\n')
    parts.append(r'\hline\end{tabular}\end{table}' + '\n')
    inputs = [AUDIT / 'fold_metrics.csv', AUDIT / 'lookup_per_fold.csv',
              DATA / 'lookup_additional_metrics.csv', DATA / 'chemistry_similarity_summary.csv',
              DATA / 'chemistry_similarity_distribution.csv.gz']
    outputs = [HERE / ('si_' + stem + '.csv') for stem in ['additional_metrics_per_fold',
        'additional_metrics_summary', 'ligand_similarity_bands_per_fold',
        'ligand_similarity_bands_summary', 'ligand_similarity_cath']]
    metadata = dict(script_sha256=sha256(Path(__file__)),
        inputs={str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        outputs={path.name: sha256(path) for path in outputs},
        all_original_aucs_cohort_counts_and_similarity_summaries_reproduced=True,
        metric_observations=240, similarity_folds=30, unique_query_instances=int(bands.query_instances.sum()),
        weighting='Unweighted means and sample SDs across ten folds; descriptive, not confidence intervals.',
        similarity_cutoff_tolerance=1e-12,
        versions={'numpy': np.__version__, 'pandas': pd.__version__})
    (HERE / 'si_supplementary_tables_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return ''.join(parts)


if __name__ == '__main__':
    raise SystemExit('Run make_summary_tables.py to generate all SI tables together.')
