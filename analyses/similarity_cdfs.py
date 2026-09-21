"""Rebuild Figure 2B empirical CDFs from the nearest-neighbor score census."""
import numpy as np
import pandas as pd
from paths import DATA, OUTPUTS
from similarity_records import load_distribution


def main():
    neighbors = load_distribution(DATA / 'chemistry_similarity_distribution.csv.gz')
    if not np.isfinite(neighbors.max_tanimoto).all() or not neighbors.max_tanimoto.between(0, 1).all():
        raise ValueError('Similarity scores must be finite and between zero and one')
    keys = ['cath', 'development_split', 'evaluation', 'reference']
    if neighbors.groupby(keys).ngroups != 30:
        raise ValueError('Expected 30 distinct fold/protocol query sets')
    for _, group in neighbors.groupby(keys):
        if group.is_sampled.any() or not group.eligible_query_ligands.eq(len(group)).all():
            raise ValueError('Every eligible ligand must be queried exactly once')
    grid = np.linspace(0, 1, 1001)
    rows = []
    for condition, (method, evaluation, label) in enumerate([
        ('random', 'validation', 'Random validation'),
        ('composite_scaffold', 'validation', 'Composite validation'),
        ('both', 'cath_lso_test', 'CATH test'),
    ]):
        selected = neighbors.loc[neighbors.development_split.eq(method) &
                                 neighbors.evaluation.eq(evaluation)]
        curves = []
        if selected.cath.nunique() != 10:
            raise ValueError('Expected ten folds in every CDF condition')
        for cath, group in selected.groupby('cath', sort=True):
            values = np.sort(group.max_tanimoto.to_numpy())
            curve = np.searchsorted(values, grid, side='right') / len(values)
            curves.append(curve)
            rows.extend(dict(condition=condition, label=label, cath=cath,
                             tanimoto=x, cumulative_fraction=y, query_count=len(values))
                        for x, y in zip(grid, curve))
        rows.extend(dict(condition=condition, label=label, cath='macro_mean',
                         tanimoto=x, cumulative_fraction=y, query_count=np.nan)
                    for x, y in zip(grid, np.mean(curves, axis=0)))
    pd.DataFrame(rows).to_csv(OUTPUTS / 'figure2_chemistry_source_cdfs.csv',
                             index=False, float_format='%.12g')


if __name__ == '__main__':
    main()
