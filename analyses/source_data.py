"""Load the ten-fold observations used in Figures 2 and 3."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from paths import ROOT as REVISIONS, OUTPUTS as HERE, DATA, SCRIPTS
INPUTS = {
    'model_metrics': DATA / 'fold_metrics.csv',
    'lookup_metrics': DATA / 'lookup_per_fold.csv',
    'protein_overlap': DATA / 'protein_overlap_per_fold.csv',
    'chemistry_overlap': DATA / 'chemistry_overlap.csv',
    'chemistry_similarity': DATA / 'chemistry_similarity_summary.csv',
}
REUSE_CONDITIONS = [
    ('random', 'validation', 'train', 'Random validation'),
    ('composite_scaffold', 'validation', 'train', 'Composite validation'),
    ('random', 'cath_lso_test', 'development', 'CATH test'),
]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_ten_folds(frame: pd.DataFrame, groups: list[str]) -> None:
    counts = frame.groupby(groups).cath.nunique()
    if not counts.eq(10).all() or frame.duplicated(groups + ['cath']).any():
        raise ValueError(f'Expected exactly ten distinct CATH folds per group: {counts.to_dict()}')
    if not np.isfinite(frame.value).all():
        raise ValueError('Non-finite figure observations')


def source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = pd.read_csv(INPUTS['model_metrics'])
    auc = model.rename(columns={'auc': 'value'})[
        ['cath', 'model', 'development', 'evaluation', 'threshold', 'n', 'value']].copy()
    lookup = pd.read_csv(INPUTS['lookup_metrics'])
    lookup = lookup.loc[lookup.identity.eq('smiles')].rename(columns={
        'cath_id': 'cath', 'method': 'development', 'partition': 'evaluation', 'auc': 'value'})
    lookup['model'] = 'ligand_lookup'
    lookup['development'] = lookup.development.replace({'composite_scaffold': 'composite'})
    lookup['threshold'] = lookup.threshold.map({5: 'Weak+', 7: 'High'})
    auc = pd.concat([auc, lookup[auc.columns]], ignore_index=True)
    check_ten_folds(auc, ['model', 'development', 'evaluation', 'threshold'])
    if len(auc) != 240:
        raise ValueError('Expected 240 AUROC observations')

    chem = pd.read_csv(INPUTS['chemistry_overlap'])
    protein = pd.read_csv(INPUTS['protein_overlap'])
    reuse_rows = []
    for group, (method, query, reference, _) in enumerate(REUSE_CONDITIONS):
        selected = chem.loc[
            chem.development_split.eq(method) & chem.evaluation.eq(query)
            & chem.reference.eq(reference) & chem.weighting.eq('rows')
            & chem.identity.isin(['canonical_isomeric', 'murcko_typed'])]
        for row in selected.itertuples():
            reuse_rows.append(dict(cath=row.cath, condition=group, identity=row.identity,
                development_split=method, query=query, reference=reference,
                value=100 * row.fraction_seen, denominator=row.denominator,
                seen=row.seen, excluded_empty_key=row.excluded_empty_key))
        selected_p = protein.loc[
            protein.method.eq(method) & protein.query_partition.eq(query)
            & protein.reference_partition.eq(reference)]
        for row in selected_p.itertuples():
            reuse_rows.append(dict(cath=row.cath_id, condition=group, identity='pocket',
                development_split=method, query=query, reference=reference,
                value=100 * row.seen_query_row_fraction, denominator=row.query_rows,
                seen=row.seen_query_rows, excluded_empty_key=0))
    reuse = pd.DataFrame(reuse_rows)
    check_ten_folds(reuse, ['condition', 'identity'])
    if len(reuse) != 90:
        raise ValueError('Expected 90 input-reuse observations')

    selected = chem.loc[
        chem.development_split.eq('random') & chem.evaluation.eq('cath_lso_test')
        & chem.reference.eq('development') & chem.identity.eq('murcko_typed')
        & chem.weighting.isin(['rows', 'unique_keys'])].copy()
    novelty = selected[['cath', 'weighting', 'denominator', 'unseen', 'excluded_empty_key']].copy()
    novelty['value'] = 100 * (1 - selected.fraction_seen)
    check_ten_folds(novelty, ['weighting'])
    if len(novelty) != 20:
        raise ValueError('Expected 20 CATH scaffold-novelty observations')
    summary = pd.read_csv(INPUTS['chemistry_similarity'])
    neighbor_rows = []
    for condition, (method, query, reference, _) in enumerate(REUSE_CONDITIONS):
        actual_method = 'both' if query == 'cath_lso_test' else method
        rows = summary.loc[summary.development_split.eq(actual_method)
            & summary.evaluation.eq(query) & summary.reference.eq(reference)
            & summary.weighting.eq('unique_canonical_ligands')]
        for row in rows.itertuples():
            neighbor_rows.append(dict(cath=row.cath, condition=condition,
                development_split=actual_method, query=query, reference=reference,
                value=row.mean_max_tanimoto, denominator=row.denominator,
                query_sampling='all'))
    neighbor = pd.DataFrame(neighbor_rows)
    check_ten_folds(neighbor, ['condition'])
    if len(neighbor) != 30:
        raise ValueError('Expected 30 nearest-neighbor fold means')
    return auc, reuse, novelty, neighbor

