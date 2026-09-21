"""Keep similarity measurements without publishing per-compound records."""
import pandas as pd

GROUP_KEYS = ['cath', 'development_split', 'evaluation', 'reference']
GROUP_CONTEXT = ['eligible_query_ligands', 'is_sampled', 'reference_ligands']


def without_structures(frame):
    """Assign one identifier per canonical query ligand across the full audit.

    Sorting the unique queries makes the identifiers repeatable when the same
    input records are used. A ligand keeps its identifier across folds.
    """
    result = frame.copy()
    column = result.columns.get_loc('canonical_isomeric')
    names = result.pop('canonical_isomeric')
    if names.isna().any():
        raise ValueError('Every query must have a canonical ligand identity')
    codes, _ = pd.factorize(names, sort=True)
    result.insert(column, 'ligand_id', [f'ligand_{code:07d}' for code in codes])
    result = result.drop(columns=['nearest_reference_canonical_isomeric'])
    return result


def as_distribution(frame):
    """Collapse per-ligand rows to counted similarity values.

    Every published figure and table reads this census as a multiset: means,
    quantiles, similarity bands and the empirical CDFs all depend on which
    values occur and how often, never on which ligand carried a value. Storing
    the distinct (similarity, row weight) pairs with their multiplicity
    therefore preserves those results exactly while removing the individual
    per-compound records.
    """
    columns = GROUP_KEYS + GROUP_CONTEXT + ['max_tanimoto', 'query_rows']
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f'Similarity records are missing columns: {missing}')
    return frame.groupby(columns, sort=True).size().reset_index(name='ligand_count')


def load_distribution(path):
    """Read counted similarity values and expand them to one row per query.

    The inverse of :func:`as_distribution`, so the analysis code downstream
    sees the same per-query frame it always did.
    """
    table = pd.read_csv(path)
    repeats = table.pop('ligand_count').to_numpy()
    if (repeats < 1).any():
        raise ValueError('Every recorded similarity value needs a positive ligand count')
    return table.loc[table.index.repeat(repeats)].reset_index(drop=True)
