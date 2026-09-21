# Generalization and shortcuts in drug discovery

Figures, numerical results, and model inference for *What does generalizable AI in drug discovery really mean?* Use Python 3.12.

## Figures and tables

```bash
pip install -r requirements.txt
python reproduce.py
```

This regenerates Figures 1–4, Figures S1–S2, and Tables S1–S7 from the supplied results. Figures go to `outputs/figures/` and tables to `outputs/si_tables/`. The plotting scripts are in `analyses/`; their numerical inputs are in `data/`.

## Model weights

Download the weights from [GitHub Releases](https://github.com/bpbrownlab/drug_discovery_generalization_shortcut_examples/releases) into the repository root. DeepDTA uses `checkpoints_deepdta.tar`; TransformerDTA uses all three `checkpoints_transformerdta_part*.tar` files.

```bash
sha256sum --check --ignore-missing model/checkpoint_archives.sha256
for archive in checkpoints_*.tar; do tar -xf "$archive"; done
pip install torch==2.7.0 transformers==4.51.3 tokenizers==0.21.1 rdkit==2026.3.3
python predict.py \
  --checkpoint checkpoints/1.10.1300.10/deepdta/random/best.pt \
  --input pairs.csv --output predictions.csv
```

The input CSV needs `pocket` and `smiles` columns. `pocket` is the sequence of selected pocket residues in their original order. Add `--device cuda` for GPU inference. Predictions are logits and probabilities at the Weak+ and High thresholds, not continuous affinities.

The TransformerDTA loader downloads the two frozen encoders at their recorded revisions. All trained components are in the release archives. Network definitions, tokenizer files, and checkpoint settings are in `model/`.

Because of PDBbind licensing restrictions, the prepared study inputs and individual prediction and projection records are not distributed here.
