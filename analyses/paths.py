"""Paths to repository data and generated outputs."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / 'analyses'
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'
RECOMPUTED = OUTPUTS / 'recomputed'
MANIFESTS = DATA / 'manifests'
for directory in (OUTPUTS, RECOMPUTED):
    directory.mkdir(parents=True, exist_ok=True)
