#!/usr/bin/env python3
"""Generate the manuscript figures and supplementary tables."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    jobs = [
        ("plot_inductive_bias_viz.py", "--out-dir", output / "figures", "--seed", 42),
        ("plot_main_figures.py",),
        ("plot_full_data.py",),
        ("make_summary_tables.py",),
        ("plot_dude_ligand_only_control.py", "--fold-means", ROOT / "data/dude_fold_means.csv",
         "--out-dir", output / "figures"),
    ]
    for script, *arguments in jobs:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "analyses" / script), *map(str, arguments)],
                       cwd=ROOT, env=env, check=True)
    print("Figures are in outputs/figures/; Tables S1–S7 are in outputs/si_tables/.")


if __name__ == "__main__":
    main()
