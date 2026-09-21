"""Score a CSV of pocket sequences and ligand SMILES using a released model."""

from pathlib import Path
import argparse

import pandas as pd
import torch

from model.load import load_checkpoint
from model.assets import tokenizers
from model.inputs import TransformerDTATokenizer, protein_string, smiles_string, encode_sequences, load_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="CSV with pocket and smiles columns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--local-files-only", action="store_true", help="Require frozen encoders in the Hugging Face cache")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Choose a new output file; existing inputs and outputs are not overwritten")

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    model, metadata = load_checkpoint(args.checkpoint, local_files_only=args.local_files_only)
    model.to(args.device)
    config = metadata["config"]
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if not {"pocket", "smiles"}.issubset(frame.columns) or frame.empty:
        raise ValueError("Input must contain nonempty pocket and smiles columns")
    if frame[["pocket", "smiles"]].apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Every input row needs both a pocket sequence and a ligand SMILES")
    frame["pocket"] = frame.pocket.map(protein_string)
    frame["smiles"] = frame.smiles.map(smiles_string)
    if frame.pocket.str.len().gt(config["pocket_length"]).any():
        raise ValueError("Pocket sequence exceeds the checkpoint's input limit")

    if metadata["model"] == "deepdta":
        pv = load_vocab(tokenizers() / "deepdta/protein_dict.json")
        lv = load_vocab(tokenizers() / "deepdta/ligand_dict.json")
    else:
        tokenizer = TransformerDTATokenizer(
            protein_bert_name=str(tokenizers() / "transformer/protein"),
            ligand_bert_name=str(tokenizers() / "transformer/ligand"),
            max_protein_length=config["pocket_length"],
            max_ligand_length=config["smiles_length"],
        )

    logits = []
    with torch.inference_mode():
        for start in range(0, len(frame), args.batch_size):
            part = frame.iloc[start:start + args.batch_size]
            if metadata["model"] == "deepdta":
                protein = torch.as_tensor(encode_sequences(part.pocket, pv, config["pocket_length"]), device=args.device)
                ligand = torch.as_tensor(encode_sequences(part.smiles, lv, config["smiles_length"]), device=args.device)
                values = model(protein, ligand)
            else:
                protein = {k: torch.as_tensor(v, device=args.device) for k, v in tokenizer.tokenize_proteins(part.pocket.tolist()).items()}
                ligand = {k: torch.as_tensor(v, device=args.device) for k, v in tokenizer.tokenize_ligands(part.smiles.tolist()).items()}
                values = model(protein["input_ids"], ligand["input_ids"], protein["attention_mask"], ligand["attention_mask"])
            logits.append(values.cpu())
    logits = torch.cat(logits)
    result = pd.DataFrame({"row": range(len(frame))})
    if "sample_id" in frame:
        result["sample_id"] = frame.sample_id
    for index, threshold in enumerate(["weak", "high"]):
        result[f"{threshold}_logit"] = logits[:, index].numpy()
        result[f"{threshold}_probability"] = logits[:, index].sigmoid().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
