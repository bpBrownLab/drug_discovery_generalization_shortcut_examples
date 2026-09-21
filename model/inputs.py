"""Input validation and tokenization matching the released checkpoints."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Union
import numpy as np
import torch

AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWYBXZOU')
DUMMY_TOKEN = "dummy"
TOKENIZATION_VERSION = "protbert-complete-pockets-chemberta-roundtrip-v3"


def protein_string(value: str) -> str:
    sequence = ''.join(value.split()).upper()
    if not sequence or set(sequence) - AMINO_ACIDS:
        raise ValueError(f'Invalid protein sequence: {value[:80]!r}')
    return sequence

def smiles_string(value: str) -> str:
    from rdkit import Chem
    # Standard .smi files may have an identifier after the SMILES field.
    parts = value.split()
    if not parts or Chem.MolFromSmiles(parts[0]) is None:
        raise ValueError(f'Invalid SMILES: {value[:80]!r}')
    return parts[0]

def load_vocab(path: Union[str, Path]) -> Dict[str, int]:
    path = Path(path)
    with path.open() as handle:
        vocab = json.load(handle)
    if dummy_index(vocab) is None:
        raise ValueError(f"{path} is missing the required {DUMMY_TOKEN!r} token")
    return {str(k): int(v) for k, v in vocab.items()}

def dummy_index(vocab: Mapping[str, int], dummy: str = DUMMY_TOKEN) -> int | None:
    if dummy not in vocab:
        return None
    return int(vocab[dummy])

def encode_sequences(
    sequences: Iterable[str],
    vocab: Mapping[str, int],
    max_length: int,
    dummy: str = DUMMY_TOKEN,
):
    """Encode sequences with the vocabulary used during training."""
    import numpy as np

    pad = int(vocab[dummy])
    materialized = [str(sequence)[:max_length] for sequence in sequences]
    encoded = np.full((len(materialized), int(max_length)), pad, dtype=np.int64)
    for index, truncated in enumerate(materialized):
        for offset, char in enumerate(truncated):
            encoded[index, offset] = int(vocab.get(char, pad))
    return encoded

class TransformerDTATokenizer:
    """ProtBERT residue tokens and ChemBERTa SMILES tokens.

    ProtBERT requires uppercase, space-separated residues. Raw whole sequences
    otherwise collapse to one unknown WordPiece token.
    """
    preprocessing_version = TOKENIZATION_VERSION

    def __init__(
        self,
        protein_bert_name: str = "Rostlab/prot_bert",
        ligand_bert_name: str = "DeepChem/ChemBERTa-100M-MLM",
        protein_revision: str | None = None,
        ligand_revision: str | None = None,
        max_protein_length: int = 2000,
        max_ligand_length: int = 500,
    ):
        from transformers import BertTokenizer, RobertaTokenizerFast

        protein_kwargs = {"revision": protein_revision} if protein_revision else {}
        ligand_kwargs = {"revision": ligand_revision} if ligand_revision else {}
        self.protein_tokenizer = BertTokenizer.from_pretrained(
            protein_bert_name, do_lower_case=False, **protein_kwargs)
        self.ligand_tokenizer = RobertaTokenizerFast.from_pretrained(ligand_bert_name, **ligand_kwargs)
        self.validate_ligand_tokenizer()
        self.protein_bert_name = protein_bert_name
        self.ligand_bert_name = ligand_bert_name
        self.protein_revision = protein_revision or "default"
        self.ligand_revision = ligand_revision or "default"
        self.max_protein_length = int(max_protein_length)
        self.max_ligand_length = int(max_ligand_length)
        if self.max_protein_length < 3 or self.max_ligand_length < 3:
            raise ValueError("Token limits must leave room for content and special tokens")

    def validate_ligand_tokenizer(self) -> None:
        # Token IDs must match the pretrained embeddings and preserve SMILES syntax.
        probes = ['CCO', 'ClCCl', 'BrCCBr', '[NH3+][C@H](C)C(=O)[O-]', 'C[C@@H](O)Br']
        for smiles in probes:
            ids = self.ligand_tokenizer(smiles, truncation=False)['input_ids']
            decoded = self.ligand_tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if self.ligand_tokenizer.unk_token_id in ids or decoded != smiles:
                raise ValueError(
                    f'Ligand tokenizer loses chemical information: {smiles!r} -> {decoded!r}. '
                    'Use a model and tokenizer exported together. The ChemBERTa-77M-MLM '
                    'export available at the time of these runs failed this check.')

    def _check_ligand_tokens(self, tokens) -> None:
        ids = np.asarray(tokens['input_ids'])
        if (ids == self.ligand_tokenizer.unk_token_id).any():
            raise ValueError('Ligand tokenizer produced unknown tokens; inspect input and vocabulary')

    @staticmethod
    def prepare_protein(sequence: str) -> str:
        sequence = ''.join(str(sequence).split()).upper()
        if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWYUZOBX'):
            raise ValueError(f"Invalid amino-acid sequence: {sequence[:60]!r}")
        return ' '.join(sequence.translate(str.maketrans('UZOB', 'XXXX')))

    def _check_protein_tokens(self, tokens) -> None:
        ids = np.asarray(tokens['input_ids'])
        mask = np.asarray(tokens['attention_mask']).astype(bool)
        if ((ids == self.protein_tokenizer.unk_token_id) & mask).any():
            raise ValueError("ProtBERT produced [UNK] for a validated residue sequence; check vocabulary and preprocessing")

    def tokenize_protein(self, sequence: str) -> dict[str, torch.Tensor]:
        self._require_complete_proteins([sequence])
        tokens = self.protein_tokenizer(
            self.prepare_protein(sequence),
            truncation=True,
            max_length=self.max_protein_length,
            padding="max_length",
            return_tensors="pt",
        )
        self._check_protein_tokens(tokens)
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }

    def tokenize_ligand(self, smiles: str) -> dict[str, torch.Tensor]:
        tokens = self.ligand_tokenizer(
            str(smiles),
            truncation=True,
            max_length=self.max_ligand_length,
            padding="max_length",
            return_tensors="pt",
        )
        self._check_ligand_tokens(tokens)
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }

    def tokenize_proteins(self, sequences: Sequence[str]) -> dict[str, np.ndarray]:
        self._require_complete_proteins(sequences)
        tokens = self.protein_tokenizer(
            [self.prepare_protein(sequence) for sequence in sequences],
            truncation=True,
            max_length=self.max_protein_length,
            padding="max_length",
            return_tensors="np",
        )
        self._check_protein_tokens(tokens)
        return {
            "input_ids": np.asarray(tokens["input_ids"]),
            "attention_mask": np.asarray(tokens["attention_mask"]),
        }

    def _require_complete_proteins(self, sequences: Sequence[str]) -> None:
        for sequence in sequences:
            if len(self.prepare_protein(sequence).split()) + 2 > self.max_protein_length:
                raise ValueError('Protein/pocket plus two special tokens exceeds the input limit; increase it instead of truncating the pocket')

    def tokenize_ligands(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        tokens = self.ligand_tokenizer(
            [str(value) for value in smiles],
            truncation=True,
            max_length=self.max_ligand_length,
            padding="max_length",
            return_tensors="np",
        )
        self._check_ligand_tokens(tokens)
        return {
            "input_ids": np.asarray(tokens["input_ids"]),
            "attention_mask": np.asarray(tokens["attention_mask"]),
        }
