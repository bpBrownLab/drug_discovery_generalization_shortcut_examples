"""Load released weights, restoring the two frozen encoders when needed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .networks import DeepDTA, TransformerDTA
from .inputs import load_vocab
from .assets import tokenizers

MODEL_DIR = Path(__file__).resolve().parent

ENCODERS = {
    "protein_bert": ("Rostlab/prot_bert", "7a894481acdc12202f0a415dd567f6cfdb698908"),
    "ligand_bert": ("DeepChem/ChemBERTa-100M-MLM", "f5c45f44d3061f0346888f5c09db17ec1146d29d"),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(state):
    """Fingerprint tensor names, shapes, dtypes, and exact CPU values."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(data).cast("B"))
    return digest.hexdigest()


def transformer_skeleton(metadata):
    """Construct the recorded architecture without allocating its weights."""
    from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel

    config = metadata["config"]
    for prefix, (name, revision) in ENCODERS.items():
        kind = "protein" if prefix == "protein_bert" else "ligand"
        if (config["tokenizers"][f"{kind}_name"], config["tokenizers"][f"{kind}_revision"]) != (name, revision):
            raise ValueError(f"Unexpected {kind} encoder or revision")
        if not metadata["architecture"][f"freeze_{prefix}"]:
            raise ValueError("Reduced checkpoints require both pretrained encoders to be frozen")
    assets = tokenizers() / "transformer"
    pc = BertConfig.from_json_file(str(assets / "protein" / "config.json"))
    lc = RobertaConfig.from_json_file(str(assets / "ligand" / "config.json"))
    pc._attn_implementation = lc._attn_implementation = "eager"
    architecture = {k: v for k, v in metadata["architecture"].items() if k != "pooling"}
    with torch.device("meta"):
        model = TransformerDTA(
            **architecture, load_pretrained=False,
            protein_d_model=pc.hidden_size, ligand_d_model=lc.hidden_size,
            prediction_head=config["prediction_head"],
        )
        model.protein_bert = BertModel(pc, add_pooling_layer=False)
        model.ligand_bert = RobertaModel(lc, add_pooling_layer=False)
    return model


def finish_transformer(model, state):
    # Strict loading rejects a missing trained layer or an unexpected tensor.
    model.load_state_dict(state, strict=True, assign=True)
    for prefix in ENCODERS:
        encoder = getattr(model, prefix)
        encoder.requires_grad_(False)
        emb = encoder.embeddings
        emb.position_ids = torch.arange(encoder.config.max_position_embeddings).expand((1, -1))
        emb.token_type_ids = torch.zeros(emb.position_ids.size(), dtype=torch.long)
    return model.eval()


def load_frozen_encoders(*, local_files_only=False, encoder_paths=None):
    """Use pinned upstream revisions, or explicitly supplied local snapshots."""
    from transformers import BertModel, RobertaModel

    encoder_paths = encoder_paths or {}
    states = {}
    for prefix, (name, revision) in ENCODERS.items():
        cls = BertModel if prefix == "protein_bert" else RobertaModel
        source = encoder_paths.get(prefix, name)
        options = {} if prefix in encoder_paths else {"revision": revision}
        encoder = cls.from_pretrained(
            source, add_pooling_layer=False, local_files_only=local_files_only,
            attn_implementation="eager", **options,
        )
        states[prefix] = encoder.state_dict()
    return states


def load_checkpoint(checkpoint, *, local_files_only=False, encoder_paths=None):
    """Return an evaluation model and its recorded training metadata.

    Complete original checkpoints remain supported. Released TransformerDTA
    files contain every trained tensor and require the pinned frozen encoders.
    """
    checkpoint = Path(checkpoint)
    digest = file_sha256(checkpoint)
    entries = json.loads((MODEL_DIR / "checkpoints.json").read_text())
    entry = next((row for row in entries if digest in {row["sha256"], row.get("original_sha256")}), None)
    if entry is None:
        raise ValueError("Checkpoint does not match any of the 40 released models")
    is_original = digest == entry.get("original_sha256", entry["sha256"])
    if digest != entry["sha256"] and not is_original:
        raise ValueError(f"Checkpoint checksum does not match: {checkpoint.name}")
    metadata = entry
    config = metadata["config"]
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if entry["model"] == "deepdta":
        protein_vocab = tokenizers() / "deepdta/protein_dict.json"
        ligand_vocab = tokenizers() / "deepdta/ligand_dict.json"
        for kind, path in [("protein", protein_vocab), ("ligand", ligand_vocab)]:
            if file_sha256(path) != metadata[f"{kind}_vocab_sha256"]:
                raise ValueError(f"The {kind} vocabulary differs from the training record")
        pv = load_vocab(protein_vocab)
        lv = load_vocab(ligand_vocab)
        model = DeepDTA(
            len(pv), len(lv), channel=config["channel"],
            protein_kernel_sizes=config["protein_kernel_sizes"],
            ligand_kernel_sizes=config["ligand_kernel_sizes"],
            fc_hidden_layers=config["fc_hidden_layers"], output_dim=2,
            prediction_head=config["prediction_head"],
        )
        model.load_state_dict(state, strict=True)
        return model.eval(), metadata

    model = transformer_skeleton(metadata)
    if not is_original:
        if entry.get("format") != "transformerdta_trainable_v1":
            raise ValueError("Unsupported reduced checkpoint format")
        expected = {key for key in model.state_dict() if not key.startswith(tuple(p + "." for p in ENCODERS))}
        if set(state) != expected:
            raise ValueError("Reduced checkpoint must contain exactly all non-encoder tensors")
        encoders = load_frozen_encoders(local_files_only=local_files_only, encoder_paths=encoder_paths)
        for prefix, tensors in encoders.items():
            if tensor_sha256(tensors) != entry["frozen_encoders"][prefix]["tensor_sha256"]:
                raise ValueError(f"Frozen encoder differs from the training checkpoint: {prefix}")
            state.update({f"{prefix}.{key}": value for key, value in tensors.items()})
    return finish_transformer(model, state), metadata
