"""DeepDTA and TransformerDTA architectures used in the Perspective."""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import os

import torch
import torch.nn as nn

PREDICTION_HEADS = ("mlp", "linear")


def _validate_prediction_head(name: str) -> str:
    normalized = str(name).lower()
    if normalized not in PREDICTION_HEADS:
        raise ValueError(
            f"prediction_head must be one of {PREDICTION_HEADS}, got {name!r}"
        )
    return normalized


class Conv1dTower(nn.Module):
    """Three Conv/ReLU stages, then global max pool."""

    def __init__(self, vocab_size: int, channel: int, kernel_sizes: list[int],
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=128)
        self.conv1 = nn.Conv1d(128, channel, kernel_sizes[0], stride, padding)
        self.conv2 = nn.Conv1d(channel, channel * 2, kernel_sizes[1], stride, padding)
        self.conv3 = nn.Conv1d(channel * 2, channel * 3, kernel_sizes[2], stride, padding)
        self.relu = nn.ReLU()
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.globalmaxpool(x)
        return x.squeeze(-1)


class DeepDTA(nn.Module):
    def __init__(
        self,
        pro_vocab_size: int,
        lig_vocab_size: int,
        channel: int = 32,
        protein_kernel_sizes: list[int] | None = None,
        ligand_kernel_sizes: list[int] | None = None,
        fc_hidden_layers: list[int] | None = None,
        output_dim: int = 2,
        prediction_head: str = "mlp",
    ):
        super().__init__()
        protein_kernel_sizes = protein_kernel_sizes or [4, 8, 12]
        ligand_kernel_sizes = ligand_kernel_sizes or [4, 6, 8]
        fc_hidden_layers = fc_hidden_layers or [1024, 1024, 512]
        self.ligand_conv = Conv1dTower(lig_vocab_size, channel, ligand_kernel_sizes)
        self.protein_conv = Conv1dTower(pro_vocab_size, channel, protein_kernel_sizes)
        self.prediction_head = _validate_prediction_head(prediction_head)
        if self.prediction_head == "mlp":
            # The saved checkpoints key on these layer names. Do not rename.
            self.fc1 = nn.Linear(channel * 6, fc_hidden_layers[0])
            self.fc2 = nn.Linear(fc_hidden_layers[0], fc_hidden_layers[1])
            self.fc3 = nn.Linear(fc_hidden_layers[1], fc_hidden_layers[2])
            self.fc4 = nn.Linear(fc_hidden_layers[2], output_dim)
        else:
            self.linear_predictor = nn.Linear(channel * 6, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, protein: torch.Tensor, ligand: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.ligand_conv(ligand), self.protein_conv(protein)), dim=1)
        if self.prediction_head == "linear":
            return self.linear_predictor(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class AffinityMLP(nn.Module):
    """GELU MLP whose layer names match the saved TransformerDTA state dict."""

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int,
                 dropout_rates: list[float] | None = None):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = [0.0, 0.25, 0.25]
        layers: list[nn.Module] = []
        in_size = input_size
        if dropout_rates[0] > 0:
            layers.append(nn.Dropout(p=dropout_rates[0]))
        for i, hidden in enumerate(hidden_sizes):
            linear = nn.Linear(in_size, hidden)
            nn.init.kaiming_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.GELU())
            drop_p = dropout_rates[i + 1] if i + 1 < len(dropout_rates) else 0.0
            if drop_p > 0:
                layers.append(nn.Dropout(p=drop_p))
            in_size = hidden
        output = nn.Linear(in_size, output_size)
        nn.init.kaiming_normal_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.model = nn.Sequential(*layers)
        self.dropout_at_inference = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and not self.dropout_at_inference:
            self.model.eval()
        return self.model(x)


class ProteinEncoder(nn.Module):
    def __init__(self, d_model: int = 1024, nhead: int = 4, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.transformer(x, src_key_padding_mask=mask)


class LigandEncoder(nn.Module):
    def __init__(self, d_model: int = 384, nhead: int = 3, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.transformer(x, src_key_padding_mask=mask)


class CrossAttentionTransformer(nn.Module):
    def __init__(self, d_model: int = 384, nhead: int = 3, num_layers: int = 1, dropout: float = 0.15):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, target, memory, target_mask=None, memory_mask=None):
        return self.transformer(
            target, memory,
            tgt_key_padding_mask=target_mask,
            memory_key_padding_mask=memory_mask,
        )


class TransformerDTA(nn.Module):
    def __init__(
        self,
        protein_bert_name: str = "Rostlab/prot_bert",
        ligand_bert_name: str = "DeepChem/ChemBERTa-100M-MLM",
        protein_bert_revision: str | None = None,
        ligand_bert_revision: str | None = None,
        freeze_protein_bert: bool = True,
        freeze_ligand_bert: bool = True,
        protein_nhead: int = 4,
        ligand_nhead: int = 3,
        protein_encoder_layers: int = 2,
        ligand_encoder_layers: int = 2,
        num_cross_layers: int = 1,
        protein_dropout: float = 0.15,
        ligand_dropout: float = 0.15,
        cross_attention_dropout: float = 0.15,
        output_dim: int = 2,
        prediction_head: str = "mlp",
        disable_protein_encoder: bool = False,
        disable_ligand_encoder: bool = False,
        cross_attention_mode: str = "ligand_query",
        bidirectional_fusion_dropout: float = 0.15,
        trim_protein_padding: bool = True,
        load_pretrained: bool = True,
        protein_d_model: int = 1024,
        ligand_d_model: int = 384,
    ):
        super().__init__()
        valid_modes = {"ligand_query", "protein_query", "bidirectional"}
        if cross_attention_mode not in valid_modes:
            raise ValueError(
                f"cross_attention_mode must be one of {sorted(valid_modes)}, "
                f"got {cross_attention_mode!r}"
            )
        self.output_dim = output_dim
        self.prediction_head = _validate_prediction_head(prediction_head)
        self.cross_attention_mode = cross_attention_mode
        self.use_protein_encoder = not disable_protein_encoder
        self.use_ligand_encoder = not disable_ligand_encoder
        self.trim_protein_padding = bool(trim_protein_padding)
        self.freeze_protein_bert = bool(freeze_protein_bert)
        self.freeze_ligand_bert = bool(freeze_ligand_bert)
        self.protein_bert = None
        self.ligand_bert = None

        if load_pretrained:
            from transformers import BertModel, RobertaModel

            protein_kwargs = {"revision": protein_bert_revision} if protein_bert_revision else {}
            ligand_kwargs = {"revision": ligand_bert_revision} if ligand_bert_revision else {}
            self.protein_bert = BertModel.from_pretrained(protein_bert_name, add_pooling_layer=False, **protein_kwargs)
            self.ligand_bert = RobertaModel.from_pretrained(ligand_bert_name, add_pooling_layer=False, **ligand_kwargs)
            self.protein_d_model = self.protein_bert.config.hidden_size
            self.ligand_d_model = self.ligand_bert.config.hidden_size
            for param in self.protein_bert.parameters():
                param.requires_grad = not freeze_protein_bert
            for param in self.ligand_bert.parameters():
                param.requires_grad = not freeze_ligand_bert
        else:
            self.protein_d_model = protein_d_model
            self.ligand_d_model = ligand_d_model

        # All new fusion paths operate in the selected ChemBERTa checkpoint's hidden space.
        # This keeps bidirectional inference practical for hundreds of
        # thousands of examples rather than introducing a second 1024-wide
        # decoder for the reverse direction.
        self.attention_d_model = self.ligand_d_model
        if self.use_protein_encoder:
            self.protein_transformer = ProteinEncoder(
                d_model=self.protein_d_model, nhead=protein_nhead,
                num_layers=protein_encoder_layers, dropout=protein_dropout,
            )
        if self.use_ligand_encoder:
            self.ligand_transformer = LigandEncoder(
                d_model=self.ligand_d_model, nhead=ligand_nhead,
                num_layers=ligand_encoder_layers, dropout=ligand_dropout,
            )
        self.protein_projection = nn.Linear(
            self.protein_d_model,
            self.attention_d_model,
        )
        if cross_attention_mode in {"ligand_query", "bidirectional"}:
            self.cross_attention = CrossAttentionTransformer(
                d_model=self.attention_d_model,
                nhead=ligand_nhead,
                num_layers=num_cross_layers,
                dropout=cross_attention_dropout,
            )
        if cross_attention_mode in {"protein_query", "bidirectional"}:
            if self.attention_d_model % protein_nhead:
                raise ValueError(
                    f"Fusion dimension {self.attention_d_model} must be divisible "
                    f"by protein_nhead={protein_nhead}"
                )
            self.reverse_cross_attention = CrossAttentionTransformer(
                d_model=self.attention_d_model,
                nhead=protein_nhead,
                num_layers=num_cross_layers,
                dropout=cross_attention_dropout,
            )
        if cross_attention_mode == "bidirectional":
            self.bidirectional_fusion = nn.Sequential(
                nn.LayerNorm(2 * self.attention_d_model),
                nn.Linear(2 * self.attention_d_model, self.attention_d_model),
                nn.GELU(),
                nn.Dropout(bidirectional_fusion_dropout),
            )
        self.pool = nn.AdaptiveAvgPool1d(1)
        if self.prediction_head == "mlp":
            self.affinity_predictor = AffinityMLP(
                input_size=self.attention_d_model,
                hidden_sizes=[256, 256],
                output_size=self.output_dim,
                dropout_rates=[0.0, 0.25, 0.25],
            )
        else:
            self.affinity_predictor = nn.Linear(
                self.attention_d_model,
                self.output_dim,
            )
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        # requires_grad=False does not disable dropout. A frozen feature
        # extractor must stay deterministic while the trainable head learns.
        for name in ('protein', 'ligand'):
            encoder = getattr(self, f'{name}_bert', None)
            if encoder is not None and getattr(self, f'freeze_{name}_bert', False):
                encoder.eval()
        return self

    def embed(self, protein_tokens, ligand_tokens, protein_attention_mask=None, ligand_attention_mask=None):
        if self.protein_bert is None or self.ligand_bert is None:
            raise RuntimeError("TransformerDTA was constructed without pretrained encoders")
        protein_embed = self.protein_bert(
            protein_tokens, attention_mask=protein_attention_mask, return_dict=True,
        ).last_hidden_state
        ligand_embed = self.ligand_bert(
            ligand_tokens, attention_mask=ligand_attention_mask, return_dict=True,
        ).last_hidden_state
        return protein_embed, ligand_embed

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return values.mean(dim=1)
        weights = attention_mask.to(dtype=values.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return values.masked_fill(~attention_mask.bool().unsqueeze(-1), 0).sum(dim=1) / denominator

    def fuse(self, protein_embed, ligand_embed, protein_attention_mask=None, ligand_attention_mask=None):
        protein_mask = ~protein_attention_mask.bool() if protein_attention_mask is not None else None
        ligand_mask = ~ligand_attention_mask.bool() if ligand_attention_mask is not None else None
        protein_encoded = (
            self.protein_transformer(protein_embed, protein_mask)
            if self.use_protein_encoder else protein_embed
        )
        ligand_encoded = (
            self.ligand_transformer(ligand_embed, ligand_mask)
            if self.use_ligand_encoder else ligand_embed
        )
        protein_fusion = self.protein_projection(protein_encoded)
        ligand_summary = None
        protein_summary = None
        if self.cross_attention_mode in {"ligand_query", "bidirectional"}:
            ligand_interaction = self.cross_attention(
                target=ligand_encoded,
                memory=protein_fusion,
                target_mask=ligand_mask,
                memory_mask=protein_mask,
            )
            ligand_summary = self._masked_mean(ligand_interaction, ligand_attention_mask)
        if self.cross_attention_mode in {"protein_query", "bidirectional"}:
            protein_interaction = self.reverse_cross_attention(
                target=protein_fusion,
                memory=ligand_encoded,
                target_mask=protein_mask,
                memory_mask=ligand_mask,
            )
            protein_summary = self._masked_mean(
                protein_interaction,
                protein_attention_mask,
            )
        if self.cross_attention_mode == "ligand_query":
            pooled = ligand_summary
        elif self.cross_attention_mode == "protein_query":
            pooled = protein_summary
        else:
            pooled = self.bidirectional_fusion(
                torch.cat((ligand_summary, protein_summary), dim=-1)
            )
        return self.affinity_predictor(pooled)

    def forward(self, protein_tokens, ligand_tokens,
                protein_attention_mask=None, ligand_attention_mask=None):
        # Protein positions that are padding for every sample cannot contribute:
        # BERT masks them as keys and cross-attention masks them as memory. Avoid
        # paying ProtBERT's quadratic attention cost for those trailing columns.
        if self.trim_protein_padding and protein_attention_mask is not None:
            used_columns = protein_attention_mask.bool().any(dim=0).nonzero()
            if used_columns.numel():
                width = int(used_columns[-1].item()) + 1
                protein_tokens = protein_tokens[:, :width]
                protein_attention_mask = protein_attention_mask[:, :width]
        if ligand_attention_mask is not None:
            used_columns = ligand_attention_mask.bool().any(dim=0).nonzero()
            if used_columns.numel():
                width = int(used_columns[-1].item()) + 1
                ligand_tokens = ligand_tokens[:, :width]
                ligand_attention_mask = ligand_attention_mask[:, :width]
        protein_embed, ligand_embed = self.embed(
            protein_tokens, ligand_tokens, protein_attention_mask, ligand_attention_mask,
        )
        return self.fuse(protein_embed, ligand_embed, protein_attention_mask, ligand_attention_mask)


def strip_module_prefix(state_dict: dict) -> dict:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key[len("module."):] if key.startswith("module.") else key: value
            for key, value in state_dict.items()}


def load_state_dict(model: nn.Module, path: str, map_location: str | torch.device = "cpu") -> nn.Module:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload in {path}")
    model.load_state_dict(strip_module_prefix(payload), strict=True)
    return model


def save_state_dict(model: nn.Module, path: str) -> None:
    to_save = model.module if hasattr(model, "module") else model
    state = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v
             for k, v in to_save.state_dict().items()}
    path=Path(path)
    temporary=path.with_name('.'+path.name+'.tmp')
    torch.save(state, temporary)
    os.replace(temporary,path)
