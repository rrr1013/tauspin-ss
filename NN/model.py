from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from config import (
    D_MODEL,
    DIM_FEEDFORWARD,
    DROPOUT,
    N_HEAD,
    N_LAYERS,
)
from dataset import CLS_TYPE, EVENT_TYPE, PFO_TYPE, TAU_TYPE, TRACK_TYPE


class TauSpinTransformer(nn.Module):
    def __init__(
        self,
        feature_dimensions: Mapping[str, int],
        decay_mode_num_embeddings: int,
        *,
        d_model: int = D_MODEL,
        n_head: int = N_HEAD,
        n_layers: int = N_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.event_projector = nn.Linear(feature_dimensions["event"], d_model)
        self.tau_projector = nn.Linear(feature_dimensions["tau"], d_model)
        self.track_projector = nn.Linear(feature_dimensions["track"], d_model)
        self.pfo_projector = nn.Linear(feature_dimensions["pfo"], d_model)

        self.object_type_embedding = nn.Embedding(6, d_model, padding_idx=0)
        self.tau_side_embedding = nn.Embedding(3, d_model, padding_idx=0)
        self.decay_mode_embedding = nn.Embedding(
            decay_mode_num_embeddings, d_model, padding_idx=0
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        object_type = batch["object_type"]
        projectors = (
            (EVENT_TYPE, "event_features", self.event_projector),
            (TAU_TYPE, "tau_features", self.tau_projector),
            (TRACK_TYPE, "track_features", self.track_projector),
            (PFO_TYPE, "pfo_features", self.pfo_projector),
        )
        tokens = None
        for type_id, feature_name, projector in projectors:
            projected = projector(batch[feature_name])
            mask = (object_type == type_id).unsqueeze(-1).to(projected.dtype)
            contribution = projected * mask
            tokens = (
                contribution
                if tokens is None
                else tokens + contribution
            )
        if tokens is None:
            raise RuntimeError("No input projectors are configured")

        tokens = (
            tokens
            + self.object_type_embedding(object_type)
            + self.tau_side_embedding(batch["tau_side"])
            + self.decay_mode_embedding(batch["decay_mode"])
        )

        batch_size = tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        cls_type = torch.full(
            (batch_size, 1),
            CLS_TYPE,
            dtype=torch.int64,
            device=tokens.device,
        )
        cls = cls + self.object_type_embedding(cls_type)
        tokens = torch.cat((cls, tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                batch["padding_mask"],
            ),
            dim=1,
        )

        encoded = self.encoder(
            tokens,
            src_key_padding_mask=padding_mask,
        )
        return self.classifier(encoded[:, 0]).squeeze(-1)
