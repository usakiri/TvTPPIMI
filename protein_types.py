from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ProteinSample:
    """Container produced by the dataset for each protein entry."""

    static: Optional[torch.Tensor]
    total_length_hint: Optional[int]
    special_tokens: Optional[torch.Tensor] = None


@dataclass
class ProteinBatch:
    """
    Collated protein batch passed to TvTPPIMI.

    Attributes:
        static: Optional tensor of shape (B, L_max, static_dim) with zero padding.
        mask: Boolean tensor (B, L_max) where True marks padded positions.
        lengths: Tensor (B,) with the true sequence lengths.
        special_tokens: Optional tensor (B, L_max) storing special token type ids.
    """

    static: Optional[torch.Tensor]
    mask: torch.Tensor
    lengths: torch.Tensor
    special_tokens: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "ProteinBatch":
        static = self.static.to(device) if self.static is not None else None
        special_tokens = self.special_tokens.to(device) if self.special_tokens is not None else None
        return ProteinBatch(
            static=static,
            mask=self.mask.to(device),
            lengths=self.lengths.to(device),
            special_tokens=special_tokens,
        )

    @property
    def batch_size(self) -> int:
        return int(self.lengths.shape[0])
