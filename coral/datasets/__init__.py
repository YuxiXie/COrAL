"""Dataset classes."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset

from coral.datasets import raw
from coral.datasets.base import (
    CollatorBase,
    RawDataset,
    RawSample,
    TokenizedDataset,
    parse_dataset,
)
from coral.datasets.prompt_only import (
    PromptOnlyBatch,
    PromptOnlyCollator,
    PromptOnlyDataset,
    PromptOnlySample,
)
from coral.datasets.raw import *  # noqa: F403
from coral.datasets.supervised import (
    SupervisedBatch,
    SupervisedCollator,
    SupervisedDataset,
    SupervisedSample,
)


__all__ = [
    'DummyDataset',
    'parse_dataset',
    'RawDataset',
    'RawSample',
    'TokenizedDataset',
    'CollatorBase',
    'PromptOnlyDataset',
    'PromptOnlyCollator',
    'PromptOnlySample',
    'PromptOnlyBatch',
    'SupervisedDataset',
    'SupervisedCollator',
    'SupervisedSample',
    'SupervisedBatch',
    *raw.__all__,
]


class DummyDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {}
