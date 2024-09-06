from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset
from oa_dag.datasets.base import RawDataset, RawSample


__all__ = ['TuluTrainDataset']


class TuluDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT)
    
    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        dialogue = [message['content'] for message in data['messages']]
        return RawSample(dialogue=dialogue)

    def __len__(self) -> int:
        return len(self.data)


class TuluReasoningTrainDataset(TuluDataset):
    NAME: str = 'tulu-sft'
    PATH: str = 'allenai/tulu-v2-sft-mixture'
    SPLIT: str = 'train'


class TuluProTrainDataset(TuluDataset):
    NAME: str = 'SciRIFF'
    PATH: str = 'allenai/SciRIFF-train-mix'
    SPLIT: str = 'train'


class TuluTrainDataset(TuluDataset):
    NAME: str = 'TULU'
    SPLIT: str = 'train'
    
    def __init__(self, path: str | None = None) -> None:
        self.reasoning_data = load_dataset('allenai/SciRIFF-train-mix', split=self.SPLIT)
        self.pro_data = load_dataset('allenai/tulu-v2-sft-mixture', split=self.SPLIT)
        self.data = list(self.reasoning_data) + list(self.pro_data)
    