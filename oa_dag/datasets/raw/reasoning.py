from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset
from oa_dag.datasets.base import RawDataset, RawSample


__all__ = ['MagpieTrainDataset']


class MagpieDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT)
    
    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        if 'conversations' in data:
            conversations = data['conversations']
            input = conversations[0]['value'].strip()
            answer = conversations[1]['value'].strip()
        else:
            instruction = data['instruction'].strip()
            input = instruction
            answer = data['response'].strip()
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)


class MagpieReasoningTrainDataset(MagpieDataset):
    NAME: str = 'MagpieReasoning'
    PATH: str = 'Magpie-Align/Magpie-Reasoning-150K'
    SPLIT: str = 'train'


class MagpieProTrainDataset(MagpieDataset):
    NAME: str = 'MagpiePro'
    PATH: str = 'Magpie-Align/Magpie-Pro-300K-Filtered'
    SPLIT: str = 'train'


class MagpieTrainDataset(MagpieDataset):
    NAME: str = 'Magpie'
    SPLIT: str = 'train'
    
    def __init__(self, path: str | None = None) -> None:
        self.reasoning_data = load_dataset('Magpie-Align/Magpie-Reasoning-150K', split=self.SPLIT)
        self.pro_data = load_dataset('Magpie-Align/Magpie-Pro-300K-Filtered', split=self.SPLIT)
        self.data = list(self.reasoning_data) + list(self.pro_data)
    