from __future__ import annotations

import regex
from typing import ClassVar
from tqdm import tqdm
from string import punctuation

from datasets import load_dataset
from oa_dag.datasets.base import RawDataset, RawSample
from oa_dag.utils import json_load, sample_from_dataset


__all__ = ['LogiCoTTrainDataset']


class LogiCoTDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT, trust_remote_code=True)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        instruction = data['instruction'].strip()
        _input = data['input'].strip()
        input = f'{instruction}\n\n### Input:\n{_input}' if _input else instruction
        answer = data['output'].strip()
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)

class LogiCoTTrainDataset(LogiCoTDataset):
    NAME: str = 'LogiCoT'
    PATH: str = 'datatune/LogiCoT'
    SPLIT: str = 'train'
