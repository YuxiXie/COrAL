from __future__ import annotations

import regex
from typing import ClassVar
from tqdm import tqdm
from string import punctuation

from coral.datasets.base import RawDataset, RawSample
from coral.utils import jsonlines_load, sample_from_dataset


__all__ = [
    'LogiQADataset', 'LogiQAZhDataset', 'LogiQAOodDataset',
    'ReclorDataset', 'ALDataset',
    'ControlDataset', 'ConjNLIDataset',
]

DIR = './datasets/LogiEval/Data'


class LogiEvalDataset(RawDataset):
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = jsonlines_load(path or self.PATH)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        instruction = data['input'][0]['content'].strip().replace('Instructions: ', '').strip()
        _input = data['input'][-1]['content'].strip().replace('\nAnswer:', '').strip()
        input = f'{instruction}\n\n### Input:\n{_input}' if _input else instruction
        answer = f'The answer is: {data["ideal"]}'
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)

class LogiQADataset(LogiEvalDataset):
    NAME: str = 'LogiQA'
    PATH: str = f'{DIR}/logiqa.jsonl'

class LogiQAZhDataset(LogiEvalDataset):
    NAME: str = 'LogiQA/zh'
    PATH: str = f'{DIR}/logiqa_zh.jsonl'

class LogiQAOodDataset(LogiEvalDataset):
    NAME: str = 'LogiQA/ood'
    PATH: str = f'{DIR}/logiqa_ood.jsonl'

class ReclorDataset(LogiEvalDataset):
    NAME: str = 'Reclor'
    PATH: str = f'{DIR}/reclor.jsonl'

class ALDataset(LogiEvalDataset):
    NAME: str = 'AL'
    PATH: str = f'{DIR}/ar_lsat.jsonl'

class ControlDataset(LogiEvalDataset):
    NAME: str = 'Control'
    PATH: str = f'{DIR}/control.jsonl'

class ConjNLIDataset(LogiEvalDataset):
    NAME: str = 'ConjNLI'
    PATH: str = f'{DIR}/conjnli.jsonl'
