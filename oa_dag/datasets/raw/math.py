from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset
from oa_dag.datasets.base import RawDataset, RawSample


__all__ = ['MetaMathTrainDataset', 'MetaMathValidDataset', 'GSM8KDataset']

def get_input(query):
    if query.find('\n') == -1:
        return ''
    return '\n'.join(query.split('\n')[1:])


class MetaMathDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        instruction = data['query'].split('\n')[0]
        _input = get_input(data['query'])
        input = f'{instruction}\n\n### Input:\n{_input}' if _input else instruction
        answer = data['response']
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)

class MetaMathTrainDataset(MetaMathDataset):
    NAME: str = 'MetaMath'
    ALIASES: tuple[str, ...] = ('meta-math',)
    PATH: str = 'meta-math/MetaMathQA'
    SPLIT: str = 'train'

class MetaMathValidDataset(MetaMathDataset):
    NAME: str = 'MetaMath/valid'
    ALIASES: tuple[str, ...] = ('meta-math/40k',)
    PATH: str = 'meta-math/MetaMathQA-40K'
    SPLIT: str = 'train'

class GSM8KDataset(RawDataset):
    NAME: str = 'GSM8K'
    PATH: str = 'openai/gsm8k'
    SPLIT: str = 'test'
    
    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, 'main', split=self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['question']
        answer = data['answer']
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)