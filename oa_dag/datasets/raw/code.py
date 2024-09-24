from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset
from oa_dag.datasets.base import RawDataset, RawSample
from oa_dag.utils import jsonlines_load


__all__ = ['MagiCoderTrainDataset']


class MagiCoderDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT, trust_remote_code=True)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['instruction'].strip()
        answer = data['response'].strip()
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)


class MagiCoderTrainDataset(MagiCoderDataset):
    NAME: str = 'MagiCoder'
    PATH: str = 'ise-uiuc/Magicoder-Evol-Instruct-110K'
    SPLIT: str = 'train'


class HumanEvalDataset(MagiCoderDataset):
    NAME: str = 'HumanEval'
    PATH: str = '/local/home/yuxi_xie/Projects/OA-DAG/datasets/human-eval/data/HumanEval.jsonl'
    
    def __init__(self, path: str | None = None) -> None:
        self.data = jsonlines_load(self.PATH)
    
    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = 'Complete the code below:\n\n{}'.format(data['prompt'])
        answer = 'Here is the final code:\n\n{}{}'.format(data['prompt'], data['canonical_solution'])
        return RawSample(input=input, answer=answer)


class MBPPTestDataset(MagiCoderDataset):
    NAME: str = 'MBPP/test'
    PATH: str = 'google-research-datasets/mbpp'
    SPLIT: str = 'test'
    
    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT, trust_remote_code=True)
    
    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = 'You are an expert Python programmer, and here is your task:\n{}'.format(data['text'])
        # input += '\nYour code should pass these tests:\n{}'.format('\n'.join(data['test_list']))
        answer = data['code']
        return RawSample(input=input, answer=answer)
