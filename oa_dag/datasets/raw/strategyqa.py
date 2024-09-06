from __future__ import annotations

from typing import ClassVar

from oa_dag.utils import jsonlines_load
from oa_dag.datasets.base import RawDataset, RawSample


__all__ = ['SQADataset']


class SQADataset(RawDataset):
    NAME: str = 'SQA'

    def __init__(self, path: str | None = None) -> None:
        self.data = jsonlines_load('/home/yuxi/Projects/OA-DAG/datasets/strategyqa.json')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        instruction = data['question'].split('\n')[0]
        input = f'{instruction}\nAnswer with Yes or No'
        # input = f'Choose your reply from the options at the end.\n\n{instruction}\nOPTIONS:\n- yes\n- no'
        answer = data['explanation'].strip()
        if answer.lower().startswith('yes.') or answer.lower().startswith('no.'):
            answer = '.'.join(answer.split('.')[1:]) + '\nThe answer is: ' + answer.split('.')[0]
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
