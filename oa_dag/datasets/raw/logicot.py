from __future__ import annotations

import regex
from typing import ClassVar
from tqdm import tqdm
from string import punctuation
from collections import defaultdict

from datasets import load_dataset
from oa_dag.datasets.base import RawDataset, RawSample
from oa_dag.utils import json_load, sample_from_dataset, jsonlines_load


__all__ = ['LogiCoTTrainDataset']


def process_logicot(rawdata):
    task_dict = defaultdict(dict)
    for dt in tqdm(rawdata):
        dt['input'] = dt['input'].strip()
        dt['instruction'] = dt['instruction'].strip()
        if dt['input'] not in task_dict[dt['instruction']]:
            task_dict[dt['instruction']][dt['input']] = []
        task_dict[dt['instruction']][dt['input']].append(dt['output'])
    data = []
    for k, v in task_dict.items():
        for kk, vv in v.items():
            vv_set = list(set(vv))
            for vvv in vv_set:
                data.append({
                    'instruction': k,
                    'input': kk,
                    'output': vvv,
                })
    return data


class LogiCoTDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = process_logicot(load_dataset(path or self.PATH, split=self.SPLIT, trust_remote_code=True))

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


class LogiCoTFilterTrainDataset(LogiCoTDataset):
    NAME: str = 'LogiCoT/filter'
    PATH: str = '/local/home/yuxi_xie/Projects/OA-DAG/datasets/logicot_filter_norm.jsonl'
    SPLIT: str = 'train'
    
    def __init__(self, path: str | None = None) -> None:
        self.data = process_logicot(jsonlines_load(path or self.PATH))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        instruction = data['instruction'].strip().replace('Instructions:', '').strip()
        _input = data['input'].strip()
        input = f'{instruction}\n\n### Input:\n{_input}' if _input else instruction
        answer = data['output'].strip()
        return RawSample(input=input, answer=answer)
    