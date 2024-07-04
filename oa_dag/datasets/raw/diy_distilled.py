from __future__ import annotations

from oa_dag.utils import json_load
from oa_dag.datasets.base import RawDataset, RawSample

__all__ = ['DistrilledDataset', 'DistrilledDatasetEval']


class DistrilledDataset(RawDataset):
    NAME: str = 'distilled'

    def __init__(self, path: str | None = None) -> None:
        self.data = json_load(path or '/share/edc/home/yuxi_xie/ShareGPT_Vicuna_unfiltered/data/mistral.json')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        dialogue = [content['content'] for content in data]
        return RawSample(dialogue=dialogue)

    def __len__(self) -> int:
        return len(self.data)

class DistrilledDatasetEval(RawDataset):
    NAME: str = 'distilled/eval'

    def __init__(self, path: str | None = None) -> None:
        self.data = json_load(path or '/share/edc/home/yuxi_xie/ShareGPT_Vicuna_unfiltered/data/mistral.json')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        # dialogue = [content['content'] for content in data]
        input = data[0]['content'] if len(data) else ''
        answer = data[1]['content'] if len(data) > 1 else ''
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)