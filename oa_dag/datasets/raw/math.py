from __future__ import annotations

from datasets import load_dataset
from oa_dag.datasets.base import RawDataset, RawSample


__all__ = ['MetaMathDataset']

def get_input(query):
    if query.find('\n') == -1:
        return ''
    return '\n'.join(query.split('\n')[1:])


class MetaMathDataset(RawDataset):
    NAME: str = 'MetaMath'
    ALIASES: tuple[str, ...] = ('meta-math',)

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or 'meta-math/MetaMathQA', split='train')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        instruction = data['query'].split('\n')[0]
        _input = get_input(data['query'])
        input = f'{instruction}\n\n### Input:\n{_input}' if _input else instruction
        answer = data['response']
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
