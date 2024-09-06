from __future__ import annotations

import regex
from typing import ClassVar
from tqdm import tqdm
from string import punctuation

from datasets import load_dataset
from oa_dag.datasets.base import RawDataset, RawSample
from oa_dag.utils import json_load, sample_from_dataset


__all__ = ['CoTTrainDataset']


def process_cot(rawdata):
    data = []
    for dt in tqdm(rawdata):
        src, tgt, ans = dt['source'].strip(), dt['target'].strip(), dt['rationale'].replace('\\n', '\n').strip()
        task = dt['task'].strip()
        
        idx = f' {ans}'.find(tgt)
        if idx < 0:
            exp = r'[\s"“`:\'\[\(]' + tgt.lower().strip('"').strip(punctuation).strip()
            idx = regex.search(exp, f' {ans.lower()}')
            idx = idx.span()[0] if idx is not None else -1
        if idx < 0 and (tgt.lower() in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 
                                        'ab', 'ac', 'ad', 'ae', 'af', 'bc', 'bd', 'be', 'bf',
                                        'cd', 'ce', 'cf', 'de', 'df', 'ef',
                                        'yes', 'no'] or regex.search(r'[\d\.]+', tgt)):
            idx = len(ans) - len(tgt)    
        if idx < 0: continue
        
        proportion = (idx + 1) / (len(ans) + 3 - len(tgt.strip('"').strip(punctuation).strip()))
        if proportion < .5: continue
        
        instruction = regex.split(r'[\n]+', src)[0]
        query = src[len(instruction):].strip()
        if task in ['rte', 'ropes', 'adversarial_qa@dbert', 'task_master', 'quoref', 'kilt_tasks@hotpotqa', 'race@middle', 'cos_e', 'drop', 'mnli', 'qed', 'trivia_qa', 'sciq', 'cosmos_qa', 'cola', 'nlg_bias', 'wiki_dialog', 'squad_v1', 'natural_questions', 'duorc@ParaphraseRC', 'task_master_input_inversion', 'anli_r2', 'web_questions', 'qnli', 'duorc@SelfRC', 'ade_corpus_v2', 'adversarial_qa@dbidaf', 'super_glue@record', 'super_glue@multirc', 'super_glue@boolq', 'adversarial_qa@droberta', 'gsm8k', 'quartz', 'root09', 'esnli', 'quarel', 'quac', 'piqa', 'openbookqa', 'wiki_hop', 'opp_115', 'qrecc', 'ecqa', 'snli', 'ai2_arc@ARC-Easy', 'wnli', 'cb', 'ddo', 'creak', 'wiki_dialog_input_inversion', 'strategyqa', 'cad', 'social_i_qa', 'qrecc_input_inversion', 'wiqa']:
            instruction, query = src, ''
        
        if ' answer is' in regex.split(r'[\n]+', ans)[-1].lower() and tgt.lower() in regex.split(r'[\n]+', ans)[-1].lower():
            response = ans
        else:
            response = f'{ans}\nThe answer is: {tgt}'
        data.append({
            'instruction': instruction,
            'query': query if query else None,
            'answer': tgt,
            'response': response,
            'task': task
        })
    import ipdb; ipdb.set_trace()
    return data


class CoTDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT, trust_remote_code=True)
        indexes = json_load('/local/home/yuxi_xie/Projects/OA-DAG/datasets/cot_len_indexes.json')
        self.data = [x for i, x in enumerate(self.data) if indexes[f'{i}'] < 512]
        self.data = process_cot(self.data)
        # self.data = sample_from_dataset(self.data, int(len(self.data) * .75))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        instruction = data['instruction'].strip()
        _input = data['query']
        input = f'{instruction}\n\n{_input}' if _input else instruction
        answer = data['response'].strip()
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)

class CoTTrainDataset(CoTDataset):
    NAME: str = 'CoTCollection'
    PATH: str = 'kaist-ai/CoT-Collection'
    SPLIT: str = 'train'
