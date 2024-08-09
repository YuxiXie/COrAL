# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Miscellaneous utilities."""

from __future__ import annotations

import dataclasses
import os
import random
import json
import codecs
import threading
from collections import OrderedDict
from typing import Any, Callable, Generator, TypeVar, cast
from typing_extensions import TypeAlias  # Python 3.10+
import scipy.stats as stats

import re
import math
import numpy as np
import optree
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from optree.typing import PyTreeTypeVar
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import ModelOutput
from transformers.tokenization_utils import BatchEncoding, PaddingStrategy, TruncationStrategy

from oa_dag.configs.constants import PROMPT_ASSISTANT, IGNORE_INDEX


__all__ = [
    'seed_everything',
    'str2bool',
    'to_device',
    'batch_retokenize',
    'is_same_tokenizer',
    'is_main_process',
    'masked_mean',
    'gather_log_probabilities',
    'get_all_reduce_mean',
    'get_all_reduce_sum',
    'get_optimizer_grouped_parameters',
]


TensorTree: TypeAlias = PyTreeTypeVar('TensorTree', torch.Tensor)
Func = TypeVar('Func', bound=Callable[..., Any])

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def seed_everything(seed: int) -> None:
    """Set global random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(string: str) -> bool:
    """Convert a string literal to a boolean value."""
    if string.lower() in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if string.lower() in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    return bool(string)


def get_subclasses(cls: type, memo: set[type] | None = None) -> Generator[type, None, None]:
    """Get all subclasses of a class recursively."""
    if memo is None:
        memo = set()

    for subclass in cls.__subclasses__():
        if subclass in memo:
            continue

        memo.add(subclass)
        yield subclass
        yield from get_subclasses(subclass, memo=memo)


__PYTREE_INITIALIZED = False
__PYTREE_REGISTRY_LOCK = threading.Lock()


def __initialize_pytree_registry_once() -> None:
    # pylint: disable-next=import-outside-toplevel,unused-import
    from oa_dag.models.score_model import ScoreModelOutput  # noqa: F401

    global __PYTREE_INITIALIZED  # pylint: disable=global-statement
    if __PYTREE_INITIALIZED:
        return

    with __PYTREE_REGISTRY_LOCK:
        if __PYTREE_INITIALIZED:
            return

        optree.register_pytree_node(
            BatchEncoding,
            lambda batch_encoding: (
                [batch_encoding.data],
                {'encoding': batch_encoding.encodings, 'n_sequences': batch_encoding.n_sequences},
            ),
            lambda metadata, children: BatchEncoding(children[0], **metadata),
            namespace='oa_dag',
        )
        optree.register_pytree_node(
            ModelOutput,
            lambda model_output: (model_output.values(), model_output.keys(), model_output.keys()),
            lambda keys, values: ModelOutput(OrderedDict(zip(keys, values))),
            namespace='oa_dag',
        )

        for model_output_class in filter(dataclasses.is_dataclass, get_subclasses(ModelOutput)):
            optree.register_pytree_node(
                model_output_class,
                lambda model_output: ([dataclasses.asdict(model_output)], type(model_output)),
                lambda metadata, children: metadata(**children[0]),
                namespace='oa_dag',
            )

        __PYTREE_INITIALIZED = True


def to_device(batch: TensorTree, device: torch.device | str | int | None) -> TensorTree:
    """Move a batch of tensors to a device."""
    if not __PYTREE_INITIALIZED:
        __initialize_pytree_registry_once()
    if device is None:
        return batch
    return optree.tree_map(lambda x: x.to(device), batch, namespace='oa_dag')


def batch_retokenize(
    input_ids: torch.LongTensor,
    src_tokenizer: PreTrainedTokenizerBase,
    dest_tokenizer: PreTrainedTokenizerBase,
    *,
    padding: bool | str | PaddingStrategy = PaddingStrategy.LONGEST,
    truncation: bool | str | TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    skip_special_tokens: bool = True,
    device: torch.device | str | int | None = None,
) -> BatchEncoding:
    """Re-tokenize a batch of input ids from one tokenizer to another."""
    output = dest_tokenizer(
        [
            text + dest_tokenizer.eos_token
            for text in src_tokenizer.batch_decode(
                input_ids,
                skip_special_tokens=skip_special_tokens,
            )
        ],
        padding=padding,
        truncation=truncation,
        return_tensors='pt',
    )
    if device is not None:
        output = to_device(output, device)
    return output


def is_same_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    other_tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """Check if two tokenizers are the same."""
    return tokenizer is other_tokenizer or (
        tokenizer.__class__ == other_tokenizer.__class__
        and tokenizer.get_vocab() == other_tokenizer.get_vocab()
    )


def is_main_process() -> bool:
    """Check if the current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def rank_zero_only(func: Func) -> Func:
    """Decorator to make a function only run on the main process."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function for the decorator."""
        if is_main_process():
            return func(*args, **kwargs)
        return None

    return cast(Func, wrapper)


def masked_mean(
    x: torch.Tensor,  # size = (B, L)
    mask: torch.BoolTensor | None = None,  # size = (B, L)
) -> torch.Tensor:  # size = ()
    """Compute the mean of a tensor with a mask."""
    if mask is None:
        return x.mean()
    return ((x * mask).sum(dim=-1) / mask.sum(dim=-1)).mean()


def gather_log_probabilities(
    logits: torch.Tensor,  # size = (B, L, V)
    labels: torch.LongTensor,  # size = (B, L)
) -> torch.Tensor:  # size = (B, L)
    """Gather log probabilities of the given labels from the logits."""
    log_probs = F.log_softmax(logits, dim=-1)  # size = (B, L, V)
    gathered_log_probs = torch.gather(  # size = (B, L, 1)
        log_probs,
        dim=-1,
        index=labels.unsqueeze(dim=-1),
    )
    return gathered_log_probs.squeeze(dim=-1)  # size = (B, L)


def broadcast_requires_grad(model):
    for param in model.parameters():
        # Temporarily convert boolean to a tensor to broadcast
        grad_tensor = torch.tensor([param.requires_grad], dtype=torch.bool, device=param.device)
        # Broadcast this tensor to all GPUs from the master GPU (rank 0)
        dist.broadcast(grad_tensor, src=0)
        # Set the parameter's requires_grad to the broadcasted value
        param.requires_grad = grad_tensor.item()


def get_all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Perform all-reduce operation on a tensor cross all ranks and return the mean."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def get_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Perform all-reduce operation on a tensor cross all ranks and return the sum."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def get_all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    """Perform all-reduce operation on a tensor cross all ranks and return the max."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def get_all_reduce_min(tensor: torch.Tensor) -> torch.Tensor:
    """Perform all-reduce operation on a tensor cross all ranks and return the max."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return tensor


def get_optimizer_grouped_parameters(
    module: nn.Module,
    weight_decay: float,
    no_decay_name_set: set[str] | None = None,
) -> list[dict[str, list[nn.Parameter] | float]]:
    """Get parameter groups with customized weight decay value."""
    if no_decay_name_set is None:
        no_decay_name_set = {'bias', 'LayerNorm.weight'}
    no_decay_name_set = set(map(str.lower, no_decay_name_set))

    named_parameters = [
        (name.lower(), param) for name, param in module.named_parameters() if param.requires_grad
    ]

    return [
        {
            'params': [
                param
                for name, param in named_parameters
                if not any(no_decay_name in name for no_decay_name in no_decay_name_set)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                param
                for name, param in named_parameters
                if any(no_decay_name in name for no_decay_name in no_decay_name_set)
            ],
            'weight_decay': 0.0,
        },
    ]


def split_prompt_response(
    texts: list[str],
    split_token: str = PROMPT_ASSISTANT,
) -> tuple[list[str], list[str]]:
    """Split prompt-response pairs into prompts and responses."""

    def split_fn(text: str) -> tuple[str, str]:
        """Split a prompt-response pair into prompt and response."""
        prompt, partition, response = text.rpartition(split_token)
        assert prompt and partition and response, f'invalid text: {text}'
        return prompt + partition, response

    return tuple(map(list, zip(*map(split_fn, texts))))


def get_variable_generator(mu_value=0.5, stderr=0.25, min_value=0.0, max_value=1.0):
    generator = stats.truncnorm(
        (min_value - mu_value) / stderr, (max_value - mu_value) / stderr,
        loc=mu_value, scale=stderr,
    )
    return generator


def pad_tensors(tensors, max_len=-1, pad_value=IGNORE_INDEX):
    tensors = [x for x in tensors]
    if max_len <= 0:
        max_len = max([len(x) for x in tensors])
    for i in range(len(tensors)):
        pad_len = max_len - len(tensors[i])
        tmp = torch.ones((pad_len,), dtype=torch.long, device=tensors[i].device)
        tensors[i] = torch.cat((tensors[i], tmp * pad_value), dim=-1).long()
    return torch.stack(tensors, dim=0)


def shuffle_and_mask(label_position_ids: torch.LongTensor, ratio_generator, left2right=False, fixed_mask_threshold=-1, device=None):
    # keep_sample = True
    # while keep_sample:
    #     if fixed_mask_threshold >= 0:
    #         mask_threshold = fixed_mask_threshold
    #     else:
    #         # sample to get masking threshold
    #         mask_threshold = ratio_generator.rvs(1)[0]
        
    #     if left2right:  # mask the right part
    #         random_noise = torch.arange(0, label_position_ids.size(-1), dtype=torch.float, device=device)
    #         random_noise = (-random_noise + label_position_ids.size(-1) - 0.5) / label_position_ids.size(-1)    # reverse to be descending
    #     else:   # randomly mask
    #         random_noise = torch.rand(label_position_ids.size(-1), device=device)
        
    #     # extract the position ids of the tokens to mask
    #     mask_label_position_ids = label_position_ids[random_noise.lt(mask_threshold).nonzero().squeeze(-1)]
    #     if mask_label_position_ids.size(0) > 0 or label_position_ids.size(0) <= 0:
    #         keep_sample = False
    
    # return mask_label_position_ids, mask_threshold
    
    if fixed_mask_threshold >= 0:
        mask_threshold = fixed_mask_threshold
    else:
        # sample to get masking threshold
        mask_threshold = ratio_generator.rvs(1)[0]
    
    if left2right:  # mask the right part
        random_noise = torch.arange(0, label_position_ids.size(-1), dtype=torch.float, device=device)
        random_noise = (-random_noise + label_position_ids.size(-1) - 0.5) / label_position_ids.size(-1)    # reverse to be descending
    else:   # randomly mask
        random_noise = torch.rand(label_position_ids.size(-1), device=device)
    
    # extract the position ids of the tokens to mask
    mask_label_position_ids = label_position_ids[random_noise.lt(mask_threshold).nonzero().squeeze(-1)]
    
    return mask_label_position_ids, mask_threshold


def add_noise(cur_input_ids: torch.LongTensor, cur_labels: torch.LongTensor, ratio_generator, fixed_replace_threshold=-1, device=None):
    if fixed_replace_threshold >= 0:
        replace_threshold = fixed_replace_threshold
    else:
        # sample the threshold for reconstruction
        replace_threshold = ratio_generator.rvs(1)[0]
    random_noise = torch.rand(cur_input_ids.size(-1), device=device)
    replace_ids = torch.logical_and(random_noise.lt(replace_threshold), cur_labels.ne(IGNORE_INDEX)).nonzero().squeeze(-1)
    return replace_ids, replace_threshold


def decode_masked_text(input_ids: torch.LongTensor, position_ids: torch.LongTensor, replace_indexes: torch.LongTensor, tokenizer, topk_ids: torch.LongTensor = None):
    if topk_ids is not None:
        _replace_indexes = replace_indexes[replace_indexes.ge(0).nonzero().squeeze(-1)]
        input_ids[_replace_indexes] = topk_ids[:_replace_indexes.size(-1), 0]
    special_id = tokenizer.encode('_')[-1]
    ids = [input_ids[position_ids.tolist().index(x)] if x in position_ids else special_id for x in range(position_ids.min().item(), position_ids.max().item() + 1)]
    text = tokenizer.decode(ids)
    return text


def corrupt_input(replace_ids: torch.LongTensor, input_ids: torch.LongTensor, position_ids: torch.LongTensor, labels: torch.LongTensor, 
                  tokenizer, device=None):
    # random tokens
    random_ids = torch.randint(tokenizer.vocab_size, replace_ids.size(), device=device)    
    # shifted results
    left_shifted_position_ids, right_shifted_position_ids = position_ids[replace_ids] - 1, position_ids[replace_ids] + 1
    if right_shifted_position_ids.max() >= input_ids.size(-1):
        right_shifted_position_ids[right_shifted_position_ids.eq(right_shifted_position_ids.max()).nonzero().squeeze(-1)] = 0
    left_shifted_ids, right_shifted_ids = input_ids[left_shifted_position_ids], input_ids[right_shifted_position_ids]
    
    for idx in range(replace_ids.size(-1)):
        _id = replace_ids[idx]
        var = random.random()
        if var < .05:
            # random tokens
            input_ids[_id] = random_ids[idx]
        elif var < .15:
            # random tokens in the context
            input_ids[_id] = input_ids[random.randint(0, input_ids.size(-1) - 1)]
        else:
            # neighboring tokens
            var = random.random()
            if var < .5:
                input_ids[_id] = left_shifted_ids[idx]
            else:
                input_ids[_id] = right_shifted_ids[idx]

    return input_ids


operators = '+-*x/÷%=()}{[]$¥<>'

def locate_quantity(topk_ids: torch.LongTensor, tokenizer):
    sequences = tokenizer.batch_decode(topk_ids[:, :1])
    quantity_seq_ids = [(True if re.search(r'\d', seq) or any(x in seq for x in operators) else False) for seq in sequences]
    return torch.tensor(quantity_seq_ids, dtype=torch.bool, device=topk_ids.device)


def replace_with_zero_one(topk_probs: torch.FloatTensor):
    batch_size, k = topk_probs.size(0), topk_probs.size(-1)
    topk_probs = topk_probs.view(-1, k).contiguous()
    random_noise = torch.rand(topk_probs.size(0), device=topk_probs.device)
    to_replace_ids = random_noise.le(.5).nonzero().squeeze(-1)
    if to_replace_ids.size(-1) > 0:
        selected_indexes = torch.randint(k, (to_replace_ids.size(-1),)).to(topk_probs.device)
        new_probs = torch.zeros_like(topk_probs[to_replace_ids], dtype=torch.float)
        for i in range(len(selected_indexes)):
            new_probs[i][selected_indexes[i].item()] = 1
        topk_probs[to_replace_ids] = new_probs
    return topk_probs.view(batch_size, -1, k).contiguous()


def get_normal_dist(mean=0.0, std=1.0, window_size=4, r=1):
    import torch.distributions as distr
    
    mean = torch.tensor([mean])
    std = torch.tensor([std])
    normal = distr.Normal(mean, std)

    # Calculate probability density function (PDF)
    x = torch.linspace(-r, r, window_size * 2 + 1)
    return torch.exp(normal.log_prob(x))
    
    # x = torch.linspace(-2, 2, window_size * 2 + 3)
    # return torch.exp(normal.log_prob(x))[2:]