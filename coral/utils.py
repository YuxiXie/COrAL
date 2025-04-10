"""Miscellaneous utilities."""

from __future__ import annotations

import dataclasses
import os
import random
import json
import codecs
import jsonlines
import threading
from collections import OrderedDict
from typing import Any, Callable, Generator, TypeVar, cast
from typing_extensions import TypeAlias  # Python 3.10+
import scipy.stats as stats

import re
import math
import time
import numpy as np
import optree
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from optree.typing import PyTreeTypeVar
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from transformers.generation.utils import LogitsProcessorList
from transformers.tokenization_utils import BatchEncoding, PaddingStrategy, TruncationStrategy

from coral.configs.constants import PROMPT_ASSISTANT, IGNORE_INDEX, metamath_accu


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


def jsonlines_load(x):
    with jsonlines.open(x, mode='r') as reader:
        data = [r for r in reader]
    return data

def jsonlines_dump(x, d, mode='w'):
    if not os.path.exists(x): mode='w'
    with jsonlines.open(x, mode=mode) as writer:
        writer.write_all(d)


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
            namespace='coral',
        )
        optree.register_pytree_node(
            ModelOutput,
            lambda model_output: (model_output.values(), model_output.keys(), model_output.keys()),
            lambda keys, values: ModelOutput(OrderedDict(zip(keys, values))),
            namespace='coral',
        )

        for model_output_class in filter(dataclasses.is_dataclass, get_subclasses(ModelOutput)):
            optree.register_pytree_node(
                model_output_class,
                lambda model_output: ([dataclasses.asdict(model_output)], type(model_output)),
                lambda metadata, children: metadata(**children[0]),
                namespace='coral',
            )

        __PYTREE_INITIALIZED = True


def to_device(batch: TensorTree, device: torch.device | str | int | None) -> TensorTree:
    """Move a batch of tensors to a device."""
    if not __PYTREE_INITIALIZED:
        __initialize_pytree_registry_once()
    if device is None:
        return batch
    return optree.tree_map(lambda x: x.to(device), batch, namespace='coral')


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
        tmp = torch.ones((pad_len,) + tensors[i].shape[1:], dtype=torch.long, device=tensors[i].device)
        tensors[i] = torch.cat((tensors[i], tmp * pad_value), dim=0).long()
    return torch.stack(tensors, dim=0)


def shuffle_and_mask(label_position_ids: torch.LongTensor, ratio_generator, left2right=False, fixed_mask_threshold=-1, device=None):
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


def add_noise(cur_input_ids: torch.LongTensor, cur_labels: torch.LongTensor, ratio_generator, force_replace=False, fixed_replace_threshold=-1, device=None):
    keep_generate = True
    while keep_generate:
        if fixed_replace_threshold >= 0:
            replace_threshold = fixed_replace_threshold
        else:
            # sample the threshold for reconstruction
            replace_threshold = ratio_generator.rvs(1)[0]
        random_noise = torch.rand(cur_input_ids.size(-1), device=device)
        replace_ids = torch.logical_and(random_noise.lt(replace_threshold), cur_labels.ne(IGNORE_INDEX)).nonzero().squeeze(-1)
        if not force_replace or replace_ids.size(-1) > 0:
            keep_generate = False
    return replace_ids, replace_threshold


def corrupt_context(cur_input_ids: torch.LongTensor, cur_labels: torch.LongTensor, 
                    raw_input_ids: torch.LongTensor, raw_labels: torch.LongTensor, raw_label_positions: torch.LongTensor,
                    context_size: int, num_contexts: int, context_inject_ratio_generator, 
                    sample_from_future: bool = False, sample_from_near: bool = False,
                    fixed_replace_threshold: float = -1, tokenizer = None):
    label_start_idx = raw_label_positions[0].item()
    keep_inject = True
    sample_from_future = False if sample_from_near else sample_from_future
    while keep_inject:
        cur_inject_cnt, prev_inject = 0, False
        context_inject_ratio = fixed_replace_threshold if fixed_replace_threshold >= 0 else context_inject_ratio_generator.rvs(1)[0]
        for j in range(num_contexts):
            gt_context = raw_input_ids.clone()[raw_label_positions[j * context_size: (j + 1) * context_size]]
            var1, var2 = torch.rand(dist.get_world_size())[dist.get_rank()], torch.rand(dist.get_world_size())[dist.get_rank()]
            # if var1 < context_inject_ratio and not prev_inject:
            if var1 < context_inject_ratio:
                if sample_from_near:
                    # fake_context_idx = random.randint(max(0, j - 2), min(num_contexts - 1, j + 2))
                    fake_context_idx = random.randint(max(0, (j - 1) * context_size), min(raw_label_positions.size(-1) - context_size, (j + 1) * context_size))
                elif sample_from_future:
                    fake_context_idx = random.randint(j, num_contexts - 1)
                else:
                    fake_context_idx = random.randint(0, num_contexts - 1)
                # fake_context = raw_input_ids.clone()[raw_label_positions[fake_context_idx * context_size: (fake_context_idx + 1) * context_size]]
                fake_context = raw_input_ids.clone()[raw_label_positions[fake_context_idx: fake_context_idx + context_size]]
                min_len = min(gt_context.size(-1), fake_context.size(-1))
                tmp_context = gt_context.clone()
                tmp_context[:min_len] = fake_context[:min_len]
                fake_context = tmp_context
                
                var11 = torch.rand(dist.get_world_size())[dist.get_rank()]
                replace_fake = torch.rand(fake_context.size(-1), device=fake_context.device)
                fake_context = fake_context * replace_fake.gt(var11 / 2) + gt_context * replace_fake.le(var11 / 2)
                
                cur_input_ids = torch.cat((cur_input_ids, fake_context), dim=-1)
                cur_labels = torch.cat((cur_labels, gt_context), dim=-1)
                cur_inject_cnt += 1
                prev_inject = True
            elif var2 < context_inject_ratio and not prev_inject:
                var3 = torch.rand(dist.get_world_size())[dist.get_rank()]
                if var3 < .5:
                    max_len = gt_context.size(-1)
                    repeat_num = random.randint(1, (max_len + 1) // 2)
                    cur_input_ids = torch.cat((cur_input_ids, gt_context[0].unsqueeze(-1).expand(repeat_num)), dim=-1)
                    cur_input_ids = torch.cat((cur_input_ids, gt_context[1: 1 + max_len - repeat_num]), dim=-1)
                    prev_inject = True
                else:
                    cur_input_ids = torch.cat((cur_input_ids, gt_context), dim=-1)
                    prev_inject = False
                cur_labels = torch.cat((cur_labels, gt_context), dim=-1)
                cur_inject_cnt += 1
            else:
                cur_input_ids = torch.cat((cur_input_ids, gt_context), dim=-1)
                cur_labels = torch.cat((cur_labels, torch.ones_like(gt_context, dtype=torch.long) * IGNORE_INDEX), dim=-1)
                prev_inject = False
        if cur_inject_cnt > 0:
            keep_inject = False
        else:
            cur_input_ids = raw_input_ids.clone()[:label_start_idx]
            cur_labels = raw_labels.clone()[:label_start_idx]
            prev_inject = False
    return cur_input_ids, cur_labels, cur_inject_cnt, context_inject_ratio


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
        if var < .1:
            # random tokens
            input_ids[_id] = random_ids[idx]
        elif var < .6:
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
    # to_replace_ids = random_noise.le(.5).nonzero().squeeze(-1)
    to_replace_ids = random_noise.le(1.0).nonzero().squeeze(-1)
    if to_replace_ids.size(-1) > 0:
        selected_indexes = torch.randint(k, (to_replace_ids.size(-1),)).to(topk_probs.device)
        new_probs = torch.zeros_like(topk_probs[to_replace_ids], dtype=torch.float)
        for i in range(len(selected_indexes)):
            new_probs[i][selected_indexes[i].item()] = 1
        topk_probs[to_replace_ids] = new_probs
    return topk_probs.view(batch_size, -1, k).contiguous()

def get_normal_dist(mean=0.0, std=1.0, forward_size=4, backward_size=4, r=3):
    import torch.distributions as distr
    
    mean = torch.tensor([mean])
    std = torch.tensor([std])
    normal = distr.Normal(mean, std)

    # Calculate probability density function (PDF)
    window_size = forward_size + backward_size
    
    x = torch.linspace(0, 2 * r, window_size + 1)
    x = torch.exp(normal.log_prob(x))
    
    return x


def sample_from_dataset(dataset, maxlen):
    maxlen = min(maxlen, len(dataset))
    return random.sample(list(dataset), maxlen)


def get_hyperbolic_dist(forward_size=4, backward_size=8, downweight=2e-5):
    def hyperbolic_distance(x, scale=1):
        return torch.arccosh(1 + ((x - 1) * scale) ** 2 / 2)
        
    x_forward = torch.arange(0, forward_size) + 1
    x_forward = hyperbolic_distance(x_forward, scale=1e16)
    
    x_backward = -torch.arange(0, backward_size + 1) + backward_size
    x_backward[-1] = 1
    x_backward = hyperbolic_distance(x_backward, scale=1e8) #* x_forward[1]

    x_max = x_forward.max().item() + 1
    x_forward = (-x_forward + x_max) / x_max
    x_backward = (-x_backward + x_max) / x_max

    x_forward[1:] = x_forward[1:] * downweight
    x_backward[-1] = x_backward[:-1].min() * 1e-1
    x = torch.cat((x_backward, x_forward), dim=-1)
    
    return x


def get_exp_dist(forward_size=4, backward_size=8, forward_scale=8, backward_scale=1/8, reverse_forward=False, device='cpu'):
    x_forward = torch.arange(0, forward_size, device=device)
    x_forward = (-forward_scale * x_forward).exp()
    if reverse_forward and len(x_forward) > 1:
        x_forward = torch.cat((x_forward[:1], 1/x_forward[1:]), dim=-1)
    
    if backward_size < 0:
        return x_forward
    if backward_size < 1:
        return torch.cat((x_forward[:1]/10, x_forward), dim=-1)
    
    x_backward = -torch.arange(0, backward_size - 1, device=device) + backward_size - 2
    x_backward = torch.cat((x_backward, torch.arange(1, 3, device=device)), dim=-1)
    x_backward = (-backward_scale * x_backward).exp()[-backward_size - 1:]
    
    return torch.cat((x_backward, x_forward), dim=-1)


def kl_divergence_log(log_p, log_q):
    if log_q.eq(0).all():
        return torch.zeros_like(log_p[0]) + torch.inf
    if (log_p - log_q).abs().max() < 1e-3:
        return torch.zeros_like(log_p[0])
    # This function calculates KL divergence in log space
    p = torch.exp(log_p)
    rst = torch.sum(p * (log_p - log_q), dim=-1)
    if rst < 0 and rst.abs() < 1e-3:
        return torch.zeros_like(log_p[0]) + rst.abs()
    return rst

def js_divergence_log(log_p, log_q):
    # Calculate the mean distribution in log space
    log_m = torch.log((torch.exp(log_p) + torch.exp(log_q)) / 2)
    return (kl_divergence_log(log_p, log_m) + kl_divergence_log(log_q, log_m)) / 2

def calculate_jsd_variance_log(distributions_log, mean_distribution_log, weights):
    if not len(distributions_log):
        return torch.zeros_like(mean_distribution_log[0]) + torch.inf
    
    # Calculate the JSD for each distribution against the mean distribution
    jsd_values = [js_divergence_log(log_dist, mean_distribution_log) * weight for log_dist, weight in zip(distributions_log, weights)]
    
    # Calculate the "variance" as the mean of these JSD values
    jsd_variance = torch.sum(torch.stack(jsd_values)) / weights.sum()
    
    return jsd_variance


def calculate_kl_variance_log(distributions_log, mean_distribution_log, weights):
    if not len(distributions_log):
        return torch.zeros_like(mean_distribution_log[0]) + torch.inf
    
    # Calculate the KLD for each distribution against the mean distribution
    kl_values = [kl_divergence_log(log_dist, mean_distribution_log) * weight for log_dist, weight in zip(distributions_log, weights)]
    
    # Calculate the "variance" as the mean of these JSD values
    jsd_variance = torch.sum(torch.stack(kl_values)) / weights.sum()
    
    return jsd_variance


def gather_kl_variance_dict(logprobs_dict: torch.FloatTensor, mean_logprobs: torch.FloatTensor, weights: torch.FloatTensor):
    batch_size, n_tokens = mean_logprobs.size(0), mean_logprobs.size(1)
    variance_logprobs = torch.zeros(batch_size, n_tokens, dtype=mean_logprobs.dtype, device=mean_logprobs.device)
    n_logprobs = torch.zeros(batch_size, n_tokens, dtype=torch.int8, device=mean_logprobs.device)
    for i in range(batch_size):
        for j in range(n_tokens):
            logprobs_list = [x for k, x in enumerate(logprobs_dict[i, j]) if x.ne(0).any()]
            variance_logprobs[i, j] = calculate_kl_variance_log(
                logprobs_list, mean_logprobs[i, j],
                weights[logprobs_dict[i, j].ne(0).any(-1).nonzero().squeeze(-1)][:len(logprobs_list)]
            )
            n_logprobs[i, j] = len(logprobs_list)
    return variance_logprobs, n_logprobs


def update_variance(logprobs: torch.FloatTensor, available_indexes: torch.LongTensor, window_size: int = 13):
    n_logprobs = logprobs.size(0)
    variance = -torch.ones(window_size, window_size, dtype=logprobs.dtype, device=logprobs.device)
    for i in range(n_logprobs):
        for j in range(n_logprobs):
            if i == j:
                variance[available_indexes[i], available_indexes[j]] = 0
                continue
            variance[available_indexes[i], available_indexes[j]] = kl_divergence_log(logprobs[j], logprobs[i])
    return variance


def extract_logprobs_into_dict(
    logprobs: torch.FloatTensor, 
    ref_position_ids_to_predict: torch.LongTensor,
    pred_start_pos: int, 
    pred_end_pos: int,
    forward_size: int = 4,
    backward_size: int = 8,
    accept_conf: torch.FloatTensor = None,
):
    batch_size, seq_length, ndim = logprobs.size(0), logprobs.size(1), logprobs.size(-1)
    window_size = forward_size + backward_size + 1  # N
    target_seq_len = pred_end_pos - pred_start_pos + 1
    
    # accpetance confidence
    if accept_conf is None:
        accept_conf = torch.ones((seq_length,), dtype=logprobs.dtype, device=logprobs.device)
    elif accept_conf.size(-1) < seq_length:
        accept_conf = torch.cat((torch.ones((seq_length - accept_conf.size(-1),), dtype=accept_conf.dtype, device=logprobs.device), accept_conf), dim=-1)
    elif accept_conf.size(-1) > seq_length:
        accept_conf = accept_conf[-seq_length:]
    accept_conf = accept_conf.cummin(-1).values + 1
    
    # pad logprobs to shift and fit the target indices
    padded_logprobs = torch.zeros(
        (batch_size, seq_length + window_size - 1, window_size, ndim,), 
        dtype=logprobs.dtype, 
        device=logprobs.device,
    ) # (1, L, N, V)
    shifted_indices = torch.arange(seq_length).unsqueeze(1) + torch.arange(window_size).unsqueeze(0)    # (L', N)
    padded_logprobs[0, shifted_indices, torch.arange(window_size)] = logprobs[0]   # (L', N, V)
    idx1 = ref_position_ids_to_predict[0].sum(-1).nonzero()[0][0]   # available start index for the seq_len dimension
    idx2 = ref_position_ids_to_predict[0, idx1].nonzero()[0][0]     # available start index for the target_pos dimension
    padded_logprobs = padded_logprobs.contiguous()[:, idx2: idx2 + target_seq_len]  # (1, T, N, V)
    
    # create pad weights
    lambda_list = get_exp_dist(forward_size=forward_size, backward_size=backward_size, device=logprobs.device)
    padded_weights = torch.zeros(batch_size, seq_length + window_size - 1, window_size, dtype=lambda_list.dtype, device=lambda_list.device)
    padded_weights[0, shifted_indices, torch.arange(window_size)] = accept_conf.unsqueeze(1).expand(accept_conf.size(-1), window_size)
    padded_weights = padded_weights.contiguous()[:, idx2: idx2 + target_seq_len]
    padded_weights = padded_weights * lambda_list
    
    ensemble_logprobs = (padded_logprobs * padded_weights.unsqueeze(-1)).sum(-2) / padded_weights.sum(-1).unsqueeze(-1)    # (1, T, V)
    return padded_logprobs, ensemble_logprobs

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('/share/edc/home/yuxi_xie/oa_dag/model-checkpoints/sft/metamath-mistral/checkpoint-18516')

def create_tree_attention_mask(
    logprobs: torch.FloatTensor, 
    forward_size: int = 1, 
    topk: int = 16, 
    maximum_seq_len: int = 100, 
    scale_factor: float = 16,
):
    n_depth, ndim = logprobs.size(0), logprobs.size(-1)
    results = torch.topk(logprobs, k=topk, dim=-1)
    topk = results.values.size(-1)
    
    # # sort tokens by accuracies
    # accuracies = results.values.clone().exp()
    # accu_matrix = torch.tensor(metamath_accu).to(accuracies.dtype).to(accuracies.device) / 100
    # accu_idx = 8 if forward_size < 1 else (8 + forward_size)
    # cnt = min(accu_idx + 1, accuracies.size(0))
    # _k = min(accu_matrix.size(-1), accuracies.size(-1))
    # accuracies[-cnt:, :_k] = accu_matrix[accu_idx - cnt + 1: accu_idx + 1, :_k]
    # sorted_conf = accuracies.view(-1).sort(descending=True)
    
    # sort tokens by confidence scores
    # confidence = results.values.clone()
    # for i in range(forward_size):
    #     # scale last few tokens
    #     confidence[-i - 1] = confidence[-i - 1] / (forward_size - i + scale_factor - 1) * scale_factor
    # confidence = confidence.exp()
    sorted_conf = results.values.exp().view(-1).sort(descending=True)
    
    positions = torch.arange(0, n_depth, dtype=torch.long, device=logprobs.device).unsqueeze(-1).expand(n_depth, topk)
    topk_indexes = torch.arange(0, topk, dtype=torch.long, device=logprobs.device).unsqueeze(0).expand(n_depth, topk)
    positions, topk_indexes = positions.contiguous().view(-1), topk_indexes.contiguous().view(-1)
    
    # initialize attention tree
    combinations, nest_combinations = [[0] * n_depth], [[0] * i for i in range(1, n_depth)]
    total_len = len(combinations) + len(nest_combinations)
    # expand attention tree
    for idx in sorted_conf.indices:
        if idx % topk == 0: continue
        pos, k = positions[idx].item(), topk_indexes[idx].item()
        if torch.isinf(results.values[pos, k]): continue
        tmp_combinations = []
        for comb in combinations:
            comb = comb[:]
            comb[pos] = k
            tmp_combinations.append(comb)
            total_len += 1
            nest_combinations.extend([comb[:i] for i in range(pos + 1, n_depth)])
            total_len += max(0, n_depth - pos - 1)
        combinations.extend(tmp_combinations)
        if total_len > maximum_seq_len: break
    combinations.extend(nest_combinations)
    
    # Sort the combinations based on their lengths and then their values
    sorted_combinations = sorted(combinations, key=lambda x: (len(x), x))
    comb_len = len(sorted_combinations) + 1
    # Initialize depth_counts to keep track of how many choices have a particular depth
    seq_ids = []
    depth_counts, prev_depth = [], 0
    for path in sorted_combinations:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
        seq_ids.append(results.indices[len(path) - 1, path[-1]])
    seq_ids = torch.stack(seq_ids, dim=0)
    # Create the attention mask
    tree_attn_mask = torch.eye(comb_len, comb_len)
    tree_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_choice = sorted_combinations[start + j]
            # retrieve ancestor position
            if len(cur_choice) == 1: continue
            ancestor_idx = []
            for c in range(len(cur_choice) - 1):
                ancestor_idx.append(sorted_combinations.index(cur_choice[:c+1]) + 1)
            tree_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]
    
    # Generate position IDs
    position_ids = torch.zeros(comb_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]
    
    # Generate retrieval indices
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_combinations)):
        cur_choice = sorted_combinations[-i-1]
        retrieve_indice = []
        if cur_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_choice)):
                retrieve_indice.append(sorted_combinations.index(cur_choice[:c+1]))
                retrieve_paths.append(cur_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)
    # if seq_ids.size(-1) > 512:
    #     import ipdb; ipdb.set_trace()
    return seq_ids, position_ids.to(seq_ids.device), tree_attn_mask.bool().to(seq_ids.device), retrieve_indices.to(seq_ids.device)


def prepare_candidate_input_output(
    prev_input_ids: torch.LongTensor, 
    candidate_ids: torch.LongTensor, 
    candidate_position_ids: torch.LongTensor,
    tree_attn_mask: torch.BoolTensor,
    pred_start_pos: int,
    pred_end_pos: int,
    forward_size: int = 4,
    backward_size: int = 8,
):
    cur_input_ids = torch.cat((prev_input_ids, candidate_ids), dim=-1)
    prev_position_ids = torch.arange(0, prev_input_ids.size(-1), dtype=torch.long, device=prev_input_ids.device)
    cur_position_ids = torch.cat((prev_position_ids, candidate_position_ids[1:] + prev_position_ids[-1]), dim=-1)
    
    tmp_position_ids_to_predict = torch.arange(forward_size + backward_size + 1, dtype=torch.long, device=cur_input_ids.device)
    position_ids_to_predict = (tmp_position_ids_to_predict - backward_size) + torch.arange(cur_position_ids.max().item() + 1, dtype=torch.long, device=cur_input_ids.device).view(-1, 1)
    position_ids_to_predict = position_ids_to_predict.masked_fill(position_ids_to_predict.lt(pred_start_pos), 0)
    position_ids_to_predict = position_ids_to_predict.masked_fill(position_ids_to_predict.gt(pred_end_pos), 0)
    
    cur_position_ids_to_predict = position_ids_to_predict[cur_position_ids]
    cur_position_ids_to_predict[:pred_start_pos - 1, :] = 0
    cur_attention_mask = torch.tril(torch.ones((cur_input_ids.size(-1), cur_input_ids.size(-1)), dtype=torch.bool, device=cur_input_ids.device))
    cur_attention_mask[-candidate_ids.size(-1) - 1:, -candidate_ids.size(-1) - 1:] = tree_attn_mask
    
    return cur_input_ids, cur_position_ids, cur_attention_mask, cur_position_ids_to_predict


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def prepare_candidates(
    input_ids: torch.LongTensor,
    logits: torch.FloatTensor,
    ref_position_ids_to_predict: torch.LongTensor,
    pred_start_pos: int,
    pred_end_pos: int,
    forward_size: int = 4,
    backward_size: int = 8,
    eval_forward_size: int = 1,
    eval_backward_size: int = 8,
    # processors: LogitsProcessorList = LogitsProcessorList(),
    topk: int = 16,
    max_new_tokens = 128,
    max_length: int = 512,
    accept_conf: torch.FloatTensor = None, 
    skip_verify: bool = False,
    verbal: bool = False,
):
    stime = time.time()
    seq_length, ndim = logits.size(1), logits.size(-1)
    logprobs = logits.view(-1, ndim).log_softmax(dim=-1).view(
        seq_length, -1, ndim
    ).unsqueeze(0)    # (1, L', N, V) --> (L' * N, V) --> (1, L', N, V)
    
    # aggregate all predicted distributions
    _, ensemble_logprobs = extract_logprobs_into_dict(
        logprobs=logprobs,
        ref_position_ids_to_predict=ref_position_ids_to_predict,
        pred_start_pos=pred_start_pos,
        pred_end_pos=pred_end_pos,
        forward_size=forward_size,
        backward_size=backward_size,
        accept_conf=accept_conf,
    )   # (1, T, V)
    
    # sample and get candidate tokens
    # token_scores = processors(input_ids, ensemble_logprobs.view(-1, ndim))  # (1 * T, V)
    # logprobs = nn.functional.log_softmax(token_scores, dim=-1)  # (1 * T, V)
    if verbal:
        print('[P1-1]', time.time() - stime)
    
    if skip_verify:
        return ensemble_logprobs
    
    stime = time.time()
    # tree attention construction
    candidate_ids, candidate_position_ids, tree_attn_mask, retrieve_indices = \
        create_tree_attention_mask(
            nn.functional.log_softmax(ensemble_logprobs.view(-1, ndim), dim=-1),
            topk=topk, 
            forward_size=min(forward_size, max(0, pred_end_pos + 1 - input_ids.size(-1))),
            maximum_seq_len=max_new_tokens,
        )
    retrieve_indices = retrieve_indices + input_ids[0, :pred_start_pos].size(-1) - 1
    if verbal:
        print('[P1-2]', time.time() - stime)
    
    stime = time.time()
    # tree attention input prepare
    cur_input_ids, cur_position_ids, cur_attention_mask, cur_position_ids_to_predict = \
        prepare_candidate_input_output(
            prev_input_ids=input_ids[0, :pred_start_pos],
            candidate_ids=candidate_ids,
            candidate_position_ids=candidate_position_ids,
            tree_attn_mask=tree_attn_mask,
            pred_start_pos=pred_start_pos,
            pred_end_pos=pred_end_pos,
            forward_size=eval_forward_size,
            backward_size=eval_backward_size,
        )
    if verbal:
        print('[P1-3]', time.time() - stime)
    
    return cur_input_ids, cur_position_ids, cur_attention_mask, cur_position_ids_to_predict, retrieve_indices, ensemble_logprobs


def extract_accept_flags(logits: torch.FloatTensor, losses: torch.FloatTensor, epsilon: float=0.2, top_p: float=0.6):
    scores = logits.contiguous().view(-1, logits.size(-1))
    
    try:
        epsilon = torch.tensor(epsilon)
        
        # Calculate the adaptive cutoff
        probabilities = scores.softmax(dim=-1)
        # entropy = torch.distributions.Categorical(logits=scores).entropy()
        entropy = -torch.sum(
            probabilities * torch.log(probabilities + torch.finfo(scores.dtype).smallest_normal), dim=-1
        )
        eta = torch.min(epsilon, torch.sqrt(epsilon) * torch.exp(-entropy))[..., None]
        indices_to_remove = probabilities < eta
        
        probabilities = probabilities.masked_fill(indices_to_remove, 2)
        maxlosses = -probabilities.min(dim=-1).values.log()
    except Exception as e:
        print(str(e))
        # torch.cuda.empty_cache()
        
        sorted_logits = torch.sort(scores, descending=False).values
        probabilities = sorted_logits.softmax(dim=-1)
        cumulative_probs = probabilities.cumsum(dim=-1)
        
        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        probabilities = probabilities.masked_fill(sorted_indices_to_remove, 2)
        maxlosses = -probabilities.min(dim=-1).values.log()
    
    return losses.le(maxlosses.view(losses.size(0), losses.size(1), -1))


def calculate_candidate_losses(
    cur_input_ids: torch.LongTensor,
    cur_position_ids_to_predict: torch.LongTensor,
    candidate_logits: torch.FloatTensor,
    retrieve_indices: torch.LongTensor,
    pred_start_pos: int,
    pred_end_pos: int,
    forward_size: int = 4,
    backward_size: int = 8,
    epsilon: float = 0.1,
):
    batch_size, target_seq_len = retrieve_indices.size(0), retrieve_indices.size(-1) - 1
    window_size = forward_size + backward_size + 1  
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    # extract losses for all candidates
    shift_logits = candidate_logits[retrieve_indices].view(-1, candidate_logits.size(-1))   # [B * L' * W, V]
    cur_labels = cur_input_ids.clone()  # flat ids
    cur_labels[:pred_start_pos] = IGNORE_INDEX
    positions_i = cur_position_ids_to_predict[retrieve_indices[0]] - pred_start_pos + 1
    positions_i = positions_i.masked_fill(positions_i.lt(1), 0)
    positions_i = positions_i.masked_fill(positions_i.ge(cur_labels[retrieve_indices].size(1)), 0)
    
    shift_labels = cur_labels[retrieve_indices][:, positions_i].view(-1)    # (B, L', W) --> (B * L' * W)
    shift_labels = shift_labels.masked_fill(positions_i.eq(0).unsqueeze(0).expand(batch_size, -1, -1).contiguous().view(-1), IGNORE_INDEX)
    shift_losses = loss_fct(shift_logits, shift_labels).view(batch_size, target_seq_len + 1, -1)   # (B, L', W)
    candidates = cur_input_ids[retrieve_indices][:, 1:]
    
    # calculate accept ratios
    shift_logits = shift_logits.contiguous().view(batch_size, -1, shift_logits.size(-1))
    try:
        accept_flags = extract_accept_flags(shift_logits, shift_losses, epsilon=epsilon)   # (B, L', W)
    except:
        # torch.cuda.empty_cache()
        n_batch = (batch_size + 31) // 32
        accept_flags = []
        for bidx in range(n_batch):
            accept_flags.append(extract_accept_flags(
                shift_logits[bidx * 32: (bidx + 1) * 32],
                shift_losses[bidx * 32: (bidx + 1) * 32],
                epsilon=epsilon,
            ))
        accept_flags = torch.cat(accept_flags, dim=0)
    
    # extract losses and accept flags
    padded_losses = torch.zeros(
        batch_size, target_seq_len + window_size, window_size, dtype=shift_losses.dtype, device=shift_losses.device,
    ) # (1, N, T, V)
    padded_flags = torch.zeros(
        batch_size, target_seq_len + window_size, window_size, dtype=accept_flags.dtype, device=accept_flags.device,
    ) # (1, N, T, V)
    shifted_indices = torch.arange(target_seq_len + 1).unsqueeze(1) + torch.arange(window_size).unsqueeze(0)    # (L', W)
    padded_losses[:, shifted_indices, torch.arange(window_size)] = shift_losses
        
    padded_flags[:, shifted_indices, torch.arange(window_size)] = accept_flags
    idx1 = cur_position_ids_to_predict.sum(-1).nonzero()[0][0]
    idx2 = cur_position_ids_to_predict[idx1].nonzero()[0][0]
    padded_losses = padded_losses[:, idx2: idx2 + target_seq_len, :]    # (B, L, W)
    padded_flags = padded_flags[:, idx2: idx2 + target_seq_len, :]      # (B, L, W)
    
    weights = get_exp_dist(
        forward_size=forward_size, backward_size=backward_size, 
        forward_scale=1, backward_scale=1, 
        reverse_forward=True, device=shift_losses.device,
    )
    weighted_losses, weighted_flags = padded_losses * weights, padded_flags * weights
    
    # losses_1: losses reliable window range
    losses_1 = weighted_losses[..., :backward_size + 2].sum(-1) / weights[..., :backward_size + 2].sum(-1)
    # losses_2: losses in forward prediction range
    weights_2 = weights[..., backward_size + 2:].sum(-1)
    weights_2 = weights_2.masked_fill(weights_2.eq(0), 1)
    losses_2 = weighted_losses[..., backward_size + 2:].sum(-1) / weights_2
    # accept_flags: only for reliable range
    accept_flags = weighted_flags[..., :backward_size + 2].sum(-1) / weights[..., :backward_size + 2].gt(0).sum(-1)
    
    all_weights = get_exp_dist(
        forward_size=forward_size, backward_size=backward_size, 
        forward_scale=1, backward_scale=1, 
        device=shift_losses.device
    )
    losses = (padded_losses * all_weights).sum(-1) / all_weights.sum(-1)
    
    return losses_1, losses_2, losses, weighted_losses[..., backward_size + 1] / weights[..., backward_size + 1], accept_flags, candidates
    

def extract_distributions(model: AutoModelForCausalLM, hidden_states: list[torch.FloatTensor]):
    proj_func = lambda hs: model.lm_head(model.model.norm(hs))
    distributions = [proj_func(hs) for hs in hidden_states]
    return distributions


def cal_kl_divergence(last_layer_hidden_states: torch.FloatTensor, prev_layers_hidden_states: list[torch.FloatTensor]):
    batch_size, vocab_size = last_layer_hidden_states.size(0), last_layer_hidden_states.size(-1)
    seq_len = prev_layers_hidden_states[0].size(1)
    
    def _kl(log_q, p):
        return F.kl_div(log_q.view(-1, vocab_size), 
                        p.view(-1, vocab_size), 
                        reduction='none').mean(-1).view(seq_len, -1).contiguous()
    
    last_layer_hidden_states = last_layer_hidden_states.view(batch_size, seq_len, -1, vocab_size)
    window_size = last_layer_hidden_states.size(-2)
    
    # log_last_layer_hidden_states = last_layer_hidden_states.log_softmax(-1)
    # kl_divergences = torch.stack([
    #     torch.stack([
    #         _kl(
    #             log_last_layer_hidden_states[bid].contiguous(), 
    #             hs[bid].softmax(-1).unsqueeze(-2).expand(seq_len, window_size, vocab_size).contiguous(),
    #         ) for hs in prev_layers_hidden_states
    #     ], dim=0) for bid in range(batch_size)
    # ], dim=0)
    last_layer_hidden_states = last_layer_hidden_states.softmax(-1)
    kl_divergences = torch.stack([
        torch.stack([
            _kl(
                hs[bid].log_softmax(-1).unsqueeze(-2).expand(seq_len, window_size, vocab_size).contiguous(),
                last_layer_hidden_states[bid].contiguous(), 
            ) for hs in prev_layers_hidden_states
        ], dim=0) for bid in range(batch_size)
    ], dim=0)
    
    return kl_divergences.transpose(0, 1).contiguous()  # (L, B, S, W)


def cal_kl_divergence_pos(last_layer_hidden_states: torch.FloatTensor, prev_layers_hidden_states: list[torch.FloatTensor], position_ids_to_predict: torch.LongTensor):
    batch_size, vocab_size = last_layer_hidden_states.size(0), last_layer_hidden_states.size(-1)
    seq_len = prev_layers_hidden_states[0].size(1)
    last_layer_hidden_states = last_layer_hidden_states.view(batch_size, seq_len, -1, vocab_size)
    window_size = last_layer_hidden_states.size(-2)
    
    def _kl(log_q, p):
        return F.kl_div(log_q.view(-1, vocab_size), 
                        p.view(-1, vocab_size), 
                        reduction='none').mean(-1).view(-1).contiguous()
    
    last_layer_hidden_states = last_layer_hidden_states.softmax(-1)
    kl_divergences = []
    for hs in prev_layers_hidden_states:
        for bid in range(batch_size):
            for sid in range(seq_len):
                pos = position_ids_to_predict[bid][sid]
                kl_divergences.append(_kl(
                    hs[bid][pos].log_softmax(-1),
                    last_layer_hidden_states[bid][sid]
                ))
    kl_divergences = torch.stack(kl_divergences, dim=0).view(len(prev_layers_hidden_states), batch_size, seq_len, -1)
    return kl_divergences
    

def cal_pred_probability(labels: torch.LongTensor, prev_layers_hidden_states: list[torch.FloatTensor]):
    batch_size, seq_len = labels.size(0), labels.size(1)
    probs = []
    for hs in prev_layers_hidden_states:
        hs = hs.softmax(-1)
        for bid in range(batch_size):
            for sid in range(seq_len):
                prb = hs[bid][sid][labels[bid][sid]]
                prb = prb.masked_fill(labels[bid][sid].eq(IGNORE_INDEX), -1)
                probs.append(prb)
    probs = torch.stack(probs, dim=0).view(len(prev_layers_hidden_states), batch_size, seq_len, -1).contiguous()
    return probs
