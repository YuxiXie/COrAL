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
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import ModelOutput
from transformers.generation.utils import LogitsProcessorList
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


def jsonlines_load(x):
    with jsonlines.open(x, mode='r') as reader:
        data = [r for r in reader]
    return data


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
    # window_size = max(forward_size, backward_size) * 2
    window_size = forward_size + backward_size
    # x = torch.linspace(-r, r, window_size + 1)
    x = torch.linspace(0, 2 * r, window_size + 1)
    x = torch.exp(normal.log_prob(x))
    
    # x = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 2e-05, 0.125, 2e-05, 2e-05, 2e-05])
    
    # mid_idx = len(x) // 2
    # return x[mid_idx - backward_size - 1: mid_idx + forward_size]
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


def get_exp_dist(forward_size=4, backward_size=8, forward_scale=8, backward_scale=1/8):
    x_forward = torch.arange(0, forward_size)
    x_forward = (-forward_scale * x_forward).exp()
    
    if backward_size < 0:
        return x_forward
    if backward_size < 1:
        return torch.cat((x_forward[:1]/10, x_forward), dim=-1)
    
    x_backward = -torch.arange(0, backward_size - 1) + backward_size - 2
    x_backward = torch.cat((x_backward, torch.arange(1, 3)), dim=-1)
    x_backward = (-backward_scale * x_backward).exp()
    
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
    input_ids: torch.LongTensor,
    logits: torch.FloatTensor, 
    ref_position_ids_to_predict: torch.LongTensor,
    pred_start_pos: int, 
    pred_end_pos: int,
    forward_size: int = 4,
    backward_size: int = 8,
):
    batch_size = input_ids.size(0)
    logprobs = logits.view(-1, logits.size(-1)).log_softmax(dim=-1).view(logits.size(1), -1, logits.size(-1)).unsqueeze(0)    # (1 * S, V) --> (1, L', N, V)
    windwo_size = forward_size + backward_size + 1
    target_seq_len = pred_end_pos - pred_start_pos + 1
    
    # extract scores from logits
    logprobs_dict = torch.zeros(batch_size, target_seq_len, windwo_size, logprobs.size(-1), dtype=logprobs.dtype, device=logprobs.device)  # (1, T, V)
    weights = get_exp_dist(forward_size=forward_size, backward_size=backward_size).to(logprobs.device)
    weights_dict = torch.zeros(batch_size, target_seq_len, windwo_size, dtype=weights.dtype, device=weights.device)  # (1, T, V)
    for i in range(batch_size):
        for j in range(windwo_size):
            cur_positions_indexes = ref_position_ids_to_predict[i, :, j].ge(pred_start_pos).nonzero().squeeze(-1)
            if cur_positions_indexes.size(-1) <= 0: continue
            cur_positions = ref_position_ids_to_predict[i, cur_positions_indexes, j]
            
            logprobs_dict[i, cur_positions - pred_start_pos, j] = logprobs[i, cur_positions_indexes, j] * weights[j]
            weights_dict[i, cur_positions - pred_start_pos, j] = weights[j]
    
    ensemble_logits = logprobs_dict.sum(-2) / weights_dict.sum(-1).unsqueeze(-1)
    return logprobs_dict, ensemble_logits


def create_tree_attention_mask(logprobs: torch.FloatTensor, forward_size: int = 1, topk: int = 16, maximum_seq_len: int = 100):
    n_depth = logprobs.size(0)
    results = torch.topk(logprobs.view(-1, logprobs.size(-1)), k=topk, dim=-1)
    topk = results.values.size(-1)
    
    # sort tokens by confidence scores
    confidence = results.values
    for i in range(forward_size):
        # scale last few tokens
        confidence[-i - 1] = confidence[-i - 1] / (forward_size - i)
    # if n_depth > forward_size:
        # scale_len = min(n_depth - forward_size, 3)
        # confidence[:scale_len] = confidence[:scale_len] / max(forward_size - 1, 2)
        # confidence[0] = confidence[0] / max(forward_size - 1, 2)
    confidence = confidence.exp().view(-1)
    sorted_conf = confidence.sort(descending=True)
    
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
    
    cur_attention_mask = torch.tril(torch.ones((cur_input_ids.size(-1), cur_input_ids.size(-1)))).bool().to(cur_input_ids.device)
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
    processors: LogitsProcessorList = LogitsProcessorList(),
    topk: int = 16,
    max_new_tokens = 256,
    max_length: int = 512,
):
    # stime = time.time()
    # aggregate all predicted distributions
    _, ensemble_logits = extract_logprobs_into_dict(
        input_ids=input_ids,
        logits=logits,
        ref_position_ids_to_predict=ref_position_ids_to_predict,
        pred_start_pos=pred_start_pos,
        pred_end_pos=pred_end_pos,
        forward_size=forward_size,
        backward_size=backward_size,
    )
    
    # sample and get candidate tokens
    token_scores = processors(input_ids, ensemble_logits.view(-1, ensemble_logits.size(-1)))  # (1 * T, V)
    logprobs = nn.functional.log_softmax(token_scores, dim=-1)  # (1 * T, V)
    # token_scores = greedy_processors(input_ids, token_scores)
    # tokens = torch.multinomial(token_scores.softmax(-1), num_samples=1).squeeze(-1).unsqueeze(0)
    # print('[P1-1]', time.time() - stime)
    
    # stime = time.time()
    # tree attention construction
    max_new_tokens = min(max_new_tokens, max_length - pred_start_pos)
    candidate_ids, candidate_position_ids, tree_attn_mask, retrieve_indices = \
        create_tree_attention_mask(
            logprobs, 
            topk=topk, 
            forward_size=forward_size,
            maximum_seq_len=max_new_tokens,
        )
    retrieve_indices = retrieve_indices + input_ids[0, :pred_start_pos].size(-1) - 1
    # print('[P1-2]', time.time() - stime)
    
    # stime = time.time()
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
    # print('[P1-3]', time.time() - stime)
    
    return cur_input_ids, cur_position_ids, cur_attention_mask, cur_position_ids_to_predict, retrieve_indices, logprobs


def calculate_candidate_losses(
    cur_input_ids: torch.LongTensor,
    cur_position_ids_to_predict: torch.LongTensor,
    candidate_logits: torch.FloatTensor,
    retrieve_indices: torch.LongTensor,
    pred_start_pos: int,
    pred_end_pos: int,
    forward_size: int = 4,
    backward_size: int = 8,
):
    # stime = time.time()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    shift_logits, shift_labels, candidates = [], [], []
    for indices in retrieve_indices:
        shift_logits.append(candidate_logits[indices])
        labels_i = cur_input_ids[indices].clone()
        labels_i[0] = IGNORE_INDEX
        positions_i = cur_position_ids_to_predict[indices] - pred_start_pos + 1
        positions_i = positions_i.masked_fill(positions_i.lt(1), 0)
        shift_labels.append(labels_i[positions_i])
        candidates.append(cur_input_ids[indices][1:])
    shift_logits, shift_labels = torch.stack(shift_logits, dim=0).view(-1, candidate_logits.size(-1)), torch.stack(shift_labels, dim=0).view(-1)
    shift_losses = loss_fct(shift_logits, shift_labels).view(retrieve_indices.size(0), retrieve_indices.size(-1), -1)
    candidates = torch.stack(candidates, dim=0).view(retrieve_indices.size(0), -1)
    # print('[P2-1]', time.time() - stime)
    
    # stime = time.time()
    batch_size, target_seq_len = retrieve_indices.size(0), retrieve_indices.size(-1) - 1
    window_size = forward_size + backward_size + 1
    losses_dict = torch.zeros(batch_size, target_seq_len, window_size, dtype=shift_losses.dtype, device=shift_losses.device)  # (1, T, V)
    weights = get_exp_dist(forward_size=forward_size, backward_size=backward_size).to(shift_losses.device)
    weights_dict = torch.zeros(batch_size, target_seq_len, window_size, dtype=weights.dtype, device=weights.device)  # (1, T, V)
    for i in range(batch_size):
        shift_losses_i, indices = shift_losses[i], retrieve_indices[i]
        cur_position_ids_to_predict_i = cur_position_ids_to_predict[indices]
        for j in range(window_size):
            cur_positions_indexes = cur_position_ids_to_predict_i[:, j].ge(pred_start_pos).nonzero().squeeze(-1)
            if cur_positions_indexes.size(-1) <= 0: continue
            cur_positions = cur_position_ids_to_predict_i[cur_positions_indexes, j]
            
            losses_dict[i, cur_positions - pred_start_pos, j] = shift_losses_i[cur_positions_indexes, j] * weights[j]
            weights_dict[i, cur_positions - pred_start_pos, j] = weights[j]
    losses = (losses_dict.sum(-1) / weights_dict.sum(-1)).mean(-1)
    # print('[P2-2]', time.time() - stime)

    return losses, candidates


