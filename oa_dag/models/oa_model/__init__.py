from __future__ import annotations

import functools
import importlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import transformers.models.auto as auto_module
from torch import distributed as dist
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.auto.auto_factory import (
    _BaseAutoModelClass,
    _LazyAutoMapping,
    auto_class_update,
    getattribute_from_module,
)
from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING_NAMES,
    model_type_to_module_name,
)
from transformers.utils.generic import ModelOutput


class _LazyAutoMappingInOA(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type: str, attr: str) -> Any:
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f'.{module_name}',
                'oa_dag.models.oa_model',
            )
        return getattribute_from_module(self._modules[module_name], attr)


MODEL_FOR_OA_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        # OA model mapping
        ('llama', 'LlamaForCausalLMOA'),
        ('mistral', 'MistralForCausalLMOA'),
    ],
)
MODEL_FOR_OA_MAPPING: OrderedDict[str, Any] = _LazyAutoMappingInOA(
    CONFIG_MAPPING_NAMES,
    MODEL_FOR_OA_MAPPING_NAMES,
)


@functools.partial(auto_class_update, head_doc='order-agnostic model')
class AutoModelForOA(_BaseAutoModelClass):
    _model_mapping: OrderedDict[str, Any] = MODEL_FOR_OA_MAPPING


setattr(auto_module, 'MODEL_FOR_OA_MAPPING', MODEL_FOR_OA_MAPPING)  # noqa: B010
setattr(auto_module, AutoModelForOA.__name__, AutoModelForOA)

@dataclass
class BaseModelOutputWithPastOA(BaseModelOutputWithPast):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    last_attention_mask: torch.BoolTensor = None

@dataclass
class OAModelOutput(ModelOutput):
    """
    Output of the order-agnostic model.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_dim)`):
            Sequence of hidden-states at the output of the last layer of the model.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None  # size = (B, L - 1, V)
    hidden_states: torch.FloatTensor | None = None  # size = (B, L - 1, E)
    past_key_values: torch.FloatTensor | None = None
    prev_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class OAModelMixin:
    """Base class for order-agnostic models."""
    
    LAYER_TYPE = nn.Module
    
    def init_oa_layer(self, config: PretrainedConfig, additional_layer: bool = False) -> None:
        """"""
        self.additional_layer = additional_layer
        if self.additional_layer:
            self.oa_layer = self.LAYER_TYPE(config, config.num_hidden_layers)
        else:
            self.oa_layer = self.model.layers[-1]