from __future__ import annotations

from typing import Any
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaPreTrainedModel, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import (
    LlamaConfig, LlamaAttention, LlamaRMSNorm, LlamaMLP,
    _CONFIG_FOR_DOC, LLAMA_INPUTS_DOCSTRING, 
    rotate_half, apply_rotary_pos_emb, repeat_kv, _prepare_4d_causal_attention_mask_for_sdpa, 
    _prepare_4d_causal_attention_mask,
)
from transformers.generation.utils import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils.doc import add_start_docstrings_to_model_forward, replace_return_docstrings

from oa_dag.models.oa_model import OAModelMixin, OAModelOutput, BaseModelOutputWithPastOA
from oa_dag.utils import pad_tensors
from oa_dag.configs.constants import IGNORE_INDEX

logger = logging.get_logger(__name__)

def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

class LlamaAttentionOA(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,    # (B, L, H)
        attention_mask: torch.Tensor | None = None,    # (B, 1, L, L)
        position_ids: torch.LongTensor | None = None,    # (B, L)
        positions_to_replace: torch.LongTensor | None = None,  # (B, 1)
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, N)
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # prepare cos & sin for positional encoding
        cos, sin = self.rotary_emb(value_states, seq_len=(torch.cat((position_ids, position_ids_to_predict), dim=-1) if position_ids_to_predict is not None else position_ids).max().item() + 1)
        # clone the original query_states for position embedding
        raw_query_states = query_states.clone()
        
        if position_ids_to_predict is not None:
            # position embedding for last layer
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, torch.cat((position_ids[:, 1:].contiguous(), position_ids[:, :1].contiguous()), dim=-1))
        else:
            # original position embedding
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        if position_ids_to_predict is not None:
            for i in range(position_ids_to_predict.size(-1)):
                # extract the corresponding query (at the next specific positions)
                query_states = torch.stack([raw_query_states[b_id, :, positions_to_replace[b_id] - 1, :].contiguous() for b_id in range(raw_query_states.size(0))], dim=0)
                # apply position encoding
                query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids_to_predict[:, i].unsqueeze(-1))
                
                # update attention weights
                update_attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attention_mask is not None:
                    corr_attention_mask = torch.stack([attention_mask[b_id, :, positions_to_replace[b_id] - 1, :].contiguous() for b_id in range(attention_mask.size(0))], dim=0)
                    update_attn_weights = update_attn_weights + corr_attention_mask
                attn_weights = torch.cat((attn_weights, update_attn_weights), dim=2)
        
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaSdpaAttentionOA(LlamaAttentionOA):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        positions_to_replace: torch.LongTensor | None = None,  # (B, 1)
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, N)
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if output_attentions:   # or position_ids_to_predict is not None:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                positions_to_replace=positions_to_replace,
                position_ids_to_predict=position_ids_to_predict,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # prepare cos & sin for positional encoding
        cos, sin = self.rotary_emb(value_states, seq_len=(torch.cat((position_ids, position_ids_to_predict), dim=-1) if position_ids_to_predict is not None else position_ids).max().item() + 1)
        # clone the original query_states for position embedding
        raw_query_states = query_states.clone()

        if position_ids_to_predict is not None:
            # position embedding for last layer
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, torch.cat((position_ids[:, 1:].contiguous(), position_ids[:, :1].contiguous()), dim=-1))
        else:
            # original position embedding
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        if position_ids_to_predict is not None:
            for i in range(position_ids_to_predict.size(-1)):
                # extract the corresponding query (at the next specific positions)
                new_query_states = torch.stack([raw_query_states[b_id, :, positions_to_replace[b_id] - 1, :].contiguous() for b_id in range(raw_query_states.size(0))], dim=0)
                # apply position encoding
                new_query_states = apply_rotary_pos_emb_single(new_query_states, cos, sin, position_ids_to_predict[:, i].unsqueeze(-1))
                # update query states
                query_states = torch.cat((query_states, new_query_states), dim=2)
                # update attention mask
                if attention_mask is not None:
                    new_attention_mask = torch.stack([attention_mask[b_id, :, positions_to_replace[b_id] - 1, :].contiguous() for b_id in range(attention_mask.size(0))], dim=0)
                    attention_mask = torch.cat((attention_mask, new_attention_mask), dim=2)
        
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, query_states.size(-2), self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttentionOA,
    # "flash_attention_2": LlamaFlashAttention2OA,
    "sdpa": LlamaSdpaAttentionOA,
}

class LlamaDecoderLayerOA(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        positions_to_replace: torch.LongTensor | None = None,  # (B, 1)
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, N)
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            positions_to_replace=positions_to_replace,
            position_ids_to_predict=position_ids_to_predict,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        if positions_to_replace is not None:
            # pad with the corresponding hidden states for residual calculation
            extended_len = hidden_states.size(1) - residual.size(1)
            add_residual = torch.stack([residual[b_id, positions_to_replace[b_id][0] - 1] for b_id in range(positions_to_replace.size(0))]).unsqueeze(1).expand(residual.size(0), extended_len, residual.size(-1))
            residual = torch.cat((residual, add_residual), dim=1)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        return outputs

class LlamaModelOA(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayerOA(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.omit_last_layer = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPastOA:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in (self.layers[:-1] if self.omit_last_layer else self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    None,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, attention_mask] if v is not None)
        return BaseModelOutputWithPastOA(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            last_attention_mask=attention_mask,
        )


class LlamaForCausalLMOA(OAModelMixin, LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    LAYER_TYPE = LlamaDecoderLayerOA

    def __init__(self, config: LlamaConfig, **kwargs: Any):
        super().__init__(config)
        self.model = LlamaModelOA(config)
        self.vocab_size = config.vocab_size
        self.init_oa_layer(config, **kwargs)
        self.model.omit_last_layer = not self.additional_layer
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OAModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        positions_to_replace: torch.LongTensor | None = None,  # (B, 1)
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, N)
        topk_probs: torch.FloatTensor | None = None,    # (B, M, K)
        topk_ids: torch.LongTensor | None = None,   # (B, M, K)
        replace_indexes: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | OAModelOutput:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:
        
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            _, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            _, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
        past_key_values_length = 0
        
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        if topk_ids is not None:
            # concatenate original input with noisy self-generations
            concat_input_ids = torch.cat((input_ids, topk_ids.view(input_ids.size(0), -1).contiguous()), dim=1).contiguous()
            # get embeddings
            concat_inputs_embeds = self.model.embed_tokens(concat_input_ids)
            orig_inputs_embeds = concat_inputs_embeds[:, :input_ids.size(-1), :].contiguous()
            noisy_inputs_embeds = concat_inputs_embeds[:, input_ids.size(-1):, :].contiguous().view(input_ids.size(0), -1, topk_ids.size(-1), concat_inputs_embeds.size(-1)).contiguous()
            # get weighted embeddings
            weighted_noisy_inputs_embeds = topk_probs.unsqueeze(-1).type(noisy_inputs_embeds.dtype).contiguous() * noisy_inputs_embeds
            weighted_noisy_inputs_embeds = weighted_noisy_inputs_embeds.sum(dim=-2) / topk_probs.sum(dim=-1).unsqueeze(-1).type(noisy_inputs_embeds.dtype)
            # replace original ones with weighted embeddings
            for i in range(input_ids.size(0)):
                replace_indexes_i = replace_indexes[i][replace_indexes[i].ge(0).nonzero().squeeze(dim=-1)]
                if labels is None or labels[i][replace_indexes_i].ne(IGNORE_INDEX).any().item():
                    orig_inputs_embeds[i][replace_indexes_i] = weighted_noisy_inputs_embeds[i][:replace_indexes_i.size(-1), :].contiguous()
            inputs_embeds = orig_inputs_embeds.contiguous()
            input_ids = None
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if return_dict:
            hidden_states = outputs.last_hidden_state
            last_attention_mask = outputs.last_attention_mask
        else:
            hidden_states = outputs[0]
            last_attention_mask = outputs[-1]
        
        hidden_states.requires_grad_()        
        if self.model.gradient_checkpointing and self.model.training:
            layer_outputs = self.model._gradient_checkpointing_func(
                self.oa_layer.__call__,
                hidden_states,
                last_attention_mask,
                position_ids,
                positions_to_replace,
                position_ids_to_predict,
            )
        else:
            layer_outputs = self.oa_layer(
                hidden_states,
                attention_mask=last_attention_mask,
                position_ids=position_ids,
                positions_to_replace=positions_to_replace,
                position_ids_to_predict=position_ids_to_predict,
            )        
        hidden_states = layer_outputs[0]
        
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            if positions_to_replace is not None:
                orig_len = position_ids.size(-1)
                # Shift so that tokens < n predict n
                shift_logits = torch.cat((logits[..., :orig_len-1, :].contiguous(), logits[..., orig_len:, :].contiguous()), dim=1)
                shift_labels = labels[..., 1:].contiguous()
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, hidden_states,) + outputs[1:-1]
            return (loss,) + output if loss is not None else output

        return OAModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=outputs.past_key_values,
            prev_hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def oa_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        positions_to_replace: torch.LongTensor | None = None,
        position_ids_to_predict: torch.LongTensor | None = None,
        temperature: float = 0.0,
        max_length: int = 512,
        top_p: float | None = None,
        top_k: int | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        min_n_tokens: int = 1,
        max_n_tokens: int = 1,
        do_sample: bool = False,
        denoising: bool = False,
        topk: int = 8,
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        
        processors = LogitsProcessorList()
        warpers = LogitsProcessorList()
        if temperature != 0.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        else:
            top_k = 1
        if top_k is not None and top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        if top_p is not None and 0.0 < top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
        
        if positions_to_replace is None:
            positions_to_replace = torch.tensor([[input_ids.size(-1)]], dtype=torch.long, device=input_ids.device)
        if position_ids_to_predict is None:
            position_ids_to_predict = torch.arange(
                input_ids.size(-1), max_length, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.size(), dtype=torch.bool)
        
        raw_input_ids = input_ids.clone().detach()
        tracks = []
        keep_generate, accumulated_n_tokens = True, 0
        topk_probs, topk_ids, replace_indexes = None, None, None
        while keep_generate:
            # forward pass to get tokens at position_ids_to_predict
            n_tokens_to_keep = min(max(min_n_tokens, accumulated_n_tokens), position_ids_to_predict.size(-1), max_n_tokens)
            accumulated_n_tokens += n_tokens_to_keep
            logits = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids, 
                positions_to_replace=positions_to_replace,
                position_ids_to_predict=position_ids_to_predict,
                topk_probs=topk_probs,
                topk_ids=topk_ids,
                replace_indexes=replace_indexes,
                return_dict=True,
            ).logits[:, input_ids.size(-1):].contiguous()
            # topk_labels = torch.topk(logits, k=10, dim=-1).indices
            
            # extract scores from logits
            token_scores = processors(input_ids, logits.view(-1, logits.size(-1)))
            token_scores = warpers(input_ids, token_scores)
            probs = nn.functional.softmax(token_scores, dim=-1)
            
            if denoising:
                results = torch.topk(nn.functional.softmax(logits, dim=-1), k=topk, dim=-1)
                topk_probs, topk_ids = results.values[:, :accumulated_n_tokens, :], results.indices[:, :accumulated_n_tokens, :]
                if position_ids_to_predict[:, :accumulated_n_tokens].max() + 1 > input_ids.size(-1):
                    replace_indexes = position_ids_to_predict[:, :accumulated_n_tokens]
                    input_ids = pad_tensors(input_ids, max_len=position_ids_to_predict[:, :accumulated_n_tokens].max() + 1, pad_value=tokenizer.pad_token_id)
                    attention_mask = pad_tensors(attention_mask, max_len=position_ids_to_predict[:, :accumulated_n_tokens].max() + 1, pad_value=True).bool()
                    position_ids = None     # torch.cat((position_ids, replace_indexes), dim=-1)
                    positions_to_replace = (position_ids_to_predict[:, :accumulated_n_tokens].max(dim=-1).values + 1).long().unsqueeze(-1)
                input_ids[0][replace_indexes[0]] = torch.multinomial(probs[:accumulated_n_tokens, ...], num_samples=1).squeeze(1)
                tracks.append([
                    input_ids[0][replace_indexes[0]].clone().tolist(),
                    replace_indexes[0].clone().tolist()
                ])
            else:
                # sample and get candidate tokens
                tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                try:
                    scores = torch.stack([probs[idx, tokens[idx]] for idx in range(tokens.size(0))], dim=0)
                except:
                    import ipdb; ipdb.set_trace()
                tokens = tokens.view(input_ids.size(0), -1)
                scores = scores.view(input_ids.size(0), -1)
                # keep tokens with the highest scores
                scores[position_ids_to_predict.eq(0)] = -math.inf
                _, sorted_ids = torch.sort(scores, dim=-1, descending=True)
                ############### TODO: keep strategy ###############
                n = accumulated_n_tokens if denoising else n_tokens_to_keep
                if not do_sample:
                    tokens_to_keep = sorted_ids[:, :n]
                else:
                    tokens_to_keep = sorted_ids[:, :int(n * 2)]
                    random_noise = torch.rand_like(tokens_to_keep, dtype=torch.float)
                    _, sorted_ids = torch.sort(random_noise, dim=-1)
                    tokens_to_keep = torch.stack([tokens_to_keep[bid][sorted_ids[bid, :n]] for bid in range(sorted_ids.size(0))], dim=0)
                ###################################################
                
                new_input_ids_list = []
                new_position_ids_list = []
                new_positions_to_replace_list = []
                new_position_ids_to_predict_list = []
                new_max_length = 0
                cur_step = []
                for bid in range(input_ids.size(0)):
                    # append the tokens to keep
                    if denoising:
                        cur_input_ids = raw_input_ids[bid][raw_input_ids[bid].ne(tokenizer.pad_token_id).nonzero().squeeze(dim=-1)]
                        cur_position_ids = position_ids[bid][raw_input_ids[bid].ne(tokenizer.pad_token_id).nonzero().squeeze(dim=-1)]
                    else:
                        cur_input_ids = input_ids[bid][attention_mask[bid].nonzero().squeeze(dim=-1)]
                        cur_position_ids = position_ids[bid][attention_mask[bid].nonzero().squeeze(dim=-1)]
                    add_input_ids = tokens[bid][tokens_to_keep[bid]]
                    add_position_ids = position_ids_to_predict[bid][tokens_to_keep[bid]]
                    new_input_ids = torch.cat((cur_input_ids, add_input_ids), dim=-1)
                    new_position_ids = torch.cat((cur_position_ids, add_position_ids), dim=-1)
                    # sort the inputs into the ascending order
                    _, tmp_idx = new_position_ids.sort()
                    new_input_ids = new_input_ids[tmp_idx]
                    new_position_ids = new_position_ids[tmp_idx]
                    cur_step.append([new_input_ids[raw_input_ids[bid].ne(tokenizer.pad_token_id).sum():].tolist(), 
                                    new_position_ids[raw_input_ids[bid].ne(tokenizer.pad_token_id).sum():].tolist()])
                    # update the positions to predict
                    new_position_ids_to_predict = [x for x in position_ids_to_predict[bid] if x not in add_position_ids]
                    if sum(new_position_ids_to_predict) <= 0: continue
                    new_position_ids_to_predict = torch.stack(new_position_ids_to_predict, dim=0)
                    
                    new_input_ids_list.append(new_input_ids)
                    new_position_ids_list.append(new_position_ids)
                    new_positions_to_replace_list.append(torch.tensor(new_input_ids.size(-1), dtype=torch.long, device=new_input_ids.device).unsqueeze(-1))
                    new_position_ids_to_predict_list.append(new_position_ids_to_predict)
                    new_max_length = max(new_max_length, new_input_ids.size(-1))
                for bid in range(len(new_input_ids_list)):
                    pad_len = new_max_length - new_input_ids_list[bid].size(-1)
                    tmp = torch.ones((pad_len,), dtype=torch.long, device=new_input_ids_list[bid].device)
                    new_input_ids_list[bid] = torch.cat((new_input_ids_list[bid], tmp * tokenizer.pad_token_id), dim=-1).long()
                    new_position_ids_list[bid] = torch.cat((new_position_ids_list[bid], tmp * 0), dim=-1).long()
                
                tracks.append(cur_step[0])
                if len(new_input_ids_list) <= 0: break
                
                input_ids = torch.stack(new_input_ids_list, dim=0)
                attention_mask = input_ids.ne(tokenizer.pad_token_id)
                position_ids = torch.stack(new_position_ids_list, dim=0)
                positions_to_replace = torch.stack(new_positions_to_replace_list, dim=0)
                if not denoising:
                    position_ids_to_predict = torch.stack(new_position_ids_to_predict_list, dim=0)
                
            print(tokenizer.decode(input_ids[0]))
            if '</s>' in tokenizer.decode(input_ids[0]) and '\n#### ' in tokenizer.decode(input_ids[0]): break
            if accumulated_n_tokens > position_ids_to_predict.max().item() + 5: break
            # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        
        return tracks, input_ids
