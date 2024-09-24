from __future__ import annotations

from typing import Any

import math

import torch
import torch.nn as nn
from transformers import MistralPreTrainedModel, PreTrainedTokenizerBase
from transformers.models.mistral.modeling_mistral import (
    MistralConfig, MistralAttention, MistralRMSNorm, MistralMLP,
    _CONFIG_FOR_DOC, MISTRAL_INPUTS_DOCSTRING, 
    rotate_half, apply_rotary_pos_emb, repeat_kv, 
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa, 
    _prepare_4d_causal_attention_mask,
)
from transformers.generation.utils import (
    LogitsProcessorList, TemperatureLogitsWarper, 
    TopKLogitsWarper, TopPLogitsWarper, EtaLogitsWarper,
)
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils.doc import add_start_docstrings_to_model_forward, replace_return_docstrings

from oa_dag.models.oa_model import OAModelMixin, OAModelOutput, BaseModelOutputWithPastOA
from oa_dag.utils import (
    prepare_candidates, calculate_candidate_losses,
)
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

class MistralAttentionOA(MistralAttention):
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
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()

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
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1)
        
        if position_ids_to_predict is not None:
            # position embedding for last layer
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)
        else:
            # original position embedding
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # repeat k/v heads if n_kv_heads < n_heads
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
            # extract the corresponding query (at the next specific positions)
            new_query_states = query_states.unsqueeze(-2).expand(bsz, query_states.size(1), q_len, position_ids_to_predict.size(-1), query_states.size(-1)).contiguous().view(bsz, query_states.size(1), -1, query_states.size(-1))
            # apply position encoding
            query_states = apply_rotary_pos_emb_single(new_query_states, cos, sin, position_ids_to_predict.view(bsz, -1))
            
            # update attention weights
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                # update attention mask
                attention_mask = attention_mask.unsqueeze(-2).expand(bsz, 1, q_len, position_ids_to_predict.size(-1), kv_seq_len).contiguous().view(bsz, 1, -1, kv_seq_len)
                attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        if position_ids_to_predict is None and attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        elif position_ids_to_predict is not None and attn_output.size() != (bsz, self.num_heads, q_len + position_ids_to_predict.size(-1), self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        cur_len = q_len if position_ids_to_predict is None else (q_len + position_ids_to_predict.size(-1))
        attn_output = attn_output.reshape(bsz, cur_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

class MistralSdpaAttentionOA(MistralAttentionOA):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, Wf + Wb + 1)
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if output_attentions:   # or position_ids_to_predict is not None:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MistralModel is using MistralSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
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
        seq_len = (max(position_ids.max().item(), position_ids_to_predict.max().item()) if position_ids_to_predict is not None else position_ids.max().item()) + 1
        # tmp_position_ids = torch.arange(0, seq_len, dtype=torch.long, device=position_ids.device).unsqueeze(0).expand(position_ids.size(0), seq_len)
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        # cos, sin = self.rotary_emb(value_states, tmp_position_ids)
        
        if position_ids_to_predict is not None:
            # position embedding for last layer
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)
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
            # extract the corresponding query (at the next specific positions)
            # (B, n_head, L, Wf+Wb+1, n_dim) --> (B, n_head, L * (Wf+Wb+1), n_dim)
            new_query_states = query_states.unsqueeze(-2).expand(bsz, query_states.size(1), q_len, position_ids_to_predict.size(-1), query_states.size(-1)).contiguous().view(bsz, query_states.size(1), -1, query_states.size(-1))
            # apply position encoding
            query_states = apply_rotary_pos_emb_single(new_query_states, cos, sin, position_ids_to_predict.view(bsz, -1))
            
            if attention_mask is not None:
                # update attention mask
                # (B, 1, L * (Wf+Wb+1), L)
                attention_mask = attention_mask.unsqueeze(-2).expand(bsz, 1, q_len, position_ids_to_predict.size(-1), kv_seq_len).contiguous().view(bsz, 1, -1, kv_seq_len)
        
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

MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttentionOA,
    # "flash_attention_2": MistralFlashAttention2OA,
    "sdpa": MistralSdpaAttentionOA,
}

class MistralDecoderLayerOA(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MISTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, Wf + Wb + 1)
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
            position_ids_to_predict=position_ids_to_predict,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        if position_ids_to_predict is not None:
            residual = residual.unsqueeze(2).expand(residual.size(0), residual.size(1), position_ids_to_predict.size(-1), residual.size(-1)).contiguous().view(residual.size(0), -1, residual.size(-1))
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

class MistralModelOA(MistralPreTrainedModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MistralDecoderLayerOA(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.oa_layer = self.layers[-1]
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.omit_last_layer = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, Wf + Wb + 1)
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
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

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
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            if attention_mask is None or attention_mask.dim() < 4:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.config.sliding_window,
                )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

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
        # last layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.training:
            hidden_states.requires_grad_()
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                self.oa_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                position_ids_to_predict,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = self.oa_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_ids_to_predict=position_ids_to_predict,
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

class MistralForCausalLMOA(OAModelMixin, MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    LAYER_TYPE = MistralDecoderLayerOA

    def __init__(self, config: MistralConfig, **kwargs: Any):
        super().__init__(config)
        self.model = MistralModelOA(config)
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

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OAModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,     # (B, L)
        attention_mask: torch.Tensor | None = None,     # (B, L)
        position_ids: torch.LongTensor | None = None,     # (B, L)
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, Wf + Wb + 1)
        topk_probs: torch.FloatTensor | None = None,    # (B, M, K)
        topk_ids: torch.LongTensor | None = None,   # (B, M, K)
        replace_indexes: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,    # (B, L, Wf + Wb + 1)
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
        
        if replace_indexes is not None:
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
            position_ids_to_predict=position_ids_to_predict,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            assert position_ids_to_predict is not None and position_ids_to_predict.size() == labels.size()
            # Flatten the tokens
            shift_logits = logits.contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels.contiguous().view(-1)
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

    def next_token_generate(
        self,
        input_ids: torch.LongTensor,    # (B, L)
        attention_mask: torch.BoolTensor | None = None,     # (B, L)
        position_ids: torch.LongTensor | None = None,       # (B, L)
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, Wf + Wb + 1)
        temperature: float = 0.0,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_length: int = 512,
        verbal: bool = False,
    ):
        processors = LogitsProcessorList()
        if temperature != 0.0:
            processors.append(TemperatureLogitsWarper(temperature))
        else:
            processors.append(TopKLogitsWarper(top_k=1, min_tokens_to_keep=1))
        
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        keep_generate, prev_end_idx, tracks = True, 0, []
        while keep_generate:
            outputs: OAModelOutput = self(
                input_ids=input_ids[:, prev_end_idx:],    # (1, L)
                attention_mask=attention_mask,    # (1, L)
                position_ids=position_ids[:, prev_end_idx:] if position_ids is not None else None,    # (1, L)
                position_ids_to_predict=position_ids_to_predict[:, prev_end_idx:, :],    # (1, L, N)
                return_dict=True,
            )
            logits = outputs.logits.contiguous().view(batch_size, input_ids[:, prev_end_idx:].size(-1), position_ids_to_predict.size(-1), -1)
            logits = logits.squeeze(-2).contiguous()[:, -1, :]    # (1, L * N, V) --> (1, L, V) (N = 1) --> (1, V)
            
            # extract scores from logits
            token_scores = processors(input_ids, logits.view(-1, logits.size(-1)))  # (1, V)
            probs = nn.functional.softmax(token_scores, dim=-1)  # (1, V)
            # sample and get candidate tokens
            tokens = torch.multinomial(probs, num_samples=1).squeeze(-1).view(batch_size, -1)   # (1, 1)
            
            input_ids = torch.cat((input_ids, tokens), dim=-1)
            tracks.append([
                input_ids[0, seq_length:].clone().tolist(),
                torch.arange(seq_length, input_ids.size(-1), dtype=torch.long, device=input_ids.device).tolist(),
            ])            
            if verbal:
                iter_cnt = len(tracks)
                print(f'[{iter_cnt}]', tokenizer.decode(input_ids[0]))
            
            if input_ids.eq(tokenizer.eos_token_id).any() or input_ids.size(-1) >= max_length:
                keep_generate = False
                break
            
            if position_ids is not None:
                position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
            position_ids_to_predict = torch.cat((
                position_ids_to_predict, torch.tensor([[[input_ids.size(-1)]]]).long().to(position_ids_to_predict.device)
            ), dim=1)
        
        return tracks, input_ids
    
    def multiple_token_generate(
        self,
        input_ids: torch.LongTensor,    # (B, L)
        attention_mask: torch.BoolTensor | None = None,     # (B, L)
        position_ids: torch.LongTensor | None = None,       # (B, L)
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, Wf + Wb + 1)
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_length: int = 512,
        block_size: int = 16,
        forward_size: int = 4,
        backward_size: int = 8,
        eval_forward_size: int = 4,
        eval_backward_size: int = 8,
        occurance_threshold: int = 8,
        topk: int = 16,
        topp: float = .99,
        max_iter_times: int = 512,
        verbal: bool = False,
        skip_verify: bool = True,
    ):
        import time
        
        processors = LogitsProcessorList()
        # processors.append(TopPLogitsWarper(top_p=topp, min_tokens_to_keep=1))
        # processors.append(EtaLogitsWarper(epsilon=1-topp))
        greedy_processors = LogitsProcessorList()
        greedy_processors.append(TopKLogitsWarper(top_k=1, min_tokens_to_keep=1))
        
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        start_idx, end_idx = seq_length, max_length - 1
        keep_generate, tracks = True, []
        pred_start_pos = fixed_seq_length = seq_length
        occurance_counts = torch.ones_like(input_ids)  # record the occurance times of predicted tokens
        iter_cnt_last, iter_cnt_mid, prev_max_start_idx = 0, 0, -1
        accept_ratios = None
        while keep_generate:
            outputs: OAModelOutput = self(
                input_ids=input_ids,    # (1, L)
                attention_mask=torch.ones_like(input_ids).bool(),    # (1, L)
                position_ids=position_ids,    # (1, L)
                position_ids_to_predict=position_ids_to_predict,    # (1, L, N)
                return_dict=True,
            )
            logits = outputs.logits.contiguous().view(batch_size, input_ids.size(-1), position_ids_to_predict.size(-1), -1)     # (1, L * N, V) --> (1, L, N, V)
            logits = logits[:, seq_length - 1:, ...].contiguous()    # (1, L, N, V) --> (1, L', N, V)
            
            pred_start_pos = fixed_seq_length
            pred_end_pos = position_ids_to_predict.max().item()
            ref_position_ids_to_predict = position_ids_to_predict[:, seq_length - 1:, :]    # (1, L', N)
            
            if skip_verify:
                logprobs = prepare_candidates(
                    input_ids=input_ids,
                    logits=logits,
                    ref_position_ids_to_predict=ref_position_ids_to_predict,
                    pred_start_pos=pred_start_pos,
                    pred_end_pos=pred_end_pos,
                    forward_size=forward_size,
                    backward_size=backward_size,
                    eval_forward_size=eval_forward_size,
                    eval_backward_size=eval_backward_size,
                    processors=processors,
                    topk=topk,
                    max_length=max_length,
                    accept_conf=accept_ratios,
                    skip_verify=skip_verify,
                )
                token_scores = greedy_processors(input_ids, logprobs.exp())  # (1 * T, V)
                probs = nn.functional.softmax(token_scores, dim=-1)  # (1 * T, V)
                tokens = torch.multinomial(probs, num_samples=1).squeeze(-1).view(batch_size, -1)
            else:
                stime = time.time()
                cur_input_ids, cur_position_ids, cur_attention_mask, cur_position_ids_to_predict, retrieve_indices, logprobs = \
                    prepare_candidates(
                        input_ids=input_ids,
                        logits=logits,
                        ref_position_ids_to_predict=ref_position_ids_to_predict,
                        pred_start_pos=pred_start_pos,
                        pred_end_pos=pred_end_pos,
                        forward_size=forward_size,
                        backward_size=backward_size,
                        eval_forward_size=eval_forward_size,
                        eval_backward_size=eval_backward_size,
                        processors=processors,
                        topk=topk,
                        max_length=max_length,
                        accept_conf=accept_ratios,
                    )
                if verbal:
                    print('[P1]', time.time() - stime)
                
                candidate_logits = self(
                    input_ids=cur_input_ids.unsqueeze(0),
                    attention_mask=cur_attention_mask.unsqueeze(0).unsqueeze(0),
                    position_ids=cur_position_ids.unsqueeze(0),
                    position_ids_to_predict=cur_position_ids_to_predict.unsqueeze(0),
                ).logits.view(cur_input_ids.size(-1), -1, logits.size(-1))  # (Lt, W, V)
                
                stime = time.time()
                token_losses, token_losses_forward, token_nt_losses, accept_flags, candidates = calculate_candidate_losses(
                    cur_input_ids=cur_input_ids,
                    cur_position_ids_to_predict=cur_position_ids_to_predict,
                    candidate_logits=candidate_logits,
                    retrieve_indices=retrieve_indices,
                    pred_start_pos=pred_start_pos,
                    pred_end_pos=pred_end_pos,
                    forward_size=eval_forward_size,
                    backward_size=eval_backward_size,
                )
                losses, losses_forward, nt_losses = token_losses.mean(-1), token_losses_forward.mean(-1), token_nt_losses.mean(-1)
                losses_gap = nt_losses - losses_forward
                losses_gap = losses_gap.masked_fill(losses_gap.lt(0), 0)
                losses, all_losses = losses + losses_gap, losses
                first_token_accept_flags = accept_flags[:, 0].ge(min(accept_flags[:, 0].max().item(), .5))
            
                # sorted_losses = losses.sort(dim=-1)
                # tmp_select_idx = first_token_accept_flags[sorted_losses.indices].nonzero().squeeze(-1)[0]
                # select_idx = sorted_losses.indices[tmp_select_idx]
                select_idx = losses.min(dim=-1).indices            
                if verbal:
                    print('[P2]', time.time() - stime)
                    
                tokens = candidates[select_idx].contiguous().view(batch_size, -1)   # (1, T)
                accept_ratios = accept_flags[select_idx]
                
            new_input_ids = torch.cat((input_ids[:, :pred_start_pos], tokens), dim=-1)
            # tokenizer.batch_decode(candidates[all_losses.sort(dim=-1).indices])
            # tokenizer.batch_decode(candidates[losses_gap.sort(dim=-1).indices])
            # tokenizer.batch_decode(candidates[nt_losses.sort(dim=-1).indices])
            # tokenizer.batch_decode(candidates[losses.sort(dim=-1).indices])
            
            # EOS
            eos_idx = new_input_ids[0].eq(tokenizer.eos_token_id).nonzero().squeeze(-1)
            if eos_idx.size(-1) > 0:
                iter_cnt_last += 1
            eos_idx = eos_idx[0] + 1 if eos_idx.size(-1) > 0 else max_length
            new_input_ids = new_input_ids[:, :eos_idx]
            tracks.append([
                new_input_ids[0, seq_length:].clone().tolist(),
                torch.arange(seq_length, new_input_ids.size(-1), dtype=torch.long).tolist(),
            ])
            if verbal:
                iter_cnt = len(tracks)
                print(f'[{iter_cnt}]', tokenizer.decode(new_input_ids[0].clone()))
                if not skip_verify:
                    print(accept_ratios.tolist())
            
            # convergence check
            min_seq_length = min(new_input_ids.size(-1), input_ids.size(-1))
            diff_idx = new_input_ids[0, :min_seq_length].ne(input_ids[0, :min_seq_length]).nonzero().squeeze(-1)
            diff_idx = diff_idx[0] if diff_idx.size(-1) > 0 else min_seq_length     # locate the difference positions
            tmp_occurance_counts = torch.ones_like(new_input_ids)
            tmp_occurance_counts[0, :diff_idx] = occurance_counts[0, :diff_idx] + 1
            occurance_counts = tmp_occurance_counts     # (B, L*) update the occurance times
            
            if not skip_verify:
                rand_var = torch.rand(accept_ratios.size(-1), device=accept_ratios.device)
                reject_indexes = accept_ratios.lt(rand_var).nonzero().squeeze(-1)
                accept_idx = reject_indexes[0] + fixed_seq_length if reject_indexes.size(-1) > 0 else occurance_counts.size(-1)
                accept_idx = (pred_start_pos + reject_indexes.min()).item() if reject_indexes.size(-1) > 0 else occurance_counts.size(-1)
            else:
                few_times_indexes = occurance_counts[0, fixed_seq_length:].le(occurance_threshold).nonzero().squeeze(-1)
                few_times_idx = few_times_indexes[0] + fixed_seq_length if few_times_indexes.size(-1) > 0 else occurance_counts.size(-1)
                few_times_idx = (pred_start_pos + few_times_indexes.min()).item() if few_times_indexes.size(-1) > 0 else occurance_counts.size(-1)
                accept_idx = few_times_idx
            
            # fixed_seq_length = min(few_times_idx, accept_idx, max(fixed_seq_length, new_input_ids.size(-1) - block_size + forward_size))
            # fixed_seq_length = min(accept_idx, max(fixed_seq_length, new_input_ids.size(-1) - block_size + forward_size))
            # fixed_seq_length = max(accept_idx, fixed_seq_length + 1)
            if tokens.size(-1) >= min(backward_size, block_size):
                fixed_seq_length = max(accept_idx, fixed_seq_length)
            start_idx, end_idx = fixed_seq_length, fixed_seq_length + block_size
            
            if start_idx <= prev_max_start_idx:
                iter_cnt_mid += 1
            else:
                iter_cnt_mid = 0
            if iter_cnt_mid > occurance_threshold:
                fixed_seq_length = start_idx = min(prev_max_start_idx + 1, new_input_ids.size(-1))
                end_idx = prev_max_start_idx + 1 + block_size
                iter_cnt_mid = 0
            prev_max_start_idx = max(prev_max_start_idx, start_idx)
            
            if (new_input_ids[0].eq(tokenizer.eos_token_id).any() and (accept_idx >= eos_idx)) or start_idx >= max_length or iter_cnt_last > forward_size or len(tracks) > max_iter_times:
                keep_generate = False
                input_ids = new_input_ids
                break
            # import ipdb; ipdb.set_trace()
            # update position_ids, position_ids_to_predict
            input_ids = new_input_ids
            if position_ids is not None:
                position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
            _position_ids_to_predict = torch.arange(forward_size + backward_size + 1, dtype=torch.long, device=input_ids.device)
            tmp_position_ids_to_predict = (_position_ids_to_predict - backward_size) + torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device).view(-1, 1)
            tmp_position_ids_to_predict = tmp_position_ids_to_predict.unsqueeze(0).expand(batch_size, input_ids.size(-1), forward_size + backward_size + 1).contiguous()
            tmp_position_ids_to_predict = tmp_position_ids_to_predict.masked_fill(tmp_position_ids_to_predict.lt(start_idx), 0)
            tmp_position_ids_to_predict = tmp_position_ids_to_predict.masked_fill(tmp_position_ids_to_predict.gt(end_idx), 0)
            
            position_ids_to_predict = tmp_position_ids_to_predict            
        
        return tracks, input_ids
    
    def oa_generate(
        self,
        input_ids: torch.LongTensor,    # (B, L)
        attention_mask: torch.BoolTensor | None = None,     # (B, L)
        position_ids: torch.LongTensor | None = None,       # (B, L)
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, Wf + Wb + 1)
        max_length: int = 512,
        max_iter_times: int = 512,
        tokenizer: PreTrainedTokenizerBase | None = None,
        temperature: float = 0.0,
        topk: int = 16,
        occurance_threshold: int = 8,
        block_size: int = 16,
        forward_size: int = 8,
        backward_size: int = 8,
        eval_forward_size: int = 4,
        eval_backward_size: int = 8,
        skip_verify: bool = False,
        left2right: bool = False,
        verbal: bool = False,
        use_cache: bool = False,
    ):
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        assert batch_size == 1, "Only support batch size 1 for now !!!"
        assert max_length > seq_length, "Input sequence length exceeds maximum length !!!"
        
        block_size = 1 if left2right else block_size
        pred_window_size = 1 if left2right else (forward_size + backward_size + 1)
        if position_ids_to_predict is None:
            position_ids_to_predict = torch.arange(pred_window_size, dtype=torch.long, device=input_ids.device)
            if left2right:  # (L, 1)
                position_ids_to_predict = (position_ids_to_predict + 1) + torch.arange(seq_length, dtype=torch.long, device=input_ids.device).view(-1, 1)
            else:           # (L, Wf + Wb + 1)
                position_ids_to_predict = (position_ids_to_predict - backward_size) + torch.arange(seq_length, dtype=torch.long, device=input_ids.device).view(-1, 1)
            position_ids_to_predict = position_ids_to_predict.unsqueeze(0).expand(batch_size, seq_length, pred_window_size).contiguous()
            for i in range(batch_size):
                start_idx = attention_mask[i].nonzero().max().item() + 1
                position_ids_to_predict[i] = (position_ids_to_predict[i] * position_ids_to_predict[i].ge(start_idx)).long()
            position_ids_to_predict = (position_ids_to_predict * position_ids_to_predict.le(seq_length + block_size)).long()
        
        if attention_mask is None:
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
        
        if left2right:            
            return self.next_token_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_ids_to_predict=position_ids_to_predict,
                temperature=temperature,
                tokenizer=tokenizer,
                max_length=max_length,
                verbal=verbal,
            )
        else:
            return self.multiple_token_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_ids_to_predict=position_ids_to_predict,
                block_size=block_size,
                forward_size=forward_size,
                backward_size=backward_size,
                eval_forward_size=eval_forward_size,
                eval_backward_size=eval_backward_size,
                occurance_threshold=occurance_threshold,
                topk=topk,
                tokenizer=tokenizer,
                max_length=max_length,
                max_iter_times=max_iter_times,
                verbal=verbal,
                skip_verify=skip_verify,
            )
