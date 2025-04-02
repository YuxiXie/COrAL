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

from coral.models.oa_model import OAModelMixin, OAModelOutput, BaseModelOutputWithPastOA
from coral.utils import (
    prepare_candidates, calculate_candidate_losses,
)
from coral.configs.constants import IGNORE_INDEX

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
        seq_len = (max(position_ids.max().item(), position_ids_to_predict.max().item()) if position_ids_to_predict is not None else position_ids.max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        
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

        if position_ids_to_predict is not None:
            # extract the corresponding query (at the next specific positions)
            new_query_states = query_states.unsqueeze(-2).expand(bsz, query_states.size(1), q_len, position_ids_to_predict.size(-1), query_states.size(-1)).contiguous().view(bsz, query_states.size(1), -1, query_states.size(-1))
            # apply position encoding
            query_states = apply_rotary_pos_emb_single(new_query_states, cos, sin, position_ids_to_predict.view(bsz, -1))
            q_len = query_states.size(-2)
            # update attention weights
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                # update attention mask
                attention_mask = attention_mask.unsqueeze(-2).expand(bsz, 1, kv_seq_len, position_ids_to_predict.size(-1), kv_seq_len).contiguous().view(bsz, 1, -1, kv_seq_len)
                attn_weights = attn_weights + attention_mask
        else:
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
            try:
                query_states = apply_rotary_pos_emb_single(new_query_states, cos, sin, position_ids_to_predict.view(bsz, -1))
            except:
                import ipdb; ipdb.set_trace()
            
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
        freeze_backbone: bool = False,
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
        if freeze_backbone:
            hidden_states = hidden_states.detach()
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

def truncate_cache(cache, max_length):
    if cache is None:
        return None
    
    if isinstance(cache, DynamicCache):
        prev_length = cache.get_seq_length()
        
        if max_length < prev_length:
            cache._seen_tokens = max_length
            for layer_idx, key_cache in enumerate(cache.key_cache):
                cache.key_cache[layer_idx] = key_cache[:, :, :max_length]
            for layer_idx, value_cache in enumerate(cache.value_cache):
                cache.value_cache[layer_idx] = value_cache[:, :, :max_length]
        
        return cache
    
    raise NotImplementedError("Truncating cache of type {} is not supported".format(type(cache)))

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
        freeze_backbone: bool = False,
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
            freeze_backbone=freeze_backbone,
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
        use_cache: bool = True,
    ):
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        
        processors = LogitsProcessorList()
        if temperature != 0.0:
            processors.append(TemperatureLogitsWarper(temperature))
        else:
            processors.append(TopKLogitsWarper(top_k=1, min_tokens_to_keep=1))
            
        # Prepare inputs
        inputs = {
            'input_ids': input_ids,    # (1, L)
            'attention_mask': torch.ones_like(input_ids, dtype=torch.bool),
            'position_ids': position_ids,    # (1, L)
            'position_ids_to_predict': position_ids_to_predict,    # (1, L, N)
            'use_cache': use_cache,
        }
        if position_ids is None:
            position_ids = inputs['attention_mask'].long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            inputs['position_ids'] = position_ids
        
        # Papare the cache
        cache_name = "past_key_values"
        if use_cache:
            inputs[cache_name] = DynamicCache()
        cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        
        # Generation
        keep_generate, prev_end_idx, tracks = True, 0, []
        while keep_generate:
            # slice model inputs if it's an input that should have the same length as `input_ids`
            if inputs[cache_name] is not None:
                if inputs['input_ids'].shape[1] != cache_position.shape[0]:
                    inputs['input_ids'] = inputs['input_ids'][:, cache_position]
                    
                    inputs['position_ids'] = inputs['position_ids'][:, cache_position]
                    inputs['position_ids_to_predict'] = inputs['position_ids_to_predict'][:, cache_position]
                    
                    if inputs['attention_mask'].dim() == 4:
                        inputs['attention_mask'] = inputs['attention_mask'][:, :, cache_position, :]
            
            # forward
            outputs: OAModelOutput = self(**inputs, return_dict=True)
            
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            logits = outputs.logits.clone().contiguous().view(batch_size, -1, position_ids_to_predict.size(-1), outputs.logits.size(-1))
            logits = logits.contiguous()[:, -1, 0, :]    # (1, L * N, V) --> (1, L, V) (N = 1) --> (1, V)
            
            # extract scores from logits
            token_scores = processors(input_ids, logits.view(-1, logits.size(-1)))  # (1, V)
            if temperature != 0:
                probs = nn.functional.softmax(token_scores, dim=-1)  # (1, V)
                # sample and get candidate tokens
                tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)   # (1, 1)
            else:
                tokens = torch.argmax(token_scores, dim=-1)    # (1, 1)
            
            input_ids = torch.cat((input_ids, tokens[:, None]), dim=-1)
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
            inputs.update({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids if position_ids is not None else attention_mask.long().cumsum(-1) - 1,
                'position_ids_to_predict': position_ids_to_predict,
            })
            
            cache = self._extract_past_from_model_output(outputs)
            inputs[cache_name] = cache
            # update cache position
            if use_cache:
                cache_position = cache_position[-1:] + 1
            else:
                past_positions = cache_position
                new_positions = torch.arange(
                    past_positions[-1] + 1, past_positions[-1] + 1 + 1, dtype=past_positions.dtype, device=past_positions.device,
                )
                cache_position = torch.cat((past_positions, new_positions))
            
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
        
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
        force_repeat: bool = True,
        epsilon: float = 0.1,
        use_cache: bool = True,
    ):
        import time
        
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        backward_size = max(-1, backward_size)
        
        # Prepare logits processor
        processors = LogitsProcessorList()
        # processors.append(TopPLogitsWarper(top_p=topp, min_tokens_to_keep=1))
        # processors.append(EtaLogitsWarper(epsilon=1-topp))
        greedy_processors = LogitsProcessorList()
        greedy_processors.append(TopKLogitsWarper(top_k=1, min_tokens_to_keep=1))
        
        # Prepare inputs
        inputs = {
            'input_ids': input_ids,    # (1, L)
            'attention_mask': torch.ones_like(input_ids, dtype=torch.bool),
            'position_ids': position_ids,    # (1, L)
            'position_ids_to_predict': position_ids_to_predict,    # (1, L, N)
            'use_cache': use_cache,
        }
        
        # Papare the cache
        cache_name = "past_key_values"
        if use_cache:
            inputs[cache_name] = DynamicCache()
        cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        
        # Generation
        occurance_counts = torch.ones((0,), dtype=torch.int16)  # record the occurance times of predicted tokens
        fixed_seq_length = prev_fixed_seq_length = seq_length
        
        block_start_idx, block_end_idx = seq_length, min(seq_length + block_size, max_length) - 1
        keep_generate, tracks = True, []
        
        iter_cnt_last = 0   # used when eos occurs
        accept_ratios = None
        
        last_token_logits = None
        while keep_generate:
            # slice model inputs if it's an input that should have the same length as `input_ids`
            if inputs[cache_name] is not None:
                if inputs['input_ids'].shape[1] != cache_position.shape[0]:
                    inputs['input_ids'] = inputs['input_ids'][:, cache_position]
                    
                    inputs['position_ids'] = inputs['position_ids'][:, cache_position]
                    inputs['position_ids_to_predict'] = inputs['position_ids_to_predict'][:, cache_position]
                    
                    if inputs['attention_mask'].dim() == 4:
                        inputs['attention_mask'] = inputs['attention_mask'][:, :, cache_position, :]
            
            # forward
            outputs: OAModelOutput = self(**inputs, return_dict=True)
            
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            logits = outputs.logits.clone().contiguous().view(batch_size, inputs['input_ids'].size(-1), position_ids_to_predict.size(-1), -1)
            if last_token_logits is not None:
                logits = torch.cat((last_token_logits, logits), dim=1)
            else:
                logits = logits[:, fixed_seq_length - inputs['position_ids'][0, 0] - 1:, ...]     # (1, L * N, V) --> (1, L, N, V) --> (1, L', N, V)
            
            # Sampling with multiple dependencies
            ref_position_ids_to_predict = position_ids_to_predict[:, -logits.size(1):]
            pred_start_pos, pred_end_pos = fixed_seq_length, ref_position_ids_to_predict.max().item()
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
                    topk=topk,
                    max_length=max_length,
                    accept_conf=accept_ratios,
                    skip_verify=skip_verify,
                )   # (1, T, V)
                token_scores = greedy_processors(input_ids, logprobs.view(-1, logprobs.size(-1)))  # (1 * T, V)
                tokens = torch.argmax(token_scores, dim=-1).squeeze(-1).view(batch_size, -1)    # (1 * T, 1) --> (1, T)
                
                token_probs = torch.gather(  # size = (1, T, 1)
                    logprobs, dim=-1, index=tokens.unsqueeze(dim=-1),
                ).squeeze(dim=-1)[0].exp()  # size = (1, T)
            else:
                stime = time.time()
                if accept_ratios is not None and accept_ratios.size(-1) > logits.size(1):
                    accept_ratios = accept_ratios[-logits.size(1):]
                
                cur_input_ids, cur_position_ids, cur_attention_mask, cur_position_ids_to_predict, retrieve_indices, _ = \
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
                        topk=topk,
                        max_length=max_length,
                        accept_conf=accept_ratios,
                        max_new_tokens=int(block_size * 16),
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
                token_losses, token_losses_forward, all_losses, token_nt_losses, accept_flags, candidates = calculate_candidate_losses(
                    cur_input_ids=cur_input_ids,
                    cur_position_ids_to_predict=cur_position_ids_to_predict,
                    candidate_logits=candidate_logits,
                    retrieve_indices=retrieve_indices,
                    pred_start_pos=pred_start_pos,
                    pred_end_pos=pred_end_pos,
                    forward_size=eval_forward_size,
                    backward_size=eval_backward_size,
                    epsilon=epsilon,
                )
                
                token_losses_gap = token_nt_losses - token_losses_forward
                token_losses_gap = token_losses_gap.masked_fill(token_losses_gap.lt(0), 0)
                token_losses = all_losses + token_losses_gap
                # token_losses = all_losses
                losses = token_losses.mean(-1)
                
                select_idx = losses.min(dim=-1).indices            
                if verbal:
                    print('[P2]', time.time() - stime)
                    
                tokens = candidates[select_idx].contiguous().view(batch_size, -1)   # (1, T)
                accept_ratios = accept_flags[select_idx]
                token_probs = (-token_losses[select_idx]).exp()
                nt_token_probs = (-token_nt_losses[select_idx]).exp()
            
            # update generated ids, model inputs, and length for next step
            new_input_ids = torch.cat((input_ids[:, :pred_start_pos], tokens), dim=-1)
            
            # EOS
            eos_indices = tokens[0].eq(tokenizer.eos_token_id).nonzero().squeeze(-1) + input_ids[:, :pred_start_pos].size(-1)
            if eos_indices.size(-1) > 0: iter_cnt_last += 1
            eos_idx = eos_indices[0].item() + 1 if eos_indices.size(-1) > 0 else max_length
            
            # Update traces
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
            
            new_occurance_counts = torch.ones_like(new_input_ids[0, seq_length:])
            if force_repeat:
                diff_indices = new_input_ids[0, seq_length:min_seq_length].ne(input_ids[0, seq_length:min_seq_length]).nonzero().squeeze(-1)
                diff_idx = diff_indices[0].item() if diff_indices.size(-1) > 0 else min_seq_length - seq_length     # locate the difference positions
                new_occurance_counts[:diff_idx] = occurance_counts[:diff_idx] + 1
            else:
                min_length = min(occurance_counts.size(-1), new_occurance_counts.size(-1))
                new_occurance_counts[:min_length] = occurance_counts[:min_length] + 1
            occurance_counts = new_occurance_counts     # (L*,) update the occurance times
            
            if not skip_verify or occurance_threshold == 0:
                rand_var = torch.rand(token_probs.size(-1), device=token_probs.device)
                if skip_verify:
                    reject_indices = rand_var.gt(token_probs).nonzero().squeeze(-1)
                else:
                    reject_indices = torch.logical_and(rand_var.gt(token_probs), rand_var.gt(nt_token_probs)).nonzero().squeeze(-1)
                accept_idx = reject_indices[0].item() if reject_indices.size(-1) > 0 else rand_var.size(-1)
                accept_pos = pred_start_pos + accept_idx
            few_times_indices = occurance_counts[pred_start_pos - seq_length:].le(occurance_threshold).nonzero().squeeze(-1)
            few_times_idx = few_times_indices[0].item() if few_times_indices.size(-1) > 0 else occurance_counts[pred_start_pos - seq_length:].size(-1)
            if not skip_verify or occurance_threshold == 0:
                accept_pos = max(pred_start_pos + few_times_idx, accept_pos)
            else:
                accept_pos = pred_start_pos + few_times_idx
            fixed_seq_length = max(accept_pos, fixed_seq_length)
            
            block_start_idx, block_end_idx = fixed_seq_length, min(fixed_seq_length + block_size, max_length) - 1
            if accept_pos >= eos_idx or block_start_idx >= max_length or iter_cnt_last > block_size or len(tracks) > max_iter_times:
                keep_generate = False
                input_ids = new_input_ids
                break
            
            # update position_ids, position_ids_to_predict
            input_ids = new_input_ids
            
            if position_ids is not None:
                position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
            _position_ids_to_predict = torch.arange(forward_size + backward_size + 1, dtype=torch.long, device=input_ids.device)
            tmp_position_ids_to_predict = (_position_ids_to_predict - backward_size).unsqueeze(0) + torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(1)
            tmp_position_ids_to_predict = tmp_position_ids_to_predict.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            tmp_position_ids_to_predict = tmp_position_ids_to_predict.masked_fill(torch.logical_or(
                tmp_position_ids_to_predict.lt(block_start_idx),
                tmp_position_ids_to_predict.gt(block_end_idx)
            ), 0)
            tmp_position_ids_to_predict = tmp_position_ids_to_predict.masked_fill(tmp_position_ids_to_predict.gt(block_end_idx), 0)
            position_ids_to_predict = tmp_position_ids_to_predict
            
            inputs['input_ids'] = input_ids
            inputs['position_ids'], inputs['position_ids_to_predict'] = position_ids, position_ids_to_predict
            inputs['attention_mask'] = torch.ones_like(input_ids, dtype=torch.bool)
            inputs['position_ids_to_predict'] = position_ids_to_predict
            
            # update past_key_values keeping its naming used in model code
            cache = self._extract_past_from_model_output(outputs)
            if min(fixed_seq_length, cache.get_seq_length()) >= input_ids.size(-1):
                fixed_seq_length = input_ids.size(-1) - 1
            if fixed_seq_length < cache.get_seq_length():
                inputs[cache_name] = truncate_cache(cache, fixed_seq_length)
            # update cache position
            if use_cache:
                cache_position = position_ids[0, inputs[cache_name].get_seq_length():]
            else:
                cache_position = position_ids[0]
            
            if 0 <= fixed_seq_length - prev_fixed_seq_length < logits.size(1):
                last_token_logits = logits[:, fixed_seq_length - prev_fixed_seq_length].unsqueeze(1)
            else:
                last_token_logits = None
            prev_fixed_seq_length = fixed_seq_length
            
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
            
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
        force_repeat: bool = True,
        left2right: bool = False,
        verbal: bool = False,
        use_cache: bool = False,
        epsilon: float = 0.1,
        add_denoising: bool = False,
    ):
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        assert batch_size == 1, "Only support batch size 1 for now !!!"
        assert max_length > seq_length, "Input sequence length exceeds maximum length !!!"
        
        if attention_mask is None:
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        
        raw_block_size = block_size
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
        
        if left2right:            
            tracks, input_ids = self.next_token_generate(
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
            tracks, input_ids = self.multiple_token_generate(
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
                force_repeat=force_repeat,
                epsilon=epsilon,
            )

        return tracks, input_ids
