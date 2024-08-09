from __future__ import annotations

from typing import Any

import math

import torch
import torch.nn as nn
from transformers import MistralPreTrainedModel, PreTrainedTokenizerBase
from transformers.models.mistral.modeling_mistral import (
    MistralConfig, MistralAttention, MistralRMSNorm, MistralMLP,
    _CONFIG_FOR_DOC, MISTRAL_INPUTS_DOCSTRING, 
    rotate_half, apply_rotary_pos_emb, repeat_kv, _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask,
)
from transformers.generation.utils import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils.doc import add_start_docstrings_to_model_forward, replace_return_docstrings

from oa_dag.models.oa_model import OAModelMixin, OAModelOutput, BaseModelOutputWithPastOA
from oa_dag.utils import pad_tensors, locate_quantity, get_normal_dist
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
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, 2 * W + 1)
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
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1)
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
            # extract the corresponding query (at the next specific positions)
            new_query_states = raw_query_states.unsqueeze(-2).expand(bsz, raw_query_states.size(1), q_len, position_ids_to_predict.size(-1), raw_query_states.size(-1)).contiguous().view(bsz, raw_query_states.size(1), -1, raw_query_states.size(-1))
            new_query_states = apply_rotary_pos_emb_single(new_query_states, cos, sin, position_ids_to_predict.view(bsz, -1))
            if attention_mask is not None:
                new_attention_mask = torch.zeros(bsz, new_query_states.size(-2), key_states.size(-2), dtype=attention_mask.dtype, device=attention_mask.device)
                for i in range(bsz):
                    # update attention mask
                    new_attention_mask[i] = attention_mask[i][0][position_ids_to_predict[i].view(-1)]
                attention_mask = new_attention_mask.unsqueeze(1)
            query_states = new_query_states
        
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
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, 2 * W + 1)
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
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, 2 * W + 1)
        topk_probs: torch.FloatTensor | None = None,    # (B, M, K)
        topk_ids: torch.LongTensor | None = None,   # (B, M, K)
        replace_indexes: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,    # (B, L, 2 * W + 1)
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
                position_ids_to_predict,
            )
        else:
            layer_outputs = self.oa_layer(
                hidden_states,
                attention_mask=last_attention_mask,
                position_ids=position_ids,
                position_ids_to_predict=position_ids_to_predict,
            )        
        hidden_states = layer_outputs[0]
        
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            assert position_ids_to_predict is not None and position_ids_to_predict.size() == labels.size()
            # Flatten the tokens
            shift_logits = logits.contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels.view(-1)
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
        max_length: int = 512,
        tokenizer: PreTrainedTokenizerBase | None = None,
        min_n_tokens: int = 1,
        topk: int = 8,
        temperature: float = 0.2,
        token_prob_threshod_to_keep: float = 0.3,
        low_conf_threshod: float = 0.6,
        mean_low_conf_threshod: float = 0.9,
        n_add_tokens: int = 16,
        block_size: int = 8,
        window_size: int = 8,
        length_penalty: float = 1.25,
        n_adjust_tokens: int = 2,
        left2right: bool = False,
        verbal: bool = False,
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now !!!"
        
        processors = LogitsProcessorList()
        warpers = LogitsProcessorList()
        if temperature != 0.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        else:
            warpers.append(TopKLogitsWarper(top_k=1, min_tokens_to_keep=1))
        
        if positions_to_replace is None:
            positions_to_replace = torch.tensor([[input_ids.size(-1)]], dtype=torch.long, device=input_ids.device)
        if position_ids_to_predict is None:
            assert max_length > input_ids.size(-1), "Input sequence length exceeds maximum length !!!"
            if left2right:
                block_size = 1  # max_length - input_ids.size(-1)
            # if window_size > 0:
            #     block_size = window_size
            end_index = min(max_length, input_ids.size(-1) + block_size)
            position_ids_to_predict = torch.arange(
                input_ids.size(-1), end_index, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.size(), dtype=torch.bool)
            
        raw_input_ids = input_ids.clone().detach()
        topk_probs, topk_ids, replace_indexes, ar_results = None, None, None, None
        keep_generate, iter_cnt, no_grow = True, 0, 0
        tracks, raw_len = [], raw_input_ids.size(-1)
        
        if left2right:
            while keep_generate:
                if verbal:
                    print(f'[{iter_cnt}]', tokenizer.decode(input_ids[0]))
                iter_cnt += 1
                
                logits = self(
                    input_ids=input_ids,    # (B, L)
                    attention_mask=attention_mask,    # (B, L)
                    position_ids=position_ids,    # (B, L)
                    positions_to_replace=positions_to_replace,    # (B, 1)
                    position_ids_to_predict=position_ids_to_predict,    # (B, N)
                    topk_probs=topk_probs,    # (B, N, k)
                    topk_ids=topk_ids,    # (B, N, k)
                    replace_indexes=replace_indexes,    # (B, N)
                    return_dict=True,
                ).logits[:, raw_input_ids.size(-1):].contiguous()
                
                pred_start_idx = positions_to_replace - raw_input_ids.size(-1)
                
                # extract scores from logits
                token_scores = processors(input_ids, logits.view(-1, logits.size(-1)))
                token_scores = warpers(input_ids, token_scores)
                probs = nn.functional.softmax(token_scores, dim=-1).unsqueeze(0)
                
                # extract topk predictions at each position
                results = torch.topk(nn.functional.softmax(logits[:, pred_start_idx:], dim=-1), k=topk, dim=-1)
                rescale_results = torch.topk(probs[:, pred_start_idx:], k=topk, dim=-1)
                top_token_probs = results.values[0, :, 0]    # (N,)
                ratio_token_probs = results.values[0, :, 0] / results.values[0, :, 1]
                
                new_input_ids = pad_tensors(input_ids, max_len=(input_ids.size(-1) + 1), pad_value=tokenizer.pad_token_id)  # (B, L)
                new_input_ids[:, input_ids.size(-1):] = rescale_results.indices[:, :1, 0]
                attention_mask = pad_tensors(attention_mask, max_len=(input_ids.size(-1) + 1), pad_value=True).bool()  # (B, L)
                position_ids = torch.cat((position_ids, position_ids_to_predict[:, :1]), dim=-1)  # (B, L)
                tracks.append([
                    new_input_ids[0, raw_input_ids.size(-1):].clone().tolist(),
                    torch.arange(raw_input_ids.size(-1), new_input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0).tolist(),
                ])
                position_ids_to_predict = position_ids_to_predict + 1
                input_ids = new_input_ids
                positions_to_replace = torch.tensor([input_ids.size(-1)]).long().unsqueeze(-1)  # (B, 1)
                eos_index = input_ids[0].eq(tokenizer.eos_token_id).nonzero().squeeze(-1)
                if eos_index.size(-1) > 0 or input_ids.size(-1) >= max_length:
                    keep_generate = False
        else:
            while keep_generate:
                # forward pass to get tokens at position_ids_to_predict
                if verbal:
                    print(f'[{iter_cnt}]', tokenizer.decode(input_ids[0]))
                iter_cnt += 1
                if input_ids.size(-1) > raw_input_ids.size(-1):
                    use_sum = True
                    batch_size = input_ids.size(-1) - raw_input_ids.size(-1) + 1
                    positions_to_replace = torch.arange(raw_input_ids.size(-1), input_ids.size(-1) + 1, dtype=torch.long, device=input_ids.device).unsqueeze(-1)
                    split_position_ids_to_predict = torch.stack([
                        torch.arange(position_ids[0, x[0] - 1] - window_size, position_ids[0, x[0] - 1] + window_size + 1, dtype=torch.long, device=position_ids_to_predict.device)
                        for x in positions_to_replace
                    ], dim=0)
                    outputs = self(
                        input_ids=input_ids.repeat(batch_size, 1),    # (B, L)
                        attention_mask=attention_mask.repeat(batch_size, 1),    # (B, L)
                        position_ids=position_ids.repeat(batch_size, 1),    # (B, L)
                        positions_to_replace=positions_to_replace,    # (B, 1)
                        position_ids_to_predict=split_position_ids_to_predict,    # (B, W * 2 + 1)
                        topk_probs=topk_probs.repeat(batch_size, 1, 1) if topk_probs is not None else None,    # (B, N, k)
                        topk_ids=topk_ids.repeat(batch_size, 1, 1) if topk_ids is not None else None,    # (B, N, k)
                        replace_indexes=replace_indexes.repeat(batch_size, 1) if replace_indexes is not None else None,    # (B, N)
                        return_dict=True,
                    )
                else:
                    use_sum = False
                    outputs = self(
                        input_ids=input_ids,    # (B, L)
                        attention_mask=attention_mask,    # (B, L)
                        position_ids=position_ids,    # (B, L)
                        positions_to_replace=positions_to_replace,    # (B, 1)
                        position_ids_to_predict=position_ids_to_predict,    # (B, N)
                        topk_probs=topk_probs,    # (B, N, k)
                        topk_ids=topk_ids,    # (B, N, k)
                        replace_indexes=replace_indexes,    # (B, N)
                        return_dict=True,
                    )
                logits = outputs.logits[:, input_ids.size(-1):].contiguous()    # (B, N, V)
                
                if use_sum:
                    tmp_logits = torch.zeros(position_ids_to_predict.size(-1), logits.size(-1), dtype=logits.dtype, device=logits.device).unsqueeze(0)
                    tmp_cnt = torch.zeros(1, position_ids_to_predict.size(-1), dtype=torch.long, device=logits.device)
                    
                    i = -1
                    tmp_logits = torch.zeros(logits.size(0), position_ids_to_predict.size(-1), logits.size(-1), dtype=logits.dtype, device=logits.device)   # (block_size + 1, block_size, V)
                    for x, xlogits in zip(split_position_ids_to_predict, logits):
                        i += 1
                        x_idx = torch.logical_and(x.ge(position_ids_to_predict.min()), x.le(position_ids_to_predict.max())).nonzero().squeeze(-1)
                        tmp_logits[i, x[x_idx] - position_ids_to_predict.min()] = xlogits[x_idx]
                    tmp_logits = tmp_logits.transpose(0, 1)   # (block_size, block_size + 1, V)
                    
                    tmp_weights = torch.zeros(position_ids_to_predict.size(-1), logits.size(0), dtype=logits.dtype, device=logits.device)
                    weights = get_normal_dist(window_size=window_size).to(logits.device)
                    for i in range(tmp_weights.size(0)):
                        start_idx, end_idx = max(-1, i - window_size) + 1, min(logits.size(0) - 2, i + window_size) + 1
                        if start_idx > end_idx: continue
                        ws, we = window_size - (i + 1 - start_idx), window_size + end_idx - i - 1
                        tmp_weights[i, start_idx: end_idx + 1] = weights[ws: we + 1]
                    
                    ar_logits = logits[:position_ids_to_predict.size(-1), window_size + 1, :].unsqueeze(0)
                    if ar_logits.size(1) != tmp_logits.size(0):
                        ar_logits = torch.cat((ar_logits, torch.zeros(1, tmp_logits.size(0) - ar_logits.size(1), ar_logits.size(-1), dtype=ar_logits.dtype, device=ar_logits.device)), dim=1)
                    ar_results = torch.topk(nn.functional.softmax(ar_logits, dim=-1), k=topk, dim=-1)
                    
                    logits = ((tmp_logits * tmp_weights.unsqueeze(-1)).sum(1) / tmp_weights.sum(-1).unsqueeze(-1)).unsqueeze(0)
                
                # extract scores from logits
                token_scores = processors(input_ids, logits.view(-1, logits.size(-1)))
                token_scores = warpers(input_ids, token_scores)
                probs = nn.functional.softmax(token_scores, dim=-1).unsqueeze(0)
                
                # extract topk predictions at each position
                results = torch.topk(nn.functional.softmax(logits, dim=-1), k=topk, dim=-1)
                rescale_results = torch.topk(probs, k=topk, dim=-1)
                top_token_probs = results.values[0, :, 0]    # (N,)
                ratio_token_probs = results.values[0, :, 0] / results.values[0, :, 1]
                
                new_input_ids = pad_tensors(raw_input_ids, max_len=max(position_ids_to_predict.max().item(), position_ids.max().item()) + 1, pad_value=tokenizer.pad_token_id)  # (B, L)
                new_input_ids[0, position_ids_to_predict[0]] = results.indices[0, :, 0]
                new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.bool)
                new_position_ids = torch.arange(0, new_input_ids.size(-1), dtype=torch.long, device=position_ids.device).unsqueeze(0)
                new_positions_to_replace = torch.tensor([[new_input_ids.size(-1)]], dtype=torch.long, device=input_ids.device)
                
                eos_indexes = new_input_ids[0].eq(tokenizer.eos_token_id).nonzero().squeeze(-1)
                if eos_indexes.size(-1) > 0:
                    idx = eos_indexes[0].item()
                    new_input_ids, new_attention_mask, new_position_ids = new_input_ids[:, :idx+1], new_attention_mask[:, :idx+1], new_position_ids[:, :idx+1]
                    new_positions_to_replace = torch.tensor([[new_input_ids.size(-1)]], dtype=torch.long, device=input_ids.device)
                
                if ar_results is not None:
                    accu_probs = torch.tensor([top_token_probs[:x + 1].mean().item() for x in range(block_size)], device=input_ids.device)
                    # diff_start_idx = torch.logical_or(
                    #     results.indices[0, :, 0].ne(ar_results.indices[0, :, 0]), results.values[0, :, 0].lt(low_conf_threshod),
                    # ).nonzero().squeeze(-1)
                    diff_start_idx = torch.logical_or(
                        results.indices[0, :, 0].ne(ar_results.indices[0, :, 0]), accu_probs.lt(.8),
                    ).nonzero().squeeze(-1)
                    # diff_start_idx = accu_probs.lt(.8).nonzero().squeeze(-1)
                    if diff_start_idx.size(-1) > 0:
                        diff_start_idx = diff_start_idx[0]
                        if diff_start_idx <= 0:
                            tmp_diff_start_idx = results.indices[0, :, 0].ne(ar_results.indices[0, :, 0]).nonzero().squeeze(-1)
                            if tmp_diff_start_idx.size(-1) > 0 and tmp_diff_start_idx[0] > 0:
                                diff_start_idx = 1
                    else:
                        diff_start_idx = position_ids_to_predict.size(-1)
                    if diff_start_idx <= 0:
                        # if new_input_ids.size(-1) > raw_input_ids.size(-1) and max(ar_results.values[0, 0, 0], results.values[0, 0, 0]) > low_conf_threshod:
                        # # if new_input_ids.size(-1) > raw_input_ids.size(-1) and results.values[0, 0, 0] > .5:
                        #     diff_start_idx = 1
                        #     new_input_ids[0, raw_input_ids.size(-1)] = ar_results.indices[0, 0, 0] if ar_results.values[0, 0, 0] > results.values[0, 0, 0] else results.indices[0, 0, 0]
                        #     # new_input_ids[0, raw_input_ids.size(-1)] = results.indices[0, 0, 0]
                        # else:
                        #     diff_start_idx = 0
                        if new_input_ids.size(-1) > raw_input_ids.size(-1):
                            if max(ar_results.values[0, 0, 0], results.values[0, 0, 0]) > low_conf_threshod:
                                diff_start_idx = 1
                                new_input_ids[0, raw_input_ids.size(-1)] = ar_results.indices[0, 0, 0] if ar_results.values[0, 0, 0] > results.values[0, 0, 0] else results.indices[0, 0, 0]
                            # elif ar_results.indices[0, 0, 0] > low_conf_threshod - .1:
                            #     diff_start_idx = 1
                            #     new_input_ids[0, raw_input_ids.size(-1)] = ar_results.indices[0, 0, 0]
                            elif no_grow > 3:
                                diff_start_idx = 1
                                new_input_ids[0, raw_input_ids.size(-1)] = ar_results.indices[0, 0, 0]
                                no_grow = 0
                            else:
                                diff_start_idx = 0
                                no_grow += 1
                        else:
                            diff_start_idx = 0
                            no_grow += 1
                    else:
                        diff_start_idx = min(diff_start_idx, (window_size + 1) // 2)
                        # diff_start_idx = min(diff_start_idx, (window_size + 1) // 4)
                        # diff_start_idx = min(diff_start_idx, window_size)
                        no_grow = 0
                    if verbal:
                        print(tokenizer.decode(new_input_ids[0, :raw_input_ids.size(-1) + diff_start_idx]))
                    raw_input_ids = new_input_ids[:, :raw_input_ids.size(-1) + diff_start_idx]
                    position_ids_to_predict = position_ids_to_predict + diff_start_idx
                    # tokenizer.decode(ar_results.indices[0, :, 0]), tokenizer.decode(results.indices[0, :, 0])
                    # ar_results.values[0, :, 0], results.values[0, :, 0]
                
                # if iter_cnt >= block_size:
                #     import ipdb; ipdb.set_trace()
                if raw_input_ids.eq(tokenizer.eos_token_id).any() or iter_cnt > 300:
                    keep_generate = False
                
                input_ids, attention_mask, position_ids, positions_to_replace = new_input_ids, new_attention_mask, new_position_ids, new_positions_to_replace
                # replace_indexes, topk_probs, topk_ids = position_ids_to_predict, results.values, results.indices
                # print([top_token_probs[:x + 1].mean().item() for x in range(block_size)])
                
                tracks.append([
                    input_ids[0, raw_len:].clone().tolist(),
                    position_ids[0, raw_len:].clone().tolist(),
                    raw_input_ids[0, raw_len:].clone().tolist(),
                ])
        
        if verbal:
            print(tokenizer.decode(input_ids[0]))
        
        return tracks, input_ids
        
        raw_input_ids = input_ids.clone().detach()
        cur_raw_input_ids = input_ids.clone().detach()
        raw_position_ids = torch.arange(0, raw_input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
        cur_raw_position_ids = raw_position_ids.clone().detach()
        raw_attention_mask = attention_mask.clone().detach()
        cur_raw_attention_mask = attention_mask.clone().detach()
        tracks = []
        keep_generate = True
        topk_probs, topk_ids, replace_indexes, prev_indexes_to_keep, prev_topk_probs = None, None, None, None, None
        predicted_position = torch.zeros(position_ids_to_predict.shape, dtype=torch.bool, device=position_ids_to_predict.device)
        cnt, xcnt, unchanged, prev_len, prb, kept_n_tokens, sliding_cnt = 0, 0, 0, 0, 0, 0, 0
        all_probs, all_seq_probs = [], []
        while keep_generate:
            cnt += 1
            # forward pass to get tokens at position_ids_to_predict
            if verbal:
                print(tokenizer.decode(input_ids[0]), '\n', kept_n_tokens, prb)
            if input_ids.size(-1) <= prev_len:
                unchanged += 1
            else:
                unchanged = 0
            prev_len = input_ids.size(-1)
            logits = self(
                input_ids=input_ids,    # (B, L)
                attention_mask=attention_mask,    # (B, L)
                position_ids=position_ids,    # (B, L)
                positions_to_replace=positions_to_replace,    # (B, 1)
                position_ids_to_predict=position_ids_to_predict,    # (B, N)
                topk_probs=topk_probs,    # (B, N, k)
                topk_ids=topk_ids,    # (B, N, k)
                replace_indexes=replace_indexes,    # (B, N)
                return_dict=True,
            ).logits[:, raw_input_ids.size(-1):].contiguous()
            
            pred_start_idx = positions_to_replace - raw_input_ids.size(-1)
            
            # extract scores from logits
            token_scores = processors(input_ids, logits.view(-1, logits.size(-1)))
            token_scores = warpers(input_ids, token_scores)
            probs = nn.functional.softmax(token_scores, dim=-1).unsqueeze(0)
            
            # extract topk predictions at each position
            results = torch.topk(nn.functional.softmax(logits[:, pred_start_idx:], dim=-1), k=topk, dim=-1)
            rescale_results = torch.topk(probs[:, pred_start_idx:], k=topk, dim=-1)
            top_token_probs = results.values[0, :, 0]    # (N,)
            ratio_token_probs = results.values[0, :, 0] / results.values[0, :, 1]
            
            ar_probs, probs = probs[:, :pred_start_idx], probs[:, pred_start_idx:]
            if pred_start_idx > 0:
                ar_results = torch.topk(nn.functional.softmax(logits[:, :pred_start_idx], dim=-1), k=topk, dim=-1)
            
            if left2right:
                new_input_ids = pad_tensors(input_ids, max_len=(input_ids.size(-1) + 1), pad_value=tokenizer.pad_token_id)  # (B, L)
                new_input_ids[:, input_ids.size(-1):] = rescale_results.indices[:, :1, 0]
                attention_mask = pad_tensors(attention_mask, max_len=(input_ids.size(-1) + 1), pad_value=True).bool()  # (B, L)
                position_ids = torch.cat((position_ids, position_ids_to_predict[:, :1]), dim=-1)  # (B, L)
                tracks.append([
                    new_input_ids[0, raw_input_ids.size(-1):].clone().tolist(),
                    torch.arange(raw_input_ids.size(-1), new_input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0).tolist(),
                ])
                position_ids_to_predict = position_ids_to_predict[:, 1:]
                input_ids = new_input_ids
                positions_to_replace = torch.tensor([input_ids.size(-1)]).long().unsqueeze(-1)  # (B, 1)
                eos_index = input_ids[0].eq(tokenizer.eos_token_id).nonzero()
                eos_index = eos_index.min() if eos_index.size(0) > 0 else None
                if eos_index is not None or input_ids.size(-1) >= max_length:
                    break
            elif window_size > 0:
                if pred_start_idx > 0:
                    ar_top_token_probs = ar_results.values[0, :, 0]
                    ar_top_token_probs = torch.cat((ar_top_token_probs[-1:], ar_top_token_probs[:-1]), dim=-1)
                    ar_top_token_probs[0] = 0
                    replace_with_ar_indexes = ar_top_token_probs.gt(top_token_probs).nonzero().squeeze(-1)
                    top_token_probs[replace_with_ar_indexes] = ar_top_token_probs[replace_with_ar_indexes]
                    results.values[0, replace_with_ar_indexes, 0] = ar_results.values[0, replace_with_ar_indexes, 0]
                    results.indices[0, replace_with_ar_indexes, 0] = ar_results.indices[0, replace_with_ar_indexes, 0]
                
                conf_flags = top_token_probs.lt(.9).long()
                high_conf_indexes, low_conf_indexes = conf_flags.eq(0).nonzero().squeeze(-1), conf_flags.nonzero().squeeze(-1)
                if high_conf_indexes.size(-1) <= 0:
                    conf_flags = top_token_probs.lt(.5).long()
                    high_conf_indexes, low_conf_indexes = conf_flags.eq(0).nonzero().squeeze(-1)[:1], conf_flags.nonzero().squeeze(-1)
                    if high_conf_indexes.size(-1) > 0:
                        low_conf_indexes = top_token_probs.ne(top_token_probs[high_conf_indexes][0]).nonzero().squeeze(-1)
                kept_n_tokens = conf_flags.size(-1)
                
                new_input_ids = pad_tensors(cur_raw_input_ids, max_len=(cur_raw_input_ids.size(-1) + kept_n_tokens), pad_value=tokenizer.pad_token_id)  # (B, L)
                if cur_raw_position_ids.size(-1) > raw_position_ids.size(-1):
                    new_input_ids[0, cur_raw_position_ids[0, raw_position_ids.size(-1):]] = cur_raw_input_ids[0, raw_input_ids.size(-1):]
                new_input_ids[0, position_ids_to_predict[0]] = results.indices[0, :, 0]
                new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.bool)
                new_position_ids = torch.arange(0, new_input_ids.size(-1), dtype=torch.long, device=position_ids.device)
                new_positions_to_replace = torch.tensor([cur_raw_input_ids.size(-1) + kept_n_tokens]).long().unsqueeze(-1)
                if low_conf_indexes.size(-1) > 0:
                    replace_indexes = position_ids_to_predict[:, low_conf_indexes]
                    topk_probs = results.values[:, low_conf_indexes, :]
                    topk_ids = results.indices[:, low_conf_indexes, :]
                else:
                    replace_indexes, topk_probs, topk_ids = None, None, None
                # if high_conf_indexes.size(-1) > 0:
                #     cur_raw_input_ids = torch.cat((cur_raw_input_ids, results.indices[:, high_conf_indexes, 0]), dim=-1)
                #     cur_raw_position_ids = torch.cat((cur_raw_position_ids, position_ids_to_predict[:, high_conf_indexes]), dim=-1)
                #     position_ids_to_predict = torch.cat((
                #         position_ids_to_predict[:, low_conf_indexes],
                #         torch.arange(position_ids_to_predict[0][-1].item() + 1, position_ids_to_predict[0][-1].item() + 1 + high_conf_indexes.size(-1), dtype=torch.long, device=position_ids.device).unsqueeze(0),
                #     ), dim=-1)
                #     import ipdb; ipdb.set_trace()
                all_probs.append(top_token_probs)
                all_seq_probs.append(tokenizer.decode(results.indices[0, :, 0]))
                input_ids, attention_mask, position_ids, positions_to_replace = new_input_ids, new_attention_mask, new_position_ids, new_positions_to_replace
                window_size -= 1
                if window_size == 0:
                    import ipdb; ipdb.set_trace()
            elif window_size > 0:
                cur_indexes_to_predict = torch.arange(0, top_token_probs.size(-1), dtype=torch.long, device=input_ids.device)
                indexes, probs = [], []
                min_accumulated_n_tokens = max(1, min((input_ids.size(-1) - cur_raw_input_ids.size(-1)) + min_n_tokens, position_ids_to_predict.size(-1)))
                for ngram in range(min_accumulated_n_tokens, window_size + 1):
                    indexes.extend([x for x in torch.combinations(cur_indexes_to_predict, r=ngram)])
                    probs.append((torch.combinations(top_token_probs.log(), r=ngram).sum(dim=-1) / ngram ** length_penalty).exp())
                probs = torch.cat(probs, dim=-1)
                indexes_to_keep = indexes[torch.topk(probs, k=1, dim=-1).indices[0]]
                prb = probs[torch.topk(probs, k=1, dim=-1).indices[0]]
                kept_n_tokens = indexes_to_keep.size(-1)
                
                all_probs.append(prb)
                all_seq_probs.append(top_token_probs)
                
                new_input_ids = pad_tensors(cur_raw_input_ids, max_len=(cur_raw_input_ids.size(-1) + kept_n_tokens), pad_value=tokenizer.pad_token_id)  # (B, L)
                new_input_ids[0, cur_raw_input_ids.size(-1):] = results.indices[0, indexes_to_keep, 0]
                new_attention_mask = pad_tensors(cur_raw_attention_mask, max_len=(cur_raw_input_ids.size(-1) + kept_n_tokens), pad_value=True).bool()  # (B, L)
                new_position_ids = torch.cat((cur_raw_position_ids, position_ids_to_predict[:, indexes_to_keep]), dim=-1)  # (B, L)
                new_positions_to_replace = torch.tensor([cur_raw_input_ids.size(-1) + kept_n_tokens]).long().unsqueeze(-1)  # (B, 1)
                
                input_ids, attention_mask, position_ids, positions_to_replace = new_input_ids, new_attention_mask, new_position_ids, new_positions_to_replace
                # track the updates
                tracks.append([
                    input_ids[0, raw_input_ids.size(-1):].clone().tolist(),
                    position_ids_to_predict[0, indexes_to_keep].clone().tolist(),
                ])
                
                if tokenizer.eos_token_id in input_ids:
                    import ipdb; ipdb.set_trace()
                
                if kept_n_tokens >= window_size:
                    tmp_cnt = 1
                    position_ids_to_predict = position_ids_to_predict + (window_size - tmp_cnt)
                    cur_raw_input_ids = input_ids[:, :-tmp_cnt]
                    cur_raw_attention_mask = attention_mask[:, :-tmp_cnt]
                    cur_raw_position_ids = position_ids[:, :-tmp_cnt]
                    # position_ids_to_predict = position_ids_to_predict + window_size
                    # cur_raw_input_ids = input_ids
                    # cur_raw_attention_mask = attention_mask
                    # cur_raw_position_ids = position_ids
                    sliding_cnt += 1
            else:
                # keep tokens where prob >= threshold
                indexes_to_keep = torch.logical_or(
                    # predicted_position[0], top_token_probs.ge(token_prob_threshod_to_keep)
                    predicted_position[0], torch.logical_and(ratio_token_probs.ge(10), top_token_probs.ge(token_prob_threshod_to_keep)),
                ).nonzero().squeeze(-1)    # (N,)
                # indexes_to_keep = top_token_probs.ge(token_prob_threshod_to_keep).nonzero().squeeze(-1)    # (N,)
                # indexes_to_keep = torch.logical_and(ratio_token_probs.ge(5), top_token_probs.ge(token_prob_threshod_to_keep)).nonzero().squeeze(-1)    # (N,)
                max_idx = indexes_to_keep.max() + 1 if indexes_to_keep.size(-1) > 0 else 1
                indexes_to_keep = torch.arange(0, max_idx, dtype=torch.long, device=input_ids.device)
                kept_n_tokens = indexes_to_keep.size(-1)
                
                # select predicted tokens to keep
                min_accumulated_n_tokens = max(1, min((input_ids.size(-1) - raw_input_ids.size(-1)) + min_n_tokens, position_ids_to_predict.size(-1)))
                if unchanged > 3:
                    min_accumulated_n_tokens = input_ids.size(-1) - raw_input_ids.size(-1) + 1
                if kept_n_tokens < min_accumulated_n_tokens:
                    _, sorted_indexes = torch.sort(top_token_probs, dim=-1, descending=True)    # (N,)
                    sorted_indexes_to_keep = sorted_indexes[:min_accumulated_n_tokens].sort().values    # (n,)
                    indexes_to_keep_to_add = torch.tensor([x for x in sorted_indexes_to_keep if x not in indexes_to_keep], dtype=torch.long, device=sorted_indexes_to_keep.device)
                    indexes_to_keep = torch.cat((indexes_to_keep, indexes_to_keep_to_add[:sorted_indexes_to_keep.size(-1) - indexes_to_keep.size(-1)]), dim=-1).sort().values
                    kept_n_tokens = indexes_to_keep.size(-1)
                new_topk_probs, new_topk_ids = results.values[:, indexes_to_keep, :], results.indices[:, indexes_to_keep, :]    # (B, N, k)
                new_replace_indexes = position_ids_to_predict[:, :kept_n_tokens]  # (B, N)
                new_is_quantity = locate_quantity(new_topk_ids[0], tokenizer)
                
                # update inputs for next iteration
                new_input_ids = pad_tensors(raw_input_ids, max_len=(raw_input_ids.size(-1) + kept_n_tokens), pad_value=tokenizer.pad_token_id)  # (B, L)
                new_input_ids[0, raw_input_ids.size(-1):] = new_topk_ids[0, :, 0]
                new_attention_mask = pad_tensors(raw_attention_mask, max_len=(raw_input_ids.size(-1) + kept_n_tokens), pad_value=True).bool()  # (B, L)
                new_position_ids = torch.cat((raw_position_ids, position_ids_to_predict[:, indexes_to_keep]), dim=-1)  # (B, L)
                new_positions_to_replace = torch.tensor([raw_input_ids.size(-1) + kept_n_tokens]).long().unsqueeze(-1)  # (B, 1)
                
                # if unchanged > 3 and new_input_ids.size(-1) <= input_ids.size(-1):
                #     import ipdb; ipdb.set_trace()
                
                if prev_topk_probs is not None: # and prev_topk_probs[0, :, 0].mean() > new_topk_probs[0, :, 0].mean():
                    remain_ids = predicted_position[0][new_position_ids[0][raw_input_ids.size(-1):] - raw_input_ids.size(-1)].nonzero().squeeze(-1)
                    if remain_ids.size(-1) <= prev_topk_probs.size(1):
                        # if new_topk_probs[0, remain_ids, 0].mean() < prev_topk_probs[0, :remain_ids.size(-1), 0].mean():
                        if new_topk_ids[0, remain_ids, 0].ne(prev_topk_ids[0, :remain_ids.size(-1), 0]).any():
                            selected_remain = (new_topk_probs[0, remain_ids, 0] < prev_topk_probs[0, :remain_ids.size(-1), 0]).nonzero().squeeze(-1)
                            # new_is_quantity[remain_ids].eq(False).nonzero().squeeze(-1)
                            if selected_remain.size(-1):
                                new_input_ids[0, remain_ids + raw_input_ids.size(-1)][selected_remain] = input_ids[0, raw_input_ids.size(-1):][selected_remain]
                                new_topk_probs[0, remain_ids, :][selected_remain, ...] = prev_topk_probs[0, :remain_ids.size(-1), :][selected_remain, ...]
                                new_topk_ids[0, remain_ids, :][selected_remain, ...] = prev_topk_ids[0, :remain_ids.size(-1), :][selected_remain, ...]
                
                input_ids, attention_mask, position_ids, positions_to_replace = new_input_ids, new_attention_mask, new_position_ids, new_positions_to_replace
                predicted_position[0][position_ids[0][raw_input_ids.size(-1):] - raw_input_ids.size(-1)] = True
                topk_probs, topk_ids, replace_indexes, is_quantity = new_topk_probs, new_topk_ids, new_replace_indexes, new_is_quantity
                
                prev_topk_probs, prev_topk_ids = topk_probs.clone(), topk_ids.clone()
                selected_replace = topk_probs[0, :, 0].lt(mean_low_conf_threshod).nonzero().squeeze(-1)
                topk_probs, topk_ids, replace_indexes = topk_probs[:, selected_replace, :], topk_ids[:, selected_replace, :], replace_indexes[:, selected_replace]
                
                # track the updates
                tracks.append([
                    input_ids[0, raw_input_ids.size(-1):].clone().tolist(),
                    position_ids_to_predict[0, indexes_to_keep].clone().tolist(),
                ])
                prev_indexes_to_keep = indexes_to_keep
                
                # check whether reached eos
                eos_index = input_ids[0].eq(tokenizer.eos_token_id).nonzero()
                eos_index = eos_index.min() if eos_index.size(0) > 0 else None
                if eos_index is not None:
                    eos_position_id = position_ids[0, eos_index]
                    # check low-confidence predictions
                    valid_token_probs = top_token_probs[:eos_position_id - raw_input_ids.size(-1) + 1]
                    low_conf_indexes = valid_token_probs.lt(low_conf_threshod).nonzero().squeeze(-1)
                    if low_conf_indexes.size(0) <= 0 or valid_token_probs.mean() > mean_low_conf_threshod:
                        if xcnt > 3: break
                        xcnt += 1
                    # regenerate on the low-confidence positions
                    position_ids_to_predict = position_ids_to_predict[:, position_ids_to_predict[0].le(eos_position_id + n_adjust_tokens).nonzero().squeeze(-1)]
                    predicted_position = predicted_position[:, :position_ids_to_predict.size(-1)]
                    shortened_indexes = position_ids[0].le(eos_position_id + n_adjust_tokens).nonzero().squeeze(-1)
                    if replace_indexes is not None:
                        shortened_indexes_for_replace = position_ids[0][replace_indexes[0]].le(eos_position_id + n_adjust_tokens).nonzero().squeeze(-1)
                        topk_probs, topk_ids, replace_indexes = topk_probs[0][shortened_indexes_for_replace, ...].unsqueeze(0), topk_ids[0][shortened_indexes_for_replace, ...].unsqueeze(0), replace_indexes[0][shortened_indexes_for_replace].unsqueeze(0)
                    input_ids, attention_mask, position_ids = input_ids[0][shortened_indexes].unsqueeze(0), attention_mask[0][shortened_indexes].unsqueeze(0), position_ids[0][shortened_indexes].unsqueeze(0)
                    positions_to_replace = torch.tensor([input_ids.size(-1)]).long().unsqueeze(-1)
                    # if low_conf_indexes.size(0) < 3:
                    #     import ipdb; ipdb.set_trace()
                elif top_token_probs.ge(low_conf_threshod).all() or top_token_probs.mean() > mean_low_conf_threshod or input_ids.size(-1) == raw_input_ids.size(-1) + position_ids_to_predict.size(-1):
                    tmp = torch.arange(position_ids_to_predict.max() + 1, min(position_ids_to_predict.max() + n_add_tokens + 1, max_length), dtype=torch.long, device=input_ids.device).unsqueeze(0)
                    position_ids_to_predict = torch.cat((position_ids_to_predict, tmp), dim=-1)
                    predicted_position = torch.zeros(position_ids_to_predict.shape, dtype=torch.bool, device=position_ids_to_predict.device)
                    predicted_position[0][position_ids[0][raw_input_ids.size(-1):] - raw_input_ids.size(-1)] = True
                
                if cnt > position_ids_to_predict.size(-1):
                    # import ipdb; ipdb.set_trace()
                    break
        
        if verbal:
            print(tokenizer.decode(input_ids[0]))
        
        return tracks, input_ids
