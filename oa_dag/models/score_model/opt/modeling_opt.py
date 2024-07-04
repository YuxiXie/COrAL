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

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import OPTModel, OPTPreTrainedModel, PretrainedConfig, PreTrainedModel
from transformers.models.opt.modeling_opt import _CONFIG_FOR_DOC, OPT_INPUTS_DOCSTRING
from transformers.utils.doc import add_start_docstrings_to_model_forward, replace_return_docstrings

from oa_dag.models.score_model import ScoreModelMixin, ScoreModelOutput


class OPTForScore(ScoreModelMixin, OPTPreTrainedModel):
    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.model = OPTModel(config)

        config.architectures = [self.__class__.__name__]
        self.init_score_head(config, hidden_size=config.word_embed_proj_dim, **kwargs)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self) -> None:
        return None

    def set_decoder(self, decoder: PreTrainedModel) -> None:
        self.model = decoder

    def get_decoder(self) -> PreTrainedModel:
        return self.model

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ScoreModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(  # pylint: disable=too-many-arguments
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | ScoreModelOutput:
        """
        Args:

        Returns:

        Examples:

        ```python
        >>> from oa_dag.models.score_model.llama.modeling_llama import LlamaForScore
        >>> from transformers import LlamaTokenizer

        >>> model = LlamaForScore.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        # get score
        >>> outputs = model(**inputs)
        >>> end_scores = outputs.end_scores
        >>> end_scores
        tensor([[0.0000]])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model.decoder(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state  # size = (B, L, E)
        return self.get_scores(
            last_hidden_state,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
