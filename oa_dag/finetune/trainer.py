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
"""Trainer class for supervised finetuning."""

from __future__ import annotations

from typing import Any
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.distributed as dist

from oa_dag.datasets import SupervisedDataset, PromptOnlyDataset
from oa_dag.trainers import SupervisedTrainer
from oa_dag.utils import get_all_reduce_mean, is_main_process, to_device


class SupervisedFinetuneTrainer(SupervisedTrainer):
    """Trainer class for supervised finetuning."""

    TRAINING_TYPE = 'sft'
    DATASET_TYPE = SupervisedDataset
    EVAL_DATA_TYPE = PromptOnlyDataset
    MODEL_TYPE = AutoModelForCausalLM
    
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return {}

        self.set_eval()
        prompts: list[str] = []
        generateds: list[str] = []

        eval_dataloader = tqdm(
            self.eval_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
        )

        cnt = 0
        for batch in eval_dataloader:
            cnt += 1
            batch = to_device(batch, self.args.device)
            with torch.no_grad():
                seq = self.model.module.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=self.args.max_length,
                    synced_gpus=True,
                    do_sample=False,
                )

            dist.barrier()

            prompt = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            generated = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
            generated = [text[len(prompt[i]) :] for i, text in enumerate(generated)]
            prompts.extend(prompt)
            generateds.extend(generated)
            # import ipdb; ipdb.set_trace()
            if cnt > 100: break
        dist.barrier()
        
        import ipdb; ipdb.set_trace()
    
    def loss(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            'loss': outputs.loss,
        }

    def train_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, Any]:
        """Performs a single training step.

        Args:
            input_ids (torch.LongTensor): input ids for causal inputs to complete with.
            labels (torch.LongTensor): labels for the full sequence.
            attention_mask (torch.BoolTensor): attention mask for the labels.

        Returns:
            dict[str, Any]: training loss, learning rate
        """
        loss = self.loss(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )['loss']
        self.model.backward(loss)
        self.model.step()

        loss = get_all_reduce_mean(loss)

        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }
