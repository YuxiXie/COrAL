"""Trainer class for supervised finetuning."""

from __future__ import annotations

import os
from typing import Any
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.distributed as dist

from coral.models.oa_model import AutoModelForOA
from coral.datasets import SupervisedDataset, PromptOnlyDataset
from coral.trainers import SupervisedTrainer
from coral.utils import get_all_reduce_mean, is_main_process, to_device, json_dump


class SupervisedFinetuneTrainer(SupervisedTrainer):
    """Trainer class for supervised finetuning."""

    TRAINING_TYPE = 'sft'
    DATASET_TYPE = SupervisedDataset
    EVAL_DATA_TYPE = PromptOnlyDataset
    MODEL_TYPE = AutoModelForOA     # AutoModelForCausalLM
    
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return {}
        
        output_dir = '/'.join(self.args.model_name_or_path.split('/')[:-1])
        os.makedirs(output_dir, exist_ok=True)

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
        json_dump({'prompts': prompts, 'outputs': generateds,}, f'{output_dir}/decoding_tracks.json')
        import ipdb; ipdb.set_trace()
    
    def loss(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        position_ids_to_predict: torch.LongTensor,  # size = (B, L, N)
    ) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids_to_predict=position_ids_to_predict,
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
        position_ids_to_predict = torch.arange(1, dtype=torch.long, device=self.args.device)
        position_ids_to_predict = (position_ids_to_predict + 1) + torch.arange(input_ids.size(-1) - 1, dtype=torch.long, device=self.args.device).view(-1, 1)
        position_ids_to_predict = position_ids_to_predict.unsqueeze(0).expand(input_ids.size(0), input_ids.size(-1) - 1, 1).contiguous()

        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()
        labels = labels[:, 1:].unsqueeze(-1).contiguous()
        
        loss = self.loss(
            input_ids=input_ids,
            position_ids_to_predict=position_ids_to_predict,
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
