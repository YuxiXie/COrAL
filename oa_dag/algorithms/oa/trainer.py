"""Trainer class for supervised finetuning."""

from __future__ import annotations

import os
import random
from argparse import Namespace
from typing import Any
from tqdm import tqdm
import deepspeed
from collections import defaultdict

import torch
from torch.nn import Module, CrossEntropyLoss
import torch.nn.functional as F
import torch.distributed as dist

from oa_dag.configs.constants import IGNORE_INDEX
from oa_dag.datasets import SupervisedDataset, PromptOnlyDataset
from oa_dag.trainers import SupervisedTrainer
from oa_dag.utils import (
    shuffle_and_mask, add_noise,
    get_variable_generator, pad_tensors, 
    get_all_reduce_mean, get_all_reduce_min,
    is_main_process, to_device, 
    decode_masked_text, corrupt_input, replace_with_zero_one,
    json_dump,
)

from oa_dag.models.oa_model import AutoModelForOA, OAModelOutput


class OASupervisedFinetuneTrainer(SupervisedTrainer):
    """Trainer class for supervised finetuning."""

    TRAINING_TYPE = 'oa-sft'
    DATASET_TYPE = SupervisedDataset
    EVAL_DATA_TYPE = SupervisedDataset  # PromptOnlyDataset
    MODEL_TYPE = AutoModelForOA
    
    def __init__(self, args: Namespace, ds_config: dict[str, Any], ds_eval_config: dict[str, Any]) -> None:
        """Initialize trainer."""
        self.ds_eval_config = ds_eval_config
        super().__init__(args, ds_config)
        
        self.context_window = self.args.context_window
        self.replace_ratio_generator = get_variable_generator(self.args.replace_ratio_mu, self.args.replace_ratio_std, self.args.replace_ratio_min, self.args.replace_ratio_max)
    
    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        super().init_models()
        # Freeze the base model
        if self.args.tune_final_layer_only:
            # self.model.base_model.requires_grad_(False)
            for name, param in self.model.base_model.named_parameters():
                if self.model.additional_layer or f'layers.{self.model.config.num_hidden_layers - 1}' not in name:
                    param.requires_grad = False
            self.model.oa_layer.requires_grad_(True)
            if not self.model.additional_layer:
                self.model.base_model.layers[-1].requires_grad_(True)
        self.model.lm_head.requires_grad_(self.args.tune_lm_head)        
    
    @property
    def extra_model_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments for initializing the model."""
        return {
            'additional_layer': self.args.additional_layer,
        }
    
    def init_engines(self) -> None:
        if not self.args.need_eval:
            super().init_engines()
        else:
            self.model = self._init_eval_engine(
            model=self.model,
            ds_config=self.ds_eval_config,
        )
    
    def _init_eval_engine(
        self,
        model: Module,
        ds_config: dict[str, Any],
    ) -> deepspeed.DeepSpeedEngine:
        engine, *_ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
        return engine
        
    @torch.no_grad()
    def eval_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, Any]:
        if self.args.do_decoding:
            self.args.eval_mask_ratio = 1.0
            self.args.reconstruct = False
        
        batch = self.masking(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                             fixed_mask_threshold=self.args.eval_mask_ratio, 
                             fixed_replace_threshold=self.args.eval_replace_ratio,
                             is_training=False)
        
        if self.args.do_decoding:
            tracks = self.model.module.oa_generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                position_ids=batch['position_ids'],
                # positions_to_replace=batch['positions_to_replace'],
                # position_ids_to_predict=batch['position_ids_to_predict'],
                max_length=self.args.max_length,
                tokenizer=self.tokenizer,
                verbal=self.args.verbal_decoding,
                left2right=self.args.left2right,
            )
            dist.barrier()            
            return tracks
        else:        
            outputs = self.loss(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                position_ids=batch['position_ids'],
                positions_to_replace=batch['positions_to_replace'],
                position_ids_to_predict=batch['position_ids_to_predict'],
                topk_probs=batch['topk_probs'],
                topk_ids=batch['topk_ids'],
                replace_indexes=batch['replace_indexes'],
                use_cache=False,
                return_logits=True,
            )
            dist.barrier()
            
            logits = outputs['logits'][:, batch['input_ids'].size(-1):].contiguous()
            gt_labels = batch['labels'][:, batch['input_ids'].size(-1):].contiguous()
            
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, logits.size(-1)), gt_labels.view(-1)).view(logits.size(0), -1)
            
            loss_dict = defaultdict(float)
            count_dict = defaultdict(float)
            for bid in range(loss.size(0)):
                start_idx = labels[bid].ne(IGNORE_INDEX).nonzero().min().item()
                for j in range(batch['position_ids_to_predict'][bid].size(-1)):
                    pid = batch['position_ids_to_predict'][bid][j].item()
                    if pid <= 0: continue
                    pid = pid - start_idx
                    if gt_labels[bid][j].ne(IGNORE_INDEX):
                        loss_dict[pid] += loss[bid][j].item()
                        count_dict[pid] += 1
            
            return {i: (torch.tensor(loss_dict[i], device=self.args.device), 
                        torch.tensor(count_dict[i], device=self.args.device)) for i in loss_dict}
        
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return {}

        output_dir = '/'.join(self.args.model_name_or_path.split('/')[:-1])
        os.makedirs(output_dir, exist_ok=True)
        # if os.path.exists(f'{output_dir}/losses.json'):
        #     assert False, '''only for evaluation'''

        self.set_eval()
        texts: list[str] = []
        tracks: list[list] = []
        predictions: list[str] = []
        losses: dict[int, list[torch.Tensor]] = defaultdict(list[torch.Tensor])
        counts: dict[int, list[torch.Tensor]] = defaultdict(list[torch.Tensor])

        eval_dataloader = tqdm(
            self.eval_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
        )
        cnt = 0
        rst_fname = 'decoding-oa-r1-w8-b8'
        for batch in eval_dataloader:
            if batch['input_ids'].size(-1) > 1000: continue
            cnt += 1
            batch = to_device(batch, self.args.device)
            if self.args.do_decoding:
                track, sequences = self.eval_step(**batch)
                tracks.append(track)
                sequences = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                predictions.extend(sequences)
                text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                texts.extend(text)
                # import ipdb; ipdb.set_trace()
                # if cnt >= 500: break
                if cnt % 10 == 0:
                    json_dump({'prompts': texts, 'outputs': predictions, 'tracks': tracks}, f'{output_dir}/{rst_fname}.json')
            else:
                loss = self.eval_step(**batch)
                for pid, values in loss.items():
                    losses[pid].append(values[0])
                    counts[pid].append(values[1])

        if self.args.do_decoding:
            # if self.args.do_sample:
            #     json_dump({'prompts': texts, 'outputs': predictions, 'tracks': tracks}, f'{output_dir}/decoding_tracks_step_sample{self.args.max_n_tokens_per_step}.json')
            # else:
            #     json_dump({'prompts': texts, 'outputs': predictions, 'tracks': tracks}, f'{output_dir}/decoding_tracks_step{self.args.max_n_tokens_per_step}.json')
            json_dump({'prompts': texts, 'outputs': predictions, 'tracks': tracks}, f'{output_dir}/{rst_fname}.json')
        else:
            # Gather results from all processes
            max_key = torch.tensor(max(losses.keys()), device=self.args.device)
            # dist.reduce(max_key, dst=0, op=dist.ReduceOp.MAX)
            dist.barrier()
            
            for key in range(max_key + 1):
                try:
                    losses[key] = torch.stack(losses[key], dim=0).sum()
                    counts[key] = torch.stack(counts[key], dim=0).sum()
                except:
                    losses[key] = torch.tensor(0, device=self.args.device)
                    counts[key] = torch.tensor(0, device=self.args.device)
                # dist.reduce(losses[key], dst=0, op=dist.ReduceOp.SUM)
                # dist.reduce(counts[key], dst=0, op=dist.ReduceOp.SUM)
                if counts[key].sum().item() > 0:
                    losses[key] = losses[key].sum().item() / counts[key].sum().item()
                else:
                    losses[key] = -1
            dist.barrier()
            
            # if is_main_process():
            losses = dict(sorted(losses.items(), key=lambda x: x[0]))
            if self.args.eval_mask_ratio < 1 or self.args.eval_replace_ratio < 1:
                json_dump(losses, f'{output_dir}/losses_ratio_msk{self.args.eval_mask_ratio}_rpl{self.args.eval_replace_ratio}.json')
            else:
                json_dump(losses, f'{output_dir}/losses.json')
        
        dist.barrier()
        assert False, '''only for evaluation'''
        
        self.set_train()
        
        return losses
    
    def loss(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        position_ids: torch.LongTensor,
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, N)
        topk_probs: torch.FloatTensor | None = None,    # (B, M, K)
        topk_ids: torch.LongTensor | None = None,   # (B, M, K)
        replace_indexes: torch.LongTensor | None = None,
        return_logits: bool = False,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs: OAModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            position_ids_to_predict=position_ids_to_predict,
            topk_probs=topk_probs,
            topk_ids=topk_ids,
            replace_indexes=replace_indexes,
            use_cache=use_cache,
        )
        if return_logits:
            return {
                'loss': outputs.loss,
                'logits': outputs.logits.detach(),
            }
        return {
            'loss': outputs.loss,
        }
    
    @torch.no_grad()
    def collect_noisy_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        replace_position_ids: list[torch.LongTensor],
        topk: int = -1,
    ) -> torch.FloatTensor:
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        
        position_ids_to_predict = torch.zeros(batch_size, seq_length, self.context_window * 2 + 1, dtype=torch.long, device=self.args.device)
        for i in range(batch_size):
            for j in range(seq_length):
                position_ids_to_predict[i][j] = torch.arange(j - self.context_window, j + self.context_window + 1, dtype=torch.long, device=self.args.device)
                position_ids_to_predict[i][j] = (position_ids_to_predict[i][j] * position_ids_to_predict[i][j].ne(j)).long()
            nonzero_idx = attention_mask[i].nonzero().squeeze(-1)
            position_ids_to_predict[i] = (position_ids_to_predict[i].gt(nonzero_idx[0]).float() * position_ids_to_predict[i].le(nonzero_idx[-1]).float() * position_ids_to_predict[i]).long()
        
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids_to_predict=position_ids_to_predict,
            ).logits.contiguous().view(batch_size, seq_length, position_ids_to_predict.size(-1), -1)
        
        tmp_logits = torch.zeros(batch_size, seq_length, logits.size(-1), dtype=logits.dtype, device=self.args.device)
        tmp_cnt = torch.zeros(batch_size, seq_length, dtype=logits.dtype, device=self.args.device)
        for i in range(batch_size):
            for j in range(seq_length):
                if position_ids_to_predict[i][j].sum() <= 0: continue
                cur_positions = position_ids_to_predict[i][j].gt(0).nonzero().squeeze(-1)
                tmp_logits[i, position_ids_to_predict[i, j, cur_positions]] = tmp_logits[i, position_ids_to_predict[i, j, cur_positions]] + logits[i, j, cur_positions]
                tmp_cnt[i, position_ids_to_predict[i, j, cur_positions]] = tmp_cnt[i, position_ids_to_predict[i, j, cur_positions]] + 1
        tmp_cnt = (tmp_cnt.eq(0).float() + tmp_cnt).unsqueeze(-1)
        
        logits_to_predict, i = [], -1
        for _replace_position_ids in pad_tensors(replace_position_ids, pad_value=0):
            i += 1
            logits_to_predict.append(tmp_logits[i, _replace_position_ids] / tmp_cnt[i, _replace_position_ids])
        logits = torch.stack(logits_to_predict, dim=0)
        
        topk = topk if topk > 0 else self.tokenizer.vocab_size
        results = torch.topk(F.softmax(logits, dim=-1), k=topk, dim=-1)
        return results.values, results.indices
    
    @torch.no_grad()
    def corruption(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        topk: int = 8,
        fixed_replace_threshold: float = -1,
    ) -> dict[str, Any]:
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        raw_position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.args.device)
        
        replace_threshold_list, replace_ratio_list, replace_position_ids, replace_indexes = [], [], [], []
        
        new_input_ids = input_ids.clone()
        position_ids_to_predict = torch.zeros(batch_size, seq_length, self.context_window * 2 + 1, 
                                              dtype=torch.long, device=self.args.device)
        padded_labels = torch.zeros(batch_size, seq_length, self.context_window * 2 + 1, 
                                    dtype=torch.long, device=self.args.device)
        ##=== adding noise ===##
        for i in range(batch_size):
            ##=== extract label positions ===##
            is_label = labels[i].ne(IGNORE_INDEX)
            label_position_ids = is_label.nonzero().squeeze(-1)
            ##=== initialize input ids with position info ===##
            cur_input_ids = input_ids[i].clone()
            cur_position_ids = raw_position_ids.clone()
            cur_labels = labels[i].clone()
            ##=== replace input ids ===##
            replace_ids, replace_threshold = add_noise(cur_input_ids, cur_labels, self.replace_ratio_generator, 
                                                       fixed_replace_threshold=fixed_replace_threshold,
                                                       device=self.args.device)
            replace_threshold_list.append(torch.tensor(replace_threshold, device=self.args.device))
            replace_ratio_list.append(torch.tensor(replace_ids.size(-1) / max(1, cur_labels.ne(IGNORE_INDEX).sum()), device=self.args.device))
            replace_indexes.append(replace_ids)
            replace_position_ids.append(cur_position_ids[replace_ids])
            
            if replace_ids.size(-1) > 0:
                cur_input_ids = corrupt_input(replace_ids, cur_input_ids, cur_position_ids, cur_labels, 
                                              self.tokenizer, device=self.args.device)
            new_input_ids[i] = cur_input_ids
            ##=== wrap for prediction ===##
            for j in range(seq_length):
                position_ids_to_predict[i][j] = torch.arange(j - self.context_window, j + self.context_window + 1, 
                                                             dtype=torch.long, device=self.args.device)
            position_ids_to_predict[i] = (position_ids_to_predict[i].ge(label_position_ids[0]).float() * position_ids_to_predict[i].le(label_position_ids[-1]).float() * position_ids_to_predict[i]).long()
            for j in range(seq_length):
                padded_labels[i][j] = cur_labels[position_ids_to_predict[i][j]]
            
        replace_threshold = torch.stack(replace_threshold_list, dim=0).mean()
        replace_ratio = torch.stack(replace_ratio_list, dim=0).mean()
        replace_count_min = torch.stack([torch.tensor(x.size(0), dtype=torch.long, device=self.args.device) for x in replace_indexes], dim=0).min()
        replace_count_min = get_all_reduce_min(replace_count_min)
        dist.barrier()
        
        ##=== add noise by weighted embeddings ===##
        topk_probs, topk_ids = None, None
        if replace_count_min > 0:
            topk_probs, topk_ids = self.collect_noisy_inputs(
                input_ids,
                attention_mask,
                replace_position_ids,
                topk=topk,
            )
            replace_indexes = pad_tensors(replace_indexes, pad_value=-1)
            topk_probs = replace_with_zero_one(topk_probs)
        
        return {
            'input_ids': new_input_ids,     # (B, L)
            'labels': padded_labels,        # (B, L, 2 * W + 1)
            'attention_mask': attention_mask,      # (B, L)
            'position_ids': raw_position_ids.unsqueeze(0).repeat(batch_size, 1),     # (B, L)
            'position_ids_to_predict': position_ids_to_predict,        # (B, L, 2 * W + 1)
            'topk_probs': topk_probs,
            'topk_ids': topk_ids,
            'replace_indexes': replace_indexes if topk_ids is not None else None,
            'replace_threshold': replace_threshold,
            'replace_ratio': replace_ratio,
        }
    
    def create_oa_batch(self, batch: SupervisedDataset) -> dict[str, Any]:
        return self.corruption(**batch)
    
    def train_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        position_ids: torch.LongTensor,
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, 2 * W + 1)
        topk_probs: torch.FloatTensor | None = None,    # (B, M, K)
        topk_ids: torch.LongTensor | None = None,   # (B, M, K)
        replace_indexes: torch.LongTensor | None = None,
        replace_threshold: torch.Tensor | None = None,
        replace_ratio: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Performs a single training step.

        Args:
            input_ids (torch.LongTensor): input ids for causal inputs to complete with.
            labels (torch.LongTensor): labels for the full sequence.
            attention_mask (torch.BoolTensor): attention mask for the labels.

        Returns:
            dict[str, Any]: training loss, learning rate
        """
        outputs = self.loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            position_ids_to_predict=position_ids_to_predict,
            topk_probs=topk_probs,
            topk_ids=topk_ids,
            replace_indexes=replace_indexes,
            use_cache=False,
            return_logits=True,
        )
        
        loss = outputs['loss']
        
        self.model.backward(loss)
        self.model.step()
        
        ############################## sanity check ##############################
        logits = outputs['logits'].detach()
        
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits, shift_labels).view(input_ids.size(0), -1, labels.size(-1))
        
        idxes = position_ids_to_predict - position_ids.unsqueeze(-1)
        loss_dn = torch.logical_and(idxes.eq(0), position_ids_to_predict.gt(0))
        loss_ar = torch.logical_and(idxes.eq(1), position_ids_to_predict.gt(0))
        loss_close = torch.logical_and(idxes.abs().le(4), position_ids_to_predict.gt(0))
        loss_dn = (loss_dn * losses).sum() / max(1, loss_dn.sum())
        loss_ar = (loss_ar * losses).sum() / max(1, loss_ar.sum())
        loss_close = (loss_close * losses).sum() / max(1, loss_close.sum())
        ##########################################################################
        
        loss = get_all_reduce_mean(loss)
        replace_threshold = get_all_reduce_mean(replace_threshold)
        replace_ratio = get_all_reduce_mean(replace_ratio)
        loss_dn = get_all_reduce_mean(loss_dn)
        loss_ar = get_all_reduce_mean(loss_ar)
        loss_close = get_all_reduce_mean(loss_close)
        
        dist.barrier()
        
        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
            'train/replace_threshold': replace_threshold.item(),
            'train/replace_ratio': replace_ratio.item(),
            'train/loss_ar': loss_ar.item(),
            'train/loss_close': loss_close.item(),
            'train/loss_dn': loss_dn.item(),
        }
