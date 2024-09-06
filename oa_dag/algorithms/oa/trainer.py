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
from transformers.generation.utils import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper

from oa_dag.configs.constants import IGNORE_INDEX
from oa_dag.datasets import SupervisedDataset, PromptOnlyDataset
from oa_dag.trainers import SupervisedTrainer
from oa_dag.utils import (
    shuffle_and_mask, add_noise,
    get_variable_generator, pad_tensors, 
    get_all_reduce_mean, get_all_reduce_min,
    is_main_process, to_device, 
    decode_masked_text, corrupt_input, replace_with_zero_one,
    json_dump, json_load,
)

from oa_dag.models.oa_model import AutoModelForOA, OAModelOutput


class OASupervisedFinetuneTrainer(SupervisedTrainer):
    """Trainer class for supervised finetuning."""

    TRAINING_TYPE = 'oa-sft'
    DATASET_TYPE = SupervisedDataset
    EVAL_DATA_TYPE = SupervisedDataset
    MODEL_TYPE = AutoModelForOA
    
    def __init__(self, args: Namespace, ds_config: dict[str, Any], ds_eval_config: dict[str, Any]) -> None:
        """Initialize trainer."""
        if args.do_decoding:
            self.EVAL_DATA_TYPE = PromptOnlyDataset
        self.ds_eval_config = ds_eval_config
        super().__init__(args, ds_config)
        
        self.forward_context_window = self.args.context_window
        self.backward_context_window = int(self.forward_context_window * self.args.n_back_pred)
        self.replace_ratio_generator = get_variable_generator(self.args.replace_ratio_mu, self.args.replace_ratio_std, self.args.replace_ratio_min, self.args.replace_ratio_max)
        
        self.replace_sampler = LogitsProcessorList()
        self.replace_sampler.append(TopKLogitsWarper(top_k=16))
        self.replace_sampler.append(TemperatureLogitsWarper(temperature=4.0))
    
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
                self.model.base_model.oa_layer.requires_grad_(True)
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
        attention_mask: torch.BoolTensor,  # size = (B, L)
        labels: torch.LongTensor | None = None,  # size = (B, L)
    ) -> dict[str, Any]:
        if self.args.do_decoding:
            self.args.eval_mask_ratio = 1.0
            self.args.reconstruct = False
        
        if self.args.do_decoding:
            tracks = self.model.module.oa_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.args.max_length,
                tokenizer=self.tokenizer,
                temperature=self.args.temperature,
                # seq_temperature=self.args.seq_temperature,
                block_size=self.args.decoding_block_size,
                forward_size=self.args.context_window,
                backward_size=int(self.args.context_window * self.args.n_back_pred),
                occurance_threshold=self.args.decoding_occurance_threshold,
                verbal=self.args.verbal_decoding,
                left2right=self.args.left2right,
                # add_denoising=self.args.add_denoising,
            )
            dist.barrier()
            return tracks
        else:
            batch = self.corruption(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                    fixed_replace_threshold=self.args.eval_replace_ratio, force_replace=True)
            outputs = self.loss(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                position_ids=batch['position_ids'],
                position_ids_to_predict=batch['position_ids_to_predict'],
                topk_probs=batch['topk_probs'],
                topk_ids=batch['topk_ids'],
                replace_indexes=batch['replace_indexes'],
                use_cache=False,
                return_logits=True,
            )
            dist.barrier()
            
            logits = outputs['logits'].detach()     # (B, L * (Wf + Wb + 1), V)
            shift_logits = logits.contiguous().view(-1, logits.size(-1))
            shift_labels = batch['labels'].view(-1)
            loss_fct = CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits, shift_labels).view(logits.size(0), -1, batch['labels'].size(-1))    # (B, L, Wf + Wb + 1)
            
            relative_positions = batch['position_ids_to_predict'] - batch['position_ids'].unsqueeze(-1)    # (B, L, Wf + Wb + 1)
            small_pos, large_pos = relative_positions.min().item(), relative_positions.max().item()
            loss_dict, count_dict = defaultdict(float), defaultdict(float)
            # losses_mask = torch.zeros_like(losses, dtype=torch.bool)
            # for bid in range(batch['replace_indexes'].size(0)):
            #     replace_indexes = batch['replace_indexes'][bid][batch['replace_indexes'][bid].ge(0).nonzero().squeeze(-1)]
            #     for i in range(losses_mask[bid].size(0)):
            #         if batch['position_ids_to_predict'][bid][i].max() <= 0: continue
            #         for j in range(losses_mask[bid][i].size(0)):
            #             if batch['position_ids_to_predict'][bid][i][j] in replace_indexes:
            #                 losses_mask[bid][i][j] = True
            losses_mask = torch.ones_like(losses, dtype=torch.bool)
            for pid in range(small_pos, large_pos + 1):
                loss_dict[pid] += (relative_positions.eq(pid) * batch['position_ids_to_predict'].gt(0) * losses_mask * losses).sum().item()
                count_dict[pid] += (relative_positions.eq(pid) * batch['position_ids_to_predict'].gt(0) * losses_mask).sum().item()
            
            return {i: (torch.tensor(loss_dict[i], device=self.args.device), 
                        torch.tensor(count_dict[i], device=self.args.device)) for i in loss_dict if i >= (- self.args.context_window * self.args.n_back_pred)}
    
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return {}

        output_dir = '/'.join(self.args.model_name_or_path.split('/')[:-1])
        os.makedirs(output_dir, exist_ok=True)

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
        if not self.args.left2right and self.args.do_decoding:
            if self.args.result_fname.startswith('oa'):
                self.args.result_fname = f'{self.args.result_fname}_tp{self.args.temperature}_stp{self.args.seq_temperature}_f{self.args.context_window}b{int(self.args.context_window * self.args.n_back_pred)}c{self.args.decoding_block_size}_t{self.args.decoding_occurance_threshold}'
            else:
                self.args.result_fname = f'oa_tp{self.args.temperature}_stp{self.args.seq_temperature}_f{self.args.context_window}b{int(self.args.context_window * self.args.n_back_pred)}c{self.args.decoding_block_size}_t{self.args.decoding_occurance_threshold}'
        elif self.args.do_decoding:
            self.args.result_fname += f'_tp{self.args.temperature}'
        if os.path.exists(f'{output_dir}/{self.args.result_fname}.json'):
            existed = json_load(f'{output_dir}/{self.args.result_fname}.json')
            texts, predictions, tracks = existed['prompts'], existed['outputs'], existed['tracks']
        for batch in eval_dataloader:
            if batch['input_ids'].size(-1) >= self.args.max_length: continue
            cnt += 1
            if len(texts) >= cnt: continue
            batch = to_device(batch, self.args.device)
            if self.args.do_decoding:
                track, sequences = self.eval_step(**batch)
                tracks.append(track)
                sequences = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                predictions.extend(sequences)
                text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                texts.extend(text)
                if cnt % 10 == 0:
                    json_dump({'prompts': texts, 'outputs': predictions, 'tracks': tracks}, f'{output_dir}/{self.args.result_fname}.json')
            else:
                loss = self.eval_step(**batch)
                for pid, values in loss.items():
                    losses[pid].append(values[0])
                    counts[pid].append(values[1])

        if self.args.do_decoding:
            json_dump({'prompts': texts, 'outputs': predictions, 'tracks': tracks}, f'{output_dir}/{self.args.result_fname}.json')
            import ipdb; ipdb.set_trace()
        else:
            # Gather results from all processes
            min_key = torch.tensor(min(losses.keys()), device=self.args.device)
            max_key = torch.tensor(max(losses.keys()), device=self.args.device)
            # dist.reduce(max_key, dst=0, op=dist.ReduceOp.MAX)
            dist.barrier()
            
            for key in range(min_key, max_key + 1):
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
            json_dump(losses, f'{output_dir}/losses_{self.args.result_fname}_rpl{self.args.eval_replace_ratio}.json')
        
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
        topk: int = 16,
        pred_gap: int = 0,
    ) -> torch.FloatTensor:
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        
        pred_gap = min(pred_gap, self.forward_context_window - 1)
        position_ids_to_predict = torch.arange(self.forward_context_window - pred_gap, dtype=torch.long, device=self.args.device)
        position_ids_to_predict = (position_ids_to_predict + pred_gap + 1) + torch.arange(seq_length, dtype=torch.long, device=self.args.device).view(-1, 1)
        position_ids_to_predict = position_ids_to_predict.unsqueeze(0).expand(batch_size, seq_length, self.forward_context_window - pred_gap).contiguous()
        for i in range(batch_size):
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
        
        if self.args.sample_to_replace:
            probs = self.replace_sampler(input_ids, logits.view(-1, logits.size(-1)))
            probs = F.softmax(probs, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1).view(logits.size(0), -1)
            return None, token_ids
        
        topk = topk if topk > 0 else self.tokenizer.vocab_size
        results = torch.topk(F.softmax(logits, dim=-1), k=topk, dim=-1)
        return results.values, results.indices
    
    @torch.no_grad()
    def corruption(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        force_replace: bool = False,
        fixed_replace_threshold: float = -1,
        is_training: bool = True,
    ) -> dict[str, Any]:
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        raw_position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.args.device)
        
        replace_threshold_list, replace_ratio_list, replace_position_ids, replace_indexes = [], [], [], []
        force_replace = force_replace and fixed_replace_threshold < 0
        
        new_input_ids = input_ids.clone()
        if force_replace:
            position_ids_to_predict = torch.arange(self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
            position_ids_to_predict = (position_ids_to_predict - self.backward_context_window) + torch.arange(seq_length, dtype=torch.long, device=self.args.device).view(-1, 1)
            position_ids_to_predict = position_ids_to_predict.unsqueeze(0).expand(batch_size, seq_length, self.backward_context_window + 1).contiguous()
            padded_labels = torch.zeros(batch_size, seq_length, self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
        else:
            # position_ids_to_predict = torch.arange(self.forward_context_window + self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
            # position_ids_to_predict = (position_ids_to_predict - self.backward_context_window) + torch.arange(seq_length, dtype=torch.long, device=self.args.device).view(-1, 1)
            # position_ids_to_predict = position_ids_to_predict.unsqueeze(0).expand(batch_size, seq_length, self.forward_context_window + self.backward_context_window + 1).contiguous()
            # padded_labels = torch.zeros(batch_size, seq_length, self.forward_context_window + self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
            position_ids_to_predict = torch.arange(self.forward_context_window, dtype=torch.long, device=self.args.device)
            position_ids_to_predict = (position_ids_to_predict + 1) + torch.arange(seq_length, dtype=torch.long, device=self.args.device).view(-1, 1)
            position_ids_to_predict = position_ids_to_predict.unsqueeze(0).expand(batch_size, seq_length, self.forward_context_window).contiguous()
            padded_labels = torch.zeros(batch_size, seq_length, self.forward_context_window, dtype=torch.long, device=self.args.device)
        # position_ids_to_predict = torch.arange(self.forward_context_window + self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
        # position_ids_to_predict = (position_ids_to_predict - self.backward_context_window) + torch.arange(seq_length, dtype=torch.long, device=self.args.device).view(-1, 1)
        # position_ids_to_predict = position_ids_to_predict.unsqueeze(0).expand(batch_size, seq_length, self.forward_context_window + self.backward_context_window + 1).contiguous()
        # padded_labels = torch.zeros(batch_size, seq_length, self.forward_context_window + self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
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
                                                       force_replace=force_replace, fixed_replace_threshold=fixed_replace_threshold,
                                                       device=self.args.device)
            replace_threshold_list.append(torch.tensor(replace_threshold, device=self.args.device).float())
            replace_ratio_list.append(torch.tensor(replace_ids.size(-1) / max(1, cur_labels.ne(IGNORE_INDEX).sum()), device=self.args.device).float())
            replace_indexes.append(replace_ids)
            replace_position_ids.append(cur_position_ids[replace_ids])
            
            if replace_ids.size(-1) > 0:
                cur_input_ids = corrupt_input(replace_ids, cur_input_ids, cur_position_ids, cur_labels, 
                                              self.tokenizer, device=self.args.device)
            new_input_ids[i] = cur_input_ids
            ##=== wrap for prediction ===##
            position_ids_to_predict[i] = (position_ids_to_predict[i].ge(label_position_ids[0]).float() * position_ids_to_predict[i].le(label_position_ids[-1]).float() * position_ids_to_predict[i]).long()
            if force_replace and replace_ids.size(-1) > 0:
                labels_mask = torch.zeros_like(cur_labels, dtype=torch.bool)
                labels_mask[replace_ids] = True
                cur_labels = (cur_labels * labels_mask + IGNORE_INDEX * labels_mask.eq(0)).long()
            for j in range(seq_length):
                padded_labels[i][j] = cur_labels[position_ids_to_predict[i][j]]
            
        replace_threshold = torch.stack(replace_threshold_list, dim=0).mean()
        replace_ratio = torch.stack(replace_ratio_list, dim=0).mean()
        replace_count_min = torch.stack([torch.tensor(x.size(0), dtype=torch.long, device=self.args.device) for x in replace_indexes], dim=0).min()
        replace_count_min = get_all_reduce_min(replace_count_min)
        dist.barrier()
        
        ##=== add noise by weighted embeddings ===##
        _probs, _ids = None, None
        # training_progress = self.global_step / len(self.train_dataloader) / self.args.epochs if is_training else 0
        # var = torch.rand(1, device=self.args.device)[0]
        # var = get_all_reduce_mean(var)
        # dist.barrier()
        # if replace_count_min > 0 and var < training_progress:
        if replace_count_min > 0:
            _probs, _ids = self.collect_noisy_inputs(
                input_ids,
                attention_mask,
                replace_position_ids,
                pred_gap=self.args.pred_gap,
            )
            if self.args.sample_to_replace:
                for i in range(_ids.size(0)):
                    new_input_ids[i][replace_indexes[i]] = _ids[i, :replace_indexes[i].size(-1)]
            else:
                replace_indexes = pad_tensors(replace_indexes, pad_value=-1)
                _probs = replace_with_zero_one(_probs)
                # for bid in range(batch_size):
                #     orig_input_ids = labels[bid].clone()
                #     for j in range(replace_indexes.size(-1)):
                #         if replace_indexes[bid][j] < 0: break
                #         idx = topk_probs[bid][j].max(dim=-1).indices
                #         orig_input_ids[replace_indexes[bid][j]] = topk_ids[bid][j][idx]
                #     padded_labels[bid] = orig_input_ids[position_ids_to_predict[bid]]
                #     import ipdb; ipdb.set_trace()
        return {
            'input_ids': new_input_ids,     # (B, L)
            'labels': padded_labels,        # (B, L, Wf + Wb + 1)
            'attention_mask': attention_mask,      # (B, L)
            'position_ids': raw_position_ids.unsqueeze(0).repeat(batch_size, 1),     # (B, L)
            'position_ids_to_predict': position_ids_to_predict,        # (B, L, Wf + Wb + 1)
            'topk_probs': _probs,
            'topk_ids': _ids,
            'replace_indexes': replace_indexes if _probs is not None else None,
            'replace_threshold': replace_threshold,
            'replace_ratio': replace_ratio,
        }
    
    def create_oa_batch(self, batch: SupervisedDataset, force_replace: bool = False, fixed_replace_threshold: float = -1) -> dict[str, Any]:
        return self.corruption(**batch, force_replace=force_replace, fixed_replace_threshold=fixed_replace_threshold)
    
    def train_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        position_ids: torch.LongTensor,
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, L, Wf + Wb + 1)
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
        
        coef = 1.0 if replace_ratio > 0 else self.args.no_noise_coef
        self.model.backward(coef * loss)
        self.model.step()
        # import ipdb; ipdb.set_trace()
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
        
        if replace_ratio > 0:
            return {
                'train/lr': self.model.optimizer.param_groups[0]['lr'],
                'train/loss': loss.item(),
                'train/denoise_loss': loss.item(),
                'train/replace_threshold': replace_threshold.item(),
                'train/replace_ratio': replace_ratio.item(),
                'train/denoise_loss_ar': loss_ar.item(),
                'train/denoise_loss_close': loss_close.item(),
                'train/denoise_loss_dn': loss_dn.item(),
            }
        return {
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
            'train/loss': loss.item(),
            'train/loss_ar': loss_ar.item(),
            'train/loss_close': loss_close.item(),
            'train/loss_dn': loss_dn.item(),
        }
