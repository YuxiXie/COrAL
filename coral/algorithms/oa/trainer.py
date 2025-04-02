"""Trainer class for supervised finetuning."""

from __future__ import annotations

import os
import time
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

from coral.configs.constants import IGNORE_INDEX
from coral.datasets import SupervisedDataset, PromptOnlyDataset
from coral.trainers import SupervisedTrainer
from coral.utils import (
    random,
    shuffle_and_mask, add_noise,
    get_variable_generator, pad_tensors, 
    get_all_reduce_mean, get_all_reduce_min,
    is_main_process, to_device, 
    decode_masked_text, corrupt_input, replace_with_zero_one,
    corrupt_context,
    extract_distributions, cal_kl_divergence, cal_pred_probability, cal_kl_divergence_pos,
    json_dump, json_load,
)

from coral.models.oa_model import AutoModelForOA, OAModelOutput

from calflops.calculate_pipline import CalFlopsPipline

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
        
        self.flops, self.durations = [], []
    
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
        if self.args.tune_backbone_only:
            for name, param in self.model.base_model.named_parameters():
                if self.model.additional_layer or f'layers.{self.model.config.num_hidden_layers - 1}' not in name:
                    param.requires_grad = True
            self.model.oa_layer.requires_grad_(False)
            if not self.model.additional_layer:
                self.model.base_model.layers[-1].requires_grad_(False)
                self.model.base_model.oa_layer.requires_grad_(False)
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
            calculate_flops_pipline = CalFlopsPipline(model=self.model,include_backPropagation=False,compute_bp_factor=2)
            calculate_flops_pipline.start_flops_calculate(ignore_list=None)
            stime = time.time()
            tracks = self.model.module.oa_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.args.max_length,
                tokenizer=self.tokenizer,
                temperature=self.args.temperature,
                block_size=self.args.decoding_block_size,
                forward_size=self.args.context_window,
                backward_size=int(self.args.context_window * self.args.n_back_pred),
                occurance_threshold=self.args.decoding_occurance_threshold,
                verbal=self.args.verbal_decoding,
                left2right=self.args.left2right,
                add_denoising=self.args.add_denoising,
                skip_verify=self.args.skip_verify,
                eval_forward_size=self.args.eval_forward_size,
                eval_backward_size=self.args.eval_backward_size,
                epsilon=self.args.epsilon,
            )
            self.durations.append(time.time() - stime)
            flops = calculate_flops_pipline.get_total_flops()
            self.flops.append(flops)
            calculate_flops_pipline.end_flops_calculate()
            dist.barrier()
            return tracks
        else:
            batch = self.corruption(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                    fixed_replace_threshold=self.args.eval_replace_ratio, force_replace=self.args.eval_replace_ratio > 0,
                                    is_training=False)
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
                output_hidden_states=True,
                output_attentions=True,
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
            losses_mask = torch.ones_like(losses, dtype=torch.bool)
            for pid in range(small_pos, large_pos + 1):
                loss_dict[pid] += (relative_positions.eq(pid) * batch['position_ids_to_predict'].gt(0) * losses_mask * losses).sum().item()
                count_dict[pid] += (relative_positions.eq(pid) * batch['position_ids_to_predict'].gt(0) * losses_mask).sum().item()

            attentions = outputs['attentions'][-1].view()
            
            
            import ipdb; ipdb.set_trace()
            distributions = extract_distributions(self.model, outputs['hidden_states'])
            # kl_divergences = cal_kl_divergence_pos(distributions[-1], distributions[:-1], batch['position_ids_to_predict'])
            # kl_divergences = cal_pred_probability(batch['labels'], distributions[:-1])
            kl_divergences = cal_kl_divergence(distributions[-1], distributions[:-1])
            L, B, S, W = kl_divergences.size(0), kl_divergences.size(1), kl_divergences.size(2), kl_divergences.size(3)
            available_flags = batch['position_ids_to_predict'].ne(0).unsqueeze(0).expand(L, B, S, W)
            available_flags = available_flags.masked_fill(kl_divergences.lt(0), False)
            kl_divergences, available_flags = kl_divergences.view(L, -1, W), available_flags.view(L, -1, W)
            
            return {i: (torch.tensor(loss_dict[i], device=self.args.device), 
                        torch.tensor(count_dict[i], device=self.args.device)) for i in loss_dict if i >= (- self.args.context_window * self.args.n_back_pred)}, \
                    (kl_divergences * available_flags).sum(1), available_flags.sum(1)
    
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
        # window_size = (self.backward_context_window + 1) if self.args.context_corrupt else (self.forward_context_window + self.backward_context_window + 1)
        window_size = self.forward_context_window + self.backward_context_window + 1
        divergences: torch.FloatTensor = torch.zeros((self.model.model.config.num_hidden_layers, window_size), device=self.args.device)
        divergences_cnt: torch.FloatTensor = torch.zeros((self.model.model.config.num_hidden_layers, window_size), device=self.args.device)

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
            if self.args.skip_verify:
                self.args.result_fname += '_skip'
            else:
                self.args.result_fname += f'_e{self.args.epsilon}_{self.args.eval_forward_size}{self.args.eval_backward_size}eval'
        elif self.args.do_decoding:
            self.args.result_fname += f'_tp{self.args.temperature}'
        self.args.result_fname += f'_s{self.args.seed}'
        if os.path.exists(f'{output_dir}/{self.args.result_fname}.json'):
            existed = json_load(f'{output_dir}/{self.args.result_fname}.json')
            texts, predictions, tracks = existed['prompts'], existed['outputs'], existed['tracks']
            self.flops, self.durations = existed['flops'], existed['duration']
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
                    json_dump({'prompts': texts, 'outputs': predictions, 'tracks': tracks, 'flops': self.flops, 'duration': self.durations}, 
                              f'{output_dir}/{self.args.result_fname}.json')
            else:
                loss, div_sum, div_cnt_sum = self.eval_step(**batch)
                for pid, values in loss.items():
                    losses[pid].append(values[0])
                    counts[pid].append(values[1])
                divergences = divergences + div_sum
                divergences_cnt = divergences_cnt + div_cnt_sum

        if self.args.do_decoding:
            json_dump({'prompts': texts, 'outputs': predictions, 'tracks': tracks, 'flops': self.flops, 'duration': self.durations}, 
                      f'{output_dir}/{self.args.result_fname}.json')
            
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
            divergences = divergences / divergences_cnt
            import ipdb; ipdb.set_trace()
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
        freeze_backbone: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
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
            freeze_backbone=freeze_backbone,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        if return_logits:
            return {
                'loss': outputs.loss,
                'logits': outputs.logits.detach(),
                'hidden_states': outputs.prev_hidden_states if output_hidden_states else None,
                'attentions': outputs.attentions if output_attentions else None,
            }
        return {
            'loss': outputs.loss,
        }
    
    @torch.no_grad()
    def collect_noisy_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        new_labels: torch.LongTensor,
        topk: int = 16,
        pred_gap: int = 0,
    ) -> torch.FloatTensor:
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        raw_input_ids = input_ids.clone()
        
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
        
        probs = self.replace_sampler(input_ids, tmp_logits.view(-1, logits.size(-1)))
        probs = F.softmax(probs, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1).view(batch_size, -1)
        for bid in range(batch_size):
            indexes = new_labels[bid].ne(IGNORE_INDEX).nonzero().squeeze(-1)
            input_ids[bid, indexes] = token_ids[bid, indexes]
        return input_ids
    
    def context_corruption(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.LongTensor,
        fixed_replace_threshold: float = -1,
    ):
        batch_size = input_ids.size(0)
        
        new_input_ids, new_labels = [], []
        position_ids_to_predict, labels_to_predict = [], []
        replace_threshold_list, replace_ratio_list = [], []
        ##=== corrupt context order ===##
        for i in range(batch_size):
            ##=== extract label positions ===##
            is_label = labels[i].ne(IGNORE_INDEX)
            label_position_ids = is_label.nonzero().squeeze(-1)
            # patition into contexts
            label_start_idx = label_position_ids[0].item()
            # context_size = max(2, label_position_ids.size(-1) // self.args.corrupt_context_num)
            context_size = min(self.args.max_corrupt_context_size, max(2, label_position_ids.size(-1) // self.args.corrupt_context_num))
            if self.args.multi_context_granularity:
                context_size = random.choice([i + 1 for i in range(context_size)])
                # context_size = random.choice([i for i in range(context_size, self.backward_context_window + 1) if i <= max(2, label_position_ids.size(-1) // self.args.corrupt_context_num)])
            num_contexts = (label_position_ids.size(-1) + context_size - 1) // context_size
            ##=== initialize input ids with position info ===##
            cur_input_ids = input_ids[i].clone()[:label_start_idx]
            cur_labels = labels[i].clone()[:label_start_idx]
            ##=== construct shuffled contexts ===##
            cur_input_ids, cur_labels, cur_inject_cnt, context_inject_ratio = corrupt_context(
                cur_input_ids=cur_input_ids, cur_labels=cur_labels,
                raw_input_ids=input_ids[i], raw_labels=labels[i], raw_label_positions=label_position_ids,
                context_size=context_size, num_contexts=num_contexts,
                context_inject_ratio_generator=self.replace_ratio_generator,
                sample_from_future=self.args.sample_from_future,
                sample_from_near=self.args.sample_from_near,
                fixed_replace_threshold=fixed_replace_threshold,
                tokenizer=self.tokenizer,
            )
            new_input_ids.append(cur_input_ids)
            new_labels.append(cur_labels)
            replace_threshold_list.append(torch.tensor(context_inject_ratio, device=self.args.device).float())
            replace_ratio_list.append(torch.tensor(cur_inject_cnt / max(1, num_contexts - 1), device=self.args.device).float())
            ##=== craft input-output target ===##
            if fixed_replace_threshold >= 0:
                cur_position_ids_to_predict = torch.arange(self.forward_context_window + self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
            else:
                cur_position_ids_to_predict = torch.arange(self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
            cur_position_ids_to_predict = (cur_position_ids_to_predict - self.backward_context_window) + torch.arange(cur_input_ids.size(-1), dtype=torch.long, device=self.args.device).view(-1, 1)
            cur_position_ids_to_predict = cur_position_ids_to_predict.masked_fill(cur_position_ids_to_predict.lt(label_start_idx), 0)
            cur_position_ids_to_predict = cur_position_ids_to_predict.masked_fill(cur_position_ids_to_predict.ge(cur_labels.size(-1)), 0)
            cur_labels_to_predict = cur_labels[cur_position_ids_to_predict]
            position_ids_to_predict.append(cur_position_ids_to_predict)
            labels_to_predict.append(cur_labels_to_predict)
        new_input_ids = pad_tensors(new_input_ids, pad_value=self.tokenizer.pad_token_id)[:, :self.args.max_length].contiguous()
        new_labels = pad_tensors(new_labels)[:, :self.args.max_length].contiguous()
        position_ids_to_predict = pad_tensors(position_ids_to_predict, pad_value=0)[:, :self.args.max_length, ...].contiguous()
        labels_to_predict = pad_tensors(labels_to_predict)[:, :self.args.max_length, ...].contiguous()
        
        replace_threshold = torch.stack(replace_threshold_list, dim=0).mean()
        replace_ratio = torch.stack(replace_ratio_list, dim=0).mean()
        replace_threshold = get_all_reduce_mean(replace_threshold)
        replace_ratio = get_all_reduce_mean(replace_ratio)
        dist.barrier()
        
        if replace_ratio - .05 < replace_threshold:
            new_input_ids = self.collect_noisy_inputs(
                input_ids,
                attention_mask,
                new_labels=new_labels,
                pred_gap=self.args.pred_gap,
            )
        if self.args.verbal_training:
            print(self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True))
        return {
            'input_ids': new_input_ids,     # (B, L)
            'labels': labels_to_predict,        # (B, L, Wf + Wb + 1)
            'attention_mask': new_input_ids.ne(self.tokenizer.pad_token_id),      # (B, L)
            'position_ids': torch.arange(0, new_input_ids.size(-1), dtype=torch.long, device=self.args.device).unsqueeze(0).repeat(batch_size, 1),     # (B, L)
            'position_ids_to_predict': position_ids_to_predict,        # (B, L, Wf + Wb + 1)
            'topk_probs': None,
            'topk_ids': None,
            'replace_indexes': None,
            'replace_threshold': replace_threshold,
            'replace_ratio': replace_ratio,
            'freeze_backbone': False,
        }
    
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
        
        if force_replace and self.args.context_corrupt:
            return self.context_corruption(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                fixed_replace_threshold=fixed_replace_threshold,
            )
        
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        raw_position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.args.device)
        
        replace_threshold_list, replace_ratio_list, replace_position_ids, replace_indexes = [], [], [], []
        
        new_input_ids = input_ids.clone()
        if not is_training:
            position_ids_to_predict = torch.arange(self.forward_context_window + self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
            position_ids_to_predict = (position_ids_to_predict - self.backward_context_window) + torch.arange(seq_length, dtype=torch.long, device=self.args.device).view(-1, 1)
            position_ids_to_predict = position_ids_to_predict.unsqueeze(0).expand(batch_size, seq_length, self.forward_context_window + self.backward_context_window + 1).contiguous()
            padded_labels = torch.zeros(batch_size, seq_length, self.forward_context_window + self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
        elif force_replace:
            position_ids_to_predict = torch.arange(self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
            position_ids_to_predict = (position_ids_to_predict - self.backward_context_window) + torch.arange(seq_length, dtype=torch.long, device=self.args.device).view(-1, 1)
            position_ids_to_predict = position_ids_to_predict.unsqueeze(0).expand(batch_size, seq_length, self.backward_context_window + 1).contiguous()
            padded_labels = torch.zeros(batch_size, seq_length, self.backward_context_window + 1, dtype=torch.long, device=self.args.device)
        else:
            position_ids_to_predict = torch.arange(self.forward_context_window, dtype=torch.long, device=self.args.device)
            position_ids_to_predict = (position_ids_to_predict + 1) + torch.arange(seq_length, dtype=torch.long, device=self.args.device).view(-1, 1)
            position_ids_to_predict = position_ids_to_predict.unsqueeze(0).expand(batch_size, seq_length, self.forward_context_window).contiguous()
            padded_labels = torch.zeros(batch_size, seq_length, self.forward_context_window, dtype=torch.long, device=self.args.device)
        
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
            'freeze_backbone': is_training and force_replace,
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
        freeze_backbone: bool = False,
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
            freeze_backbone=freeze_backbone,
        )
        
        loss = outputs['loss']
        
        coef = 1.0 if replace_ratio > 0 else self.args.no_noise_coef
        self.model.backward(coef * loss)
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
