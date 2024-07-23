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
    decode_masked_text, corrupt_input,
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
        
        if not self.args.dynamic_mask_ratio_mu:
            self.mask_ratio_generator = get_variable_generator(self.args.mask_ratio_mu, self.args.mask_ratio_std, self.args.mask_ratio_min, self.args.mask_ratio_max)
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
                positions_to_replace=batch['positions_to_replace'],
                position_ids_to_predict=batch['position_ids_to_predict'],
                max_length=self.args.max_length,
                tokenizer=self.tokenizer,
                verbal=self.args.verbal_decoding,
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
                # if cnt >= 100: break
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
            json_dump({'prompts': texts, 'outputs': predictions, 'tracks': tracks}, f'{output_dir}/decoding_tracks_step_denoise.json')
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
            if self.args.eval_mask_ratio < 1:
                json_dump(losses, f'{output_dir}/losses_ratio{self.args.eval_mask_ratio}.json')
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
        positions_to_replace: torch.LongTensor | None = None,  # (B, 1)
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
            positions_to_replace=positions_to_replace,
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
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        replace_position_ids: list[torch.LongTensor],
        topk: int = -1,
    ) -> torch.FloatTensor:
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        raw_position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.args.device)
        
        input_ids_list, attention_mask_list, position_ids_list = [], [], []
        positions_to_replace_list, position_ids_to_predict_list = [], []
        max_add_positions, max_seq_len = 0, 0
        for i in range(batch_size):
            replace_ids = replace_position_ids[i]            
            cur_input_ids = input_ids[i][attention_mask[i].nonzero().squeeze(dim=-1)].clone()
            corrupt_ids, _ = shuffle_and_mask(labels[i].ne(IGNORE_INDEX).nonzero().squeeze(), self.mask_ratio_generator, device=self.args.device)
            cur_input_ids[corrupt_ids] = IGNORE_INDEX
            selected_indexes = cur_input_ids.ne(IGNORE_INDEX).nonzero().squeeze(dim=-1)
            cur_input_ids = cur_input_ids[selected_indexes]
            input_ids_list.append(cur_input_ids)
            position_ids_list.append(raw_position_ids[selected_indexes])
            positions_to_replace_list.append(torch.tensor(cur_input_ids.size(0), dtype=torch.long, device=self.args.device))
            position_ids_to_predict_list.append(replace_ids)
            max_add_positions = max(max_add_positions, replace_ids.size(0))
            max_seq_len = max(max_seq_len, cur_input_ids.size(0))
        for i in range(batch_size):
            pad_len = max_seq_len - input_ids_list[i].size(-1)
            attention_mask_list.append(torch.cat((
                torch.ones_like(input_ids_list[i], dtype=torch.bool),
                torch.zeros((pad_len,), dtype=torch.bool, device=self.args.device),
            ), dim=-1).bool())
            tmp = torch.ones((pad_len,), dtype=torch.long, device=self.args.device)
            input_ids_list[i] = torch.cat((input_ids_list[i], tmp * self.tokenizer.pad_token_id), dim=-1).long()
            position_ids_list[i] = torch.cat((position_ids_list[i], tmp * 0), dim=-1).long()
            
            to_add_len = max_add_positions - position_ids_to_predict_list[i].size(-1)
            tmp = torch.ones((to_add_len,), dtype=torch.long, device=self.args.device)
            position_ids_to_predict_list[i] = torch.cat((position_ids_to_predict_list[i], tmp * 0), dim=-1).long()
        
        new_input_ids = torch.stack(input_ids_list, dim=0)
        new_attention_mask = torch.stack(attention_mask_list, dim=0)
        position_ids=torch.stack(position_ids_list, dim=0)
        positions_to_replace=torch.stack(positions_to_replace_list, dim=0).unsqueeze(-1)
        position_ids_to_predict=torch.stack(position_ids_to_predict_list, dim=0)
        # #################### sanity check ####################
        # idx = 0
        # _cur_input_ids, cur_position_ids = new_input_ids[idx], position_ids[idx]
        # _id = self.tokenizer.encode('_')[-1]
        # vis_ids = [_cur_input_ids[cur_position_ids.eq(idx).nonzero()[0].item()].item() if idx in cur_position_ids else _id for idx in range(cur_position_ids.min().item(), cur_position_ids.max().item() + 1)]
        # import ipdb; ipdb.set_trace()
        # ######################################################
        with torch.no_grad():
            logits = self.model(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=position_ids,
                positions_to_replace=positions_to_replace,
                position_ids_to_predict=position_ids_to_predict,
            ).logits[:, new_input_ids.size(-1):, :]
        
        topk = topk if topk > 0 else self.tokenizer.vocab_size
        results = torch.topk(F.softmax(logits, dim=-1), k=topk, dim=-1)
        
        return results.values, results.indices
    
    @torch.no_grad()
    def masking(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        topk: int = 8,
        fixed_mask_threshold: float = -1,
        fixed_replace_threshold: float = -1,
        is_training: bool = True,
    ) -> dict[str, Any]:
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        raw_position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.args.device)
        
        ##=== get mask / shuffle ratio (generator) ===##
        if self.args.dynamic_mask_ratio_mu:
            mask_ratio_mu = self.args.min_mask_ratio_mu + (self.args.max_mask_ratio_mu - self.args.min_mask_ratio_mu) * (self.global_step / len(self.train_dataloader) / self.args.epochs)
            # mask_ratio_min = self.args.mask_ratio_min   # TODO: original version - fixed min mask_ratio
            mask_ratio_mu = min(mask_ratio_mu, self.args.mask_ratio_mu)
            mask_ratio_min = min(self.args.mask_ratio_min, mask_ratio_mu - 0.05)   # TODO: dynamic min mask_ratio
            self.mask_ratio_generator = get_variable_generator(mask_ratio_mu, self.args.mask_ratio_std, mask_ratio_min, self.args.mask_ratio_max)
        if self.args.mage_shuffle:
            shuffle_ratio = self.args.init_shuffle_ratio_mage + (self.args.max_shuffle_ratio_mage - self.args.init_shuffle_ratio_mage) * (self.global_step / len(self.train_dataloader) / self.args.epochs)

        ##=== init variables ===##
        mask_threshold_list, mask_ratio_list = [], []
        if self.args.reconstruct:
            replace_threshold_list, replace_ratio_list = [], []
            replace_position_ids, replace_indexes = [], []
        input_ids_list, labels_list, attention_mask_list, position_ids_list = [], [], [], []
        positions_to_replace_list, position_ids_to_predict_list, add_labels_list, add_label_tags_list = [], [], [], []
        max_add_positions, max_seq_len = 0, 0
        insert_cnt, replace_cnt, self_replace_cnt = 0, 0, 0
        
        to_insert = torch.rand(1, device=self.args.device).mean()
        to_insert = get_all_reduce_mean(to_insert)
        dist.barrier()
        to_insert = to_insert.lt(.5).item() if self.args.insert_with_prob > 0 else False
        
        ##=== masking & adding noise ===##
        for i in range(batch_size):
            ##=== extract label positions ===##
            is_label = labels[i].ne(IGNORE_INDEX)
            label_position_ids = is_label.nonzero().squeeze(-1)
            # import ipdb; ipdb.set_trace()
            ##=== shuffle and mask ===##
            mask_label_position_ids, mask_threshold = shuffle_and_mask(label_position_ids, self.mask_ratio_generator,
                                                                       fixed_mask_threshold=fixed_mask_threshold, 
                                                                       device=self.args.device)
            mask_threshold_list.append(torch.tensor(mask_threshold, device=self.args.device))
            mask_ratio_list.append(torch.tensor(mask_label_position_ids.size(0) / max(1, label_position_ids.size(-1)), device=self.args.device))
            ##=== mask dedicated tokens ===##
            mask_attention_mask = attention_mask[i].clone()
            mask_attention_mask[mask_label_position_ids] = False
            mask_attention_mask[-1] = False     # mask the final token
            end_index = mask_attention_mask.nonzero().max()
            if is_training:
                mask_attention_mask[end_index + 1] = True   # for training only -- the token to predict next
            
            ##=== initialize input ids with position info ===##
            # select the initial token (at end_idx+1) to predict
            select_index = torch.randint(mask_attention_mask.nonzero().size(0), (1,)).to(self.args.device)
            # init input_ids
            cur_input_ids = input_ids[i].clone()
            cur_input_ids[mask_label_position_ids] = self.tokenizer.pad_token_id
            cur_input_ids[end_index + 1] = input_ids[i][select_index.item()]
            cur_input_ids = cur_input_ids[mask_attention_mask.nonzero().squeeze(dim=-1)]
            # init position ids
            cur_position_ids = raw_position_ids.clone()
            cur_position_ids[end_index + 1] = raw_position_ids[select_index.item()]
            cur_position_ids = cur_position_ids[mask_attention_mask.nonzero().squeeze(dim=-1)]
            
            ##=== rearrange unmasked tokens to get labels ===##
            cur_labels = labels[i][cur_position_ids]
            # extract position ids to predict
            position_ids_to_predict = raw_position_ids[mask_label_position_ids]
            add_cur_labels = labels[i][position_ids_to_predict]
            add_label_tags = torch.ones_like(add_cur_labels)
            
            ##=== replace input ids ===##
            if self.args.reconstruct:
                replace_ids, new_replace_ids, replace_threshold = add_noise(cur_input_ids, cur_labels, self.replace_ratio_generator, 
                                                                            replace_with_prob=self.args.replace_with_prob, 
                                                                            fixed_replace_threshold=fixed_replace_threshold,
                                                                            device=self.args.device)
                replace_threshold_list.append(torch.tensor(replace_threshold, device=self.args.device))
                replace_ratio_list.append(torch.tensor(replace_ids.size(0) / max(1, labels[i, cur_position_ids].ne(IGNORE_INDEX).sum()), device=self.args.device))
                # update the positions to predict
                position_ids_to_predict = torch.cat((position_ids_to_predict, cur_position_ids[replace_ids]), dim=-1)
                add_cur_labels = torch.cat((add_cur_labels, cur_labels[replace_ids]), dim=-1)
                add_label_tags = torch.cat((add_label_tags, torch.zeros_like(cur_labels[replace_ids])), dim=-1)
                
                # replace input ids
                cur_replace_ids = new_replace_ids if self.args.replace_with_prob < 1 else replace_ids
                replace_indexes.append(cur_replace_ids)
                replace_position_ids.append(cur_position_ids[cur_replace_ids])
                if cur_replace_ids.size(-1) > 0 and cur_labels.ne(IGNORE_INDEX).any().item():
                    if to_insert and random.random() < self.args.insert_with_prob:
                        insert_cnt += 1
                        cur_input_ids, cur_position_ids, cur_labels = corrupt_input(cur_replace_ids, cur_input_ids, input_ids[i], cur_position_ids, cur_labels, labels[i], 
                                                                                    self.tokenizer, insert_ratio=self.args.insert_with_prob, max_length=self.args.max_length, 
                                                                                    device=self.args.device)
                    else:
                        replace_cnt += 1
                        cur_input_ids, cur_position_ids, cur_labels = corrupt_input(cur_replace_ids, cur_input_ids, input_ids[i], cur_position_ids, cur_labels, labels[i], 
                                                                                    self.tokenizer, max_length=self.args.max_length, device=self.args.device)
            
            max_seq_len = max(cur_labels.size(-1), max_seq_len)
            max_add_positions = max(max_add_positions, position_ids_to_predict.size(-1))
            
            input_ids_list.append(cur_input_ids)
            labels_list.append(cur_labels)
            attention_mask_list.append(torch.ones_like(cur_input_ids, dtype=torch.bool))
            position_ids_list.append(cur_position_ids)
            positions_to_replace_list.append(torch.tensor(cur_input_ids.size(-1) - 1, device=self.args.device))
            position_ids_to_predict_list.append(position_ids_to_predict)
            add_labels_list.append(add_cur_labels)
            add_label_tags_list.append(add_label_tags)
        
        ##=== arrange samples with padding ===##
        for i in range(batch_size):
            # pad all input samples to be of the same length
            pad_len = max_seq_len - input_ids_list[i].size(-1)
            attention_mask_list[i] = torch.cat((
                torch.ones_like(input_ids_list[i], dtype=torch.bool),
                torch.zeros((pad_len,), dtype=torch.bool, device=self.args.device),
            ), dim=-1).bool()
            # shuffle input orders
            if self.args.mage_shuffle:
                if torch.rand(1) < shuffle_ratio:
                    # shuffle the order of inputs
                    random_noise = torch.rand(input_ids_list[i].size(-1), device=self.args.device)
                    _, shuffled_ids = torch.sort(random_noise)
                    # create inputs with the new order
                    input_ids_list[i] = input_ids_list[i][shuffled_ids]
                    labels_list[i] = labels_list[i][shuffled_ids]
                    position_ids_list[i] = position_ids_list[i][shuffled_ids]
            tmp = torch.ones((pad_len,), dtype=torch.long, device=self.args.device)
            input_ids_list[i] = torch.cat((input_ids_list[i], tmp * self.tokenizer.pad_token_id), dim=-1).long()
            tag = torch.cat((torch.ones_like(labels_list[i]) * 2, tmp * IGNORE_INDEX), dim=-1).long()
            labels_list[i] = torch.cat((labels_list[i], tmp * IGNORE_INDEX), dim=-1).long()
            position_ids_list[i] = torch.cat((position_ids_list[i], tmp * 0), dim=-1).long()
            
            # pad position_ids_to_predict to be of the same number
            cur_seq_len = position_ids_to_predict_list[i].size(-1)
            to_add_len = max_add_positions - cur_seq_len
            tmp = torch.ones((to_add_len,), dtype=torch.long, device=self.args.device)
            position_ids_to_predict_list[i] = torch.cat((position_ids_to_predict_list[i], tmp * 0), dim=-1).long()
            add_labels_list[i] = torch.cat((add_labels_list[i], tmp * IGNORE_INDEX), dim=-1).long()
            add_label_tags_list[i] = torch.cat((add_label_tags_list[i], tmp * IGNORE_INDEX), dim=-1).long()
            labels_list[i] = torch.cat((labels_list[i], add_labels_list[i]), dim=-1).long()
            add_label_tags_list[i] = torch.cat((tag, add_label_tags_list[i]), dim=-1).long()
        
        mask_threshold = torch.stack(mask_threshold_list, dim=0).mean()
        mask_ratio = torch.stack(mask_ratio_list, dim=0).mean()
        if self.args.reconstruct:
            replace_threshold = torch.stack(replace_threshold_list, dim=0).mean()
            replace_ratio = torch.stack(replace_ratio_list, dim=0).mean()
            replace_count_min = torch.stack([torch.tensor(x.size(0), dtype=torch.long, device=self.args.device) for x in replace_indexes], dim=0).min()
            if insert_cnt > 0:
                replace_count_min = replace_count_min * 0
            replace_count_min = get_all_reduce_min(replace_count_min)
        dist.barrier()
        
        ##=== add noise by weighted embeddings ===##
        topk_probs, topk_ids = None, None
        if self.args.reconstruct and not self.args.random_noise and len(replace_position_ids) and replace_count_min > 0:
            topk_probs, topk_ids = self.collect_noisy_inputs(
                input_ids,
                labels,
                attention_mask,
                replace_position_ids,
                topk=topk,
            )
            replace_indexes = pad_tensors(replace_indexes, pad_value=-1)
            replace_cnt, self_replace_cnt = 0, batch_size
        
        return {
            'input_ids': torch.stack(input_ids_list, dim=0),
            'labels': torch.stack(labels_list, dim=0),
            'attention_mask': torch.stack(attention_mask_list, dim=0),
            'position_ids': torch.stack(position_ids_list, dim=0),
            'positions_to_replace': torch.stack(positions_to_replace_list, dim=0).unsqueeze(-1),
            'position_ids_to_predict': torch.stack(position_ids_to_predict_list, dim=0),
            'topk_probs': topk_probs,
            'topk_ids': topk_ids,
            'replace_indexes': replace_indexes if self.args.reconstruct else None,
            'mask_threshold': mask_threshold,
            'mask_ratio': mask_ratio,
            'replace_threshold': replace_threshold if self.args.reconstruct else torch.zeros_like(mask_threshold),
            'replace_ratio': replace_ratio if self.args.reconstruct else torch.zeros_like(mask_ratio),
            'add_label_tags': torch.stack(add_label_tags_list, dim=0),
            'insert_part': torch.tensor(insert_cnt / batch_size, dtype=torch.float, device=self.args.device),
            'replace_part': torch.tensor(replace_cnt / batch_size, dtype=torch.float, device=self.args.device),
            'self_replace_part': torch.tensor(self_replace_cnt / batch_size, dtype=torch.float, device=self.args.device),
        }
    
    def create_oa_batch(self, batch: SupervisedDataset) -> dict[str, Any]:
        return self.masking(**batch)
    
    def vanilla_shuffle_train_step(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> dict[str, Any]:
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        raw_position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.args.device)
        
        # unshuffled inputs in left-to-right order
        cur_position_ids = raw_position_ids.unsqueeze(0).view(-1, seq_length).expand(batch_size, seq_length)
        cur_input_ids = input_ids
        cur_labels = labels
        
        shuffle_ratio = self.args.init_shuffle_ratio + (self.args.max_shuffle_ratio - self.args.init_shuffle_ratio) * self.global_step / len(self.train_dataloader) / self.args.epochs
        
        all_input_ids, all_labels, all_attention_mask, all_position_ids = [], [], [], []        
        flag = not self.args.exclude_l2r_order
        for cnt in range(self.args.n_shuffle + 1):
            if flag:
                all_input_ids.append(cur_input_ids)
                all_labels.append(cur_labels)
                all_attention_mask.append(attention_mask)
                all_position_ids.append(cur_position_ids)
            
            if cnt >= self.args.n_shuffle: break
            
            flag = self.args.exclude_l2r_order
            input_ids_list, labels_list, position_ids_list = [], [], []
            for i in range(batch_size):
                # extract label positions
                is_label = labels[i].ne(IGNORE_INDEX)
                label_position_ids = is_label.nonzero().squeeze(-1)
                
                # organize label positions into groups (e.g. multi-turn dialogue)
                output_ids_groups, cur_j = [], 0
                for j in range(1, label_position_ids.size(-1)):
                    if label_position_ids[j] - label_position_ids[j - 1] == 1: continue
                    output_ids_groups.append(label_position_ids[cur_j:j])
                    cur_j = j
                output_ids_groups.append(label_position_ids[cur_j:])
                
                input_ids_copy = input_ids[i].clone()
                labels_copy = labels[i].clone()
                position_ids_copy = raw_position_ids.clone()
                for label_position_ids_group in output_ids_groups:
                    if torch.rand(1) < shuffle_ratio:
                        # shuffle the order of labels
                        random_noise = torch.rand(label_position_ids_group.size(-1), device=self.args.device)
                        _, shuffled_ids = torch.sort(random_noise)
                        shuffled_position_ids = label_position_ids_group[shuffled_ids]
                        # create inputs with the new order
                        input_ids_copy[label_position_ids_group] = input_ids_copy[shuffled_position_ids]
                        labels_copy[label_position_ids_group] = labels_copy[shuffled_position_ids]
                        position_ids_copy[label_position_ids_group] = position_ids_copy[shuffled_position_ids]
                        flag = True
                input_ids_list.append(input_ids_copy)
                labels_list.append(labels_copy)
                position_ids_list.append(position_ids_copy)
            # gather inputs into a batch
            cur_input_ids = torch.stack(input_ids_list, dim=0)
            cur_labels = torch.stack(labels_list, dim=0)
            cur_position_ids = torch.stack(position_ids_list, dim=0)
        
        loss = self.loss(
            input_ids=torch.cat(all_input_ids, dim=0),
            labels=torch.cat(all_labels, dim=0),
            attention_mask=torch.cat(all_attention_mask, dim=0),
            position_ids=torch.cat(all_position_ids, dim=0),
            position_ids_to_predict=torch.zeros((all_input_ids[0].size(0) * len(all_input_ids), 0), dtype=torch.long, device=self.args.device),
        )['loss']
        self.model.backward(loss)
        self.model.step()

        loss = get_all_reduce_mean(loss)
        
        dist.barrier()

        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }
    
    def train_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        position_ids: torch.LongTensor,
        positions_to_replace: torch.LongTensor | None = None,  # (B, 1)
        position_ids_to_predict: torch.LongTensor | None = None,    # (B, N)
        topk_probs: torch.FloatTensor | None = None,    # (B, M, K)
        topk_ids: torch.LongTensor | None = None,   # (B, M, K)
        replace_indexes: torch.LongTensor | None = None,
        mask_threshold: torch.Tensor | None = None,
        mask_ratio: torch.Tensor | None = None,
        replace_threshold: torch.Tensor | None = None,
        replace_ratio: torch.Tensor | None = None,
        add_label_tags: torch.LongTensor | None = None,
        insert_part: torch.LongTensor | None = None,
        replace_part: torch.LongTensor | None = None,
        self_replace_part: torch.LongTensor | None = None,
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
            positions_to_replace=positions_to_replace,
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
        logits = outputs['logits']
        
        orig_len = input_ids.size(-1)
        shift_logits = torch.cat((logits[..., :orig_len-1, :].contiguous(), logits[..., orig_len:, :].contiguous()), dim=1)
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, self.model.module.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits, shift_labels).view(input_ids.size(0), -1)
        
        shift_tags = add_label_tags[..., 1:].contiguous()
        loss_ar = (losses * shift_tags.eq(2)).sum() / max(1, shift_tags.eq(2).sum())
        loss_oa = (losses * shift_tags.eq(1)).sum() / max(1, shift_tags.eq(1).sum())
        loss_dn = (losses * shift_tags.eq(0)).sum() / max(1, shift_tags.eq(0).sum())
        
        # idx = 0
        # text = decode_masked_text(input_ids[idx], position_ids[idx], replace_indexes[idx], self.tokenizer, topk_ids=topk_ids[idx] if topk_ids is not None else None)
        # import ipdb; ipdb.set_trace()
        ##########################################################################
        
        loss = get_all_reduce_mean(loss)
        mask_threshold = get_all_reduce_mean(mask_threshold)
        mask_ratio = get_all_reduce_mean(mask_ratio)
        if self.args.reconstruct:
            replace_threshold = get_all_reduce_mean(replace_threshold)
            replace_ratio = get_all_reduce_mean(replace_ratio)
        loss_ar = get_all_reduce_mean(loss_ar)
        loss_oa = get_all_reduce_mean(loss_oa)
        loss_dn = get_all_reduce_mean(loss_dn)
        
        insert_part = get_all_reduce_mean(insert_part)
        replace_part = get_all_reduce_mean(replace_part)
        self_replace_part = get_all_reduce_mean(self_replace_part)
        
        dist.barrier()
        
        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
            'train/mask_threshold': mask_threshold.item(),
            'train/mask_ratio': mask_ratio.item(),
            'train/replace_threshold': replace_threshold.item() if self.args.reconstruct else 0,
            'train/replace_ratio': replace_ratio.item() if self.args.reconstruct else 0,
            'train/loss_ar': loss_ar.item(),
            'train/loss_oa': loss_oa.item(),
            'train/loss_dn': loss_dn.item(),
            'train/insert_part': insert_part.item(),
            'train/replace_part': replace_part.item(),
            'train/self_replace_part': self_replace_part.item(),
        }
