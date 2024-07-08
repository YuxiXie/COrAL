"""Trainer class for supervised finetuning."""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Any
from tqdm import tqdm
import deepspeed
from collections import defaultdict

import torch
from torch.nn import Module, CrossEntropyLoss
import torch.distributed as dist

from oa_dag.configs.constants import IGNORE_INDEX
from oa_dag.datasets import SupervisedDataset, PromptOnlyDataset
from oa_dag.trainers import SupervisedTrainer
from oa_dag.utils import get_variable_generator, get_all_reduce_mean, is_main_process, to_device, json_dump

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
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        raw_position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.args.device)
        
        if self.args.vanilla_shuffle:
            return self.vanilla_shuffle_train_step(input_ids, labels, attention_mask, raw_position_ids)
        
        max_add_positions, max_seq_len = 0, 0
        input_ids_list, labels_list, attention_mask_list, position_ids_list = [], [], [], []
        positions_to_replace_list, position_ids_to_predict_list, add_labels_list = [], [], []
        for i in range(batch_size):
            # extract label positions
            is_label = labels[i].ne(IGNORE_INDEX)
            label_position_ids = is_label.nonzero().squeeze(-1)
            
            # shuffle and mask
            mask_threshold = self.args.eval_mask_ratio
            random_noise = torch.rand(label_position_ids.size(-1), device=self.args.device)
            mask_label_position_ids = label_position_ids[random_noise.le(mask_threshold).nonzero().squeeze(-1)]
            
            mask_attention_mask = attention_mask[i].clone()
            mask_attention_mask[mask_label_position_ids] = False
            
            cur_labels = labels[i].clone()
            cur_labels = cur_labels[mask_attention_mask.nonzero().squeeze()]
            max_seq_len = max(cur_labels.size(-1), max_seq_len)
            
            # init input_ids
            cur_input_ids = input_ids[i].clone()
            cur_input_ids[mask_label_position_ids] = self.tokenizer.pad_token_id
            cur_input_ids = cur_input_ids[mask_attention_mask.nonzero().squeeze()]
            # init position ids
            cur_position_ids = raw_position_ids.clone()
            cur_position_ids = cur_position_ids[mask_attention_mask.nonzero().squeeze()]
            
            # extract position ids to predict
            position_ids_to_predict = raw_position_ids[mask_label_position_ids]
            add_cur_labels = labels[i][position_ids_to_predict]
            max_add_positions = max(max_add_positions, position_ids_to_predict.size(-1))
            
            input_ids_list.append(cur_input_ids)
            labels_list.append(cur_labels)
            attention_mask_list.append(mask_attention_mask)
            position_ids_list.append(cur_position_ids)
            positions_to_replace_list.append(torch.tensor(cur_input_ids.size(-1), device=self.args.device))
            position_ids_to_predict_list.append(position_ids_to_predict)
            add_labels_list.append(add_cur_labels)
        
        for i in range(batch_size):
            # pad all input samples to be of the same length
            pad_len = max_seq_len - input_ids_list[i].size(-1)
            attention_mask_list[i] = torch.cat((
                torch.ones((input_ids_list[i].size(-1),), dtype=torch.bool, device=self.args.device),
                torch.zeros((pad_len,), dtype=torch.bool, device=self.args.device),
            ), dim=-1).bool()
            tmp = torch.ones((pad_len,), dtype=torch.long, device=self.args.device)
            input_ids_list[i] = torch.cat((input_ids_list[i], tmp * self.tokenizer.pad_token_id), dim=-1).long()
            labels_list[i] = torch.cat((labels_list[i], tmp * IGNORE_INDEX), dim=-1).long()
            position_ids_list[i] = torch.cat((position_ids_list[i], tmp * 0), dim=-1).long()
            
            # pad position_ids_to_predict to be of the same number
            cur_seq_len = position_ids_to_predict_list[i].size(-1)
            to_add_len = max_add_positions - cur_seq_len                
            tmp = torch.ones((to_add_len,), dtype=torch.long, device=self.args.device)
            position_ids_to_predict_list[i] = torch.cat((position_ids_to_predict_list[i], tmp * 0), dim=-1).long()
            add_labels_list[i] = torch.cat((add_labels_list[i], tmp * IGNORE_INDEX), dim=-1).long()
            labels_list[i] = torch.cat((labels_list[i], add_labels_list[i]), dim=-1).long()
        
        new_input_ids = torch.stack(input_ids_list, dim=0)
        new_labels = torch.stack(labels_list, dim=0)
        new_attention_mask = torch.stack(attention_mask_list, dim=0)
        position_ids=torch.stack(position_ids_list, dim=0)
        positions_to_replace=torch.stack(positions_to_replace_list, dim=0).unsqueeze(-1)
        position_ids_to_predict=torch.stack(position_ids_to_predict_list, dim=0)
        
        if self.args.do_decoding:
            with torch.no_grad():
                tracks = self.model.module.oa_generate(
                    input_ids=new_input_ids,
                    attention_mask=new_attention_mask,
                    position_ids=position_ids,
                    positions_to_replace=positions_to_replace,
                    position_ids_to_predict=position_ids_to_predict,
                    temperature=self.args.temperature,
                    max_length=self.args.max_length,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    tokenizer=self.tokenizer,
                    max_n_tokens=self.args.max_n_tokens_per_step,
                    do_sample=self.args.do_sample,
                    denoising=self.args.denoising,
                )
            dist.barrier()
            
            return tracks
        else:        
            with torch.no_grad():
                outputs = self.model(
                    input_ids=new_input_ids,
                    labels=new_labels,
                    attention_mask=new_attention_mask,
                    position_ids=position_ids,
                    positions_to_replace=positions_to_replace,
                    position_ids_to_predict=position_ids_to_predict,
                )
            dist.barrier()
            
            logits = outputs.logits[:, input_ids_list[0].size(-1):].contiguous()
            gt_labels = torch.stack(labels_list, dim=0)[:, input_ids_list[0].size(-1):].contiguous()
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, logits.size(-1)), gt_labels.view(-1)).view(batch_size, -1)
            
            loss_dict = defaultdict(float)
            count_dict = defaultdict(float)
            for bid in range(loss.size(0)):
                start_idx = labels[bid].ne(IGNORE_INDEX).nonzero().min().item()
                for j in range(position_ids_to_predict[bid].size(-1)):
                    pid = position_ids_to_predict[bid][j].item()
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
                if cnt >= 100: break
            else:
                loss = self.eval_step(**batch)
                for pid, values in loss.items():
                    losses[pid].append(values[0])
                    counts[pid].append(values[1])

        if self.args.do_decoding:
            if self.args.do_sample:
                json_dump({'prompts': texts, 'outputs': predictions, 'tracks': tracks}, f'{output_dir}/decoding_tracks_step_sample{self.args.max_n_tokens_per_step}.json')
            else:
                json_dump({'prompts': texts, 'outputs': predictions, 'tracks': tracks}, f'{output_dir}/decoding_tracks_step{self.args.max_n_tokens_per_step}.json')
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
                losses[key] = losses[key].sum().item() / counts[key].sum().item()
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
    ) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs: OAModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            positions_to_replace=positions_to_replace,
            position_ids_to_predict=position_ids_to_predict,
            use_cache=False
        )
        # #################### sanity check ####################
        # logits = outputs.logits.detach()
        # orig_len = position_ids.size(-1)
        # shift_logits = torch.cat((logits[..., :orig_len-1, :].contiguous(), logits[..., orig_len:, :].contiguous()), dim=1)
        # shift_labels = labels[..., 1:].contiguous()
        # shift_logits = shift_logits.view(-1, self.model.module.config.vocab_size)
        # shift_labels = shift_labels.view(-1).to(shift_logits.device)
        # loss_fct = CrossEntropyLoss(reduction='none')
        # loss = loss_fct(shift_logits, shift_labels).view(input_ids.size(0), -1)
        # idx = 0
        # selected_ids = labels[idx][input_ids.size(-1):].ge(0).nonzero().squeeze()
        # losses = loss[idx][input_ids.size(-1) - 1:][selected_ids]
        # special_ids = losses.lt(2).nonzero().squeeze()
        # tokens = [self.tokenizer.decode([x]) for x in labels[idx][input_ids.size(-1):][selected_ids[special_ids]]]
        # import ipdb; ipdb.set_trace()
        # ######################################################
        return {
            'loss': outputs.loss,
        }
        
    def vanilla_shuffle_train_step(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        raw_position_ids: torch.LongTensor,
    ) -> dict[str, Any]:
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        
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
    ) -> dict[str, Any]:
        """Performs a single training step.

        Args:
            input_ids (torch.LongTensor): input ids for causal inputs to complete with.
            labels (torch.LongTensor): labels for the full sequence.
            attention_mask (torch.BoolTensor): attention mask for the labels.

        Returns:
            dict[str, Any]: training loss, learning rate
        """
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        raw_position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.args.device)
        
        if self.args.vanilla_shuffle:
            return self.vanilla_shuffle_train_step(input_ids, labels, attention_mask, raw_position_ids)

        if self.args.dynamic_mask_ratio_mu:
            mask_ratio_mu = self.args.min_mask_ratio_mu + (self.args.max_mask_ratio_mu - self.args.min_mask_ratio_mu) * (self.global_step / len(self.train_dataloader) / self.args.epochs)
            # mask_ratio_min = self.args.mask_ratio_min
            mask_ratio_mu = min(mask_ratio_mu, self.args.mask_ratio_mu)
            mask_ratio_min = mask_ratio_mu - 0.05   # TODO: magic number
            self.mask_ratio_generator = get_variable_generator(mask_ratio_mu, self.args.mask_ratio_std, mask_ratio_min, self.args.mask_ratio_max)
        if self.args.mage_shuffle:
            shuffle_ratio = self.args.init_shuffle_ratio_mage + (self.args.max_shuffle_ratio_mage - self.args.init_shuffle_ratio_mage) * (self.global_step / len(self.train_dataloader) / self.args.epochs)
        
        mask_threshold_list, mask_ratio_list = [], []   # ratio and proportion to mask the inputs to predict simutaneously
        replace_threshold_list, replace_ratio_list = [], []
        all_input_ids, all_labels, all_attention_mask, all_position_ids = [], [], [], []
        all_positions_to_replace, all_position_ids_to_prediction = [], []
        max_add_positions, max_seq_len = 0, 0
        for _ in range(self.args.n_shuffle):
            input_ids_list, labels_list, attention_mask_list, position_ids_list = [], [], [], []
            positions_to_replace_list, position_ids_to_predict_list, add_labels_list = [], [], []
            for i in range(batch_size):
                ##=== extract label positions ===##
                is_label = labels[i].ne(IGNORE_INDEX)
                label_position_ids = is_label.nonzero().squeeze(-1)
                
                ##=== shuffle and mask ===##
                keep_sample = True
                while keep_sample:
                    # sample to get masking threshold                    
                    mask_threshold = self.mask_ratio_generator.rvs(1)[0]
                    if self.args.mage_consecutive:  # mask the right part
                        random_noise = torch.arange(0, label_position_ids.size(-1), dtype=torch.float, device=self.args.device)
                        random_noise = (-random_noise + label_position_ids.size(-1) - 0.5) / label_position_ids.size(-1)    # reverse to be descending
                    else:   # randomly mask
                        random_noise = torch.rand(label_position_ids.size(-1), device=self.args.device)
                    # extract the position ids of the tokens to mask
                    mask_label_position_ids = label_position_ids[random_noise.lt(mask_threshold).nonzero().squeeze(-1)]
                    if mask_label_position_ids.size(0) > 0 or label_position_ids.size(0) <= 0:
                        keep_sample = False
                mask_threshold_list.append(torch.tensor(mask_threshold, device=self.args.device))
                mask_ratio_list.append(torch.tensor(mask_label_position_ids.size(0) / max(1, label_position_ids.size(-1)), device=self.args.device))
                ##=== mask dedicated tokens ===##
                mask_attention_mask = attention_mask[i].clone()
                mask_attention_mask[mask_label_position_ids] = False
                mask_attention_mask[-1] = False     # mask the final token
                end_index = mask_attention_mask.nonzero().max()
                mask_attention_mask[end_index + 1] = True   # for training only -- the token to predict next
                
                ##=== initialize input ids with position info ===##
                # select the initial token (at end_idx+1) to predict
                select_index = torch.randint(mask_attention_mask.nonzero().size(0), (1,)).to(self.args.device)
                # init input_ids
                cur_input_ids = input_ids[i].clone()
                cur_input_ids[mask_label_position_ids] = self.tokenizer.pad_token_id
                cur_input_ids[end_index + 1] = input_ids[i][select_index.item()]
                cur_input_ids = cur_input_ids[mask_attention_mask.nonzero().squeeze()]
                # init position ids
                cur_position_ids = raw_position_ids.clone()
                cur_position_ids[end_index + 1] = raw_position_ids[select_index.item()]
                cur_position_ids = cur_position_ids[mask_attention_mask.nonzero().squeeze()]
                
                ##=== rearrange unmasked tokens to get labels ===##
                cur_labels = labels[i][cur_position_ids]
                max_seq_len = max(cur_labels.size(-1), max_seq_len)
                # extract position ids to predict
                position_ids_to_predict = raw_position_ids[mask_label_position_ids]
                add_cur_labels = labels[i][position_ids_to_predict]
                
                ##=== replace input ids ===##
                if self.args.reconstruct:
                    # sample the threshold for reconstruction
                    replace_threshold = self.replace_ratio_generator.rvs(1)[0]
                    random_noise = torch.rand(cur_input_ids.size(-1), device=self.args.device)
                    # replace_ids = torch.logical_and(random_noise.lt(replace_threshold), cur_labels.ne(IGNORE_INDEX)).nonzero().squeeze(-1) # TODO
                    replace_ids = torch.logical_and(random_noise.le(1), cur_labels.ne(IGNORE_INDEX)).nonzero().squeeze(-1) # TODO
                    replace_threshold_list.append(torch.tensor(replace_threshold, device=self.args.device))
                    replace_ratio_list.append(torch.tensor(replace_ids.size(0) / max(1, labels[i, cur_position_ids].ne(IGNORE_INDEX).sum()), device=self.args.device))
                    if replace_ids.size(0) > 0:
                        # update the positions to predict
                        position_ids_to_predict = torch.cat((position_ids_to_predict, cur_position_ids[replace_ids]), dim=-1)
                        add_cur_labels = torch.cat((add_cur_labels, cur_labels[replace_ids]), dim=-1)
                        # replace input ids to reconstruct
                        if self.args.replace_with_prob < 1:
                            self.args.replace_with_prob = replace_threshold # TODO
                            # replace with probability < 1, so otherwise it should remain the same
                            random_noise = torch.rand(replace_ids.size(-1), device=self.args.device)
                            new_replace_ids = replace_ids[random_noise.lt(self.args.replace_with_prob).nonzero().squeeze(-1)]
                        # #################### sanity check ####################
                        # _cur_input_ids = cur_input_ids.clone()
                        # text = self.tokenizer.decode([_cur_input_ids[cur_position_ids.eq(idx).nonzero()[0].item()].item() if idx in cur_position_ids else 583 for idx in range(cur_position_ids.min().item(), cur_position_ids.max().item() + 1)])
                        # _cur_input_ids[new_replace_ids] = torch.randint(self.tokenizer.vocab_size, _cur_input_ids[new_replace_ids].size(), device=self.args.device)
                        # text1 = self.tokenizer.decode([_cur_input_ids[cur_position_ids.eq(idx).nonzero()[0].item()].item() if idx in cur_position_ids else 583 for idx in range(cur_position_ids.min().item(), cur_position_ids.max().item() + 1)])
                        # _cur_input_ids[replace_ids] = torch.randint(self.tokenizer.vocab_size, _cur_input_ids[replace_ids].size(), device=self.args.device)
                        # text2 = self.tokenizer.decode([_cur_input_ids[cur_position_ids.eq(idx).nonzero()[0].item()].item() if idx in cur_position_ids else 583 for idx in range(cur_position_ids.min().item(), cur_position_ids.max().item() + 1)])
                        # import ipdb; ipdb.set_trace()
                        # ######################################################
                        # replace input ids
                        if self.args.replace_with_prob < 1 and new_replace_ids.size(0) > 0:
                            cur_input_ids[new_replace_ids] = torch.randint(self.tokenizer.vocab_size, cur_input_ids[new_replace_ids].size(), device=self.args.device)
                        else:
                            cur_input_ids[replace_ids] = torch.randint(self.tokenizer.vocab_size, cur_input_ids[replace_ids].size(), device=self.args.device)
                max_add_positions = max(max_add_positions, position_ids_to_predict.size(-1))
                
                input_ids_list.append(cur_input_ids)
                labels_list.append(cur_labels)
                attention_mask_list.append(torch.ones_like(cur_input_ids, dtype=torch.bool))
                position_ids_list.append(cur_position_ids)
                positions_to_replace_list.append(torch.tensor(cur_input_ids.size(-1) - 1, device=self.args.device))
                position_ids_to_predict_list.append(position_ids_to_predict)
                add_labels_list.append(add_cur_labels)
            
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
                labels_list[i] = torch.cat((labels_list[i], tmp * IGNORE_INDEX), dim=-1).long()
                position_ids_list[i] = torch.cat((position_ids_list[i], tmp * 0), dim=-1).long()
                
                # pad position_ids_to_predict to be of the same number
                cur_seq_len = position_ids_to_predict_list[i].size(-1)
                to_add_len = max_add_positions - cur_seq_len                
                tmp = torch.ones((to_add_len,), dtype=torch.long, device=self.args.device)
                position_ids_to_predict_list[i] = torch.cat((position_ids_to_predict_list[i], tmp * 0), dim=-1).long()
                add_labels_list[i] = torch.cat((add_labels_list[i], tmp * IGNORE_INDEX), dim=-1).long()
                labels_list[i] = torch.cat((labels_list[i], add_labels_list[i]), dim=-1).long()
            
            # gather inputs into a batch
            all_input_ids.append(torch.stack(input_ids_list, dim=0))
            all_labels.append(torch.stack(labels_list, dim=0))
            all_attention_mask.append(torch.stack(attention_mask_list, dim=0))
            all_position_ids.append(torch.stack(position_ids_list, dim=0))
            all_positions_to_replace.append(torch.stack(positions_to_replace_list, dim=0).unsqueeze(-1))
            all_position_ids_to_prediction.append(torch.stack(position_ids_to_predict_list, dim=0))
        
        loss = self.loss(
            input_ids=torch.cat(all_input_ids, dim=0),
            labels=torch.cat(all_labels, dim=0),
            attention_mask=torch.cat(all_attention_mask, dim=0),
            position_ids=torch.cat(all_position_ids, dim=0),
            positions_to_replace=torch.cat(all_positions_to_replace, dim=0),
            position_ids_to_predict=torch.cat(all_position_ids_to_prediction, dim=0),
        )['loss']
        
        self.model.backward(loss)
        self.model.step()
        
        mask_threshold = torch.stack(mask_threshold_list, dim=0).mean()
        mask_ratio = torch.stack(mask_ratio_list, dim=0).mean()
        if self.args.reconstruct:
            replace_threshold = torch.stack(replace_threshold_list, dim=0).mean()
            replace_ratio = torch.stack(replace_ratio_list, dim=0).mean()
        
        loss = get_all_reduce_mean(loss)
        mask_threshold = get_all_reduce_mean(mask_threshold)
        mask_ratio = get_all_reduce_mean(mask_ratio)
        if self.args.reconstruct:
            replace_threshold = get_all_reduce_mean(replace_threshold)
            replace_ratio = get_all_reduce_mean(replace_ratio)
        
        dist.barrier()
        
        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
            'train/mask_threshold': mask_threshold.item(),
            'train/mask_ratio': mask_ratio.item(),
            'train/replace_threshold': replace_threshold.item() if self.args.reconstruct else 0,
            'train/replace_ratio': replace_ratio.item() if self.args.reconstruct else 0,
        }
