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
"""Trainer base class for supervised training."""

from __future__ import annotations

import abc
import argparse
from typing import Any, ClassVar
import re

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler
from transformers.integrations.deepspeed import HfDeepSpeedConfig, deepspeed_load_checkpoint

from oa_dag.configs import ADAM_BETAS
from oa_dag.datasets import TokenizedDataset, DummyDataset
from oa_dag.models import load_pretrained_models
from oa_dag.trainers.base import TrainerBase
from oa_dag.utils import get_optimizer_grouped_parameters, is_main_process, to_device, get_all_reduce_mean


class SupervisedTrainer(TrainerBase):
    """Trainer base class for supervised training.

    Abstract methods:
        loss: Compute supervised training loss.
        train_step: Perform a single training step.
    """

    TRAINING_TYPE: ClassVar[str] = 'supervised'
    DATASET_TYPE: ClassVar[type[TokenizedDataset]]
    EVAL_DATA_TYPE: ClassVar[type[TokenizedDataset]]
    MODEL_TYPE = AutoModelForCausalLM

    model: deepspeed.DeepSpeedEngine
    ds_config: dict[str, Any]

    extra_model_kwargs: dict[str, Any] | None = None
    extra_tokenizer_kwargs: dict[str, Any] | None = None

    def __init__(self, args: argparse.Namespace, ds_config: dict[str, Any]) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_config = ds_config
        self.global_step = 0

        self.init_models()
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_config is not None and self.ds_config['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_config)

        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='right',
            auto_model_type=self.MODEL_TYPE,
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs=self.extra_model_kwargs,
            auto_tokenizer_kwargs=self.extra_tokenizer_kwargs,
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        train_dataset = self.DATASET_TYPE(
            self.args.train_datasets,
            tokenizer=self.tokenizer,
            model_type=self.args.model_type,
            lazy_tokenization=self.args.lazy_tokenization,
        )

        if self.args.need_eval:
            if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
                train_dataset, eval_dataset = train_dataset.split_train_test(
                    split_ratio=self.args.eval_split_ratio,
                )
            elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
                eval_dataset = self.EVAL_DATA_TYPE(
                    self.args.eval_datasets,
                    tokenizer=self.tokenizer,
                    model_type=self.args.model_type,
                    lazy_tokenization=True,
                )
            else:
                raise ValueError('Either `eval_datasets` or `eval_split_ratio` should be provided.')

            self.eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=True),
                batch_size=self.args.per_device_eval_batch_size,
            )
        else:
            self.eval_dataloader = None

        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True),
            batch_size=self.args.per_device_train_batch_size,
        )
        if self.args.no_noise:
            self.no_noise_train_dataloader = DataLoader(
                train_dataset,
                collate_fn=train_dataset.get_collator(),
                sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True, seed=self.args.seed + 1),
                batch_size=self.args.per_device_train_batch_size,
            )
        else:
            self.no_noise_train_dataloader = DataLoader(DummyDataset(len(self.train_dataloader)))

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.args.total_training_steps = len(self.train_dataloader) * self.args.epochs
        if self.args.no_noise and not self.args.no_denoise:
            self.args.gradient_accumulation_steps *= 2
            self.ds_config['train_batch_size'] *= 2
            self.ds_config['gradient_accumulation_steps'] *= 2
            self.args.total_training_steps *= 2

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            self.model,
            self.args.weight_decay,
        )
        if (
            self.ds_config['zero_optimization'].get('offload_optimizer', {}).get('device', 'none')
            != 'none'
        ):
            optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                betas=ADAM_BETAS,
            )
        else:
            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                betas=ADAM_BETAS,
            )

        lr_scheduler_update_steps = self.args.total_training_steps // self.ds_config['gradient_accumulation_steps']
        num_warmup_steps = int(lr_scheduler_update_steps * self.args.lr_warmup_ratio)
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=lr_scheduler_update_steps,
        )

        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            args=self.args,
            config=self.ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
        )
        
        if self.args.resume_from_ckpt is not None:
            deepspeed_load_checkpoint(self.model, self.args.resume_from_ckpt)

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @abc.abstractmethod
    def loss(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute supervised training loss."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform a single training step."""
        raise NotImplementedError
    
    def create_oa_batch(self, batch: TokenizedDataset) -> dict[str, Any]:
        return batch
    
    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')
        
        progress_bar = tqdm(
            total=self.args.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.args.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )
        
        steps_trained_in_current_epoch, epochs_trained = 0, 0
        if self.args.resume_from_ckpt is not None:
            steps_trained_in_current_epoch = self.model.global_steps * self.args.gradient_accumulation_steps // (2 if self.args.no_noise and not self.args.no_denoise else 1)
            if steps_trained_in_current_epoch > 0:
                progress_bar.update(steps_trained_in_current_epoch)
            self.global_step = steps_trained_in_current_epoch
            if not steps_trained_in_current_epoch:
                _step = int(re.search(r'\b\d+\b', self.args.resume_from_ckpt)[0])
                steps_trained_in_current_epoch = _step
                progress_bar.update(steps_trained_in_current_epoch)
                self.global_step = steps_trained_in_current_epoch
            epochs_trained = steps_trained_in_current_epoch // len(self.train_dataloader)
            steps_trained_in_current_epoch %= len(self.train_dataloader)

        if self.args.need_eval:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.logger.log(self.eval(), step=0)

        for epoch in range(self.args.epochs):
            self.train_dataloader.sampler.set_epoch(epoch)
            if epoch < epochs_trained: continue

            for batch, no_noise_batch in zip(self.train_dataloader, self.no_noise_train_dataloader):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                
                ##=== generate batch (with noise) ===##
                self.set_eval()
                if self.args.no_noise:
                    if not self.args.no_denoise:
                        oa_batch = self.create_oa_batch(to_device(batch, self.args.device), force_replace=True)
                        # torch.cuda.empty_cache()
                    noa_batch = self.create_oa_batch(to_device(no_noise_batch, self.args.device), fixed_replace_threshold=0.0)
                else:
                    oa_batch = self.create_oa_batch(to_device(batch, self.args.device))
                # torch.cuda.empty_cache()
                
                ##=== training ===##
                self.set_train()
                if not self.args.no_denoise:
                    info = self.train_step(**oa_batch)
                    # torch.cuda.empty_cache()
                if self.args.no_noise:
                    noa_info = self.train_step(**noa_batch)
                    # torch.cuda.empty_cache()
                    if self.args.no_denoise:
                        info = noa_info
                get_accelerator().empty_cache()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.args.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)
                if self.args.no_noise:
                    self.logger.log(noa_info, step=self.global_step)

                if self.global_step % self.args.save_interval == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    # self.model.save_checkpoint(self.args.output_dir, tag=self.global_step)
                    self.save(global_steps=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.args.need_eval
                    and self.args.eval_strategy == 'steps'
                    and self.global_step % self.args.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)

            if self.args.need_eval and self.args.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)

            self.save(global_steps=self.global_step)
            
            self.model.tput_timer.update_epoch_count()

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for model."""
        if mode:
            self.model.train()
            if self.args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.model.eval()
            if self.args.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()
