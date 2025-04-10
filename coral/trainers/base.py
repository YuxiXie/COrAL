"""Trainer base class."""

from __future__ import annotations

import abc
import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, ClassVar

import deepspeed
import torch.distributed as dist
from transformers import CONFIG_NAME, WEIGHTS_NAME, PreTrainedModel, PreTrainedTokenizerBase

from coral.logger import Logger
from coral.utils import is_main_process


class TrainerBase(metaclass=abc.ABCMeta):
    """Trainer base class.

    Abstract methods:
        init_models: Initialize model and tokenizer.
        init_datasets: Initialize training and evaluation datasets.
        init_engines: Initialize DeepSpeed engines.
        train: Train model.
        set_train: Set training mode for all models.
    """

    TRAINING_TYPE: ClassVar[str]

    tokenizer: PreTrainedTokenizerBase

    args: argparse.Namespace
    logger: Logger

    @abc.abstractmethod
    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        raise NotImplementedError

    @abc.abstractmethod
    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        raise NotImplementedError

    @abc.abstractmethod
    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        raise NotImplementedError

    def init_logger(self) -> None:
        """Set logger."""
        if self.args.log_type is None:
            self.logger = Logger(config=self.args)
            return

        time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        self.args.log_dir = self.args.log_dir or self.args.output_dir
        self.args.log_project = self.args.log_project or 'safe-rlhf'
        model_type = self.args.output_dir.strip('/').split('/')[-1]
        # self.args.log_run_name = self.args.log_run_name or f'{self.TRAINING_TYPE}-{model_type}-{time}'
        self.args.log_run_name = f'{self.args.log_run_name}-{model_type}-{time}'

        self.logger = Logger(
            log_type=self.args.log_type,
            log_dir=self.args.log_dir,
            log_project=self.args.log_project,
            log_run_name=self.args.log_run_name,
            config=self.args,
        )

    @abc.abstractmethod
    def train(self) -> None:
        """Train model."""
        raise NotImplementedError

    def eval(self) -> dict[str, Any]:
        """Evaluate model."""
        return {}

    @abc.abstractmethod
    def set_train(self, mode: bool = True) -> None:
        """Set training mode for all models."""
        raise NotImplementedError

    def set_eval(self) -> None:
        """Set model to evaluation mode."""
        self.set_train(mode=False)

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        ds_config: dict[str, Any] | None = None,
        global_steps: int = -1,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        dist.barrier()

        if model is None:
            model = self.model  # pylint: disable=no-member
        if ds_config is None:
            ds_config = self.ds_config  # pylint: disable=no-member
        
        output_dir = self.args.output_dir
        if global_steps > 0:
            output_dir = os.path.join(output_dir, f'checkpoint-{global_steps}')
            os.makedirs(output_dir, exist_ok=True)

        self.logger.print(f'Saving model to "{output_dir}" ...')

        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        model_to_save: PreTrainedModel = getattr(model, 'module', model)
        if is_main_process():
            model_to_save.config.to_json_file(output_config_file)
            self.tokenizer.save_pretrained(output_dir)

        if self.args.save_16bit:
            self.logger.print('Saving 16-bit model...')
            model.save_16bit_model(output_dir)
        else:
            # Save model checkpoint
            if ds_config['zero_optimization']['stage'] >= 2:
                self.logger.print('Saving DeepSpeed Checkpoints...')
                model.save_checkpoint(output_dir)
                self.logger.print('Converting DeepSpeed Checkpoints to Hugging Face format...')
                if is_main_process():
                    subprocess.check_call(
                        [sys.executable, 'zero_to_fp32.py', '.', WEIGHTS_NAME],  # noqa: S603
                        cwd=output_dir,
                    )
                dist.barrier()
            else:
                self.logger.print('Saving Hugging Face Checkpoints...')
                if is_main_process():
                    model_to_save.save_pretrained(output_dir, is_main_process=True)

        self.logger.print('Model saved!')
