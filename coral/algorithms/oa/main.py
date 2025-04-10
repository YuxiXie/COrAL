import sys
sys.path.append('/local/home/yuxi_xie/Projects/COrAsL')

import argparse

import deepspeed
import torch
import torch.distributed as dist
from transformers import SchedulerType
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available

from coral.configs import get_deepspeed_train_config, get_deepspeed_eval_config
from coral.datasets import parse_dataset
from coral.algorithms.oa.trainer import OASupervisedFinetuneTrainer
from coral.logger import set_logger_level
from coral.utils import seed_everything, str2bool


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='deepspeed --module coral.finetune.deepspeed',
        description='Supervised finetune a model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='Path to the model checkpoint or its name.',
        required=True,
    )
    model_parser.add_argument(
        '--resume_from_ckpt',
        type=str,
        default=None,
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )
    model_parser.add_argument(
        '--trust_remote_code',
        type=str2bool,
        default=False,
        help='Whether to trust the remote code.',
    )
    model_parser.add_argument(
        '--additional_layer',
        default=False,
        action='store_true',
    )

    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--model_type',
        type=str,
        default='mistral-instruct',
    )
    dataset_parser.add_argument(
        '--train_datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
        help='Dataset name(s) registered in the raw dataset.',
        required=True,
    )
    dataset_parser.add_argument(
        '--eval_datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
        help='Dataset name(s) registered in the raw dataset.',
    )
    dataset_parser.add_argument(
        '--not_lazy_tokenization',
        default=False,
        action='store_true',
    )

    # Training
    training_parser = parser.add_argument_group('training')
    training_parser.add_argument(
        '--verbal_training',
        default=False,
        action='store_true',
    )
    training_parser.add_argument(
        '--sample_from_near',
        default=False,
        action='store_true',
    )
    training_parser.add_argument(
        '--sample_from_future',
        default=False,
        action='store_true',
    )
    training_parser.add_argument(
        '--max_corrupt_context_size',
        default=4,
        type=int,
    )
    training_parser.add_argument(
        '--multi_context_granularity',
        default=False,
        action='store_true',
    )
    training_parser.add_argument(
        '--corrupt_context_num',
        default=8,
        type=int,
    )
    training_parser.add_argument(
        '--context_corrupt',
        default=False,
        action='store_true',
    )
    training_parser.add_argument(
        '--sample_to_replace',
        default=False,
        action='store_true',
    )
    training_parser.add_argument(
        '--no_noise_coef',
        default=1,
        type=float,
    )
    training_parser.add_argument(
        '--no_noise',
        default=False,
        action='store_true',
    )
    training_parser.add_argument(
        '--no_denoise',
        default=False,
        action='store_true',
    )
    # - for denoising
    training_parser.add_argument(
        '--pred_gap',
        default=0,
        type=int,
    )
    training_parser.add_argument(
        '--context_window',
        default=8,
        type=int,
    )
    training_parser.add_argument(
        '--n_back_pred',
        default=1,
        type=float,
    )
    training_parser.add_argument(
        '--replace_ratio_min',
        type=float,
        default=0.0,
    )
    training_parser.add_argument(
        '--replace_ratio_max',
        type=float,
        default=0.2,
    )
    training_parser.add_argument(
        '--replace_ratio_mu',
        type=float,
        default=0.05,
    )
    training_parser.add_argument(
        '--replace_ratio_std',
        type=float,
        default=0.1,
    )
    # - for partial fine-tuning
    training_parser.add_argument(
        '--tune_final_layer_only',
        default=False,
        action='store_true',
    )
    training_parser.add_argument(
        '--tune_backbone_only',
        default=False,
        action='store_true',
    )
    training_parser.add_argument(
        '--tune_lm_head',
        default=False,
        action='store_true',
    )
    training_parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Total number of training epochs to perform.',
    )
    training_parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the training dataloader.',
    )
    training_parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the evaluation dataloader.',
    )
    training_parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )
    training_parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for actor model.',
    )
    training_parser.add_argument(
        '--lr',
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Initial learning rate (after the potential warmup period) to use.',
    )
    training_parser.add_argument(
        '--lr_scheduler_type',
        type=SchedulerType,
        default='cosine',
        help='The scheduler type to use.',
        choices=[
            'linear',
            'cosine',
            'cosine_with_restarts',
            'polynomial',
            'constant',
            'constant_with_warmup',
        ],
    )
    training_parser.add_argument(
        '--lr_warmup_ratio',
        type=float,
        default=0.0,
        help='Ratio of warm steps over total training steps for the lr scheduler.',
    )
    training_parser.add_argument(
        '--weight_decay',
        type=float,
        default=1.0e-6,
        help='Weight decay to use.',
    )
    training_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='A seed for reproducible training.',
    )
    training_parser.add_argument(
        '--fp16',
        type=str2bool,
        default=False,
        help='Whether to use float16 precision.',
    )
    training_parser.add_argument(
        '--bf16',
        type=str2bool,
        default=False,
        help='Whether to use bfloat16 precision.',
    )
    training_parser.add_argument(
        '--tf32',
        type=str2bool,
        default=None,
        help='Whether to use tf32 mix precision.',
    )

    # Evaluation
    evaluation_parser = parser.add_argument_group('evaluation')
    evaluation_parser.add_argument(
        '--epsilon',
        type=float,
        default=0.1,
    )
    evaluation_parser.add_argument(
        '--eval_forward_size',
        type=int,
        default=4,
    )
    evaluation_parser.add_argument(
        '--eval_backward_size',
        type=int,
        default=4,
    )
    evaluation_parser.add_argument(
        '--skip_verify',
        default=False,
        action='store_true',
    )
    evaluation_parser.add_argument(
        '--decoding_occurance_threshold',
        type=int,
        default=3,
    )
    evaluation_parser.add_argument(
        '--left2right',
        default=False,
        action='store_true',
    )
    evaluation_parser.add_argument(
        '--add_denoising',
        default=False,
        action='store_true',
    )
    evaluation_parser.add_argument(
        '--decoding_block_size',
        type=int,
        default=16,
    )
    evaluation_parser.add_argument(
        '--result_fname',
        type=str,
        default='results',
    )
    evaluation_parser.add_argument(
        '--max_n_tokens_per_step',
        type=int,
        default=1,
    )
    evaluation_parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
    )
    evaluation_parser.add_argument(
        '--seq_temperature',
        type=float,
        default=1.0,
    )
    evaluation_parser.add_argument(
        '--top_p',
        type=float,
        default=None,
    )
    evaluation_parser.add_argument(
        '--top_k',
        type=int,
        default=None,
    )
    evaluation_parser.add_argument(
        '--eval_strategy',
        type=str,
        default='epoch',
        help='The evaluation strategy to adopt.',
        choices=['epoch', 'steps'],
    )
    evaluation_parser.add_argument(
        '--eval_interval',
        type=int,
        default=1000000,
        help='The interval to evaluate the model.',
    )
    evaluation_parser.add_argument(
        '--need_eval',
        default=False,
        help='Whether to evaluate the model during training.',
        action='store_true',
    )
    evaluation_parser.add_argument(
        '--verbal_decoding',
        default=False,
        action='store_true',
    )
    evaluation_parser.add_argument(
        '--do_decoding',
        default=False,
        action='store_true',
    )
    evaluation_parser.add_argument(
        '--eval_replace_ratio',
        type=float,
        default=0.0,
    )
    evaluation_parser.add_argument(
        '--eval_split_ratio',
        type=float,
        default=None,
        help='The split ratio of the evaluation dataset.',
    )

    # Logging
    logging_parser = parser.add_argument_group('logging')
    logging_parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the model.',
    )
    logging_parser.add_argument(
        '--log_type',
        type=str,
        help='The type of logging.',
        default='wandb',
        choices=['wandb', 'tensorboard'],
    )
    logging_parser.add_argument(
        '--log_dir',
        type=str,
        help='The directory to store the logs.',
        default=None,
    )
    logging_parser.add_argument(
        '--log_project',
        type=str,
        help='The project name for logging.',
        default=None,
    )
    logging_parser.add_argument(
        '--log_run_name',
        type=str,
        help='The run name for logging.',
        default=None,
    )
    logging_parser.add_argument(
        '--save_16bit',
        action='store_true',
        help='Whether to save the model in 16-bit precision.',
    )
    logging_parser.add_argument(
        '--save_interval',
        type=int,
        default=1000000,
        help='The interval to save the model.',
    )

    # DeepSpeed
    deepspeed_parser = parser.add_argument_group('deepspeed')
    deepspeed_parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training on GPUs',
    )
    deepspeed_parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='ZeRO optimization stage for models.',
    )
    deepspeed_parser.add_argument(
        '--offload',
        type=str,
        default='none',
        choices=['none', 'parameter', 'optimizer', 'all'],
        help='Offload parameters and/or optimizer states to CPU.',
    )
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    if args.local_rank == -1:
        parser.error('`local_rank` not set, please use DeepSpeed launcher to run this script.')
    if args.fp16 and args.bf16:
        parser.error('Cannot use both bf16 and fp16 precision.')
    if args.bf16 and not is_torch_bf16_gpu_available():
        parser.error(
            'bf16 precision is not supported on this GPU. '
            'Please disable `--bf16` flag or use another precision flag (e.g., `--fp16`).',
        )
    if args.tf32 is not None and is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
    
    if not args.tune_final_layer_only and not args.tune_backbone_only:
        args.tune_lm_head = True
    args.lazy_tokenization = not args.not_lazy_tokenization

    return args


def main() -> None:
    """Main training routine."""
    args = parse_arguments()

    from datetime import timedelta
    deepspeed.init_distributed(
        # dist_backend='nccl' if dist.is_nccl_available() else 'gloo',
        # timeout=timedelta(seconds=7200000)
    )

    args.global_rank = dist.get_rank()
    args.device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(args.device)
    seed_everything(args.seed)
    set_logger_level()

    dist.barrier()

    ds_config = get_deepspeed_train_config(
        micro_batch_size_per_gpu=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        offload=args.offload,
        stage=args.zero_stage,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    if args.need_eval:
        ds_eval_config = get_deepspeed_eval_config(
            stage=args.zero_stage,
            offload=args.offload,
            fp16=args.fp16,
            bf16=args.bf16,
        )
    else:
        ds_eval_config = None

    trainer = OASupervisedFinetuneTrainer(args, ds_config, ds_eval_config)
    trainer.train()
    # trainer.save()


if __name__ == '__main__':
    main()
