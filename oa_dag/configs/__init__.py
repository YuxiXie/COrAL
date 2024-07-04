"""Configurations and constants."""

from oa_dag.configs import constants
from oa_dag.configs.constants import *  # noqa: F403
from oa_dag.configs.deepspeed_config import (
    TEMPLATE_DIR,
    get_deepspeed_eval_config,
    get_deepspeed_train_config,
)


__all__ = [
    *constants.__all__,
    'TEMPLATE_DIR',
    'get_deepspeed_eval_config',
    'get_deepspeed_train_config',
]
