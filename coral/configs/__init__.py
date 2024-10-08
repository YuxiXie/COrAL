"""Configurations and constants."""

from coral.configs import constants
from coral.configs.constants import *  # noqa: F403
from coral.configs.deepspeed_config import (
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
