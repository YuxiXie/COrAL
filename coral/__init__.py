from coral import algorithms, configs, datasets, models, trainers, utils
from coral.algorithms import *  # noqa: F403
from coral.configs import *  # noqa: F403
from coral.datasets import *  # noqa: F403
from coral.models import *  # noqa: F403
from coral.trainers import *  # noqa: F403
from coral.utils import *  # noqa: F403
from coral.version import __version__


__all__ = [
    *algorithms.__all__,
    *configs.__all__,
    *datasets.__all__,
    *models.__all__,
    *trainers.__all__,
    *utils.__all__,
]
