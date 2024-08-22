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
"""Raw datasets."""

from oa_dag.datasets.raw.alpaca import AlpacaDataset
from oa_dag.datasets.raw.firefly import FireflyDataset
from oa_dag.datasets.raw.hh_rlhf import (
    HhRLHFDialogueDataset,
    HhRLHFHarmlessDialogueDataset,
    HhRLHFHelpfulDialogueDataset,
)
from oa_dag.datasets.raw.moss import MOSS002SFT, MOSS003SFT
from oa_dag.datasets.raw.safe_rlhf import (
    SafeRLHF10KTrainDataset,
    SafeRLHF30KTestDataset,
    SafeRLHF30KTrainDataset,
    SafeRLHFDataset,
    SafeRLHFTestDataset,
    SafeRLHFTrainDataset,
)

from oa_dag.datasets.raw.diy_distilled import DistrilledDataset, DistrilledDatasetEval
from oa_dag.datasets.raw.math import MetaMathTrainDataset, MetaMathValidDataset, GSM8KDataset, MATHDataset


__all__ = [
    'AlpacaDataset',
    'FireflyDataset',
    'HhRLHFDialogueDataset',
    'HhRLHFHarmlessDialogueDataset',
    'HhRLHFHelpfulDialogueDataset',
    'MOSS002SFT',
    'MOSS003SFT',
    'SafeRLHFDataset',
    'SafeRLHFTrainDataset',
    'SafeRLHFTestDataset',
    'SafeRLHF30KTrainDataset',
    'SafeRLHF30KTestDataset',
    'SafeRLHF10KTrainDataset',
    
    'DistrilledDataset',
    'DistrilledDatasetEval',
    
    'MetaMathTrainDataset',
    'MetaMathValidDataset',
    'GSM8KDataset',
    'MATHDataset',
]
