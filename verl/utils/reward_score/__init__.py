# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from .multiply import compute_score as multiply_compute_score
from .countdown import compute_score as countdown_compute_score
from .gsm8k import compute_score as gsm8k_compute_score
from .math import compute_score as math_compute_score
from .qa_em import compute_score as qa_em_compute_score
from .agentgym import compute_score as agentgym_compute_score

SUPPORTED_REWARD_SCORE_FNS = {
    'multiply': multiply_compute_score,
    'countdown': countdown_compute_score,
    'gsm8k': gsm8k_compute_score,
    'math': math_compute_score,
    'qa_em': qa_em_compute_score,
    'agentgym': agentgym_compute_score,
}
