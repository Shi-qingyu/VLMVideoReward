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

from importlib.metadata import version, PackageNotFoundError


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = 'vllm'
package_version = get_version(package_name)


def _patch_transformers_aimv2_duplicate_register():
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except Exception:
        return

    register_fn = CONFIG_MAPPING.register
    if getattr(register_fn, "_verl_aimv2_patched", False):
        return

    def safe_register(key, value, exist_ok=False):
        try:
            return register_fn(key, value, exist_ok=exist_ok)
        except ValueError as exc:
            if key == "aimv2" and "already used by a Transformers config" in str(exc):
                return None
            raise

    safe_register._verl_aimv2_patched = True
    CONFIG_MAPPING.register = safe_register


_patch_transformers_aimv2_duplicate_register()

if package_version <= '0.6.3':
    vllm_mode = 'customized'
    from .vllm_rollout import vLLMRollout
    from .fire_vllm_rollout import FIREvLLMRollout
else:
    vllm_mode = 'spmd'
    from .vllm_rollout_spmd import vLLMRollout, vLLMRollout_MultiTurn_ToolCall, vLLMRollout_MultiTurn_ResizeImage, vLLMAsyncRollout
