# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional
import uuid

from torch import Tensor


class FairseqIncrementalState(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4()) # 由uuid.uuid4()生成一组唯一码

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key) # 由一组唯一码和当前名称组成的新名称

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key) # 返回由一组唯一码和当前名称(默认为:'attn_state')组成的新名称
        if incremental_state is None or full_key not in incremental_state: # 由一组唯一码和当前名称(默认为:'attn_state')组成的新名称不在字典内,故返回None
            return None
        return incremental_state[full_key]  # 返回一组唯一码和当前名称(默认为:'attn_state')组成的新名称对应的incremental_state值

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module. 用于为nn.Module设置增量状态的辅助器"""
        if incremental_state is not None: # value为传入的saved_state,存储到当目前解码时间步为止(包括当前)的'prev_key'/'prev_value'/'prev_key_padding_mask'字典
            full_key = self._get_full_incremental_state_key(key)  # 返回由一组唯一码和当前名称(默认为:'attn_state')组成的新名称
            incremental_state[full_key] = value
            # 将{唯一码+'attn_state':到当目前解码时间步为止(包括当前)的'prev_key'/'prev_value'/'prev_key_padding_mask'}存入incremental_state字典中
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(b for b in cls.__bases__ if b != FairseqIncrementalState)
    return cls
