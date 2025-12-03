# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Strategies using PyTorch distributed.checkpoint as an underlying format. """
import io
import os
import pickle
import warnings
from collections import ChainMap, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import product
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast, get_args

import torch
from packaging.version import Version as PkgVersion
from torch.distributed import checkpoint
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import Shard
from torch.distributed._shard.sharded_tensor import ShardedTensor as TorchShardedTensor
from torch.distributed._shard.sharded_tensor import ShardedTensorMetadata, TensorProperties
from torch.distributed.checkpoint import (
    BytesStorageMetadata,
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    FileSystemWriter,
    LoadPlan,
    Metadata,
    ReadItem,
    SavePlan,
    TensorStorageMetadata,
    WriteItem,
)
from torch.distributed.checkpoint.planner import SavePlanner, WriteItemType
from torch.distributed.checkpoint._nested_dict import FLATTEN_MAPPING, unflatten_state_dict
from torch.distributed.checkpoint._traverse import OBJ_PATH, traverse_state_dict
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner_helpers import _create_write_items

from ...utils import get_torch_version, is_torch_min_version
from ..core import CheckpointingException
from ..dict_utils import nested_values
from ..mapping import (
    ShardedBase,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    StateDict,
    is_main_replica,
)
from .async_utils import AsyncRequest
from .base import (
    AsyncSaveShardedStrategy,
    LoadShardedStrategy,
    StrategyAction,
    register_default_strategy,
)
from .cached_metadata_filesystem_reader import CachedMetadataFileSystemReader
from .eccheck_manager import ECCHECKManager
from .filesystem_async import FileSystemWriterAsync
from .resharding import (
    TensorReformulationMetadata,
    apply_nd_flattened_tensors_reformulation,
    is_nd_flattened_tensor,
    nd_flattened_tensor_reformulated_global_shape,
    restore_nd_flattened_tensors_formulation,
)
from .state_dict_saver import save_state_dict_async_finalize, save_state_dict_async_plan
from .state_dict_decomposer import DecomposedStateDict, TensorMetadata
from time import time

try:
    if not torch.cuda.is_available():
        raise ImportError
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    from torch.distributed._tensor import DTensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from megatron.core.msc_utils import MultiStorageClientFeature

MSC_PREFIX = "msc://"

_metadata_fn: str = ".metadata"

_TORCH_STRATEGIES_REGISTERED = False


def register_default_torch_strategies():
    global _TORCH_STRATEGIES_REGISTERED

    if _TORCH_STRATEGIES_REGISTERED:
        return
    """Register default strategies related to PyT Distributed backend."""
    register_default_strategy(
        StrategyAction.LOAD_SHARDED, 'torch_dist', 1, TorchDistLoadShardedStrategy()
    )
    register_default_strategy(
        StrategyAction.SAVE_SHARDED, 'torch_dist', 1, TorchDistSaveShardedStrategy('torch_dist', 1)
    )
    _TORCH_STRATEGIES_REGISTERED = True


logger = getLogger(__name__)


def flatten_state_dict(
    state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, Dict[str, OBJ_PATH]]:
    """Flattens state dict into a single level dict.

    It's a copy of torch.distributed.checkpoint._nested_dict.flatten_state_dict
    which also accepts ShardedBase tensors as terminal objects

    Args:
        state_dict (ShardedStateDict): state dict to be flattened

    Returns (tuple): flattened state dict and a mapping allowing to recreate the original one

    """
    flattened = {}
    mappings = {}

    def flat_copy(path: OBJ_PATH, value: Any) -> None:
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    traverse_state_dict(state_dict, flat_copy, lambda x: isinstance(x, (torch.Tensor, ShardedBase)))
    return flattened, mappings


def sharded_tensor_to_torch_sharded_tensor(
    sh_tens: List[ShardedTensor],
    rank: Optional[int] = None,
    load_legacy_1d_flatten_tensors: bool = False,
) -> TorchShardedTensor:
    """Convert MCore ShardedTensor to PyT ShardedTensor. PyT requires information about all chunks.

    On high-level, this function follows the logic of
    torch.distributed.fsdp._shard_utils._create_chunk_sharded_tensor.
    Additionally, it saves `prepend_axis_num` and `has_flattened_range` (specific to MCore)
    as attributes for further restoration in `_unwrap_pyt_sharded_tensor`.

    NOTE: this function assumes regular (grid) sharding of the MCore ShardedTensor.
    The only local irregularities could be introduced with a `flattened_range` attribute.

    This function handles 2 different type of ShardedTensors:
    1. Non-flat regular ShardedTensors (`not has_flattened_range`)
    2. N-D flattened ShardedTensors (`has_flattened_range`)

    (1) type are saved according to their original shape.
    Type (2) however requires global shape adjustment for efficiency:
    we treat [X, Y, Z] global shape tensor with local shape [x, y, z]
    as a [X // x, Y // y, Z // z, x * y * z] tensor with last axis
    partitioned according to `flattened_range` slices.
    This will need special handling while resharding.

    Args:
        sh_tens (List[ShardedTensor]): list of sharded tensors to convert
        rank (int, optional): current process rank passed to PyT ShardedTensor.
            If None, assumes rank in the default pg.
        load_legacy_1d_flatten_tensors (bool, optional): flag indicating if 1-D flattened tensors
            should be loaded in a legacy way. Defaults to False.

    Returns (TorchShardedTensor): PyT ShardedTensor containing all passed shards.

    """
    if rank is None:
        rank = torch.distributed.get_rank()

    some_sh_ten = sh_tens[0]
    has_flattened_range = some_sh_ten.flattened_range is not None

    for sh_ten in sh_tens:
        assert (sh_ten.flattened_range is not None) == has_flattened_range, sh_tens
        if not sh_ten.data.is_contiguous():
            sh_ten.data = sh_ten.data.contiguous()

    if load_legacy_1d_flatten_tensors and len(some_sh_ten.global_shape) == 1:
        # Legacy 1-D flattened tensors are loaded as non-flat regular ShardedTensors
        has_flattened_range = False

    local_global_offsets = {}

    prepend_axis_num = sh_tens[0].prepend_axis_num
    # Determine local shards according to tensor type (see docs)
    if has_flattened_range:
        # Type (3) case: N-D flattened ShardedTensors
        for sh_ten in sh_tens:
            local_global_offsets.setdefault(sh_ten.local_chunk_offset_in_global(), []).append(
                sh_ten
            )
            assert sh_ten.data.ndim == 1, sh_ten
            sh_ten.data = sh_ten.data.view((1,) * len(sh_ten.global_shape) + (-1,))

        # Global shape reformulation:
        global_shape = nd_flattened_tensor_reformulated_global_shape(some_sh_ten)
        offsets_shape = (1,) * len(
            some_sh_ten.global_shape
        )  # reformulated global shape has shape equal ti number of local chunks

        local_shards = [
            Shard.from_tensor_and_offsets(
                sh_ten.data,
                list(
                    sh_ten.local_chunk_offset_in_global() + (sh_ten.flattened_range.start,)
                ),  # additional flattened offset
                rank,
            )
            for sh_ten in sh_tens
        ]
    else:
        # Type (1) case: non-flat regular ShardedTensors
        for sh_ten in sh_tens:
            local_global_offsets.setdefault(sh_ten.global_offset, []).append(sh_ten)
            sh_ten.data = sh_ten.data.view(
                (1,) * prepend_axis_num + sh_ten.local_shape
            )  # adjust to prepended_axis_num

        global_shape = some_sh_ten.global_shape
        offsets_shape = some_sh_ten.data.shape  # includes prepended axes

        local_shards = [
            Shard.from_tensor_and_offsets(
                sh_ten.data, list(sh_ten.global_offset), rank  # simple case
            )
            for sh_ten in sh_tens
        ]

    # Create a ShardedTensor without invoking communication. Determine global shards
    world_size = torch.distributed.get_world_size()
    shard_metadata = []
    # NOTE: here we assume a regular grid of shards
    for fragment_offsets in product(*map(range, some_sh_ten.axis_fragmentations)):
        offset = tuple(map(lambda x: x[0] * x[1], zip(fragment_offsets, offsets_shape)))
        if offset in local_global_offsets:
            # local shard
            placement = f"rank:{rank}/cuda"
            for sh_ten in local_global_offsets[offset]:
                if has_flattened_range:
                    assert offset == sh_ten.local_chunk_offset_in_global()
                    # This is not an actual offset, but an offset of the whole shard
                    # This is needed for a PyT Dist internal integrity check
                    offset = sh_ten.local_chunk_offset_in_global() + (0,)
                    size = (1,) * len(offsets_shape) + global_shape[-1:]
                else:
                    size = sh_ten.data.shape
                shard_metadata.append(ShardMetadata(offset, size, placement))

        else:
            # pylint: disable=line-too-long
            # for shards from other ranks we provide simplistic data - this information will be discarded
            # during TorchShardedTensor._init_from_local_shards_and_global_metadata call.
            # Due to a bug in PyT 24.05 container we must specify some concrete rank within a world size.
            # The exact rank doesn't matter as long as it's different than my rank - hence (rank + 1) % WS.
            placement = f"rank:{(rank + 1) % world_size}/cuda"
            if has_flattened_range:
                offset = offset + (0,)
                size = (1,) * len(offsets_shape) + global_shape[-1:]
            else:
                size = offsets_shape
            shard_metadata.append(ShardMetadata(offset, size, placement))

    tensor = some_sh_ten.data
    sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=shard_metadata,
        size=torch.Size(global_shape),
        tensor_properties=TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        ),
    )
    pyt_sh_ten = TorchShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards, sharded_tensor_metadata=sharded_tensor_metadata, process_group=None
    )
    # Store MCore related data as PyTShardedTensor attribute.
    # This won't be stored in the checkpoint, only for runtime purposes
    pyt_sh_ten.mcore_sh_ten = sh_ten.without_data()
    pyt_sh_ten.mcore_metadata = {}
    if has_flattened_range:
        pyt_sh_ten.mcore_metadata['nd_reformulated_orig_global_shape'] = sh_ten.global_shape
    return pyt_sh_ten


def mcore_to_pyt_state_dict(
    state_dict: Dict[str, List[ShardedBase]],
    is_loading: bool = False,
    init_device: torch.device = torch.device("cpu"),
    load_legacy_1d_flatten_tensors: bool = False,
) -> Dict[str, Union[TorchShardedTensor, io.BytesIO]]:
    """Convert state dict with ShardedTensors and ShardedObjects
    to state dict compatible with PyT Dist format.

    Operates in-place and returns the original state dict.

    Args:
        state_dict (Dict[str, List[ShardedBase]]): flattened state dict, where values
            are lists of either ShardedTensor or ShardedObjects.
        is_loading (bool, optional): flag indicating if loading or saving. Defaults to False.
        init_device (torch.device, optional): device to initialize potentially missing tensors
            during loading. Defaults to 'cpu'.

    Returns (Dict[str, Union[TorchShardedTensor, io.BytesIO]]): original dictionary with values
        converted either into PyT ShardedTensors or io.BytesIO.

    """
    rank = torch.distributed.get_rank()
    pyt_state_dict = {}

    def _mcore_to_torch_sharded_tensor(sh_tens: List[ShardedTensor]) -> TorchShardedTensor:
        """Build a PyT ShardedTensor from given shards.

        During loading:
        - if data is None, initialize it with an empty tensor (will be used to copy the data into)
        - if `allow_shape_mismatch` is True, the data is initialized with zeros
            prior to loading (not all parts of the tensor will be read from the checkpoint)
        """
        assert all(isinstance(sh_ten, ShardedTensor) for sh_ten in sh_tens), sh_tens
        for sh_ten in sh_tens:
            if sh_ten.data is None:
                if is_loading:
                    sh_ten.init_data(
                        init_device,
                        init_fn=torch.zeros if sh_ten.allow_shape_mismatch else torch.empty,
                    )
                else:
                    raise CheckpointingException(f'`data` attr is None for {sh_ten}')
            else:
                sh_ten.data = sh_ten.data.detach()
                if sh_ten.allow_shape_mismatch and is_loading:
                    sh_ten.data.zero_()

        torch_sh_ten = sharded_tensor_to_torch_sharded_tensor(
            sh_tens, rank, load_legacy_1d_flatten_tensors
        )
        torch_sh_ten.key = sh_tens[0].key
        return torch_sh_ten

    def _mcore_to_torch_sharded_object(sh_objs: List[ShardedObject]) -> io.BytesIO:
        """Build io.BytesIO from given sharded objects data."""
        assert all(isinstance(sh_obj, ShardedObject) for sh_obj in sh_objs), sh_objs
        serialized_data = io.BytesIO()
        torch.save([sh_obj.data for sh_obj in sh_objs], serialized_data)
        return serialized_data

    for k, v in state_dict.items():
        if isinstance(v[0], ShardedTensor):
            v = cast(List[ShardedTensor], v)
            pyt_state_dict[k] = _mcore_to_torch_sharded_tensor(v)
        else:
            v = cast(List[ShardedObject], v)
            pyt_state_dict[k] = _mcore_to_torch_sharded_object(v)

    return pyt_state_dict


def _unwrap_pyt_sharded_tensor(sh_ten: TorchShardedTensor) -> List[torch.Tensor]:
    """Unwrap tensor from PyT ShardedTensor instance.

    If `prepend_axis_num` was non-zero (which is specific to MCore ShardedTensor)
    then the tensor has additional singleton dimensions which should be squeezed.
    """
    mcore_sh_ten = sh_ten.mcore_sh_ten
    ret_tensors = []
    for sh in sh_ten.local_shards():
        ten = sh.tensor
        if mcore_sh_ten.flattened_range is not None:
            assert ten.shape[:-1] == (1,) * (len(ten.shape) - 1), ten.shape
            ten = ten.view(-1)
        else:
            for _ in range(mcore_sh_ten.prepend_axis_num):
                assert ten.size(0) == 1
                ten = ten[0]  # NOTE: ten.squeeze(0) uses more memory for FP8 tensors
        ret_tensors.append(ten)
    return ret_tensors


def _replace_state_dict_keys_with_sharded_keys(
    sharded_state_dict: ShardedStateDict, keep_only_main_replica: bool = False
) -> Tuple[Dict[str, List[ShardedBase]], FLATTEN_MAPPING, Dict[str, List[str]]]:
    """Group ShardedBase objects by keys and
    return mappings required for recreating the original dict."""
    flat_sd, flat_mapping = flatten_state_dict(sharded_state_dict)
    rename_mapping = defaultdict(list)
    new_flat_sd = defaultdict(list)
    for k, sh_base in flat_sd.items():
        assert isinstance(sh_base, ShardedBase), type(sh_base)
        key = sh_base.unique_key if isinstance(sh_base, ShardedObject) else sh_base.key
        if is_main_replica(sh_base.replica_id) or not keep_only_main_replica:
            rename_mapping[key].append(k)
            new_flat_sd[key].append(sh_base)
    return new_flat_sd, flat_mapping, rename_mapping


def _replace_sharded_keys_with_state_dict_keys(
    state_dict: Dict[str, List[Union[torch.Tensor, io.BytesIO]]],
    flat_mapping: FLATTEN_MAPPING,
    rename_mapping: Dict[str, List[str]],
):
    """Inverse of _replace_state_dict_keys_with_sharded_keys."""
    recovered_sd = {}
    for k, tensors in state_dict.items():
        assert len(tensors) == len(rename_mapping[k])
        for ten, recovered_k in zip(tensors, rename_mapping[k]):
            recovered_sd[recovered_k] = ten

    return unflatten_state_dict(recovered_sd, flat_mapping)


def _restore_dict_types(x: Union[dict, list, Any], keys_template: Union[dict, list, Any]):
    """Recursively update `x` keys, based on `keys_template`."""
    if isinstance(keys_template, dict):
        assert isinstance(x, dict), type(x)
        for k, v in keys_template.items():
            if not isinstance(k, str):
                assert str(k) in x, (k, x.keys)
                x[k] = x.pop(str(k))
            _restore_dict_types(x[k], v)
    elif isinstance(keys_template, list):
        assert isinstance(x, list), type(x)
        for x_val, templ_val in zip(x, keys_template):
            _restore_dict_types(x_val, templ_val)


@dataclass(frozen=True)
class MCoreSavePlan(SavePlan):
    """SavePlan with MCore specific data."""

    mcore_data: Optional[Dict[str, Dict[str, Any]]] = None  # Mcore related data about each tensor


class MCoreSavePlanner(DefaultSavePlanner):
    """Differs with the default planner by saving BytesIO objects on all ranks.

    In the integration of MCore with PyT Distributed format, BytesIO objects
    come from ShardedObjects, which should be treated as separate objects on each rank
    (not common on all ranks).

    Also, the objects are already packed in io.BytesIO, so no need to redo it
    in transform_object.
    """

    def __init__(
        self,
        *args,
        dedup_replicated_tensors: Optional[bool] = None,
        nd_flattened_global_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        can_run_decentralized_global_plan: bool = True,
        **kwargs,
    ) -> None:
        # `dedup_replicated_tensors` was deprecated in 2.3; this check avoids warnings
        # during saving.
        if get_torch_version() <= PkgVersion("2.2"):
            kwargs['dedup_replicated_tensors'] = dedup_replicated_tensors
        super().__init__(*args, **kwargs)
        self.nd_flattened_global_shapes = nd_flattened_global_shapes or {}
        self.can_run_decentralized_global_plan = can_run_decentralized_global_plan
        if can_run_decentralized_global_plan:
            assert (
                not dedup_replicated_tensors
            ), 'Cannot run decentralized plan with dedup_replicated_tensors=True'
            assert (
                not self.flatten_state_dict
            ), 'Cannot run decentralized plan with flatten_state_dict=True'

    def create_local_plan(self) -> SavePlan:
        """Adds IOBytes write request on non-coordinator ranks."""

        # NOTE: for PyT 2.4.0a0 we can't rely on `create_default_local_save_plan` because
        # some alpha versions (specifically 2.4.0a0+f70bd71a48 in 24.06 NGC PyTorch container)
        # add iobytes request only on coordinator ranks and some alpha versions
        # (specifically 2.4.0a0+3bcc3cddb5 in 24.07 NGC PyTorch container)
        # add those requests on all ranks. We inline a simplified version of this method below.
        write_items = []
        for fqn, obj in self.state_dict.items():
            assert not HAVE_DTENSOR or not isinstance(
                obj, DTensor
            )  # translation from MCore ShardedTensors shouldn't result in DTensors
            # Create write requests for tensor and bytes values.
            # For MCore, these should be already non-duplicates.
            write_items += _create_write_items(fqn, obj)

        self.plan = MCoreSavePlan(
            items=write_items,
            planner_data=self.mappings,
            mcore_data={
                k: sh_ten.mcore_metadata
                for k, sh_ten in self.state_dict.items()
                if isinstance(sh_ten, TorchShardedTensor)
            },
        )
        return self.plan

    def create_global_plan(self, all_plans: List[MCoreSavePlan]) -> Tuple[List[SavePlan], Metadata]:
        """Merges MCore data for all plans."""
        global_plan, metadata = super().create_global_plan(all_plans)
        metadata.mcore_data = dict(
            ChainMap(*(plan.mcore_data for plan in all_plans))  # type: ignore[arg-type]
        )
        return global_plan, metadata

    def create_decentralized_global_plan(self, local_plan: SavePlan) -> SavePlan:
        """Nothing to do, just some checks.

        Args:
            local_plan (SavePlan): local plan to turn to a global plan
                (without interactions with other ranks)

        Returns:
            SavePlan - locally transformed plan equivalent to the plan that would be
                created by the coordinator
        """
        assert (
            not self.flatten_state_dict
        ), 'Cannot run decentralized plan with flatten_state_dict=True'
        assert not local_plan.planner_data, 'Planner data should be empty with decentralized plan'
        return local_plan

    def transform_object(self, write_item: WriteItem, object: Any):
        """Make no transformations - bytes objects are already serialized."""
        return object


class MCoreLoadPlanner(DefaultLoadPlanner):
    """Adds global shape validation to the default planner.

    If global shape validation can be ignored (shouldn't!), the default
    load planner can be used.
    """

    def __init__(
        self,
        *args,
        shapes_validation_sharded_tensors: Iterable[ShardedTensor] = (),
        allow_shape_mismatch_sharded_tensors: Optional[Dict[str, ShardedTensor]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.shapes_validation_sharded_tensors = shapes_validation_sharded_tensors
        self.allow_shape_mismatch_sharded_tensors = allow_shape_mismatch_sharded_tensors
        self._intermediate_read_item_and_target: Optional[Tuple[ReadItem, torch.Tensor]] = None

    @staticmethod
    def _expected_shape(sh_ten):
        return (
            nd_flattened_tensor_reformulated_global_shape(sh_ten)
            if is_nd_flattened_tensor(sh_ten)
            else sh_ten.global_shape
        )

    def _validate_global_shapes(self, metadata, sharded_tensors):
        for sh_ten in sharded_tensors:
            if sh_ten.key not in metadata.state_dict_metadata:
                raise KeyError(
                    f"{sh_ten.key} from model not in state dict:"
                    f" {sorted(metadata.state_dict_metadata.keys())}"
                )
            loaded_shape = metadata.state_dict_metadata[sh_ten.key].size
            expected_shape = self._expected_shape(sh_ten)
            if loaded_shape != expected_shape:
                if is_nd_flattened_tensor(sh_ten) and len(sh_ten.global_shape) == 1:
                    # Handle legacy 1-D flattened tensors checkpoint format
                    # where the global shape is not stored in the metadata
                    expected_shape = sh_ten.global_shape
                    if loaded_shape == expected_shape:
                        continue
                _msg = (
                    f'Global shape mismatch for loaded ({loaded_shape})'
                    f' and expected ({expected_shape}) tensor'
                    f' for key {sh_ten.key}'
                )
                raise CheckpointingException(_msg)

    @contextmanager
    def _temporarily_bypass_shape_validation(self):
        """
        Temporarily set the size of tensors to their expected shapes to bypass DCP shape validation.
        This is used when validating the shapes during local plan creation.
        """
        if not self.allow_shape_mismatch_sharded_tensors:
            yield
            return

        tensor_metadata = self.metadata.state_dict_metadata
        metadata_with_sizes = [
            (tensor_metadata[key], tensor_metadata[key].size, sharded_tensor)
            for key, sharded_tensor in self.allow_shape_mismatch_sharded_tensors.items()
        ]
        try:
            # Temporarily set sizes to expected shapes
            for md, _, sharded_tensor in metadata_with_sizes:
                md.size = self._expected_shape(sharded_tensor)
            yield
        finally:
            # Restore original sizes after yield
            for md, size, _ in metadata_with_sizes:
                md.size = size

    def create_local_plan(self) -> LoadPlan:
        """Runs additional shapes validation."""
        self._validate_global_shapes(self.metadata, self.shapes_validation_sharded_tensors)

        with self._temporarily_bypass_shape_validation():
            local_plan = super().create_local_plan()

        return local_plan

    def resolve_tensor(self, read_item: ReadItem):
        """Override to add FP8 support.

        Narrowing the Float8Tensor can create incontiguous tensors and there are
        no `copy` kernels for such cases. This method creates a contiguous FP8
        tensors so that the subsequent `copy_` in FileSystemReader succeeds.
        Note that this requires tracking the original tensor
        (as `self._intermediate_read_item_and_target` attribute)
        and restoring it in `commit_tensor` method.
        """
        target_tensor = super().resolve_tensor(read_item)
        if (
            not target_tensor.is_contiguous()
            and HAVE_TE
            and isinstance(target_tensor, Float8Tensor)
        ):
            self._intermediate_read_item_and_target = (read_item, target_tensor)
            target_tensor = Float8Tensor.make_like(
                target_tensor, data=target_tensor._data.contiguous()
            )
        return target_tensor

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        """Restores the original FP8 tensor saved in `resolve_tensor`."""
        if self._intermediate_read_item_and_target is not None:
            interm_read_item, target_tensor = self._intermediate_read_item_and_target
            assert (
                interm_read_item is read_item
            ), '`commit_tensor` method should be called right after `resolve_tensor`'
            target_tensor.copy_(tensor)
            tensor = target_tensor
            self._intermediate_read_item_and_target = None
        return super().commit_tensor(read_item, tensor)


class TorchDistSaveShardedStrategy(AsyncSaveShardedStrategy):
    """Async save strategy for the PyT Distributed format.

    The idea is to translate MCore ShardedTensors into PyT ShardedTensors
    and use the async-adjusted torch.distributed.checkpoint saving mechanism
    provided by the FileSystemWriterAsync writer.
    """

    def __init__(
        self,
        backend: str,
        version: int,
        keep_only_main_replica: bool = True,
        thread_count: int = 1,
        cached_metadata: bool = False,
        separation_hint: Optional[str] = None,
    ):
        """Adds parameters specific to PyT Distributed format
        Args:
            backend (str): format backend string
            version (int): format version
            keep_only_main_replica (bool, optional): PyT Distributed has a mechanism
                for deduplication, but replica_id aware deduplication is more coherent.
                Default is True (recommended to keep it).
            thread_count (int, optional): threads to use during saving.
                Affects the number of files in the checkpoint (saving ranks * num_threads).
            cached_metadata (bool, optional): Enables using cached global metadata to avoid
                gathering local metadata every checkpointing invocation
            separation_hint(str, optional): If provided, all tensors whose keys have this
                prefix will be saved to a separate file.
        """
        super().__init__(backend, version)
        self.keep_only_main_replica = keep_only_main_replica
        self.thread_count = thread_count

        # Cached SavePlans to skip plan in `save_state_dict_async_plan`
        # cached outcome of `SavePlan.prepare_global_plan`,
        # which aggregates local plans from all ranks
        self.cached_central_plan: SavePlan = None
        # cached outcome of `SavePlan.prepare_local_plan` describes how local state_dict is written
        self.cached_local_plan: SavePlan = None
        # Cached global metadata, only `coordinator` for dist-ckpt holds
        # if central plans are consistent over iters
        self.cached_global_metadata: Metadata = None
        # This variable records if the ckpt structures are consistent
        # so the following checkpoint savings reuse `cached_global_metadata`
        self.validated_cache_reuse: bool = False
        # The knob to enable cached metadata communication in saving
        self.use_cached_ckpt_structure: bool = cached_metadata

        self.separation_hint = separation_hint

        self.validated_loaded_metadata_reuse = False
        
        # Initialize EC-CHECK manager (singleton instance shared with Load strategy)
        self.eccheck_manager = ECCHECKManager()
        self.eccheck_manager.init_eccheck_if_enabled()
        
        # Initialize strategy-specific EC-CHECK state
        self.eccheck_preallocate_cpu_buffer = True  # Preallocate CPU buffer for tensor data
        self.eccheck_use_continuous_buffer = True  # Use continuous buffer for tensor data
        self.decomposed_state_dict = None
        self.preallocated_cpu_buffer = None
        self.eccheck_serialized_metadata = None
        self.eccheck_global_registry = None
        self.eccheck_recv_encoding_buffers = None
        self.eccheck_p2p_buffers = None
        self.ecc_write_buckets = []

    def _get_p2p_partner_rank(self, my_rank: int, world_size: int) -> int:
        """Get P2P partner rank using the shared manager."""
        return self.eccheck_manager.get_p2p_partner_rank(my_rank, world_size)

    def _allocate_recv_encoding_buffers_phase2(self, global_registry):
        """
        Allocate TWO large receive buffers for peer encoded packets (one per encoding thread).
        
        Each buffer is equal to peer's total data size, aligned to buffer_size (64MB).
        This is called after metadata exchange when peer data sizes are known.
        
        Args:
            global_registry (GlobalMetadataRegistry): Complete metadata from all ranks
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two receive buffers (one for thread1, one for thread2)
        """
        return self.eccheck_manager.allocate_recv_encoding_buffers_phase2(global_registry)

    def _allocate_p2p_buffers(self, global_registry):
        """
        Allocate TWO large continuous buffers for P2P stage:
        - Buffer 1: Store own data/parity (based on P2P role)
        - Buffer 2: Store P2P partner's data/parity
        
        Buffer sizes are determined from metadata in global_registry.
        This is called after metadata exchange when data sizes are known.
        
        Args:
            global_registry (GlobalMetadataRegistry): Complete metadata from all ranks
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'own_buffer' and 'partner_buffer'
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        p2p_partner_rank = self._get_p2p_partner_rank(rank, world_size)
        
        # ===== Get own data size from metadata =====
        own_metadata = global_registry.rank_metadata.get(rank, [])
        own_total_size = sum(meta.size_bytes for meta in own_metadata)
        
        if rank % 2 == 0:
            own_metadata_updated = [
                replace(meta, source_rank=rank, target_rank=rank, chunk_type='data')
                for meta in own_metadata
            ]  
        else:
            own_metadata_updated = [
                replace(meta, source_rank=rank, target_rank=rank, chunk_type='parity')
                for meta in own_metadata
            ]
            
        # ===== Get P2P partner's data size from metadata =====
        partner_metadata = global_registry.rank_metadata.get(p2p_partner_rank, [])
        partner_total_size = sum(meta.size_bytes for meta in partner_metadata)
        
        # Update partner_metadata: set source_rank to p2p_partner_rank and target_rank to rank
        if rank % 2 == 0:
            partner_metadata_updated = [
                replace(meta, source_rank=p2p_partner_rank, target_rank=rank, chunk_type='data')
                for meta in partner_metadata
            ]
        else:
            partner_metadata_updated = [
                replace(meta, source_rank=p2p_partner_rank, target_rank=rank, chunk_type='parity')
                for meta in partner_metadata
            ]

        # ===== Calculate maximum data size across all ranks (for pipeline synchronization) =====
        if torch.distributed.is_initialized():
            # Get all ranks' data sizes from global_registry and compute max locally
            # (We already have all ranks' metadata, so no need for all_reduce/all_gather)
            all_total_bytes_list = []
            for r in range(world_size):
                rank_metadata = global_registry.rank_metadata.get(r, [])
                rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
                all_total_bytes_list.append(rank_total_size)
            
            # Compute maximum locally (all ranks have the same global_registry)
            max_total_bytes = max(all_total_bytes_list)
        else:
            max_total_bytes = max(own_total_size, partner_total_size)
        
        # ===== Align both sizes to buffer_size (64MB) using maximum for pipeline sync =====
        eccheck_buffer_size = self.eccheck_manager.eccheck_buffer_size
        # Use maximum size for pipeline synchronization (all ranks use same size)
        own_pipeline_size = max_total_bytes
        partner_pipeline_size = max_total_bytes
        own_aligned_size = ((own_pipeline_size + eccheck_buffer_size - 1) // eccheck_buffer_size) * eccheck_buffer_size
        partner_aligned_size = ((partner_pipeline_size + eccheck_buffer_size - 1) // eccheck_buffer_size) * eccheck_buffer_size
        
        logger.info(
            f"EC-CHECK: Allocating P2P buffers based on metadata\n"
            f"  P2P partner rank: {p2p_partner_rank}\n"
            f"  Own data size: {own_total_size / (1024**3):.2f} GB "
            f"(actual), {max_total_bytes / (1024**3):.2f} GB (pipeline max), "
            f"{own_aligned_size / (1024**3):.2f} GB (aligned)\n"
            f"  Partner data size: {partner_total_size / (1024**3):.2f} GB "
            f"(actual), {max_total_bytes / (1024**3):.2f} GB (pipeline max), "
            f"{partner_aligned_size / (1024**3):.2f} GB (aligned)\n"
            f"  Total P2P memory: {(own_aligned_size + partner_aligned_size) / (1024**3):.2f} GB"
        )
        
        # ===== Allocate two large continuous buffers =====
        # Buffer 1: Own data/parity
        own_buffer = torch.empty(own_aligned_size, dtype=torch.uint8)
        
        # Buffer 2: Partner's data/parity
        partner_buffer = torch.empty(partner_aligned_size, dtype=torch.uint8)
        
        logger.info(
            f"EC-CHECK: Allocated P2P buffers:\n"
            f"  Own buffer: {own_aligned_size / (1024**3):.2f} GB "
            f"({own_aligned_size / (1024**2):.0f} MB)\n"
            f"  Partner buffer: {partner_aligned_size / (1024**3):.2f} GB "
            f"({partner_aligned_size / (1024**2):.0f} MB)"
        )
        
        # ===== Package own and partner metadata/buffers into eccheck_bytes_data format =====
        # Get non-tensor data from global_registry
        own_non_tensor_data = global_registry.rank_non_tensor_data.get(rank, {})
        partner_non_tensor_data = global_registry.rank_non_tensor_data.get(p2p_partner_rank, {})
        
        # Serialize own metadata
        own_non_tensor_data_bytes = pickle.dumps(own_non_tensor_data)
        own_tensor_keys_data_bytes = pickle.dumps(own_metadata_updated)
        own_non_tensor_size = len(own_non_tensor_data_bytes)
        own_tensor_keys_size = len(own_tensor_keys_data_bytes)
        own_tensor_buffer_size = own_total_size  # Use actual size for metadata (not padded)
        
        # Serialize partner metadata
        partner_non_tensor_data_bytes = pickle.dumps(partner_non_tensor_data)
        partner_tensor_keys_data_bytes = pickle.dumps(partner_metadata_updated)
        partner_non_tensor_size = len(partner_non_tensor_data_bytes)
        partner_tensor_keys_size = len(partner_tensor_keys_data_bytes)
        partner_tensor_buffer_size = partner_total_size  # Use actual size for metadata (not padded)
        
        # Store actual sizes and pipeline sizes for later use
        own_actual_size = own_total_size
        partner_actual_size = partner_total_size
        p2p_pipeline_total_bytes = max_total_bytes
        
        # Create own serialized metadata (similar to eccheck_serialized_metadata)
        own_serialized_metadata = {
            'non_tensor_data': own_non_tensor_data_bytes,
            'tensor_keys_data': own_tensor_keys_data_bytes,
            'non_tensor_size': own_non_tensor_size,
            'tensor_keys_size': own_tensor_keys_size,
            'tensor_buffer_size': own_tensor_buffer_size,
            'eccheck_file': f'__{rank}_p2p_own.distcp',  # P2P own file
            'eccheck_file_path': None,  # Will be set later if needed
        }
        
        # Create partner serialized metadata
        partner_serialized_metadata = {
            'non_tensor_data': partner_non_tensor_data_bytes,
            'tensor_keys_data': partner_tensor_keys_data_bytes,
            'non_tensor_size': partner_non_tensor_size,
            'tensor_keys_size': partner_tensor_keys_size,
            'tensor_buffer_size': partner_tensor_buffer_size,
            'eccheck_file': f'__{p2p_partner_rank}_p2p_partner.distcp',  # P2P partner file
            'eccheck_file_path': None,  # Will be set later if needed
        }
        
        # Create eccheck_bytes_data format for own data
        own_eccheck_bytes_data = [
            ('eccheck_metadata', own_serialized_metadata),
            ('eccheck_continuous_buffer', own_buffer),  # Own buffer
        ]
        
        # Create eccheck_bytes_data format for partner data
        partner_eccheck_bytes_data = [
            ('eccheck_metadata', partner_serialized_metadata),
            ('eccheck_continuous_buffer', partner_buffer),  # Partner buffer
        ]
        
        # ===== Package into WriteBucket format =====
        # WriteBucket = Tuple[Path, str, Tuple[list, list]]
        # Format: (file_path, storage_key, (bytes_data, tensor_data))
        from pathlib import Path
        
        # Get checkpoint_dir (should be set in save() method)
        checkpoint_dir = getattr(self, 'current_checkpoint_dir', None)
        if checkpoint_dir is None:
            logger.warning("EC-CHECK: checkpoint_dir not available, using file_name as path")
            checkpoint_dir = Path(".")
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        # Generate file names for own and partner
        own_file_name = own_serialized_metadata['eccheck_file']
        partner_file_name = partner_serialized_metadata['eccheck_file']
        
        # Build full file paths using checkpoint_dir
        own_file_path = checkpoint_dir / own_file_name
        partner_file_path = checkpoint_dir / partner_file_name
        
        # Update serialized metadata with full paths
        own_serialized_metadata['eccheck_file_path'] = str(own_file_path)
        partner_serialized_metadata['eccheck_file_path'] = str(partner_file_path)
        
        # Create WriteBucket for own data
        own_write_bucket = (
            own_file_path,              # file_path (full path with checkpoint_dir)
            own_file_name,              # storage_key (used in metadata)
            (own_eccheck_bytes_data, []),  # (bytes_data, tensor_data)
        )
        
        # Create WriteBucket for partner data
        partner_write_bucket = (
            partner_file_path,          # file_path (full path with checkpoint_dir)
            partner_file_name,          # storage_key (used in metadata)
            (partner_eccheck_bytes_data, []),  # (bytes_data, tensor_data)
        )
        
        self.ecc_write_buckets.append(own_write_bucket)
        self.ecc_write_buckets.append(partner_write_bucket)
        
        logger.info(
            f"EC-CHECK: Packaged P2P data into eccheck_bytes_data and WriteBucket format:\n"
            f"  Own metadata: {own_non_tensor_size / 1024:.2f} KB (non-tensor) + "
            f"{own_tensor_keys_size / 1024:.2f} KB (tensor keys), "
            f"{own_tensor_buffer_size / (1024**3):.2f} GB (buffer)\n"
            f"  Own WriteBucket: {own_file_name}\n"
            f"  Partner metadata: {partner_non_tensor_size / 1024:.2f} KB (non-tensor) + "
            f"{partner_tensor_keys_size / 1024:.2f} KB (tensor keys), "
            f"{partner_tensor_buffer_size / (1024**3):.2f} GB (buffer)\n"
            f"  Partner WriteBucket: {partner_file_name}"
        )
        
        return {
            'own_buffer': own_buffer,
            'partner_buffer': partner_buffer,
            "own_write_bucket": own_write_bucket,
            "partner_write_bucket": partner_write_bucket,
            'own_actual_size': own_actual_size,
            'partner_actual_size': partner_actual_size,
            'p2p_pipeline_total_bytes': p2p_pipeline_total_bytes,
        }
    
    def _get_eccheck_buffers(self):
        """Get EC-CHECK buffers for FileSystemWriterAsync.
        
        Note: Returns data, encoding, and parity buffers.
        Receive buffers will be allocated by FileSystemWriterAsync
        after metadata exchange.
        """
        return self.eccheck_manager.get_eccheck_buffers()

    def __del__(self):
        """Cleanup EC-CHECK resources when strategy is destroyed.
        
        Note: Manager cleanup is handled by the manager itself (singleton).
        We don't need to cleanup here since the manager is shared.
        """
        pass

    def async_save(
        self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
    ) -> AsyncRequest:
        """Translates MCore ShardedTensors to PyT ShardedTensors & saves in PyT Distributed format.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to save
            checkpoint_dir (Path): checkpoint directory

        Returns: None
        """
        # Store checkpoint_dir for EC-CHECK preparation
        self.current_checkpoint_dir = checkpoint_dir
        
        # Translate the state dict
        (sharded_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(
                sharded_state_dict, self.keep_only_main_replica
            )
        )
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
        from megatron.training import get_args as input_args
        args = input_args()
        # Use PyT saving mechanism
        writer = FileSystemWriterAsync(
            checkpoint_dir,
            separation_hint=self.separation_hint,
            thread_count=self.thread_count,
            use_msc=MultiStorageClientFeature.is_enabled(),
            use_eccheck=self.eccheck_manager.use_eccheck,
            eccheck_native=self.eccheck_manager._eccheck_native,  # Pass pre-initialized C++ module
            eccheck_buffers=self._get_eccheck_buffers(),  # Pass pre-allocated buffers
        )

        # This should be set differently if we run in a smaller process group than the default
        coordinator = 0
        # Try twice to validate the generated `central_plan` is the same across iterations
        # If so, reuse `cached_central_plan` and `cached_global_metadata`
        # From the 3rd iteration, `save_state_dict_async_plan` will not generate `global_metadata`
        # (return None) so `self.cached_global_metadata` is reused
        args_cached_plans = None
        loaded_all_plans = None
        if self.use_cached_ckpt_structure:
            loaded_all_plans = getattr(self.cached_global_metadata, "all_local_plans", None)
            if loaded_all_plans is None:
                logger.debug(
                    "no all_local_plans in metadata - can't verify global metadata reuse..."
                )

            args_cached_plans = (
                self.cached_central_plan,
                self.cached_local_plan,
                self.validated_cache_reuse,
            )

        (
            save_state_dict_ret,
            self.cached_central_plan,
            self.cached_local_plan,
            self.validated_cache_reuse,
            self.validated_loaded_metadata_reuse,
            planner
        ) = save_state_dict_async_plan(
            pyt_state_dict,
            writer,
            None,
            coordinator,
            planner=MCoreSavePlanner(
                dedup_replicated_tensors=not self.keep_only_main_replica, flatten_state_dict=False
            ),
            cached_ckpt_structure=args_cached_plans,
            loaded_all_plans=loaded_all_plans,
        )
        # EC-CHECK mode: decompose state_dict and preallocate CPU memory
        if self.eccheck_manager.use_eccheck:
            self._prepare_eccheck_data(self.cached_central_plan, planner)
            # Pass EC-CHECK state to writer if available
            writer.decomposed_state_dict = self.decomposed_state_dict
            writer.preallocated_cpu_buffer = self.preallocated_cpu_buffer
            writer.eccheck_serialized_metadata = self.eccheck_serialized_metadata
            writer.eccheck_global_registry = self.eccheck_global_registry
            # Pass the updated receive buffers (TWO large buffers allocated based on peer size)
            writer.eccheck_recv_encoding_buffers = self.eccheck_recv_encoding_buffers
            # Pass P2P buffers (own_buffer and partner_buffer)
            writer.eccheck_p2p_buffers = self.eccheck_p2p_buffers
            
            writer.ecc_write_buckets = self.ecc_write_buckets
            # In EC-CHECK mode, call prepare_write_data to create write_buckets
            # It will use the metadata we just prepared
            writer.prepare_write_data(self.cached_central_plan, planner)
        else:
            start = time()
            writer.prepare_write_data(self.cached_central_plan, planner)
            end = time()
            logger.debug(f"{time()} rank: {rank}, write(async) time: {end - start}")
            
        rank = torch.distributed.get_rank()
        if self.use_cached_ckpt_structure:
            if (
                loaded_all_plans
                and self.cached_global_metadata
                and self.validated_loaded_metadata_reuse
            ):
                if coordinator == rank:
                    logger.debug(
                        f"rank: {rank}, reuse global metadata from loaded"
                        f" .metadata, {save_state_dict_ret[1]}"
                    )
                    save_state_dict_ret = list(save_state_dict_ret)
                    save_state_dict_ret[1] = self.cached_global_metadata

            elif self.validated_cache_reuse:
                logger.debug(f"rank: {rank}, cache validated")
                if save_state_dict_ret[1]:  # when global_metadata is not cached
                    self.cached_global_metadata = save_state_dict_ret[1]  # Cache Metadata
                # Only Coordinator rank holds cached global_metadata
                # (None is returned for global_metadata)
                elif coordinator == rank:
                    logger.debug(
                        f"rank: {rank}, reuse global metadata cached from previous"
                        f" save iteration, {save_state_dict_ret[1]}"
                    )
                    save_state_dict_ret = list(save_state_dict_ret)
                    save_state_dict_ret[1] = self.cached_global_metadata

        return self._get_save_and_finalize_callbacks(writer, save_state_dict_ret)

    def _prepare_eccheck_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        EC-CHECK preparation: organize data for serialization-free encoding.
        
        This method performs the following steps:
        1. Process plan items like normal mode (separate bytes and tensors)
        2. Organize tensors for EC-CHECK (extract metadata and data)
        3. Preallocate CPU memory buffer for tensors
        4. Prepare write buckets for async transfer
        
        Args:
            plan (SavePlan): save plan from PyTorch distributed checkpoint
            planner (SavePlanner): save planner to resolve data
        """
        from torch.distributed.checkpoint.filesystem import _StoragePrefix
        
        start_total = time()
        logger.info("EC-CHECK: Starting serialization-free checkpoint preparation")
        
        # Step 1: Process plan items (similar to normal mode)
        start = time()
        storage_plan: _StoragePrefix = plan.storage_data
        
        # Separate items into BYTE_IO (non-tensor) and TENSOR
        non_tensor_data = {}
        tensor_infos = []
        tensor_data_list = []
        
        logger.info(f"EC-CHECK: Processing {len(plan.items)} items from SavePlan")
        byte_io_count = 0
        tensor_count = 0
        none_data_count = 0
        
        for item in plan.items:
            data = planner.resolve_data(item)
            
            # Debug: check for None data
            if data is None:
                none_data_count += 1
                if none_data_count <= 5:
                    logger.warning(f"EC-CHECK SAVE: Found None data for item: fqn={item.index.fqn}, type={item.type}")
                continue  # Skip None data items
            
            if item.type == WriteItemType.BYTE_IO:
                # Non-tensor data (e.g., extra_state)
                # BytesIO objects need special handling to preserve format
                import io
                if isinstance(data, io.BytesIO):
                    # Store the BytesIO content directly as bytes
                    # We'll also store metadata to indicate this was a BytesIO
                    non_tensor_data[item.index.fqn] = {
                        '_eccheck_type': 'BytesIO',
                        '_eccheck_data': data.getvalue()
                    }
                else:
                    non_tensor_data[item.index.fqn] = data
                byte_io_count += 1
            else:
                # Tensor data - create TensorInfo
                # Extract and store serializable fields from WriteItem.index
                from .state_dict_decomposer import TensorInfo
                
                tensor_info = TensorInfo(
                    key=item.index.fqn,  # Keep FQN as base key
                    shape=tuple(data.shape),
                    dtype=data.dtype,
                    device=data.device,
                    numel=data.numel(),
                    size_bytes=data.numel() * data.element_size(),
                    offset=0,  # Will be calculated below
                    global_offset=tuple(item.index.offset),  # Extract offset as tuple (serializable)
                    shard_index=item.index.index,  # Extract index (serializable)
                )
                tensor_infos.append(tensor_info)
                tensor_data_list.append(data)
                tensor_count += 1
        
        logger.info(
            f"EC-CHECK: Processed {byte_io_count} BytesIO items, {tensor_count} tensor items"
            + (f", skipped {none_data_count} None items" if none_data_count > 0 else "")
        )
        
        # Calculate offsets for tensor data
        offset = 0
        for info in tensor_infos:
            info.offset = offset
            offset += info.size_bytes
        
        # Create decomposed structure
        self.decomposed_state_dict = DecomposedStateDict(
            non_tensor_data=non_tensor_data,
            tensor_infos=tensor_infos,
            tensor_data=tensor_data_list,
        )
        
        process_time = time() - start
        
        # Log statistics
        stats = self.decomposed_state_dict.get_statistics()
        logger.info(
            f"EC-CHECK: Processed plan items in {process_time:.2f}s\n"
            f"  Non-tensor items: {len(non_tensor_data)}\n"
            f"  Tensor items: {len(tensor_data_list)}\n"
            f"  Non-tensor data: {stats['non_tensor_size_bytes'] / 1024:.2f} KB "
            f"({stats['non_tensor_percentage']:.4f}%)\n"
            f"  Tensor keys: {stats['tensor_keys_size_bytes'] / 1024:.2f} KB "
            f"({stats['tensor_keys_percentage']:.4f}%)\n"
            f"  Tensor data: {stats['tensor_data_size_bytes'] / (1024**3):.2f} GB "
            f"({stats['tensor_data_percentage']:.2f}%)"
        )
        
        # Step 2: Preallocate CPU memory buffer if enabled
        if self.eccheck_preallocate_cpu_buffer:
            start = time()
            total_size = self.decomposed_state_dict.total_tensor_size_bytes
            logger.info(f"EC-CHECK: Preallocating CPU buffer of {total_size / (1024**3):.2f} GB")
            
            if self.preallocated_cpu_buffer == None:
                if self.eccheck_manager.eccheck_pin_memory and torch.cuda.is_available():
                    self.preallocated_cpu_buffer = torch.empty(
                        total_size, dtype=torch.uint8).pin_memory()
                    logger.debug("EC-CHECK: Using pinned memory for CPU buffer")
                else:
                    self.preallocated_cpu_buffer = torch.empty(
                        total_size, dtype=torch.uint8
                    )
                
            prealloc_time = time() - start
            logger.debug(f"EC-CHECK: CPU buffer preallocation took {prealloc_time:.2f}s")
        else:
            prealloc_time = 0
        
        # Step 3: Prepare write buckets for async transfer
        start = time()
        self._prepare_eccheck_write_buckets(plan, self.current_checkpoint_dir)
        bucket_time = time() - start
        logger.debug(f"EC-CHECK: Write bucket preparation took {bucket_time:.2f}s")
        
        # Validate decomposition
        if not self.validate_eccheck_decomposition():
            raise RuntimeError("EC-CHECK: Decomposition validation failed")
        
        # Step 4: Broadcast and exchange metadata
        start = time()
        self.eccheck_global_registry = self._broadcast_and_exchange_metadata()
        metadata_time = time() - start
        logger.info(f"EC-CHECK: Metadata exchange completed in {metadata_time:.2f}s")
        
        # Step 5: Allocate receive buffers based on peer data size
        start = time()
        if self.eccheck_recv_encoding_buffers == None:
            self.eccheck_recv_encoding_buffers = self._allocate_recv_encoding_buffers_phase2(self.eccheck_global_registry)
        buffer_alloc_time = time() - start
        logger.info(f"EC-CHECK: Receive buffer allocation completed in {buffer_alloc_time:.2f}s")
        
        # Step 6: Allocate P2P buffers based on metadata
        start = time()
        if self.eccheck_p2p_buffers is None:
            self.eccheck_p2p_buffers = self._allocate_p2p_buffers(self.eccheck_global_registry)
            # Store to manager for reuse in load phase
            self.eccheck_manager.eccheck_p2p_buffers = self.eccheck_p2p_buffers
            
        p2p_buffer_alloc_time = time() - start
        logger.info(f"EC-CHECK: P2P buffer allocation completed in {p2p_buffer_alloc_time:.2f}s")
        
        total_time = time() - start_total
        logger.info(
            f"EC-CHECK: Preparation completed in {total_time:.2f}s\n"
            f"  Item processing: {process_time:.2f}s\n"
            f"  Preallocation: {prealloc_time:.2f}s\n"
            f"  Bucket prep: {bucket_time:.2f}s\n"
            f"  Metadata exchange: {metadata_time:.2f}s\n"
            f"  Receive buffer allocation: {buffer_alloc_time:.2f}s\n"
            f"  P2P buffer allocation: {p2p_buffer_alloc_time:.2f}s"
        )
        
    def _broadcast_and_exchange_metadata(self):
        """
        All-to-all metadata exchange using torch.distributed.all_gather.
        
        Each rank broadcasts its metadata to all other ranks.
        After this call, all ranks have complete metadata from all peers.
        
        Returns:
            GlobalMetadataRegistry: Complete metadata from all ranks
        """
        from .state_dict_decomposer import GlobalMetadataRegistry
        import pickle
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        logger.info(f"EC-CHECK: [Rank {rank}] Starting metadata exchange with {world_size} ranks")
        
        # ===== Step 1: Prepare local metadata (both tensor and non-tensor) =====
        local_tensor_metadata = self._prepare_local_metadata_for_broadcast(rank, world_size)
        local_non_tensor_data = self.decomposed_state_dict.non_tensor_data
        
        # Package both together
        local_package = {
            'tensor_metadata': local_tensor_metadata,
            'non_tensor_data': local_non_tensor_data,
        }
        
        logger.info(
            f"EC-CHECK: [Rank {rank}] Local metadata: "
            f"{len(local_tensor_metadata)} tensor items, "
            f"{len(local_non_tensor_data)} non-tensor items"
        )
        
        # ===== Step 2: All-gather complete metadata using all_gather_object =====
        # This automatically handles serialization, padding, and deserialization
        # Transmits both tensor_metadata and non_tensor_data
        all_packages = [None] * world_size
        torch.distributed.all_gather_object(all_packages, local_package)
        
        # ===== Step 3: Build rank_metadata and rank_non_tensor_data dicts =====
        rank_metadata = {}
        rank_non_tensor_data = {}
        for i, package in enumerate(all_packages):
            rank_metadata[i] = package['tensor_metadata']
            rank_non_tensor_data[i] = package['non_tensor_data']
        
        logger.info(
            f"EC-CHECK: [Rank {rank}] Received metadata from all {world_size} ranks"
        )
        
        # Create registry with both tensor and non-tensor metadata
        registry = GlobalMetadataRegistry(
            rank_metadata=rank_metadata,
            rank_non_tensor_data=rank_non_tensor_data
        )
        
        # Log statistics
        stats = registry.get_statistics()
        logger.info(
            f"EC-CHECK: Global metadata exchange complete:\n"
            f"  Total ranks: {stats['total_ranks']}\n"
            f"  Total tensor items: {stats['total_tensor_chunks']}\n"
            f"  Total non-tensor items: {stats['total_non_tensor_items']}\n"
            f"  Metadata size: {stats['total_metadata_bytes'] / 1024:.2f} KB (actual transmitted)\n"
            f"  Tensor data size: {stats['total_tensor_data_bytes'] / (1024**3):.2f} GB (referenced, not transmitted)\n"
            f"  Per-rank tensor items: {stats['per_rank_tensor_items']}\n"
            f"  Per-rank non-tensor items: {stats['per_rank_non_tensor_items']}"
        )
        
        return registry

    def _prepare_local_metadata_for_broadcast(self, my_rank: int, world_size: int):
        """
        Prepare local tensor metadata for broadcasting to all ranks.
        
        Currently implements simple strategy:
        - Each rank keeps its own data chunks locally (target_rank = my_rank)
        - Future: Add parity chunk generation for redundancy
        
        Args:
            my_rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            List[TensorMetadata]: Serializable metadata for broadcasting
        """
        from .state_dict_decomposer import TensorMetadata
        
        local_metadata = []
        
        for info in self.decomposed_state_dict.tensor_infos:
            # Create metadata for data chunk
            data_meta = TensorMetadata(
                key=info.key,
                shape=info.shape,
                dtype=str(info.dtype),
                size_bytes=info.size_bytes,
                global_offset=info.global_offset if info.global_offset is not None else (),
                shard_index=info.shard_index if info.shard_index is not None else 0,
                chunk_type='data',
                target_rank=my_rank,  # Data stays on same rank
                source_rank=my_rank,
            )
            local_metadata.append(data_meta)
        
        return local_metadata
        
    def _prepare_eccheck_write_buckets(self, plan: SavePlan, checkpoint_dir: Optional[Path] = None) -> None:
        """
        Prepare write buckets for EC-CHECK mode.
        
        In EC-CHECK mode, each node stores three components in ONE file:
        1. Serialized non-tensor key-value pairs
        2. Serialized tensor keys (tensor_infos)
        3. Tensor data buffer (will be filled during preload)
        
        File structure:
        [Header: sizes of 3 components] [Component 1] [Component 2] [Component 3]
        
        Args:
            plan (SavePlan): save plan
            checkpoint_dir (Optional[Path]): checkpoint directory (if available)
        """
        from torch.distributed.checkpoint.filesystem import _StoragePrefix
        
        storage_plan: _StoragePrefix = plan.storage_data
        
        # Serialize Components 1 & 2 in CPU memory
        non_tensor_data = pickle.dumps(self.decomposed_state_dict.non_tensor_data)
        tensor_keys_data = pickle.dumps(self.decomposed_state_dict.tensor_infos)
        
        # Calculate sizes for header
        non_tensor_size = len(non_tensor_data)
        tensor_keys_size = len(tensor_keys_data)
        tensor_buffer_size = self.decomposed_state_dict.total_tensor_size_bytes
        
        # Create single file for all three components
        # Use standard .distcp extension for compatibility, but with EC-CHECK content
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # Use the same naming convention as standard distributed checkpoint
        # Format: __{rank}_{thread_id}.distcp
        eccheck_file = f"__{rank}_0.distcp"  # Single file per rank, thread 0
        
        # Build full path if checkpoint_dir is available
        if checkpoint_dir is not None:
            eccheck_path = os.path.join(str(checkpoint_dir), eccheck_file)
        else:
            # Will be set by FileSystemWriterAsync later
            eccheck_path = None
        
        # Store the serialized metadata for later use by FileSystemWriterAsync
        self.eccheck_serialized_metadata = {
            'non_tensor_data': non_tensor_data,
            'tensor_keys_data': tensor_keys_data,
            'non_tensor_size': non_tensor_size,
            'tensor_keys_size': tensor_keys_size,
            'tensor_buffer_size': tensor_buffer_size,
            'eccheck_file': eccheck_file,  # Filename
            'eccheck_file_path': eccheck_path,  # Full path (if available)
        }
        
        logger.debug(
            f"EC-CHECK: Prepared single file structure:\n"
            f"  File: {eccheck_file}\n"
            f"  Component 1 size: {non_tensor_size / 1024:.2f} KB\n"
            f"  Component 2 size: {tensor_keys_size / 1024:.2f} KB\n"
            f"  Component 3 size: {tensor_buffer_size / (1024**3):.2f} GB"
        )
    
    def validate_eccheck_decomposition(self) -> bool:
        """
        Simple validation: check if state_dict is correctly decomposed into three components.
        
        Validates:
        1. non_tensor_data is a dict
        2. tensor_infos is a list (tensor keys)
        3. tensor_data is a list of tensors
        4. Counts match between tensor_infos and tensor_data
        
        Returns:
            bool: True if decomposition is valid, False otherwise
        """
        if not self.eccheck_manager.use_eccheck:
            logger.warning("EC-CHECK: Validation skipped - EC-CHECK is not enabled")
            return False
        
        if not self.decomposed_state_dict:
            logger.error("EC-CHECK: Validation failed - State dict not decomposed yet")
            return False
        
        decomposed = self.decomposed_state_dict
        
        # Check 1: Non-tensor key-value pairs (dict)
        if not isinstance(decomposed.non_tensor_data, dict):
            logger.error(
                f"EC-CHECK: Component 1 failed - non_tensor_data should be dict, "
                f"got {type(decomposed.non_tensor_data).__name__}"
            )
            return False
        
        # Check 2: Tensor keys (list)
        if not isinstance(decomposed.tensor_infos, list):
            logger.error(
                f"EC-CHECK: Component 2 failed - tensor_infos should be list, "
                f"got {type(decomposed.tensor_infos).__name__}"
            )
            return False
        
        # Check 3: Tensor data (list)
        if not isinstance(decomposed.tensor_data, list):
            logger.error(
                f"EC-CHECK: Component 3 failed - tensor_data should be list, "
                f"got {type(decomposed.tensor_data).__name__}"
            )
            return False
        
        # Check 4: Counts match
        if len(decomposed.tensor_infos) != len(decomposed.tensor_data):
            logger.error(
                f"EC-CHECK: Count mismatch - {len(decomposed.tensor_infos)} tensor_infos "
                f"vs {len(decomposed.tensor_data)} tensors"
            )
            return False
        
        # All checks passed
        logger.info(
            f"EC-CHECK: Decomposition validation passed ✓\n"
            f"  Component 1 (non-tensor dict): {len(decomposed.non_tensor_data)} keys\n"
            f"  Component 2 (tensor keys list): {len(decomposed.tensor_infos)} tensors\n"
            f"  Component 3 (tensor data list): {len(decomposed.tensor_data)} tensors"
        )
            
        # Log device info for verification
        if len(decomposed.tensor_infos) > 0:
            devices = set(info.device.type for info in decomposed.tensor_infos)
            logger.debug(f"EC-CHECK: Tensor devices: {devices}")
        
        return True

    def _get_save_and_finalize_callbacks(self, writer, save_state_dict_ret) -> AsyncRequest:
        save_fn_args = writer.get_save_function_and_args()
        save_fn, preload_fn, save_args = save_fn_args

        def finalize_fn():
            save_state_dict_async_finalize(*save_state_dict_ret)
            torch.distributed.barrier()

        return AsyncRequest(save_fn, save_args, [finalize_fn], preload_fn=preload_fn)

    def can_handle_sharded_objects(self):
        return True


def _get_filesystem_reader(
    checkpoint_dir: Union[str, Path], cache_metadata: bool = False
) -> FileSystemReader:
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.torch.MultiStorageFileSystemReader(checkpoint_dir, thread_count=2)

    if cache_metadata:
        return CachedMetadataFileSystemReader(checkpoint_dir)

    return FileSystemReader(checkpoint_dir)


def get_reformulation_metadata(
    sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
) -> Dict[str, TensorReformulationMetadata]:
    """Reads MCore data for N-D flattened tensors from checkpoint metadata during ckpt load.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict to load
        checkpoint_dir (Path): checkpoint directory

    Returns:
        Dict[str, TensorReformulationMetadata] - dictionary that maps keys of every
            N-D flattened tensor from the sharded_state_dict to its original global shape
            as stored in `mcore_data` in the checkpoint.
    """
    fs_reader = _get_filesystem_reader(checkpoint_dir)
    ckpt_metadata = fs_reader.read_metadata()
    reformulation_metadata = {}
    for sh_ten in nested_values(sharded_state_dict):
        if not is_nd_flattened_tensor(sh_ten):
            continue
        try:
            ckpt_global_shape = ckpt_metadata.mcore_data[sh_ten.key][
                'nd_reformulated_orig_global_shape'
            ]
        except KeyError as e:
            if len(sh_ten.global_shape) == 1:
                warnings.warn(
                    f'Legacy checkpoint format detected for 1-D flattened tensor {sh_ten}. '
                    'Skip metadata reformulation.'
                )
                continue
            raise CheckpointingException(
                f'Cannot find global shape metadata for N-D flattened tensor {sh_ten} '
                f'in checkpoint metadata: {ckpt_metadata.mcore_data}'
            ) from e

        reformulation_metadata[sh_ten.key] = TensorReformulationMetadata(
            ckpt_global_shape, ckpt_metadata.state_dict_metadata[sh_ten.key].size
        )
    return reformulation_metadata


class TorchDistLoadShardedStrategy(LoadShardedStrategy):
    """Basic load strategy for the PyT Distributed format."""

    def __init__(self):
        self.cached_global_metadata: Optional[Metadata] = None
        super().__init__()
        
        # Initialize EC-CHECK manager (singleton instance shared with Save strategy)
        self.eccheck_manager = ECCHECKManager()
        self.eccheck_manager.init_eccheck_if_enabled()
        
        # Initialize strategy-specific EC-CHECK state
        self.eccheck_p2p_buffers = None
    
    def _get_p2p_partner_rank(self, my_rank: int, world_size: int) -> int:
        """Get P2P partner rank using the shared manager."""
        return self.eccheck_manager.get_p2p_partner_rank(my_rank, world_size)
    
    def _is_eccheck_checkpoint(self, checkpoint_dir: Path) -> bool:
        """Check if the checkpoint is in EC-CHECK format.
        
        EC-CHECK checkpoints are .distcp files with 'ECCK' magic number in the header.
        
        Args:
            checkpoint_dir (Path): checkpoint directory
            
        Returns:
            bool: True if this is an EC-CHECK checkpoint
        """
        import struct
        
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return False
        
        # Get current rank to find the corresponding file
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Check for EC-CHECK format: __{rank}_0.distcp with ECCK magic number
        potential_file = checkpoint_dir / f"__{rank}_0.distcp"
        
        if not potential_file.exists():
            return False
        
        # Read first 4 bytes to check for ECCK magic number
        try:
            with open(potential_file, 'rb') as f:
                magic = f.read(4)
                return magic == b'ECCK'
        except:
            return False
    
    def _restore_dict_types_lenient(self, x: Union[dict, list, Any], keys_template: Union[dict, list, Any]):
        """Lenient version of _restore_dict_types that skips missing keys.
        
        This is needed for EC-CHECK where different ranks may have different keys.
        """
        if isinstance(keys_template, dict):
            if not isinstance(x, dict):
                return
            
            for k, v in keys_template.items():
                # Convert non-string keys
                if not isinstance(k, str):
                    str_k = str(k)
                    if str_k in x:
                        x[k] = x.pop(str_k)
                    else:
                        # Key doesn't exist - skip it
                        continue
                
                # Recursively restore types if key exists
                if k in x:
                    self._restore_dict_types_lenient(x[k], v)
                # If key doesn't exist, just skip it (lenient behavior)
                
        elif isinstance(keys_template, list):
            if not isinstance(x, list):
                return
            for x_val, templ_val in zip(x, keys_template):
                self._restore_dict_types_lenient(x_val, templ_val)
    
    def _load_ecccheck_p2p_checkpoint(self, checkpoint_dir: Path) -> Tuple:
        from .filesystem_async import EccheckMappedFile
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        p2p_partner_rank = self._get_p2p_partner_rank(rank, world_size)
        
        checkpoint_dir = Path(checkpoint_dir)
        eccheck_p2p_own_file = checkpoint_dir / f"__{rank}_p2p_own.distcp"
        eccheck_p2p_partner_file = checkpoint_dir / f"__{p2p_partner_rank}_p2p_partner.distcp"
        
        if not eccheck_p2p_own_file.exists():
            mapped_file_own = EccheckMappedFile(None, None, None, None, None)
            mapped_file_partner = EccheckMappedFile(None, None, None, None, None)
            return mapped_file_own, mapped_file_partner
        
        # Load the decomposed state dict from file
        # Returns tuple: (EccheckMappedFile, non_tensor_data, List[TensorMetadata])
        mapped_file_own = FileSystemWriterAsync.load_eccheck_bytes_from_file(
            str(eccheck_p2p_own_file), my_rank=rank
        )
        mapped_file_partner = FileSystemWriterAsync.load_eccheck_bytes_from_file(
            str(eccheck_p2p_partner_file), my_rank=p2p_partner_rank
        )
        
        # Package both together
        local_package = {
            'tensor_metadata': mapped_file_partner.local_metadata,
            'non_tensor_data': mapped_file_partner.non_tensor_data,
        }
        
        # ===== Step 2: All-gather complete metadata using all_gather_object =====
        # This automatically handles serialization, padding, and deserialization
        # Transmits both tensor_metadata and non_tensor_data
        all_packages = [None] * world_size
        torch.distributed.all_gather_object(all_packages, local_package)

        rank_metadata = {}
        rank_non_tensor_data = {}
        for i, package in enumerate(all_packages):
            rank_metadata[i] = package['tensor_metadata']
            rank_non_tensor_data[i] = package['non_tensor_data']       

        # Create registry with both tensor and non-tensor metadata
        from .state_dict_decomposer import GlobalMetadataRegistry
        registry = GlobalMetadataRegistry(
            rank_metadata=rank_metadata,
            rank_non_tensor_data=rank_non_tensor_data
        )
        
        # ===== Step 3: Prepare P2P buffers (own_buffer and partner_buffer) =====
        # Check if buffers exist in manager and can be reused, or allocate new ones
        if self.eccheck_manager.eccheck_p2p_buffers is not None:
            existing_buffers = self.eccheck_manager.eccheck_p2p_buffers
            existing_own_size = existing_buffers['own_buffer'].numel()
            existing_partner_size = existing_buffers['partner_buffer'].numel()
            
            # Calculate required buffer sizes from registry
            own_metadata = registry.rank_metadata.get(rank, [])
            own_total_size = sum(meta.size_bytes for meta in own_metadata)
            partner_metadata = registry.rank_metadata.get(p2p_partner_rank, [])
            partner_total_size = sum(meta.size_bytes for meta in partner_metadata)
            
            # Calculate maximum size across all ranks (for pipeline synchronization)
            all_total_bytes_list = []
            for r in range(world_size):
                rank_metadata = registry.rank_metadata.get(r, [])
                rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
                all_total_bytes_list.append(rank_total_size)
            max_total_bytes = max(all_total_bytes_list)
            
            # Calculate aligned sizes
            eccheck_buffer_size = self.eccheck_manager.eccheck_buffer_size
            needed_own_size = ((max_total_bytes + eccheck_buffer_size - 1) // eccheck_buffer_size) * eccheck_buffer_size
            needed_partner_size = needed_own_size  # Same size for pipeline sync
            
            if (existing_own_size >= needed_own_size and 
                existing_partner_size >= needed_partner_size):
                # Reuse existing buffers
                logger.info(
                    f"EC-CHECK: Reusing existing P2P buffers from manager "
                    f"(own: {existing_own_size / (1024**3):.2f} GB >= {needed_own_size / (1024**3):.2f} GB, "
                    f"partner: {existing_partner_size / (1024**3):.2f} GB >= {needed_partner_size / (1024**3):.2f} GB)"
                )
                self.eccheck_p2p_buffers = existing_buffers
            else:
                # Existing buffers too small, reallocate
                logger.info(
                    f"EC-CHECK: Existing buffers too small, reallocating "
                    f"(own: {existing_own_size / (1024**3):.2f} GB < {needed_own_size / (1024**3):.2f} GB or "
                    f"partner: {existing_partner_size / (1024**3):.2f} GB < {needed_partner_size / (1024**3):.2f} GB)"
                )
                self.eccheck_p2p_buffers = self._allocate_p2p_buffers(registry)
                self.eccheck_manager.eccheck_p2p_buffers = self.eccheck_p2p_buffers
        else:
            # First-time allocation (e.g., after process restart)
            logger.info("EC-CHECK: Allocating P2P buffers from checkpoint metadata")
            self.eccheck_p2p_buffers = self._allocate_p2p_buffers(registry)
            self.eccheck_manager.eccheck_p2p_buffers = self.eccheck_p2p_buffers
        
        paired_rank = self._get_p2p_partner_rank(rank, world_size)
        
        # Get self metadata form peer rank in global registry
        metadata_in_peer = registry.rank_metadata.get(paired_rank, [])
        meta_type = metadata_in_peer[0].chunk_type
        recv_total_size = sum(meta.size_bytes for meta in metadata_in_peer)
    
        recv_own_buffer = torch.empty(recv_total_size, dtype=torch.uint8)
        
        # Simple P2P exchange placeholder. This will be extended into a full
        # EC-CHECK recovery pipeline (encoding + XOR + P2P) in later steps.
        self._run_eccheck_p2p_pipeline_simple(
            rank=rank,
            world_size=world_size,
            registry=registry,
            mapped_file_own=mapped_file_own,
            mapped_file_partner=mapped_file_partner,
            recv_own_buffer=recv_own_buffer,
            recv_total_size=recv_total_size,
        )
        
        # Return EccheckMappedFile, non_tensor_data, and local_metadata for each file
        return mapped_file_own, mapped_file_partner
    
    def _run_eccheck_p2p_pipeline_simple(
        self,
        rank: int,
        world_size: int,
        registry,
        mapped_file_own,
        mapped_file_partner,
        recv_own_buffer: torch.Tensor,
        recv_total_size: int,
    ) -> None:
        """EC-CHECK recovery pipeline for rank2 single-failure scenario.

        This implements a chunked pipeline that drives C++ encoding, XOR, and P2P
        workers to recover lost data. The pipeline structure mirrors the save-side
        implementation for consistency.
        """
        import queue
        import ctypes
        import mmap
        
        # === Step 1: Prepare recv_encoding_buffers (if not already allocated) ===
        if self.eccheck_manager.eccheck_recv_encoding_buffers is None:
            logger.info("EC-CHECK: Allocating recv_encoding_buffers for load pipeline")
            self.eccheck_manager.eccheck_recv_encoding_buffers = (
                self.eccheck_manager.allocate_recv_encoding_buffers_phase2(registry)
            )
        
        recv_buffer_thread1, recv_buffer_thread2 = self.eccheck_manager.eccheck_recv_encoding_buffers
        recv_buffer_base_addr_thread1 = int(recv_buffer_thread1.data_ptr())
        recv_buffer_base_addr_thread2 = int(recv_buffer_thread2.data_ptr())
        recv_buffer_offset_thread1 = 0
        recv_buffer_offset_thread2 = 0
        
        # === Step 2: Buffer helper functions (from ECCHECKManager) ===
        mgr = self.eccheck_manager
        
        def get_free_data_buffer():
            mgr._poll_and_release_buffers()
            try:
                return mgr._free_data_buffer_queue.get(timeout=5.0)
            except queue.Empty:
                logger.error("EC-CHECK: TIMEOUT waiting for free data buffer - possible deadlock!")
                return mgr._free_data_buffer_queue.get()
        
        def get_free_encoding_buffer():
            mgr._poll_and_release_buffers()
            try:
                return mgr._free_encoding_buffer_queue.get(timeout=5.0)
            except queue.Empty:
                logger.error("EC-CHECK: TIMEOUT waiting for free encoding buffer - possible deadlock!")
                return mgr._free_encoding_buffer_queue.get()
        
        def get_free_parity_buffer():
            mgr._poll_and_release_buffers()
            try:
                return mgr._free_parity_buffer_queue.get(timeout=5.0)
            except queue.Empty:
                logger.error("EC-CHECK: TIMEOUT waiting for free parity buffer - possible deadlock!")
                return mgr._free_parity_buffer_queue.get()
        
        # === Step 3: P2P buffers base addresses ===
        p2p_own_buffer_base_addr = 0
        p2p_partner_buffer_base_addr = 0
        p2p_own_buffer_offset = 0
        p2p_partner_buffer_offset = 0
        
        if self.eccheck_p2p_buffers is not None:
            own_buffer = self.eccheck_p2p_buffers['own_buffer']
            partner_buffer = self.eccheck_p2p_buffers['partner_buffer']
            p2p_own_buffer_base_addr = int(own_buffer.data_ptr())
            p2p_partner_buffer_base_addr = int(partner_buffer.data_ptr())
        
        # === Step 4: Set load mode in C++ native module ===
        # For rank2 recovery scenario, set failed_rank=2
        # TODO: In the future, this could be determined dynamically based on which rank failed
        failed_rank = 2  # Hardcoded for now
        self.eccheck_manager._eccheck_native.set_load_mode(True, failed_rank)
        logger.info(f"EC-CHECK: Set load mode (failed_rank={failed_rank})")
        
        # === Step 5: Compute unified total_bytes for pipeline synchronization ===
        all_total_bytes_list = []
        for r in range(world_size):
            rank_metadata = registry.rank_metadata.get(r, [])
            rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
            all_total_bytes_list.append(rank_total_size)
        
        if len(all_total_bytes_list) == 0:
            return
        
        max_total_bytes = max(all_total_bytes_list)
        eccheck_buffer_size = self.eccheck_manager.eccheck_buffer_size
        total_bytes = max_total_bytes
        
        # === Step 5: Determine data source based on rank role ===
        # For rank2 recovery scenario:
        # - rank0/3: read from mapped_file_own (their own data/parity)
        # - rank1/2: read from mapped_file_partner (received from step2)
        if rank == 0 or rank == 3:
            source_mmap = mapped_file_own.mmap_object if mapped_file_own.mmap_object is not None else None
            source_file_size = mapped_file_own.file_size if mapped_file_own.file_size is not None else 0
        else:
            source_mmap = mapped_file_partner.mmap_object if mapped_file_partner.mmap_object is not None else None
            source_file_size = mapped_file_partner.file_size if mapped_file_partner.file_size is not None else 0
        
        # Calculate actual data size (skip header: 32 bytes + Component 1 + Component 2)
        # Component 3 (tensor buffer) starts after header + Component 1 + Component 2
        # For simplicity, we'll read from the tensor buffer portion directly
        # The header parsing is already done in load_eccheck_bytes_from_file
        # We need to find the offset where Component 3 (tensor buffer) starts
        tensor_buffer_start_offset = 32  # After header
        if source_mmap is not None:
            # Parse header to get Component 1 and Component 2 sizes
            header_bytes = source_mmap[:32]
            import struct
            magic, non_tensor_size, tensor_keys_size, tensor_buffer_size = struct.unpack('4sQQQ', header_bytes)
            tensor_buffer_start_offset = 32 + non_tensor_size + tensor_keys_size
        
        # === Step 6: Pre-process partner_file for rank0/3 (fill zeros before pipeline) ===
        # Similar to save stage: read actual data and pad with zeros to max_total_bytes
        # This ensures pipeline can always send full chunks without boundary checks
        if (rank == 0 or rank == 3) and mapped_file_partner.mmap_object is not None:
            # Parse partner_file header to get tensor buffer info
            partner_header_bytes = mapped_file_partner.mmap_object[:32]
            import struct
            partner_magic, partner_non_tensor_size, partner_tensor_keys_size, partner_tensor_buffer_size = struct.unpack('4sQQQ', partner_header_bytes)
            partner_tensor_buffer_start_offset = 32 + partner_non_tensor_size + partner_tensor_keys_size
            
            # Get partner_buffer for preprocessing
            if self.eccheck_p2p_buffers is not None:
                partner_buffer = self.eccheck_p2p_buffers['partner_buffer']
                
                # Read actual data from partner_file into partner_buffer
                partner_actual_bytes = min(partner_tensor_buffer_size, max_total_bytes)
                if partner_actual_bytes > 0:
                    if mapped_file_partner.memory_address is not None:
                        partner_source_addr = mapped_file_partner.memory_address + partner_tensor_buffer_start_offset
                        partner_dest_addr = p2p_partner_buffer_base_addr
                        
                        # Copy actual data using ctypes
                        source_ptr = ctypes.cast(partner_source_addr, ctypes.POINTER(ctypes.c_uint8))
                        dest_ptr = ctypes.cast(partner_dest_addr, ctypes.POINTER(ctypes.c_uint8))
                        ctypes.memmove(dest_ptr, source_ptr, partner_actual_bytes)
                    else:
                        # Fallback: use mmap slice and torch
                        partner_source_data = mapped_file_partner.mmap_object[
                            partner_tensor_buffer_start_offset:partner_tensor_buffer_start_offset + partner_actual_bytes
                        ]
                        import numpy as np
                        np_array = np.frombuffer(partner_source_data, dtype=np.uint8)
                        partner_buffer[:partner_actual_bytes].copy_(torch.from_numpy(np_array))
                
                    # Fill remaining space with zeros (for pipeline synchronization)
                    if partner_actual_bytes < max_total_bytes:
                        padding_size = max_total_bytes - partner_actual_bytes
                        partner_buffer[partner_actual_bytes:max_total_bytes].fill_(0)
                        logger.debug(
                            f"EC-CHECK: [Rank {rank}] Pre-filled partner_buffer with zeros "
                            f"({padding_size / (1024**2):.2f} MB padding)"
                        )
                
                logger.info(
                    f"EC-CHECK: [Rank {rank}] Pre-processed partner_file: "
                    f"{partner_actual_bytes / (1024**2):.2f} MB actual data, "
                    f"{(max_total_bytes - partner_actual_bytes) / (1024**2):.2f} MB padding"
                )
            else:
                logger.warning(f"EC-CHECK: [Rank {rank}] partner_buffer not available for preprocessing")
        elif (rank == 0 or rank == 3) and mapped_file_partner.memory_address is not None:
            # Test mode: Pre-process partner_file data
            if self.eccheck_p2p_buffers is not None:
                partner_buffer = self.eccheck_p2p_buffers['partner_buffer']
                partner_actual_bytes = min(mapped_file_partner.file_size, max_total_bytes)
                
                if partner_actual_bytes > 0:
                    # Copy from memory_address directly
                    source_ptr = ctypes.cast(mapped_file_partner.memory_address, ctypes.POINTER(ctypes.c_uint8))
                    dest_ptr = ctypes.cast(p2p_partner_buffer_base_addr, ctypes.POINTER(ctypes.c_uint8))
                    ctypes.memmove(dest_ptr, source_ptr, partner_actual_bytes)
                
                # Fill remaining space with zeros
                if partner_actual_bytes < max_total_bytes:
                    padding_size = max_total_bytes - partner_actual_bytes
                    partner_buffer[partner_actual_bytes:max_total_bytes].fill_(0)
                    logger.debug(
                        f"EC-CHECK: [Rank {rank}] Test mode: Pre-filled partner_buffer with zeros "
                        f"({padding_size / (1024**2):.2f} MB padding)"
                    )
        
        # === Step 7: Reset encoding completion flags and activate buffer poller ===
        self.eccheck_manager._eccheck_native.reset_encoding_completion_flags()
        if mgr._buffer_poller_active_event:
            mgr._buffer_poller_active_event.set()
            logger.info("EC-CHECK: Activated buffer poller for load pipeline")
        
        try:
            # === Step 8: Main pipeline loop ===
            processed = 0
            
            while processed < total_bytes:
                take = min(eccheck_buffer_size, total_bytes - processed)
                
                # Get free buffers
                cur_buffer_addr = get_free_data_buffer()
                enc_addr1 = get_free_encoding_buffer()
                enc_addr2 = get_free_encoding_buffer()
                parity_addr1 = get_free_parity_buffer()
                parity_addr2 = get_free_parity_buffer()
                
                # Calculate recv addresses (64-byte aligned)
                recv_buffer_offset_thread1_aligned = ((recv_buffer_offset_thread1 + 63) // 64) * 64
                recv_buffer_offset_thread2_aligned = ((recv_buffer_offset_thread2 + 63) // 64) * 64
                
                recv_addr_thread1 = recv_buffer_base_addr_thread1 + recv_buffer_offset_thread1_aligned
                recv_addr_thread2 = recv_buffer_base_addr_thread2 + recv_buffer_offset_thread2_aligned
                recv_chunk_size = take
                
                recv_buffer_offset_thread1 = recv_buffer_offset_thread1_aligned + recv_chunk_size
                recv_buffer_offset_thread2 = recv_buffer_offset_thread2_aligned + recv_chunk_size
                
                # Calculate P2P write addresses (64-byte aligned)
                if p2p_own_buffer_base_addr != 0:
                    p2p_own_buffer_offset_aligned = ((p2p_own_buffer_offset + 63) // 64) * 64
                    p2p_partner_buffer_offset_aligned = ((p2p_partner_buffer_offset + 63) // 64) * 64
                    
                    p2p_own_write_addr = p2p_own_buffer_base_addr + p2p_own_buffer_offset_aligned
                    p2p_partner_write_addr = p2p_partner_buffer_base_addr + p2p_partner_buffer_offset_aligned
                    
                    p2p_own_buffer_offset = p2p_own_buffer_offset_aligned + take
                    p2p_partner_buffer_offset = p2p_partner_buffer_offset_aligned + take
                else:
                    p2p_own_write_addr = 0
                    p2p_partner_write_addr = 0
                
                # Prepare Step 2 P2P transfer parameters
                step2_send_addr = 0
                step2_recv_data_addr = 0
                step2_size = 0
                
                if rank == 0 or rank == 3:
                    # Sender: use pre-processed partner_buffer (already filled with zeros)
                    if self.eccheck_p2p_buffers is not None:
                        # Calculate offset in partner_buffer
                        partner_buffer_offset = processed
                        step2_send_addr = p2p_partner_buffer_base_addr + partner_buffer_offset
                        step2_size = take  # Always send full chunk (zeros already filled in preprocessing)
                    else:
                        logger.warning(f"EC-CHECK: [Rank {rank}] partner_buffer not available, skipping Step2 P2P")
                        step2_send_addr = 0
                        step2_size = 0
                    
                    # rank0/3: copy from own_file to data_buffer before submitting pipeline
                    buffer_ptr = ctypes.cast(cur_buffer_addr, ctypes.POINTER(ctypes.c_uint8))
                    buffer_array = ctypes.cast(buffer_ptr, ctypes.POINTER(ctypes.c_uint8 * take))
                    
                    if source_mmap is not None and processed < source_file_size - tensor_buffer_start_offset:
                        # Real mmap file: Calculate source offset in Component 3 (tensor buffer)
                        source_offset = tensor_buffer_start_offset + processed
                        bytes_to_copy = min(take, source_file_size - source_offset)
                        
                        if bytes_to_copy > 0:
                            # Read from mmap
                            source_data = source_mmap[source_offset:source_offset + bytes_to_copy]
                            ctypes.memmove(buffer_array.contents, source_data, bytes_to_copy)
                            
                            # Pad with zeros if needed
                            if take > bytes_to_copy:
                                padding_size = take - bytes_to_copy
                                padding_ptr = ctypes.cast(
                                    ctypes.addressof(buffer_array.contents) + bytes_to_copy,
                                    ctypes.POINTER(ctypes.c_uint8)
                                )
                                ctypes.memset(padding_ptr, 0, padding_size)
                        else:
                            # Past actual data, fill with zeros
                            ctypes.memset(buffer_array.contents, 0, take)
                    elif mapped_file_own.memory_address is not None:
                        # Test mode: Copy directly from memory_address (no header offset for test data)
                        source_addr = mapped_file_own.memory_address + processed
                        bytes_to_copy = min(take, mapped_file_own.file_size - processed)
                        
                        if bytes_to_copy > 0:
                            source_ptr = ctypes.cast(source_addr, ctypes.POINTER(ctypes.c_uint8))
                            ctypes.memmove(buffer_array.contents, source_ptr, bytes_to_copy)
                            
                            # Pad with zeros if needed
                            if take > bytes_to_copy:
                                padding_size = take - bytes_to_copy
                                padding_ptr = ctypes.cast(
                                    ctypes.addressof(buffer_array.contents) + bytes_to_copy,
                                    ctypes.POINTER(ctypes.c_uint8)
                                )
                                ctypes.memset(padding_ptr, 0, padding_size)
                        else:
                            # Past actual data, fill with zeros
                            ctypes.memset(buffer_array.contents, 0, take)
                        logger.debug(f"EC-CHECK: [Rank {rank}] Test mode: Copied {bytes_to_copy} bytes from own_file to data_buffer")
                    else:
                        # No source data available, fill with zeros
                        logger.warning(f"EC-CHECK: [Rank {rank}] No source data available, filling data_buffer with zeros")
                        ctypes.memset(buffer_array.contents, 0, take)
                else:
                    # rank1/2: receive partner_file chunk into data_buffer via Step 2 P2P
                    step2_recv_data_addr = cur_buffer_addr
                    step2_size = take
                
                # Submit complete load pipeline chunk (Step2 P2P -> Encoding -> XOR -> Step6 P2P)
                # This single call handles the entire pipeline internally in C++
                self.eccheck_manager._eccheck_native.submit_load_pipeline_chunk(
                    step2_send_addr=step2_send_addr,           # rank0/3: partner_file chunk addr; rank1/2: 0
                    step2_recv_data_addr=step2_recv_data_addr,  # rank1/2: data_buffer addr; rank0/3: 0
                    step2_size=step2_size,                      # Step 2 transfer size
                    data_addr=cur_buffer_addr,                  # Data buffer address (own_file for rank0/3, received for rank1/2)
                    size=take,                                  # Data size
                    encoding_addr1=enc_addr1,                   # Thread1 encoding buffer
                    encoding_addr2=enc_addr2,                   # Thread2 encoding buffer
                    recv_addr_thread1=recv_addr_thread1,        # Thread1 receive address
                    recv_addr_thread2=recv_addr_thread2,        # Thread2 receive address
                    recv_chunk_size=recv_chunk_size,            # Receive chunk size
                    parity_addr1=parity_addr1,                  # Thread1 parity buffer
                    parity_addr2=parity_addr2,                  # Thread2 parity buffer
                    p2p_own_write_addr=p2p_own_write_addr,       # P2P own buffer write address
                    p2p_partner_write_addr=p2p_partner_write_addr  # P2P partner buffer write address
                )
                
                processed += take
            
            # === Step 8: Send sentinel and wait for completion ===
            logger.info("EC-CHECK: Load pipeline: Sending sentinel to encoding threads")
            # For rank 2 and rank 3 in load mode, only submit sentinel to thread2
            # (thread1 doesn't process any tasks in load mode for these ranks)
            if rank in [2, 3]:
                self.eccheck_manager._eccheck_native.submit_data_for_encoding_thread2(0, 0, 0, 0, 0, 0, 0, 0)
            else:
                self.eccheck_manager._eccheck_native.submit_data_for_encoding_thread1(0, 0, 0, 0, 0, 0, 0, 0)
                self.eccheck_manager._eccheck_native.submit_data_for_encoding_thread2(0, 0, 0, 0, 0, 0, 0, 0)
            
            logger.info("EC-CHECK: Load pipeline: Waiting for encoding threads to complete...")
            self.eccheck_manager._eccheck_native.wait_for_encoding_completion()
            
            # Wait for P2P workers to complete
            # Note: C++ should have wait_for_p2p_workers or similar, but for now we'll rely on
            # the encoding completion which should ensure P2P is done
            torch.cuda.synchronize()
            logger.info("EC-CHECK: Load pipeline: Pipeline completed")
            
        finally:
            # Deactivate buffer poller
            if mgr._buffer_poller_active_event:
                mgr._buffer_poller_active_event.clear()
                logger.info("EC-CHECK: Deactivated buffer poller after load pipeline")
    
    def prepare_for_load_pipeline_test(self):
        """
        Prepare environment for load pipeline testing.
        
        This function ensures all prerequisites are met:
        1. Distributed environment is initialized
        2. EC-CHECK is enabled via args
        3. ECCHECKManager is initialized
        4. C++ native module is ready
        
        Returns:
            bool: True if preparation successful, False otherwise
        """
        # === Step 1: Check distributed environment ===
        if not torch.distributed.is_initialized():
            logger.error("EC-CHECK TEST: Distributed environment not initialized!")
            logger.error("  Please initialize distributed environment first:")
            logger.error("    torch.distributed.init_process_group(...)")
            return False
        
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        
        if world_size < 4:
            logger.error(f"EC-CHECK TEST: Need at least 4 ranks, got {world_size}")
            return False
        
        logger.info(f"EC-CHECK TEST: Distributed environment OK (rank={rank}, world_size={world_size})")
        
        # === Step 2: Ensure args.use_eccheck is True ===
        # The init_eccheck_if_enabled() checks get_args().use_eccheck
        # We need to mock or set this before calling init_eccheck_if_enabled()
        try:
            from megatron.training import get_args as input_args
            args = input_args()
            
            # Set use_eccheck if not already set
            if not hasattr(args, 'use_eccheck') or not args.use_eccheck:
                logger.info("EC-CHECK TEST: Setting args.use_eccheck = True")
                args.use_eccheck = True
        except Exception as e:
            logger.warning(f"EC-CHECK TEST: Could not get/set args: {e}")
            logger.warning("  Will try to proceed anyway...")
        
        # === Step 3: Ensure ECCHECKManager is initialized ===
        # Create strategy instance (which initializes manager)
        if self.eccheck_manager is None:
            logger.error("EC-CHECK TEST: eccheck_manager is None!")
            return False
        
        # Force initialization if not already done
        if self.eccheck_manager._eccheck_native is None:
            logger.info("EC-CHECK TEST: Initializing ECCHECKManager...")
            try:
                self.eccheck_manager.init_eccheck_if_enabled()
            except Exception as e:
                logger.error(f"EC-CHECK TEST: Failed to initialize ECCHECKManager: {e}")
                return False
        
        # === Step 4: Verify C++ native module is ready ===
        if self.eccheck_manager._eccheck_native is None:
            logger.error("EC-CHECK TEST: C++ native module is None!")
            logger.error("  This usually means:")
            logger.error("    1. args.use_eccheck is False")
            logger.error("    2. Distributed environment not initialized")
            logger.error("    3. eccheck_native.so file not found")
            logger.error("    4. C++ module initialization failed")
            return False
        
        logger.info(f"EC-CHECK TEST: C++ native module is ready (rank={rank})")
        
        # === Step 5: Verify buffers are allocated ===
        if self.eccheck_manager.eccheck_data_buffers is None:
            logger.warning("EC-CHECK TEST: Data buffers not allocated yet (will be allocated during pipeline)")
        else:
            logger.info(f"EC-CHECK TEST: Data buffers allocated ({len(self.eccheck_manager.eccheck_data_buffers)} buffers)")
        
        logger.info(f"EC-CHECK TEST: Preparation complete for rank {rank}")
        return True
    
    def test_load_pipeline(self, test_data_size: int = 64 * 1024 * 1024):
        """
        Test load pipeline without requiring actual checkpoint files.
        
        This function creates mock data structures and directly calls the load pipeline
        to test the rank2 recovery flow. All 4 ranks will participate, but the pipeline
        will simulate rank2 failure recovery.
        
        Args:
            test_data_size (int): Size of test data in bytes (default: 64MB)
        """
        import mmap
        import ctypes
        import struct
        from .filesystem_async import EccheckMappedFile
        from .state_dict_decomposer import GlobalMetadataRegistry, TensorMetadata
        
        # === Preparation ===
        if not self.prepare_for_load_pipeline_test():
            logger.error("EC-CHECK TEST: Preparation failed, aborting test")
            return False
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 4
        
        if world_size < 4:
            logger.warning(f"EC-CHECK TEST: World size {world_size} < 4, test may not work correctly")
        
        logger.info(f"EC-CHECK TEST: Starting load pipeline test (rank={rank}, world_size={world_size}, test_data_size={test_data_size} bytes)")
        
        # === Step 1: Create mock TensorMetadata for all ranks ===
        # For simplicity, each rank has one tensor chunk
        rank_metadata = {}
        for r in range(world_size):
            # Create metadata for this rank
            # In real scenario, rank0/1 have data, rank2/3 have parity
            # For test, we'll create data for all ranks
            chunk_type = 'data' if r < 2 else 'parity'
            metadata = TensorMetadata(
                key=f"test_tensor_rank_{r}",
                shape=(test_data_size,),
                dtype='torch.uint8',
                size_bytes=test_data_size,
                global_offset=(0,),
                shard_index=0,
                chunk_type=chunk_type,
                target_rank=r,
                source_rank=r
            )
            rank_metadata[r] = [metadata]
        
        # Create GlobalMetadataRegistry
        registry = GlobalMetadataRegistry(
            rank_metadata=rank_metadata,
            rank_non_tensor_data={r: {} for r in range(world_size)}
        )
        
        logger.info(f"EC-CHECK TEST: Created registry with {len(rank_metadata)} ranks")
        
        # === Step 2: Create mock EccheckMappedFile for own_file and partner_file ===
        # We'll use torch tensors as data source, then create mmap-like objects
        
        # For own_file: rank0/3 have their own data/parity
        # For partner_file: rank0/3 have partner's data/parity (for Step2 P2P send)
        p2p_partner_rank = self._get_p2p_partner_rank(rank, world_size)
        
        # Create test data: fill with rank-specific pattern for verification
        own_data = torch.full((test_data_size,), rank, dtype=torch.uint8)
        partner_data = torch.full((test_data_size,), p2p_partner_rank, dtype=torch.uint8)
        
        # Create mmap-like objects from torch tensors
        # Use PyTorch's data_ptr() to get the actual memory address
        # This is more reliable than ctypes.addressof for torch tensors
        own_memory_address = int(own_data.data_ptr())
        partner_memory_address = int(partner_data.data_ptr())
        
        # Create EccheckMappedFile objects
        # Note: We use None for mmap_object since we're using torch tensor memory
        # The memory_address points to the actual data
        mapped_file_own = EccheckMappedFile(
            mmap_object=None,  # Not a real mmap, but we keep data in torch tensor
            memory_address=own_memory_address,
            file_size=test_data_size,
            local_metadata=rank_metadata.get(rank, []),
            non_tensor_data={}
        )
        
        mapped_file_partner = EccheckMappedFile(
            mmap_object=None,  # Not a real mmap, but we keep data in torch tensor
            memory_address=partner_memory_address,
            file_size=test_data_size,
            local_metadata=rank_metadata.get(p2p_partner_rank, []),
            non_tensor_data={}
        )
        
        # Keep references to prevent garbage collection
        mapped_file_own._data_ref = own_data
        mapped_file_partner._data_ref = partner_data
        
        logger.info(f"EC-CHECK TEST: Created mock mapped files (own_addr={own_memory_address}, partner_addr={partner_memory_address})")
        
        # === Step 3: Create recv_own_buffer ===
        # This is where rank2 will receive the recovered data
        recv_own_buffer = torch.empty(test_data_size, dtype=torch.uint8)
        recv_total_size = test_data_size
        
        logger.info(f"EC-CHECK TEST: Created recv_own_buffer (size={recv_total_size} bytes)")
        
        # === Step 4: Call the pipeline ===
        try:
            logger.info(f"EC-CHECK TEST: Calling _run_eccheck_p2p_pipeline_simple...")
            self._run_eccheck_p2p_pipeline_simple(
                rank=rank,
                world_size=world_size,
                registry=registry,
                mapped_file_own=mapped_file_own,
                mapped_file_partner=mapped_file_partner,
                recv_own_buffer=recv_own_buffer,
                recv_total_size=recv_total_size,
            )
            logger.info(f"EC-CHECK TEST: Pipeline completed successfully for rank {rank}")
            
            # === Step 5: Verify results (for rank2) ===
            if rank == 2:
                # Check if own_buffer contains recovered d2
                if self.eccheck_p2p_buffers is not None:
                    own_buffer = self.eccheck_p2p_buffers['own_buffer']
                    partner_buffer = self.eccheck_p2p_buffers['partner_buffer']
                    
                    logger.info(f"EC-CHECK TEST: Rank2 verification:")
                    logger.info(f"  own_buffer shape: {own_buffer.shape}, dtype: {own_buffer.dtype}")
                    logger.info(f"  partner_buffer shape: {partner_buffer.shape}, dtype: {partner_buffer.dtype}")
                    
                    # Check if buffers are not all zeros (basic sanity check)
                    own_nonzero = torch.count_nonzero(own_buffer).item()
                    partner_nonzero = torch.count_nonzero(partner_buffer).item()
                    
                    logger.info(f"  own_buffer non-zero elements: {own_nonzero} / {own_buffer.numel()}")
                    logger.info(f"  partner_buffer non-zero elements: {partner_nonzero} / {partner_buffer.numel()}")
                    
                    if own_nonzero == 0:
                        logger.warning("EC-CHECK TEST: WARNING - own_buffer is all zeros!")
                    if partner_nonzero == 0:
                        logger.warning("EC-CHECK TEST: WARNING - partner_buffer is all zeros!")
                else:
                    logger.warning("EC-CHECK TEST: Rank2 - eccheck_p2p_buffers is None")
            
            logger.info(f"EC-CHECK TEST: Test completed successfully for rank {rank}")
            return True
            
        except Exception as e:
            logger.error(f"EC-CHECK TEST: Pipeline failed with error: {e}", exc_info=True)
            raise
            if mgr._buffer_poller_active_event:
                mgr._buffer_poller_active_event.clear()
                logger.info("EC-CHECK: Deactivated buffer poller after load pipeline completion")
            
            # Final poll to ensure all buffers are released
            mgr._poll_and_release_buffers()

    def _allocate_p2p_buffers(self, global_registry):
        """
        Allocate TWO large continuous buffers for P2P stage (load phase).
        This is a simplified version that only allocates buffers without creating write buckets.
        
        Args:
            global_registry (GlobalMetadataRegistry): Complete metadata from all ranks
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'own_buffer' and 'partner_buffer'
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        p2p_partner_rank = self._get_p2p_partner_rank(rank, world_size)
        
        # ===== Get own data size from metadata =====
        own_metadata = global_registry.rank_metadata.get(rank, [])
        own_total_size = sum(meta.size_bytes for meta in own_metadata)
        
        # ===== Get P2P partner's data size from metadata =====
        partner_metadata = global_registry.rank_metadata.get(p2p_partner_rank, [])
        partner_total_size = sum(meta.size_bytes for meta in partner_metadata)
        
        # ===== Calculate maximum data size across all ranks (for pipeline synchronization) =====
        if torch.distributed.is_initialized():
            all_total_bytes_list = []
            for r in range(world_size):
                rank_metadata = global_registry.rank_metadata.get(r, [])
                rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
                all_total_bytes_list.append(rank_total_size)
            max_total_bytes = max(all_total_bytes_list)
        else:
            max_total_bytes = max(own_total_size, partner_total_size)
        
        # ===== Align both sizes to buffer_size (64MB) using maximum for pipeline sync =====
        eccheck_buffer_size = self.eccheck_manager.eccheck_buffer_size
        own_pipeline_size = max_total_bytes
        partner_pipeline_size = max_total_bytes
        own_aligned_size = ((own_pipeline_size + eccheck_buffer_size - 1) // eccheck_buffer_size) * eccheck_buffer_size
        partner_aligned_size = ((partner_pipeline_size + eccheck_buffer_size - 1) // eccheck_buffer_size) * eccheck_buffer_size
        
        logger.info(
            f"EC-CHECK: Allocating P2P buffers for load phase based on metadata\n"
            f"  P2P partner rank: {p2p_partner_rank}\n"
            f"  Own data size: {own_total_size / (1024**3):.2f} GB "
            f"(actual), {max_total_bytes / (1024**3):.2f} GB (pipeline max), "
            f"{own_aligned_size / (1024**3):.2f} GB (aligned)\n"
            f"  Partner data size: {partner_total_size / (1024**3):.2f} GB "
            f"(actual), {max_total_bytes / (1024**3):.2f} GB (pipeline max), "
            f"{partner_aligned_size / (1024**3):.2f} GB (aligned)\n"
            f"  Total P2P memory: {(own_aligned_size + partner_aligned_size) / (1024**3):.2f} GB"
        )
        
        # ===== Allocate two large continuous buffers =====
        own_buffer = torch.empty(own_aligned_size, dtype=torch.uint8)
        partner_buffer = torch.empty(partner_aligned_size, dtype=torch.uint8)
        
        logger.info(
            f"EC-CHECK: Allocated P2P buffers for load phase:\n"
            f"  Own buffer: {own_aligned_size / (1024**3):.2f} GB "
            f"({own_aligned_size / (1024**2):.0f} MB)\n"
            f"  Partner buffer: {partner_aligned_size / (1024**3):.2f} GB "
            f"({partner_aligned_size / (1024**2):.0f} MB)"
        )
        
        return {
            'own_buffer': own_buffer,
            'partner_buffer': partner_buffer,
        }
    
    def _load_eccheck_metadata_broadcast(self, checkpoint_dir: Path) -> Tuple[Dict[str, Any], List[TensorMetadata]]:
        """
        Broadcast the metadata from the mapped file to all ranks.
        """
        return self._load_ecccheck_p2p_checkpoint(checkpoint_dir)

    def _load_eccheck_checkpoint(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Load checkpoint saved in EC-CHECK format.
        
        Args:
            sharded_state_dict (ShardedStateDict): template showing what to load
            checkpoint_dir (Path): checkpoint directory
            
        Returns:
            StateDict: loaded state dict with structure matching sharded_state_dict
        """
        from .filesystem_async import FileSystemWriterAsync
        from .state_dict_decomposer import reconstruct_state_dict
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Find the EC-CHECK file for this rank
        # Ensure checkpoint_dir is a Path object
        checkpoint_dir = Path(checkpoint_dir)
        # EC-CHECK uses standard .distcp extension but with custom ECCK format
        eccheck_file = checkpoint_dir / f'__{rank}_0.distcp'
        
        if not eccheck_file.exists():
            raise FileNotFoundError(
                f"EC-CHECK file not found for rank {rank}: {eccheck_file}"
            )
        
        logger.info(f"Loading EC-CHECK checkpoint from {eccheck_file}")
        
        # Load the decomposed state dict from file
        decomposed = FileSystemWriterAsync.load_eccheck_components_from_file(
            str(eccheck_file)
        )

        # Build index map from loaded tensor_infos
        # Map: (fqn, global_offset) → (tensor_info, tensor_data)
        logger.info(f"EC-CHECK: Building index map from {len(decomposed.tensor_infos)} tensor infos")
        
        index_to_data = {}
        for info, tensor in zip(decomposed.tensor_infos, decomposed.tensor_data):
            # Use (fqn, global_offset) as unique key
            # global_offset is already a tuple from TensorInfo
            index_key = (info.key, info.global_offset)
            index_to_data[index_key] = (info, tensor)
        
        # Also add non-tensor data keyed by FQN only
        non_tensor_by_fqn = decomposed.non_tensor_data
        
        logger.info(
            f"Successfully loaded EC-CHECK checkpoint for rank {rank} "
            f"({len(decomposed.tensor_data)} tensors, "
            f"{decomposed.total_tensor_size_bytes / (1024**3):.2f} GB)"
        )
        
        # Save original sharded_state_dict for type restoration later
        orig_sharded_state_dict = sharded_state_dict
        
        # Generate PyT-compatible state dict from sharded_state_dict
        # This creates the structure that standard load expects
        # IMPORTANT: Do NOT use keep_only_main_replica=True, as standard load doesn't use it
        (keyed_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        )
        
        # Create a mapping from loaded data using metadata_index
        # For tensors: use metadata_index for precise matching
        # For objects: use FQN
        matched_count = 0
        unmatched_count = 0
        
        for key, sh_base_list in keyed_state_dict.items():
            for sh_base in sh_base_list:
                if isinstance(sh_base, ShardedObject):
                    # For ShardedObject, match by FQN
                    if key in non_tensor_by_fqn:
                        value = non_tensor_by_fqn[key]
                        
                        # Handle EC-CHECK wrapped BytesIO data
                        if isinstance(value, dict) and '_eccheck_type' in value:
                            if value['_eccheck_type'] == 'BytesIO':
                                # Reconstruct and deserialize
                                bytes_data = value['_eccheck_data']
                                bytes_io = io.BytesIO(bytes_data)
                                deserialized_list = torch.load(bytes_io, map_location='cpu', weights_only=False)
                                value = deserialized_list[0] if isinstance(deserialized_list, list) else deserialized_list
                        
                        sh_base.data = value
                        matched_count += 1
                    else:
                        unmatched_count += 1
                
                elif isinstance(sh_base, ShardedTensor):
                    # For ShardedTensor, match by (fqn, global_offset)
                    sh_offset = tuple(sh_base.global_offset) if hasattr(sh_base.global_offset, '__iter__') else (sh_base.global_offset,)
                    
                    # Construct lookup key
                    lookup_key = (key, sh_offset)
                    
                    if lookup_key in index_to_data:
                        # Direct match found!
                        info, tensor = index_to_data[lookup_key]
                        sh_base.data = tensor
                        matched_count += 1
                        
                        # Verify the data is not None
                        if tensor is None:
                            logger.error(f"EC-CHECK: Matched key {lookup_key} but tensor is None!")
                    else:
                        unmatched_count += 1
                        # Keep data as None - this is normal in distributed checkpoints
                        # Different ranks have different parameter shards
        
        logger.info(f"EC-CHECK: Matched {matched_count} ShardedBase objects")
        
        # Step 1: Unwrap ShardedTensors and ShardedObjects
        # Convert from ShardedBase objects to actual data
        unwrapped_state_dict = {}
        for key, sh_base_list in keyed_state_dict.items():
            if len(sh_base_list) == 0:
                continue
            
            sh_base = sh_base_list[0]
            if isinstance(sh_base, ShardedTensor):
                # For ShardedTensor, unwrap and handle prepend_axis_num
                # Similar to _unwrap_pyt_sharded_tensor
                tensors = []
                for sh in sh_base_list:
                    ten = sh.data
                    if ten is None:
                        tensors.append(None)
                        continue
                    
                    # Handle flattened_range (similar to _unwrap_pyt_sharded_tensor)
                    if sh.flattened_range is not None:
                        assert ten.shape[:-1] == (1,) * (len(ten.shape) - 1), ten.shape
                        ten = ten.view(-1)
                    else:
                        # Squeeze prepend_axis_num dimensions
                        for _ in range(sh.prepend_axis_num):
                            if ten.size(0) == 1:
                                ten = ten[0]
                    
                    tensors.append(ten)
                
                unwrapped_state_dict[key] = tensors
            elif isinstance(sh_base, ShardedObject):
                # For ShardedObject, collect data into a list (must match rename_mapping length)
                # Standard format expects List[data] for ShardedObjects
                data_list = [sh.data for sh in sh_base_list]
                unwrapped_state_dict[key] = data_list
        
        # Step 2: Convert keyed keys back to original state_dict keys
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
            unwrapped_state_dict, flat_mapping, rename_mapping  # type: ignore[arg-type]
        )
        
        # Step 3: Restore dict types (convert string keys back to original types if needed)
        # Note: Use a lenient version that skips missing keys
        self._restore_dict_types_lenient(mcore_state_dict, orig_sharded_state_dict)
        
        return mcore_state_dict
    
    def _populate_sharded_base_objects(self, sharded_state_dict: ShardedStateDict, flat_state_dict: Dict[str, Any]) -> None:
        """Populate ShardedBase objects in sharded_state_dict with data from flat_state_dict.
        
        Args:
            sharded_state_dict: nested dict containing ShardedTensor/ShardedObject (modified in-place)
            flat_state_dict: flat dict with FQN keys and loaded data
        """
        import io
        
        # Recursively find and populate all ShardedBase objects
        for sh_base in nested_values(sharded_state_dict):
            if not isinstance(sh_base, ShardedBase):
                continue
            
            # Get the key for this ShardedBase object
            if isinstance(sh_base, ShardedObject):
                key = sh_base.unique_key
            else:
                key = sh_base.key
            
            # Find matching data in flat_state_dict
            if key not in flat_state_dict:
                logger.warning(f"Key {key} not found in loaded data")
                continue
            
            value = flat_state_dict[key]
            
            # Handle EC-CHECK wrapped BytesIO data
            if isinstance(value, dict) and '_eccheck_type' in value:
                if value['_eccheck_type'] == 'BytesIO':
                    # Reconstruct BytesIO from stored bytes
                    bytes_data = value['_eccheck_data']
                    bytes_io = io.BytesIO(bytes_data)
                    # For ShardedObject, deserialize the BytesIO content
                    # The BytesIO contains torch.save'd list of data
                    deserialized_list = torch.load(bytes_io, map_location='cpu', weights_only=False)
                    # Extract the first element (standard format is [data])
                    value = deserialized_list[0] if isinstance(deserialized_list, list) else deserialized_list
            
            # Assign data to ShardedBase object
            sh_base.data = value

    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Translates MCore ShardedTensors to PyT ShardedTensors & loads from PyT Distributed fmt.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict with mapping
                information to instruct loading
            checkpoint_dir (Path): checkpoint directory

        Returns: loaded state dict
        """
        # Check if this is an EC-CHECK format checkpoint
        if self._is_eccheck_checkpoint(checkpoint_dir):
            logger.info(f"Detected EC-CHECK format checkpoint at {checkpoint_dir}")
            mapped_file_own, mapped_file_partner = self._load_ecccheck_p2p_checkpoint(checkpoint_dir)
            
            
            return self._load_eccheck_checkpoint(sharded_state_dict, checkpoint_dir)
        
        # Apply N-D tensors resharding
        reformulation_metadata = get_reformulation_metadata(sharded_state_dict, checkpoint_dir)
        sharded_state_dict, formulation_restore_data = apply_nd_flattened_tensors_reformulation(
            sharded_state_dict, reformulation_metadata
        )

        # Check if there are legacy 1-D flattened tensors in the checkpoint
        has_legacy_1d_flattened_tensors = False
        for sh_ten in nested_values(sharded_state_dict):
            if is_nd_flattened_tensor(sh_ten) and sh_ten.key not in reformulation_metadata:
                has_legacy_1d_flattened_tensors = True
                break

        flexible_shape_sharded_tensors = [
            sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and not sh_ten.allow_shape_mismatch
        ]
        allow_shape_mismatch_sharded_tensors = {
            sh_ten.key: sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and sh_ten.allow_shape_mismatch
        }

        orig_sharded_state_dict = sharded_state_dict
        # MCore state dict to PyT Distributed compatible
        (sharded_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        )
        pyt_state_dict = mcore_to_pyt_state_dict(
            sharded_state_dict, True, load_legacy_1d_flatten_tensors=has_legacy_1d_flattened_tensors
        )
        # Load PyT Distributed format
        fsr = _get_filesystem_reader(checkpoint_dir, cache_metadata=True)
        checkpoint.load_state_dict(
            pyt_state_dict,
            fsr,
            planner=MCoreLoadPlanner(
                shapes_validation_sharded_tensors=flexible_shape_sharded_tensors,
                allow_shape_mismatch_sharded_tensors=allow_shape_mismatch_sharded_tensors,
            ),
        )

        self.cached_global_metadata = (
            fsr.read_metadata()
        )  # no storage interaction thanks to caching

        pyt_state_dict = cast(
            Dict[str, Union[TorchShardedTensor, List[io.BytesIO]]], pyt_state_dict
        )
        # Unwrap ShardedTensors and return to original state dict
        mcore_state_dict = {
            k: v if not isinstance(v, TorchShardedTensor) else _unwrap_pyt_sharded_tensor(v)
            for k, v in pyt_state_dict.items()
        }
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
            mcore_state_dict, flat_mapping, rename_mapping  # type: ignore[arg-type]
        )
        _restore_dict_types(mcore_state_dict, orig_sharded_state_dict)
        # Apply N-D tensors resharding postprocessing
        mcore_state_dict = restore_nd_flattened_tensors_formulation(
            mcore_state_dict, formulation_restore_data
        )
        return mcore_state_dict

    def load_tensors_metadata(self, checkpoint_dir: Path, metadata: Metadata = None):
        """Uses tensors metadata stored in the metadata file."""
        if metadata is None:
            fs_reader = _get_filesystem_reader(checkpoint_dir)
            metadata = fs_reader.read_metadata()

        mcore_data = getattr(metadata, 'mcore_data', {})
        sharded_metadata = {}
        for k, tp in metadata.state_dict_metadata.items():
            if not isinstance(tp, TensorStorageMetadata):
                continue  # load only tensors

            nd_orig_global_shape = mcore_data.get(k, {}).get('nd_reformulated_orig_global_shape')
            if nd_orig_global_shape is None:
                # Regular tensor
                sharded_metadata[k] = ShardedTensor.from_rank_offsets(
                    k, torch.empty(tp.size, **tp.properties.__dict__, device='meta')
                ).without_data()
            else:
                # N-D flattened tensor
                unflat_ten = torch.empty(
                    nd_orig_global_shape, **tp.properties.__dict__, device='meta'
                )
                flat_ten = unflat_ten.flatten()
                sharded_metadata[k] = ShardedTensor.from_rank_offsets_flat(
                    k,
                    flat_ten,
                    unflat_ten.shape,
                    flattened_range=slice(0, unflat_ten.numel()),  # whole slice
                ).without_data()

        return sharded_metadata

    def load_sharded_metadata(self, checkpoint_dir: Path) -> ShardedStateDict:
        """Uses tensors and objects metadata stored in the metadata file."""
        fs_reader = _get_filesystem_reader(checkpoint_dir)
        metadata = fs_reader.read_metadata()

        sharded_metadata = {}
        for metadata_key, storage_metadata in metadata.state_dict_metadata.items():
            if not isinstance(storage_metadata, BytesStorageMetadata):
                continue
            sh_obj = ShardedObject.empty_from_unique_key(metadata_key)
            sharded_metadata[sh_obj.unique_key] = sh_obj

        sharded_metadata.update(self.load_tensors_metadata(checkpoint_dir, metadata))
        return sharded_metadata

    def remove_sharded_tensors(self, checkpoint_dir: str, key_prefix: str):
        """Removes checkpoint files whose keys have the given prefix.

        Performs the following steps:
        1. checks whether there are files that start with the key_prefix
        2. loads metadata
        3. removes all entries from the metadata that start with the key_prefix
        4. resaves the new metadata and removes the old metadata
        5. removes the relevant files
        """

        assert is_torch_min_version(
            "2.3.0"
        ), f'torch >= 2.3.0 is required for remove_sharded_tensors'

        distckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith("distcp")]
        files_to_remove = [f for f in distckpt_files if f.startswith(key_prefix)]

        if not files_to_remove:
            warnings.warn(
                f'There are no files in {checkpoint_dir} that begin with "{key_prefix}".'
                f' Skipping removal.'
            )
            return

        fs_reader = FileSystemReader(checkpoint_dir)
        original_metadata = fs_reader.read_metadata()

        new_state_dict_metadata = {}
        new_planner_data = {}
        new_storage_data = {}
        for k in original_metadata.state_dict_metadata.keys():
            if k.startswith(key_prefix):
                continue
            new_state_dict_metadata[k] = original_metadata.state_dict_metadata[k]
        original_planner_data = original_metadata.planner_data
        if original_planner_data is not None:
            for k in original_planner_data.keys():
                if k.startswith(key_prefix):
                    continue
                new_planner_data[k] = original_metadata.planner_data[k]
        original_storage_data = original_metadata.storage_data
        if original_storage_data is not None:
            for k in original_storage_data.keys():
                if k.fqn.startswith(key_prefix):
                    continue
                new_storage_data[k] = original_metadata.storage_data[k]
        metadata = Metadata(
            state_dict_metadata=new_state_dict_metadata,
            planner_data=new_planner_data,
            storage_data=new_storage_data,
        )
        fs_writer = FileSystemWriter(checkpoint_dir)
        metadata_filename = cast(Path, fs_writer.fs.concat_path(fs_writer.path, _metadata_fn))
        tmp_path = cast(
            metadata_filename,  # type: ignore[valid-type]
            fs_writer.fs.concat_path(fs_writer.path, f"{_metadata_fn}.tmp"),
        )
        old_path = cast(
            metadata_filename,  # type: ignore[valid-type]
            fs_writer.fs.concat_path(fs_writer.path, f"{_metadata_fn}.bck"),
        )
        ## save the new metadata
        with fs_writer.fs.create_stream(tmp_path, "wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            try:
                os.fsync(metadata_file.fileno())
            except AttributeError:
                os.sync()
        ## move the old metadata
        fs_writer.fs.rename(fs_writer.metadata_path, old_path)
        try:
            ## rename the new metadata
            fs_writer.fs.rename(tmp_path, fs_writer.metadata_path)

            ## finally, remove the files we want to drop
            for f in files_to_remove:
                fs_writer.fs.rm_file(checkpoint_dir / f)
        except Exception as e:
            fs_writer.fs.rename(old_path, fs_writer.metadata_path)
            raise e
        else:
            fs_writer.fs.rm_file(old_path)

    def can_handle_sharded_objects(self):
        return True

    def check_backend_compatibility(self, loaded_version):
        pass  # TODO

    def check_version_compatibility(self, loaded_version):
        pass  # TODO
