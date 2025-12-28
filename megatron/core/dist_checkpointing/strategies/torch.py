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
from .eclatin_manager import ECLATINManager
from .gemini_manager import GeminiManager
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
        
        # Initialize ECLATIN manager (singleton instance shared with Load strategy)
        self.eclatin_manager = ECLATINManager()
        self.eclatin_manager.init_eclatin_if_enabled()
        
        # Initialize Gemini manager (singleton instance for replica-level data transfer)
        from .gemini_manager import GeminiManager
        self.gemini_manager = GeminiManager()
        self.gemini_manager.init_gemini_if_enabled()
        
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
        
        # Initialize strategy-specific ECLATIN state
        self.eclatin_preallocate_cpu_buffer = True  # Preallocate CPU buffer for tensor data
        self.eclatin_use_continuous_buffer = True  # Use continuous buffer for tensor data
        # Note: decomposed_state_dict and preallocated_cpu_buffer are shared with ECCHECK
        self.eclatin_serialized_metadata = None
        self.eclatin_global_registry = None
        self.eclatin_blocks = None  # 4 persistent blocks (data_block_1/2, parity_block_1/2)
        self.ecl_write_buckets = []  # WriteBuckets for 4 blocks

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

    def _get_eclatin_buffers(self):
        """Get ECLATIN buffers for FileSystemWriterAsync.
        
        Note: Returns data and recv buffers (pooled).
        The 4 persistent blocks (data_block_1/2, parity_block_1/2) are allocated
        in _allocate_eclatin_blocks after metadata exchange.
        """
        return self.eclatin_manager.get_eclatin_buffers()

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
        # Store checkpoint_dir for EC-CHECK/ECLATIN preparation
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

        # Create FileSystemWriterAsync with EC-CHECK, ECLATIN, or Gemini parameters
        if self.eclatin_manager.use_eclatin:
            writer = FileSystemWriterAsync(
                checkpoint_dir,
                separation_hint=self.separation_hint,
                thread_count=self.thread_count,
                use_msc=MultiStorageClientFeature.is_enabled(),
                use_eclatin=self.eclatin_manager.use_eclatin,
                eclatin_native=self.eclatin_manager._eclatin_native,  # Pass pre-initialized C++ module
                eclatin_buffers=self._get_eclatin_buffers(),  # Pass pre-allocated buffers
            )
        elif self.gemini_manager.use_gemini and self.gemini_manager.use_gemini_optimized:
            writer = FileSystemWriterAsync(
                checkpoint_dir,
                separation_hint=self.separation_hint,
                thread_count=self.thread_count,
                use_msc=MultiStorageClientFeature.is_enabled(),
                use_gemini=self.gemini_manager.use_gemini,
                gemini_native=self.gemini_manager.get_native_module(),  # Pass pre-initialized C++ module
            )
        else:
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
        rank = torch.distributed.get_rank()
        # ECLATIN mode: decompose state_dict and preallocate CPU memory
        if self.eclatin_manager.use_eclatin:
            self._prepare_eclatin_data(self.cached_central_plan, planner)
            # Pass ECLATIN state to writer if available
            writer.decomposed_state_dict = self.decomposed_state_dict
            writer.preallocated_cpu_buffer = self.preallocated_cpu_buffer
            writer.eclatin_serialized_metadata = self.eclatin_serialized_metadata
            writer.eclatin_global_registry = self.eclatin_global_registry
            # Pass the 4 persistent blocks (data_block_1/2, parity_block_1/2)
            writer.eclatin_blocks = self.eclatin_blocks
            
            writer.ecl_write_buckets = self.ecl_write_buckets
            # In ECLATIN mode, call prepare_write_data to create write_buckets
            # It will use the metadata we just prepared
            writer.prepare_write_data(self.cached_central_plan, planner)
        # Gemini mode: decompose state_dict and preallocate CPU memory for replica exchange
        elif self.gemini_manager.use_gemini and self.gemini_manager.use_gemini_optimized:
            self._prepare_gemini_data(self.cached_central_plan, planner)
            # Pass Gemini state to writer if available
            writer.decomposed_state_dict = self.decomposed_state_dict
            writer.preallocated_cpu_buffer = self.preallocated_cpu_buffer
            
            # In Gemini mode, call prepare_write_data to create write_buckets
            # It will use the decomposed state_dict we just prepared
            writer.prepare_write_data(self.cached_central_plan, planner)
        # EC-CHECK mode: decompose state_dict and preallocate CPU memory
        elif self.eccheck_manager.use_eccheck:
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

    def _prepare_gemini_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        Gemini preparation: organize data for replica-level transfer.
        
        This method performs the following steps:
        1. Process plan items (separate bytes and tensors)
        2. Create DecomposedStateDict for efficient GPU-to-CPU transfer
        3. Preallocate CPU memory buffer for tensors
        
        Similar to EC-CHECK but simplified for Gemini's replica exchange use case.
        
        Args:
            plan (SavePlan): save plan from PyTorch distributed checkpoint
            planner (SavePlanner): save planner to resolve data
        """
        from torch.distributed.checkpoint.filesystem import _StoragePrefix
        
        start_total = time()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(f"Gemini: [Rank {rank}] Starting replica-level checkpoint preparation")
        
        # Step 1: Process plan items
        storage_plan: _StoragePrefix = plan.storage_data
        
        # Separate items into BYTE_IO (non-tensor) and TENSOR
        non_tensor_data = {}
        tensor_infos = []
        tensor_data_list = []
        
        logger.info(f"Gemini: [Rank {rank}] Processing {len(plan.items)} items from SavePlan")
        byte_io_count = 0
        tensor_count = 0
        none_data_count = 0
        
        for item in plan.items:
            data = planner.resolve_data(item)
            
            if data is None:
                none_data_count += 1
                continue
            
            if item.type == WriteItemType.BYTE_IO:
                # Non-tensor data
                # Convert BytesIO to bytes for proper serialization
                if hasattr(data, 'getvalue'):  # BytesIO object
                    data.seek(0)
                    data_bytes = data.getvalue()
                    non_tensor_data[item.index.fqn] = data_bytes
                else:
                    non_tensor_data[item.index.fqn] = data
                byte_io_count += 1
            else:
                # Tensor data - create TensorInfo
                from .state_dict_decomposer import TensorInfo
                
                tensor_info = TensorInfo(
                    key=item.index.fqn,
                    shape=tuple(data.shape),
                    dtype=data.dtype,
                    device=data.device,
                    numel=data.numel(),
                    size_bytes=data.numel() * data.element_size(),
                    offset=0,  # Will be calculated below
                    global_offset=tuple(item.index.offset) if hasattr(item.index, 'offset') else None,
                    shard_index=item.index.index if hasattr(item.index, 'index') else None,
                )
                tensor_infos.append(tensor_info)
                tensor_data_list.append(data)
                tensor_count += 1
        
        logger.info(
            f"Gemini: [Rank {rank}] Processed {byte_io_count} BytesIO items, {tensor_count} tensor items"
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
        
        # Log statistics
        stats = self.decomposed_state_dict.get_statistics()
        logger.info(
            f"Gemini: [Rank {rank}] Created DecomposedStateDict:\n"
            f"  Non-tensor data: {stats['non_tensor_size_bytes'] / 1024:.2f} KB\n"
            f"  Tensor data: {stats['tensor_data_size_bytes'] / (1024**3):.2f} GB\n"
            f"  Total tensors: {stats['num_tensors']}"
        )
        
        # Step 2: Preallocate CPU buffer
        total_tensor_size = self.decomposed_state_dict.total_tensor_size_bytes
        
        if self.preallocated_cpu_buffer is None or self.preallocated_cpu_buffer.numel() < total_tensor_size:
            logger.info(
                f"Gemini: [Rank {rank}] Allocating preallocated CPU buffer: "
                f"{total_tensor_size / (1024**3):.2f} GB"
            )
            
            # Use pinned memory for faster GPU-to-CPU transfer
            if torch.cuda.is_available():
                self.preallocated_cpu_buffer = torch.empty(
                    total_tensor_size, dtype=torch.uint8
                ).pin_memory()
                logger.info(f"Gemini: [Rank {rank}] Allocated pinned memory buffer")
            else:
                self.preallocated_cpu_buffer = torch.empty(
                    total_tensor_size, dtype=torch.uint8
                )
                logger.info(f"Gemini: [Rank {rank}] Allocated regular CPU buffer")
        else:
            logger.info(
                f"Gemini: [Rank {rank}] Reusing existing preallocated CPU buffer: "
                f"{self.preallocated_cpu_buffer.numel() / (1024**3):.2f} GB"
            )
        
        total_time = time() - start_total
        logger.info(
            f"Gemini: [Rank {rank}] Preparation completed in {total_time:.2f}s"
        )

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

    def _prepare_eclatin_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        ECLATIN preparation: organize data for serialization-free checkpointing.
        
        This method performs the following steps:
        1. Process plan items like normal mode (separate bytes and tensors)
        2. Organize tensors for ECLATIN (extract metadata and data)
        3. Preallocate CPU memory buffer for tensors
        4. Prepare write buckets for async transfer
        5. Broadcast and exchange metadata
        6. Allocate 4 persistent blocks (data_block_1/2, parity_block_1/2)
        
        Args:
            plan (SavePlan): save plan from PyTorch distributed checkpoint
            planner (SavePlanner): save planner to resolve data
        """
        from torch.distributed.checkpoint.filesystem import _StoragePrefix
        from time import time
        
        start_total = time()
        logger.info("ECLATIN: Starting serialization-free checkpoint preparation")
        
        # Step 1: Process plan items (similar to ECCHECK)
        start = time()
        storage_plan: _StoragePrefix = plan.storage_data
        
        # Separate items into BYTE_IO (non-tensor) and TENSOR
        non_tensor_data = {}
        tensor_infos = []
        tensor_data_list = []
        
        logger.info(f"ECLATIN: Processing {len(plan.items)} items from SavePlan")
        byte_io_count = 0
        tensor_count = 0
        none_data_count = 0
        
        for item in plan.items:
            data = planner.resolve_data(item)
            
            # Debug: check for None data
            if data is None:
                none_data_count += 1
                if none_data_count <= 5:
                    logger.warning(f"ECLATIN SAVE: Found None data for item: fqn={item.index.fqn}, type={item.type}")
                continue  # Skip None data items
            
            if item.type == WriteItemType.BYTE_IO:
                # Non-tensor data (e.g., extra_state)
                import io
                if isinstance(data, io.BytesIO):
                    non_tensor_data[item.index.fqn] = {
                        '_eccheck_type': 'BytesIO',
                        '_eccheck_data': data.getvalue()
                    }
                else:
                    non_tensor_data[item.index.fqn] = data
                byte_io_count += 1
            else:
                # Tensor data - create TensorInfo
                from .state_dict_decomposer import TensorInfo
                
                tensor_info = TensorInfo(
                    key=item.index.fqn,
                    shape=tuple(data.shape),
                    dtype=data.dtype,
                    device=data.device,
                    numel=data.numel(),
                    size_bytes=data.numel() * data.element_size(),
                    offset=0,  # Will be calculated below
                    global_offset=tuple(item.index.offset),
                    shard_index=item.index.index,
                )
                tensor_infos.append(tensor_info)
                tensor_data_list.append(data)
                tensor_count += 1
        
        logger.info(
            f"ECLATIN: Processed {byte_io_count} BytesIO items, {tensor_count} tensor items"
            + (f", skipped {none_data_count} None items" if none_data_count > 0 else "")
        )
        
        # Calculate offsets for tensor data
        offset = 0
        for info in tensor_infos:
            info.offset = offset
            offset += info.size_bytes
        
        # Create decomposed structure (reuse from ECCHECK if available, otherwise create new)
        if self.decomposed_state_dict is None:
            from .state_dict_decomposer import DecomposedStateDict
            self.decomposed_state_dict = DecomposedStateDict(
                non_tensor_data=non_tensor_data,
                tensor_infos=tensor_infos,
                tensor_data=tensor_data_list,
            )
        else:
            # Update existing decomposed_state_dict
            self.decomposed_state_dict.non_tensor_data = non_tensor_data
            self.decomposed_state_dict.tensor_infos = tensor_infos
            self.decomposed_state_dict.tensor_data = tensor_data_list
        
        process_time = time() - start
        
        # Log statistics
        stats = self.decomposed_state_dict.get_statistics()
        logger.info(
            f"ECLATIN: Processed plan items in {process_time:.2f}s\n"
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
        if self.eclatin_preallocate_cpu_buffer:
            start = time()
            total_size = self.decomposed_state_dict.total_tensor_size_bytes
            logger.info(f"ECLATIN: Preallocating CPU buffer of {total_size / (1024**3):.2f} GB")
            
            if self.preallocated_cpu_buffer is None:
                if self.eclatin_manager.eclatin_pin_memory and torch.cuda.is_available():
                    self.preallocated_cpu_buffer = torch.empty(
                        total_size, dtype=torch.uint8).pin_memory()
                    logger.info("ECLATIN: Using pinned memory for CPU buffer")
                else:
                    self.preallocated_cpu_buffer = torch.empty(
                        total_size, dtype=torch.uint8
                    )
                    logger.info("ECLATIN: Using non-pinned memory for CPU buffer")
            
            prealloc_time = time() - start
            logger.debug(f"ECLATIN: CPU buffer preallocation took {prealloc_time:.2f}s")
        else:
            prealloc_time = 0
        
        # Step 3: Prepare write buckets for async transfer
        # Note: WriteBuckets for 4 blocks will be created in _allocate_eclatin_blocks
        # This step is a placeholder for consistency with ECCHECK flow
        start = time()
        bucket_time = time() - start
        logger.debug(f"ECLATIN: Write bucket preparation (will be done in block allocation)")
        
        # Step 4: Validate decomposition
        if not self.validate_eclatin_decomposition():
            raise RuntimeError("ECLATIN: Decomposition validation failed")
        
        # Step 5: Broadcast and exchange metadata (reuse ECCHECK method)
        start = time()
        self.eclatin_global_registry = self._broadcast_and_exchange_metadata()
        metadata_time = time() - start
        logger.info(f"ECLATIN: Metadata exchange completed in {metadata_time:.2f}s")
        
        # Step 6: Allocate 4 persistent blocks (data_block_1/2, parity_block_1/2)
        start = time()
        if self.eclatin_blocks is None:
            self.eclatin_blocks = self._allocate_eclatin_blocks(self.eclatin_global_registry)
        block_alloc_time = time() - start
        logger.info(f"ECLATIN: Block allocation completed in {block_alloc_time:.2f}s")
        
        total_time = time() - start_total
        logger.info(
            f"ECLATIN: Preparation completed in {total_time:.2f}s\n"
            f"  Item processing: {process_time:.2f}s\n"
            f"  Preallocation: {prealloc_time:.2f}s\n"
            f"  Bucket prep: {bucket_time:.2f}s\n"
            f"  Metadata exchange: {metadata_time:.2f}s\n"
            f"  Block allocation: {block_alloc_time:.2f}s"
        )

    def validate_eclatin_decomposition(self) -> bool:
        """
        Validate ECLATIN decomposition structure.
        
        Validates:
        1. non_tensor_data is a dict
        2. tensor_infos is a list (tensor keys)
        3. tensor_data is a list of tensors
        4. Counts match between tensor_infos and tensor_data
        
        Returns:
            bool: True if decomposition is valid, False otherwise
        """
        if not self.eclatin_manager.use_eclatin:
            logger.warning("ECLATIN: Validation skipped - ECLATIN is not enabled")
            return False
        
        if not self.decomposed_state_dict:
            logger.error("ECLATIN: Validation failed - State dict not decomposed yet")
            return False
        
        decomposed = self.decomposed_state_dict
        
        # Check 1: Non-tensor key-value pairs (dict)
        if not isinstance(decomposed.non_tensor_data, dict):
            logger.error(
                f"ECLATIN: Component 1 failed - non_tensor_data should be dict, "
                f"got {type(decomposed.non_tensor_data).__name__}"
            )
            return False
        
        # Check 2: Tensor keys (list)
        if not isinstance(decomposed.tensor_infos, list):
            logger.error(
                f"ECLATIN: Component 2 failed - tensor_infos should be list, "
                f"got {type(decomposed.tensor_infos).__name__}"
            )
            return False
        
        # Check 3: Tensor data (list)
        if not isinstance(decomposed.tensor_data, list):
            logger.error(
                f"ECLATIN: Component 3 failed - tensor_data should be list, "
                f"got {type(decomposed.tensor_data).__name__}"
            )
            return False
        
        # Check 4: Counts match
        if len(decomposed.tensor_infos) != len(decomposed.tensor_data):
            logger.error(
                f"ECLATIN: Component count mismatch - tensor_infos has {len(decomposed.tensor_infos)} items, "
                f"tensor_data has {len(decomposed.tensor_data)} items"
            )
            return False
        
        # Check 5: Total size matches
        calculated_size = sum(info.size_bytes for info in decomposed.tensor_infos)
        if calculated_size != decomposed.total_tensor_size_bytes:
            logger.warning(
                f"ECLATIN: Size mismatch - calculated {calculated_size} bytes, "
                f"but total_tensor_size_bytes is {decomposed.total_tensor_size_bytes} bytes"
            )
            # This is a warning, not an error, as it might be due to rounding
        
        logger.debug("ECLATIN: Decomposition validation passed")
        return True

    def _allocate_eclatin_blocks(self, global_registry):
        """
        Allocate 4 persistent blocks for ECLATIN:
        - data_block_1: First data block
        - data_block_2: Second data block
        - parity_block_1: First parity block (from parity1 pipeline)
        - parity_block_2: Second parity block (from parity2 pipeline)
        
        All blocks are aligned to the maximum size across all ranks for pipeline synchronization.
        This ensures all ranks use the same block sizes.
        
        Args:
            global_registry: GlobalMetadataRegistry from all ranks
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'data_block_1', 'data_block_2', 
                                    'parity_block_1', 'parity_block_2'
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # ===== Get own data size from metadata =====
        own_metadata = global_registry.rank_metadata.get(rank, [])
        own_total_size = sum(meta.size_bytes for meta in own_metadata)
        
        # ===== Calculate maximum data size across all ranks =====
        if torch.distributed.is_initialized():
            # Get all ranks' data sizes from global_registry and compute max locally
            all_total_bytes_list = []
            for r in range(world_size):
                rank_metadata = global_registry.rank_metadata.get(r, [])
                rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
                all_total_bytes_list.append(rank_total_size)
            
            # Compute maximum locally (all ranks have the same global_registry)
            max_total_bytes = max(all_total_bytes_list)
        else:
            max_total_bytes = own_total_size
        
        # ===== Align block size to buffer_size (64MB) using half of maximum =====
        eclatin_buffer_size = self.eclatin_manager.eclatin_buffer_size
        # Each block only needs half of max_total_bytes (data is split into two halves)
        half_max_total_bytes = max_total_bytes // 2
        aligned_half_block_size = ((half_max_total_bytes + eclatin_buffer_size - 1) // eclatin_buffer_size) * eclatin_buffer_size
        
        logger.info(
            f"ECLATIN: Allocating 4 persistent blocks based on metadata\n"
            f"  Own data size: {own_total_size / (1024**3):.2f} GB (actual), "
            f"{max_total_bytes / (1024**3):.2f} GB (pipeline max), "
            f"{aligned_half_block_size / (1024**3):.2f} GB (aligned half block size)\n"
            f"  All blocks will use aligned half size: {aligned_half_block_size / (1024**3):.2f} GB "
            f"({aligned_half_block_size / (1024**2):.0f} MB)"
        )
        
        # ===== Allocate 4 large continuous buffers =====
        # All blocks use the same aligned half size (each block stores half of the data)
        data_block_1 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        data_block_2 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        parity_block_1 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        parity_block_2 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        
        logger.info(
            f"ECLATIN: Allocated 4 persistent blocks:\n"
            f"  data_block_1: {aligned_half_block_size / (1024**3):.2f} GB "
            f"({aligned_half_block_size / (1024**2):.0f} MB)\n"
            f"  data_block_2: {aligned_half_block_size / (1024**3):.2f} GB "
            f"({aligned_half_block_size / (1024**2):.0f} MB)\n"
            f"  parity_block_1: {aligned_half_block_size / (1024**3):.2f} GB "
            f"({aligned_half_block_size / (1024**2):.0f} MB)\n"
            f"  parity_block_2: {aligned_half_block_size / (1024**3):.2f} GB "
            f"({aligned_half_block_size / (1024**2):.0f} MB)\n"
            f"  Total memory: {4 * aligned_half_block_size / (1024**3):.2f} GB"
        )
        
        # ===== Package blocks with metadata =====
        # Align with EC-CHECK: use decomposed_state_dict.non_tensor_data directly
        own_non_tensor_data = self.decomposed_state_dict.non_tensor_data
        tensor_infos = self.decomposed_state_dict.tensor_infos  # List[TensorInfo] with offsets
        own_non_tensor_data_bytes = pickle.dumps(own_non_tensor_data)
        own_tensor_keys_data_bytes = pickle.dumps(tensor_infos)
        own_non_tensor_size = len(own_non_tensor_data_bytes)
        own_tensor_keys_size = len(own_tensor_keys_data_bytes)
        own_tensor_buffer_size = self.decomposed_state_dict.total_tensor_size_bytes
        
        # Create serialized metadata for all blocks (same metadata for all)
        block_serialized_metadata = {
            'non_tensor_data': own_non_tensor_data_bytes,
            'tensor_keys_data': own_tensor_keys_data_bytes,
            'non_tensor_size': own_non_tensor_size,
            'tensor_keys_size': own_tensor_keys_size,
            'tensor_buffer_size': own_tensor_buffer_size,
        }
        
        # Store actual size and pipeline size for later use
        own_actual_size = own_total_size
        block_pipeline_total_bytes = max_total_bytes
        
        # ===== Package blocks into WriteBucket format =====
        # Similar to ECCHECK's P2P buffers, create WriteBuckets for each block
        from pathlib import Path
        
        # Get checkpoint_dir
        checkpoint_dir = getattr(self, 'current_checkpoint_dir', None)
        if checkpoint_dir is None:
            logger.warning("ECLATIN: checkpoint_dir not available, using file_name as path")
            checkpoint_dir = Path(".")
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        # Create WriteBuckets for 4 blocks
        # Format: (file_path, storage_key, (bytes_data, tensor_data))
        block_names = ['data_block_1', 'data_block_2', 'parity_block_1', 'parity_block_2']
        block_tensors = [data_block_1, data_block_2, parity_block_1, parity_block_2]
        
        for block_name, block_tensor in zip(block_names, block_tensors):
            # Create eccheck_bytes_data format (reuse ECCHECK format for compatibility)
            block_eclatin_bytes_data = [
                ('eclatin_metadata', block_serialized_metadata),
                ('eclatin_continuous_buffer', block_tensor),
            ]
            
            # Generate file name
            file_name = f'__{rank}_{block_name}.distcp'
            file_path = checkpoint_dir / file_name
            
            # Create WriteBucket
            write_bucket = (
                file_path,              # file_path (full path with checkpoint_dir)
                file_name,              # storage_key (used in metadata)
                (block_eclatin_bytes_data, []),  # (bytes_data, tensor_data)
            )
            
            self.ecl_write_buckets.append(write_bucket)
        
        # Package blocks into dictionary
        blocks = {
            'data_block_1': data_block_1,
            'data_block_2': data_block_2,
            'parity_block_1': parity_block_1,
            'parity_block_2': parity_block_2,
            'metadata': block_serialized_metadata,
            'actual_size': own_actual_size,
            'pipeline_size': block_pipeline_total_bytes,
            'aligned_size': aligned_half_block_size,
        }
        
        logger.info(
            f"ECLATIN: Packaged 4 blocks with metadata and WriteBuckets:\n"
            f"  Metadata: {own_non_tensor_size / 1024:.2f} KB (non-tensor) + "
            f"{own_tensor_keys_size / 1024:.2f} KB (tensor keys), "
            f"{own_tensor_buffer_size / (1024**3):.2f} GB (buffer actual size)\n"
            f"  Pipeline size: {block_pipeline_total_bytes / (1024**3):.2f} GB\n"
            f"  Aligned half block size: {aligned_half_block_size / (1024**3):.2f} GB\n"
            f"  Created {len(block_names)} WriteBuckets"
        )
        
        return blocks
        
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
        
        # Initialize Gemini manager (singleton instance shared with Save strategy)
        self.gemini_manager = GeminiManager()
        self.gemini_manager.init_gemini_if_enabled()
        # Initialize ECLATIN manager (singleton instance shared with Save strategy)
        self.eclatin_manager = ECLATINManager()
        self.eclatin_manager.init_eclatin_if_enabled()
        
        # Initialize strategy-specific EC-CHECK state
        self.eccheck_p2p_buffers = None
        
        # Initialize rank2 recovery state
        self.eccheck_recovered_buffer = None
        self.eccheck_recovered_metadata = None
        self.eccheck_recovered_registry = None
        
        # Initialize strategy-specific ECLATIN state
        self.eclatin_blocks = None
        self.eclatin_recv_buffers = None
        self.eclatin_recovered_buffer = None
        self.eclatin_recovered_metadata = None
        self.eclatin_recovered_registry = None
    
        self.pairing_map = {0: 2, 2: 0, 1: 3, 3: 1}
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
    
    def _is_eclatin_checkpoint(self, checkpoint_dir: Path) -> bool:
        """Check if the checkpoint is in ECLATIN format.
        
        ECLATIN checkpoints are .distcp files with 'ECLT' magic number in the header.
        
        Args:
            checkpoint_dir (Path): checkpoint directory
            
        Returns:
            bool: True if this is an ECLATIN checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return False
        
        # Get current rank to find the corresponding file
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Check for ECLATIN format: __{rank}_0.distcp with ECLT magic number
        potential_file = checkpoint_dir / f"__{rank}_0.distcp"
        
        if not potential_file.exists():
            return False
        
        # Read first 4 bytes to check for ECLT magic number
        try:
            with open(potential_file, 'rb') as f:
                magic = f.read(4)
                return magic == b'ECLT'
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
        # Temporary hardcoding for rank2 failure recovery
        failed_rank = 2
        
        checkpoint_dir = Path(checkpoint_dir)
        eccheck_p2p_own_file = checkpoint_dir / f"__{rank}_p2p_own.distcp"
        eccheck_p2p_partner_file = checkpoint_dir / f"__{p2p_partner_rank}_p2p_partner.distcp"

        # Default placeholders
        mapped_file_own = EccheckMappedFile(None, None, None, None, None)
        mapped_file_partner = EccheckMappedFile(None, None, None, None, None)

        # Load own file if present
        if eccheck_p2p_own_file.exists():
            mapped_file_own = FileSystemWriterAsync.load_eccheck_bytes_from_file(
                str(eccheck_p2p_own_file), my_rank=rank
            )

        # Load partner file if present
        if eccheck_p2p_partner_file.exists():
            mapped_file_partner = FileSystemWriterAsync.load_eccheck_bytes_from_file(
                str(eccheck_p2p_partner_file), my_rank=p2p_partner_rank
            )
        
        # 如果 failed_rank 缺失 partner 文件，则 partner 点对点 send 元数据给 failed_rank，避免大对象广播
        # 仅当 failed_rank 缺失 partner 文件时触发点对点补元数据
        local_missing = (
            failed_rank >= 0
            and rank == failed_rank
            and not eccheck_p2p_partner_file.exists()
        )
        if torch.distributed.is_initialized():
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            missing_any = torch.tensor(int(local_missing), device=device)
            torch.distributed.all_reduce(missing_any, op=torch.distributed.ReduceOp.SUM)
            need_recover = bool(missing_any.item())
        else:
            need_recover = local_missing

        partner_rank = self._get_p2p_partner_rank(failed_rank, world_size)
        if need_recover and rank in (failed_rank, partner_rank):
            import pickle
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if rank == partner_rank:
                payload = pickle.dumps({
                    'tensor_metadata': mapped_file_own.local_metadata or [],
                    'non_tensor_data': mapped_file_own.non_tensor_data or {},
                })
                buf = torch.tensor(list(payload), dtype=torch.uint8, device=device)
                size = torch.tensor([buf.numel()], dtype=torch.int64, device=device)
                torch.distributed.send(size, dst=failed_rank)
                torch.distributed.send(buf, dst=failed_rank)
            elif rank == failed_rank:
                size = torch.empty(1, dtype=torch.int64, device=device)
                torch.distributed.recv(size, src=partner_rank)
                buf = torch.empty(int(size.item()), dtype=torch.uint8, device=device)
                torch.distributed.recv(buf, src=partner_rank)
                obj = pickle.loads(bytes(buf.cpu().tolist()))
                mapped_file_partner = EccheckMappedFile(
                    mmap_object=None,
                    memory_address=None,
                    file_size=None,
                    local_metadata=obj.get('tensor_metadata'),
                    non_tensor_data=obj.get('non_tensor_data'),
                )

        # Package both together
        local_package = {
            'tensor_metadata': mapped_file_partner.local_metadata or [],
            'non_tensor_data': mapped_file_partner.non_tensor_data or {},
        }
        
        if local_package['tensor_metadata'] is None or local_package['non_tensor_data'] is None:
            logger.error(f"EC-CHECK: [Rank {rank}] Local metadata is None, skipping metadata exchange")
            return mapped_file_own, mapped_file_partner
        
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
        
        # For rank2 recovery: save recovered data for later use in _load_eccheck_checkpoint
        if rank == failed_rank:
            logger.info(f"EC-CHECK: [Rank {rank}] Saving recovered buffer for _load_eccheck_checkpoint")
            # Store recovered data in instance variables for _load_eccheck_checkpoint to use
            self.eccheck_recovered_buffer = recv_own_buffer
            self.eccheck_recovered_metadata = mapped_file_own
            self.eccheck_recovered_registry = registry
        
        # Return EccheckMappedFile, non_tensor_data, and local_metadata for each file
        return mapped_file_own, mapped_file_partner
    
    def _load_eclatin_block_checkpoint(self, checkpoint_dir: Path) -> Tuple:
        """Load ECLATIN checkpoint data and recover rank2.
        
        Similar to EC-CHECK for rank2 recovery.
        
        Args:
            checkpoint_dir (Path): checkpoint directory
        
        Returns:
            Tuple: (mapped_file_own, None) - placeholder for compatibility
        """
        from .filesystem_async import FileSystemWriterAsync
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # ECLATIN recovers rank2 (same as EC-CHECK)
        failed_rank = 2
        
        checkpoint_dir = Path(checkpoint_dir)
        
        # ===== Step 1: Load main file to extract metadata =====
        eclatin_main_file = checkpoint_dir / f'__{rank}_0.distcp'
        if not eclatin_main_file.exists():
            logger.warning(f"ECLATIN: Main file not found for rank {rank}: {eclatin_main_file}")
            return None, None
        
        # Load main file, extract Component 1 and Component 2 (metadata)
        mapped_file_own = FileSystemWriterAsync.load_eclatin_bytes_from_file(
            str(eclatin_main_file), my_rank=rank
        )
        
        # ===== Step 2: Metadata exchange (similar to EC-CHECK) =====
        local_package = {
            'tensor_metadata': mapped_file_own.local_metadata or [],
            'non_tensor_data': mapped_file_own.non_tensor_data or {},
        }
        
        if local_package['tensor_metadata'] is None or local_package['non_tensor_data'] is None:
            logger.error(f"ECLATIN: [Rank {rank}] Local metadata is None, skipping metadata exchange")
            return mapped_file_own, None
        
        # All-gather complete metadata using all_gather_object
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
        
        # ===== Step 3: Allocate 4 blocks uniformly (reuse save phase logic) =====
        if self.eclatin_blocks is None:
            # Use the same _allocate_eclatin_blocks method as save phase
            # This allocates 4 blocks based on registry metadata
            self.eclatin_blocks = self._allocate_eclatin_blocks(registry)
            logger.info(f"ECLATIN: [Rank {rank}] Allocated 4 blocks using registry metadata")
        
        # ===== Step 4: rank0/1/3 load block data from files =====
        if rank != 2:
            # rank0/1/3: Load block data from files into allocated blocks
            self._load_eclatin_blocks_from_files(checkpoint_dir, rank)
        
        # ===== Step 5: rank2 allocate recv buffers =====
        if rank == 2:
            if self.eclatin_recv_buffers is None:
                self.eclatin_recv_buffers = self._allocate_eclatin_load_recv_buffers(registry)
            
            if self.eclatin_recovered_buffer is None:
                own_metadata = registry.rank_metadata.get(rank, [])
                total_size = sum(meta.size_bytes for meta in own_metadata)
                self.eclatin_recovered_buffer = torch.empty(total_size, dtype=torch.uint8)
        
        # ===== Step 6: Run recovery pipeline =====
        own_metadata = registry.rank_metadata.get(rank, [])
        total_size = sum(meta.size_bytes for meta in own_metadata)
        
        self._run_eclatin_recovery_pipeline(
            rank=rank,
            world_size=world_size,
            registry=registry,
            eclatin_blocks=self.eclatin_blocks,
            recv_buffers=self.eclatin_recv_buffers if rank == 2 else None,
            recovered_buffer=self.eclatin_recovered_buffer if rank == 2 else None,
            total_size=total_size,
        )
        
        # ===== Step 7: rank2 save recovered buffer =====
        if rank == failed_rank:
            logger.info(f"ECLATIN: [Rank {rank}] Saving recovered buffer for _load_eclatin_checkpoint")
            self.eclatin_recovered_metadata = mapped_file_own
            self.eclatin_recovered_registry = registry
        
        return mapped_file_own, None
    
    def _load_eclatin_blocks_from_files(self, checkpoint_dir: Path, rank: int) -> None:
        """Load block data from files into already allocated blocks.
        
        Note: Blocks are already allocated via _allocate_eclatin_blocks.
        This method only loads data from files into the allocated blocks.
        
        Args:
            checkpoint_dir (Path): checkpoint directory
            rank (int): current rank
        """
        # rank0: Load data_block_2, parity_block_2 (send to rank2)
        if rank == 0:
            self._load_block_data_from_file(
                checkpoint_dir, rank, 'data_block_2', self.eclatin_blocks['data_block_2']
            )
            self._load_block_data_from_file(
                checkpoint_dir, rank, 'parity_block_2', self.eclatin_blocks['parity_block_2']
            )
        
        # rank1: Load data_block_1, parity_block_1 (send to rank2)
        elif rank == 1:
            self._load_block_data_from_file(
                checkpoint_dir, rank, 'data_block_1', self.eclatin_blocks['data_block_1']
            )
            self._load_block_data_from_file(
                checkpoint_dir, rank, 'parity_block_1', self.eclatin_blocks['parity_block_1']
            )
        
        # rank3: Load data_block_1, data_block_2 (send to rank2)
        elif rank == 3:
            self._load_block_data_from_file(
                checkpoint_dir, rank, 'data_block_1', self.eclatin_blocks['data_block_1']
            )
            self._load_block_data_from_file(
                checkpoint_dir, rank, 'data_block_2', self.eclatin_blocks['data_block_2']
            )
        
        # rank2: No need to load blocks (will receive from others)
    
    def _load_block_data_from_file(
        self, checkpoint_dir: Path, rank: int, block_name: str, block_tensor: torch.Tensor
    ) -> None:
        """Load block data from file into allocated block tensor using mmap.
        
        File format: __{rank}_{block_name}.distcp
        Only loads Component 3 (block data) into block_tensor.
        Uses mmap for zero-copy access, similar to EC-CHECK.
        
        Args:
            checkpoint_dir (Path): checkpoint directory
            rank (int): current rank
            block_name (str): block name (e.g., 'data_block_1')
            block_tensor (torch.Tensor): pre-allocated tensor to load data into
        """
        import struct
        import numpy as np
        import mmap
        
        file_path = checkpoint_dir / f'__{rank}_{block_name}.distcp'
        
        if not file_path.exists():
            raise FileNotFoundError(f"ECLATIN: Block file not found: {file_path}")
        
        logger.info(f"ECLATIN: [Rank {rank}] Loading {block_name} from {file_path} using mmap")
        
        # Open file and memory-map it
        f = open(file_path, "rb")
        mm = None
        try:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Seek back to start
            
            # Memory-map the entire file (zero-copy access)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Close file handle - mmap is independent of the file handle
            f.close()
            f = None
            
            # Parse header from mmap
            header_bytes = mm[:32]
            if len(header_bytes) != 32:
                raise RuntimeError(f"ECLATIN: Invalid file header (expected 32 bytes, got {len(header_bytes)})")
            
            magic, non_tensor_size, tensor_keys_size, tensor_buffer_size = struct.unpack('4sQQQ', header_bytes)
            
            if magic != b'ECLT':
                raise RuntimeError(f"ECLATIN: Invalid magic number (expected b'ECLT', got {magic})")
            
            # Calculate Component 3 offset (skip Component 1 and Component 2)
            offset = 32 + non_tensor_size + tensor_keys_size
            
            # Read Component 3 (block data) directly from mmap into block_tensor
            # Note: block_tensor size should match tensor_buffer_size (aligned_half_block_size)
            expected_size = block_tensor.numel()
            if tensor_buffer_size > expected_size:
                logger.warning(
                    f"ECLATIN: Block data size ({tensor_buffer_size}) > allocated size ({expected_size}), "
                    f"truncating to {expected_size}"
                )
                read_size = expected_size
            else:
                read_size = tensor_buffer_size
            
            # Read data directly from mmap (zero-copy numpy view)
            source_data = mm[offset:offset + read_size]
            if len(source_data) != read_size:
                raise RuntimeError(
                    f"ECLATIN: Failed to read block data from mmap "
                    f"(expected {read_size} bytes, got {len(source_data)})"
                )
            
            # Copy to block_tensor using numpy (zero-copy from mmap)
            block_tensor_np = block_tensor.numpy()
            block_tensor_np[:read_size] = np.frombuffer(source_data, dtype=np.uint8)
            
            # Fill remaining with zeros if needed
            if read_size < expected_size:
                block_tensor_np[read_size:] = 0
            
            logger.debug(
                f"ECLATIN: [Rank {rank}] Loaded {block_name} ({read_size / (1024**2):.2f} MB) "
                f"from mmap (zero-copy)"
            )
            
        finally:
            if mm is not None:
                try:
                    mm.close()
                except:
                    pass
            if f is not None:
                f.close()
    
    def _allocate_eclatin_blocks(self, global_registry):
        """
        Allocate 4 persistent blocks for ECLATIN load.
        
        Similar to save phase but simplified - only allocates blocks without WriteBuckets.
        
        Args:
            global_registry: GlobalMetadataRegistry from all ranks
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'data_block_1', 'data_block_2', 
                                    'parity_block_1', 'parity_block_2'
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # ===== Get own data size from metadata =====
        own_metadata = global_registry.rank_metadata.get(rank, [])
        own_total_size = sum(meta.size_bytes for meta in own_metadata)
        
        # ===== Calculate maximum data size across all ranks =====
        if torch.distributed.is_initialized():
            all_total_bytes_list = []
            for r in range(world_size):
                rank_metadata = global_registry.rank_metadata.get(r, [])
                rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
                all_total_bytes_list.append(rank_total_size)
            max_total_bytes = max(all_total_bytes_list)
        else:
            max_total_bytes = own_total_size
        
        # ===== Align block size to buffer_size (64MB) using half of maximum =====
        eclatin_buffer_size = self.eclatin_manager.eclatin_buffer_size
        half_max_total_bytes = max_total_bytes // 2
        aligned_half_block_size = ((half_max_total_bytes + eclatin_buffer_size - 1) // eclatin_buffer_size) * eclatin_buffer_size
        
        logger.info(
            f"ECLATIN: [Load] Allocating 4 persistent blocks based on metadata\n"
            f"  Own data size: {own_total_size / (1024**3):.2f} GB (actual), "
            f"{max_total_bytes / (1024**3):.2f} GB (pipeline max), "
            f"{aligned_half_block_size / (1024**3):.2f} GB (aligned half block size)"
        )
        
        # ===== Allocate 4 large continuous buffers =====
        data_block_1 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        data_block_2 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        parity_block_1 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        parity_block_2 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        
        logger.info(
            f"ECLATIN: [Load] Allocated 4 persistent blocks:\n"
            f"  data_block_1: {aligned_half_block_size / (1024**3):.2f} GB\n"
            f"  data_block_2: {aligned_half_block_size / (1024**3):.2f} GB\n"
            f"  parity_block_1: {aligned_half_block_size / (1024**3):.2f} GB\n"
            f"  parity_block_2: {aligned_half_block_size / (1024**3):.2f} GB\n"
            f"  Total memory: {4 * aligned_half_block_size / (1024**3):.2f} GB"
        )
        
        # Return only blocks dictionary (no WriteBuckets needed for load)
        blocks = {
            'data_block_1': data_block_1,
            'data_block_2': data_block_2,
            'parity_block_1': parity_block_1,
            'parity_block_2': parity_block_2,
        }
        
        return blocks
    
    def _allocate_eclatin_load_recv_buffers(self, global_registry) -> Dict[str, torch.Tensor]:
        """Allocate recv buffers for rank0 load recovery.
        
        Delegates to manager's allocate_eclatin_load_recv_buffers method.
        
        Args:
            global_registry: GlobalMetadataRegistry
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 6 recv buffers (rank0 only)
        """
        return self.eclatin_manager.allocate_eclatin_load_recv_buffers(global_registry)
    
    def _extract_decomposed_from_buffer(
        self,
        recv_own_buffer: torch.Tensor,
        mapped_file_own,
        registry
    ):
        """Extract DecomposedStateDict from recovered buffer and metadata for rank2.
        
        This function extracts tensors from recv_own_buffer using metadata from
        mapped_file_own.local_metadata and builds a DecomposedStateDict structure.
        
        Args:
            recv_own_buffer: Buffer containing recovered tensor data
            mapped_file_own: EccheckMappedFile with local_metadata and non_tensor_data
            registry: GlobalMetadataRegistry with complete metadata
            
        Returns:
            DecomposedStateDict: Decomposed state dict with tensor_infos, tensor_data, and non_tensor_data
        """
        import io
        from .state_dict_decomposer import TensorInfo, DecomposedStateDict
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        try:
            # Get metadata for this rank (rank2's own metadata)
            local_metadata = mapped_file_own.local_metadata or []
            non_tensor_data = mapped_file_own.non_tensor_data or {}
            
            if not local_metadata:
                logger.warning(f"EC-CHECK: [Rank {rank}] No local_metadata available for reconstruction")
                return None
            
            logger.info(f"EC-CHECK: [Rank {rank}] Reconstructing from {len(local_metadata)} tensor metadata entries")
            
            # CRITICAL FIX: Use original tensor_infos with offset if available
            # The recovered buffer contains data in the order saved (with proper offsets),
            # not in sequential order. Using sequential current_offset causes mismatches.
            tensor_infos = []
            tensor_data = []
            
            # Check if mapped_file_own has original tensor_infos with offset information
            if hasattr(mapped_file_own, 'tensor_infos') and mapped_file_own.tensor_infos:
                logger.info(f"EC-CHECK: [Rank {rank}] Using tensor_infos with offset from mapped_file")
                # Use original tensor_infos with correct offset
                for info in mapped_file_own.tensor_infos:
                    start = info.offset  # Use offset from TensorInfo (from save phase)
                    end = start + info.size_bytes
                    
                    if end > recv_own_buffer.numel():
                        logger.error(f"EC-CHECK: [Rank {rank}] Buffer overflow: end={end}, buffer_size={recv_own_buffer.numel()}")
                        return None
                    
                    tensor_bytes = recv_own_buffer[start:end]
                    
                    # Reshape to original tensor
                    try:
                        tensor = tensor_bytes.view(info.dtype).reshape(info.shape).clone()
                        tensor_data.append(tensor)
                        tensor_infos.append(info)
                    except Exception as e:
                        logger.error(f"EC-CHECK: [Rank {rank}] Failed to reshape tensor {info.key}: {e}")
                        return None
            else:
                # Fallback: sequential extraction (may cause mismatches if data order differs)
                logger.warning(f"EC-CHECK: [Rank {rank}] No tensor_infos with offset found, using sequential extraction (may cause mismatches)")
                current_offset = 0
                
                for meta in local_metadata:
                    # Convert dtype string to torch.dtype
                    # meta.dtype is a string like 'torch.float32' or 'float32'
                    dtype_str = meta.dtype.replace('torch.', '') if 'torch.' in meta.dtype else meta.dtype
                    try:
                        dtype = getattr(torch, dtype_str)
                    except AttributeError:
                        logger.warning(f"EC-CHECK: [Rank {rank}] Unknown dtype {meta.dtype}, using float32")
                        dtype = torch.float32
                    
                    # Calculate element size for numel calculation
                    # Use torch._utils._element_size instead of creating a tensor
                    # to avoid GPU memory allocation
                    element_size = torch._utils._element_size(dtype)
                    numel = meta.size_bytes // element_size
                    
                    # Create TensorInfo
                    tensor_info = TensorInfo(
                        key=meta.key,
                        shape=meta.shape,
                        dtype=dtype,
                        device=torch.device('cpu'),
                        numel=numel,
                        size_bytes=meta.size_bytes,
                        offset=current_offset,
                        global_offset=meta.global_offset,
                        shard_index=meta.shard_index
                    )
                    tensor_infos.append(tensor_info)
                    
                    # Extract tensor from buffer
                    start = current_offset
                    end = start + meta.size_bytes
                    if end > recv_own_buffer.numel():
                        logger.error(f"EC-CHECK: [Rank {rank}] Buffer overflow: end={end}, buffer_size={recv_own_buffer.numel()}")
                        return None
                    
                    tensor_bytes = recv_own_buffer[start:end]
                    
                    # Reshape to original tensor
                    try:
                        # Convert uint8 buffer to target dtype and reshape
                        # First view as target dtype, then reshape to original shape
                        tensor = tensor_bytes.view(dtype).reshape(meta.shape).clone()
                        tensor_data.append(tensor)
                    except Exception as e:
                        logger.error(f"EC-CHECK: [Rank {rank}] Failed to reshape tensor {meta.key}: {e}")
                        return None
                    
                    current_offset = end
            
            logger.info(f"EC-CHECK: [Rank {rank}] Extracted {len(tensor_data)} tensors from buffer")
            
            # Create and return DecomposedStateDict
            decomposed = DecomposedStateDict(
                non_tensor_data=non_tensor_data,
                tensor_infos=tensor_infos,
                tensor_data=tensor_data
            )
            
            logger.info(f"EC-CHECK: [Rank {rank}] Successfully created DecomposedStateDict with {len(tensor_data)} tensors")
            return decomposed
            
        except Exception as e:
            logger.error(f"EC-CHECK: [Rank {rank}] Failed to extract DecomposedStateDict from buffer: {e}", exc_info=True)
            raise
    
    def _load_gemini_checkpoint_recovery_asio(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Load checkpoint for rank2 failure recovery using ASIO for data transfer (OPTIMIZED).
        
        This method uses Gemini's ASIO-based communication for efficient data transfer
        between rank0 and rank2 during recovery. Optimized to minimize data copies.
        
        Optimizations:
        1. Zero-copy mmap: Direct send from mmap without intermediate copy
        2. Direct tensor receive: Receive into torch tensor, avoid numpy->bytes conversion
        3. Memoryview parsing: Use memoryview to avoid bytes slicing copies
        4. Shared memory: Use from_numpy without copy() when safe
        
        Process:
        1. Rank0 reads replica file via mmap and sends directly (zero-copy)
        2. Rank0 sends data to rank2 using ASIO (non-blocking, high-performance)
        3. Rank2 receives into torch tensor directly (zero-copy)
        4. Both ranks restore their state_dict with minimal copies
        
        Args:
            sharded_state_dict: Sharded state dict template for loading
            checkpoint_dir: Checkpoint directory
            
        Returns:
            StateDict: Loaded state dict
        """
        import mmap
        import numpy as np
        import ctypes
        
        rank = torch.distributed.get_rank()
        paired_rank = self.pairing_map.get(rank, None)
        checkpoint_dir = Path(checkpoint_dir)
        
        logger.info(f"rank: {rank}, starting Gemini checkpoint recovery with ASIO (OPTIMIZED) for rank2 failure")
        
        # Ensure Gemini native module is initialized
        if self.gemini_manager._gemini_native is None:
            logger.warning(f"rank: {rank}, Gemini native module not initialized, falling back to standard recovery")
            return self._load_gemini_checkpoint_recovery(sharded_state_dict, checkpoint_dir)
        
        # Only rank0 (pair_rank=2) and rank2 participate
        if rank == 0 and paired_rank == 2:
            # Rank0: Read replica file and send to rank2 via ASIO
            logger.info(f"rank: {rank}, reading replica file for rank2 recovery (ASIO OPTIMIZED mode)")
            
            # Find replica file: __0_0_replica2_rank0.distcp
            replica_files = list(checkpoint_dir.glob(f"*_replica{paired_rank}_rank{rank}*.distcp"))
            
            if not replica_files:
                logger.error(f"rank: {rank}, no replica file found for rank2 recovery")
                raise FileNotFoundError(f"No replica file found for rank2 recovery")
            
            replica_file_path = replica_files[0]
            logger.info(f"rank: {rank}, found replica file: {replica_file_path}")
            
            # Step 1: Open file with mmap (keep mmap alive for zero-copy send)
            try:
                f = open(replica_file_path, 'rb')
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                replica_file_size = len(mm)
                
                logger.info(f"rank: {rank}, opened replica file with mmap: {replica_file_size / (1024**2):.2f} MB")
            
            except Exception as e:
                logger.error(f"rank: {rank}, failed to open replica file: {e}", exc_info=True)
                raise
            
            # Step 2: Send size information first
            size_array = np.array([replica_file_size], dtype=np.int64)
            logger.info(f"rank: {rank}, sending size to rank2 via ASIO: {replica_file_size / (1024**2):.2f} MB")
            
            # Create numpy array view of mmap (zero-copy, read-only)
            # Do this before sending to ensure the view is created
            mmap_np = np.frombuffer(mm, dtype=np.uint8)
            
            try:
                # Send size (8 bytes) using raw memory address
                size_addr = size_array.ctypes.data
                self.gemini_manager._gemini_native.send_buffer(size_addr, size_array.nbytes)
                logger.info(f"rank: {rank}, size sent successfully")
                
                # Step 3: Send replica data directly from mmap (ZERO-COPY)
                logger.info(f"rank: {rank}, sending replica data to rank2 via ASIO (zero-copy from mmap)...")
                
                # Get mmap buffer address from numpy array
                mmap_addr = mmap_np.ctypes.data
                self.gemini_manager._gemini_native.send_buffer(mmap_addr, replica_file_size)
                
                logger.info(f"rank: {rank}, replica data sent successfully via ASIO (zero-copy)")
                
            except Exception as e:
                logger.error(f"rank: {rank}, ASIO send failed: {e}", exc_info=True)
                raise
            finally:
                # Delete numpy array reference first to release mmap
                del mmap_np
                # Clean up mmap and file after send completes
                mm.close()
                f.close()
            
            # Step 4: Rank0 also needs to load its own checkpoint from file
            logger.info(f"rank: {rank}, loading own checkpoint from saved file")
            return self._load_from_saved_checkpoint_file(sharded_state_dict, checkpoint_dir)
            
        elif rank == 2:
            # Rank2: Receive data from rank0 via ASIO and restore state_dict
            logger.info(f"rank: {rank}, receiving replica data from rank0 via ASIO (OPTIMIZED) for recovery")
            
            try:
                # Step 1: Receive size information first
                size_buffer = np.zeros(1, dtype=np.int64)
                size_addr = size_buffer.ctypes.data
                self.gemini_manager._gemini_native.receive_buffer(size_addr, size_buffer.nbytes)
                remote_size = int(size_buffer[0])
                
                logger.info(f"rank: {rank}, received size from rank0 via ASIO: {remote_size / (1024**2):.2f} MB")
                
                # Step 2: Create receive buffer as torch tensor (pinned for faster GPU transfer if needed)
                if torch.cuda.is_available():
                    remote_tensor = torch.empty(remote_size, dtype=torch.uint8).pin_memory()
                    logger.info(f"rank: {rank}, allocated pinned memory tensor for receive")
                else:
                    remote_tensor = torch.empty(remote_size, dtype=torch.uint8)
                
                # Step 3: Receive replica data via ASIO directly into tensor (ZERO-COPY)
                logger.info(f"rank: {rank}, receiving replica data from rank0 via ASIO (zero-copy into tensor)...")
                remote_addr = remote_tensor.data_ptr()
                self.gemini_manager._gemini_native.receive_buffer(remote_addr, remote_size)
                logger.info(f"rank: {rank}, received replica data via ASIO: {remote_size / (1024**2):.2f} MB")
                
                # Step 4: Parse received data (OPTIMIZED - avoid unnecessary copies)
                # Check if this is Gemini optimized format
                use_gemini_optimized = False
                try:
                    from megatron.training import get_args
                    args = get_args()
                    use_gemini_optimized = getattr(args, 'use_gemini', False) and getattr(args, 'use_gemini_optimized', False)
                except:
                    pass
                
                if use_gemini_optimized and remote_size >= 8:
                    # Parse as Gemini optimized format: [metadata_size(8)] + [metadata_bytes] + [buffer_bytes]
                    # Use tensor slicing to avoid copies
                    metadata_size_tensor = remote_tensor[:8]
                    metadata_size = int.from_bytes(metadata_size_tensor.cpu().numpy().tobytes(), byteorder='little')
                    
                    logger.info(f"rank: {rank}, parsing received data as Gemini optimized format, metadata_size: {metadata_size / 1024:.2f} KB")
                    
                    # Extract metadata (only copy small metadata portion)
                    metadata_tensor = remote_tensor[8:8+metadata_size]
                    metadata_bytes = metadata_tensor.cpu().numpy().tobytes()
                    metadata_buffer = io.BytesIO(metadata_bytes)
                    gemini_metadata = torch.load(metadata_buffer, map_location='cpu', weights_only=False)
                    
                    # Extract buffer (ZERO-COPY - use tensor slice directly)
                    buffer_tensor = remote_tensor[8+metadata_size:]
                    
                    logger.info(
                        f"rank: {rank}, parsed Gemini data (OPTIMIZED): "
                        f"metadata_size={metadata_size / 1024:.2f} KB, "
                        f"buffer_size={buffer_tensor.numel() / (1024**2):.2f} MB"
                    )
                    
                    # Create write_buckets structure for Gemini format
                    replica_buckets = [(
                        checkpoint_dir / f"__{rank}_0.distcp",
                        'gemini_optimized_local',
                        (
                            [('gemini_metadata', gemini_metadata), ('gemini_buffer', buffer_tensor)],
                            []
                        )
                    )]
                    
                    # Step 5: Restore state_dict from Gemini format
                    logger.info(f"rank: {rank}, restoring state_dict from Gemini format (zero-copy)...")
                    loaded_state_dict = self._restore_state_dict_from_gemini_format(
                        replica_buckets, sharded_state_dict
                    )
                else:
                    # Standard pickle format (need to convert to bytes)
                    logger.info(f"rank: {rank}, parsing received data as standard pickle format")
                    remote_bytes = remote_tensor.cpu().numpy().tobytes()
                    remote_data_io = io.BytesIO(remote_bytes)
                    replica_buckets = torch.load(remote_data_io, weights_only=False)
                    
                    logger.info(f"rank: {rank}, deserialized replica data, restoring state_dict from memory...")
                    
                    # Step 5: Restore state_dict from replica_buckets
                    loaded_state_dict = self._restore_state_dict_from_write_buckets(
                        replica_buckets, sharded_state_dict
                    )
                
                logger.info(f"rank: {rank}, successfully restored state_dict from rank0's replica data (ASIO OPTIMIZED)")
                return loaded_state_dict
                
            except Exception as e:
                logger.error(f"rank: {rank}, ASIO receive failed: {e}", exc_info=True)
                raise
        else:
            # Should not reach here
            logger.error(f"rank: {rank}, unexpected rank in ASIO recovery")
            raise RuntimeError(f"Unexpected rank {rank} in ASIO recovery")
    
    def _load_gemini_checkpoint_recovery(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Load checkpoint for rank2 failure recovery scenario.
        
        In this scenario, rank2 has no data and needs to recover from rank0's replica.
        Only rank0 (pair_rank=2) and rank2 need to participate.
        
        Process:
        1. Rank0 (pair_rank=2) reads replica file and sends size to rank2
        2. Rank2 creates receive buffer based on the size
        3. Rank0 sends replica data to rank2
        4. Rank2 deserializes and restores state_dict from received data
        
        Args:
            sharded_state_dict: Sharded state dict template for loading
            checkpoint_dir: Checkpoint directory
            
        Returns:
            StateDict: Loaded state dict (only rank2 returns valid data)
        """
        import mmap
        import numpy as np
        from .async_utils import get_or_create_pair_process_group
        
        rank = torch.distributed.get_rank()
        paired_rank = self.pairing_map.get(rank, None)
        checkpoint_dir = Path(checkpoint_dir)
        
        logger.info(f"rank: {rank}, starting Gemini checkpoint recovery for rank2 failure")
        
        # Only rank0 (pair_rank=2) and rank2 participate
        if rank == 0 and paired_rank == 2:
            # Rank0: Read replica file and send to rank2
            logger.info(f"rank: {rank}, reading replica file for rank2 recovery")
            
            # Find replica file: __0_0_replica2_rank0.distcp
            replica_files = list(checkpoint_dir.glob(f"*_replica{paired_rank}_rank{rank}*.distcp"))
            
            if not replica_files:
                logger.error(f"rank: {rank}, no replica file found for rank2 recovery")
                raise FileNotFoundError(f"No replica file found for rank2 recovery")
            
            replica_file_path = replica_files[0]
            logger.info(f"rank: {rank}, found replica file: {replica_file_path}")
            
            # Step 1: Read replica file using mmap (zero-copy)
            try:
                with open(replica_file_path, 'rb') as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    replica_file_size = len(mm)
                    replica_data_bytes = mm[:]
                    mm.close()
                
                logger.info(f"rank: {rank}, read replica file: {replica_file_size / (1024**2):.2f} MB")
            
            except Exception as e:
                logger.error(f"rank: {rank}, failed to read replica file: {e}", exc_info=True)
                raise
            
            # Step 2: Create pair process group for communication with rank2
            pair_group = get_or_create_pair_process_group(rank, paired_rank)
            
            # Step 3: Exchange file size with rank2
            local_size_tensor = torch.tensor([replica_file_size], dtype=torch.int64)
            remote_size_tensor = torch.tensor([0], dtype=torch.int64)
            
            # Use all_gather to exchange sizes
            size_list = [local_size_tensor, remote_size_tensor]
            torch.distributed.all_gather(size_list, local_size_tensor, group=pair_group)
            
            logger.info(f"rank: {rank}, exchanged size with rank2: {replica_file_size / (1024**2):.2f} MB")
            
            # Step 4: Convert replica data to tensor for broadcast
            replica_np = np.frombuffer(replica_data_bytes, dtype=np.uint8).copy()
            replica_tensor = torch.from_numpy(replica_np)
            
            logger.info(f"rank: {rank}, prepared replica tensor: {replica_tensor.numel() / (1024**2):.2f} MB")
            
            # Step 5: Send replica data to rank2 using broadcast
            # Rank0 is the source (src=0 in global ranks)
            torch.distributed.broadcast(replica_tensor, src=rank, group=pair_group)
            
            logger.info(f"rank: {rank}, sent replica data to rank2")
            
            # Step 6: Rank0 also needs to load its own checkpoint from file
            logger.info(f"rank: {rank}, loading own checkpoint from saved file")
            return self._load_from_saved_checkpoint_file(sharded_state_dict, checkpoint_dir)
        elif rank == 2:
            # Rank2: Receive data from rank0 and restore state_dict
            logger.info(f"rank: {rank}, receiving replica data from rank0 for recovery")
            
            # Step 1: Create pair process group for communication with rank0
            pair_group = get_or_create_pair_process_group(rank, 0)
            
            # Step 2: Receive file size from rank0
            local_size_tensor = torch.tensor([0], dtype=torch.int64)
            remote_size_tensor = torch.tensor([0], dtype=torch.int64)
            
            # Use all_gather to exchange sizes
            size_list = [remote_size_tensor, local_size_tensor]
            torch.distributed.all_gather(size_list, local_size_tensor, group=pair_group)
            
            remote_size = size_list[0].item()
            logger.info(f"rank: {rank}, received size from rank0: {remote_size / (1024**2):.2f} MB")
            
            # Step 3: Create receive buffer
            remote_tensor = torch.empty(remote_size, dtype=torch.uint8)
            
            # Step 4: Receive replica data from rank0 using broadcast
            torch.distributed.broadcast(remote_tensor, src=0, group=pair_group)
            
            logger.info(f"rank: {rank}, received replica data from rank0: {remote_size / (1024**2):.2f} MB")
            
            # Step 5: Deserialize received data
            # Check if this is Gemini optimized format
            remote_bytes = remote_tensor.numpy().tobytes()
            
            # Try to detect Gemini optimized format
            use_gemini_optimized = False
            try:
                from megatron.training import get_args
                args = get_args()
                use_gemini_optimized = getattr(args, 'use_gemini', False) and getattr(args, 'use_gemini_optimized', False)
            except:
                pass
            
            if use_gemini_optimized and len(remote_bytes) >= 8:
                # Parse as Gemini optimized format: [metadata_size(8)] + [metadata_bytes] + [buffer_bytes]
                metadata_size = int.from_bytes(remote_bytes[:8], byteorder='little')
                
                logger.info(f"rank: {rank}, parsing received data as Gemini optimized format, metadata_size: {metadata_size / 1024:.2f} KB")
                
                # Extract metadata
                metadata_bytes = remote_bytes[8:8+metadata_size]
                metadata_buffer = io.BytesIO(metadata_bytes)
                gemini_metadata = torch.load(metadata_buffer, map_location='cpu', weights_only=False)
                
                # Extract buffer
                buffer_bytes = remote_bytes[8+metadata_size:]
                buffer_np = np.frombuffer(buffer_bytes, dtype=np.uint8)
                gemini_buffer = torch.from_numpy(buffer_np.copy())
                
                logger.info(
                    f"rank: {rank}, parsed Gemini data: "
                    f"metadata_size={metadata_size / 1024:.2f} KB, "
                    f"buffer_size={len(buffer_bytes) / (1024**2):.2f} MB"
                )
                
                # Create write_buckets structure for Gemini format
                replica_buckets = [(
                    checkpoint_dir / f"__{rank}_0.distcp",
                    'gemini_optimized_local',
                    (
                        [('gemini_metadata', gemini_metadata), ('gemini_buffer', gemini_buffer)],
                        []
                    )
                )]
                
                # Step 6: Restore state_dict from Gemini format
                logger.info(f"rank: {rank}, restoring state_dict from Gemini format...")
                loaded_state_dict = self._restore_state_dict_from_gemini_format(
                    replica_buckets, sharded_state_dict
                )
            else:
                # Standard pickle format
                logger.info(f"rank: {rank}, parsing received data as standard pickle format")
                remote_data_io = io.BytesIO(remote_bytes)
                replica_buckets = torch.load(remote_data_io, weights_only=False)
                
                logger.info(f"rank: {rank}, deserialized replica data, restoring state_dict from memory...")
                
                # Step 6: Restore state_dict from replica_buckets
                loaded_state_dict = self._restore_state_dict_from_write_buckets(
                    replica_buckets, sharded_state_dict
                )
            
            logger.info(f"rank: {rank}, successfully restored state_dict from rank0's replica data")
            return loaded_state_dict
    
    def _load_from_saved_checkpoint_file(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Load checkpoint from saved checkpoint file (OPTIMIZED for non-recovery ranks).
        
        This method is used by ranks that are not participating in rank2 recovery.
        It reads the checkpoint file that was saved by _write_bytes_to_file_with_queue.
        
        Optimizations:
        1. Keep mmap alive: Avoid copying entire file to memory
        2. Lazy parsing: Parse header first, then extract only needed parts
        3. Zero-copy buffer: Use memoryview for buffer extraction
        
        Args:
            sharded_state_dict: Sharded state dict template for loading
            checkpoint_dir: Checkpoint directory
            
        Returns:
            StateDict: Loaded state dict from saved file
        """
        import mmap
        import numpy as np
        
        rank = torch.distributed.get_rank()
        checkpoint_dir = Path(checkpoint_dir)
        
        logger.info(f"rank: {rank}, loading checkpoint from saved file (OPTIMIZED)")
        
        # Find the checkpoint file for this rank: __{rank}_0.distcp
        checkpoint_files = list(checkpoint_dir.glob(f"__{rank}_0.distcp"))
        
        if not checkpoint_files:
            logger.error(f"rank: {rank}, no checkpoint file found")
            raise FileNotFoundError(f"No checkpoint file found for rank {rank}")
        
        checkpoint_file_path = checkpoint_files[0]
        logger.info(f"rank: {rank}, found checkpoint file: {checkpoint_file_path}")
        
        # Open file with mmap (keep it alive for zero-copy access)
        checkpoint_file_size = 0
        
        try:
            f = open(checkpoint_file_path, 'rb')
            # Get file size
            f.seek(0, 2)  # Seek to end
            checkpoint_file_size = f.tell()
            f.seek(0)  # Seek back to start
            
            # Memory-map the file for zero-copy access
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            logger.info(f"rank: {rank}, opened checkpoint file with mmap: {checkpoint_file_size / (1024**2):.2f} MB")
        
        except Exception as e:
            logger.error(f"rank: {rank}, failed to open checkpoint file: {e}", exc_info=True)
            raise
        
        # Deserialize the checkpoint data (OPTIMIZED - avoid copying entire file)
        # Gemini optimized format: [metadata_size(8)] + [metadata_bytes] + [buffer_bytes]
        # Standard format: pickle serialized write_buckets
        try:
            # Priority 1: Check command-line flags to determine format
            use_gemini_optimized = False
            try:
                from megatron.training import get_args
                args = get_args()
                use_gemini_optimized = getattr(args, 'use_gemini', False) and getattr(args, 'use_gemini_optimized', False)
                if use_gemini_optimized:
                    logger.info(f"rank: {rank}, using Gemini optimized format (from args flags)")
            except Exception as e:
                # Args not available, will auto-detect format
                logger.debug(f"rank: {rank}, cannot access args, will auto-detect format: {e}")
            
            # Parse checkpoint data based on format (OPTIMIZED)
            if use_gemini_optimized or (not use_gemini_optimized and checkpoint_file_size >= 8):
                # Try Gemini format first (if flag is set, or auto-detect)
                if checkpoint_file_size >= 8:
                    # Read only the header (8 bytes) to determine format
                    metadata_size = int.from_bytes(mm[:8], byteorder='little')
                    
                    # Sanity check: metadata_size should be reasonable (< 10MB for metadata)
                    is_valid_gemini = (0 < metadata_size < 10 * 1024 * 1024 and 
                                      (8 + metadata_size) <= checkpoint_file_size)
                    
                    if use_gemini_optimized or is_valid_gemini:
                        # Parse as Gemini optimized format (ZERO-COPY)
                        logger.info(f"rank: {rank}, parsing as Gemini optimized format (zero-copy), metadata_size: {metadata_size / 1024:.2f} KB")
                        
                        # Extract metadata (only copy small metadata portion)
                        metadata_bytes = mm[8:8+metadata_size]
                        metadata_buffer = io.BytesIO(metadata_bytes)
                        gemini_metadata = torch.load(metadata_buffer, map_location='cpu', weights_only=False)
                        
                        # Extract buffer (ZERO-COPY - use memoryview to avoid copy)
                        buffer_offset = 8 + metadata_size
                        buffer_size = checkpoint_file_size - buffer_offset
                        
                        # Create numpy array from mmap buffer (zero-copy view)
                        buffer_np = np.frombuffer(mm, dtype=np.uint8, count=buffer_size, offset=buffer_offset)
                        
                        # Convert to torch tensor (make writable copy to avoid warning)
                        # Note: We need to copy here because torch.from_numpy requires writable buffer
                        # This is the only unavoidable copy in the optimized path
                        gemini_buffer = torch.from_numpy(np.array(buffer_np, copy=True))
                        
                        logger.info(
                            f"rank: {rank}, loaded Gemini checkpoint (OPTIMIZED): "
                            f"metadata_size={metadata_size / 1024:.2f} KB, "
                            f"buffer_size={buffer_size / (1024**2):.2f} MB"
                        )
                        
                        # Create write_buckets structure compatible with restoration
                        # Format: [(file_path, storage_key, (bytes_data, tensor_data))]
                        # Use a custom class to hold mmap references
                        class WriteBucketWithRefs:
                            def __init__(self, file_path, storage_key, data, mmap_ref=None, file_ref=None):
                                self.file_path = file_path
                                self.storage_key = storage_key
                                self.data = data
                                self._mmap_ref = mmap_ref
                                self._file_ref = file_ref
                            
                            def __iter__(self):
                                # Make it behave like a tuple for unpacking
                                return iter([self.file_path, self.storage_key, self.data])
                            
                            def __getitem__(self, index):
                                return [self.file_path, self.storage_key, self.data][index]
                            
                            def __len__(self):
                                return 3
                        
                        write_bucket = WriteBucketWithRefs(
                            checkpoint_file_path,
                            'gemini_optimized_local',
                            (
                                [('gemini_metadata', gemini_metadata), ('gemini_buffer', gemini_buffer)],
                                []
                            ),
                            mmap_ref=mm,
                            file_ref=f
                        )
                        
                        write_buckets = [write_bucket]
                    else:
                        # Not valid Gemini format, try standard pickle
                        logger.info(f"rank: {rank}, not valid Gemini format (metadata_size={metadata_size}), trying standard pickle")
                        checkpoint_data_bytes = mm[:]  # Copy entire file for pickle
                        checkpoint_buffer = io.BytesIO(checkpoint_data_bytes)
                        write_buckets = torch.load(checkpoint_buffer, map_location='cpu', weights_only=False)
                        mm.close()
                        f.close()
                else:
                    # File too small for Gemini format, try standard pickle
                    logger.warning(f"rank: {rank}, file too small ({checkpoint_file_size} bytes) for Gemini format, trying standard pickle")
                    checkpoint_data_bytes = mm[:]  # Copy entire file for pickle
                    checkpoint_buffer = io.BytesIO(checkpoint_data_bytes)
                    write_buckets = torch.load(checkpoint_buffer, map_location='cpu', weights_only=False)
                    mm.close()
                    f.close()
            else:
                # Standard pickle format
                logger.info(f"rank: {rank}, using standard pickle format")
                checkpoint_data_bytes = mm[:]  # Copy entire file for pickle
                checkpoint_buffer = io.BytesIO(checkpoint_data_bytes)
                write_buckets = torch.load(checkpoint_buffer, map_location='cpu', weights_only=False)
                mm.close()
                f.close()
            
            logger.info(f"rank: {rank}, deserialized checkpoint data, got {len(write_buckets)} buckets")
        
        except Exception as e:
            logger.error(f"rank: {rank}, failed to deserialize checkpoint data: {e}", exc_info=True)
            # Clean up on error
            try:
                mm.close()
                f.close()
            except:
                pass
            raise
        
        # Restore state_dict from write_buckets
        # Check if this is Gemini optimized format
        # Priority 1: Check command-line flags if available
        is_gemini_format = False
        try:
            from megatron.training import get_args
            args = get_args()
            is_gemini_format = getattr(args, 'use_gemini', False) and getattr(args, 'use_gemini_optimized', False)
            if is_gemini_format:
                logger.info(f"rank: {rank}, detected Gemini optimized format from args flags")
        except Exception as e:
            # Args not available, fall back to format detection
            logger.debug(f"rank: {rank}, cannot access args, will detect format from data: {e}")
        
        # Priority 2: If flags not set, check write_buckets structure
        if not is_gemini_format:
            is_gemini_format = (
                len(write_buckets) > 0 and 
                isinstance(write_buckets[0], tuple) and 
                len(write_buckets[0]) >= 3 and 
                write_buckets[0][1] == 'gemini_optimized_local'
            )
            if is_gemini_format:
                logger.info(f"rank: {rank}, detected Gemini optimized format from write_buckets structure")
        
        if is_gemini_format:
            logger.info(f"rank: {rank}, restoring from Gemini optimized format")
            loaded_state_dict = self._restore_state_dict_from_gemini_format(
                write_buckets, sharded_state_dict
            )
        else:
            logger.info(f"rank: {rank}, restoring from standard format")
            loaded_state_dict = self._restore_state_dict_from_write_buckets(
                write_buckets, sharded_state_dict
            )
        
        logger.info(f"rank: {rank}, successfully restored state_dict from saved checkpoint file")
        return loaded_state_dict
    
    def _load_gemini_checkpoint(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Load checkpoint using Gemini (replica) backup data with mmap.
        
        This method:
        1. Finds the replica checkpoint file for the paired rank
        2. Uses mmap to read the replica data (zero-copy)
        3. Exchanges replica data with the paired rank
        4. Writes received backup data to original checkpoint file location
        5. Loads state_dict from the restored checkpoint data
        
        Args:
            sharded_state_dict: Sharded state dict template for loading
            checkpoint_dir: Checkpoint directory
            
        Returns:
            StateDict: Loaded state dict from backup data
        """
        import mmap
        import numpy as np
        
        rank = torch.distributed.get_rank()
        
        # Get pairing map (same as in async_utils)

        paired_rank = self.pairing_map.get(rank, None)
        
        checkpoint_dir = Path(checkpoint_dir)
        
        # Find replica file: original file with _replica{paired_rank}_rank{rank} suffix
        # Example: __0_0.distcp -> __0_0_replica2_rank0.distcp
        replica_files = list(checkpoint_dir.glob(f"*_replica{paired_rank}_rank{rank}*.distcp"))
        
        replica_file_path = replica_files[0]
        logger.info(f"rank: {rank}, found replica file: {replica_file_path}")
        
        # Step 1: Use mmap to read replica file (zero-copy)
        replica_data_bytes = None
        replica_file_size = 0
        
        try:
            with open(replica_file_path, 'rb') as f:
                # Get file size
                f.seek(0, 2)  # Seek to end
                replica_file_size = f.tell()
                f.seek(0)  # Seek back to start
                
                # Memory-map the file for zero-copy access
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
                # Read all data from mmap
                replica_data_bytes = mm[:]
                
                # Close mmap (file handle will be closed automatically)
                mm.close()
            
            logger.info(f"rank: {rank}, read {replica_file_size / (1024**2):.2f} MB from replica file using mmap")
        
        except Exception as e:
            logger.error(f"rank: {rank}, failed to read replica file with mmap: {e}, "
                        f"falling back to normal load", exc_info=True)
            # return self.load(sharded_state_dict, checkpoint_dir)
        
        # Step 2: Create pair process group for communication
        from .async_utils import get_or_create_pair_process_group
        pair_group = get_or_create_pair_process_group(rank, paired_rank)
        
        # Step 3: Exchange data sizes first
        size_tensor = torch.tensor([replica_file_size], dtype=torch.long, device='cpu')
        gathered_sizes = [torch.zeros_like(size_tensor) for _ in range(2)]
        torch.distributed.all_gather(gathered_sizes, size_tensor, group=pair_group)
        
        # Determine paired rank's data size
        pair_ranks = [min(rank, paired_rank), max(rank, paired_rank)]
        my_idx = pair_ranks.index(rank)
        paired_idx = 1 - my_idx
        remote_size = gathered_sizes[paired_idx].item()
        
        logger.info(f"rank: {rank}, local replica size: {replica_file_size / (1024**2):.2f} MB, "
                   f"paired rank {paired_rank} replica size: {remote_size / (1024**2):.2f} MB")
        
        # Step 4: Convert bytes to tensor and send to paired rank
        replica_array = np.frombuffer(replica_data_bytes, dtype=np.uint8)
        replica_tensor = torch.from_numpy(replica_array.copy()).cpu()
        
        # Allocate receive buffer
        remote_tensor = torch.zeros(remote_size, dtype=torch.uint8, device='cpu')
        
        # Step 5: Exchange data using broadcast
        lower_global_rank = pair_ranks[0]
        higher_global_rank = pair_ranks[1]
        
        if rank == lower_global_rank:
            # Lower rank: send first, then receive
            torch.distributed.broadcast(replica_tensor, src=lower_global_rank, group=pair_group)
            torch.distributed.broadcast(remote_tensor, src=higher_global_rank, group=pair_group)
        else:
            # Higher rank: receive first, then send
            torch.distributed.broadcast(remote_tensor, src=lower_global_rank, group=pair_group)
            torch.distributed.broadcast(replica_tensor, src=higher_global_rank, group=pair_group)
        
        logger.info(f"rank: {rank}, exchanged replica data with rank {paired_rank}, "
                   f"received {remote_size / (1024**2):.2f} MB")
        
        # Step 6: Deserialize received data and directly restore state_dict from memory
        # Check if this is Gemini optimized format
        remote_bytes = remote_tensor.numpy().tobytes()
        
        # Try to detect Gemini optimized format
        use_gemini_optimized = False
        try:
            from megatron.training import get_args
            args = get_args()
            use_gemini_optimized = getattr(args, 'use_gemini', False) and getattr(args, 'use_gemini_optimized', False)
        except:
            pass
        
        if use_gemini_optimized and len(remote_bytes) >= 8:
            # Parse as Gemini optimized format: [metadata_size(8)] + [metadata_bytes] + [buffer_bytes]
            metadata_size = int.from_bytes(remote_bytes[:8], byteorder='little')
            
            logger.info(f"rank: {rank}, parsing received replica as Gemini optimized format, metadata_size: {metadata_size / 1024:.2f} KB")
            
            # Extract metadata
            metadata_bytes = remote_bytes[8:8+metadata_size]
            metadata_buffer = io.BytesIO(metadata_bytes)
            gemini_metadata = torch.load(metadata_buffer, map_location='cpu', weights_only=False)
            
            # Extract buffer
            buffer_bytes = remote_bytes[8+metadata_size:]
            buffer_np = np.frombuffer(buffer_bytes, dtype=np.uint8)
            gemini_buffer = torch.from_numpy(buffer_np.copy())
            
            logger.info(
                f"rank: {rank}, parsed Gemini replica data: "
                f"metadata_size={metadata_size / 1024:.2f} KB, "
                f"buffer_size={len(buffer_bytes) / (1024**2):.2f} MB"
            )
            
            # Create write_buckets structure for Gemini format
            replica_buckets = [(
                checkpoint_dir / f"__{rank}_0.distcp",
                'gemini_optimized_local',
                (
                    [('gemini_metadata', gemini_metadata), ('gemini_buffer', gemini_buffer)],
                    []
                )
            )]
            
            # Step 7: Restore state_dict from Gemini format
            logger.info(f"rank: {rank}, restoring state_dict from Gemini replica...")
            loaded_state_dict = self._restore_state_dict_from_gemini_format(
                replica_buckets, sharded_state_dict
            )
        else:
            # Standard pickle format
            logger.info(f"rank: {rank}, parsing received replica as standard pickle format")
            remote_data_io = io.BytesIO(remote_bytes)
            
            # Deserialize the checkpoint data (write_buckets)
            replica_buckets = torch.load(remote_data_io, weights_only=False)
            logger.info(f"rank: {rank}, deserialized replica data, restoring state_dict from memory...")
            
            # Step 7: Directly restore state_dict from replica_buckets
            # Parse write_buckets and extract data to populate sharded_state_dict
            loaded_state_dict = self._restore_state_dict_from_write_buckets(
                replica_buckets, sharded_state_dict
            )
        
        logger.info(f"rank: {rank}, successfully restored state_dict from backup data in memory")
        return loaded_state_dict
        
    def _restore_state_dict_from_gemini_format(
        self, write_buckets: List, sharded_state_dict: ShardedStateDict
    ) -> StateDict:
        """Restore state_dict from Gemini optimized format (OPTIMIZED).
        
        Gemini optimized format stores checkpoint as:
        - gemini_metadata: contains non_tensor_data and tensor_infos
        - gemini_buffer: continuous buffer with all tensor data (torch.Tensor or numpy array)
        
        Optimizations:
        1. Zero-copy tensor views: Use .view() instead of .clone() when safe
        2. Direct buffer slicing: Avoid intermediate copies
        3. Conditional cloning: Only clone when necessary for memory safety
        
        Args:
            write_buckets: List with single bucket containing Gemini format data
            sharded_state_dict: Template sharded state dict to populate
            
        Returns:
            StateDict: Restored state dict
        """
        rank = torch.distributed.get_rank()
        logger.info(f"rank: {rank}, restoring state_dict from Gemini optimized format (OPTIMIZED)")
        
        # Extract gemini_metadata and gemini_buffer
        if len(write_buckets) == 0 or len(write_buckets[0]) < 3:
            raise ValueError(f"rank: {rank}, invalid Gemini write_buckets format")
        
        _, storage_key, (bytes_data, _) = write_buckets[0]
        
        gemini_metadata = None
        gemini_buffer = None
        
        for key, value in bytes_data:
            if key == 'gemini_metadata':
                gemini_metadata = value
            elif key == 'gemini_buffer':
                gemini_buffer = value
        
        if gemini_metadata is None or gemini_buffer is None:
            raise ValueError(f"rank: {rank}, missing gemini_metadata or gemini_buffer")
        
        # Ensure gemini_buffer is a torch.Tensor for zero-copy operations
        if not isinstance(gemini_buffer, torch.Tensor):
            logger.info(f"rank: {rank}, converting gemini_buffer to torch.Tensor")
            import numpy as np
            if isinstance(gemini_buffer, np.ndarray):
                gemini_buffer = torch.from_numpy(gemini_buffer)
            else:
                raise TypeError(f"rank: {rank}, gemini_buffer must be torch.Tensor or numpy.ndarray, got {type(gemini_buffer)}")
        
        logger.info(
            f"rank: {rank}, extracted Gemini data (OPTIMIZED): "
            f"buffer_size={gemini_buffer.numel() / (1024**2):.2f} MB, "
            f"num_tensors={len(gemini_metadata.get('tensor_infos', []))}"
        )
        
        # Rebuild state_dict from gemini_metadata and gemini_buffer
        # The metadata contains:
        # - non_tensor_data: dict of non-tensor items
        # - tensor_infos: list of dicts with keys: 'key', 'shape', 'dtype', 'offset', 'size_bytes', etc.
        
        non_tensor_data_raw = gemini_metadata.get('non_tensor_data', {})
        tensor_infos = gemini_metadata.get('tensor_infos', [])
        
        # Deserialize non_tensor_data
        # In Gemini format, BytesIO content was extracted as bytes before saving
        # We need to convert bytes back to BytesIO and then deserialize
        non_tensor_data = {}
        for key, data in non_tensor_data_raw.items():
            if isinstance(data, bytes):
                # Bytes data from BytesIO.getvalue() - need to deserialize
                try:
                    data_io = io.BytesIO(data)
                    deserialized_list = torch.load(data_io, map_location='cpu', weights_only=False)
                    non_tensor_data[key] = deserialized_list
                    logger.debug(f"rank: {rank}, deserialized bytes for key: {key}, got {len(deserialized_list) if isinstance(deserialized_list, list) else 1} items")
                except Exception as e:
                    logger.warning(f"rank: {rank}, failed to deserialize bytes for key {key}: {e}")
                    # Fallback: use as is
                    non_tensor_data[key] = data
            elif isinstance(data, io.BytesIO):
                # Still a BytesIO object (shouldn't happen with new code, but handle it)
                try:
                    data.seek(0)
                    deserialized_list = torch.load(data, map_location='cpu', weights_only=False)
                    non_tensor_data[key] = deserialized_list
                    logger.debug(f"rank: {rank}, deserialized BytesIO for key: {key}")
                except Exception as e:
                    logger.warning(f"rank: {rank}, failed to deserialize BytesIO for key {key}: {e}")
                    non_tensor_data[key] = data
            else:
                # Other types - use as is (shouldn't happen in normal case)
                non_tensor_data[key] = data
                logger.debug(f"rank: {rank}, using data as-is for key: {key}, type: {type(data)}")
        
        logger.info(f"rank: {rank}, processed {len(non_tensor_data)} non-tensor items from metadata")
        
        # Step 1: Reconstruct tensors from buffer (OPTIMIZED - minimize copies)
        tensor_dict = {}
        for info in tensor_infos:
            key = info['key']
            shape = tuple(info['shape'])
            dtype_str = info['dtype']
            offset = info['offset']
            size_bytes = info['size_bytes']
            
            # Parse dtype string (e.g., 'torch.float32' -> torch.float32)
            if dtype_str.startswith('torch.'):
                dtype_name = dtype_str.split('.')[1]
                dtype = getattr(torch, dtype_name, torch.float32)
            else:
                dtype = torch.float32  # Default fallback
            
            # Calculate element size for dtype
            element_size = {
                torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
                torch.float64: 8, torch.int32: 4, torch.int64: 8,
                torch.int16: 2, torch.int8: 1, torch.uint8: 1,
                torch.bool: 1,
            }.get(dtype, 4)
            
            # Calculate number of elements
            numel = size_bytes // element_size
            
            # Extract tensor data from buffer (ZERO-COPY when possible)
            # gemini_buffer is uint8, need to view as target dtype
            start_offset = offset
            end_offset = offset + size_bytes
            
            # Create a view of the buffer (zero-copy slice)
            tensor_bytes = gemini_buffer[start_offset:end_offset]
            
            # Convert to target dtype and reshape (OPTIMIZED)
            # Use .view() for zero-copy type conversion, then reshape
            # Only clone if we need to ensure memory contiguity for reshape
            try:
                # Try zero-copy path: view + reshape without clone
                tensor = tensor_bytes.view(dtype)[:numel].reshape(shape)
                
                # Check if tensor is contiguous; if not, make it contiguous
                # This is necessary for some operations but avoids unnecessary clones
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                
                # Note: We don't clone here unless necessary
                # The tensor shares memory with gemini_buffer, which is safe
                # as long as gemini_buffer stays alive (it's referenced in write_buckets)
                
            except Exception as e:
                # Fallback: clone if zero-copy path fails
                logger.debug(f"rank: {rank}, zero-copy failed for {key}, using clone: {e}")
                tensor = tensor_bytes.view(dtype)[:numel].reshape(shape).clone()
            
            tensor_dict[key] = tensor
            logger.debug(f"rank: {rank}, reconstructed tensor (zero-copy): {key}, shape: {shape}, dtype: {dtype}")
        
        logger.info(f"rank: {rank}, reconstructed {len(tensor_dict)} tensors, {len(non_tensor_data)} non-tensor items")
        
        # Step 2: Match reconstructed data with sharded_state_dict
        # Use the same matching logic as _restore_state_dict_from_write_buckets
        (keyed_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        )
        
        matched_count = 0
        unmatched_count = 0
        
        for key, sh_base_list in keyed_state_dict.items():
            for idx, sh_base in enumerate(sh_base_list):
                if isinstance(sh_base, ShardedObject):
                    # Match non-tensor data
                    if key in non_tensor_data:
                        data = non_tensor_data[key]
                        if isinstance(data, list) and idx < len(data):
                            sh_base.data = data[idx]
                            logger.debug(f"rank: {rank}, matched ShardedObject: {key}[{idx}], data type: {type(data[idx])}")
                        else:
                            sh_base.data = data
                            logger.debug(f"rank: {rank}, matched ShardedObject (whole): {key}, data type: {type(data)}")
                        matched_count += 1
                    else:
                        unmatched_count += 1
                        logger.debug(f"rank: {rank}, unmatched ShardedObject: {key}")
                
                elif isinstance(sh_base, ShardedTensor):
                    # Match tensor data
                    if key in tensor_dict:
                        sh_base.data = tensor_dict[key]
                        matched_count += 1
                        logger.debug(f"rank: {rank}, matched ShardedTensor: {key}")
                    else:
                        unmatched_count += 1
                        logger.debug(f"rank: {rank}, unmatched ShardedTensor: {key}")
        
        logger.info(f"rank: {rank}, matched {matched_count} items, unmatched {unmatched_count} items")
        
        # Step 3: Unwrap and convert to regular state dict
        unwrapped_state_dict = {}
        for key, sh_base_list in keyed_state_dict.items():
            if len(sh_base_list) == 0:
                continue
            
            sh_base = sh_base_list[0]
            if isinstance(sh_base, ShardedTensor):
                tensors = []
                for sh in sh_base_list:
                    ten = sh.data
                    if ten is None:
                        tensors.append(None)
                        continue
                    
                    # Handle prepend_axis_num: remove singleton dimensions added during save
                    if hasattr(sh, 'prepend_axis_num') and sh.prepend_axis_num > 0:
                        for _ in range(sh.prepend_axis_num):
                            if isinstance(ten, torch.Tensor) and ten.size(0) == 1:
                                ten = ten[0]  # Remove first singleton dimension
                    
                    tensors.append(ten)
                unwrapped_state_dict[key] = tensors
            elif isinstance(sh_base, ShardedObject):
                # For ShardedObject, create a list of data
                data_list = [sh.data for sh in sh_base_list]
                # Keep as list format (consistent with _restore_state_dict_from_write_buckets)
                unwrapped_state_dict[key] = data_list
        
        # Step 4: Convert back to MCore format
        orig_sharded_state_dict = sharded_state_dict
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
            unwrapped_state_dict, flat_mapping, rename_mapping
        )
        
        # Step 5: Restore dict types
        self._restore_dict_types_lenient(mcore_state_dict, orig_sharded_state_dict)
        
        logger.info(f"rank: {rank}, successfully restored {len(mcore_state_dict)} items from Gemini format")
        return mcore_state_dict
    
    def _restore_state_dict_from_write_buckets(
        self, write_buckets: List, sharded_state_dict: ShardedStateDict
    ) -> StateDict:
        """Restore state_dict directly from write_buckets in memory.
        
        write_buckets structure: [(file_name, storage_key, (bytes_data, tensor_data)), ...]
        - bytes_data: [(WriteItem, data), ...] for non-tensor data
        - tensor_data: [(WriteItem, tensor), ...] for tensor data
        
        This method directly extracts data from write_buckets and populates sharded_state_dict,
        avoiding the need to write to temporary files.
        
        Args:
            write_buckets: List of write buckets from deserialized checkpoint data
            sharded_state_dict: Template sharded state dict to populate
            
        Returns:
            StateDict: Restored state dict
        """
        rank = torch.distributed.get_rank()
        
        logger.info(f"rank: {rank}, restoring state_dict directly from write_buckets in memory")
        
        # Step 1: Build a mapping from FQN to data
        # FQN (Fully Qualified Name) is the key in WriteItem.index.fqn
        fqn_to_data = {}
        fqn_to_tensor_list = {}  # For tensors with same FQN but different offsets
        
        for bucket in write_buckets:
            if isinstance(bucket, tuple) and len(bucket) >= 3:
                file_name, storage_key, (bytes_data, tensor_data) = bucket
                
                # Process bytes_data (non-tensor data: metadata, ShardedObjects, etc.)
                if isinstance(bytes_data, list):
                    for write_item, data in bytes_data:
                        fqn = write_item.index.fqn
                        
                        # BytesIO objects contain serialized data (list of ShardedObject.data)
                        # We need to deserialize them
                        if isinstance(data, io.BytesIO):
                            # Reset position to beginning
                            data.seek(0)
                            # Deserialize: torch.save([sh_obj.data for sh_obj in sh_objs], ...)
                            # So torch.load returns a list
                            deserialized_list = torch.load(data, map_location='cpu', weights_only=False)
                            # Store the deserialized list
                            fqn_to_data[fqn] = deserialized_list
                            logger.debug(f"rank: {rank}, extracted and deserialized BytesIO for FQN: {fqn}, "
                                       f"got {len(deserialized_list) if isinstance(deserialized_list, list) else 1} items")
                        else:
                            # Other types of data (shouldn't happen in standard format)
                            fqn_to_data[fqn] = data
                            logger.debug(f"rank: {rank}, extracted bytes_data for FQN: {fqn}")
                
                # Process tensor_data (ShardedTensors)
                if isinstance(tensor_data, list):
                    for write_item, tensor in tensor_data:
                        fqn = write_item.index.fqn
                        offset = tuple(write_item.index.offset) if hasattr(write_item.index, 'offset') else ()
                        shard_index = write_item.index.index if hasattr(write_item.index, 'index') else 0
                        
                        # Create unique key: (fqn, offset, shard_index)
                        key = (fqn, offset, shard_index)
                        
                        if fqn not in fqn_to_tensor_list:
                            fqn_to_tensor_list[fqn] = []
                        fqn_to_tensor_list[fqn].append((offset, shard_index, tensor))
                        
                        logger.debug(f"rank: {rank}, extracted tensor for FQN: {fqn}, offset: {offset}, index: {shard_index}")
        
        logger.info(f"rank: {rank}, extracted {len(fqn_to_data)} non-tensor items, "
                   f"{len(fqn_to_tensor_list)} tensor groups from write_buckets")
        
        # Step 2: Generate PyT-compatible state dict from sharded_state_dict
        orig_sharded_state_dict = sharded_state_dict
        (keyed_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        )
        
        # Step 3: Match and populate data
        matched_count = 0
        unmatched_count = 0
        
        for key, sh_base_list in keyed_state_dict.items():
            for idx, sh_base in enumerate(sh_base_list):
                if isinstance(sh_base, ShardedObject):
                    # Match by FQN for ShardedObject
                    if key in fqn_to_data:
                        # fqn_to_data[key] is a list: [sh_obj.data for sh_obj in sh_objs]
                        # Each sh_base in sh_base_list corresponds to one element in the list
                        data_list = fqn_to_data[key]
                        if isinstance(data_list, list) and idx < len(data_list):
                            sh_base.data = data_list[idx]
                            matched_count += 1
                            logger.debug(f"rank: {rank}, matched ShardedObject: {key}[{idx}]")
                        else:
                            # Fallback: if not a list or index out of range, use the whole thing
                            sh_base.data = data_list
                            matched_count += 1
                            logger.debug(f"rank: {rank}, matched ShardedObject (fallback): {key}")
                    else:
                        unmatched_count += 1
                        logger.debug(f"rank: {rank}, unmatched ShardedObject: {key}")
                
                elif isinstance(sh_base, ShardedTensor):
                    # Match by (FQN, offset) for ShardedTensor
                    sh_offset = tuple(sh_base.global_offset) if hasattr(sh_base.global_offset, '__iter__') else (sh_base.global_offset,)
                    
                    if key in fqn_to_tensor_list:
                        # Find matching tensor by offset
                        found = False
                        for offset, shard_index, tensor in fqn_to_tensor_list[key]:
                            if offset == sh_offset:
                                sh_base.data = tensor
                                matched_count += 1
                                found = True
                                logger.debug(f"rank: {rank}, matched ShardedTensor: {key}, offset: {sh_offset}")
                                break
                        
                        if not found:
                            unmatched_count += 1
                            logger.debug(f"rank: {rank}, unmatched ShardedTensor: {key}, offset: {sh_offset}")
                    else:
                        unmatched_count += 1
                        logger.debug(f"rank: {rank}, unmatched ShardedTensor (no FQN): {key}")
        
        logger.info(f"rank: {rank}, matched {matched_count} items, unmatched {unmatched_count} items")
        
        # Step 4: Unwrap and convert to MCore format
        unwrapped_state_dict = {}
        for key, sh_base_list in keyed_state_dict.items():
            if len(sh_base_list) == 0:
                continue
            
            sh_base = sh_base_list[0]
            if isinstance(sh_base, ShardedTensor):
                tensors = []
                for sh in sh_base_list:
                    ten = sh.data
                    if ten is None:
                        tensors.append(None)
                        continue
                    
                    # Handle prepend_axis_num: remove singleton dimensions added during save
                    # These are extra dimensions at the beginning of the tensor
                    # e.g., [1, 1024, 1024] -> [1024, 1024] when prepend_axis_num=1
                    if hasattr(sh, 'prepend_axis_num') and sh.prepend_axis_num > 0:
                        for _ in range(sh.prepend_axis_num):
                            if isinstance(ten, torch.Tensor) and ten.size(0) == 1:
                                ten = ten[0]  # Remove first singleton dimension
                    
                    tensors.append(ten)
                unwrapped_state_dict[key] = tensors
            elif isinstance(sh_base, ShardedObject):
                # For ShardedObject, create a list of data
                data_list = [sh.data for sh in sh_base_list]
                # If there's only one element, unwrap it (standard case)
                # Otherwise keep as list (for replicated objects)
                
                if len(data_list) == 1:
                    unwrapped_state_dict[key] = data_list
                else:
                    unwrapped_state_dict[key] = data_list
        
        # Step 5: Convert back to MCore format
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
            unwrapped_state_dict, flat_mapping, rename_mapping
        )
        
        # Step 6: Restore dict types
        self._restore_dict_types_lenient(mcore_state_dict, orig_sharded_state_dict)
        
        logger.info(f"rank: {rank}, successfully restored state_dict from write_buckets in memory")
        return mcore_state_dict
    
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
                
                # Do not pre-fill partner_buffer here; pipeline loop will copy actual data
                # and pad with zeros per chunk to max_total_bytes.
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
            pass
    
    def _run_eclatin_recovery_pipeline(
        self,
        rank: int,
        world_size: int,
        registry,
        eclatin_blocks: Dict[str, torch.Tensor],
        recv_buffers: Optional[Dict[str, torch.Tensor]],
        recovered_buffer: Optional[torch.Tensor],
        total_size: int,
    ) -> None:
        """
        Run ECLATIN recovery pipeline to recover rank2 data.
        
        Similar to EC-CHECK for rank2 recovery:
        - rank2: Receives 6 blocks from rank0/1/3, runs load_recover to recover 4 blocks
        - rank0/1/3: Send their blocks to rank2 using load_send_blocks
        
        Args:
            rank (int): Current rank
            world_size (int): Total number of ranks
            registry: GlobalMetadataRegistry
            eclatin_blocks (Dict[str, torch.Tensor]): 4 allocated blocks (all ranks)
            recv_buffers (Optional[Dict[str, torch.Tensor]]): 6 recv buffers (rank2 only)
            recovered_buffer (Optional[torch.Tensor]): Buffer to store recovered data (rank2 only)
            total_size (int): Total size of data to recover
        """
        import ctypes
        
        if not self.eclatin_manager.use_eclatin:
            logger.warning("ECLATIN: Manager not enabled, skipping recovery pipeline")
            return
        
        if self.eclatin_manager._eclatin_native is None:
            logger.error("ECLATIN: Native module not initialized")
            return
        
        # === Step 1: Set load mode in C++ native module ===
        failed_rank = 2  # ECLATIN recovers rank2
        self.eclatin_manager._eclatin_native.set_load_mode(True, failed_rank)
        logger.info(f"ECLATIN: [Rank {rank}] Set load mode (failed_rank={failed_rank})")
        
        # === Step 1.5: Initialize load connections ===
        # Get network configuration for load mode ports
        # All ranks need rank2's network config to get the correct ports
        net_config_rank2 = self.eclatin_manager._get_eclatin_network_config(2, world_size)
        rank2_ip = net_config_rank2['rank_ips'].get(2, net_config_rank2['my_ip'])
        
        # All ranks use rank2's recv ports (rank2 listens, rank0/1/3 connect)
        load_recv_rank0_data2_port = net_config_rank2['ports']['load_recv_rank0_data2']
        load_recv_rank0_parity2_port = net_config_rank2['ports']['load_recv_rank0_parity2']
        load_recv_rank1_data1_port = net_config_rank2['ports']['load_recv_rank1_data1']
        load_recv_rank1_parity1_port = net_config_rank2['ports']['load_recv_rank1_parity1']
        load_recv_rank3_data1_port = net_config_rank2['ports']['load_recv_rank3_data1']
        load_recv_rank3_data2_port = net_config_rank2['ports']['load_recv_rank3_data2']
        
        # Similar to EC-CHECK: rank2 starts accept operations first, then other ranks connect
        if rank == 2:
            # rank2: Initialize accept operations (will start accept threads)
            logger.info(f"ECLATIN: [Rank 2] Initializing load accept connections...")
            self.eclatin_manager._eclatin_native.init_load_connections(
                rank,
                rank2_ip,
                load_recv_rank0_data2_port,
                load_recv_rank0_parity2_port,
                load_recv_rank1_data1_port,
                load_recv_rank1_parity1_port,
                load_recv_rank3_data1_port,
                load_recv_rank3_data2_port
            )
            logger.info(f"ECLATIN: [Rank 2] Accept operations started, waiting for other ranks...")
        
        # Synchronize: ensure rank2's acceptors are ready before other ranks connect
        torch.distributed.barrier()
        
        if rank != 2:
            # rank0/1/3: Connect to rank2 (will block until connected)
            logger.info(f"ECLATIN: [Rank {rank}] Connecting load send sockets to rank2...")
            self.eclatin_manager._eclatin_native.init_load_connections(
                rank,
                rank2_ip,
                load_recv_rank0_data2_port,
                load_recv_rank0_parity2_port,
                load_recv_rank1_data1_port,
                load_recv_rank1_parity1_port,
                load_recv_rank3_data1_port,
                load_recv_rank3_data2_port
            )
            logger.info(f"ECLATIN: [Rank {rank}] Load send sockets connected")
        
        # Wait for all connections to be established
        logger.info(f"ECLATIN: [Rank {rank}] Waiting for load connections to be established...")
        self.eclatin_manager._eclatin_native.wait_for_load_connections(timeout_seconds=30)
        
        # Synchronize to ensure all connections are established
        torch.distributed.barrier()
        logger.info(f"ECLATIN: [Rank {rank}] Load connections initialized")
        
        # === Step 2: rank2: Receive blocks and recover ===
        if rank == 2:
            if recv_buffers is None or recovered_buffer is None:
                logger.error("ECLATIN: [Rank 2] recv_buffers or recovered_buffer is None")
                return
            
            # Get base addresses for recv buffers
            rank0_data2_addr = int(recv_buffers['rank0_data2'].data_ptr())
            rank0_parity2_addr = int(recv_buffers['rank0_parity2'].data_ptr())
            rank1_data1_addr = int(recv_buffers['rank1_data1'].data_ptr())
            rank1_parity1_addr = int(recv_buffers['rank1_parity1'].data_ptr())
            rank3_data1_addr = int(recv_buffers['rank3_data1'].data_ptr())
            rank3_data2_addr = int(recv_buffers['rank3_data2'].data_ptr())
            
            # Get base addresses for recovered blocks
            recovered_data1_addr = int(eclatin_blocks['data_block_1'].data_ptr())
            recovered_data2_addr = int(eclatin_blocks['data_block_2'].data_ptr())
            recovered_parity1_addr = int(eclatin_blocks['parity_block_1'].data_ptr())
            recovered_parity2_addr = int(eclatin_blocks['parity_block_2'].data_ptr())
            
            # Calculate aligned half block size (same as save phase)
            aligned_half_block_size = eclatin_blocks['data_block_1'].numel()
            
            logger.info(
                f"ECLATIN: [Rank 2] Starting recovery pipeline\n"
                f"  Recv buffers: {aligned_half_block_size / (1024**3):.2f} GB each\n"
                f"  Recovered blocks: {aligned_half_block_size / (1024**3):.2f} GB each"
            )
            
            # Call C++ load_recover: receives 6 blocks and recovers 4 blocks
            self.eclatin_manager._eclatin_native.load_recover(
                rank0_data2_addr, rank0_parity2_addr,
                rank1_data1_addr, rank1_parity1_addr,
                rank3_data1_addr, rank3_data2_addr,
                recovered_data1_addr, recovered_data2_addr,
                recovered_parity1_addr, recovered_parity2_addr,
                aligned_half_block_size
            )
            
            logger.info("ECLATIN: [Rank 2] Recovery pipeline completed")
            
            # Copy recovered blocks to recovered_buffer (combine data_block_1 and data_block_2)
            # CRITICAL FIX: Use actual_tensor_buffer_size // 2 as split point (same as save phase's actual_data_bytes // 2)
            # Save phase splits actual data at actual_data_bytes // 2, not pipeline_total_bytes // 2
            # Calculate actual_tensor_buffer_size from registry (same as save phase's actual_data_bytes)
            actual_tensor_buffer_size = 0
            for r in range(world_size):
                rank_metadata = registry.rank_metadata.get(r, [])
                rank_actual_size = sum(meta.size_bytes for meta in rank_metadata)
                if rank_actual_size > actual_tensor_buffer_size:
                    actual_tensor_buffer_size = rank_actual_size
            
            # Use actual_tensor_buffer_size // 2 (same as save phase's actual_data_bytes // 2)
            # This ensures the split point matches save phase exactly
            half_actual_data = actual_tensor_buffer_size // 2  # Same split point as save phase
            
            if recovered_buffer.numel() >= total_size:
                # Copy first half: from data_block_1[0:half_actual_data]
                first_half_actual = min(half_actual_data, total_size)
                recovered_buffer[:first_half_actual].copy_(
                    eclatin_blocks['data_block_1'][:first_half_actual]
                )
                
                # Copy second half: from data_block_2[0:remaining_data] if total_size > half_actual_data
                if total_size > half_actual_data:
                    second_half_size = total_size - half_actual_data
                    recovered_buffer[first_half_actual:total_size].copy_(
                        eclatin_blocks['data_block_2'][:second_half_size]
                    )
                
                logger.info(
                    f"ECLATIN: [Rank 2] Copied recovered data to buffer "
                    f"({total_size / (1024**3):.2f} GB): "
                    f"actual_tensor_buffer_size={actual_tensor_buffer_size / (1024**3):.2f} GB, "
                    f"half_actual_data={half_actual_data / (1024**3):.2f} GB, "
                    f"first half {first_half_actual / (1024**3):.2f} GB from data_block_1, "
                    f"second half {(total_size - first_half_actual) / (1024**3):.2f} GB from data_block_2"
                )
            else:
                logger.warning(
                    f"ECLATIN: [Rank 2] recovered_buffer too small "
                    f"({recovered_buffer.numel()} < {total_size})"
                )
        
        # === Step 3: rank0/1/3: Send blocks to rank2 ===
        else:
            aligned_half_block_size = eclatin_blocks['data_block_1'].numel()
            
            if rank == 0:
                # rank0 sends: data_block_2, parity_block_2
                data2_addr = int(eclatin_blocks['data_block_2'].data_ptr())
                parity2_addr = int(eclatin_blocks['parity_block_2'].data_ptr())
                
                logger.info(f"ECLATIN: [Rank 0] Sending data_block_2 and parity_block_2 to rank2")
                self.eclatin_manager._eclatin_native.load_send_blocks(
                    'rank0_data2', data2_addr,
                    'rank0_parity2', parity2_addr,
                    aligned_half_block_size
                )
            
            elif rank == 1:
                # rank1 sends: data_block_1, parity_block_1
                data1_addr = int(eclatin_blocks['data_block_1'].data_ptr())
                parity1_addr = int(eclatin_blocks['parity_block_1'].data_ptr())
                
                logger.info(f"ECLATIN: [Rank 1] Sending data_block_1 and parity_block_1 to rank2")
                self.eclatin_manager._eclatin_native.load_send_blocks(
                    'rank1_data1', data1_addr,
                    'rank1_parity1', parity1_addr,
                    aligned_half_block_size
                )
            
            elif rank == 3:
                # rank3 sends: data_block_1, data_block_2
                data1_addr = int(eclatin_blocks['data_block_1'].data_ptr())
                data2_addr = int(eclatin_blocks['data_block_2'].data_ptr())
                
                logger.info(f"ECLATIN: [Rank 3] Sending data_block_1 and data_block_2 to rank2")
                self.eclatin_manager._eclatin_native.load_send_blocks(
                    'rank3_data1', data1_addr,
                    'rank3_data2', data2_addr,
                    aligned_half_block_size
                )
            
            logger.info(f"ECLATIN: [Rank {rank}] Sent blocks to rank2")
        
        # Synchronize all ranks
        torch.distributed.barrier()
        logger.info(f"ECLATIN: [Rank {rank}] Recovery pipeline completed")
    
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
        
        # Check if rank2 has recovered data from P2P pipeline
        if (rank == 2 and hasattr(self, 'eccheck_recovered_buffer') 
            and self.eccheck_recovered_buffer is not None):
            logger.info(f"EC-CHECK: [Rank {rank}] Using recovered data from P2P pipeline")
            
            # Extract tensors from recovered buffer using saved metadata
            decomposed = self._extract_decomposed_from_buffer(
                self.eccheck_recovered_buffer,
                self.eccheck_recovered_metadata,
                self.eccheck_recovered_registry
            )
            
            # Clear saved data
            self.eccheck_recovered_buffer = None
            self.eccheck_recovered_metadata = None
            self.eccheck_recovered_registry = None
        else:
            # Normal path: load from file
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

    def _load_eclatin_checkpoint(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Load checkpoint saved in ECLATIN format (mirror EC-CHECK flow)."""
        from .filesystem_async import FileSystemWriterAsync
        from .state_dict_decomposer import reconstruct_state_dict
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Recovery path: rank2 may have recovered buffer
        #if (False):
        if (rank == 2 and hasattr(self, 'eclatin_recovered_buffer')
            and self.eclatin_recovered_buffer is not None):
            logger.info(f"ECLATIN: [Rank {rank}] Using recovered data from recovery pipeline")
            decomposed = self._extract_decomposed_from_buffer(
                self.eclatin_recovered_buffer,
                self.eclatin_recovered_metadata,
                self.eclatin_recovered_registry
            )
            self.eclatin_recovered_buffer = None
            self.eclatin_recovered_metadata = None
            self.eclatin_recovered_registry = None
        else:
            checkpoint_dir = Path(checkpoint_dir)
            eclatin_file = checkpoint_dir / f'__{rank}_0.distcp'
            if not eclatin_file.exists():
                raise FileNotFoundError(f"ECLATIN file not found for rank {rank}: {eclatin_file}")
            logger.info(f"Loading ECLATIN checkpoint from {eclatin_file}")
            decomposed = FileSystemWriterAsync.load_eclatin_components_from_file(str(eclatin_file))
        
        # Build index map: (key, global_offset) -> (info, tensor)
        logger.info(f"ECLATIN: Building index map from {len(decomposed.tensor_infos)} tensor infos")
        index_to_data = {}
        for info, tensor in zip(decomposed.tensor_infos, decomposed.tensor_data):
            # Normalize global_offset to tuple format for consistent matching
            info_offset = info.global_offset
            if info_offset is None:
                info_offset = ()
            elif not isinstance(info_offset, tuple):
                info_offset = tuple(info_offset) if hasattr(info_offset, '__iter__') else (info_offset,)
            index_key = (info.key, info_offset)
            index_to_data[index_key] = (info, tensor)
        
        non_tensor_by_fqn = decomposed.non_tensor_data
        logger.info(
            f"Successfully loaded ECLATIN checkpoint for rank {rank} "
            f"({len(decomposed.tensor_data)} tensors, "
            f"{decomposed.total_tensor_size_bytes / (1024**3):.2f} GB)"
        )
        
        orig_sharded_state_dict = sharded_state_dict
        (keyed_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        )
        
        matched_count = 0
        unmatched_count = 0
        for key, sh_base_list in keyed_state_dict.items():
            for sh_base in sh_base_list:
                if isinstance(sh_base, ShardedObject):
                    if key in non_tensor_by_fqn:
                        value = non_tensor_by_fqn[key]
                        # Handle BytesIO wrapper (support both '_eccheck_type' and '_eclatin_type' for backward compatibility)
                        if isinstance(value, dict) and ('_eccheck_type' in value or '_eclatin_type' in value):
                            wrapper_type = value.get('_eccheck_type') or value.get('_eclatin_type')
                            if wrapper_type == 'BytesIO':
                                bytes_data = value.get('_eccheck_data') or value.get('_eclatin_data')
                                bytes_io = io.BytesIO(bytes_data)
                                deserialized_list = torch.load(bytes_io, map_location='cpu', weights_only=False)
                                value = deserialized_list[0] if isinstance(deserialized_list, list) else deserialized_list
                        sh_base.data = value
                        matched_count += 1
                    else:
                        unmatched_count += 1
                        logger.warning(f"ECLATIN: [Rank {rank}] Unmatched ShardedObject: key={key}")
                elif isinstance(sh_base, ShardedTensor):
                    sh_offset = tuple(sh_base.global_offset) if hasattr(sh_base.global_offset, '__iter__') else (sh_base.global_offset,)
                    lookup_key = (key, sh_offset)
                    if lookup_key in index_to_data:
                        _, tensor = index_to_data[lookup_key]
                        sh_base.data = tensor
                        matched_count += 1
                        if tensor is None:
                            logger.error(f"ECLATIN: Matched key {lookup_key} but tensor is None!")
                    else:
                        unmatched_count += 1
                        logger.warning(f"ECLATIN: [Rank {rank}] Unmatched ShardedTensor: key={key}, global_offset={sh_offset}, lookup_key={lookup_key}")
        
        logger.info(f"ECLATIN: Matched {matched_count} ShardedBase objects")
        if unmatched_count > 0:
            logger.warning(f"ECLATIN: {unmatched_count} ShardedBase objects were not matched - this may cause NaN after several iterations!")
        
        # Unwrap to plain state_dict
        unwrapped_state_dict = {}
        for key, sh_base_list in keyed_state_dict.items():
            if len(sh_base_list) == 0:
                continue
            sh_base = sh_base_list[0]
            if isinstance(sh_base, ShardedTensor):
                tensors = []
                for sh in sh_base_list:
                    ten = sh.data
                    if ten is None:
                        tensors.append(None)
                        continue
                    if sh.flattened_range is not None:
                        assert ten.shape[:-1] == (1,) * (len(ten.shape) - 1), ten.shape
                        ten = ten.view(-1)
                    else:
                        for _ in range(sh.prepend_axis_num):
                            if ten.size(0) == 1:
                                ten = ten[0]
                    tensors.append(ten)
                unwrapped_state_dict[key] = tensors
            elif isinstance(sh_base, ShardedObject):
                unwrapped_state_dict[key] = [sh.data for sh in sh_base_list]
        
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
            unwrapped_state_dict, flat_mapping, rename_mapping  # type: ignore[arg-type]
        )
        self._restore_dict_types_lenient(mcore_state_dict, orig_sharded_state_dict)
        return mcore_state_dict

    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Translates MCore ShardedTensors to PyT ShardedTensors & loads from PyT Distributed fmt.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict with mapping
                information to instruct loading
            checkpoint_dir (Path): checkpoint directory

        Returns: loaded state dict
        """
        # Check if this is an EC-CHECK format checkpoint
        rank = torch.distributed.get_rank()
        pair_rank = self.pairing_map.get(rank, None)
        from megatron.training import get_args as use_args
        input_args = use_args()
        
        # Gemini checkpoint recovery for rank2 failure scenario
        # Only rank0 and rank2 participate in recovery, but ALL ranks must synchronize
        recovered_state_dict = None
        if input_args.use_gemini and input_args.use_gemini_hardware_failure:
            # Check if this is a rank2 recovery scenario
            is_rank2_recovery = (rank == 2 or pair_rank == 2)
            start_recovery_time = time()
            if is_rank2_recovery:
                logger.info(f"rank: {rank}, using Gemini checkpoint recovery for rank2 failure")
                # Only rank0 (pair_rank=2) and rank2 participate in recovery
                # Use ASIO-based recovery if Gemini optimized mode is enabled
                if input_args.use_gemini_optimized:
                    logger.info(f"rank: {rank}, using ASIO-based recovery for rank2 failure")
                    recovered_state_dict = self._load_gemini_checkpoint_recovery_asio(sharded_state_dict, checkpoint_dir)
                else:
                    logger.info(f"rank: {rank}, using standard recovery for rank2 failure")
                    recovered_state_dict = self._load_gemini_checkpoint_recovery(sharded_state_dict, checkpoint_dir)
            else:
                logger.info(f"rank: {rank}, not participating in rank2 recovery, loading from own checkpoint file")
                # Other ranks (rank1, rank3) load from their own saved checkpoint files
                recovered_state_dict = self._load_from_saved_checkpoint_file(sharded_state_dict, checkpoint_dir)
            
            # ALL ranks must synchronize here (including rank1 and rank3)
            # This ensures no rank proceeds to collective operations while others are still in recovery
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                logger.info(f"rank: {rank}, synchronized after Gemini recovery check")
            end_recovery_time = time()
            recovery_time = end_recovery_time - start_recovery_time
            logger.info(f"rank: {rank}, Gemini hardware failure recovery time: {recovery_time:.2f} seconds")
            # All ranks have loaded their data, return it directly
            if recovered_state_dict:
                logger.info(f"rank: {rank}, returning loaded state dict")
                return recovered_state_dict
        
        if input_args.use_gemini and input_args.use_gemini_software_failure:
            logger.info(f"rank: {rank}, using Gemini checkpoint recovery for software failure")
            start_recovery_time = time()
            recovered_state_dict = self._load_from_saved_checkpoint_file(sharded_state_dict, checkpoint_dir)
            end_recovery_time = time()
            recovery_time = end_recovery_time - start_recovery_time
            logger.info(f"rank: {rank}, Gemini software failure recovery time: {recovery_time:.2f} seconds")
            return recovered_state_dict
        
        # Normal Gemini checkpoint load (mutual exchange between paired ranks, and all rank recovery from peer replication)
        # if input_args.use_gemini:
        #     logger.info(f"rank: {rank}, using Gemini checkpointing (normal mode)")
        #     # Load directly from backup data and return the state_dict
        #     return self._load_gemini_checkpoint(sharded_state_dict, checkpoint_dir)
        
        if input_args.use_eccheck and (self._is_eccheck_checkpoint(checkpoint_dir) or rank == 2):
            logger.info(f"Detected EC-CHECK format checkpoint at {checkpoint_dir}")
            # Load P2P checkpoint data (for rank2 recovery, this prepares the buffer)
            mapped_file_own, mapped_file_partner = self._load_ecccheck_p2p_checkpoint(checkpoint_dir)
            
            # _load_eccheck_checkpoint will use recovered data if available (rank2)
            return self._load_eccheck_checkpoint(sharded_state_dict, checkpoint_dir)
        
        if input_args.use_eclatin and (self._is_eclatin_checkpoint(checkpoint_dir) or rank == 2):
            logger.info(f"Detected ECLATIN format checkpoint at {checkpoint_dir}")
            # Load P2P checkpoint data (for rank2 recovery, this prepares the buffer)
            mapped_file_own, mapped_file_partner = self._load_eclatin_block_checkpoint(checkpoint_dir)
            
            # _load_eclatin_checkpoint will use recovered data if available (rank2)
            return self._load_eclatin_checkpoint(sharded_state_dict, checkpoint_dir)
        
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
        logger.info(f"rank: {rank}, starting checkpoint.load_state_dict")
        
        fsr = _get_filesystem_reader(checkpoint_dir, cache_metadata=True)
        checkpoint.load_state_dict(
            pyt_state_dict,
            fsr,
            planner=MCoreLoadPlanner(
                shapes_validation_sharded_tensors=flexible_shape_sharded_tensors,
                allow_shape_mismatch_sharded_tensors=allow_shape_mismatch_sharded_tensors,
            ),
        )
        logger.info(f"rank: {rank}, finished checkpoint.load_state_dict")
        
        self.cached_global_metadata = (
            fsr.read_metadata()
        )  # no storage interaction thanks to caching
        logger.info(f"rank: {rank}, finished read_metadata")

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
            # _ensure_metadata_accessible is called by _get_filesystem_reader
            # so all ranks should now have access to .metadata
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
        # _ensure_metadata_accessible is called by _get_filesystem_reader
        # so all ranks should now have access to .metadata
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
