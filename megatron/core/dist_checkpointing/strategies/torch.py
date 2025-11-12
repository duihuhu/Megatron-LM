# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Strategies using PyTorch distributed.checkpoint as an underlying format. """
import io
import os
import pickle
import warnings
from collections import ChainMap, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
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
from .filesystem_async import FileSystemWriterAsync
from .resharding import (
    TensorReformulationMetadata,
    apply_nd_flattened_tensors_reformulation,
    is_nd_flattened_tensor,
    nd_flattened_tensor_reformulated_global_shape,
    restore_nd_flattened_tensors_formulation,
)
from .state_dict_saver import save_state_dict_async_finalize, save_state_dict_async_plan
from .state_dict_decomposer import DecomposedStateDict, TensorInfo
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


def register_default_torch_strategies():
    """Register default strategies related to PyT Distributed backend."""
    register_default_strategy(
        StrategyAction.LOAD_SHARDED, 'torch_dist', 1, TorchDistLoadShardedStrategy()
    )
    register_default_strategy(
        StrategyAction.SAVE_SHARDED, 'torch_dist', 1, TorchDistSaveShardedStrategy('torch_dist', 1)
    )


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
        
        # Initialize EC-CHECK if enabled
        self._eccheck_native = None
        self._init_eccheck_if_enabled()

    def _init_eccheck_if_enabled(self):
        """Initialize EC-CHECK C++ module if enabled and distributed environment is ready."""
        try:
            from megatron.training import get_args as input_args
            args = input_args()
            self.use_eccheck = True
            if not getattr(args, 'use_eccheck', False):
                return
                
            # Check if distributed environment is initialized
            if not torch.distributed.is_initialized():
                logger.warning("EC-CHECK: Distributed environment not initialized, skipping EC-CHECK initialization")
                return
                
            # Initialize EC-CHECK C++ module
            self._init_eccheck_native()
            
            # Start persistent buffer poller thread
            self._start_buffer_poller_thread()
            
        except Exception as e:
            logger.warning(f"EC-CHECK: Failed to initialize during strategy creation: {e}")
            self._eccheck_native = None

    def _init_eccheck_native(self):
        """Initialize EC-CHECK C++ native module."""
        eccheck_native = None
        try:
            # Direct import .so file without modifying sys.path or affecting other packages
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Find .so file
            import glob as _glob_module
            so_files = _glob_module.glob(os.path.join(current_dir, "eccheck_native*.so"))
            
            if not so_files:
                raise ImportError(f"No eccheck_native.so file found in {current_dir}")
            
            # Load .so file directly using importlib
            import importlib.util as _importlib_util
            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location("eccheck_native", so_path)
            eccheck_native = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(eccheck_native)
            logger.debug(f"EC-CHECK: Loaded .so file from {so_path}")
            
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            paired_rank = self._get_paired_rank(rank, world_size)
            
            # Create instance with error handling
            try:
                # IMPORTANT: This constructor call will BLOCK until:
                # 1. Send and recv threads are started
                # 2. Both NCCL communicators (0to1 and 1to0) are fully initialized
                # 3. All threads are ready for data exchange
                # Only after all initialization is complete will this call return.
                logger.info(f"EC-CHECK: Creating C++ native module (this will block until NCCL is initialized)...")
                print(f"EC-CHECK: [Rank {rank}] Creating C++ native module (blocking until NCCL initialization completes)...")
                
                self._eccheck_native = eccheck_native.ECCHECKNative(rank, world_size, paired_rank)
                
                # If we reach here, NCCL communicators are ready and threads are running
                logger.info(f"EC-CHECK: C++ native module initialized successfully (rank={rank}, world_size={world_size}, paired_rank={paired_rank})")
                print(f"EC-CHECK: [Rank {rank}] C++ native module initialized - NCCL communicators ready for data exchange")
                
                # Initialize EC-CHECK buffers
                self._init_eccheck_buffers()
        
            except Exception as e:
                logger.warning(f"EC-CHECK: Failed to create C++ native module instance: {e}")
                # Try to stop the pipeline if it was partially created
                try:
                    if hasattr(self, '_eccheck_native') and self._eccheck_native is not None:
                        self._eccheck_native.stop_pipeline()
                except:
                    pass
                self._eccheck_native = None
                raise e
            
        except ImportError as e:
            logger.warning(f"EC-CHECK: C++ native module not available: {e}, EC-CHECK functionality will not work")
            self._eccheck_native = None
        except Exception as e:
            logger.warning(f"EC-CHECK: Failed to initialize C++ native module: {e}, EC-CHECK functionality will not work")
            self._eccheck_native = None

    def _get_paired_rank(self, my_rank: int, world_size: int) -> int:
        """Get the paired rank for parity exchange."""
        if world_size % 2 != 0:
            raise ValueError(f"EC-CHECK: World size must be even for pairing, got {world_size}")
        
        half_size = world_size // 2
        
        if my_rank < half_size:
            # First half pairs with second half
            paired_rank = my_rank + half_size
        else:
            # Second half pairs with first half
            paired_rank = my_rank - half_size
        
        logger.debug(f"EC-CHECK: Rank {my_rank} paired with Rank {paired_rank}")
        return paired_rank

    def _init_eccheck_buffers(self):
        """Initialize EC-CHECK buffers during C++ module initialization.
        
        Note: Only allocates data and encoding buffers at initialization.
        Receive and parity buffers will be allocated by FileSystemWriterAsync
        after metadata exchange, when peer data sizes are known.
        """
        rank = torch.distributed.get_rank()
        logger.info("EC-CHECK: Initializing buffers for EC-CHECK (data and encoding only)")
        print(f"EC-CHECK: Initializing buffers for EC-CHECK (rank={rank}, data and encoding only)")
        
        # EC-CHECK configuration parameters
        self.eccheck_data_buffers_count = 12
        self.eccheck_encoding_buffers_count = 24  # data_count * m (12 * 2)
        self.eccheck_buffer_size = 64 * 1024 * 1024  # 64MB
        self.eccheck_pin_memory = True
        
        self.eccheck_preallocate_cpu_buffer = True  # Preallocate CPU buffer for tensor data
        self.eccheck_use_continuous_buffer = True  # Use continuous buffer for tensor data
        
        # Initialize state for EC-CHECK preparation
        self.decomposed_state_dict = None
        self.preallocated_cpu_buffer = None
        self.eccheck_serialized_metadata = None
        
        # Allocate data buffers for storing original tensor data
        self.eccheck_data_buffers = self._allocate_data_buffers()
        
        # Allocate encoding buffers for encoded packets
        self.eccheck_encoding_buffers = self._allocate_encoding_buffers()
        
        # Allocate receive buffers for peer encoded packets
        self.eccheck_recv_encoding_buffers = None
        
        # Allocate parity buffers for XOR computation results
        self.eccheck_parity_buffers = self._allocate_parity_buffers()
        
        # Initialize free buffer queues for Phase 3
        import queue
        self._free_data_buffer_queue = queue.Queue()
        for buffer in self.eccheck_data_buffers:
            self._free_data_buffer_queue.put(int(buffer.data_ptr()))
        
        self._free_encoding_buffer_queue = queue.Queue()
        for buffer in self.eccheck_encoding_buffers:
            self._free_encoding_buffer_queue.put(int(buffer.data_ptr()))
        
        self._free_parity_buffer_queue = queue.Queue()
        for buffer in self.eccheck_parity_buffers:
            self._free_parity_buffer_queue.put(int(buffer.data_ptr()))

        logger.info(f"EC-CHECK: Buffer initialization completed - "
                   f"Data buffers: {len(self.eccheck_data_buffers)}, "
                   f"Encoding buffers: {len(self.eccheck_encoding_buffers)}, "
                   f"Parity buffers: {len(self.eccheck_parity_buffers)}")
        print(f"EC-CHECK: Buffer initialization completed (rank={rank}) - "
              f"Data buffers: {len(self.eccheck_data_buffers)}, "
              f"Encoding buffers: {len(self.eccheck_encoding_buffers)}, "
              f"Parity buffers: {len(self.eccheck_parity_buffers)}")

    def _allocate_data_buffers(self):
        """Allocate data buffers for storing original tensor data."""
        logger.info(f"EC-CHECK: Allocating data buffers ({self.eccheck_data_buffers_count} buffers, {self.eccheck_buffer_size // (1024*1024)}MB each)")
        
        data_buffers = []
        for i in range(self.eccheck_data_buffers_count):
            buffer = torch.empty(self.eccheck_buffer_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
            data_buffers.append(buffer)
            logger.debug(f"EC-CHECK: Allocated data buffer {i}: {self.eccheck_buffer_size} bytes")
        
        logger.info(f"EC-CHECK: Allocated {len(data_buffers)} data buffers")
        return data_buffers

    def _allocate_encoding_buffers(self):
        """Allocate encoding buffers for encoded packets."""
        logger.info(f"EC-CHECK: Allocating encoding buffers ({self.eccheck_encoding_buffers_count} buffers, {self.eccheck_buffer_size // (1024*1024)}MB each)")
        
        encoding_buffers = []
        for i in range(self.eccheck_encoding_buffers_count):
            buffer = torch.empty(self.eccheck_buffer_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
            encoding_buffers.append(buffer)
            logger.debug(f"EC-CHECK: Allocated encoding buffer {i}: {self.eccheck_buffer_size} bytes")
        
        logger.info(f"EC-CHECK: Allocated {len(encoding_buffers)} encoding buffers")
        return encoding_buffers
 
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
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        paired_rank = self._get_paired_rank(rank, world_size)
        
        # Get peer's total data size from global registry
        peer_metadata = global_registry.rank_metadata.get(paired_rank, [])
        peer_total_size = sum(meta.size_bytes for meta in peer_metadata)
        
        # Align peer's data size to buffer_size (64MB)
        aligned_size = ((peer_total_size + self.eccheck_buffer_size - 1) // self.eccheck_buffer_size) * self.eccheck_buffer_size
        
        logger.info(
            f"EC-CHECK: Allocating TWO receive buffers based on peer data size\n"
            f"  Paired rank: {paired_rank}\n"
            f"  Peer data size: {peer_total_size / (1024**3):.2f} GB\n"
            f"  Aligned buffer size (per buffer): {aligned_size / (1024**3):.2f} GB\n"
            f"  Total receive memory: {2 * aligned_size / (1024**3):.2f} GB"
        )
        
        # # Allocate two large continuous buffers (one for each encoding thread)
        recv_buffer_thread1 = torch.empty(aligned_size, dtype=torch.uint8)
        recv_buffer_thread2 = torch.empty(aligned_size, dtype=torch.uint8)
        
        logger.info(
            f"EC-CHECK: Allocated TWO receive buffers: {aligned_size / (1024**3):.2f} GB each "
            f"({aligned_size / (1024**2):.0f} MB each)"
        )
        
        return (recv_buffer_thread1, recv_buffer_thread2)

    def _allocate_parity_buffers(self):
        """Allocate parity buffers for XOR computation results.
        
        Note: Parity buffer count should match encoding buffer count (24) to support
        pipelined operations where each data chunk needs 2 parity buffers (one per thread).
        """
        # Use encoding buffer count instead of data buffer count
        # Each data chunk needs 2 parity buffers (thread1 and thread2)
        parity_buffer_count = self.eccheck_encoding_buffers_count
        logger.info(f"EC-CHECK: Allocating parity buffers ({parity_buffer_count} buffers)")
        
        parity_buffers = []
        for i in range(parity_buffer_count):
            buffer = torch.empty(self.eccheck_buffer_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
            parity_buffers.append(buffer)
            logger.debug(f"EC-CHECK: Allocated parity buffer {i}: {self.eccheck_buffer_size} bytes")
        
        logger.info(f"EC-CHECK: Allocated {len(parity_buffers)} parity buffers")
        return parity_buffers

    def _poll_and_release_buffers(self):
        """Poll C++ for buffers ready to be released and put them back to queues."""
        if self._eccheck_native is None:
            return
        
        # Get data buffers ready for release
        data_buffers = self._eccheck_native.get_data_buffers_to_release()
        for data_addr in data_buffers:
            try:
                self._free_data_buffer_queue.put_nowait(data_addr)
                logger.debug(f"EC-CHECK: Released data buffer at address {data_addr}")
            except Exception:
                logger.error(f"EC-CHECK: Data buffer queue is full, cannot release buffer {data_addr}")
        
        # Get encoding buffers ready for release
        encoding_buffers = self._eccheck_native.get_encoding_buffers_to_release()
        for encoding_addr in encoding_buffers:
            try:
                self._free_encoding_buffer_queue.put_nowait(encoding_addr)
                logger.debug(f"EC-CHECK: Released encoding buffer at address {encoding_addr}")
            except Exception:
                logger.error(f"EC-CHECK: Encoding buffer queue is full, cannot release buffer {encoding_addr}")
        
        # Get parity buffers ready for release
        parity_buffers = self._eccheck_native.get_parity_buffers_to_release()
        for parity_addr in parity_buffers:
            try:
                self._free_parity_buffer_queue.put_nowait(parity_addr)
                logger.debug(f"EC-CHECK: Released parity buffer at address {parity_addr}")
            except Exception:
                logger.error(f"EC-CHECK: Parity buffer queue is full, cannot release buffer {parity_addr}")
    
    def _start_buffer_poller_thread(self):
        """Start a persistent background thread to poll and release buffers."""
        import threading
        
        if hasattr(self, '_buffer_poller_thread') and self._buffer_poller_thread is not None:
            logger.warning("EC-CHECK: Buffer poller thread already started")
            return
        
        # Create control events
        self._buffer_poller_stop_event = threading.Event()
        self._buffer_poller_active_event = threading.Event()
        
        def buffer_poller_worker():
            """Persistent background thread that polls for buffer releases."""
            logger.info("EC-CHECK: Buffer poller thread started")
            poll_count = 0
            
            while not self._buffer_poller_stop_event.is_set():
                # Only poll when active
                if self._buffer_poller_active_event.is_set():
                    self._poll_and_release_buffers()
                    poll_count += 1
                    if poll_count % 1000 == 0:
                        logger.debug(f"EC-CHECK: Buffer poller running (polled {poll_count} times)")
                
                # Sleep briefly to avoid busy waiting
                from time import sleep
                sleep(0.001)  # 1ms
            
            logger.info("EC-CHECK: Buffer poller thread stopping")
        
        # Start the daemon thread
        self._buffer_poller_thread = threading.Thread(target=buffer_poller_worker, daemon=True)
        self._buffer_poller_thread.start()
        logger.info("EC-CHECK: Buffer poller thread created and started")
    
    def _stop_buffer_poller_thread(self):
        """Stop the persistent buffer poller thread."""
        if not hasattr(self, '_buffer_poller_thread') or self._buffer_poller_thread is None:
            return
        
        logger.info("EC-CHECK: Stopping buffer poller thread...")
        
        # Signal the thread to stop
        if self._buffer_poller_stop_event:
            self._buffer_poller_stop_event.set()
        
        # Wait for thread to finish
        if self._buffer_poller_thread.is_alive():
            self._buffer_poller_thread.join(timeout=2.0)
            if self._buffer_poller_thread.is_alive():
                logger.warning("EC-CHECK: Buffer poller thread did not stop in time")
            else:
                logger.info("EC-CHECK: Buffer poller thread stopped successfully")
        
        self._buffer_poller_thread = None
        self._buffer_poller_stop_event = None
        self._buffer_poller_active_event = None

    def _get_eccheck_buffers(self):
        """Get EC-CHECK buffers for FileSystemWriterAsync.
        
        Note: Returns data, encoding, and parity buffers.
        Receive buffers will be allocated by FileSystemWriterAsync
        after metadata exchange.
        """
        if not hasattr(self, 'eccheck_data_buffers'):
            return None
        
        return {
            'data_buffers': self.eccheck_data_buffers,
            'encoding_buffers': self.eccheck_encoding_buffers,
            'parity_buffers': self.eccheck_parity_buffers,
            'free_data_buffer_queue': self._free_data_buffer_queue,
            'free_encoding_buffer_queue': self._free_encoding_buffer_queue,
            'free_parity_buffer_queue': self._free_parity_buffer_queue,
            # Pass buffer poller control objects
            'buffer_poller_active_event': self._buffer_poller_active_event,
            'poll_and_release_buffers': self._poll_and_release_buffers,
            # Note: recv_encoding_buffers will be allocated by FileSystemWriterAsync after metadata exchange
        }

    def __del__(self):
        """Cleanup EC-CHECK resources when strategy is destroyed."""
        try:
            if hasattr(self, '_eccheck_native') and self._eccheck_native is not None:
                # Stop the C++ pipeline
                self._eccheck_native.stop_pipeline()
                logger.info("EC-CHECK: C++ native module stopped in strategy destructor")
        except Exception as e:
            logger.warning(f"EC-CHECK: Error during strategy cleanup: {e}")

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
            use_eccheck=args.use_eccheck,
            eccheck_native=self._eccheck_native,  # Pass pre-initialized C++ module
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
        if self.use_eccheck:
            self._prepare_eccheck_data(self.cached_central_plan, planner)
            # Pass EC-CHECK state to writer if available
            writer.decomposed_state_dict = self.decomposed_state_dict
            writer.preallocated_cpu_buffer = self.preallocated_cpu_buffer
            writer.eccheck_serialized_metadata = self.eccheck_serialized_metadata
            writer.eccheck_global_registry = self.eccheck_global_registry
            # Pass the updated receive buffers (TWO large buffers allocated based on peer size)
            writer.eccheck_recv_encoding_buffers = self.eccheck_recv_encoding_buffers
            
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
                if self.eccheck_pin_memory and torch.cuda.is_available():
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
        
        total_time = time() - start_total
        logger.info(
            f"EC-CHECK: Preparation completed in {total_time:.2f}s\n"
            f"  Item processing: {process_time:.2f}s\n"
            f"  Preallocation: {prealloc_time:.2f}s\n"
            f"  Bucket prep: {bucket_time:.2f}s\n"
            f"  Metadata exchange: {metadata_time:.2f}s\n"
            f"  Buffer allocation: {buffer_alloc_time:.2f}s"
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
        if not self.use_eccheck:
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
