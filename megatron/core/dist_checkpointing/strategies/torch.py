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
import logging
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
from .ecnaive_manager import ECNAIVEManager
from .gemini_manager import GeminiManager
from .gemini_replicas_manager import GeminiReplicasManager
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


def filter_embeddings_from_sharded_state_dict(sharded_state_dict: ShardedStateDict) -> ShardedStateDict:
    """
    Filter out embedding layers from sharded state dict for redundancy backup.
    
    This function removes word_embeddings and position_embeddings from the state dict
    to reduce network traffic and storage when using redundancy backup strategies
    (Gemini/EC series). The embeddings are saved separately by rank 0.
    
    Args:
        sharded_state_dict: The sharded state dict to filter
        
    Returns:
        Filtered sharded state dict without embedding layers
    """
    try:
        from megatron.training import get_args
        args = get_args()
        save_embeddings_separately = getattr(args, 'save_embeddings_separately', False)
        
        if not save_embeddings_separately:
            return sharded_state_dict
        
        filtered = {}
        embedding_keys_filtered = []
        
        for key, value in sharded_state_dict.items():
            # Filter out word_embeddings and position_embeddings
            if 'embedding.word_embeddings.weight' in key or \
               'embedding.position_embeddings.weight' in key:
                embedding_keys_filtered.append(key)
                continue
            filtered[key] = value
        
        if embedding_keys_filtered:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            logger.info(
                f"Rank {rank}: Filtered {len(embedding_keys_filtered)} embedding keys for redundancy backup"
            )
            logger.debug(f"Rank {rank}: Filtered keys: {embedding_keys_filtered}")
        
        return filtered
    except Exception as e:
        # If any error occurs (e.g., get_args() fails), just return original state dict
        logger.warning(f"Failed to filter embeddings: {e}, returning original state dict")
        return sharded_state_dict


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
        
        # Initialize EC-NAIVE manager (singleton instance shared with Load strategy)
        self.ecnaive_manager = ECNAIVEManager()
        self.ecnaive_manager.init_ecnaive_if_enabled()
        
        # Initialize Gemini manager (singleton instance for replica-level data transfer)
        from .gemini_manager import GeminiManager
        self.gemini_manager = GeminiManager()
        self.gemini_manager.init_gemini_if_enabled()
        
        # Initialize Gemini Replicas manager (singleton instance for multi-replica data transfer)
        self.gemini_replicas_manager = GeminiReplicasManager()
        self.gemini_replicas_manager.init_gemini_replicas_if_enabled()
        
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
        self.eclatin_recv_buffers_layerwise = None  # 4 recv buffers for layerwise mode (continuous, allocated in strategy)
        
        # Initialize strategy-specific EC-NAIVE state
        self.ecnaive_preallocate_cpu_buffer = True  # Preallocate CPU buffer for tensor data
        self.ecnaive_use_continuous_buffer = True  # Use continuous buffer for tensor data
        # Note: decomposed_state_dict and preallocated_cpu_buffer are shared with ECCHECK/ECLATIN
        self.ecnaive_serialized_metadata = None
        self.ecnaive_global_registry = None
        self.ecnaive_blocks = None  # 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1)
        self.ec_write_buckets = []  # WriteBuckets for 4 blocks

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
    
    def _get_ecnaive_buffers(self):
        """Get EC-NAIVE buffers for FileSystemWriterAsync.
        
        Note: Returns data and parity buffers (pooled).
        The 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1) are allocated
        in _allocate_ecnaive_blocks after metadata exchange.
        """
        if not self.ecnaive_manager.use_ecnaive:
            return None
        return self.ecnaive_manager.get_ecnaive_buffers()

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
        
        # Filter embeddings for backup strategies (Gemini/EC)
        # This reduces network traffic and storage in redundancy backups
        pyt_state_dict = filter_embeddings_from_sharded_state_dict(pyt_state_dict)
        
        from megatron.training import get_args as input_args
        args = input_args()
        # Use PyT saving mechanism

        # Create FileSystemWriterAsync with EC-CHECK, ECLATIN, EC-NAIVE, Gemini, or Gemini Replicas parameters
        if self.eclatin_manager.use_eclatin:
            from megatron.training import get_args
            args = get_args()
            use_eclatin_layerwise = getattr(args, 'use_eclatin_layerwise', False)
            
            writer = FileSystemWriterAsync(
                checkpoint_dir,
                separation_hint=self.separation_hint,
                thread_count=self.thread_count,
                use_msc=MultiStorageClientFeature.is_enabled(),
                use_eclatin=self.eclatin_manager.use_eclatin,
                use_eclatin_layerwise=use_eclatin_layerwise,
                eclatin_native=self.eclatin_manager._eclatin_native,  # Pass pre-initialized C++ module
                eclatin_buffers=self._get_eclatin_buffers(),  # Pass pre-allocated buffers
            )
            # Pass layerwise recv buffers if available
            if use_eclatin_layerwise and self.eclatin_recv_buffers_layerwise is not None:
                writer.eclatin_recv_buffers_layerwise = self.eclatin_recv_buffers_layerwise
                
        elif self.ecnaive_manager.use_ecnaive:
            from megatron.training import get_args
            args = get_args()
            
            writer = FileSystemWriterAsync(
                checkpoint_dir,
                separation_hint=self.separation_hint,
                thread_count=self.thread_count,
                use_msc=MultiStorageClientFeature.is_enabled(),
                use_ecnaive=self.ecnaive_manager.use_ecnaive,
                ecnaive_native=self.ecnaive_manager._ecnaive_native,  # Pass pre-initialized C++ module
                ecnaive_buffers=self._get_ecnaive_buffers(),  # Pass pre-allocated buffers
            )
                
        elif self.gemini_replicas_manager.use_gemini_replicas and self.gemini_replicas_manager.use_gemini_replicas_optimized:
            writer = FileSystemWriterAsync(
                checkpoint_dir,
                separation_hint=self.separation_hint,
                thread_count=self.thread_count,
                use_msc=MultiStorageClientFeature.is_enabled(),
                use_gemini_replicas=self.gemini_replicas_manager.use_gemini_replicas,
                gemini_replicas_native=self.gemini_replicas_manager.get_native_module(),  # Pass pre-initialized C++ module
                gemini_replicas_num=self.gemini_replicas_manager.num_replicas,  # Pass number of replicas
                use_rdma=self.gemini_replicas_manager.use_rdma,  # Pass RDMA flag for buffer registration
            )
        elif self.gemini_manager.use_gemini and self.gemini_manager.use_gemini_optimized:
            writer = FileSystemWriterAsync(
                checkpoint_dir,
                separation_hint=self.separation_hint,
                thread_count=self.thread_count,
                use_msc=MultiStorageClientFeature.is_enabled(),
                use_gemini=self.gemini_manager.use_gemini,
                gemini_native=self.gemini_manager.get_native_module(),  # Pass pre-initialized C++ module
                use_rdma=self.gemini_manager.use_rdma,  # Pass RDMA flag for transport selection
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
                use_rdma=self.eccheck_manager.use_rdma,  # Pass RDMA flag for EC-CHECK transport
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
            
            # Pass layerwise recv buffers (allocated in _prepare_eclatin_data)
            if use_eclatin_layerwise and self.eclatin_recv_buffers_layerwise is not None:
                writer.eclatin_recv_buffers_layerwise = self.eclatin_recv_buffers_layerwise
                logger.info(
                    f"ECLATIN: Passed layerwise recv buffers to writer "
                    f"(size: {self.eclatin_recv_buffers_layerwise[0].numel() / (1024**2):.0f} MB each)"
                )
            
            # In ECLATIN mode, call prepare_write_data to create write_buckets
            # It will use the metadata we just prepared
            writer.prepare_write_data(self.cached_central_plan, planner)
        # EC-NAIVE mode: decompose state_dict and preallocate CPU memory
        elif self.ecnaive_manager.use_ecnaive:
            self._prepare_ecnaive_data(self.cached_central_plan, planner)
            # Pass EC-NAIVE state to writer if available
            writer.decomposed_state_dict = self.decomposed_state_dict
            writer.preallocated_cpu_buffer = self.preallocated_cpu_buffer
            writer.ecnaive_serialized_metadata = self.ecnaive_serialized_metadata
            writer.ecnaive_global_registry = self.ecnaive_global_registry
            # Pass the 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1)
            writer.ecnaive_blocks = self.ecnaive_blocks
            writer.ec_write_buckets = self.ec_write_buckets
            
            # In EC-NAIVE mode, call prepare_write_data to create write_buckets
            # It will use the metadata we just prepared
            writer.prepare_write_data(self.cached_central_plan, planner)
        # Gemini Replicas mode: decompose state_dict and preallocate CPU memory for multi-replica exchange
        elif self.gemini_replicas_manager.use_gemini_replicas and self.gemini_replicas_manager.use_gemini_replicas_optimized:
            self._prepare_gemini_replicas_data(self.cached_central_plan, planner)
            # Pass Gemini Replicas state to writer if available
            writer.decomposed_state_dict = self.decomposed_state_dict
            writer.preallocated_cpu_buffer = self.preallocated_cpu_buffer
            # Pass preallocated remote buffers (optimization: only allocate once)
            if hasattr(self, 'gemini_replicas_remote_buffers'):
                writer.gemini_replicas_remote_buffers = self.gemini_replicas_remote_buffers
                writer.gemini_replicas_remote_buffer_sizes = self.gemini_replicas_remote_buffer_sizes
            
            # In Gemini Replicas mode, call prepare_write_data to create write_buckets
            # It will use the decomposed state_dict we just prepared
            writer.prepare_write_data(self.cached_central_plan, planner)
        # Gemini mode: decompose state_dict and preallocate CPU memory for replica exchange
        elif self.gemini_manager.use_gemini and self.gemini_manager.use_gemini_optimized:
            self._prepare_gemini_data(self.cached_central_plan, planner)
            # Pass Gemini state to writer if available
            writer.decomposed_state_dict = self.decomposed_state_dict
            writer.preallocated_cpu_buffer = self.preallocated_cpu_buffer
            # Pass preallocated remote buffers (optimization: only allocate once)
            if hasattr(self, 'gemini_remote_buffer'):
                writer.gemini_remote_buffer = self.gemini_remote_buffer
                writer.gemini_remote_buffer_size = self.gemini_remote_buffer_size
            # Pass cached pair process group (optimization: avoid repeated lookups)
            if hasattr(self, 'gemini_pair_group'):
                writer.gemini_pair_group = self.gemini_pair_group
            
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
        
        send_buffer_needs_registration = False
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
            send_buffer_needs_registration = True
        else:
            logger.info(
                f"Gemini: [Rank {rank}] Reusing existing preallocated CPU buffer: "
                f"{self.preallocated_cpu_buffer.numel() / (1024**3):.2f} GB"
            )
        
        # Register send buffer for RDMA if enabled (on first allocation)
        if self.gemini_manager.use_rdma and send_buffer_needs_registration:
            logger.info(f"Gemini: [Rank {rank}] Registering preallocated_cpu_buffer (send buffer) for RDMA")
            self.gemini_manager.register_buffer(self.preallocated_cpu_buffer)
        
        # Step 3: Exchange buffer sizes and preallocate remote buffers (receive buffers)
        # This optimization moves buffer allocation from _gemini_preload_to_continuous_buffer
        # to here, so it only happens once in the first iteration
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        if world_size >= 4:  # Gemini requires at least 4 ranks
            # Get paired rank
            pairing_map = {0: 2, 2: 0, 1: 3, 3: 1}
            paired_rank = pairing_map.get(rank, None)
            
            if paired_rank is not None:
                from ..strategies.async_utils import get_or_create_pair_process_group
                # Cache pair_group for reuse (optimization: avoid repeated lookups)
                if not hasattr(self, 'gemini_pair_group') or self.gemini_pair_group is None:
                    self.gemini_pair_group = get_or_create_pair_process_group(rank, paired_rank)
                    logger.info(f"Gemini: [Rank {rank}] Created and cached pair process group")
                else:
                    logger.debug(f"Gemini: [Rank {rank}] Reusing cached pair process group")
                
                pair_group = self.gemini_pair_group
                
                # Exchange buffer sizes (local_buffer_size)
                # Note: metadata size will be exchanged later in _gemini_preload_to_continuous_buffer
                # because metadata is generated there
                local_buffer_size = total_tensor_size
                size_tensor = torch.tensor([local_buffer_size], dtype=torch.long, device='cpu')
                gathered_sizes = [torch.zeros_like(size_tensor) for _ in range(2)]
                torch.distributed.all_gather(gathered_sizes, size_tensor, group=pair_group)
                
                pair_ranks = [min(rank, paired_rank), max(rank, paired_rank)]
                my_idx = pair_ranks.index(rank)
                paired_idx = 1 - my_idx
                remote_buffer_size = gathered_sizes[paired_idx][0].item()
                
                logger.info(
                    f"Gemini: [Rank {rank}] Exchanged buffer sizes with rank {paired_rank}: "
                    f"local={local_buffer_size / (1024**2):.2f} MB, "
                    f"remote={remote_buffer_size / (1024**2):.2f} MB"
                )
                
                # Allocate remote buffer if needed (or reuse existing)
                recv_buffer_needs_registration = False
                if not hasattr(self, 'gemini_remote_buffer') or self.gemini_remote_buffer is None or \
                   self.gemini_remote_buffer.numel() < remote_buffer_size:
                    self.gemini_remote_buffer = torch.empty(remote_buffer_size, dtype=torch.uint8, device='cpu')
                    logger.info(
                        f"Gemini: [Rank {rank}] Allocated remote buffer: "
                        f"{remote_buffer_size / (1024**2):.2f} MB"
                    )
                    recv_buffer_needs_registration = True
                else:
                    logger.info(
                        f"Gemini: [Rank {rank}] Reusing existing remote buffer: "
                        f"{self.gemini_remote_buffer.numel() / (1024**2):.2f} MB"
                    )
                
                # Store remote buffer size for later use
                self.gemini_remote_buffer_size = remote_buffer_size
                
                # Register recv buffer for RDMA if enabled (on first allocation)
                if self.gemini_manager.use_rdma and recv_buffer_needs_registration:
                    logger.info(f"Gemini: [Rank {rank}] Registering gemini_remote_buffer (recv buffer) for RDMA")
                    self.gemini_manager.register_buffer(self.gemini_remote_buffer)
            else:
                logger.warning(f"Gemini: [Rank {rank}] No paired rank found for buffer exchange")
        
        total_time = time() - start_total
        logger.info(
            f"Gemini: [Rank {rank}] Preparation completed in {total_time:.2f}s"
        )
    
    def _prepare_gemini_replicas_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        Gemini Replicas preparation: organize data for multi-replica transfer.
        
        This method performs the following steps:
        1. Process plan items (separate bytes and tensors)
        2. Create DecomposedStateDict for efficient GPU-to-CPU transfer
        3. Preallocate CPU memory buffer for tensors
        
        Similar to Gemini but supports multiple replicas with round-robin placement.
        
        Args:
            plan (SavePlan): save plan from PyTorch distributed checkpoint
            planner (SavePlanner): save planner to resolve data
        """
        from torch.distributed.checkpoint.filesystem import _StoragePrefix
        
        start_total = time()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        num_replicas = self.gemini_replicas_manager.num_replicas
        logger.info(
            f"Gemini Replicas: [Rank {rank}] Starting multi-replica checkpoint preparation "
            f"({num_replicas} replicas)"
        )
        
        # Step 1: Process plan items
        storage_plan: _StoragePrefix = plan.storage_data
        
        # Separate items into BYTE_IO (non-tensor) and TENSOR
        non_tensor_data = {}
        tensor_infos = []
        tensor_data_list = []
        
        logger.info(f"Gemini Replicas: [Rank {rank}] Processing {len(plan.items)} items from SavePlan")
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
            f"Gemini Replicas: [Rank {rank}] Processed {byte_io_count} BytesIO items, {tensor_count} tensor items"
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
            f"Gemini Replicas: [Rank {rank}] Created DecomposedStateDict:\n"
            f"  Non-tensor data: {stats['non_tensor_size_bytes'] / 1024:.2f} KB\n"
            f"  Tensor data: {stats['tensor_data_size_bytes'] / (1024**3):.2f} GB\n"
            f"  Total tensors: {stats['num_tensors']}"
        )
        
        # Step 2: Preallocate CPU buffer (send buffer)
        total_tensor_size = self.decomposed_state_dict.total_tensor_size_bytes
        
        send_buffer_needs_registration = False
        if self.preallocated_cpu_buffer is None or self.preallocated_cpu_buffer.numel() < total_tensor_size:
            logger.info(
                f"Gemini Replicas: [Rank {rank}] Allocating preallocated CPU buffer: "
                f"{total_tensor_size / (1024**3):.2f} GB"
            )
            
            # Use pinned memory for faster GPU-to-CPU transfer
            if torch.cuda.is_available():
                self.preallocated_cpu_buffer = torch.empty(
                    total_tensor_size, dtype=torch.uint8
                ).pin_memory()
                logger.info(f"Gemini Replicas: [Rank {rank}] Allocated pinned memory buffer")
            else:
                self.preallocated_cpu_buffer = torch.empty(
                    total_tensor_size, dtype=torch.uint8
                )
                logger.info(f"Gemini Replicas: [Rank {rank}] Allocated regular CPU buffer")
            send_buffer_needs_registration = True
        else:
            logger.info(
                f"Gemini Replicas: [Rank {rank}] Reusing existing preallocated CPU buffer: "
                f"{self.preallocated_cpu_buffer.numel() / (1024**3):.2f} GB"
            )
        
        # Register send buffer for RDMA if enabled (on first allocation)
        if self.gemini_replicas_manager.use_rdma and send_buffer_needs_registration:
            logger.info(f"Gemini Replicas: [Rank {rank}] Registering preallocated_cpu_buffer (send buffer) for RDMA")
            self.gemini_replicas_manager.register_buffer(self.preallocated_cpu_buffer)
        
        # Step 3: Exchange buffer sizes and preallocate remote buffers (receive buffers)
        # This optimization moves buffer allocation from _gemini_replicas_preload_to_continuous_buffer
        # to here, so it only happens once in the first iteration
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        if world_size > 1:
            # Calculate source ranks (ranks that will send data to this rank)
            source_ranks = []
            for src_rank in range(world_size):
                if src_rank == rank:
                    continue
                src_targets = self.gemini_replicas_manager._calculate_target_ranks(src_rank, world_size)
                if rank in src_targets:
                    source_ranks.append(src_rank)
            
            if len(source_ranks) > 0:
                # Create or get global gloo group for CPU tensor communication
                from ..strategies.async_utils import get_or_create_global_gloo_group
                global_gloo_group = get_or_create_global_gloo_group()
                
                # Exchange buffer sizes using all_gather
                local_buffer_size = total_tensor_size
                size_tensor = torch.tensor([local_buffer_size], dtype=torch.long, device='cpu')
                all_sizes = [torch.zeros_like(size_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(all_sizes, size_tensor, group=global_gloo_group)
                
                # Extract sizes for all ranks
                rank_sizes = {r: all_sizes[r][0].item() for r in range(world_size)}
                logger.info(
                    f"Gemini Replicas: [Rank {rank}] Exchanged buffer sizes with all ranks: "
                    f"local={local_buffer_size / (1024**2):.2f} MB, "
                    f"sources={source_ranks}"
                )
                
                # Initialize remote buffers dict if not exists
                if not hasattr(self, 'gemini_replicas_remote_buffers'):
                    self.gemini_replicas_remote_buffers = {}
                if not hasattr(self, 'gemini_replicas_remote_buffer_sizes'):
                    self.gemini_replicas_remote_buffer_sizes = {}
                
                # Allocate remote buffer for each source rank
                for src_rank in source_ranks:
                    remote_buffer_size = rank_sizes[src_rank]
                    
                    # Allocate remote buffer if needed (or reuse existing)
                    recv_buffer_needs_registration = False
                    if src_rank not in self.gemini_replicas_remote_buffers or \
                       self.gemini_replicas_remote_buffers[src_rank] is None or \
                       self.gemini_replicas_remote_buffers[src_rank].numel() < remote_buffer_size:
                        self.gemini_replicas_remote_buffers[src_rank] = torch.empty(
                            remote_buffer_size, dtype=torch.uint8, device='cpu'
                        )
                        logger.info(
                            f"Gemini Replicas: [Rank {rank}] Allocated remote buffer for source rank {src_rank}: "
                            f"{remote_buffer_size / (1024**2):.2f} MB"
                        )
                        recv_buffer_needs_registration = True
                    else:
                        logger.info(
                            f"Gemini Replicas: [Rank {rank}] Reusing existing remote buffer for source rank {src_rank}: "
                            f"{self.gemini_replicas_remote_buffers[src_rank].numel() / (1024**2):.2f} MB"
                        )
                    
                    # Store remote buffer size for later use
                    self.gemini_replicas_remote_buffer_sizes[src_rank] = remote_buffer_size
                    
                    # Register recv buffer for RDMA if enabled (on first allocation)
                    if self.gemini_replicas_manager.use_rdma and recv_buffer_needs_registration:
                        logger.info(
                            f"Gemini Replicas: [Rank {rank}] Registering remote buffer for source rank {src_rank} "
                            f"(recv buffer) for RDMA"
                        )
                        self.gemini_replicas_manager.register_buffer(self.gemini_replicas_remote_buffers[src_rank])
        
        total_time = time() - start_total
        logger.info(
            f"Gemini Replicas: [Rank {rank}] Preparation completed in {total_time:.2f}s"
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
        
        # Step 7: For layerwise mode, allocate recv buffers based on layer sizes
        from megatron.training import get_args
        args = get_args()
        use_eclatin_layerwise = getattr(args, 'use_eclatin_layerwise', False)
        
        recv_buffer_alloc_time = 0
        if use_eclatin_layerwise:
            start = time()
            if self.eclatin_recv_buffers_layerwise is None:
                self.eclatin_recv_buffers_layerwise = self._allocate_eclatin_layerwise_recv_buffers(
                    self.eclatin_global_registry
                )
            recv_buffer_alloc_time = time() - start
            logger.info(f"ECLATIN: Layerwise recv buffer allocation completed in {recv_buffer_alloc_time:.2f}s")
        
        total_time = time() - start_total
        logger.info(
            f"ECLATIN: Preparation completed in {total_time:.2f}s\n"
            f"  Item processing: {process_time:.2f}s\n"
            f"  Preallocation: {prealloc_time:.2f}s\n"
            f"  Bucket prep: {bucket_time:.2f}s\n"
            f"  Metadata exchange: {metadata_time:.2f}s\n"
            f"  Block allocation: {block_alloc_time:.2f}s"
            + (f"\n  Recv buffer allocation: {recv_buffer_alloc_time:.2f}s" if use_eclatin_layerwise else "")
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

    def _prepare_ecnaive_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        EC-NAIVE preparation: organize data for serialization-free checkpointing.
        
        This method performs the following steps:
        1. Process plan items like normal mode (separate bytes and tensors)
        2. Organize tensors for EC-NAIVE (extract metadata and data)
        3. Preallocate CPU memory buffer for tensors
        4. Prepare write buckets for async transfer
        5. Broadcast and exchange metadata
        6. Allocate 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1)
        
        Args:
            plan (SavePlan): save plan from PyTorch distributed checkpoint
            planner (SavePlanner): save planner to resolve data
        """
        from torch.distributed.checkpoint.filesystem import _StoragePrefix
        from time import time
        
        start_total = time()
        logger.info("EC-NAIVE: Starting serialization-free checkpoint preparation")
        
        # Step 1: Process plan items (similar to ECLATIN)
        start = time()
        storage_plan: _StoragePrefix = plan.storage_data
        
        # Separate items into BYTE_IO (non-tensor) and TENSOR
        non_tensor_data = {}
        tensor_infos = []
        tensor_data_list = []
        
        logger.info(f"EC-NAIVE: Processing {len(plan.items)} items from SavePlan")
        byte_io_count = 0
        tensor_count = 0
        none_data_count = 0
        
        for item in plan.items:
            data = planner.resolve_data(item)
            
            # Debug: check for None data
            if data is None:
                none_data_count += 1
                if none_data_count <= 5:
                    logger.warning(f"EC-NAIVE SAVE: Found None data for item: fqn={item.index.fqn}, type={item.type}")
                continue  # Skip None data items
            
            if item.type == WriteItemType.BYTE_IO:
                # Non-tensor data (e.g., extra_state)
                import io
                if isinstance(data, io.BytesIO):
                    non_tensor_data[item.index.fqn] = {
                        '_ecnaive_type': 'BytesIO',
                        '_ecnaive_data': data.getvalue()
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
            f"EC-NAIVE: Processed {byte_io_count} BytesIO items, {tensor_count} tensor items"
            + (f", skipped {none_data_count} None items" if none_data_count > 0 else "")
        )
        
        # Calculate offsets for tensor data
        offset = 0
        for info in tensor_infos:
            info.offset = offset
            offset += info.size_bytes
        
        # Create decomposed structure (reuse from ECCHECK/ECLATIN if available, otherwise create new)
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
            f"EC-NAIVE: Processed plan items in {process_time:.2f}s\n"
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
        if self.ecnaive_preallocate_cpu_buffer:
            start = time()
            total_size = self.decomposed_state_dict.total_tensor_size_bytes
            logger.info(f"EC-NAIVE: Preallocating CPU buffer of {total_size / (1024**3):.2f} GB")
            
            if self.preallocated_cpu_buffer is None:
                if self.ecnaive_manager.ecnaive_pin_memory and torch.cuda.is_available():
                    self.preallocated_cpu_buffer = torch.empty(
                        total_size, dtype=torch.uint8).pin_memory()
                    logger.info("EC-NAIVE: Using pinned memory for CPU buffer")
                else:
                    self.preallocated_cpu_buffer = torch.empty(
                        total_size, dtype=torch.uint8
                    )
                    logger.info("EC-NAIVE: Using non-pinned memory for CPU buffer")
            
            prealloc_time = time() - start
            logger.debug(f"EC-NAIVE: CPU buffer preallocation took {prealloc_time:.2f}s")
        else:
            prealloc_time = 0
        
        # Step 3: Prepare write buckets for async transfer
        # Note: WriteBuckets for 4 blocks will be created in _allocate_ecnaive_blocks
        # This step is a placeholder for consistency with ECLATIN flow
        start = time()
        bucket_time = time() - start
        logger.debug(f"EC-NAIVE: Write bucket preparation (will be done in block allocation)")
        
        # Step 4: Validate decomposition
        if not self.validate_ecnaive_decomposition():
            raise RuntimeError("EC-NAIVE: Decomposition validation failed")
        
        # Step 5: Broadcast and exchange metadata (reuse ECCHECK method)
        start = time()
        self.ecnaive_global_registry = self._broadcast_and_exchange_metadata()
        metadata_time = time() - start
        logger.info(f"EC-NAIVE: Metadata exchange completed in {metadata_time:.2f}s")
        
        # Step 6: Allocate 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1)
        start = time()
        if self.ecnaive_blocks is None:
            self.ecnaive_blocks = self._allocate_ecnaive_blocks(self.ecnaive_global_registry)
        block_alloc_time = time() - start
        logger.info(f"EC-NAIVE: Block allocation completed in {block_alloc_time:.2f}s")
        
        total_time = time() - start_total
        logger.info(
            f"EC-NAIVE: Preparation completed in {total_time:.2f}s\n"
            f"  Item processing: {process_time:.2f}s\n"
            f"  Preallocation: {prealloc_time:.2f}s\n"
            f"  Bucket prep: {bucket_time:.2f}s\n"
            f"  Metadata exchange: {metadata_time:.2f}s\n"
            f"  Block allocation: {block_alloc_time:.2f}s"
        )

    def validate_ecnaive_decomposition(self) -> bool:
        """
        Validate EC-NAIVE decomposition structure.
        
        Validates:
        1. non_tensor_data is a dict
        2. tensor_infos is a list (tensor keys)
        3. tensor_data is a list of tensors
        4. Counts match between tensor_infos and tensor_data
        
        Returns:
            bool: True if decomposition is valid, False otherwise
        """
        if not self.ecnaive_manager.use_ecnaive:
            logger.warning("EC-NAIVE: Validation skipped - EC-NAIVE is not enabled")
            return False
        
        if not self.decomposed_state_dict:
            logger.error("EC-NAIVE: Validation failed - State dict not decomposed yet")
            return False
        
        decomposed = self.decomposed_state_dict
        
        # Check 1: Non-tensor key-value pairs (dict)
        if not isinstance(decomposed.non_tensor_data, dict):
            logger.error(
                f"EC-NAIVE: Component 1 failed - non_tensor_data should be dict, "
                f"got {type(decomposed.non_tensor_data).__name__}"
            )
            return False
        
        # Check 2: Tensor keys (list)
        if not isinstance(decomposed.tensor_infos, list):
            logger.error(
                f"EC-NAIVE: Component 2 failed - tensor_infos should be list, "
                f"got {type(decomposed.tensor_infos).__name__}"
            )
            return False
        
        # Check 3: Tensor data (list)
        if not isinstance(decomposed.tensor_data, list):
            logger.error(
                f"EC-NAIVE: Component 3 failed - tensor_data should be list, "
                f"got {type(decomposed.tensor_data).__name__}"
            )
            return False
        
        # Check 4: Counts match
        if len(decomposed.tensor_infos) != len(decomposed.tensor_data):
            logger.error(
                f"EC-NAIVE: Component count mismatch - tensor_infos has {len(decomposed.tensor_infos)} items, "
                f"tensor_data has {len(decomposed.tensor_data)} items"
            )
            return False
        
        # Check 5: Total size matches
        calculated_size = sum(info.size_bytes for info in decomposed.tensor_infos)
        if calculated_size != decomposed.total_tensor_size_bytes:
            logger.warning(
                f"EC-NAIVE: Size mismatch - calculated {calculated_size} bytes, "
                f"but total_tensor_size_bytes is {decomposed.total_tensor_size_bytes} bytes"
            )
            # This is a warning, not an error, as it might be due to rounding
        
        logger.debug("EC-NAIVE: Decomposition validation passed")
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
        
        # ===== Register buffers for RDMA if enabled =====
        if self.eclatin_manager.use_rdma:
            logger.info("ECLATIN: Registering 4 persistent blocks for RDMA...")
            self.eclatin_manager.register_buffer(data_block_1)
            self.eclatin_manager.register_buffer(data_block_2)
            self.eclatin_manager.register_buffer(parity_block_1)
            self.eclatin_manager.register_buffer(parity_block_2)
            logger.info("ECLATIN: RDMA buffer registration complete")
        
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
   

    def _allocate_ecnaive_blocks(self, global_registry):
        """
        Allocate 4 persistent blocks for EC-NAIVE:
        - data0: Local data block (kept, not sent)
        - recv_parity1: Receive p_{(i+1),1} from rank (i+1)
        - recv_parity0: Receive p_{(i+2),0} from rank (i+2)
        - recv_data1: Receive d_{(i+3),1} from rank (i+3)
        
        All blocks are aligned to the maximum size across all ranks for pipeline synchronization.
        This ensures all ranks use the same block sizes.
        
        Args:
            global_registry: GlobalMetadataRegistry from all ranks
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'data0', 'recv_parity1', 
                                    'recv_parity0', 'recv_data1'
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
        ecnaive_buffer_size = self.ecnaive_manager.ecnaive_buffer_size
        # Each block only needs half of max_total_bytes (data is split into two halves)
        half_max_total_bytes = max_total_bytes // 2
        aligned_half_block_size = ((half_max_total_bytes + ecnaive_buffer_size - 1) // ecnaive_buffer_size) * ecnaive_buffer_size
        
        logger.info(
            f"EC-NAIVE: Allocating 4 persistent blocks based on metadata\n"
            f"  Own data size: {own_total_size / (1024**3):.2f} GB (actual), "
            f"{max_total_bytes / (1024**3):.2f} GB (pipeline max), "
            f"{aligned_half_block_size / (1024**3):.2f} GB (aligned half block size)\n"
            f"  All blocks will use aligned half size: {aligned_half_block_size / (1024**3):.2f} GB "
            f"({aligned_half_block_size / (1024**2):.0f} MB)"
        )
        
        # ===== Allocate 4 large continuous buffers =====
        # All blocks use the same aligned half size (each block stores half of the data)
        data0 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        recv_parity1 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        recv_parity0 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        recv_data1 = torch.empty(aligned_half_block_size, dtype=torch.uint8)
        
        logger.info(
            f"EC-NAIVE: Allocated 4 persistent blocks:\n"
            f"  data0: {aligned_half_block_size / (1024**3):.2f} GB "
            f"({aligned_half_block_size / (1024**2):.0f} MB)\n"
            f"  recv_parity1: {aligned_half_block_size / (1024**3):.2f} GB "
            f"({aligned_half_block_size / (1024**2):.0f} MB)\n"
            f"  recv_parity0: {aligned_half_block_size / (1024**3):.2f} GB "
            f"({aligned_half_block_size / (1024**2):.0f} MB)\n"
            f"  recv_data1: {aligned_half_block_size / (1024**3):.2f} GB "
            f"({aligned_half_block_size / (1024**2):.0f} MB)\n"
            f"  Total memory: {4 * aligned_half_block_size / (1024**3):.2f} GB"
        )
        
        # ===== Package blocks with metadata =====
        # Align with EC-CHECK/ECLATIN: use decomposed_state_dict.non_tensor_data directly
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
        # Similar to ECCHECK/ECLATIN's P2P buffers, create WriteBuckets for each block
        from pathlib import Path
        
        # Get checkpoint_dir
        checkpoint_dir = getattr(self, 'current_checkpoint_dir', None)
        if checkpoint_dir is None:
            logger.warning("EC-NAIVE: checkpoint_dir not available, using file_name as path")
            checkpoint_dir = Path(".")
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        # Create WriteBuckets for 4 blocks
        # Format: (file_path, storage_key, (bytes_data, tensor_data))
        block_names = ['data0', 'recv_parity1', 'recv_parity0', 'recv_data1']
        block_tensors = [data0, recv_parity1, recv_parity0, recv_data1]
        
        for block_name, block_tensor in zip(block_names, block_tensors):
            # Create ecnaive_bytes_data format (reuse ECLATIN format for compatibility)
            block_ecnaive_bytes_data = [
                ('ecnaive_metadata', block_serialized_metadata),
                ('ecnaive_continuous_buffer', block_tensor),
            ]
            
            # Generate file name
            file_name = f'__{rank}_{block_name}.distcp'
            file_path = checkpoint_dir / file_name
            
            # Create WriteBucket
            write_bucket = (
                file_path,              # file_path (full path with checkpoint_dir)
                file_name,              # storage_key (used in metadata)
                (block_ecnaive_bytes_data, []),  # (bytes_data, tensor_data)
            )
            
            self.ec_write_buckets.append(write_bucket)
        
        # Package blocks into dictionary
        blocks = {
            'data0': data0,
            'recv_parity1': recv_parity1,
            'recv_parity0': recv_parity0,
            'recv_data1': recv_data1,
            'metadata': block_serialized_metadata,
            'actual_size': own_actual_size,
            'pipeline_size': block_pipeline_total_bytes,
            'aligned_size': aligned_half_block_size,
        }
        
        logger.info(
            f"EC-NAIVE: Packaged 4 blocks with metadata and WriteBuckets:\n"
            f"  Metadata: {own_non_tensor_size / 1024:.2f} KB (non-tensor) + "
            f"{own_tensor_keys_size / 1024:.2f} KB (tensor keys), "
            f"{own_tensor_buffer_size / (1024**3):.2f} GB (buffer actual size)\n"
            f"  Pipeline size: {block_pipeline_total_bytes / (1024**3):.2f} GB\n"
            f"  Aligned half block size: {aligned_half_block_size / (1024**3):.2f} GB\n"
            f"  Created {len(block_names)} WriteBuckets"
        )
        
        return blocks
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
        
        # ===== Register buffers for RDMA if enabled =====
        if self.eclatin_manager.use_rdma:
            logger.info("ECLATIN: Registering 4 persistent blocks for RDMA...")
            self.eclatin_manager.register_buffer(data_block_1)
            self.eclatin_manager.register_buffer(data_block_2)
            self.eclatin_manager.register_buffer(parity_block_1)
            self.eclatin_manager.register_buffer(parity_block_2)
            logger.info("ECLATIN: RDMA buffer registration complete")
        
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
    
    def _allocate_eclatin_layerwise_recv_buffers(self, global_registry):
        """
        Allocate 4 large continuous recv buffers for ECLATIN layerwise mode.
        
        Similar to EC-CHECK's recv encoding buffers, but for ECLATIN's 4 recv buffers:
        - recv1_parity1: For receiving data for parity1 XOR encoding
        - recv2_parity1: For receiving data for parity1 XOR encoding
        - recv1_parity2: For receiving data for parity2 XOR encoding
        - recv2_parity2: For receiving data for parity2 XOR encoding
        
        Each buffer size = global maximum of sum of all layers' half_aligned sizes (aligned to buffer_size).
        All ranks use the same buffer size for pipeline synchronization.
        
        Args:
            global_registry: GlobalMetadataRegistry from all ranks
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            4 recv buffers (recv1_parity1, recv2_parity1, recv1_parity2, recv2_parity2)
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # Step 1: Organize tensors by layer from tensor_infos
        # Extract layer groups from tensor_infos (using same logic as _extract_layer_groups)
        layer_groups = {}
        
        def extract_layer_number(fqn: str) -> int:
            """Extract layer number from FQN - using same patterns as _extract_layer_groups.
            
            Supports patterns:
            - decoder.layers.N.
            - encoder.layers.N.
            - transformer.layers.N.
            - model.layers.N.
            - layers.N.
            - .layer.N., _layers_N_, .blocks.N., etc.
            """
            import re
            # Use the same patterns as _extract_layer_groups for consistency
            patterns = [
                r'\.layers\.(\d+)\.',      # .layers.N. (matches decoder.layers.0., module.decoder.layers.0., etc.)
                r'^layers\.(\d+)\.',       # layers.N. at start
                r'\.layer\.(\d+)\.',       # .layer.N.
                r'^layer\.(\d+)\.',        # layer.N. at start
                r'_layers_(\d+)_',         # _layers_N_
                r'_layer_(\d+)_',          # _layer_N_
                r'\.blocks\.(\d+)\.',      # .blocks.N.
                r'^blocks\.(\d+)\.',       # blocks.N. at start
                r'_blocks_(\d+)_',         # _blocks_N_
            ]
            for pattern in patterns:
                match = re.search(pattern, fqn)
                if match:
                    return int(match.group(1))
            return -1  # Non-layer tensor
        
        # Add logging to see actual FQN formats (INFO level for troubleshooting)
        sample_keys = [info.key for info in self.decomposed_state_dict.tensor_infos[:5]]
        logger.info(f"ECLATIN: Sample tensor keys for layer extraction: {sample_keys}")
        
        # Strategy: Since FQN doesn't contain layer number (e.g., "decoder.layers.xxx"),
        # we need to infer layer number from tensor order and FQN patterns.
        # For ShardedTensors, same FQN appears multiple times for different layers.
        # First pass: count occurrences of each FQN pattern (for inference)
        import re
        fqn_to_occurrences = {}
        for tensor_info in self.decomposed_state_dict.tensor_infos:
            fqn = tensor_info.key
            # Normalize to base pattern (remove layer number if present)
            base_fqn = fqn
            if re.search(r'\.layers\.\d+\.', fqn):
                base_fqn = re.sub(r'\.layers\.\d+\.', '.layers.', fqn)
            elif re.search(r'^layers\.\d+\.', fqn):
                base_fqn = re.sub(r'^layers\.\d+\.', 'layers.', fqn)
            # If FQN contains .layers. but no number, use as-is
            
            fqn_to_occurrences[base_fqn] = fqn_to_occurrences.get(base_fqn, 0) + 1
        
        # Determine if this is a layer-based FQN pattern
        # If same FQN appears multiple times (e.g., 6 times for 6 layers), it's a layer tensor
        layer_fqn_patterns = set()
        for fqn, count in fqn_to_occurrences.items():
            if count > 1 and ('layers.' in fqn or 'layer.' in fqn):
                layer_fqn_patterns.add(fqn)
        
        logger.info(f"ECLATIN: Found {len(layer_fqn_patterns)} layer FQN patterns (appearing multiple times)")
        if layer_fqn_patterns and logger.isEnabledFor(logging.DEBUG):
            for pattern in sorted(list(layer_fqn_patterns))[:5]:
                logger.debug(f"  Layer pattern: {pattern} (appears {fqn_to_occurrences[pattern]} times)")
        
        # Second pass: assign layer numbers based on FQN pattern and occurrence order
        fqn_to_layer_counter = {}  # Track current layer number for each FQN pattern
        
        # Group tensor_infos by layer
        for tensor_info in self.decomposed_state_dict.tensor_infos:
            fqn = tensor_info.key
            
            # Try direct extraction first
            layer_num = extract_layer_number(fqn)
            
            # If not found, try to infer from FQN pattern
            if layer_num == -1:
                base_fqn = fqn
                # Normalize to base pattern (remove layer number if present)
                if re.search(r'\.layers\.\d+\.', fqn):
                    base_fqn = re.sub(r'\.layers\.\d+\.', '.layers.', fqn)
                elif re.search(r'^layers\.\d+\.', fqn):
                    base_fqn = re.sub(r'^layers\.\d+\.', 'layers.', fqn)
                
                # If this is a layer pattern (appears multiple times), assign layer number based on occurrence
                if base_fqn in layer_fqn_patterns:
                    if base_fqn not in fqn_to_layer_counter:
                        fqn_to_layer_counter[base_fqn] = 0
                    layer_num = fqn_to_layer_counter[base_fqn]
                    fqn_to_layer_counter[base_fqn] += 1
            
            layer_key = f"layer_{layer_num}" if layer_num >= 0 else "non_layer"
            
            if layer_key not in layer_groups:
                layer_groups[layer_key] = []
            layer_groups[layer_key].append(tensor_info)
        
        # Debug: log a few examples of layer extraction results
        if logger.isEnabledFor(logging.DEBUG):
            sample_extractions = []
            for tensor_info in self.decomposed_state_dict.tensor_infos[:10]:
                layer_num = extract_layer_number(tensor_info.key)
                sample_extractions.append((tensor_info.key, layer_num))
            logger.debug(f"ECLATIN: Sample layer extraction results: {sample_extractions}")
        
        # Log layer extraction results
        layer_keys = [k for k in layer_groups.keys() if k != 'non_layer']
        num_layers = len(layer_keys)
        num_non_layer = len(layer_groups.get('non_layer', []))
        logger.info(
            f"ECLATIN: Extracted {num_layers} layers, {num_non_layer} non-layer tensors"
        )
        if logger.isEnabledFor(logging.DEBUG) and layer_keys:
            logger.debug(f"ECLATIN: Layer keys found: {sorted(layer_keys)}")
        
        # Step 2: Calculate per-layer sizes (own sizes) and non-layer size
        layer_sizes = {}
        non_layer_size = 0
        
        for layer_key, tensor_infos in layer_groups.items():
            if layer_key == "non_layer":
                non_layer_size = sum(info.size_bytes for info in tensor_infos)
                continue
            layer_id = int(layer_key.split('_')[1])
            layer_size = sum(info.size_bytes for info in tensor_infos)
            layer_sizes[layer_id] = layer_size
        
        # Step 3: All-gather per-layer sizes and non-layer sizes, then calculate maximums
        max_non_layer_size = non_layer_size
        avg_layer_size = 0
        
        if torch.distributed.is_initialized():
            # All-gather both layer sizes and non-layer sizes
            all_layer_sizes_list = [None] * world_size
            all_non_layer_sizes_list = [None] * world_size
            
            torch.distributed.all_gather_object(all_layer_sizes_list, layer_sizes)
            torch.distributed.all_gather_object(all_non_layer_sizes_list, non_layer_size)
            
            # Calculate maximum for each layer
            all_layer_sizes_dict = {}
            for rank_layer_sizes in all_layer_sizes_list:
                for layer_id, layer_size in rank_layer_sizes.items():
                    if layer_id not in all_layer_sizes_dict:
                        all_layer_sizes_dict[layer_id] = []
                    all_layer_sizes_dict[layer_id].append(layer_size)
            
            layer_max_sizes = {}
            for layer_id, sizes_list in all_layer_sizes_dict.items():
                layer_max_sizes[layer_id] = max(sizes_list)
            
            # Calculate maximum non-layer size across all ranks
            max_non_layer_size = max(all_non_layer_sizes_list)
            
            # Calculate average layer size (for splitting non-layer data)
            if layer_max_sizes:
                avg_layer_size = sum(layer_max_sizes.values()) // len(layer_max_sizes)
        else:
            # Single rank: use own sizes
            layer_max_sizes = layer_sizes.copy()
            if layer_max_sizes:
                avg_layer_size = sum(layer_max_sizes.values()) // len(layer_max_sizes)
        
        # Step 4: Align per-layer sizes to buffer_size
        eclatin_buffer_size = self.eclatin_manager.eclatin_buffer_size
        layer_aligned_sizes = {}
        for layer_id, max_size in layer_max_sizes.items():
            aligned_size = ((max_size + eclatin_buffer_size - 1) // eclatin_buffer_size) * eclatin_buffer_size
            layer_aligned_sizes[layer_id] = aligned_size
        
        # Step 4.1: Add virtual layers for non-layer data
        if max_non_layer_size > 0 and avg_layer_size > 0:
            # Calculate number of virtual layers needed based on avg_layer_size
            num_virtual_layers = (max_non_layer_size + avg_layer_size - 1) // avg_layer_size
            
            # Calculate virtual layer capacity: max_non_layer_size / num_virtual_layers
            # This ensures each virtual layer won't exceed its capacity
            virtual_layer_capacity = (max_non_layer_size + num_virtual_layers - 1) // num_virtual_layers
            
            # Analyze non-layer tensor sizes
            non_layer_tensors = layer_groups.get('non_layer', [])
            non_layer_tensor_sizes = [(info.key, info.size_bytes) for info in non_layer_tensors]
            non_layer_tensor_sizes_sorted = sorted(non_layer_tensor_sizes, key=lambda x: x[1], reverse=True)
            max_single_tensor_size = non_layer_tensor_sizes_sorted[0][1] if non_layer_tensor_sizes_sorted else 0
            
            logger.info(
                f"ECLATIN: Non-layer data size: {non_layer_size / (1024**2):.2f} MB (own), "
                f"{max_non_layer_size / (1024**2):.2f} MB (max across ranks), "
                f"splitting into {num_virtual_layers} virtual layers "
                f"(capacity: {virtual_layer_capacity / (1024**2):.2f} MB per layer)"
            )
            
            # Log top 10 largest non-layer tensors
            # logger.info(f"ECLATIN: Top 10 largest non-layer tensors:")
            # for i, (key, size) in enumerate(non_layer_tensor_sizes_sorted[:10]):
            #     logger.info(f"  {i+1}. {key}: {size / (1024**2):.2f} MB ({size} bytes)")
            
            # Adjust capacity if single tensor exceeds it
            original_virtual_layer_capacity = virtual_layer_capacity
            if max_single_tensor_size > virtual_layer_capacity:
                logger.warning(
                    f"ECLATIN: Largest single non-layer tensor ({max_single_tensor_size / (1024**2):.2f} MB) "
                    f"exceeds virtual layer capacity ({virtual_layer_capacity / (1024**2):.2f} MB). "
                    f"Adjusting capacity to accommodate largest tensor."
                )
                # Set capacity to max single tensor size
                virtual_layer_capacity = max_single_tensor_size
                logger.info(
                    f"ECLATIN: Adjusted virtual layer capacity: "
                    f"{original_virtual_layer_capacity / (1024**2):.2f} MB -> {virtual_layer_capacity / (1024**2):.2f} MB"
                )
            
            # Add virtual layers to layer_max_sizes and layer_aligned_sizes
            # Use layer IDs starting after the last real layer
            max_real_layer_id = max(layer_max_sizes.keys()) if layer_max_sizes else -1
            virtual_layer_base_id = max_real_layer_id + 1
            
            # For buffer allocation, we need to accommodate both:
            # 1. Large tensors (single large tensor per layer): use adjusted virtual_layer_capacity
            # 2. Packed small tensors: use original_virtual_layer_capacity
            # Use the larger of the two to be safe
            large_layer_aligned_size = ((virtual_layer_capacity + eclatin_buffer_size - 1) // eclatin_buffer_size) * eclatin_buffer_size
            small_layer_aligned_size = ((original_virtual_layer_capacity + eclatin_buffer_size - 1) // eclatin_buffer_size) * eclatin_buffer_size
            max_virtual_layer_aligned_size = max(large_layer_aligned_size, small_layer_aligned_size)
            
            for vl_id in range(num_virtual_layers):
                virtual_layer_id = virtual_layer_base_id + vl_id
                
                # Use max capacity to ensure buffer is large enough for any layer type
                layer_max_sizes[virtual_layer_id] = virtual_layer_capacity
                layer_aligned_sizes[virtual_layer_id] = max_virtual_layer_aligned_size
            
            logger.info(
                f"ECLATIN: Created {num_virtual_layers} virtual layers (IDs {virtual_layer_base_id} to {virtual_layer_base_id + num_virtual_layers - 1}) "
                f"for non-layer data in buffer allocation\n"
                f"  Max aligned size per virtual layer: {max_virtual_layer_aligned_size / (1024**2):.2f} MB "
                f"(supports both large tensors and packed small tensors)"
            )
        
        # Step 5: Calculate own total_half_aligned
        own_total_half_aligned = sum(aligned_size // 2 for aligned_size in layer_aligned_sizes.values())
        
        # Step 6: All-gather all ranks' total_half_aligned and get maximum
        if torch.distributed.is_initialized():
            all_total_half_aligned_list = [None] * world_size
            torch.distributed.all_gather_object(
                all_total_half_aligned_list, 
                own_total_half_aligned
            )
            
            # Calculate global maximum
            max_total_half_aligned = max(all_total_half_aligned_list)
        else:
            max_total_half_aligned = own_total_half_aligned
        
        # Step 7: Align global maximum to buffer_size
        aligned_total_recv_size = ((max_total_half_aligned + eclatin_buffer_size - 1) // eclatin_buffer_size) * eclatin_buffer_size
        
        # Handle zero size buffer case (when no layers detected)
        if aligned_total_recv_size == 0:
            logger.warning(
                "ECLATIN: No layers detected or all layers have zero size. "
                "This may indicate an issue with layer extraction. "
                "Allocating minimum size buffers (1 buffer_size)."
            )
            # Allocate minimum size buffers (at least 1 buffer_size)
            aligned_total_recv_size = eclatin_buffer_size
        
        logger.info(
            f"ECLATIN: Allocating 4 continuous recv buffers for layerwise mode\n"
            f"  Number of layers: {len(layer_aligned_sizes)}\n"
            f"  Own total half_aligned: {own_total_half_aligned / (1024**3):.2f} GB\n"
            f"  Global max total half_aligned: {max_total_half_aligned / (1024**3):.2f} GB\n"
            f"  Aligned buffer size (per buffer): {aligned_total_recv_size / (1024**3):.2f} GB "
            f"({aligned_total_recv_size / (1024**2):.0f} MB)\n"
            f"  Total recv memory: {4 * aligned_total_recv_size / (1024**3):.2f} GB"
        )
        
        # Step 8: Allocate 4 large continuous buffers (all ranks use same size)
        recv_buffer_parity1_1 = torch.empty(
            aligned_total_recv_size, 
            dtype=torch.uint8, 
            pin_memory=self.eclatin_manager.eclatin_pin_memory
        )
        recv_buffer_parity1_2 = torch.empty(
            aligned_total_recv_size, 
            dtype=torch.uint8, 
            pin_memory=self.eclatin_manager.eclatin_pin_memory
        )
        recv_buffer_parity2_1 = torch.empty(
            aligned_total_recv_size, 
            dtype=torch.uint8, 
            pin_memory=self.eclatin_manager.eclatin_pin_memory
        )
        recv_buffer_parity2_2 = torch.empty(
            aligned_total_recv_size, 
            dtype=torch.uint8, 
            pin_memory=self.eclatin_manager.eclatin_pin_memory
        )
        
        logger.info(
            f"ECLATIN: Allocated 4 continuous recv buffers: "
            f"{aligned_total_recv_size / (1024**3):.2f} GB each "
            f"({aligned_total_recv_size / (1024**2):.0f} MB each)"
        )
        
        # Register buffers for RDMA if enabled
        if self.eclatin_manager.use_rdma:
            logger.info("ECLATIN: Registering 4 layerwise recv buffers for RDMA...")
            self.eclatin_manager.register_buffer(recv_buffer_parity1_1)
            self.eclatin_manager.register_buffer(recv_buffer_parity1_2)
            self.eclatin_manager.register_buffer(recv_buffer_parity2_1)
            self.eclatin_manager.register_buffer(recv_buffer_parity2_2)
            logger.info("ECLATIN: RDMA recv buffer registration complete")
        
        return (
            recv_buffer_parity1_1,
            recv_buffer_parity1_2,
            recv_buffer_parity2_1,
            recv_buffer_parity2_2
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
    
    def _derive_metadata_from_sharded_state_dict(self, sharded_state_dict: ShardedStateDict, my_rank: int):
        """
        Derive tensor metadata from sharded_state_dict for failed rank recovery.
        
        This is used when a rank doesn't have its checkpoint file (e.g., after node failure)
        but needs to participate in metadata exchange. The sharded_state_dict describes
        what tensors this rank needs to load, including all necessary metadata.
        
        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict describing tensors to load
            my_rank (int): current rank
            
        Returns:
            Tuple[List[TensorMetadata], Dict]: (tensor_metadata_list, non_tensor_data_dict)
        """
        from .state_dict_decomposer import TensorMetadata
        from ..mapping import ShardedTensor
        from ..dict_utils import nested_values
        
        local_metadata = []
        non_tensor_data = {}
        
        logger.info(f"[Rank {my_rank}] Deriving metadata from sharded_state_dict for failed node recovery")
        
        # Traverse sharded_state_dict to find all ShardedTensor objects
        for sh_ten in nested_values(sharded_state_dict):
            if not isinstance(sh_ten, ShardedTensor):
                continue
            
            # Calculate size_bytes from shape and dtype
            element_size = torch.tensor([], dtype=sh_ten.dtype).element_size()
            numel = 1
            for dim in sh_ten.local_shape:
                numel *= dim
            size_bytes = numel * element_size
            
            # Create TensorMetadata from ShardedTensor
            meta = TensorMetadata(
                key=sh_ten.key,
                shape=sh_ten.local_shape,
                dtype=str(sh_ten.dtype),
                size_bytes=size_bytes,
                global_offset=sh_ten.global_offset if sh_ten.global_offset is not None else (),
                shard_index=sh_ten.replica_id if isinstance(sh_ten.replica_id, int) else 0,
                chunk_type='data',
                target_rank=my_rank,
                source_rank=my_rank,
            )
            local_metadata.append(meta)
        
        logger.info(
            f"[Rank {my_rank}] Derived {len(local_metadata)} tensor metadata entries from sharded_state_dict\n"
            f"  Total size: {sum(m.size_bytes for m in local_metadata) / (1024**3):.2f} GB"
        )
        
        return local_metadata, non_tensor_data
        
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

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """Initialize load strategy.
        
        Args:
            checkpoint_dir: Optional checkpoint directory. If provided, RDMA buffers
                          will be prepared during initialization (for rank0 only).
                          If None, buffers will be prepared on first load() call.
        """
        self.cached_global_metadata: Optional[Metadata] = None
        super().__init__()
        
        # Initialize EC-CHECK manager (singleton instance shared with Save strategy)
        self.eccheck_manager = ECCHECKManager()
        self.eccheck_manager.init_eccheck_if_enabled()
        
        # Initialize Gemini manager (singleton instance shared with Save strategy)
        self.gemini_manager = GeminiManager()
        self.gemini_manager.init_gemini_if_enabled()
        
        # Initialize Gemini Replicas manager (singleton instance shared with Save strategy)
        self.gemini_replicas_manager = GeminiReplicasManager()
        self.gemini_replicas_manager.init_gemini_replicas_if_enabled()
        
        # Initialize ECLATIN manager (singleton instance shared with Save strategy)
        self.eclatin_manager = ECLATINManager()
        self.eclatin_manager.init_eclatin_if_enabled()
        
        # Initialize EC-NAIVE manager (singleton instance shared with Save strategy)
        from .ecnaive_manager import ECNAIVEManager
        self.ecnaive_manager = ECNAIVEManager()
        self.ecnaive_manager.init_ecnaive_if_enabled()
        
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
        
        # Pre-allocated ECLATIN load buffers (allocated on first load, reused on subsequent loads)
        self.eclatin_preallocated_blocks = None  # Dict[str, torch.Tensor]: 4 blocks
        self.eclatin_preallocated_recv_buffers = None  # Dict[str, torch.Tensor]: 6 recv buffers (rank2 only)
        self.eclatin_preallocated_recovered_buffer = None  # torch.Tensor: recovered data buffer (rank2 only)
        
        # Initialize strategy-specific EC-NAIVE state
        self.ecnaive_blocks = None  # 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1)
        self.ecnaive_recv_buffers = None  # 2 recv buffers (rank2 only): recv_data1, recv_parity0
        self.ecnaive_recovered_buffer = None  # Recovered data buffer (rank2 only)
        self.ecnaive_recovered_metadata = None
        self.ecnaive_recovered_registry = None
        
        # Pre-allocated EC-NAIVE load buffers (allocated on first load, reused on subsequent loads)
        self.ecnaive_preallocated_blocks = None  # Dict[str, torch.Tensor]: 4 blocks
        self.ecnaive_preallocated_recv_buffers = None  # Dict[str, torch.Tensor]: 2 recv buffers (rank2 only)
        self.ecnaive_preallocated_recovered_buffer = None  # torch.Tensor: recovered data buffer (rank2 only)
        
        # Initialize Gemini recovery buffers (pre-allocated for rank2 recovery)
        self.gemini_recovery_buffer_replica = None  # Buffer for receiving replica data
        self.gemini_recovery_buffer_rank0 = None    # Buffer for receiving rank0 data
        self._allocate_gemini_recovery_buffers()
        
        # Initialize Gemini Replicas recovery buffers (pre-allocated for rank2 recovery)
        # Gemini Replicas needs to receive from multiple ranks (rank0, rank1, rank3)
        self.gemini_replicas_recovery_buffers = {}  # Dict[int, torch.Tensor]: rank -> buffer
        self._allocate_gemini_replicas_recovery_buffers()
        
        # Initialize Gemini RDMA send buffers for rank0 (mmap files + RDMA-friendly buffers)
        self.gemini_rdma_send_buffers = {}  # Dict[str, dict]: 'replica' and 'own' -> {mmap, buffer, registered}
        self.gemini_mmap_files = {}  # Dict[str, tuple]: 'replica' and 'own' -> (file_handle, mmap_handle)
        self.gemini_rdma_checkpoint_dir = checkpoint_dir  # Track which checkpoint_dir buffers are prepared for
        
        # Initialize Gemini Replicas RDMA send buffers for sender ranks (rank0, rank1, rank3)
        # Each rank sends one file to rank2 during recovery
        self.gemini_replicas_rdma_send_buffers = {}  # Dict[str, dict]: buffer_name -> {mmap, buffer, registered}
        self.gemini_replicas_mmap_files = {}  # Dict[str, tuple]: buffer_name -> (file_handle, mmap_handle)
        self.gemini_replicas_rdma_checkpoint_dir = None  # Track which checkpoint_dir buffers are prepared for
    
        self.pairing_map = {0: 2, 2: 0, 1: 3, 3: 1}
        
        # If checkpoint_dir provided, prepare RDMA buffers now (rank0 only)
        # # this is desgin for use ,but checkpoint_dir can not be pass
        # if checkpoint_dir is not None:
        #     self._prepare_gemini_rdma_buffers_if_needed(checkpoint_dir)
    
    def _prepare_gemini_rdma_buffers_if_needed(self, checkpoint_dir: Path):
        """Prepare send buffers if needed (wrapper function).
        
        This function checks if buffers need to be prepared and calls the actual
        preparation function. Buffers are always prepared (for ASIO/RDMA), but
        RDMA registration only happens when use_rdma is enabled.
        Can be called from __init__ or load().
        
        Args:
            checkpoint_dir: Checkpoint directory
        """
        if not torch.distributed.is_initialized():
            return  # Skip if distributed not initialized
        
        rank = torch.distributed.get_rank()
        if rank != 0:
            return  # Only rank0 needs send buffers
        
        checkpoint_dir_str = str(checkpoint_dir)
        
        # Check if buffers already prepared for this checkpoint_dir
        if self.gemini_rdma_checkpoint_dir == checkpoint_dir_str:
            logger.debug(f"rank: {rank}, send buffers already prepared for {checkpoint_dir_str}")
            return
        
        # Cleanup old buffers if checkpoint_dir changed
        if self.gemini_rdma_send_buffers:
            logger.info(f"rank: {rank}, checkpoint_dir changed, cleaning up old send buffers...")
            self._cleanup_gemini_rdma_send_buffers()
        
        # Prepare new buffers (always prepare, RDMA registration happens inside based on use_rdma flag)
        logger.info(f"rank: {rank}, preparing send buffers for {checkpoint_dir_str}...")
        self._prepare_gemini_rdma_send_buffers(checkpoint_dir)
        self.gemini_rdma_checkpoint_dir = checkpoint_dir_str
    
    def _prepare_gemini_rdma_send_buffers(self, checkpoint_dir: Path):
        """Prepare send buffers for Gemini load (rank0 only).
        
        This method:
        1. Opens checkpoint files with mmap
        2. If RDMA enabled: Attempts to register mmap buffers with RDMA
        3. If RDMA registration fails: Allocates aligned C++ buffers and copies data
        4. If RDMA enabled: Registers the aligned buffers with RDMA
        
        Buffers are always prepared (for ASIO/RDMA), but RDMA registration only
        happens when use_rdma is enabled.
        
        Args:
            checkpoint_dir: Checkpoint directory containing replica files
        """
        import mmap
        import numpy as np
        
        rank = torch.distributed.get_rank()
        if rank != 0:
            return  # Only rank0 needs send buffers
        
        paired_rank = self.pairing_map.get(rank, None)
        if paired_rank != 2:
            return  # Only for rank0->rank2 recovery
        
        checkpoint_dir = Path(checkpoint_dir)
        
        # Check if RDMA is enabled
        use_rdma = self.gemini_manager.use_rdma if hasattr(self.gemini_manager, 'use_rdma') else False
        transport_mode = "RDMA" if use_rdma else "ASIO"
        
        logger.info(f"rank: {rank}, preparing send buffers for Gemini load ({transport_mode} mode)")
        
        try:
            # Find checkpoint files
            replica_files = list(checkpoint_dir.glob(f"*_replica{paired_rank}_rank{rank}*.distcp"))
            own_checkpoint_files = list(checkpoint_dir.glob(f"__{rank}_0.distcp"))
            
            if not replica_files or not own_checkpoint_files:
                logger.warning(f"rank: {rank}, checkpoint files not found, skipping RDMA buffer preparation")
                return
            
            replica_file_path = replica_files[0]
            own_checkpoint_path = own_checkpoint_files[0]
            
            logger.info(f"rank: {rank}, found checkpoint files:\n"
                       f"  replica: {replica_file_path}\n"
                       f"  own: {own_checkpoint_path}")
            
            # Process replica file
            self._prepare_single_rdma_send_buffer('replica', replica_file_path)
            
            # Process own checkpoint file
            self._prepare_single_rdma_send_buffer('own', own_checkpoint_path)
            
            logger.info(f"rank: {rank}, RDMA send buffers prepared successfully")
            
        except Exception as e:
            logger.error(f"rank: {rank}, failed to prepare RDMA send buffers: {e}", exc_info=True)
            # Clean up any partial allocations
            self._cleanup_gemini_rdma_send_buffers()
    
    def _prepare_single_rdma_send_buffer(self, buffer_name: str, file_path: Path):
        """Prepare a single send buffer from a checkpoint file.
        
        If RDMA is enabled, attempts to register the buffer for RDMA operations.
        If RDMA is disabled, just prepares the buffer for ASIO operations.
        
        Args:
            buffer_name: 'replica' or 'own'
            file_path: Path to checkpoint file
        """
        import mmap
        import numpy as np
        import ctypes
        
        rank = torch.distributed.get_rank()
        
        # Check if RDMA is enabled
        use_rdma = self.gemini_manager.use_rdma if hasattr(self.gemini_manager, 'use_rdma') else False
        has_register_func = hasattr(self.gemini_manager._gemini_native, 'register_buffer')
        
        try:
            # Open file with mmap
            f = open(file_path, 'rb')
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            file_size = len(mm)
            
            logger.info(f"rank: {rank}, opened {buffer_name} file with mmap: {file_size / (1024**2):.2f} MB")
            
            # Get mmap buffer address
            mmap_np = np.frombuffer(mm, dtype=np.uint8)
            mmap_addr = mmap_np.ctypes.data
            
            # Try to register mmap buffer for RDMA if enabled
            registered = False
            use_copy = False
            
            if use_rdma and has_register_func:
                try:
                    self.gemini_manager._gemini_native.register_buffer(mmap_addr, file_size)
                    logger.info(f"rank: {rank}, successfully registered mmap buffer for {buffer_name} (RDMA zero-copy)")
                    registered = True
                    
                except Exception as e:
                    logger.warning(f"rank: {rank}, failed to register mmap buffer for {buffer_name}: {e}")
                    logger.info(f"rank: {rank}, allocating aligned buffer for {buffer_name}")
                    
                    # Allocate aligned buffer (page-aligned for RDMA)
                    aligned_buffer = torch.empty(file_size, dtype=torch.uint8, pin_memory=True)
                    
                    # Copy data from mmap to aligned buffer
                    logger.info(f"rank: {rank}, copying {file_size / (1024**2):.2f} MB from mmap to aligned buffer...")
                    aligned_buffer_np = aligned_buffer.numpy()
                    aligned_buffer_np[:] = mmap_np[:]
                    logger.info(f"rank: {rank}, copy completed for {buffer_name}")
                    
                    # Register aligned buffer
                    aligned_addr = aligned_buffer.data_ptr()
                    self.gemini_manager._gemini_native.register_buffer(aligned_addr, file_size)
                    logger.info(f"rank: {rank}, successfully registered aligned buffer for {buffer_name} (RDMA)")
                    
                    # Use aligned buffer
                    mmap_addr = aligned_addr
                    registered = True
                    use_copy = True
                    
                    # Store aligned buffer
                    self.gemini_rdma_send_buffers[buffer_name] = {
                        'addr': mmap_addr,
                        'size': file_size,
                        'registered': registered,
                        'use_copy': use_copy,
                        'buffer': aligned_buffer,  # Keep tensor alive
                        'numpy_ref': None
                    }
                    self.gemini_mmap_files[buffer_name] = (f, mm)
                    return
            
            # Store mmap handles (RDMA registered or ASIO mode)
            self.gemini_mmap_files[buffer_name] = (f, mm)
            self.gemini_rdma_send_buffers[buffer_name] = {
                'addr': mmap_addr,
                'size': file_size,
                'registered': registered,
                'use_copy': use_copy,
                'buffer': None,
                'numpy_ref': mmap_np  # Keep reference to prevent GC
            }
            
            if not use_rdma:
                logger.info(f"rank: {rank}, prepared {buffer_name} buffer for ASIO (no RDMA registration)")
                
        except Exception as e:
            logger.error(f"rank: {rank}, failed to prepare {buffer_name} buffer: {e}", exc_info=True)
            raise
    
    def _cleanup_gemini_rdma_send_buffers(self):
        """Clean up RDMA send buffers and mmap files."""
        rank = torch.distributed.get_rank()
        
        # Unregister RDMA buffers
        for buffer_name, buffer_info in self.gemini_rdma_send_buffers.items():
            if buffer_info.get('registered', False):
                try:
                    self.gemini_manager._gemini_native.unregister_buffer(buffer_info['addr'])
                    logger.info(f"rank: {rank}, unregistered RDMA buffer for {buffer_name}")
                except Exception as e:
                    logger.warning(f"rank: {rank}, failed to unregister {buffer_name}: {e}")
        
        # Close mmap files
        for buffer_name, (f, mm) in self.gemini_mmap_files.items():
            try:
                mm.close()
                f.close()
                logger.info(f"rank: {rank}, closed mmap file for {buffer_name}")
            except Exception as e:
                logger.warning(f"rank: {rank}, failed to close mmap for {buffer_name}: {e}")
        
        self.gemini_rdma_send_buffers.clear()
        self.gemini_mmap_files.clear()
    
    def _prepare_gemini_replicas_rdma_send_buffers_if_needed(self, checkpoint_dir: Path):
        """Prepare Gemini Replicas send buffers if needed (wrapper function).
        
        This function checks if buffers need to be prepared and calls the actual
        preparation function. Buffers are always prepared (for ASIO/RDMA), but
        RDMA registration only happens when use_rdma is enabled.
        
        For Gemini Replicas recovery, sender ranks are:
        - rank0: sends own data to rank2
        - rank1: sends own data to rank2  
        - rank3: sends rank2's replica to rank2
        
        Args:
            checkpoint_dir: Checkpoint directory
        """
        if not torch.distributed.is_initialized():
            return
        
        rank = torch.distributed.get_rank()
        
        # Only sender ranks (0, 1, 3) need send buffers
        if rank not in [0, 1, 3]:
            return
        
        checkpoint_dir_str = str(checkpoint_dir)
        
        # Check if buffers already prepared for this checkpoint_dir
        if self.gemini_replicas_rdma_checkpoint_dir == checkpoint_dir_str:
            logger.debug(f"rank: {rank}, Gemini Replicas send buffers already prepared for {checkpoint_dir_str}")
            return
        
        # Cleanup old buffers if checkpoint_dir changed
        if self.gemini_replicas_rdma_send_buffers:
            logger.info(f"rank: {rank}, checkpoint_dir changed, cleaning up old Gemini Replicas send buffers...")
            self._cleanup_gemini_replicas_rdma_send_buffers()
        
        # Prepare new buffers
        self._prepare_gemini_replicas_rdma_send_buffers(checkpoint_dir)
        self.gemini_replicas_rdma_checkpoint_dir = checkpoint_dir_str
    
    def _prepare_gemini_replicas_rdma_send_buffers(self, checkpoint_dir: Path):
        """Prepare Gemini Replicas send buffers for recovery.
        
        Each sender rank prepares one file:
        - rank0: __0_0.distcp (own data)
        - rank1: __1_0.distcp (own data)
        - rank3: __3_0_replica2_rank3.distcp (rank2's replica)
        
        Args:
            checkpoint_dir: Checkpoint directory
        """
        rank = torch.distributed.get_rank()
        checkpoint_dir = Path(checkpoint_dir)
        
        # Check if RDMA is enabled
        use_rdma = self.gemini_replicas_manager.use_rdma if hasattr(self.gemini_replicas_manager, 'use_rdma') else False
        transport_mode = "RDMA" if use_rdma else "ASIO"
        
        logger.info(f"rank: {rank}, preparing Gemini Replicas send buffers ({transport_mode} mode)")
        
        try:
            # Determine which file to send based on rank
            if rank == 3:
                # rank3 sends rank2's replica
                send_files = list(checkpoint_dir.glob(f"__{rank}_0_replica2_rank{rank}.distcp"))
                if not send_files:
                    send_files = list(checkpoint_dir.glob(f"*_replica2_rank{rank}*.distcp"))
            else:
                # rank0, rank1 send their own data
                send_files = list(checkpoint_dir.glob(f"__{rank}_0.distcp"))
            
            if not send_files:
                logger.warning(f"rank: {rank}, checkpoint file not found, skipping buffer preparation")
                return
            
            send_file_path = send_files[0]
            
            logger.info(f"rank: {rank}, found checkpoint file: {send_file_path}")
            
            # Prepare send buffer
            self._prepare_single_gemini_replicas_rdma_send_buffer('send', send_file_path)
            
            logger.info(f"rank: {rank}, Gemini Replicas send buffer prepared successfully")
            
        except Exception as e:
            logger.error(f"rank: {rank}, failed to prepare Gemini Replicas send buffers: {e}", exc_info=True)
            # Clean up any partial allocations
            self._cleanup_gemini_replicas_rdma_send_buffers()
    
    def _prepare_single_gemini_replicas_rdma_send_buffer(self, buffer_name: str, file_path: Path):
        """Prepare a single Gemini Replicas send buffer from a checkpoint file.
        
        Similar to Gemini's _prepare_single_rdma_send_buffer, but for Gemini Replicas.
        
        Args:
            buffer_name: Buffer identifier (e.g., 'send')
            file_path: Path to checkpoint file
        """
        import mmap
        import numpy as np
        
        rank = torch.distributed.get_rank()
        
        try:
            # Open file with mmap (zero-copy read)
            f = open(file_path, 'rb')
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            file_size = len(mm)
            
            logger.info(f"rank: {rank}, opened file with mmap: {file_size / (1024**2):.2f} MB")
            
            # Store mmap reference
            self.gemini_replicas_mmap_files[buffer_name] = (f, mm)
            
            # Check if RDMA is enabled
            use_rdma = self.gemini_replicas_manager.use_rdma if hasattr(self.gemini_replicas_manager, 'use_rdma') else False
            
            if use_rdma and self.gemini_replicas_manager.is_initialized():
                # For RDMA: Create aligned buffer, copy data, register
                logger.info(f"rank: {rank}, preparing RDMA buffer for {buffer_name}...")
                
                # Allocate aligned buffer (pinned memory for better RDMA performance)
                if torch.cuda.is_available():
                    aligned_buffer = torch.empty(file_size, dtype=torch.uint8).pin_memory()
                else:
                    aligned_buffer = torch.empty(file_size, dtype=torch.uint8)
                
                # Copy data from mmap to buffer
                mmap_np = np.frombuffer(mm, dtype=np.uint8)
                aligned_buffer_np = aligned_buffer.numpy()
                np.copyto(aligned_buffer_np, mmap_np)
                
                logger.info(f"rank: {rank}, copied {file_size / (1024**2):.2f} MB to aligned buffer")
                
                # Register buffer for RDMA
                buffer_addr = aligned_buffer.data_ptr()
                try:
                    self.gemini_replicas_manager.register_buffer(aligned_buffer)
                    logger.info(f"rank: {rank}, registered {buffer_name} buffer for RDMA")
                    
                    # Store buffer info
                    self.gemini_replicas_rdma_send_buffers[buffer_name] = {
                        'buffer': aligned_buffer,
                        'addr': buffer_addr,
                        'size': file_size,
                        'registered': True
                    }
                except Exception as e:
                    logger.error(f"rank: {rank}, failed to register {buffer_name} for RDMA: {e}")
                    raise
                    
            else:
                # For ASIO: Just keep mmap reference
                logger.info(f"rank: {rank}, prepared {buffer_name} buffer for ASIO (no RDMA registration)")
                
                # Create numpy view of mmap (for ASIO send)
                mmap_np = np.frombuffer(mm, dtype=np.uint8)
                mmap_addr = mmap_np.ctypes.data
                
                # Store buffer info (ASIO will use mmap directly)
                self.gemini_replicas_rdma_send_buffers[buffer_name] = {
                    'addr': mmap_addr,
                    'size': file_size,
                    'registered': False,
                    'mmap_np': mmap_np  # Keep numpy view alive
                }
                
        except Exception as e:
            logger.error(f"rank: {rank}, failed to prepare {buffer_name} buffer: {e}", exc_info=True)
            raise
    
    def _cleanup_gemini_replicas_rdma_send_buffers(self):
        """Clean up Gemini Replicas RDMA send buffers and mmap files."""
        rank = torch.distributed.get_rank()
        
        # Unregister RDMA buffers
        for buffer_name, buffer_info in self.gemini_replicas_rdma_send_buffers.items():
            if buffer_info.get('registered', False):
                try:
                    buffer = buffer_info.get('buffer')
                    if buffer is not None:
                        self.gemini_replicas_manager.unregister_buffer(buffer)
                    logger.info(f"rank: {rank}, unregistered Gemini Replicas RDMA buffer for {buffer_name}")
                except Exception as e:
                    logger.warning(f"rank: {rank}, failed to unregister {buffer_name}: {e}")
        
        # Close mmap files
        for buffer_name, (f, mm) in self.gemini_replicas_mmap_files.items():
            try:
                mm.close()
                f.close()
                logger.info(f"rank: {rank}, closed Gemini Replicas mmap file for {buffer_name}")
            except Exception as e:
                logger.warning(f"rank: {rank}, failed to close mmap for {buffer_name}: {e}")
        
        self.gemini_replicas_rdma_send_buffers.clear()
        self.gemini_replicas_mmap_files.clear()
        self.gemini_replicas_rdma_checkpoint_dir = None
    
    def _allocate_gemini_recovery_buffers(self):
        """Pre-allocate large buffers for Gemini recovery to avoid allocation overhead.
        
        Only allocates for rank2 (the recovery rank). Allocates two buffers:
        1. replica buffer: for receiving rank2's own backup data
        2. rank0 buffer: for receiving rank0's checkpoint data
        
        Buffer size is set to 4GB by default, which should be enough for most models.
        If the actual data size exceeds this, we'll fall back to dynamic allocation.
        """
        try:
            from megatron.training import get_args
            args = get_args()
            
            # Only allocate for rank2 and only if Gemini is enabled
            if not (hasattr(args, 'use_gemini') and args.use_gemini):
                return
            
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank != 2:
                return
            
            # Default buffer size: 4GB (adjustable via args if needed)
            buffer_size_gb = getattr(args, 'gemini_recovery_buffer_size_gb', 2)
            buffer_size_bytes = buffer_size_gb * 1024 * 1024 * 1024
            
            logger.info(f"rank: {rank}, allocating Gemini recovery buffers: {buffer_size_gb} GB each")
            
            # Allocate pinned memory buffers for faster GPU transfer
            if torch.cuda.is_available():
                self.gemini_recovery_buffer_replica = torch.empty(
                    buffer_size_bytes, dtype=torch.uint8
                ).pin_memory()
                self.gemini_recovery_buffer_rank0 = torch.empty(
                    buffer_size_bytes, dtype=torch.uint8
                ).pin_memory()
                logger.info(f"rank: {rank}, allocated pinned memory buffers for Gemini recovery")
            else:
                self.gemini_recovery_buffer_replica = torch.empty(
                    buffer_size_bytes, dtype=torch.uint8
                )
                self.gemini_recovery_buffer_rank0 = torch.empty(
                    buffer_size_bytes, dtype=torch.uint8
                )
                logger.info(f"rank: {rank}, allocated CPU buffers for Gemini recovery")
            
            logger.info(
                f"rank: {rank}, Gemini recovery buffers allocated successfully: "
                f"{buffer_size_bytes / (1024**3):.2f} GB x 2"
            )
            
            # Register buffers for RDMA if enabled (check both use_rdma flag and native module capability)
            use_rdma = getattr(args, 'use_rdma', False) if hasattr(args, 'use_rdma') else False
            if use_rdma and hasattr(self.gemini_manager._gemini_native, 'register_buffer'):
                try:
                    replica_addr = self.gemini_recovery_buffer_replica.data_ptr()
                    rank0_addr = self.gemini_recovery_buffer_rank0.data_ptr()
                    
                    self.gemini_manager._gemini_native.register_buffer(replica_addr, buffer_size_bytes)
                    logger.info(f"rank: {rank}, registered replica recovery buffer for RDMA ({buffer_size_gb} GB)")
                    
                    self.gemini_manager._gemini_native.register_buffer(rank0_addr, buffer_size_bytes)
                    logger.info(f"rank: {rank}, registered rank0 recovery buffer for RDMA ({buffer_size_gb} GB)")
                    
                except Exception as e:
                    logger.warning(f"rank: {rank}, failed to register recovery buffers for RDMA: {e}")
                    logger.warning(f"rank: {rank}, will use unregistered buffers (may fall back to temp buffers)")
            elif use_rdma:
                logger.info(f"rank: {rank}, RDMA enabled but native module not available, skipping buffer registration")
            
        except Exception as e:
            logger.warning(f"Failed to allocate Gemini recovery buffers: {e}")
            # Fall back to dynamic allocation
            self.gemini_recovery_buffer_replica = None
            self.gemini_recovery_buffer_rank0 = None
    
    def _get_gemini_recovery_buffer(self, buffer_type: str, required_size: int) -> torch.Tensor:
        """Get pre-allocated buffer or allocate dynamically if needed.
        
        Args:
            buffer_type: 'replica' or 'rank0'
            required_size: Required buffer size in bytes
            
        Returns:
            torch.Tensor: Buffer to use (either pre-allocated or newly allocated)
        """
        if buffer_type == 'replica':
            preallocated_buffer = self.gemini_recovery_buffer_replica
        elif buffer_type == 'rank0':
            preallocated_buffer = self.gemini_recovery_buffer_rank0
        else:
            raise ValueError(f"Invalid buffer_type: {buffer_type}")
        
        # Check if pre-allocated buffer is available and large enough
        if preallocated_buffer is not None and preallocated_buffer.numel() >= required_size:
            logger.info(
                f"Using pre-allocated {buffer_type} buffer: "
                f"{required_size / (1024**2):.2f} MB / {preallocated_buffer.numel() / (1024**2):.2f} MB"
            )
            # Return a view of the required size
            return preallocated_buffer[:required_size]
        else:
            # Fall back to dynamic allocation
            logger.warning(
                f"Pre-allocated {buffer_type} buffer not available or too small "
                f"(required: {required_size / (1024**2):.2f} MB), allocating dynamically"
            )
            if torch.cuda.is_available():
                return torch.empty(required_size, dtype=torch.uint8).pin_memory()
            else:
                return torch.empty(required_size, dtype=torch.uint8)
    
    def _allocate_gemini_replicas_recovery_buffers(self):
        """Pre-allocate buffers for Gemini Replicas recovery.
        
        For rank2 recovery, we need to receive data from multiple ranks:
        - rank0: sends its own data (rank2 had rank0's backup)
        - rank1: sends its own data (rank2 had rank1's backup)
        - rank3: sends rank2's replica (rank3 had rank2's backup)
        
        We allocate 3 buffers for rank2 to receive from these 3 sources.
        """
        try:
            from megatron.training import get_args
            args = get_args()
            
            # Only allocate for rank2 and only if Gemini Replicas is enabled
            if not (hasattr(args, 'use_gemini_replicas') and args.use_gemini_replicas):
                return
            
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank != 2:
                return
            
            # Default buffer size: 4GB per buffer (adjustable via args if needed)
            buffer_size_gb = getattr(args, 'gemini_replicas_recovery_buffer_size_gb', 2)
            buffer_size_bytes = buffer_size_gb * 1024 * 1024 * 1024
            
            # Ranks that will send data to rank2 during recovery
            source_ranks = [0, 1, 3]
            
            logger.info(
                f"rank: {rank}, allocating Gemini Replicas recovery buffers: "
                f"{buffer_size_gb} GB x {len(source_ranks)} (for ranks {source_ranks})"
            )
            
            # Allocate one buffer for each source rank
            for src_rank in source_ranks:
                if torch.cuda.is_available():
                    self.gemini_replicas_recovery_buffers[src_rank] = torch.empty(
                        buffer_size_bytes, dtype=torch.uint8
                    ).pin_memory()
                else:
                    self.gemini_replicas_recovery_buffers[src_rank] = torch.empty(
                        buffer_size_bytes, dtype=torch.uint8
                    )
            
            if torch.cuda.is_available():
                logger.info(f"rank: {rank}, allocated pinned memory buffers for Gemini Replicas recovery")
            else:
                logger.info(f"rank: {rank}, allocated CPU buffers for Gemini Replicas recovery")
            
            logger.info(
                f"rank: {rank}, Gemini Replicas recovery buffers allocated successfully: "
                f"{buffer_size_bytes / (1024**3):.2f} GB x {len(source_ranks)} = "
                f"{buffer_size_bytes * len(source_ranks) / (1024**3):.2f} GB total"
            )
            
            # Register buffers for RDMA if enabled
            use_rdma = getattr(args, 'use_rdma', False) if hasattr(args, 'use_rdma') else False
            if use_rdma and self.gemini_replicas_manager.is_initialized():
                try:
                    for src_rank in source_ranks:
                        buffer = self.gemini_replicas_recovery_buffers[src_rank]
                        buffer_addr = buffer.data_ptr()
                        
                        self.gemini_replicas_manager.register_buffer(buffer)
                        logger.info(
                            f"rank: {rank}, registered recovery buffer for source rank {src_rank} "
                            f"for RDMA ({buffer_size_gb} GB)"
                        )
                    
                    logger.info(f"rank: {rank}, all Gemini Replicas recovery buffers registered for RDMA")
                    
                except Exception as e:
                    logger.warning(f"rank: {rank}, failed to register recovery buffers for RDMA: {e}")
                    logger.warning(f"rank: {rank}, will use unregistered buffers (may fall back to temp buffers)")
            elif use_rdma:
                logger.info(f"rank: {rank}, RDMA enabled but Gemini Replicas not initialized, skipping buffer registration")
            
        except Exception as e:
            logger.warning(f"Failed to allocate Gemini Replicas recovery buffers: {e}")
            # Fall back to dynamic allocation
            self.gemini_replicas_recovery_buffers = {}
    
    def _get_gemini_replicas_recovery_buffer(self, source_rank: int, required_size: int) -> torch.Tensor:
        """Get pre-allocated buffer for Gemini Replicas recovery or allocate dynamically.
        
        Args:
            source_rank: The rank we're receiving from (0, 1, or 3)
            required_size: Required buffer size in bytes
            
        Returns:
            torch.Tensor: Buffer to use (either pre-allocated or newly allocated)
        """
        preallocated_recovery_buffer = self.gemini_replicas_recovery_buffers.get(source_rank)
        
        # Check if pre-allocated buffer is available and large enough
        if preallocated_recovery_buffer is not None and preallocated_recovery_buffer.numel() >= required_size:
            logger.info(
                f"Using pre-allocated buffer for rank{source_rank}: "
                f"{required_size / (1024**2):.2f} MB / {preallocated_recovery_buffer.numel() / (1024**2):.2f} MB"
            )
            # Return a view of the required size
            return preallocated_recovery_buffer[:required_size]
        else:
            # Fall back to dynamic allocation
            logger.warning(
                f"Pre-allocated buffer for rank{source_rank} not available or too small "
                f"(required: {required_size / (1024**2):.2f} MB), allocating dynamically"
            )
            if torch.cuda.is_available():
                buffer = torch.empty(required_size, dtype=torch.uint8).pin_memory()
            else:
                buffer = torch.empty(required_size, dtype=torch.uint8)
            
            # Register buffer for RDMA if needed
            from .gemini_replicas_manager import GeminiReplicasManager
            manager = GeminiReplicasManager()
            if manager.use_rdma and manager.is_initialized():
                try:
                    logger.info(f"Registering Gemini Replicas recovery buffer for source rank {source_rank} (RDMA)...")
                    manager.register_buffer(buffer)
                    logger.info(f"Gemini Replicas recovery buffer for source rank {source_rank} registered for RDMA")
                except Exception as e:
                    logger.warning(f"Failed to register Gemini Replicas recovery buffer for RDMA: {e}")
            
            return buffer
    
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
    
    def _is_ecnaive_checkpoint(self, checkpoint_dir: Path) -> bool:
        """Check if the checkpoint is in EC-NAIVE format.
        
        EC-NAIVE checkpoints are .distcp files with 'ECNV' magic number in the header.
        (EC-NAIVE uses the same file format as ECLATIN, but with ECNV magic number)
        
        Args:
            checkpoint_dir (Path): checkpoint directory
            
        Returns:
            bool: True if this is an EC-NAIVE checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return False
        
        # Get current rank to find the corresponding file
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Check for EC-NAIVE format: __{rank}_0.distcp with ECNV magic number
        potential_file = checkpoint_dir / f"__{rank}_0.distcp"
        
        if not potential_file.exists():
            return False
        
        # Read first 4 bytes to check for ECNV magic number
        try:
            with open(potential_file, 'rb') as f:
                magic = f.read(4)
                return magic == b'ECNV'
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
        
        # Determine failed_rank based on flags
        from megatron.training import get_args as use_args
        input_args = use_args()
        if input_args.use_eccheck_software_failure:
            failed_rank = 1  # rank1 software failure
            logger.info(f"EC-CHECK: [Rank {rank}] Software failure recovery mode (failed_rank=1)")
        else:
            failed_rank = 2  # Default: rank2 hardware failure
            logger.info(f"EC-CHECK: [Rank {rank}] Hardware failure recovery mode (failed_rank=2)")
        
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
            
        meta_start_time = time()
        
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
        meta_end_time = time()
        meta_time = meta_end_time - meta_start_time
        logger.info(f"EC-CHECK: [Rank {rank}] Metadata exchange completed in {meta_time:.2f}s")
        # ===== Step 3: Prepare P2P buffers (own_buffer and partner_buffer) =====
        # Check if buffers exist in manager and can be reused, or allocate new ones
        if self.eccheck_manager.eccheck_p2p_buffers is not None:
            existing_buffers = self.eccheck_manager.eccheck_p2p_buffers
            existing_own_size = existing_buffers['own_buffer'].numel()
            existing_partner_size = existing_buffers['partner_buffer'].numel()
            
            # Calculate required buffer sizes from registry
            own_metadata = registry.rank_metadata.get(rank, [])
            # own_total_size = sum(meta.size_bytes for meta in own_metadata)
            partner_metadata = registry.rank_metadata.get(p2p_partner_rank, [])
            # partner_total_size = sum(meta.size_bytes for meta in partner_metadata)
            
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
                # logger.info(
                #     f"EC-CHECK: Reusing existing P2P buffers from manager "
                #     f"(own: {existing_own_size / (1024**3):.2f} GB >= {needed_own_size / (1024**3):.2f} GB, "
                #     f"partner: {existing_partner_size / (1024**3):.2f} GB >= {needed_partner_size / (1024**3):.2f} GB)"
                # )
                self.eccheck_p2p_buffers = existing_buffers
            else:
                # Existing buffers too small, reallocate
                # logger.info(
                #     f"EC-CHECK: Existing buffers too small, reallocating "
                #     f"(own: {existing_own_size / (1024**3):.2f} GB < {needed_own_size / (1024**3):.2f} GB or "
                #     f"partner: {existing_partner_size / (1024**3):.2f} GB < {needed_partner_size / (1024**3):.2f} GB)"
                # )
                self.eccheck_p2p_buffers = self._allocate_p2p_buffers(registry)
                self.eccheck_manager.eccheck_p2p_buffers = self.eccheck_p2p_buffers
        else:
            # First-time allocation (e.g., after process restart)
            # logger.info("EC-CHECK: Allocating P2P buffers from checkpoint metadata")
            self.eccheck_p2p_buffers = self._allocate_p2p_buffers(registry)
            self.eccheck_manager.eccheck_p2p_buffers = self.eccheck_p2p_buffers
        
        paired_rank = self._get_p2p_partner_rank(rank, world_size)
        
        # Get self metadata form peer rank in global registry
        metadata_in_peer = registry.rank_metadata.get(paired_rank, [])
        # meta_type = metadata_in_peer[0].chunk_type
        recv_total_size = sum(meta.size_bytes for meta in metadata_in_peer)
    
        recv_own_buffer = torch.empty(recv_total_size, dtype=torch.uint8)
        
        # Simple P2P exchange placeholder. This will be extended into a full
        # EC-CHECK recovery pipeline (encoding + XOR + P2P) in later steps.
        data_start_time = time()
        # Get failed_rank from args (same logic as above)
        from megatron.training import get_args as use_args
        input_args = use_args()
        if input_args.use_eccheck_software_failure:
            failed_rank = 1
        else:
            failed_rank = 2
        
        self._run_eccheck_p2p_pipeline_simple(
            rank=rank,
            world_size=world_size,
            registry=registry,
            mapped_file_own=mapped_file_own,
            mapped_file_partner=mapped_file_partner,
            recv_own_buffer=recv_own_buffer,
            recv_total_size=recv_total_size,
            failed_rank=failed_rank,  # Pass failed_rank parameter
        )
        data_end_time = time()
        data_time = data_end_time - data_start_time
        logger.info(f"EC-CHECK: [Rank {rank}] Data exchange completed in {data_time:.2f}s")
        # For rank2 recovery: save recovered data for later use in _load_eccheck_checkpoint
        if rank == failed_rank:
            logger.info(f"EC-CHECK: [Rank {rank}] Saving recovered buffer for _load_eccheck_checkpoint")
            # Store recovered data in instance variables for _load_eccheck_checkpoint to use
            self.eccheck_recovered_buffer = recv_own_buffer
            self.eccheck_recovered_metadata = mapped_file_own
            self.eccheck_recovered_registry = registry
        
        # Return EccheckMappedFile, non_tensor_data, and local_metadata for each file
        return mapped_file_own, mapped_file_partner
    
    def _load_eclatin_block_checkpoint(self, checkpoint_dir: Path, sharded_state_dict: ShardedStateDict = None) -> Tuple:
        """Load ECLATIN checkpoint data and recover rank2.
        
        Similar to EC-CHECK for rank2 recovery.
        
        Args:
            checkpoint_dir (Path): checkpoint directory
            sharded_state_dict (ShardedStateDict): sharded state dict for failed rank to derive metadata
        
        Returns:
            Tuple: (mapped_file_own, None) - placeholder for compatibility
        """
        from .filesystem_async import FileSystemWriterAsync
        from time import time
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # Determine failed_rank based on flags
        from megatron.training import get_args as use_args
        input_args = use_args()
        if input_args.use_eclatin_software_failure:
            failed_rank = 2  # rank2 software failure (can read local files)
            logger.info(f"ECLATIN: [Rank {rank}] Software failure recovery mode (failed_rank=2, reading local files)")
        else:
            failed_rank = 2  # Default: rank2 hardware failure (needs network recovery)
            logger.info(f"ECLATIN: [Rank {rank}] Hardware failure recovery mode (failed_rank=2, network recovery)")
        
        checkpoint_dir = Path(checkpoint_dir)
        
        # Store checkpoint_dir for software failure recovery
        self._current_checkpoint_dir = checkpoint_dir
        
        # ===== Step 1: Load main file to extract metadata =====
        eclatin_main_file = checkpoint_dir / f'__{rank}_0.distcp'
        
        # Handle failed rank that doesn't have checkpoint file
        if not eclatin_main_file.exists():
            if rank == failed_rank:
                logger.warning(f"ECLATIN: [Rank {rank}] Main file not found (failed node), deriving metadata from sharded_state_dict")
                from .filesystem_async import EclatinMappedFile
                
                # Derive metadata from sharded_state_dict
                if sharded_state_dict is not None:
                    local_metadata, non_tensor_data = self._derive_metadata_from_sharded_state_dict(sharded_state_dict, rank)
                    logger.info(f"ECLATIN: [Rank {rank}] Derived {len(local_metadata)} tensor metadata entries from sharded_state_dict")
                else:
                    logger.warning(f"ECLATIN: [Rank {rank}] sharded_state_dict is None, using empty metadata")
                    local_metadata = []
                    non_tensor_data = {}
                
                mapped_file_own = EclatinMappedFile(
                    mmap_object=None,
                    memory_address=None,
                    file_size=None,
                    local_metadata=local_metadata,
                    non_tensor_data=non_tensor_data,
                    tensor_infos=[]
                )
            else:
                logger.error(f"ECLATIN: [Rank {rank}] Main file not found: {eclatin_main_file}")
                return None, None
        else:
            # Load main file, extract Component 1 and Component 2 (metadata)
            mapped_file_own = FileSystemWriterAsync.load_eclatin_bytes_from_file(
                str(eclatin_main_file), my_rank=rank
            )
        
        meta_start_time = time()
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
        meta_end_time = time()
        meta_time = meta_end_time - meta_start_time
        logger.info(f"ECLATIN: [Rank {rank}] Metadata exchange completed in {meta_time:.2f}s")
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
                
                # Check if pre-allocated recovered buffer exists and is large enough
                if (self.eclatin_preallocated_recovered_buffer is not None and
                    self.eclatin_preallocated_recovered_buffer.numel() >= total_size):
                    # Reuse pre-allocated buffer (create view)
                    self.eclatin_recovered_buffer = self.eclatin_preallocated_recovered_buffer[:total_size]
                    logger.info(
                        f"ECLATIN: [Rank {rank}] Reusing pre-allocated recovered buffer: "
                        f"{total_size / (1024**3):.2f} GB / "
                        f"{self.eclatin_preallocated_recovered_buffer.numel() / (1024**3):.2f} GB"
                    )
                else:
                    # Allocate new buffer
                    pin_memory = torch.cuda.is_available() and getattr(self.eclatin_manager, 'eclatin_pin_memory', False)
                    self.eclatin_recovered_buffer = torch.empty(total_size, dtype=torch.uint8, pin_memory=pin_memory)
                    
                    # Store for future reuse
                    self.eclatin_preallocated_recovered_buffer = self.eclatin_recovered_buffer
                    
                    logger.info(
                        f"ECLATIN: [Rank {rank}] Allocated and cached recovered buffer: "
                        f"{total_size / (1024**3):.2f} GB"
                    )
        
        # ===== Step 6: Run recovery pipeline =====
        own_metadata = registry.rank_metadata.get(rank, [])
        total_size = sum(meta.size_bytes for meta in own_metadata)

        # Store mapped_file_own for software failure recovery (needed in early exit)
        self._eclatin_mapped_file_own = mapped_file_own

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
    
    def _load_ecnaive_block_checkpoint(self, checkpoint_dir: Path, sharded_state_dict: ShardedStateDict = None) -> Tuple:
        """Load EC-NAIVE checkpoint data and recover rank2.

        Similar to ECLATIN but simplified:
        - rank2: Receives 2 blocks (d_{3,1} from rank3, p_{0,0} from rank0)
        - rank2: Recovers d_{2,0} using XOR: d_{2,0} = d_{3,1} XOR p_{0,0}
        - rank0/3: Send their blocks to rank2

        Args:
            checkpoint_dir (Path): checkpoint directory
            sharded_state_dict (ShardedStateDict): sharded state dict for failed rank to derive metadata

        Returns:
            Tuple: (mapped_file_own, None) - placeholder for compatibility
        """
        from .filesystem_async import FileSystemWriterAsync
        from time import time

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        # Check for software failure mode
        from megatron.training import get_args
        input_args = get_args()
        if input_args.use_ecnaive_software_failure:
            failed_rank = 2
            logger.info("EC-NAIVE: Software failure recovery mode")

            from .state_dict_decomposer import GlobalMetadataRegistry

            # Get metadata (reuse existing logic)
            checkpoint_dir = Path(checkpoint_dir)

            # Load main file to extract metadata
            ecnaive_main_file = checkpoint_dir / f'__{rank}_0.distcp'

            # Handle failed rank that doesn't have checkpoint file
            if not ecnaive_main_file.exists():
                if rank == failed_rank:
                    logger.warning(f"EC-NAIVE: [Rank {rank}] Main file not found (failed node), deriving metadata from sharded_state_dict")
                    from .filesystem_async import EclatinMappedFile

                    if sharded_state_dict is not None:
                        local_metadata, non_tensor_data = self._derive_metadata_from_sharded_state_dict(sharded_state_dict, rank)
                        logger.info(f"EC-NAIVE: [Rank {rank}] Derived {len(local_metadata)} tensor metadata entries from sharded_state_dict")
                    else:
                        logger.warning(f"EC-NAIVE: [Rank {rank}] sharded_state_dict is None, using empty metadata")
                        local_metadata = []
                        non_tensor_data = {}

                    mapped_file_own = EclatinMappedFile(
                        mmap_object=None,
                        memory_address=None,
                        file_size=None,
                        local_metadata=local_metadata,
                        non_tensor_data=non_tensor_data,
                        tensor_infos=[]
                    )
                else:
                    logger.error(f"EC-NAIVE: [Rank {rank}] Main file not found: {ecnaive_main_file}")
                    return None, None
            else:
                # Load main file, extract Component 1 and Component 2 (metadata)
                mapped_file_own = FileSystemWriterAsync.load_ecnaive_bytes_from_file(
                    str(ecnaive_main_file), my_rank=rank
                )

            # Metadata exchange
            local_package = {
                'tensor_metadata': mapped_file_own.local_metadata or [],
                'non_tensor_data': mapped_file_own.non_tensor_data or {},
            }

            if local_package['tensor_metadata'] is None or local_package['non_tensor_data'] is None:
                logger.error(f"EC-NAIVE: [Rank {rank}] Local metadata is None, skipping metadata exchange")
                return mapped_file_own, None

            # All-gather complete metadata using all_gather_object
            all_metadata = [{}] * world_size
            if torch.distributed.is_initialized():
                torch.distributed.all_gather_object(all_metadata, local_package)

            # Build rank_metadata and rank_non_tensor_data dicts for GlobalMetadataRegistry
            rank_metadata = {}
            rank_non_tensor_data = {}
            for r in range(world_size):
                rank_metadata[r] = all_metadata[r]['tensor_metadata']
                rank_non_tensor_data[r] = all_metadata[r]['non_tensor_data']

            # Create registry with both tensor and non-tensor metadata
            registry = GlobalMetadataRegistry(
                rank_metadata=rank_metadata,
                rank_non_tensor_data=rank_non_tensor_data
            )

            # Software failure recovery: only rank 2 performs data transfer using existing connections
            if rank == 2:
                # Read local d20
                d20_path = checkpoint_dir / "__2_data0.distcp"
                with open(d20_path, 'rb') as f:
                    d20_data = f.read()
                d20_size = len(d20_data)

                # Receive d21
                d21_buffer = torch.zeros(d20_size, dtype=torch.uint8)
                d21_addr = d21_buffer.data_ptr()
                self.ecnaive_manager._ecnaive_native.software_recv_data1(d21_addr, d20_size)

                # Create merged buffer (align with hardware version)
                total_size = d20_size + d20_size
                self.ecnaive_recovered_buffer = torch.zeros(total_size, dtype=torch.uint8)

                # Data layout: first half d20, second half d21
                self.ecnaive_recovered_buffer[:d20_size] = torch.frombuffer(d20_data, dtype=torch.uint8)
                self.ecnaive_recovered_buffer[d20_size:] = d21_buffer

                # Save metadata (align with hardware version)
                self.ecnaive_recovered_metadata = mapped_file_own
                self.ecnaive_recovered_registry = registry

                logger.info(f"EC-NAIVE: [Rank 2] Software recovery completed: d20_size={d20_size}, total={total_size}")

            elif rank == 3:
                # Rank 3: Ready for software failure recovery but doesn't actively send
                # The connection is already established, rank 2 will initiate the transfer
                logger.info(f"EC-NAIVE: [Rank 3] Ready for software failure recovery data transfer")

            # All ranks return metadata (rank 2 will use recovered buffer, others use normal loading)
            return mapped_file_own, registry

        # If not software failure mode, proceed with normal hardware recovery
        # EC-NAIVE recovers rank2 (same as ECLATIN)
        failed_rank = 2
        
        checkpoint_dir = Path(checkpoint_dir)
        
        # ===== Step 0: Initialize EC-NAIVE load mode =====
        # This must be called before any load operations to set is_load_mode_ and rank_
        if self.ecnaive_manager.use_ecnaive:
            self.ecnaive_manager.init_ecnaive_load(rank, world_size)
            logger.info(f"EC-NAIVE: [Rank {rank}] Load mode initialized")
        
        # ===== Step 1: Load main file to extract metadata =====
        ecnaive_main_file = checkpoint_dir / f'__{rank}_0.distcp'
        
        # Handle failed rank that doesn't have checkpoint file
        if not ecnaive_main_file.exists():
            if rank == failed_rank:
                logger.warning(f"EC-NAIVE: [Rank {rank}] Main file not found (failed node), deriving metadata from sharded_state_dict")
                from .filesystem_async import EclatinMappedFile  # Reuse ECLATIN's mapped file structure
                
                # Derive metadata from sharded_state_dict
                if sharded_state_dict is not None:
                    local_metadata, non_tensor_data = self._derive_metadata_from_sharded_state_dict(sharded_state_dict, rank)
                    logger.info(f"EC-NAIVE: [Rank {rank}] Derived {len(local_metadata)} tensor metadata entries from sharded_state_dict")
                else:
                    logger.warning(f"EC-NAIVE: [Rank {rank}] sharded_state_dict is None, using empty metadata")
                    local_metadata = []
                    non_tensor_data = {}
                
                mapped_file_own = EclatinMappedFile(
                    mmap_object=None,
                    memory_address=None,
                    file_size=None,
                    local_metadata=local_metadata,
                    non_tensor_data=non_tensor_data,
                    tensor_infos=[]
                )
            else:
                logger.error(f"EC-NAIVE: [Rank {rank}] Main file not found: {ecnaive_main_file}")
                return None, None
        else:
            # Load main file, extract Component 1 and Component 2 (metadata)
            # EC-NAIVE uses ECNV magic number
            mapped_file_own = FileSystemWriterAsync.load_ecnaive_bytes_from_file(
                str(ecnaive_main_file), my_rank=rank
            )
        
        meta_start_time = time()
        # ===== Step 2: Metadata exchange (similar to ECLATIN) =====
        local_package = {
            'tensor_metadata': mapped_file_own.local_metadata or [],
            'non_tensor_data': mapped_file_own.non_tensor_data or {},
        }
        
        if local_package['tensor_metadata'] is None or local_package['non_tensor_data'] is None:
            logger.error(f"EC-NAIVE: [Rank {rank}] Local metadata is None, skipping metadata exchange")
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
        meta_end_time = time()
        meta_time = meta_end_time - meta_start_time
        logger.info(f"EC-NAIVE: [Rank {rank}] Metadata exchange completed in {meta_time:.2f}s")
        
        # ===== Step 3: Allocate 4 blocks uniformly (reuse save phase logic) =====
        if self.ecnaive_blocks is None:
            # Use the same _allocate_ecnaive_blocks method as save phase
            # This allocates 4 blocks based on registry metadata
            self.ecnaive_blocks = self._allocate_ecnaive_blocks(registry)
            logger.info(f"EC-NAIVE: [Rank {rank}] Allocated 4 blocks using registry metadata")
        
        # ===== Step 4: rank0/3 load block data from files =====
        if rank != 2:
            # rank0/3: Load block data from files into allocated blocks
            # rank1 doesn't participate in load mode
            if rank == 0 or rank == 3:
                self._load_ecnaive_blocks_from_files(checkpoint_dir, rank)
        
        # ===== Step 5: rank2 allocate recv buffers =====
        if rank == 2:
            if self.ecnaive_recv_buffers is None:
                self.ecnaive_recv_buffers = self.ecnaive_manager.allocate_ecnaive_load_recv_buffers(registry)
            
            if self.ecnaive_recovered_buffer is None:
                own_metadata = registry.rank_metadata.get(rank, [])
                total_size = sum(meta.size_bytes for meta in own_metadata)
                
                # Check if pre-allocated recovered buffer exists and is large enough
                if (self.ecnaive_preallocated_recovered_buffer is not None and
                    self.ecnaive_preallocated_recovered_buffer.numel() >= total_size):
                    # Reuse pre-allocated buffer (create view)
                    self.ecnaive_recovered_buffer = self.ecnaive_preallocated_recovered_buffer[:total_size]
                    logger.info(
                        f"EC-NAIVE: [Rank {rank}] Reusing pre-allocated recovered buffer: "
                        f"{total_size / (1024**3):.2f} GB / "
                        f"{self.ecnaive_preallocated_recovered_buffer.numel() / (1024**3):.2f} GB"
                    )
                else:
                    # Allocate new buffer
                    pin_memory = torch.cuda.is_available() and getattr(self.ecnaive_manager, 'ecnaive_pin_memory', False)
                    self.ecnaive_recovered_buffer = torch.empty(total_size, dtype=torch.uint8, pin_memory=pin_memory)
                    
                    # Store for future reuse
                    self.ecnaive_preallocated_recovered_buffer = self.ecnaive_recovered_buffer
                    
                    logger.info(
                        f"EC-NAIVE: [Rank {rank}] Allocated and cached recovered buffer: "
                        f"{total_size / (1024**3):.2f} GB"
                    )
        
        # ===== Step 6: Run recovery pipeline =====
        own_metadata = registry.rank_metadata.get(rank, [])
        total_size = sum(meta.size_bytes for meta in own_metadata)
        
        self._run_ecnaive_recovery_pipeline(
            rank=rank,
            world_size=world_size,
            registry=registry,
            ecnaive_blocks=self.ecnaive_blocks,
            recv_buffers=self.ecnaive_recv_buffers if rank == 2 else None,
            recovered_buffer=self.ecnaive_recovered_buffer if rank == 2 else None,
            total_size=total_size,
        )
        
        # ===== Step 7: rank2 save recovered buffer =====
        if rank == failed_rank:
            logger.info(f"EC-NAIVE: [Rank {rank}] Saving recovered buffer for _load_ecnaive_checkpoint")
            self.ecnaive_recovered_metadata = mapped_file_own
            self.ecnaive_recovered_registry = registry
        
        return mapped_file_own, None
    
    def _load_eclatin_layerwise_block_checkpoint(self, checkpoint_dir: Path, sharded_state_dict: ShardedStateDict = None) -> Tuple:
        """Load ECLATIN checkpoint data for layerwise recovery.
        
        Similar to _load_eclatin_block_checkpoint but uses layerwise recovery pipeline.
        
        Args:
            checkpoint_dir (Path): checkpoint directory
            sharded_state_dict (ShardedStateDict): sharded state dict for failed rank to derive metadata
        
        Returns:
            Tuple: (mapped_file_own, None) - placeholder for compatibility
        """
        from .filesystem_async import FileSystemWriterAsync
        from time import time
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # Determine failed_rank based on flags
        from megatron.training import get_args as use_args
        input_args = use_args()
        if input_args.use_eclatin_software_failure:
            failed_rank = 2  # rank2 software failure (can read local files)
            logger.info(f"ECLATIN: [Rank {rank}] Software failure recovery mode (failed_rank=2, reading local files)")
        else:
            failed_rank = 2  # Default: rank2 hardware failure (needs network recovery)
            logger.info(f"ECLATIN: [Rank {rank}] Hardware failure recovery mode (failed_rank=2, network recovery)")
        
        checkpoint_dir = Path(checkpoint_dir)
        
        # Store checkpoint_dir for software failure recovery
        self._current_checkpoint_dir = checkpoint_dir
        
        logger.info(f"ECLATIN Layerwise: [Rank {rank}] Loading block checkpoint for layerwise recovery")
        
        # ===== Step 1: Load main file to extract metadata =====
        eclatin_main_file = checkpoint_dir / f'__{rank}_0.distcp'
        
        # Handle failed rank that doesn't have checkpoint file
        if not eclatin_main_file.exists():
            if rank == failed_rank:
                logger.warning(f"ECLATIN Layerwise: [Rank {rank}] Main file not found (failed node), deriving metadata from sharded_state_dict")
                from .filesystem_async import EclatinMappedFile
                
                # Derive metadata from sharded_state_dict
                if sharded_state_dict is not None:
                    local_metadata, non_tensor_data = self._derive_metadata_from_sharded_state_dict(sharded_state_dict, rank)
                    logger.info(f"ECLATIN Layerwise: [Rank {rank}] Derived {len(local_metadata)} tensor metadata entries from sharded_state_dict")
                else:
                    logger.warning(f"ECLATIN Layerwise: [Rank {rank}] sharded_state_dict is None, using empty metadata")
                    local_metadata = []
                    non_tensor_data = {}
                
                mapped_file_own = EclatinMappedFile(
                    mmap_object=None,
                    memory_address=None,
                    file_size=None,
                    local_metadata=local_metadata,
                    non_tensor_data=non_tensor_data,
                    tensor_infos=[]
                )
            else:
                logger.error(f"ECLATIN Layerwise: [Rank {rank}] Main file not found: {eclatin_main_file}")
                return None, None
        else:
            # Load main file, extract Component 1 and Component 2 (metadata)
            mapped_file_own = FileSystemWriterAsync.load_eclatin_bytes_from_file(
                str(eclatin_main_file), my_rank=rank
            )
        
        # ===== Step 2: Metadata exchange =====
        meta_start_time = time()
        local_package = {
            'tensor_metadata': mapped_file_own.local_metadata or [],
            'non_tensor_data': mapped_file_own.non_tensor_data or {},
        }
        
        if local_package['tensor_metadata'] is None or local_package['non_tensor_data'] is None:
            logger.error(f"ECLATIN Layerwise: [Rank {rank}] Local metadata is None, skipping metadata exchange")
            return mapped_file_own, None
        
        # All-gather complete metadata
        all_packages = [None] * world_size
        torch.distributed.all_gather_object(all_packages, local_package)
        
        rank_metadata = {}
        rank_non_tensor_data = {}
        for i, package in enumerate(all_packages):
            rank_metadata[i] = package['tensor_metadata']
            rank_non_tensor_data[i] = package['non_tensor_data']
        
        # Create registry
        from .state_dict_decomposer import GlobalMetadataRegistry
        registry = GlobalMetadataRegistry(
            rank_metadata=rank_metadata,
            rank_non_tensor_data=rank_non_tensor_data
        )
        meta_end_time = time()
        logger.info(f"ECLATIN Layerwise: [Rank {rank}] Metadata exchange completed in {meta_end_time - meta_start_time:.2f}s")
        
        # ===== Step 3: Allocate 4 blocks =====
        if self.eclatin_blocks is None:
            self.eclatin_blocks = self._allocate_eclatin_blocks(registry)
            logger.info(f"ECLATIN Layerwise: [Rank {rank}] Allocated 4 blocks")
        
        # ===== Step 4: rank0/1/3 load block data from files =====
        if rank != 2:
            self._load_eclatin_blocks_from_files(checkpoint_dir, rank)
            logger.info(f"ECLATIN Layerwise: [Rank {rank}] Loaded blocks from files")
        
        # ===== Step 5: rank2 allocate recv buffers (per-layer) =====
        if rank == 2:
            if self.eclatin_recv_buffers is None:
                self.eclatin_recv_buffers = self._allocate_eclatin_load_recv_buffers(registry)
                logger.info(f"ECLATIN Layerwise: [Rank {rank}] Allocated recv buffers")
            
            if self.eclatin_recovered_buffer is None:
                own_metadata = registry.rank_metadata.get(rank, [])
                total_size = sum(meta.size_bytes for meta in own_metadata)
                
                # Check if pre-allocated recovered buffer exists and is large enough
                if (self.eclatin_preallocated_recovered_buffer is not None and
                    self.eclatin_preallocated_recovered_buffer.numel() >= total_size):
                    # Reuse pre-allocated buffer (create view)
                    self.eclatin_recovered_buffer = self.eclatin_preallocated_recovered_buffer[:total_size]
                    logger.info(
                        f"ECLATIN Layerwise: [Rank {rank}] Reusing pre-allocated recovered buffer: "
                        f"{total_size / (1024**3):.2f} GB / "
                        f"{self.eclatin_preallocated_recovered_buffer.numel() / (1024**3):.2f} GB"
                    )
                else:
                    # Allocate new buffer
                    pin_memory = torch.cuda.is_available() and getattr(self.eclatin_manager, 'eclatin_pin_memory', False)
                    self.eclatin_recovered_buffer = torch.empty(total_size, dtype=torch.uint8, pin_memory=pin_memory)
                    
                    # Store for future reuse
                    self.eclatin_preallocated_recovered_buffer = self.eclatin_recovered_buffer
                    
                    logger.info(
                        f"ECLATIN Layerwise: [Rank {rank}] Allocated and cached recovered buffer: "
                        f"{total_size / (1024**3):.2f} GB"
                    )
        
        # Store registry for later use in layerwise pipeline
        self.eclatin_recovered_metadata = mapped_file_own
        self.eclatin_recovered_registry = registry
        
        logger.info(f"ECLATIN Layerwise: [Rank {rank}] Block checkpoint loaded, ready for layerwise recovery")
        
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
    
    def _load_ecnaive_blocks_from_files(self, checkpoint_dir: Path, rank: int) -> None:
        """Load EC-NAIVE block data from files into allocated blocks.
        
        EC-NAIVE load mode:
        - rank0: Load data0 (p_{0,0} will be recalculated in recovery pipeline)
        - rank3: Load data0 (d_{3,1} will be recalculated in recovery pipeline)
        - rank1/2: No blocks to load (rank2 is failed, rank1 doesn't participate)
        
        Note: In save mode, rank0 sends p_{0,0} to rank2 and rank3 sends d_{3,1} to rank0,
        but neither rank saves these blocks directly. They need to be recalculated in the
        recovery pipeline from data0 and the second half of data (from main file).
        
        Args:
            checkpoint_dir (Path): checkpoint directory
            rank (int): current rank
        """
        if rank == 0:
            # rank0: Load data0 (needed for recalculating p_{0,0})
            # p_{0,0} will be recalculated in recovery pipeline from data0 and data1 (second half)
            logger.info(f"EC-NAIVE: [Rank 0] Loading data0 block from file")
            self._load_block_data_from_file(
                checkpoint_dir, rank, 'data0', self.ecnaive_blocks['data0']
            )
            logger.info(f"EC-NAIVE: [Rank 0] data0 loaded, p_{0,0} will be recalculated in recovery pipeline")
        
        elif rank == 3:
            # rank3: Load data0 (needed for recalculating d_{3,1})
            # d_{3,1} will be recalculated in recovery pipeline from data0 and the second half
            logger.info(f"EC-NAIVE: [Rank 3] Loading data0 block from file")
            self._load_block_data_from_file(
                checkpoint_dir, rank, 'data0', self.ecnaive_blocks['data0']
            )
            logger.info(f"EC-NAIVE: [Rank 3] data0 loaded, d_{3,1} will be recalculated in recovery pipeline")
        
        # rank1/2: No blocks to load
        # rank2 is failed, rank1 doesn't participate in load mode
    
    def _load_block_data_from_file(
        self, checkpoint_dir: Path, rank: int, block_name: str, block_tensor: torch.Tensor
    ) -> None:
        """Load block data from file into allocated block tensor using mmap.
        
        File format: __{rank}_{block_name}.distcp
        Only loads Component 3 (block data) into block_tensor.
        Uses mmap for zero-copy access, similar to EC-CHECK.
        Supports both ECLATIN (ECLT) and EC-NAIVE (ECNV) formats.
        
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
            raise FileNotFoundError(f"EC: Block file not found: {file_path}")
        
        logger.info(f"EC: [Rank {rank}] Loading {block_name} from {file_path} using mmap")
        
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
                raise RuntimeError(f"EC: Invalid file header (expected 32 bytes, got {len(header_bytes)})")
            
            magic, non_tensor_size, tensor_keys_size, tensor_buffer_size = struct.unpack('4sQQQ', header_bytes)
            
            # Support both ECLATIN (ECLT) and EC-NAIVE (ECNV) formats
            if magic not in (b'ECLT', b'ECNV'):
                raise RuntimeError(f"EC: Invalid magic number (expected b'ECLT' or b'ECNV', got {magic})")
            
            format_name = "ECLATIN" if magic == b'ECLT' else "EC-NAIVE"
            
            # Calculate Component 3 offset (skip Component 1 and Component 2)
            offset = 32 + non_tensor_size + tensor_keys_size
            
            # Read Component 3 (block data) directly from mmap into block_tensor
            # Note: block_tensor size should match tensor_buffer_size (aligned_half_block_size)
            expected_size = block_tensor.numel()
            if tensor_buffer_size > expected_size:
                logger.warning(
                    f"{format_name}: Block data size ({tensor_buffer_size}) > allocated size ({expected_size}), "
                    f"truncating to {expected_size}"
                )
                read_size = expected_size
            else:
                read_size = tensor_buffer_size
            
            # Read data directly from mmap (zero-copy numpy view)
            source_data = mm[offset:offset + read_size]
            if len(source_data) != read_size:
                raise RuntimeError(
                    f"{format_name}: Failed to read block data from mmap "
                    f"(expected {read_size} bytes, got {len(source_data)})"
                )
            
            # Copy to block_tensor using numpy (zero-copy from mmap)
            block_tensor_np = block_tensor.numpy()
            block_tensor_np[:read_size] = np.frombuffer(source_data, dtype=np.uint8)
            
            # Fill remaining with zeros if needed
            if read_size < expected_size:
                block_tensor_np[read_size:] = 0
            
            logger.debug(
                f"{format_name}: [Rank {rank}] Loaded {block_name} ({read_size / (1024**2):.2f} MB) "
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
            f"ECLATIN: [Load] Preparing 4 persistent blocks based on metadata\n"
            f"  Own data size: {own_total_size / (1024**3):.2f} GB (actual), "
            f"{max_total_bytes / (1024**3):.2f} GB (pipeline max), "
            f"{aligned_half_block_size / (1024**3):.2f} GB (aligned half block size)"
        )
        
        # ===== Check if pre-allocated buffers exist and are large enough =====
        block_names = ['data_block_1', 'data_block_2', 'parity_block_1', 'parity_block_2']
        
        if (self.eclatin_preallocated_blocks is not None and
            all(name in self.eclatin_preallocated_blocks for name in block_names) and
            all(self.eclatin_preallocated_blocks[name].numel() >= aligned_half_block_size for name in block_names)):
            # Reuse pre-allocated buffers (create views)
            blocks = {
                name: self.eclatin_preallocated_blocks[name][:aligned_half_block_size]
                for name in block_names
            }
            logger.info(
                f"ECLATIN: [Load] Reusing pre-allocated blocks: "
                f"{aligned_half_block_size / (1024**3):.2f} GB x 4 = "
                f"{4 * aligned_half_block_size / (1024**3):.2f} GB"
            )
            # Register buffers for RDMA if enabled (reused buffers should already be registered)
            # But we check and register if not already done
            if self.eclatin_manager.use_rdma:
                logger.info("ECLATIN: [Load] Verifying RDMA registration for reused blocks...")
                for block_name in block_names:
                    block_tensor = self.eclatin_preallocated_blocks[block_name]
                    # Check if already registered by looking at buffer address
                    buffer_addr = block_tensor.data_ptr()
                    if buffer_addr not in self.eclatin_manager.registered_buffers:
                        logger.info(f"ECLATIN: [Load] Registering reused block {block_name} for RDMA...")
                        self.eclatin_manager.register_buffer(block_tensor)
                logger.info("ECLATIN: [Load] RDMA registration verification complete")
        else:
            # Allocate new buffers
            pin_memory = torch.cuda.is_available() and getattr(self.eclatin_manager, 'eclatin_pin_memory', False)
            
            data_block_1 = torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=pin_memory)
            data_block_2 = torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=pin_memory)
            parity_block_1 = torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=pin_memory)
            parity_block_2 = torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=pin_memory)
            
            # Store for future reuse
            self.eclatin_preallocated_blocks = {
                'data_block_1': data_block_1,
                'data_block_2': data_block_2,
                'parity_block_1': parity_block_1,
                'parity_block_2': parity_block_2,
            }
            
            blocks = self.eclatin_preallocated_blocks
            
            logger.info(
                f"ECLATIN: [Load] Allocated and cached 4 persistent blocks:\n"
                f"  data_block_1: {aligned_half_block_size / (1024**3):.2f} GB\n"
                f"  data_block_2: {aligned_half_block_size / (1024**3):.2f} GB\n"
                f"  parity_block_1: {aligned_half_block_size / (1024**3):.2f} GB\n"
                f"  parity_block_2: {aligned_half_block_size / (1024**3):.2f} GB\n"
                f"  Total memory: {4 * aligned_half_block_size / (1024**3):.2f} GB"
            )
            
            # Register buffers for RDMA if enabled
            if self.eclatin_manager.use_rdma:
                logger.info("ECLATIN: [Load] Registering 4 persistent blocks for RDMA...")
                self.eclatin_manager.register_buffer(data_block_1)
                self.eclatin_manager.register_buffer(data_block_2)
                self.eclatin_manager.register_buffer(parity_block_1)
                self.eclatin_manager.register_buffer(parity_block_2)
                logger.info("ECLATIN: [Load] RDMA buffer registration complete")
        
        return blocks
    
    def _allocate_ecnaive_blocks(self, global_registry):
        """
        Allocate 4 persistent blocks for EC-NAIVE load.
        
        Similar to save phase but simplified - only allocates blocks without WriteBuckets.
        
        Args:
            global_registry: GlobalMetadataRegistry from all ranks
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'data0', 'recv_parity1', 
                                    'recv_parity0', 'recv_data1'
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
        ecnaive_buffer_size = self.ecnaive_manager.ecnaive_buffer_size
        # Each block only needs half of max_total_bytes (data is split into two halves)
        half_max_total_bytes = max_total_bytes // 2
        aligned_half_block_size = ((half_max_total_bytes + ecnaive_buffer_size - 1) // ecnaive_buffer_size) * ecnaive_buffer_size
        
        logger.info(
            f"EC-NAIVE: [Load] Preparing 4 persistent blocks based on metadata\n"
            f"  Own data size: {own_total_size / (1024**3):.2f} GB (actual), "
            f"{max_total_bytes / (1024**3):.2f} GB (pipeline max), "
            f"{aligned_half_block_size / (1024**3):.2f} GB (aligned half block size)"
        )
        
        # ===== Check if pre-allocated buffers exist and are large enough =====
        block_names = ['data0', 'recv_parity1', 'recv_parity0', 'recv_data1']
        
        if (self.ecnaive_preallocated_blocks is not None and
            all(name in self.ecnaive_preallocated_blocks for name in block_names) and
            all(self.ecnaive_preallocated_blocks[name].numel() >= aligned_half_block_size for name in block_names)):
            # Reuse pre-allocated buffers (create views)
            blocks = {
                name: self.ecnaive_preallocated_blocks[name][:aligned_half_block_size]
                for name in block_names
            }
            logger.info(
                f"EC-NAIVE: [Load] Reusing pre-allocated blocks: "
                f"{aligned_half_block_size / (1024**3):.2f} GB x 4 = "
                f"{4 * aligned_half_block_size / (1024**3):.2f} GB"
            )
        else:
            # Allocate new buffers
            pin_memory = torch.cuda.is_available() and getattr(self.ecnaive_manager, 'ecnaive_pin_memory', False)
            
            data0 = torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=pin_memory)
            recv_parity1 = torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=pin_memory)
            recv_parity0 = torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=pin_memory)
            recv_data1 = torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=pin_memory)
            
            # Store for future reuse
            self.ecnaive_preallocated_blocks = {
                'data0': data0,
                'recv_parity1': recv_parity1,
                'recv_parity0': recv_parity0,
                'recv_data1': recv_data1,
            }
            
            blocks = self.ecnaive_preallocated_blocks
            
            logger.info(
                f"EC-NAIVE: [Load] Allocated and cached 4 persistent blocks:\n"
                f"  data0: {aligned_half_block_size / (1024**3):.2f} GB\n"
                f"  recv_parity1: {aligned_half_block_size / (1024**3):.2f} GB\n"
                f"  recv_parity0: {aligned_half_block_size / (1024**3):.2f} GB\n"
                f"  recv_data1: {aligned_half_block_size / (1024**3):.2f} GB\n"
                f"  Total memory: {4 * aligned_half_block_size / (1024**3):.2f} GB"
            )
        
        return blocks
    
    def _allocate_eclatin_load_recv_buffers(self, global_registry) -> Dict[str, torch.Tensor]:
        """Allocate recv buffers for rank2 load recovery.
        
        Reuses pre-allocated buffers if available and large enough, otherwise allocates new ones.
        
        Args:
            global_registry: GlobalMetadataRegistry
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 6 recv buffers (rank2 only)
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        if rank != 2:
            return {}
        
        # Calculate required buffer size
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        max_total_bytes = 0
        for r in range(world_size):
            rank_metadata = global_registry.rank_metadata.get(r, [])
            rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
            max_total_bytes = max(max_total_bytes, rank_total_size)
        
        eclatin_buffer_size = self.eclatin_manager.eclatin_buffer_size
        half_max_total_bytes = max_total_bytes // 2
        aligned_half_block_size = ((half_max_total_bytes + eclatin_buffer_size - 1) // eclatin_buffer_size) * eclatin_buffer_size
        
        recv_buffer_names = ['rank0_data2', 'rank0_parity2', 'rank1_data1', 'rank1_parity1', 'rank3_data1', 'rank3_data2']
        
        # Check if pre-allocated buffers exist and are large enough
        if (self.eclatin_preallocated_recv_buffers is not None and
            all(name in self.eclatin_preallocated_recv_buffers for name in recv_buffer_names) and
            all(self.eclatin_preallocated_recv_buffers[name].numel() >= aligned_half_block_size for name in recv_buffer_names)):
            # Reuse pre-allocated buffers (create views)
            recv_buffers = {
                name: self.eclatin_preallocated_recv_buffers[name][:aligned_half_block_size]
                for name in recv_buffer_names
            }
            logger.info(
                f"ECLATIN: [Rank {rank}] Reusing pre-allocated recv buffers: "
                f"{aligned_half_block_size / (1024**3):.2f} GB x 6 = "
                f"{6 * aligned_half_block_size / (1024**3):.2f} GB"
            )
        else:
            # Allocate new buffers
            pin_memory = torch.cuda.is_available() and getattr(self.eclatin_manager, 'eclatin_pin_memory', False)
            
            recv_buffers = {
                name: torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=pin_memory)
                for name in recv_buffer_names
            }
            
            # Store for future reuse
            self.eclatin_preallocated_recv_buffers = recv_buffers
            
            logger.info(
                f"ECLATIN: [Rank {rank}] Allocated and cached 6 recv buffers:\n"
                f"  Buffer size: {aligned_half_block_size / (1024**3):.2f} GB each\n"
                f"  Total memory: {6 * aligned_half_block_size / (1024**3):.2f} GB"
            )
        
        return recv_buffers
    
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
        """Load checkpoint for rank2 failure recovery using ASIO/RDMA for data transfer (OPTIMIZED).
        
        This method uses Gemini's ASIO or RDMA-based communication for efficient data transfer
        between rank0 and rank2 during recovery. Optimized to minimize data copies.
        
        Optimizations:
        1. Zero-copy mmap: Direct send from mmap without intermediate copy
        2. Direct tensor receive: Receive into torch tensor, avoid numpy->bytes conversion
        3. Memoryview parsing: Use memoryview to avoid bytes slicing copies
        4. Shared memory: Use from_numpy without copy() when safe
        5. RDMA buffer registration: Register mmap and tensor buffers for zero-copy RDMA transfer
        
        Process:
        1. Rank0 reads replica file via mmap and sends directly (zero-copy)
        2. Rank0 sends data to rank2 using ASIO/RDMA (non-blocking, high-performance)
           - If RDMA enabled: Registers mmap buffers for zero-copy RDMA transfer
        3. Rank2 receives into torch tensor directly (zero-copy)
           - If RDMA enabled: Registers tensor buffers for zero-copy RDMA receive
        4. Both ranks restore their state_dict with minimal copies
        
        RDMA Support:
        - Automatically detects if RDMA is enabled via gemini_native module
        - Registers send/receive buffers for RDMA operations
        - Falls back to ASIO if RDMA registration fails
        - Unregisters buffers after transfer to free RDMA resources
        
        Args:
            sharded_state_dict: Sharded state dict template for loading
            checkpoint_dir: Checkpoint directory
            
        Returns:
            StateDict: Loaded state dict
        """
        import numpy as np
        
        rank = torch.distributed.get_rank()
        paired_rank = self.pairing_map.get(rank, None)
        checkpoint_dir = Path(checkpoint_dir)
        
        # Detect if RDMA is enabled
        from megatron.training import get_args as use_args
        input_args = use_args()
        use_rdma = input_args.use_rdma
        transport_mode = "RDMA" if use_rdma else "ASIO"
        logger.info(f"rank: {rank}, starting Gemini checkpoint recovery with {transport_mode} (OPTIMIZED) for rank2 failure")
        
        if use_rdma:
            logger.info(f"rank: {rank}, RDMA buffer registration enabled for load operations")
        
        # Ensure Gemini native module is initialized
        if self.gemini_manager._gemini_native is None:
            logger.warning(f"rank: {rank}, Gemini native module not initialized, falling back to standard recovery")
            return self._load_gemini_checkpoint_recovery(sharded_state_dict, checkpoint_dir)
        
        # Prepare RDMA send buffers for rank0 (if not already prepared or checkpoint_dir changed)
        # Note: If checkpoint_dir was provided during __init__, buffers are already prepared.
        # This is a fast check (string comparison only) if buffers are ready.
        # Only prepare if: (1) not initialized with checkpoint_dir, or (2) checkpoint_dir changed
        # if use_rdma and str(checkpoint_dir) != self.gemini_rdma_checkpoint_dir:
        #     self._prepare_gemini_rdma_buffers_if_needed(checkpoint_dir)
        
        # Only rank0 (pair_rank=2) and rank2 participate
        if rank == 0 and paired_rank == 2:
            # Rank0: Use pre-prepared RDMA buffers to send data to rank2
            logger.info(f"rank: {rank}, starting rank2 recovery - using pre-prepared {transport_mode} buffers")
            
            try:
                # Get buffer info from pre-prepared buffers
                if not self.gemini_rdma_send_buffers:
                    logger.error(f"rank: {rank}, RDMA send buffers not prepared")
                    raise RuntimeError("RDMA send buffers not prepared")
                
                replica_info = self.gemini_rdma_send_buffers.get('replica')
                own_info = self.gemini_rdma_send_buffers.get('own')
                
                if not replica_info or not own_info:
                    logger.error(f"rank: {rank}, incomplete RDMA send buffers")
                    raise RuntimeError("Incomplete RDMA send buffers")
                
                replica_file_size = replica_info['size']
                own_file_size = own_info['size']
                
                logger.info(
                    f"rank: {rank}, using pre-prepared buffers:\n"
                    f"  replica (rank2): {replica_file_size / (1024**2):.2f} MB (registered: {replica_info['registered']}, use_copy: {replica_info['use_copy']})\n"
                    f"  own (rank0): {own_file_size / (1024**2):.2f} MB (registered: {own_info['registered']}, use_copy: {own_info['use_copy']})"
                )
                send_start_time = time()
                # Step 2: Send metadata (both file sizes) to rank2
                metadata_array = np.array([replica_file_size, own_file_size], dtype=np.int64)
                metadata_addr = metadata_array.ctypes.data
                self.gemini_manager._gemini_native.send_buffer(metadata_addr, metadata_array.nbytes)
                logger.info(f"rank: {rank}, sent metadata (both file sizes) to rank2")
                
                # Step 3: Send replica data (rank2's backup) to rank2
                logger.info(f"rank: {rank}, sending replica data (rank2's backup) to rank2...")
                self.gemini_manager._gemini_native.send_buffer(replica_info['addr'], replica_file_size)
                logger.info(f"rank: {rank}, sent replica data to rank2: {replica_file_size / (1024**2):.2f} MB")
                
                # Step 4: Send own data (rank0's checkpoint) to rank2
                logger.info(f"rank: {rank}, sending own checkpoint data (for rank2 backup) to rank2...")
                self.gemini_manager._gemini_native.send_buffer(own_info['addr'], own_file_size)
                logger.info(f"rank: {rank}, sent own checkpoint data to rank2: {own_file_size / (1024**2):.2f} MB")
                send_end_time = time()
                send_time = send_end_time - send_start_time
                logger.info(f"rank: {rank}, all data sent successfully to rank2 in {send_time:.2f}s")
                
            except Exception as e:
                logger.error(f"rank: {rank}, {transport_mode} send failed: {e}", exc_info=True)
                raise
            
            # Step 5: Rank0 loads its own checkpoint from file
            logger.info(f"rank: {rank}, loading own checkpoint from saved file")
            loaded_state_dict = self._load_from_saved_checkpoint_file(sharded_state_dict, checkpoint_dir)
            return loaded_state_dict
            
        elif rank == 2:
            # Rank2: Receive metadata and all data from rank0, then restore
            logger.info(f"rank: {rank}, starting rank2 recovery - receiving data from rank0 via {transport_mode} (OPTIMIZED)")
            
            try:
                # Step 1: Receive metadata (both file sizes)
                start_time = time()
                metadata_buffer = np.zeros(2, dtype=np.int64)
                metadata_addr = metadata_buffer.ctypes.data
                self.gemini_manager._gemini_native.receive_buffer(metadata_addr, metadata_buffer.nbytes)
                
                replica_size = int(metadata_buffer[0])  # rank2's backup data size
                rank0_size = int(metadata_buffer[1])    # rank0's checkpoint data size
                
                logger.info(
                    f"rank: {rank}, received metadata from rank0:\n"
                    f"  replica (rank2 backup): {replica_size / (1024**2):.2f} MB\n"
                    f"  rank0 checkpoint: {rank0_size / (1024**2):.2f} MB"
                )
                
                # Step 2: Receive replica data (rank2's backup for recovery)
                # Note: Buffer is already registered for RDMA during initialization (_allocate_gemini_recovery_buffers)
                logger.info(f"rank: {rank}, receiving replica data (own backup) from rank0...")
                replica_tensor = self._get_gemini_recovery_buffer('replica', replica_size)
                
                replica_addr = replica_tensor.data_ptr()
                self.gemini_manager._gemini_native.receive_buffer(replica_addr, replica_size)
                logger.info(f"rank: {rank}, received replica data: {replica_size / (1024**2):.2f} MB")
                
                # Step 3: Receive rank0's checkpoint data (for rank2 backup)
                # Note: Buffer is already registered for RDMA during initialization (_allocate_gemini_recovery_buffers)
                logger.info(f"rank: {rank}, receiving rank0's checkpoint data (for backup)...")
                rank0_tensor = self._get_gemini_recovery_buffer('rank0', rank0_size)
                
                rank0_addr = rank0_tensor.data_ptr()
                self.gemini_manager._gemini_native.receive_buffer(rank0_addr, rank0_size)
                recv_time = time()
                logger.info(f"rank: {rank}, gemini asio received rank0 and rank2's checkpoint data time: {recv_time - start_time}")
                
                # Step 4: Save rank0's data as replica file (optional, for future recovery)
                # try:
                #     replica_filename = checkpoint_dir / f"__{rank}_0_replica0_rank{rank}.distcp"
                #     logger.info(f"rank: {rank}, saving rank0's backup to: {replica_filename}")
                    
                #     with open(replica_filename, 'wb') as f_replica:
                #         rank0_data_bytes = rank0_tensor.cpu().numpy().tobytes()
                #         f_replica.write(rank0_data_bytes)
                    
                #     logger.info(f"rank: {rank}, successfully saved rank0's backup data")
                # except Exception as e:
                #     logger.warning(f"rank: {rank}, failed to save rank0's backup data: {e}")
                #     # Continue - this is optional
                
                # Step 5: Parse and restore state_dict from replica data
                logger.info(f"rank: {rank}, parsing replica data for recovery...")
                
                # Check if this is Gemini optimized format
                use_gemini_optimized = False
                try:
                    from megatron.training import get_args
                    args = get_args()
                    use_gemini_optimized = getattr(args, 'use_gemini', False) and getattr(args, 'use_gemini_optimized', False)
                except:
                    pass
                
                if use_gemini_optimized and replica_size >= 8:
                    # Parse as Gemini optimized format
                    metadata_size_tensor = replica_tensor[:8]
                    metadata_size = int.from_bytes(metadata_size_tensor.cpu().numpy().tobytes(), byteorder='little')
                    
                    logger.info(f"rank: {rank}, parsing as Gemini optimized format, metadata_size: {metadata_size / 1024:.2f} KB")
                    
                    # Extract metadata
                    metadata_tensor = replica_tensor[8:8+metadata_size]
                    metadata_bytes = metadata_tensor.cpu().numpy().tobytes()
                    metadata_buffer_io = io.BytesIO(metadata_bytes)
                    gemini_metadata = torch.load(metadata_buffer_io, map_location='cpu', weights_only=False)
                    
                    # Extract buffer (zero-copy)
                    buffer_tensor = replica_tensor[8+metadata_size:]
                    
                    logger.info(
                        f"rank: {rank}, parsed Gemini data: "
                        f"metadata_size={metadata_size / 1024:.2f} KB, "
                        f"buffer_size={buffer_tensor.numel() / (1024**2):.2f} MB"
                    )
                    
                    # Create write_buckets structure
                    replica_buckets = [(
                        checkpoint_dir / f"__{rank}_0.distcp",
                        'gemini_optimized_local',
                        (
                            [('gemini_metadata', gemini_metadata), ('gemini_buffer', buffer_tensor)],
                            []
                        )
                    )]
                    
                    # Restore state_dict
                    logger.info(f"rank: {rank}, restoring state_dict from Gemini format...")
                    loaded_state_dict = self._restore_state_dict_from_gemini_format(
                        replica_buckets, sharded_state_dict
                    )
                    end_time = time()
                    logger.info(f"rank: {rank}, gemini asio recovery rank2's checkpoint data time: {end_time - start_time}")
                else:
                    # Standard pickle format
                    logger.info(f"rank: {rank}, parsing as standard pickle format")
                    replica_bytes = replica_tensor.cpu().numpy().tobytes()
                    replica_data_io = io.BytesIO(replica_bytes)
                    replica_buckets = torch.load(replica_data_io, weights_only=False)
                    
                    logger.info(f"rank: {rank}, restoring state_dict from replica data...")
                    loaded_state_dict = self._restore_state_dict_from_write_buckets(
                        replica_buckets, sharded_state_dict
                    )
                
                logger.info(f"rank: {rank}, gemini asio successfully restored state_dict and saved rank0's backup")
                return loaded_state_dict
                
            except Exception as e:
                logger.error(f"rank: {rank}, recovery failed: {e}", exc_info=True)
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
            # Rank0: Read both files first, then send metadata and data to rank2
            logger.info(f"rank: {rank}, starting rank2 recovery - reading both checkpoint files")
            
            # Step 1: Find both files
            replica_files = list(checkpoint_dir.glob(f"*_replica{paired_rank}_rank{rank}*.distcp"))
            if not replica_files:
                logger.error(f"rank: {rank}, no replica file found for rank2 recovery")
                raise FileNotFoundError(f"No replica file found for rank2 recovery")
            replica_file_path = replica_files[0]
            
            own_checkpoint_files = list(checkpoint_dir.glob(f"__{rank}_0.distcp"))
            if not own_checkpoint_files:
                logger.error(f"rank: {rank}, no own checkpoint file found")
                raise FileNotFoundError(f"No own checkpoint file found for rank {rank}")
            own_checkpoint_path = own_checkpoint_files[0]
            
            logger.info(f"rank: {rank}, found replica file: {replica_file_path}")
            logger.info(f"rank: {rank}, found own checkpoint file: {own_checkpoint_path}")
            
            # Step 2: Read both files
            try:
                with open(replica_file_path, 'rb') as f:
                    mm_replica = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    replica_file_size = len(mm_replica)
                    replica_data_bytes = mm_replica[:]
                    mm_replica.close()
                
                with open(own_checkpoint_path, 'rb') as f:
                    mm_own = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    own_file_size = len(mm_own)
                    own_data_bytes = mm_own[:]
                    mm_own.close()
                
                logger.info(
                    f"rank: {rank}, read both files:\n"
                    f"  replica (rank2): {replica_file_size / (1024**2):.2f} MB\n"
                    f"  own (rank0): {own_file_size / (1024**2):.2f} MB"
                )
            except Exception as e:
                logger.error(f"rank: {rank}, failed to read files: {e}", exc_info=True)
                raise
            
            # Step 3: Create pair process group
            pair_group = get_or_create_pair_process_group(rank, paired_rank)
            
            # Step 4: Send metadata (both file sizes) to rank2
            metadata_tensor = torch.tensor([replica_file_size, own_file_size], dtype=torch.int64)
            dummy_tensor = torch.zeros(2, dtype=torch.int64)
            
            size_list = [metadata_tensor, dummy_tensor]
            torch.distributed.all_gather(size_list, metadata_tensor, group=pair_group)
            
            logger.info(f"rank: {rank}, sent metadata (both file sizes) to rank2")
            
            # Step 5: Send replica data to rank2
            replica_np = np.frombuffer(replica_data_bytes, dtype=np.uint8).copy()
            replica_tensor = torch.from_numpy(replica_np)
            
            logger.info(f"rank: {rank}, sending replica data to rank2: {replica_tensor.numel() / (1024**2):.2f} MB")
            torch.distributed.broadcast(replica_tensor, src=rank, group=pair_group)
            logger.info(f"rank: {rank}, sent replica data to rank2")
            
            # Step 6: Send own data to rank2
            own_np = np.frombuffer(own_data_bytes, dtype=np.uint8).copy()
            own_tensor = torch.from_numpy(own_np)
            
            logger.info(f"rank: {rank}, sending own checkpoint data to rank2: {own_tensor.numel() / (1024**2):.2f} MB")
            torch.distributed.broadcast(own_tensor, src=rank, group=pair_group)
            logger.info(f"rank: {rank}, sent own checkpoint data to rank2")
            
            # Step 7: Rank0 loads its own checkpoint from file
            logger.info(f"rank: {rank}, loading own checkpoint from saved file")
            return self._load_from_saved_checkpoint_file(sharded_state_dict, checkpoint_dir)
        elif rank == 2:
            # Rank2: Receive metadata and all data from rank0, then restore
            logger.info(f"rank: {rank}, starting rank2 recovery - receiving data from rank0")
            
            # Step 1: Create pair process group
            pair_group = get_or_create_pair_process_group(rank, 0)
            
            data_start_time = time()
            # Step 2: Receive metadata (both file sizes)
            local_metadata_dummy = torch.zeros(2, dtype=torch.int64)
            remote_metadata = torch.zeros(2, dtype=torch.int64)
            
            metadata_list = [remote_metadata, local_metadata_dummy]
            torch.distributed.all_gather(metadata_list, local_metadata_dummy, group=pair_group)
            
            replica_size = int(metadata_list[0][0].item())
            rank0_size = int(metadata_list[0][1].item())
            
            logger.info(
                f"rank: {rank}, received metadata from rank0:\n"
                f"  replica (rank2 backup): {replica_size / (1024**2):.2f} MB\n"
                f"  rank0 checkpoint: {rank0_size / (1024**2):.2f} MB"
            )
            
            # Step 3: Receive replica data (rank2's backup for recovery)
            logger.info(f"rank: {rank}, receiving replica data (own backup) from rank0...")
            replica_tensor = self._get_gemini_recovery_buffer('replica', replica_size)
            torch.distributed.broadcast(replica_tensor, src=0, group=pair_group)
            logger.info(f"rank: {rank}, received replica data: {replica_size / (1024**2):.2f} MB")
            
            # Step 4: Receive rank0's checkpoint data (for rank2 backup)
            logger.info(f"rank: {rank}, receiving rank0's checkpoint data (for backup)...")
            rank0_tensor = self._get_gemini_recovery_buffer('rank0', rank0_size)
            torch.distributed.broadcast(rank0_tensor, src=0, group=pair_group)
            data_end_time = time()
            data_time = data_end_time - data_start_time
            logger.info(f"rank: {rank}, receive from rank0 checkpoint data in {data_time:.2f} seconds")

            # Step 5: Save rank0's data as replica file
            # try:
            #     replica_filename = checkpoint_dir / f"__{rank}_0_replica0_rank{rank}.distcp"
            #     logger.info(f"rank: {rank}, saving rank0's backup to: {replica_filename}")
                
            #     with open(replica_filename, 'wb') as f_replica:
            #         rank0_data_bytes = rank0_tensor.numpy().tobytes()
            #         f_replica.write(rank0_data_bytes)
                
            #     logger.info(f"rank: {rank}, successfully saved rank0's backup data")
            # except Exception as e:
            #     logger.warning(f"rank: {rank}, failed to save rank0's backup data: {e}")
                # Continue - this is optional
            
            # Step 6: Parse and restore state_dict from replica data
            logger.info(f"rank: {rank}, parsing replica data for recovery...")
            replica_bytes = replica_tensor.numpy().tobytes()
            
            # Check if this is Gemini optimized format
            use_gemini_optimized = False
            try:
                from megatron.training import get_args
                args = get_args()
                use_gemini_optimized = getattr(args, 'use_gemini', False) and getattr(args, 'use_gemini_optimized', False)
            except:
                pass
            
            if use_gemini_optimized and len(replica_bytes) >= 8:
                # Parse as Gemini optimized format
                metadata_size = int.from_bytes(replica_bytes[:8], byteorder='little')
                
                logger.info(f"rank: {rank}, parsing as Gemini optimized format, metadata_size: {metadata_size / 1024:.2f} KB")
                
                # Extract metadata
                metadata_bytes = replica_bytes[8:8+metadata_size]
                metadata_buffer = io.BytesIO(metadata_bytes)
                gemini_metadata = torch.load(metadata_buffer, map_location='cpu', weights_only=False)
                
                # Extract buffer
                buffer_bytes = replica_bytes[8+metadata_size:]
                buffer_np = np.frombuffer(buffer_bytes, dtype=np.uint8)
                gemini_buffer = torch.from_numpy(buffer_np.copy())
                
                logger.info(
                    f"rank: {rank}, parsed Gemini data: "
                    f"metadata_size={metadata_size / 1024:.2f} KB, "
                    f"buffer_size={len(buffer_bytes) / (1024**2):.2f} MB"
                )
                
                # Create write_buckets structure
                replica_buckets = [(
                    checkpoint_dir / f"__{rank}_0.distcp",
                    'gemini_optimized_local',
                    (
                        [('gemini_metadata', gemini_metadata), ('gemini_buffer', gemini_buffer)],
                        []
                    )
                )]
                
                # Restore state_dict
                logger.info(f"rank: {rank}, restoring state_dict from Gemini format...")
                loaded_state_dict = self._restore_state_dict_from_gemini_format(
                    replica_buckets, sharded_state_dict
                )
            else:
                deserialize_start_time = time()
                # Standard pickle format
                logger.info(f"rank: {rank}, parsing as standard pickle format")
                replica_data_io = io.BytesIO(replica_bytes)
                replica_buckets = torch.load(replica_data_io, weights_only=False)
                deserialize_end_time = time()
                logger.info(f"rank: {rank}, restoring state_dict from replica data...")
                deserialize_time = deserialize_end_time - deserialize_start_time
                logger.info(f"rank: {rank}, deserialize state_dict from replica data in {deserialize_time:.2f} seconds")
                loaded_state_dict = self._restore_state_dict_from_write_buckets(
                    replica_buckets, sharded_state_dict
                )
                recovery_end_time = time()
                recovery_time = recovery_end_time - deserialize_end_time
                logger.info(f"rank: {rank}, recovery state_dict from replica data in {recovery_time:.2f} seconds")
            logger.info(f"rank: {rank}, successfully restored state_dict and saved rank0's backup")
            return loaded_state_dict
    
    def _init_gemini_replicas_recovery_native(self, rank: int, world_size: int):
        """Initialize a temporary C++ ASIO module for recovery-specific topology.
        
        For rank2 recovery:
        - rank0 sends to rank2
        - rank1 sends to rank2
        - rank3 sends to rank2
        - rank2 receives from rank0, rank1, rank3
        
        Args:
            rank: Current rank
            world_size: Total number of ranks
            
        Returns:
            C++ native module instance for recovery
        """
        import os
        import socket
        
        logger.info(f"rank: {rank}, initializing Gemini Replicas recovery ASIO connections")
        
        # Load C++ module
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            import glob as _glob_module
            so_files = _glob_module.glob(os.path.join(current_dir, "gemini_replicas_native*.so"))
            
            if not so_files:
                raise RuntimeError(f"Gemini Replicas: No gemini_replicas_native.so file found in {current_dir}")
            
            import importlib.util as _importlib_util
            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location("gemini_replicas_native", so_path)
            gemini_replicas_native = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(gemini_replicas_native)
            logger.info(f"rank: {rank}, loaded gemini_replicas_native.so from {so_path}")
        except Exception as e:
            logger.error(f"rank: {rank}, failed to load gemini_replicas_native.so: {e}")
            raise
        
        # Get network configuration
        base_ip = os.environ.get('GEMINI_REPLICAS_BASE_IP')
        if not base_ip:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('8.8.8.8', 80))
                base_ip = s.getsockname()[0]
                s.close()
            except Exception:
                base_ip = os.environ.get('MASTER_ADDR', '127.0.0.1')
        
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(os.environ.get('GEMINI_REPLICAS_BASE_PORT', master_port + 30000))
        recovery_base_port = base_port + 10000  # Use different port range for recovery
        
        # Exchange IP addresses across all ranks via all_gather
        rank_ips = {}
        if torch.distributed.is_initialized():
            try:
                ip_list = [None] * world_size
                torch.distributed.all_gather_object(ip_list, base_ip)
                
                for r, ip in enumerate(ip_list):
                    rank_ips[r] = ip
                
                logger.info(
                    f"rank: {rank}, IP exchange completed - All rank IPs: {rank_ips}"
                )
            except Exception as e:
                logger.warning(
                    f"rank: {rank}, failed to exchange IPs via all_gather, using local IP: {e}"
                )
                for r in range(world_size):
                    rank_ips[r] = base_ip
        else:
            logger.warning(f"rank: {rank}, distributed not initialized, using local IP for all ranks")
            for r in range(world_size):
                rank_ips[r] = base_ip
        
        # Port allocation strategy for recovery:
        # rank2 listens on ONE port: recovery_base_port + 2
        # All senders (rank0, rank1, rank3) connect to the same port
        # The acceptor will handle multiple incoming connections sequentially
        
        rank2_recv_port = recovery_base_port + 2
        
        # Define recovery topology
        if rank in [0, 1, 3]:
            # Senders: send to rank2 only
            target_ranks = [2]
            target_ips = [rank_ips[2]]  # Use rank2's actual IP from all_gather
            # All senders connect to rank2's listening port
            target_ports = [rank2_recv_port]
            source_ranks = []  # Senders don't receive
            num_source_ranks = 0
            # Sender's recv port (not used but required by API)
            my_recv_port = recovery_base_port + rank * 10
        elif rank == 2:
            # Receiver: receive from rank0, rank1, rank3
            target_ranks = []  # Receiver doesn't send (or sends dummy)
            target_ips = []
            target_ports = []
            source_ranks = [0, 1, 3]
            num_source_ranks = 3
            # rank2 listens on a single port for all incoming connections
            my_recv_port = rank2_recv_port
        else:
            raise ValueError(f"rank: {rank}, invalid rank for recovery")
        
        my_ip = rank_ips[rank]  # Use this rank's IP from all_gather
        
        logger.info(
            f"rank: {rank}, recovery ASIO config:\n"
            f"  My IP: {my_ip}, My recv port: {my_recv_port}\n"
            f"  Target ranks: {target_ranks}, Target IPs: {target_ips}, Target ports: {target_ports}\n"
            f"  Source ranks: {source_ranks}, Num sources: {num_source_ranks}"
        )
        
        # Phase 1: All ranks create C++ instances and start acceptors
        logger.info(f"rank: {rank}, creating recovery C++ native module (Phase 1: acceptor)...")
        
        # Create C++ instance (this starts the acceptor for receiving connections)
        recovery_native = gemini_replicas_native.GeminiReplicasNative(
            rank, world_size,
            target_ranks,
            target_ips,
            target_ports,
            my_ip,
            my_recv_port,
            num_source_ranks
        )
        
        logger.info(f"rank: {rank}, acceptor started, waiting for all ranks to start acceptors...")
        
        # CRITICAL: Wait for ALL ranks to start their acceptors before anyone tries to connect
        torch.distributed.barrier()
        
        # Add a small delay to ensure acceptors are fully ready
        import time
        time.sleep(0.5)
        
        logger.info(f"rank: {rank}, all ranks ready, finalizing recovery connections (Phase 2: connect)...")
        
        # Phase 2: All ranks connect to their targets
        recovery_native.finalize_connections()
        
        logger.info(f"rank: {rank}, connections finalized, waiting for all ranks to complete...")
        
        # Wait for all connections to be established
        torch.distributed.barrier()
        
        logger.info(f"rank: {rank}, recovery ASIO connections initialized successfully")
        return recovery_native
    
    def _load_gemini_replicas_checkpoint_recovery(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Load checkpoint for Gemini Replicas hardware failure recovery (rank2 failure scenario).
        
        Recovery process when rank2 fails with 3 replicas and 4 ranks:
        
        Background - Normal Gemini Replicas topology:
        - rank0 targets: [0, 1, 2] -> rank0 sends its data to rank1 and rank2
        - rank1 targets: [1, 2, 3] -> rank1 sends its data to rank2 and rank3
        - rank2 targets: [2, 3, 0] -> rank2 sends its data to rank3 and rank0
        - rank3 targets: [3, 0, 1] -> rank3 sends its data to rank0 and rank1
        
        Saved files during normal checkpointing:
        - rank0 saves: __0_0.distcp (local), __0_0_replica2_rank0.distcp (from rank2), __0_0_replica3_rank0.distcp (from rank3)
        - rank1 saves: __1_0.distcp (local), __1_0_replica2_rank1.distcp (from rank2), __1_0_replica3_rank1.distcp (from rank3)
        - rank2 saves: __2_0.distcp (local), __2_0_replica0_rank2.distcp (from rank0), __2_0_replica1_rank2.distcp (from rank1)
        - rank3 saves: __3_0.distcp (local), __3_0_replica0_rank3.distcp (from rank0), __3_0_replica1_rank3.distcp (from rank1), __3_0_replica2_rank3.distcp (from rank2)
        
        Recovery when rank2 fails:
        - rank0: sends its local data (__0_0.distcp) to rank2 (rank0's own data, which rank2 had as backup)
        - rank1: sends its local data (__1_0.distcp) to rank2 (rank1's own data, which rank2 had as backup)
        - rank3: sends rank2's replica (__3_0_replica2_rank3.distcp) to rank2 (rank2's original data)
        - rank2: receives from rank0, rank1, rank3 and uses rank3's data to recover its own checkpoint
        
        Uses mmap for zero-copy file access and torch.distributed for network transfer.
        Data sizes are broadcasted first using all_gather, then point-to-point send/recv for actual data.
        
        Args:
            sharded_state_dict: Sharded state dict template for loading
            checkpoint_dir: Checkpoint directory
            
        Returns:
            StateDict: Loaded state dict
        """
        import mmap
        import numpy as np
        
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        checkpoint_dir = Path(checkpoint_dir)
        
        logger.info(f"rank: {rank}, starting Gemini Replicas checkpoint recovery for rank2 failure")
        
        # Calculate which ranks participate in recovery
        # rank0, rank1, rank3 are senders; rank2 is receiver
        participating_ranks = [0, 1, 2, 3]
        
        if rank not in participating_ranks:
            logger.info(f"rank: {rank}, not participating in rank2 recovery, loading from own checkpoint file")
            return self._load_from_saved_checkpoint_file(sharded_state_dict, checkpoint_dir)
        
        # For recovery, we need to set up special ASIO connections
        # Create a temporary C++ native module instance with recovery-specific topology
        recovery_native = self._init_gemini_replicas_recovery_native(rank, world_size)
        
        if rank in [0, 1, 3]:
            # Sender ranks: Read replica files and send to rank2
            logger.info(f"rank: {rank}, reading replica files for rank2 recovery")
            
            # Determine which file to read based on rank
            # rank0: sends own local data (rank2 was in rank0's target list [0,1,2])
            # rank1: sends own local data (rank2 was in rank1's target list [1,2,3])
            # rank3: sends rank2's data that was replicated to rank3 (rank3 was in rank2's target list [2,3,0])
            
            if rank == 3:
                # rank3 has rank2's replica data (because rank3 is in rank2's target list)
                # File pattern: __3_0_replica2_rank3.distcp
                replica_files = list(checkpoint_dir.glob(f"__{rank}_0_replica2_rank{rank}.distcp"))
                if not replica_files:
                    # Try alternative patterns
                    replica_files = list(checkpoint_dir.glob(f"*_replica2_rank{rank}*.distcp"))
            else:
                # rank0, rank1 send their own local data (not replicas)
                # Because rank2 was in their target lists, they send their primary data to rank2
                replica_files = list(checkpoint_dir.glob(f"__{rank}_0.distcp"))
            
            if not replica_files:
                logger.error(f"rank: {rank}, no replica file found for rank2 recovery")
                raise FileNotFoundError(f"No replica file found for rank2 recovery at rank {rank}")
            
            replica_file_path = replica_files[0]
            logger.info(f"rank: {rank}, found replica file: {replica_file_path}")
            
            # Step 1: Open file with mmap (zero-copy)
            try:
                f = open(replica_file_path, 'rb')
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                replica_file_size = len(mm)
                
                logger.info(f"rank: {rank}, opened replica file with mmap: {replica_file_size / (1024**2):.2f} MB")
            except Exception as e:
                logger.error(f"rank: {rank}, failed to open replica file: {e}", exc_info=True)
                raise
            
            # Step 2: Broadcast file sizes to all ranks using torch.distributed
            # (Still use torch.distributed for metadata sync, but ASIO for data)
            # Need to use gloo backend for CPU tensors when NCCL is the default backend
            from .async_utils import get_or_create_global_gloo_group
            global_gloo_group = get_or_create_global_gloo_group()
            
            size_tensor = torch.tensor([replica_file_size], dtype=torch.int64, device='cpu')
            
            # Gather all sizes at all ranks using gloo group
            all_sizes = [torch.zeros(1, dtype=torch.int64, device='cpu') for _ in range(world_size)]
            torch.distributed.all_gather(all_sizes, size_tensor, group=global_gloo_group)
            
            logger.info(f"rank: {rank}, broadcasted size to all ranks: {replica_file_size / (1024**2):.2f} MB")
            
            # Step 3: Send data to rank2 using C++ ASIO (zero-copy from mmap)
            try:
                # Create numpy view of mmap (zero-copy, read-only)
                mmap_np = np.frombuffer(mm, dtype=np.uint8)
                mmap_addr = mmap_np.ctypes.data
                
                logger.info(f"rank: {rank}, sending data to rank2 via C++ ASIO (zero-copy from mmap)...")
                logger.info(f"rank: {rank}, data size: {replica_file_size / (1024**2):.2f} MB")
                
                # Submit send buffer to C++ module
                recovery_native.submit_send_buffer(mmap_addr, replica_file_size)
                
                # Execute exchange (this will send data via ASIO)
                logger.info(f"rank: {rank}, executing ASIO exchange...")
                recovery_native.execute_exchange()
                
                logger.info(f"rank: {rank}, data sent successfully to rank2 via C++ ASIO")
                
            except Exception as e:
                logger.error(f"rank: {rank}, failed to send data via C++ ASIO: {e}", exc_info=True)
                raise
            finally:
                # Clean up mmap (keep numpy view alive until after send)
                del mmap_np
                mm.close()
                f.close()
            
            # Step 5: Load own checkpoint from file
            logger.info(f"rank: {rank}, loading own checkpoint from saved file")
            return self._load_from_saved_checkpoint_file(sharded_state_dict, checkpoint_dir)
            
        elif rank == 2:
            # Receiver rank: Receive data from rank0, rank1, rank3 and merge
            logger.info(f"rank: {rank}, receiving replica data from rank0, rank1, rank3 for recovery")
            
            try:
                # Step 1: Receive size information via broadcast
                # Need to use gloo backend for CPU tensors when NCCL is the default backend
                from .async_utils import get_or_create_global_gloo_group
                global_gloo_group = get_or_create_global_gloo_group()
                
                size_tensor = torch.zeros(1, dtype=torch.int64, device='cpu')
                all_sizes = [torch.zeros(1, dtype=torch.int64, device='cpu') for _ in range(world_size)]
                torch.distributed.all_gather(all_sizes, size_tensor, group=global_gloo_group)
                
                # Extract sizes from sender ranks
                rank0_size = int(all_sizes[0][0])
                rank1_size = int(all_sizes[1][0])
                rank3_size = int(all_sizes[3][0])
                
                logger.info(
                    f"rank: {rank}, received sizes via broadcast:\n"
                    f"  rank0: {rank0_size / (1024**2):.2f} MB\n"
                    f"  rank1: {rank1_size / (1024**2):.2f} MB\n"
                    f"  rank3: {rank3_size / (1024**2):.2f} MB"
                )
                
                # Step 2: Receive data from rank0, rank1, rank3 using C++ ASIO
                start_time = time()
                recv_buffers = {}
                source_ranks_to_recv = [0, 1, 3]
                
                logger.info(f"rank: {rank}, expecting data from source ranks: {source_ranks_to_recv}")
                
                # Use pre-allocated buffers for each source
                for src_rank in source_ranks_to_recv:
                    src_size = int(all_sizes[src_rank][0].item())
                    
                    # Get pre-allocated buffer or allocate dynamically
                    recv_buffer = self._get_gemini_replicas_recovery_buffer(src_rank, src_size)
                    recv_buffers[src_rank] = recv_buffer
                    
                    # Submit receive buffer to C++ module
                    recv_addr = recv_buffer.data_ptr()
                    recovery_native.submit_recv_buffer(src_rank, recv_addr, src_size)
                    
                    logger.info(f"rank: {rank}, submitted receive buffer for rank{src_rank}: {src_size / (1024**2):.2f} MB")
                
                # Step 3: Execute ASIO exchange (receive from all sources)
                # Submit a dummy send buffer (rank2 doesn't send, but API may require it)
                dummy_buffer = torch.zeros(8, dtype=torch.uint8)
                recovery_native.submit_send_buffer(dummy_buffer.data_ptr(), 8)
                
                logger.info(f"rank: {rank}, executing C++ ASIO exchange to receive from all sources...")
                recovery_native.execute_exchange()
                
                end_time = time() - start_time
                logger.info(f"rank: {rank}, data received successfully from all source ranks via C++ ASIO , use time {end_time:.2f}s")

                # Step 4: Parse received data and reconstruct state_dict
                # For Gemini Replicas recovery, we need to merge data from multiple sources
                # Typically rank3's data is the primary data, rank0 and rank1 are backups
                
                # Check if Gemini optimized format is used
                use_gemini_replicas_optimized = False
                try:
                    from megatron.training import get_args
                    args = get_args()
                    use_gemini_replicas_optimized = getattr(args, 'use_gemini_replicas_optimized', False)
                except:
                    pass
                
                # Use rank3's data as primary (rank3 has rank2's original data)
                primary_rank = 3
                if primary_rank in recv_buffers:
                    primary_buffer = recv_buffers[primary_rank]
                    logger.info(f"rank: {rank}, using rank{primary_rank}'s data as primary for recovery")
                    
                    if use_gemini_replicas_optimized and primary_buffer.numel() >= 8:
                        # Parse as Gemini optimized format
                        metadata_size_tensor = primary_buffer[:8]
                        metadata_size = int.from_bytes(metadata_size_tensor.cpu().numpy().tobytes(), byteorder='little')
                        
                        logger.info(f"rank: {rank}, parsing as Gemini Replicas optimized format, metadata_size: {metadata_size / 1024:.2f} KB")
                        
                        # Extract metadata (Gemini Replicas uses pickle)
                        metadata_tensor = primary_buffer[8:8+metadata_size]
                        metadata_bytes = metadata_tensor.cpu().numpy().tobytes()
                        import pickle
                        import io
                        
                        # Create custom Unpickler to handle torch objects with persistent IDs
                        class TorchUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                # Handle torch classes normally
                                if module == 'torch':
                                    return getattr(torch, name)
                                return super().find_class(module, name)
                            
                            def persistent_load(self, pid):
                                # Handle persistent IDs for torch objects
                                logger.debug(f"rank: {rank}, persistent_load called with pid: {pid}")
                                
                                if isinstance(pid, tuple):
                                    typename = pid[0] if len(pid) > 0 else None
                                    if typename and 'Storage' in typename:
                                        raise pickle.UnpicklingError(
                                            f"Unexpected storage object in metadata: {pid}"
                                        )
                                
                                logger.warning(f"rank: {rank}, unhandled persistent_load pid: {pid}, returning as-is")
                                return pid
                        
                        try:
                            metadata_buffer = io.BytesIO(metadata_bytes)
                            unpickler = TorchUnpickler(metadata_buffer)
                            gemini_metadata = unpickler.load()
                        except Exception as e:
                            logger.error(f"rank: {rank}, failed to unpickle metadata: {e}")
                            # Fallback: try torch.load
                            metadata_buffer = io.BytesIO(metadata_bytes)
                            gemini_metadata = torch.load(metadata_buffer, map_location='cpu', weights_only=False)
                        
                        # Extract buffer (zero-copy)
                        buffer_tensor = primary_buffer[8+metadata_size:]
                        
                        logger.info(
                            f"rank: {rank}, parsed Gemini Replicas data: "
                            f"metadata_size={metadata_size / 1024:.2f} KB, "
                            f"buffer_size={buffer_tensor.numel() / (1024**2):.2f} MB"
                        )
                        
                        # Create write_buckets structure
                        replica_buckets = [(
                            checkpoint_dir / f"__{rank}_0.distcp",
                            'gemini_optimized_local',
                            (
                                [('gemini_metadata', gemini_metadata), ('gemini_buffer', buffer_tensor)],
                                []
                            )
                        )]
                        
                        # Restore state_dict
                        logger.info(f"rank: {rank}, restoring state_dict from Gemini Replicas format...")
                        loaded_state_dict = self._restore_state_dict_from_gemini_format(
                            replica_buckets, sharded_state_dict
                        )
                    else:
                        # Standard pickle format
                        logger.info(f"rank: {rank}, parsing as standard pickle format")
                        primary_bytes = primary_buffer.cpu().numpy().tobytes()
                        primary_data_io = io.BytesIO(primary_bytes)
                        replica_buckets = torch.load(primary_data_io, weights_only=False)
                        
                        logger.info(f"rank: {rank}, deserialized replica data, restoring state_dict...")
                        loaded_state_dict = self._restore_state_dict_from_write_buckets(
                            replica_buckets, sharded_state_dict
                        )
                    
                    logger.info(f"rank: {rank}, successfully restored state_dict from Gemini Replicas recovery")
                    return loaded_state_dict
                else:
                    raise RuntimeError(f"rank: {rank}, primary rank {primary_rank} not in received buffers")
                    
            except Exception as e:
                logger.error(f"rank: {rank}, Gemini Replicas recovery failed: {e}", exc_info=True)
                raise
        else:
            # Should not reach here
            logger.error(f"rank: {rank}, unexpected rank in Gemini Replicas recovery")
            raise RuntimeError(f"Unexpected rank {rank} in Gemini Replicas recovery")
    
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
            use_gemini_replicas_optimized = False
            try:
                from megatron.training import get_args
                args = get_args()
                use_gemini_optimized = getattr(args, 'use_gemini', False) and getattr(args, 'use_gemini_optimized', False)
                use_gemini_replicas_optimized = getattr(args, 'use_gemini_replicas', False) and getattr(args, 'use_gemini_replicas_optimized', False)
                
                if use_gemini_optimized:
                    logger.info(f"rank: {rank}, using Gemini optimized format (from args flags)")
                elif use_gemini_replicas_optimized:
                    logger.info(f"rank: {rank}, using Gemini Replicas optimized format (from args flags)")
            except Exception as e:
                # Args not available, will auto-detect format
                logger.debug(f"rank: {rank}, cannot access args, will auto-detect format: {e}")
            
            # Parse checkpoint data based on format (OPTIMIZED)
            # Support both Gemini and Gemini Replicas optimized formats
            use_optimized_format = use_gemini_optimized or use_gemini_replicas_optimized
            
            if use_optimized_format or (not use_optimized_format and checkpoint_file_size >= 8):
                # Try optimized format first (if flag is set, or auto-detect)
                if checkpoint_file_size >= 8:
                    # Read only the header (8 bytes) to determine format
                    metadata_size = int.from_bytes(mm[:8], byteorder='little')
                    
                    # Sanity check: metadata_size should be reasonable (< 10MB for metadata)
                    is_valid_optimized = (0 < metadata_size < 10 * 1024 * 1024 and 
                                         (8 + metadata_size) <= checkpoint_file_size)
                    
                    if use_optimized_format or is_valid_optimized:
                        # Parse as optimized format (Gemini or Gemini Replicas) - ZERO-COPY
                        format_name = "Gemini Replicas" if use_gemini_replicas_optimized else "Gemini"
                        logger.info(f"rank: {rank}, parsing as {format_name} optimized format (zero-copy), metadata_size: {metadata_size / 1024:.2f} KB")
                        
                        # Extract metadata (only copy small metadata portion)
                        metadata_bytes = mm[8:8+metadata_size]
                        
                        # Gemini Replicas uses pickle, Gemini uses torch.save
                        if use_gemini_replicas_optimized:
                            # import pickle
                            # gemini_metadata = pickle.loads(metadata_bytes)
                            # Use torch.load instead of pickle.loads to properly handle torch objects
                            # torch.load internally provides persistent_load for torch.dtype, torch.device, etc.
                            metadata_buffer = io.BytesIO(metadata_bytes)
                            gemini_metadata = torch.load(metadata_buffer, map_location='cpu', weights_only=False)
                        else:
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
                            f"rank: {rank}, loaded {format_name} checkpoint (OPTIMIZED): "
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
                        
                        # Use appropriate storage key based on format
                        storage_key = 'gemini_replicas_optimized_local' if use_gemini_replicas_optimized else 'gemini_optimized_local'
                        
                        write_bucket = WriteBucketWithRefs(
                            checkpoint_file_path,
                            storage_key,
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
        failed_rank: int = 2,  # Add failed_rank parameter, default to 2 for backward compatibility
    ) -> None:
        """EC-CHECK recovery pipeline for single-failure scenario.

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
        # Use failed_rank parameter passed from caller
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
        logger.info(f"EC-CHECK: load pipeline total_bytes: {total_bytes}")
        # === Step 5: Determine data source based on rank role and failed_rank ===
        # For rank2 recovery scenario:
        #   - rank0/3: read from mapped_file_own (their own data/parity)
        #   - rank1/2: read from mapped_file_partner (received from step2)
        # For rank1 recovery scenario:
        #   - rank0: read from mapped_file_own (d0 data to send to rank1)
        #   - rank1: will receive d0 from rank0 via ASIO (no file read needed)
        #   - rank2/3: read from mapped_file_own (their own data)
        if failed_rank == 1:
            # rank1 software failure: rank0 sends d0 to rank1
            if rank == 0:
                source_mmap = mapped_file_own.mmap_object if mapped_file_own.mmap_object is not None else None
                source_file_size = mapped_file_own.file_size if mapped_file_own.file_size is not None else 0
            elif rank == 1:
                # rank1 doesn't read from file, will receive from rank0 via ASIO
                source_mmap = None
                source_file_size = 0
            else:
                # rank2/3 read from own file
                source_mmap = mapped_file_own.mmap_object if mapped_file_own.mmap_object is not None else None
                source_file_size = mapped_file_own.file_size if mapped_file_own.file_size is not None else 0
        else:
            # rank2 hardware failure (original logic)
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
        
        # === Step 7: Early exit for rank1 software failure - simple P2P transfer (no encoding/XOR buffers needed) ===
        # rank1 software failure recovery only needs simple P2P transfer, no encoding/XOR pipeline needed
        if failed_rank == 1:
            logger.info(f"EC-CHECK: [Rank {rank}] rank1 software failure recovery - using simple synchronous P2P send/recv (no worker queue)")
            
            if rank == 0:
                # Rank0: Send d0 directly to rank1 via simple synchronous P2P send
                logger.info(f"EC-CHECK: [Rank {rank}] Sending d0 to rank1 via simple synchronous P2P send (one-time transfer)")
                
                # Read d0 from own file and send all at once via simple P2P send
                if source_mmap is not None:
                    # Get rank0's own data size from registry (to match rank1's recv_total_size calculation)
                    rank0_metadata = registry.rank_metadata.get(rank, [])
                    send_total_size = sum(meta.size_bytes for meta in rank0_metadata)
                    
                    # Calculate tensor buffer size and offset from file header
                    header_bytes = source_mmap[:32]
                    import struct
                    magic, non_tensor_size, tensor_keys_size, tensor_buffer_size = struct.unpack('4sQQQ', header_bytes)
                    tensor_buffer_start_offset = 32 + non_tensor_size + tensor_keys_size
                    
                    # Verify that registry size matches file header size
                    if send_total_size != tensor_buffer_size:
                        logger.warning(
                            f"EC-CHECK: [Rank {rank}] Size mismatch: registry={send_total_size}, "
                            f"file_header={tensor_buffer_size}, using registry size to match rank1's recv_total_size"
                        )
                    
                    # Allocate buffer for send_total_size data (rank0's own data size)
                    send_buffer = torch.empty(send_total_size, dtype=torch.uint8)
                    send_addr = int(send_buffer.data_ptr())
                    
                    # Copy all data from mmap to buffer at once
                    chunk_data = source_mmap[tensor_buffer_start_offset:tensor_buffer_start_offset + send_total_size]
                    import ctypes
                    ctypes.memmove(
                        ctypes.cast(send_addr, ctypes.POINTER(ctypes.c_uint8)),
                        chunk_data,
                        send_total_size
                    )
                    
                    # Send all data at once via simple synchronous P2P send (no worker queue)
                    logger.info(f"EC-CHECK: [Rank {rank}] Sending {send_total_size / (1024**2):.2f} MB in one transfer")
                    self.eccheck_manager._eccheck_native.simple_p2p_send(
                        buffer_addr=send_addr,
                        size=send_total_size
                    )
                    
                    logger.info(f"EC-CHECK: [Rank {rank}] Finished sending d0 to rank1: {send_total_size / (1024**2):.2f} MB")
                else:
                    logger.error(f"EC-CHECK: [Rank {rank}] No source mmap available for sending d0")
                    
            elif rank == 1:
                # Rank1: Receive d0 from rank0 via simple synchronous P2P recv
                logger.info(f"EC-CHECK: [Rank {rank}] Receiving d0 from rank0 via simple synchronous P2P recv (one-time transfer)")
                
                # Receive all data at once directly into recv_own_buffer
                recv_addr = int(recv_own_buffer.data_ptr())
                
                # Receive all data at once via simple synchronous P2P recv (no worker queue)
                logger.info(f"EC-CHECK: [Rank {rank}] Receiving {recv_total_size / (1024**2):.2f} MB in one transfer")
                self.eccheck_manager._eccheck_native.simple_p2p_recv(
                    buffer_addr=recv_addr,
                    size=recv_total_size
                )
                
                logger.info(f"EC-CHECK: [Rank {rank}] Finished receiving d0 from rank0: {recv_total_size / (1024**2):.2f} MB")
                
            else:
                # rank2/3: No action needed for rank1 recovery
                logger.info(f"EC-CHECK: [Rank {rank}] No action needed for rank1 recovery")
            
            # Synchronize all ranks and return early (skip full encoding/XOR pipeline)
            # if torch.distributed.is_initialized():
            #     torch.distributed.barrier()
            #     logger.info(f"EC-CHECK: [Rank {rank}] Synchronized after simple P2P transfer")
            
            logger.info(f"EC-CHECK: [Rank {rank}] rank1 software failure recovery completed (simple synchronous P2P, no worker queue)")
            return
        
        # === Step 7: Reset encoding completion flags and activate buffer poller (for rank2 hardware failure) ===
        self.eccheck_manager._eccheck_native.reset_encoding_completion_flags()
        if mgr._buffer_poller_active_event:
            mgr._buffer_poller_active_event.set()
            logger.info("EC-CHECK: Activated buffer poller for load pipeline")
        
        try:
            # === Step 8: Main pipeline loop (for rank2 hardware failure recovery) ===
            processed = 0
            #total_bytes = 20 * eccheck_buffer_size
            while processed < total_bytes:
                take = min(eccheck_buffer_size, total_bytes - processed)
                
                # Get free buffers
                cur_buffer_addr = get_free_data_buffer()
                # Load mode only needs thread2 encoding buffer (parity index 1)
                enc_addr2 = get_free_encoding_buffer()
                # Load mode parity buffer allocation based on failed_rank
                if failed_rank == 1:
                    # rank1 recovery: only rank1 needs to receive, no parity needed
                    if rank == 1:
                        parity_addr2 = 0  # rank1 receives d0 directly, no XOR needed
                    else:
                        parity_addr2 = 0
                else:
                    # rank2 recovery: rank2/3 need parity buffer
                    if rank == 2 or rank == 3:
                        parity_addr2 = get_free_parity_buffer()
                    else:
                        parity_addr2 = 0
                
                # Calculate recv addresses (64-byte aligned) - based on failed_rank
                if failed_rank == 1:
                    # rank1 recovery: only rank1 receives
                    if rank == 1:
                        recv_buffer_offset_thread2_aligned = ((recv_buffer_offset_thread2 + 63) // 64) * 64
                        recv_addr_thread2 = recv_buffer_base_addr_thread2 + recv_buffer_offset_thread2_aligned
                        recv_chunk_size = take
                        recv_buffer_offset_thread2 = recv_buffer_offset_thread2_aligned + recv_chunk_size
                    else:
                        recv_addr_thread2 = 0
                        recv_chunk_size = 0
                elif rank == 2 or rank == 3:
                    recv_buffer_offset_thread2_aligned = ((recv_buffer_offset_thread2 + 63) // 64) * 64
                    recv_addr_thread2 = recv_buffer_base_addr_thread2 + recv_buffer_offset_thread2_aligned
                    recv_chunk_size = take
                    recv_buffer_offset_thread2 = recv_buffer_offset_thread2_aligned + recv_chunk_size
                else:
                    recv_addr_thread2 = 0
                    recv_chunk_size = 0
                
                # Calculate P2P write addresses for Step6 (based on failed_rank)
                if failed_rank == 1:
                    # rank1 recovery: rank1 needs partner_buffer for receiving d0
                    if rank == 1 and p2p_partner_buffer_base_addr != 0:
                        p2p_partner_buffer_offset_aligned = ((p2p_partner_buffer_offset + 63) // 64) * 64
                        p2p_partner_write_addr = p2p_partner_buffer_base_addr + p2p_partner_buffer_offset_aligned
                        p2p_partner_buffer_offset = p2p_partner_buffer_offset_aligned + take
                    else:
                        p2p_partner_write_addr = 0
                elif rank == 2 and p2p_partner_buffer_base_addr != 0:
                    p2p_partner_buffer_offset_aligned = ((p2p_partner_buffer_offset + 63) // 64) * 64
                    p2p_partner_write_addr = p2p_partner_buffer_base_addr + p2p_partner_buffer_offset_aligned
                    p2p_partner_buffer_offset = p2p_partner_buffer_offset_aligned + take
                else:
                    p2p_partner_write_addr = 0  # Not needed for rank0/1/3
                
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
                        #         ctypes.memset(padding_ptr, 0, padding_size)
                        # else:
                        #     # Past actual data, fill with zeros
                        #     ctypes.memset(buffer_array.contents, 0, take)
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
                                # ctypes.memset(padding_ptr, 0, padding_size)
                        # else:
                        #     # Past actual data, fill with zeros
                        #     ctypes.memset(buffer_array.contents, 0, take)
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
                # Simplified interface: only pass necessary parameters based on rank
                if rank == 0:
                    # rank0: Step2 send, then encoding and send
                    # Only need thread2 encoding buffer (parity index 1)
                    self.eccheck_manager._eccheck_native.submit_load_pipeline_chunk(
                        step2_send_addr=step2_send_addr,
                        step2_recv_data_addr=0,
                        step2_size=step2_size,
                        data_addr=cur_buffer_addr,
                        size=take,
                        encoding_addr=enc_addr2,  # Only thread2 encoding buffer
                        recv_addr=0,              # Sender doesn't need recv
                        recv_chunk_size=0,
                        parity_addr=0,            # Sender doesn't need parity buffer
                        p2p_partner_write_addr=0   # Not needed for sender
                    )
                elif rank == 1:
                    # rank1: Step2 recv, then encoding and send
                    self.eccheck_manager._eccheck_native.submit_load_pipeline_chunk(
                        step2_send_addr=0,
                        step2_recv_data_addr=step2_recv_data_addr,
                        step2_size=step2_size,
                        data_addr=cur_buffer_addr,
                        size=take,
                        encoding_addr=enc_addr2,  # Only thread2 encoding buffer
                        recv_addr=0,
                        recv_chunk_size=0,
                        parity_addr=0,
                        p2p_partner_write_addr=0  # Not needed for rank1
                    )
                elif rank == 2:
                    # rank2: Step2 recv, then encoding, receive rank0's encoding and do XOR
                    self.eccheck_manager._eccheck_native.submit_load_pipeline_chunk(
                        step2_send_addr=0,
                        step2_recv_data_addr=step2_recv_data_addr,
                        step2_size=step2_size,
                        data_addr=cur_buffer_addr,
                        size=take,
                        encoding_addr=enc_addr2,  # Only thread2 encoding buffer
                        recv_addr=recv_addr_thread2,  # Receive rank0's encoding
                        recv_chunk_size=recv_chunk_size,
                        parity_addr=parity_addr2,     # Store XOR result d2
                        p2p_partner_write_addr=p2p_partner_write_addr  # For Step6: receive d3
                    )
                elif rank == 3:
                    # rank3: Step2 send, then encoding, receive rank1's encoding and do XOR
                    self.eccheck_manager._eccheck_native.submit_load_pipeline_chunk(
                        step2_send_addr=step2_send_addr,
                        step2_recv_data_addr=0,
                        step2_size=step2_size,
                        data_addr=cur_buffer_addr,
                        size=take,
                        encoding_addr=enc_addr2,  # Only thread2 encoding buffer
                        recv_addr=recv_addr_thread2,  # Receive rank1's encoding
                        recv_chunk_size=recv_chunk_size,
                        parity_addr=parity_addr2,     # Store XOR result d3
                        p2p_partner_write_addr=0      # Not needed for rank3
                )
                
                processed += take
            
            # === Step 8: Send sentinel and wait for completion ===
            logger.info("EC-CHECK: Load pipeline: Sending sentinel to load encoder worker")
            # All ranks submit sentinel to load encoder worker
            self.eccheck_manager._eccheck_native.submit_load_encoding_sentinel()
            
            logger.info("EC-CHECK: Load pipeline: Waiting for XOR worker to complete (Step6 tasks will be submitted)...")
            # Wait for XOR worker to complete (which will submit all Step6 tasks)
            # Note: wait_for_encoding_completion waits for all workers including Step6,
            # so we need to wait twice: first for XOR, then send Step6 sentinel, then wait again
            import time
            while True:
                # Check if XOR worker is completed (but not Step6 workers yet)
                # We'll use a simple polling approach: wait a bit, then check
                time.sleep(0.1)
                # Try to wait, but this will wait for all workers including Step6
                # So we'll just wait once and send Step6 sentinel before the final wait
                break
            
            # Wait for XOR to complete (all Step6 tasks should be submitted by now)
            # Note: This will also wait for Step6, but Step6 sentinel hasn't been sent yet
            # So we need to send Step6 sentinel first, then wait
            # Send sentinel to Step6 P2P workers based on failed_rank
            if failed_rank == 1:
                # rank1 recovery: only rank1 sends sentinel
                if rank == 1:
                    logger.info("EC-CHECK: Load pipeline: Sending sentinel to Step6 P2P workers (rank1 recovery)")
                    self.eccheck_manager._eccheck_native.submit_load_step6_p2p_sentinel()
            elif rank == 2 or rank == 3:
                # rank2 recovery: rank2/3 send sentinel
                logger.info("EC-CHECK: Load pipeline: Sending sentinel to Step6 P2P workers (rank2 recovery)")
                self.eccheck_manager._eccheck_native.submit_load_step6_p2p_sentinel()
            
            logger.info("EC-CHECK: Load pipeline: Waiting for all load workers to complete...")
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
        
        # === Step 0: Check for software failure mode ===
        from megatron.training import get_args as use_args
        from time import time
        from pathlib import Path
        input_args = use_args()
        failed_rank = 2  # ECLATIN recovers rank2
        
        # Early exit for rank2 software failure - read local files directly (no network/XOR needed)
        if input_args.use_eclatin_software_failure:
            if rank == failed_rank:
                logger.info(f"ECLATIN: [Rank {rank}] rank2 software failure recovery - reading local files directly (no network/XOR)")
                logger.info(f"ECLATIN: [Rank {rank}] recovered_buffer size: {recovered_buffer.numel() if recovered_buffer is not None else 'None'}")

                if recovered_buffer is None:
                    logger.error(f"ECLATIN: [Rank {rank}] recovered_buffer is None")
                    return
                
                if eclatin_blocks is None:
                    logger.error(f"ECLATIN: [Rank {rank}] eclatin_blocks is None")
                    return
                
                # Get checkpoint_dir
                checkpoint_dir = getattr(self, '_current_checkpoint_dir', None)
                if checkpoint_dir is None:
                    logger.error(f"ECLATIN: [Rank {rank}] checkpoint_dir not available")
                    return
                
                checkpoint_dir = Path(checkpoint_dir)
                start_time = time()
                
                # Step 1: Load data_block_1 from local file
                logger.info(f"ECLATIN: [Rank {rank}] Loading data_block_1 from local file")
                self._load_block_data_from_file(
                    checkpoint_dir, rank, 'data_block_1', eclatin_blocks['data_block_1']
                )
                
                # Step 2: Load data_block_2 from local file
                logger.info(f"ECLATIN: [Rank {rank}] Loading data_block_2 from local file")
                self._load_block_data_from_file(
                    checkpoint_dir, rank, 'data_block_2', eclatin_blocks['data_block_2']
                )
                
                # Step 3: Combine data_block_1 and data_block_2 into recovered_buffer
                # Calculate split point (same as hardware recovery)
                actual_tensor_buffer_size = 0
                for r in range(world_size):
                    rank_metadata = registry.rank_metadata.get(r, [])
                    rank_actual_size = sum(meta.size_bytes for meta in rank_metadata)
                    if rank_actual_size > actual_tensor_buffer_size:
                        actual_tensor_buffer_size = rank_actual_size
                
                half_actual_data = actual_tensor_buffer_size // 2
                
                if recovered_buffer.numel() >= total_size:
                    copy_start_time = time()
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
                    
                    copy_end_time = time()
                    logger.info(
                        f"ECLATIN: [Rank {rank}] Combined data_block_1 and data_block_2 into recovered_buffer "
                        f"({total_size / (1024**2):.2f} MB) in {copy_end_time - copy_start_time:.2f} seconds"
                    )
                else:
                    logger.warning(
                        f"ECLATIN: [Rank {rank}] recovered_buffer too small "
                        f"({recovered_buffer.numel()} < {total_size})"
                    )
                
                end_time = time()
                logger.info(f"ECLATIN: [Rank {rank}] rank2 software failure recovery completed in {end_time - start_time:.2f} seconds")

                # Save recovered buffer info for _load_eclatin_checkpoint
                logger.info(f"ECLATIN: [Rank {rank}] Starting to save recovered buffer info")
                mapped_file_own = getattr(self, '_eclatin_mapped_file_own', None)
                logger.info(f"ECLATIN: [Rank {rank}] mapped_file_own available: {mapped_file_own is not None}")
                if mapped_file_own is not None:
                    self.eclatin_recovered_metadata = mapped_file_own
                    self.eclatin_recovered_registry = registry
                    # Save the recovered buffer content
                    if recovered_buffer is not None:
                        logger.info(f"ECLATIN: [Rank {rank}] Cloning recovered_buffer of size {recovered_buffer.numel()}")
                        self.eclatin_recovered_buffer = recovered_buffer.clone()
                        logger.info(f"ECLATIN: [Rank {rank}] Successfully saved eclatin_recovered_buffer of size {self.eclatin_recovered_buffer.numel()}")
                    else:
                        logger.error(f"ECLATIN: [Rank {rank}] recovered_buffer is None, cannot save!")
                    logger.info(f"ECLATIN: [Rank {rank}] Saved recovered buffer info for _load_eclatin_checkpoint")
                else:
                    logger.error(f"ECLATIN: [Rank {rank}] mapped_file_own not available, cannot save recovery info!")

                # Synchronize all ranks and return early (skip network/XOR pipeline)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                    logger.info(f"ECLATIN: [Rank {rank}] Synchronized after software failure recovery")

                return
            else:
                # rank0/1/3: No action needed for software failure recovery
                logger.info(f"ECLATIN: [Rank {rank}] No action needed for rank2 software failure recovery")
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                return
        
        # === Step 1: Set load mode in C++ native module ===
        # Continue with normal hardware failure recovery pipeline
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
        
        start_time = time()
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
            end_time = time()
            logger.info(f"ECLATIN: [Rank {rank}] Recovery pipeline completed in {end_time - start_time:.4f} seconds")
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
                copy_start_time = time()
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
                copy_end_time = time()
                logger.info(f"ECLATIN: [Rank 2] Copied recovered data to buffer in {copy_end_time - copy_start_time:.2f} seconds")
                # logger.info(
                #     f"ECLATIN: [Rank 2] Copied recovered data to buffer "
                #     f"({total_size / (1024**3):.2f} GB): "
                #     f"actual_tensor_buffer_size={actual_tensor_buffer_size / (1024**3):.2f} GB, "
                #     f"half_actual_data={half_actual_data / (1024**3):.2f} GB, "
                #     f"first half {first_half_actual / (1024**3):.2f} GB from data_block_1, "
                #     f"second half {(total_size - first_half_actual) / (1024**3):.2f} GB from data_block_2"
                # )
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
                end_time = time()
                logger.info(f"ECLATIN: [Rank {rank}] Recovery pipeline completed in {end_time - start_time:.2f} seconds")
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
                end_time = time()
                logger.info(f"ECLATIN: [Rank {rank}] Recovery pipeline completed in {end_time - start_time:.2f} seconds")
            
            logger.info(f"ECLATIN: [Rank {rank}] Sent blocks to rank2")
        
        # Synchronize all ranks
        torch.distributed.barrier()
        # end_time = time()
        # logger.info(f"ECLATIN: [Rank {rank}] Recovery pipeline completed in {end_time - start_time:.2f} seconds")
        logger.info(f"ECLATIN: [Rank {rank}] Recovery pipeline completed")
    
    def _run_ecnaive_recovery_pipeline(
        self,
        rank: int,
        world_size: int,
        registry,
        ecnaive_blocks: Dict[str, torch.Tensor],
        recv_buffers: Optional[Dict[str, torch.Tensor]],
        recovered_buffer: Optional[torch.Tensor],
        total_size: int,
    ) -> None:
        """
        Run EC-NAIVE recovery pipeline to recover rank2 data.
        
        EC-NAIVE load mode:
        - rank2: Receives d_{3,1} from rank3 and p_{0,0} from rank0, then XOR recovers d_{2,0}
        - rank0: Recalculates p_{0,0} from data0 and data1, then sends to rank2
        - rank3: Recalculates d_{3,1} from data0 and second half, then sends to rank2
        
        Args:
            rank (int): Current rank
            world_size (int): Total number of ranks
            registry: GlobalMetadataRegistry
            ecnaive_blocks (Dict[str, torch.Tensor]): 4 allocated blocks (all ranks)
            recv_buffers (Optional[Dict[str, torch.Tensor]]): 2 recv buffers (rank2 only)
            recovered_buffer (Optional[torch.Tensor]): Buffer to store recovered data (rank2 only)
            total_size (int): Total size of data to recover
        """
        from time import time
        
        if not self.ecnaive_manager.use_ecnaive:
            logger.warning("EC-NAIVE: Manager not enabled, skipping recovery pipeline")
            return
        
        if self.ecnaive_manager._ecnaive_native is None:
            logger.error("EC-NAIVE: Native module not initialized")
            return
        
        # === Step 1: Ensure load mode is initialized ===
        # Load mode should already be initialized in _load_ecnaive_block_checkpoint
        # But we check here to be safe
        failed_rank = 2  # EC-NAIVE recovers rank2
        logger.info(f"EC-NAIVE: [Rank {rank}] Starting recovery pipeline (failed_rank={failed_rank})")
        
        start_time = time()
        
        # === Step 2: rank2: Receive blocks and recover ===
        if rank == 2:
            if recv_buffers is None or recovered_buffer is None:
                logger.error("EC-NAIVE: [Rank 2] recv_buffers or recovered_buffer is None")
                return
            
            # Get base addresses for recv buffers
            recv_data1_addr = int(recv_buffers['recv_data1'].data_ptr())  # For d_{3,1}
            recv_parity0_addr = int(recv_buffers['recv_parity0'].data_ptr())  # For p_{0,0}
            
            # Get base address for recovered buffer (d_{2,0})
            # Note: recv_data1 will be directly written to recovered_buffer position
            recovered_data0_addr = int(recovered_buffer.data_ptr())
            
            # Calculate aligned block size (same as save phase)
            # EC-NAIVE uses full block size (not half like ECLATIN)
            aligned_block_size = ecnaive_blocks['data0'].numel()
            
            logger.info(
                f"EC-NAIVE: [Rank 2] Starting recovery pipeline\n"
                f"  Recv buffers: {aligned_block_size / (1024**3):.2f} GB each\n"
                f"  Recovered buffer: {total_size / (1024**3):.2f} GB"
            )
            
            # Submit recovery tasks to C++ pipeline
            # The C++ pipeline will:
            # 1. Receive d_{3,1} from rank3 (directly to recovered_buffer)
            # 2. Receive p_{0,0} from rank0 (to recv_parity0 buffer)
            # 3. Perform XOR: d_{2,0} = d_{3,1} XOR p_{0,0}
            # Note: recv_data1_addr and recovered_data0_addr should be the same for zero-copy
            self.ecnaive_manager._ecnaive_native.submit_ecnaive_load_recovery(
                recv_data1_addr=recovered_data0_addr,  # d_{3,1} directly to final position
                recv_parity0_addr=recv_parity0_addr,    # p_{0,0} temporary buffer
                recovered_data0_addr=recovered_data0_addr,  # d_{2,0} output (same as recv_data1)
                size=aligned_block_size
            )
            
            # Submit sentinel to signal pipeline completion
            self.ecnaive_manager._ecnaive_native.submit_load_recv_sentinel()
            
            # Wait for recovery to complete
            logger.info("EC-NAIVE: [Rank 2] Waiting for recovery pipeline to complete...")
            self.ecnaive_manager._ecnaive_native.wait_for_load_completion()
            
            logger.info("EC-NAIVE: [Rank 2] Recovery pipeline completed")
            end_time = time()
            logger.info(f"EC-NAIVE: [Rank {rank}] Recovery pipeline completed in {end_time - start_time:.2f} seconds")
            
            # Note: recovered_buffer already contains d_{2,0} (no need to copy)
            # The C++ pipeline writes directly to recovered_buffer
        
        # === Step 3: rank0/3: Recalculate and send blocks ===
        else:
            aligned_block_size = ecnaive_blocks['data0'].numel()
            
            if rank == 0:
                # rank0: Recalculate p_{0,0} from data0 and data1, then send
                # For now, we'll use data0 as a placeholder and send it
                # TODO: Actually recalculate p_{0,0} from data0 and data1 (second half from main file)
                # This requires loading the second half from the main file and encoding
                
                # Get data0 address (p_{0,0} should be recalculated, but for now use data0)
                # In a full implementation, we'd need to:
                # 1. Load second half from main file
                # 2. Encode data0 and second half to get p_{0,0}
                # 3. Send p_{0,0}
                
                # For now, use a temporary buffer or recalculate
                # Actually, we can use recv_parity0 block as temporary buffer for encoding
                parity0_addr = int(ecnaive_blocks['recv_parity0'].data_ptr())
                data0_addr = int(ecnaive_blocks['data0'].data_ptr())
                
                # TODO: Recalculate p_{0,0} = encode(data0, data1)[0]
                # For now, we'll assume p_{0,0} is already in recv_parity0 (if saved separately)
                # Or we need to recalculate it here
                
                logger.info(f"EC-NAIVE: [Rank 0] Sending p_{0,0} to rank2")
                # Note: This is a placeholder - in full implementation, p_{0,0} should be recalculated
                self.ecnaive_manager._ecnaive_native.submit_load_send_rank0_parity0(
                    send_addr=parity0_addr,  # TODO: Should be recalculated p_{0,0}
                    size=aligned_block_size
                )
                
                # Submit sentinel
                self.ecnaive_manager._ecnaive_native.submit_load_send_sentinel()
                
                end_time = time()
                logger.info(f"EC-NAIVE: [Rank {rank}] Recovery pipeline completed in {end_time - start_time:.2f} seconds")
            
            elif rank == 3:
                # rank3: Recalculate d_{3,1} from data0 and second half, then send
                # For now, we'll use recv_data1 block as placeholder
                # TODO: Actually recalculate d_{3,1} from data0 and second half
                
                # Get recv_data1 address (d_{3,1} should be recalculated, but for now use recv_data1)
                # In a full implementation, we'd need to:
                # 1. Load second half from main file
                # 2. Use second half as d_{3,1}
                
                data1_addr = int(ecnaive_blocks['recv_data1'].data_ptr())
                
                # TODO: Recalculate d_{3,1} = second half of rank3's data
                # For now, we'll assume d_{3,1} is already in recv_data1 (if saved separately)
                # Or we need to recalculate it here
                
                logger.info(f"EC-NAIVE: [Rank 3] Sending d_{3,1} to rank2")
                # Note: This is a placeholder - in full implementation, d_{3,1} should be recalculated
                self.ecnaive_manager._ecnaive_native.submit_load_send_rank3_data1(
                    send_addr=data1_addr,  # TODO: Should be recalculated d_{3,1}
                    size=aligned_block_size
                )
                
                # Submit sentinel
                self.ecnaive_manager._ecnaive_native.submit_load_send_sentinel()
                
                end_time = time()
                logger.info(f"EC-NAIVE: [Rank {rank}] Recovery pipeline completed in {end_time - start_time:.2f} seconds")
            
            logger.info(f"EC-NAIVE: [Rank {rank}] Sent blocks to rank2")
        
        # Synchronize all ranks
        torch.distributed.barrier()
        logger.info(f"EC-NAIVE: [Rank {rank}] Recovery pipeline completed")
    
    def _run_eclatin_layerwise_recovery_pipeline(
        self,
        rank: int,
        world_size: int,
        registry,
        eclatin_blocks: Dict[str, torch.Tensor],
        recv_buffers: Optional[Dict[str, torch.Tensor]],
        layer_groups: Dict,
        sharded_state_dict: ShardedStateDict,
    ) -> None:
        """
        Run ECLATIN layerwise recovery pipeline to recover rank2 data layer-by-layer.
        
        This method sets up the infrastructure and submits layers to C++ for pipeline processing.
        The C++ worker will handle: receive → recover → H2D in a pipelined manner.
        
        Args:
            rank (int): Current rank
            world_size (int): Total number of ranks
            registry: GlobalMetadataRegistry
            eclatin_blocks (Dict[str, torch.Tensor]): 4 allocated blocks (all ranks)
            recv_buffers (Optional[Dict[str, torch.Tensor]]): 6 recv buffers (rank2 only)
            layer_groups (Dict): Layer-organized tensors
            sharded_state_dict: For extracting GPU tensor information
        """
        import ctypes
        from time import time
        
        if not self.eclatin_manager.use_eclatin:
            logger.warning("ECLATIN Layerwise: Manager not enabled, skipping recovery pipeline")
            return
        
        if self.eclatin_manager._eclatin_native is None:
            logger.error("ECLATIN Layerwise: Native module not initialized")
            return
        
        logger.info(f"ECLATIN Layerwise: [Rank {rank}] Starting layerwise recovery pipeline")
        
        # === Step 1: Set load mode in C++ native module ===
        failed_rank = 2  # ECLATIN recovers rank2
        self.eclatin_manager._eclatin_native.set_load_mode(True, failed_rank)
        
        # === Step 2: Initialize load connections (same as standard ECLATIN) ===
        net_config_rank2 = self.eclatin_manager._get_eclatin_network_config(2, world_size)
        rank2_ip = net_config_rank2['rank_ips'].get(2, net_config_rank2['my_ip'])
        
        load_recv_rank0_data2_port = net_config_rank2['ports']['load_recv_rank0_data2']
        load_recv_rank0_parity2_port = net_config_rank2['ports']['load_recv_rank0_parity2']
        load_recv_rank1_data1_port = net_config_rank2['ports']['load_recv_rank1_data1']
        load_recv_rank1_parity1_port = net_config_rank2['ports']['load_recv_rank1_parity1']
        load_recv_rank3_data1_port = net_config_rank2['ports']['load_recv_rank3_data1']
        load_recv_rank3_data2_port = net_config_rank2['ports']['load_recv_rank3_data2']
        
        if rank == 2:
            logger.info(f"ECLATIN Layerwise: [Rank 2] Initializing load accept connections...")
            self.eclatin_manager._eclatin_native.init_load_connections(
                rank, rank2_ip,
                load_recv_rank0_data2_port, load_recv_rank0_parity2_port,
                load_recv_rank1_data1_port, load_recv_rank1_parity1_port,
                load_recv_rank3_data1_port, load_recv_rank3_data2_port
            )
        
        torch.distributed.barrier()
        
        if rank != 2:
            logger.info(f"ECLATIN Layerwise: [Rank {rank}] Connecting load send sockets to rank2...")
            self.eclatin_manager._eclatin_native.init_load_connections(
                rank, rank2_ip,
                load_recv_rank0_data2_port, load_recv_rank0_parity2_port,
                load_recv_rank1_data1_port, load_recv_rank1_parity1_port,
                load_recv_rank3_data1_port, load_recv_rank3_data2_port
            )
        
        logger.info(f"ECLATIN Layerwise: [Rank {rank}] Waiting for load connections...")
        self.eclatin_manager._eclatin_native.wait_for_load_connections(timeout_seconds=30)
        torch.distributed.barrier()
        
        logger.info(f"ECLATIN Layerwise: [Rank {rank}] Load connections initialized")
        
        # === Step 3: Process each layer ===
        start_time = time()
        
        # Note: In layerwise mode, actual recovery and H2D transfer happens in C++ worker
        # Here we just submit tasks to the C++ pipeline
        # For non-rank2, we still need to send data layer-by-layer (to be implemented)
        
        logger.info(f"ECLATIN Layerwise: [Rank {rank}] Layerwise recovery pipeline setup completed")
        logger.info(f"ECLATIN Layerwise: Note - Actual recovery will happen in C++ layerwise_load_worker")
        logger.info(f"ECLATIN Layerwise: [Rank {rank}] Pipeline will be driven by load() method's layer-by-layer processing")
        
        # Synchronize all ranks
        torch.distributed.barrier()
        end_time = time()
        logger.info(f"ECLATIN Layerwise: [Rank {rank}] Pipeline setup completed in {end_time - start_time:.2f}s")
    
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

    def _load_eclatin_layerwise_checkpoint(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Load checkpoint saved in ECLATIN layerwise format with pipelined recovery and model initialization.
        
        This method implements three-stage pipeline:
        1. Network data reception (layer-by-layer)
        2. Recovery computation (layer-by-layer)
        3. Model initialization/CPU→GPU transfer (layer-by-layer)
        
        The pipeline allows overlapping these stages for improved performance.
        """
        from .filesystem_async import FileSystemWriterAsync
        from time import time
        import logging
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        logger.info(f"ECLATIN Layerwise Load: [Rank {rank}] Starting layerwise checkpoint loading")
        start_time = time()
        
        # Step 1: Load block checkpoint data using layerwise-specific method
        checkpoint_dir = Path(checkpoint_dir)
        mapped_file_own, mapped_file_partner = self._load_eclatin_layerwise_block_checkpoint(checkpoint_dir, sharded_state_dict)
        
        # Step 2: Extract layer information from checkpoint metadata
        # This will tell us how the data is organized into layers
        logger.info(f"ECLATIN Layerwise Load: [Rank {rank}] Extracting layer metadata")
        
        # Load metadata from the first block to understand layer organization
        # In layerwise format, metadata includes layer offsets and sizes
        eclatin_file = checkpoint_dir / f'__{rank}_0.distcp'
        
        # Handle failed rank that doesn't have checkpoint file
        if not eclatin_file.exists():
            if rank == 2:
                logger.warning(f"ECLATIN Layerwise Load: [Rank {rank}] File not found (failed node), using recovered metadata")
                # For failed rank, use the metadata from mapped_file_own (already derived from sharded_state_dict)
                # Create a minimal DecomposedStateDict for layer organization
                from .state_dict_decomposer import DecomposedStateDict, TensorInfo
                
                # Convert TensorMetadata back to TensorInfo for layer organization
                tensor_infos = []
                for meta in mapped_file_own.local_metadata:
                    tensor_info = TensorInfo(
                        key=meta.key,
                        shape=meta.shape,
                        dtype=torch.dtype(meta.dtype.replace('torch.', '')) if isinstance(meta.dtype, str) else meta.dtype,
                        device=torch.device('cpu'),
                        numel=meta.shape[0] if len(meta.shape) > 0 else 1,
                        size_bytes=meta.size_bytes,
                        offset=0,
                        global_offset=meta.global_offset,
                        shard_index=meta.shard_index,
                    )
                    tensor_infos.append(tensor_info)
                
                decomposed = DecomposedStateDict(
                    non_tensor_data=mapped_file_own.non_tensor_data,
                    tensor_infos=tensor_infos,
                    tensor_data=[],  # No actual data yet
                    total_tensor_size_bytes=sum(meta.size_bytes for meta in mapped_file_own.local_metadata)
                )
            else:
                raise FileNotFoundError(f"ECLATIN file not found for rank {rank}: {eclatin_file}")
        else:
            # Parse file to extract layer metadata
            # TODO: This needs to be implemented in FileSystemWriterAsync
            # For now, use standard loading and organize by layers ourselves
            decomposed = FileSystemWriterAsync.load_eclatin_components_from_file(str(eclatin_file))
        
        # Step 3: Organize tensors by layer (similar to save logic)
        layer_groups = self._organize_tensors_by_layer_for_load(decomposed, sharded_state_dict)
        
        logger.info(f"ECLATIN Layerwise Load: [Rank {rank}] Organized into {len(layer_groups)} layer groups")
        
        # Step 4: Run layerwise recovery pipeline (setup connections)
        # This will initialize C++ for layerwise processing
        self._run_eclatin_layerwise_recovery_pipeline(
            rank=rank,
            world_size=world_size,
            registry=self.eclatin_recovered_registry,
            eclatin_blocks=self.eclatin_blocks,
            recv_buffers=self.eclatin_recv_buffers if rank == 2 else None,
            layer_groups=layer_groups,
            sharded_state_dict=sharded_state_dict,
        )
        
        logger.info(f"ECLATIN Layerwise Load: [Rank {rank}] Layerwise pipeline initialized")
        
        # Step 5: Process each layer with pipeline
        # For non-recovery ranks: just organize and initialize model layer-by-layer
        # For rank2: receive → recover → initialize (pipelined)
        
        orig_sharded_state_dict = sharded_state_dict
        (keyed_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        )
        
        # Count total ShardedBase objects for debugging
        total_sharded_tensors = sum(1 for sh_base_list in keyed_state_dict.values() 
                                    for sh_base in sh_base_list if isinstance(sh_base, ShardedTensor))
        total_sharded_objects = sum(1 for sh_base_list in keyed_state_dict.values() 
                                   for sh_base in sh_base_list if isinstance(sh_base, ShardedObject))
        logger.info(f"ECLATIN Layerwise Load: [Rank {rank}] Total ShardedBase objects: "
                   f"{total_sharded_tensors + total_sharded_objects} "
                   f"({total_sharded_tensors} tensors, {total_sharded_objects} objects)")
        
        # Load layer-by-layer using C++ pipeline for rank2 recovery
        matched_count = 0
        unmatched_count = 0
        
        # For rank2, use C++ layerwise load worker for pipelined recovery + H2D
        use_cpp_pipeline = (rank == 2 and 
                           self.eclatin_manager.use_eclatin and 
                           self.eclatin_manager._eclatin_native is not None)
        
        for layer_key in sorted(layer_groups.keys()):
            layer_tensors = layer_groups[layer_key]
            
            # Extract numeric layer_id from layer_key (e.g., "layer_0" -> 0)
            # Note: Both real layers and virtual layers (for non-layer tensors) use "layer_N" format
            if layer_key.startswith("layer_"):
                layer_id = int(layer_key.split("_")[1])
                logger.info(f"ECLATIN Layerwise Load: [Rank {rank}] Processing layer {layer_id} ({len(layer_tensors)} tensors)")
            else:
                # This shouldn't happen in layerwise mode - all tensors should be in layer groups
                logger.warning(f"ECLATIN Layerwise Load: [Rank {rank}] Unexpected non-layer group: {layer_key} ({len(layer_tensors)} tensors)")
                
                # Fallback: Handle as standard CPU->GPU transfer
                index_to_data = {}
                for info, tensor in layer_tensors:
                    info_offset = info.global_offset
                    if info_offset is None:
                        info_offset = ()
                    elif not isinstance(info_offset, tuple):
                        info_offset = tuple(info_offset) if hasattr(info_offset, '__iter__') else (info_offset,)
                    index_key = (info.key, info_offset)
                    index_to_data[index_key] = (info, tensor)
                
                # Match and transfer non-layer tensors
                for key, sh_base_list in keyed_state_dict.items():
                    for sh_base in sh_base_list:
                        if isinstance(sh_base, ShardedTensor):
                            sh_offset = tuple(sh_base.global_offset) if hasattr(sh_base.global_offset, '__iter__') else (sh_base.global_offset,)
                            lookup_key = (key, sh_offset)
                            if lookup_key in index_to_data:
                                _, tensor = index_to_data[lookup_key]
                                # Standard CPU->GPU transfer for non-layer tensors
                                if tensor is not None and tensor.device.type == 'cpu':
                                    tensor = tensor.cuda(non_blocking=True)
                                sh_base.data = tensor
                                matched_count += 1
                
                # Skip to next group after processing non-layer tensors
                continue
            
            # Continue with layer processing (both real layers and virtual layers for non-layer tensors)...
            
            # Build index map for this layer
            index_to_data = {}
            layer_size = 0
            for info, tensor in layer_tensors:
                info_offset = info.global_offset
                if info_offset is None:
                    info_offset = ()
                elif not isinstance(info_offset, tuple):
                    info_offset = tuple(info_offset) if hasattr(info_offset, '__iter__') else (info_offset,)
                index_key = (info.key, info_offset)
                index_to_data[index_key] = (info, tensor)
                if tensor is not None:
                    layer_size += tensor.numel() * tensor.element_size()
            
            if use_cpp_pipeline and layer_size > 0:
                # Rank2: Use C++ pipeline for recovery + H2D transfer
                # Prepare GPU tensor info for H2D transfer
                gpu_tensors_info = []
                cpu_offset = 0
                
                for key, sh_base_list in keyed_state_dict.items():
                    for sh_base in sh_base_list:
                        if isinstance(sh_base, ShardedTensor):
                            sh_offset = tuple(sh_base.global_offset) if hasattr(sh_base.global_offset, '__iter__') else (sh_base.global_offset,)
                            lookup_key = (key, sh_offset)
                            if lookup_key in index_to_data:
                                info, tensor = index_to_data[lookup_key]
                                if tensor is not None and hasattr(sh_base, 'data') and isinstance(sh_base.data, torch.Tensor):
                                    # Prepare info for C++ H2D transfer
                                    gpu_ptr = int(sh_base.data.data_ptr()) if sh_base.data.is_cuda else 0
                                    if gpu_ptr > 0:
                                        tensor_size = tensor.numel() * tensor.element_size()
                                        shape = list(tensor.shape)
                                        fqn = info.key
                                        gpu_tensors_info.append((gpu_ptr, cpu_offset, tensor_size, shape, fqn))
                                        cpu_offset += tensor_size
                
                # Submit layer to C++ pipeline
                if gpu_tensors_info:
                    # Get buffer addresses (simplified - actual implementation needs proper buffer management)
                    recv_rank0_data2_addr = int(self.eclatin_recv_buffers['rank0_data2'].data_ptr()) if self.eclatin_recv_buffers else 0
                    recv_rank0_parity2_addr = int(self.eclatin_recv_buffers['rank0_parity2'].data_ptr()) if self.eclatin_recv_buffers else 0
                    recv_rank1_data1_addr = int(self.eclatin_recv_buffers['rank1_data1'].data_ptr()) if self.eclatin_recv_buffers else 0
                    recv_rank1_parity1_addr = int(self.eclatin_recv_buffers['rank1_parity1'].data_ptr()) if self.eclatin_recv_buffers else 0
                    recv_rank3_data1_addr = int(self.eclatin_recv_buffers['rank3_data1'].data_ptr()) if self.eclatin_recv_buffers else 0
                    recv_rank3_data2_addr = int(self.eclatin_recv_buffers['rank3_data2'].data_ptr()) if self.eclatin_recv_buffers else 0
                    
                    recovered_data1_addr = int(self.eclatin_blocks['data_block_1'].data_ptr())
                    recovered_data2_addr = int(self.eclatin_blocks['data_block_2'].data_ptr())
                    recovered_parity1_addr = int(self.eclatin_blocks['parity_block_1'].data_ptr())
                    recovered_parity2_addr = int(self.eclatin_blocks['parity_block_2'].data_ptr())
                    
                    logger.info(f"ECLATIN Layerwise Load: [Rank {rank}] Submitting layer {layer_id} to C++ pipeline")
                    self.eclatin_manager._eclatin_native.submit_layer_wise_load(
                        int(layer_id),
                        gpu_tensors_info,
                        int(recv_rank0_data2_addr),
                        int(recv_rank0_parity2_addr),
                        int(recv_rank1_data1_addr),
                        int(recv_rank1_parity1_addr),
                        int(recv_rank3_data1_addr),
                        int(recv_rank3_data2_addr),
                        int(recovered_data1_addr),
                        int(recovered_data2_addr),
                        int(recovered_parity1_addr),
                        int(recovered_parity2_addr),
                        int(layer_size)
                    )
            
            # Match tensors with sharded_state_dict for this layer
            for key, sh_base_list in keyed_state_dict.items():
                for sh_base in sh_base_list:
                    if isinstance(sh_base, ShardedTensor):
                        sh_offset = tuple(sh_base.global_offset) if hasattr(sh_base.global_offset, '__iter__') else (sh_base.global_offset,)
                        lookup_key = (key, sh_offset)
                        if lookup_key in index_to_data:
                            if use_cpp_pipeline:
                                # Rank2: C++ pipeline has already written data directly to sh_base.data GPU memory
                                # Do NOT overwrite with CPU tensor from index_to_data
                                # sh_base.data already contains the recovered data on GPU
                                matched_count += 1
                            else:
                                # Non-rank2 or fallback: Load CPU tensor and transfer to GPU
                                _, tensor = index_to_data[lookup_key]
                                if tensor is not None and tensor.device.type == 'cpu':
                                    tensor = tensor.cuda(non_blocking=True)
                                sh_base.data = tensor
                                matched_count += 1
        
        # Wait for all C++ layerwise load tasks to complete
        if use_cpp_pipeline:
            logger.info(f"ECLATIN Layerwise Load: [Rank {rank}] Waiting for C++ pipeline to complete...")
            self.eclatin_manager._eclatin_native.wait_all_load_layers_complete()
            logger.info(f"ECLATIN Layerwise Load: [Rank {rank}] C++ pipeline completed")
        
        # Process non-layer tensors (similar to standard loading)
        non_tensor_by_fqn = decomposed.non_tensor_data
        for key, sh_base_list in keyed_state_dict.items():
            for sh_base in sh_base_list:
                if isinstance(sh_base, ShardedObject):
                    if key in non_tensor_by_fqn:
                        value = non_tensor_by_fqn[key]
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
                        logger.warning(f"ECLATIN Layerwise Load: [Rank {rank}] Unmatched ShardedObject: key={key}, unique_key={sh_base.unique_key}")
        
        logger.info(f"ECLATIN Layerwise Load: Matched {matched_count} ShardedBase objects")
        if unmatched_count > 0:
            logger.warning(f"ECLATIN Layerwise Load: {unmatched_count} ShardedBase objects were not matched")
        
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
        
        end_time = time()
        logger.info(
            f"ECLATIN Layerwise Load: [Rank {rank}] Completed in {end_time - start_time:.2f}s, "
            f"loaded {matched_count} tensors"
        )
        
        return mcore_state_dict
    
    def _organize_tensors_by_layer_for_load(self, decomposed, sharded_state_dict):
        """Organize loaded tensors by layer for layerwise processing.
        
        This mirrors the layer extraction logic used during save, including FQN pattern inference.
        MUST use exactly the same logic as _allocate_eclatin_layerwise_recv_buffers to ensure consistency.
        """
        import re
        
        layer_groups = {}  # layer_id -> list of (info, tensor)
        
        def extract_layer_number(fqn: str) -> int:
            """Extract layer number from FQN - using same patterns as save mode.
            
            Supports patterns:
            - decoder.layers.N.
            - encoder.layers.N.
            - transformer.layers.N.
            - model.layers.N.
            - layers.N.
            - .layer.N., _layers_N_, .blocks.N., etc.
            """
            # Use the same patterns as save mode for consistency
            patterns = [
                r'\.layers\.(\d+)\.',      # .layers.N. (matches decoder.layers.0., module.decoder.layers.0., etc.)
                r'^layers\.(\d+)\.',       # layers.N. at start
                r'\.layer\.(\d+)\.',       # .layer.N.
                r'^layer\.(\d+)\.',        # layer.N. at start
                r'_layers_(\d+)_',         # _layers_N_
                r'_layer_(\d+)_',          # _layer_N_
                r'\.blocks\.(\d+)\.',      # .blocks.N.
                r'^blocks\.(\d+)\.',       # blocks.N. at start
                r'_blocks_(\d+)_',         # _blocks_N_
            ]
            for pattern in patterns:
                match = re.search(pattern, fqn)
                if match:
                    return int(match.group(1))
            return -1  # Non-layer tensor
        
        # First pass: count occurrences of each FQN pattern (for inference)
        # This is critical to match save mode behavior
        fqn_to_occurrences = {}
        for info in decomposed.tensor_infos:
            fqn = info.key
            # Normalize to base pattern (remove layer number if present)
            base_fqn = fqn
            if re.search(r'\.layers\.\d+\.', fqn):
                base_fqn = re.sub(r'\.layers\.\d+\.', '.layers.', fqn)
            elif re.search(r'^layers\.\d+\.', fqn):
                base_fqn = re.sub(r'^layers\.\d+\.', 'layers.', fqn)
            
            fqn_to_occurrences[base_fqn] = fqn_to_occurrences.get(base_fqn, 0) + 1
        
        # Determine if this is a layer-based FQN pattern
        # If same FQN appears multiple times (e.g., 6 times for 6 layers), it's a layer tensor
        layer_fqn_patterns = set()
        for fqn, count in fqn_to_occurrences.items():
            if count > 1 and ('layers.' in fqn or 'layer.' in fqn):
                layer_fqn_patterns.add(fqn)
        
        logger.info(f"ECLATIN Load: Found {len(layer_fqn_patterns)} layer FQN patterns (appearing multiple times)")
        if layer_fqn_patterns and logger.isEnabledFor(logging.DEBUG):
            for pattern in sorted(list(layer_fqn_patterns))[:5]:
                logger.debug(f"  Layer pattern: {pattern} (appears {fqn_to_occurrences[pattern]} times)")
        
        # Second pass: assign layer numbers based on FQN pattern and occurrence order
        fqn_to_layer_counter = {}  # Track current layer number for each FQN pattern
        
        # Group tensors by layer using SAME logic as save mode
        for info, tensor in zip(decomposed.tensor_infos, decomposed.tensor_data):
            fqn = info.key
            
            # Try direct extraction first
            layer_num = extract_layer_number(fqn)
            
            # If not found, try to infer from FQN pattern (SAME as save mode)
            if layer_num == -1:
                base_fqn = fqn
                # Normalize to base pattern (remove layer number if present)
                if re.search(r'\.layers\.\d+\.', fqn):
                    base_fqn = re.sub(r'\.layers\.\d+\.', '.layers.', fqn)
                elif re.search(r'^layers\.\d+\.', fqn):
                    base_fqn = re.sub(r'^layers\.\d+\.', 'layers.', fqn)
                
                # If this is a layer pattern (appears multiple times), assign layer number based on occurrence
                if base_fqn in layer_fqn_patterns:
                    if base_fqn not in fqn_to_layer_counter:
                        fqn_to_layer_counter[base_fqn] = 0
                    layer_num = fqn_to_layer_counter[base_fqn]
                    fqn_to_layer_counter[base_fqn] += 1
            
            layer_key = f"layer_{layer_num}" if layer_num >= 0 else "non_layer"
            
            if layer_key not in layer_groups:
                layer_groups[layer_key] = []
            layer_groups[layer_key].append((info, tensor))
        
        # Log layer extraction results (same as save mode)
        layer_keys = [k for k in layer_groups.keys() if k != 'non_layer']
        num_layers = len(layer_keys)
        num_non_layer = len(layer_groups.get('non_layer', []))
        logger.info(
            f"ECLATIN Load: Organized into {num_layers} layers, {num_non_layer} non-layer tensors (before virtual layer splitting)"
        )
        if logger.isEnabledFor(logging.DEBUG) and layer_keys:
            logger.debug(f"ECLATIN Load: Layer keys found: {sorted(layer_keys)}")
        
        # Split non-layer tensors into virtual layers (SAME as save mode)
        if num_non_layer > 0:
            logger.info(f"ECLATIN Load: Splitting {num_non_layer} non-layer tensors into virtual layers...")
            layer_groups = self._split_non_layer_into_virtual_layers_for_load(layer_groups, layer_keys)
        
        return layer_groups
    
    def _split_non_layer_into_virtual_layers_for_load(self, layer_groups, real_layer_keys):
        """Split non-layer tensors into virtual layers for load.
        
        This MUST use the SAME algorithm as save mode to ensure consistency.
        The algorithm splits based on tensor sizes and capacity limits.
        
        Args:
            layer_groups: Dict with layer_groups including "non_layer"
            real_layer_keys: List of real layer keys (for determining virtual layer base ID)
        
        Returns:
            Updated layer_groups with virtual layers added
        """
        non_layer_tensors = layer_groups.get('non_layer', [])
        if not non_layer_tensors:
            return layer_groups
        
        # Calculate average layer size and virtual layer capacity (SAME as save mode)
        total_non_layer_size = sum(
            info.size_bytes for info, tensor in non_layer_tensors
        )
        
        # Get average real layer size
        real_layer_sizes = []
        for layer_key in real_layer_keys:
            if layer_key in layer_groups:
                layer_size = sum(info.size_bytes for info, tensor in layer_groups[layer_key])
                real_layer_sizes.append(layer_size)
        
        avg_layer_size = sum(real_layer_sizes) // len(real_layer_sizes) if real_layer_sizes else total_non_layer_size
        
        # Calculate number of virtual layers needed
        num_virtual_layers = (total_non_layer_size + avg_layer_size - 1) // avg_layer_size
        original_virtual_layer_capacity = (total_non_layer_size + num_virtual_layers - 1) // num_virtual_layers
        
        logger.info(
            f"ECLATIN Load: Splitting non-layer data ({total_non_layer_size / (1024**2):.2f} MB) "
            f"into {num_virtual_layers} virtual layers "
            f"(capacity: {original_virtual_layer_capacity / (1024**2):.2f} MB per layer)"
        )
        
        # Sort tensors by size (largest first) - SAME as save mode
        non_layer_tensors_sorted = sorted(
            non_layer_tensors,
            key=lambda x: x[0].size_bytes,  # info.size_bytes
            reverse=True
        )
        
        # Split into large and small tensors
        large_tensors = []
        small_tensors = []
        
        for info, tensor in non_layer_tensors_sorted:
            if info.size_bytes >= original_virtual_layer_capacity:
                large_tensors.append((info, tensor))
            else:
                small_tensors.append((info, tensor))
        
        logger.info(
            f"ECLATIN Load: {len(large_tensors)} large tensors, {len(small_tensors)} small tensors"
        )
        
        # Assign virtual layers
        virtual_layer_tensors = {}
        current_virtual_layer = 0
        
        # First, assign large tensors (each gets its own layer)
        for info, tensor in large_tensors:
            virtual_layer_tensors[current_virtual_layer] = [(info, tensor)]
            logger.debug(
                f"ECLATIN Load: Virtual layer {current_virtual_layer}: Large tensor {info.key}, "
                f"size={info.size_bytes / (1024**2):.2f} MB"
            )
            current_virtual_layer += 1
        
        # Then, pack small tensors together
        if small_tensors:
            virtual_layer_tensors[current_virtual_layer] = []
            current_virtual_size = 0
            
            for info, tensor in small_tensors:
                # If adding this tensor would exceed capacity, start new layer
                if current_virtual_size > 0 and current_virtual_size + info.size_bytes > original_virtual_layer_capacity:
                    logger.debug(
                        f"ECLATIN Load: Virtual layer {current_virtual_layer}: Packed "
                        f"{len(virtual_layer_tensors[current_virtual_layer])} tensors, "
                        f"size={current_virtual_size / (1024**2):.2f} MB"
                    )
                    current_virtual_layer += 1
                    virtual_layer_tensors[current_virtual_layer] = []
                    current_virtual_size = 0
                
                virtual_layer_tensors[current_virtual_layer].append((info, tensor))
                current_virtual_size += info.size_bytes
            
            # Record last layer
            if virtual_layer_tensors[current_virtual_layer]:
                logger.debug(
                    f"ECLATIN Load: Virtual layer {current_virtual_layer}: Packed "
                    f"{len(virtual_layer_tensors[current_virtual_layer])} tensors, "
                    f"size={current_virtual_size / (1024**2):.2f} MB"
                )
        
        # Add virtual layers to layer_groups
        max_real_layer_id = max(
            [int(k.split('_')[1]) for k in real_layer_keys if k.startswith('layer_')],
            default=-1
        )
        virtual_layer_base_id = max_real_layer_id + 1
        
        for vl_id, tensors in virtual_layer_tensors.items():
            virtual_layer_id = virtual_layer_base_id + vl_id
            layer_key = f"layer_{virtual_layer_id}"
            layer_groups[layer_key] = tensors
            logger.debug(f"ECLATIN Load: Created virtual layer {layer_key} with {len(tensors)} tensors")
        
        # Remove non_layer group
        if 'non_layer' in layer_groups:
            del layer_groups['non_layer']
        
        logger.info(
            f"ECLATIN Load: Created {len(virtual_layer_tensors)} virtual layers "
            f"(IDs {virtual_layer_base_id} to {virtual_layer_base_id + len(virtual_layer_tensors) - 1})"
        )
        
        return layer_groups
    
    def _load_eclatin_checkpoint(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Load checkpoint saved in ECLATIN format (mirror EC-CHECK flow)."""
        from .filesystem_async import FileSystemWriterAsync
        from .state_dict_decomposer import reconstruct_state_dict

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(f"ECLATIN: [Rank {rank}] _load_eclatin_checkpoint called")
        logger.info(f"ECLATIN: [Rank {rank}] Checking recovery buffers - eclatin_recovered_buffer: {hasattr(self, 'eclatin_recovered_buffer') and self.eclatin_recovered_buffer is not None}")
        logger.info(f"ECLATIN: [Rank {rank}] Checking recovery buffers - eclatin_recovered_metadata: {hasattr(self, 'eclatin_recovered_metadata') and self.eclatin_recovered_metadata is not None}")
        
        # Recovery path: rank2 may have recovered buffer
        #if (False):
        logger.info(f"ECLATIN: [Rank {rank}] Checking if should use recovery path: rank==2: {rank == 2}, has_buffer: {hasattr(self, 'eclatin_recovered_buffer')}, buffer_not_none: {self.eclatin_recovered_buffer is not None if hasattr(self, 'eclatin_recovered_buffer') else False}")
        if (rank == 2 and hasattr(self, 'eclatin_recovered_buffer')
            and self.eclatin_recovered_buffer is not None):
            logger.info(f"ECLATIN: [Rank {rank}] Using recovered data from recovery pipeline")
            logger.info(f"ECLATIN: [Rank {rank}] Recovery buffer size: {self.eclatin_recovered_buffer.numel()}")
            decomposed = self._extract_decomposed_from_buffer(
                self.eclatin_recovered_buffer,
                self.eclatin_recovered_metadata,
                self.eclatin_recovered_registry
            )
            logger.info(f"ECLATIN: [Rank {rank}] Successfully extracted decomposed data from recovery buffer")
            self.eclatin_recovered_buffer = None
            self.eclatin_recovered_metadata = None
            self.eclatin_recovered_registry = None
        else:
            checkpoint_dir = Path(checkpoint_dir)
            eclatin_file = checkpoint_dir / f'__{rank}_0.distcp'
            logger.info(f"ECLATIN: [Rank {rank}] Using fallback path - loading from file {eclatin_file}")
            if not eclatin_file.exists():
                logger.error(f"ECLATIN: [Rank {rank}] ECLATIN file not found: {eclatin_file}")
                raise FileNotFoundError(f"ECLATIN file not found for rank {rank}: {eclatin_file}")
            logger.info(f"ECLATIN: [Rank {rank}] Loading ECLATIN checkpoint from {eclatin_file}")
            decomposed = FileSystemWriterAsync.load_eclatin_components_from_file(str(eclatin_file))
            logger.info(f"ECLATIN: [Rank {rank}] Successfully loaded ECLATIN checkpoint from file")
        
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
            f"ECLATIN: [Rank {rank}] Successfully loaded ECLATIN checkpoint "
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
                        # logger.warning(f"ECLATIN: [Rank {rank}] Unmatched ShardedTensor: key={key}, global_offset={sh_offset}, lookup_key={lookup_key}")
        
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
    
    def _load_ecnaive_checkpoint(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Load checkpoint saved in EC-NAIVE format (similar to ECLATIN flow)."""
        from .filesystem_async import FileSystemWriterAsync
        from .state_dict_decomposer import reconstruct_state_dict
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Recovery path: rank2 may have recovered buffer
        if (rank == 2 and hasattr(self, 'ecnaive_recovered_buffer')
            and self.ecnaive_recovered_buffer is not None):
            logger.info(f"EC-NAIVE: [Rank {rank}] Using recovered data from recovery pipeline")
            decomposed = self._extract_decomposed_from_buffer(
                self.ecnaive_recovered_buffer,
                self.ecnaive_recovered_metadata,
                self.ecnaive_recovered_registry
            )
            self.ecnaive_recovered_buffer = None
            self.ecnaive_recovered_metadata = None
            self.ecnaive_recovered_registry = None
        else:
            checkpoint_dir = Path(checkpoint_dir)
            ecnaive_file = checkpoint_dir / f'__{rank}_0.distcp'
            if not ecnaive_file.exists():
                raise FileNotFoundError(f"EC-NAIVE file not found for rank {rank}: {ecnaive_file}")
            logger.info(f"Loading EC-NAIVE checkpoint from {ecnaive_file}")
            # EC-NAIVE uses ECNV magic number
            decomposed = FileSystemWriterAsync.load_ecnaive_components_from_file(str(ecnaive_file))
        
        # Build index map: (key, global_offset) -> (info, tensor)
        logger.info(f"EC-NAIVE: Building index map from {len(decomposed.tensor_infos)} tensor infos")
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
            f"Successfully loaded EC-NAIVE checkpoint for rank {rank} "
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
                        if isinstance(value, dict) and ('_eccheck_type' in value or '_eclatin_type' in value or '_ecnaive_type' in value):
                            wrapper_type = value.get('_eccheck_type') or value.get('_eclatin_type') or value.get('_ecnaive_type')
                            if wrapper_type == 'BytesIO':
                                bytes_data = value.get('_eccheck_data') or value.get('_eclatin_data') or value.get('_ecnaive_data')
                                bytes_io = io.BytesIO(bytes_data)
                                deserialized_list = torch.load(bytes_io, map_location='cpu', weights_only=False)
                                value = deserialized_list[0] if isinstance(deserialized_list, list) else deserialized_list
                        sh_base.data = value
                        matched_count += 1
                    else:
                        unmatched_count += 1
                        logger.warning(f"EC-NAIVE: [Rank {rank}] Unmatched ShardedObject: key={key}")
                elif isinstance(sh_base, ShardedTensor):
                    sh_offset = tuple(sh_base.global_offset) if hasattr(sh_base.global_offset, '__iter__') else (sh_base.global_offset,)
                    lookup_key = (key, sh_offset)
                    if lookup_key in index_to_data:
                        _, tensor = index_to_data[lookup_key]
                        sh_base.data = tensor
                        matched_count += 1
                        if tensor is None:
                            logger.error(f"EC-NAIVE: Matched key {lookup_key} but tensor is None!")
                    else:
                        unmatched_count += 1
        
        logger.info(f"EC-NAIVE: Matched {matched_count} ShardedBase objects")
        if unmatched_count > 0:
            logger.warning(f"EC-NAIVE: {unmatched_count} ShardedBase objects were not matched - this may cause NaN after several iterations!")
        
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
        
        # Prepare RDMA send buffers early for Gemini recovery (rank0 only)
        # This ensures buffers are ready BEFORE rank2 starts waiting, eliminating the 2.9s delay
        if input_args.use_gemini and input_args.use_gemini_hardware_failure and input_args.use_gemini_optimized:
            if rank == 0 and pair_rank == 2:
                # Prepare buffers now if not already prepared
                # This includes: opening mmap, allocating aligned buffers, copying data, registering RDMA
                prepare_start = time()
                self._prepare_gemini_rdma_buffers_if_needed(checkpoint_dir)
                prepare_end = time()
                logger.info(f"rank: {rank}, RDMA send buffers preparation time: {(prepare_end - prepare_start)*1000:.2f}ms")
        
            # Prepare Gemini Replicas RDMA send buffers early for recovery (sender ranks only)
        # This ensures buffers are ready BEFORE rank2 starts waiting
        if input_args.use_gemini_replicas and input_args.use_gemini_replicas_hardware_failure and input_args.use_gemini_replicas_optimized:
            if rank in [0, 1, 3]:
                # Prepare buffers now if not already prepared
                # This includes: opening mmap, allocating aligned buffers, copying data, registering RDMA
                prepare_start = time()
                self._prepare_gemini_replicas_rdma_send_buffers_if_needed(checkpoint_dir)
                prepare_end = time()
                logger.info(f"rank: {rank}, Gemini Replicas RDMA send buffers preparation time: {(prepare_end - prepare_start)*1000:.2f}ms")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
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
                    end_recovery_time = time()
                    recovery_time = end_recovery_time - start_recovery_time
                    logger.info(f"rank: {rank}, gemini asio recovery time: {recovery_time:.2f} seconds")
                else:
                    logger.info(f"rank: {rank}, using standard recovery for rank2 failure")
                    recovered_state_dict = self._load_gemini_checkpoint_recovery(sharded_state_dict, checkpoint_dir)
                    end_recovery_time = time()
                    recovery_time = end_recovery_time - start_recovery_time
                    logger.info(f"rank: {rank}, gemini standard recovery time: {recovery_time:.2f} seconds")
            else:
                logger.info(f"rank: {rank}, not participating in rank2 recovery, loading from own checkpoint file")
                # Other ranks (rank1, rank3) load from their own saved checkpoint files
                recovered_state_dict = self._load_from_saved_checkpoint_file(sharded_state_dict, checkpoint_dir)
                load_end_time = time()
                load_time = load_end_time - start_recovery_time
                logger.info(f"rank: {rank}, gemini asio load time: {load_time:.2f} seconds")
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
            # torch.distributed.barrier()
            end_recovery_time = time()
            recovery_time = end_recovery_time - start_recovery_time
            logger.info(f"rank: {rank}, Gemini software failure recovery time: {recovery_time:.2f} seconds")
            return recovered_state_dict
                
        # Gemini Replicas checkpoint recovery for rank2 failure scenario
        # rank0, rank1, rank3 send data to rank2; rank2 receives and recovers
        if input_args.use_gemini_replicas and input_args.use_gemini_replicas_hardware_failure:
            logger.info(f"rank: {rank}, using Gemini Replicas checkpoint recovery for rank2 failure")
            start_recovery_time = time()
            
            # All participating ranks (0, 1, 2, 3) execute recovery logic
            recovered_state_dict = self._load_gemini_replicas_checkpoint_recovery(sharded_state_dict, checkpoint_dir)
            
            # ALL ranks must synchronize here
            # if torch.distributed.is_initialized():
            #     torch.distributed.barrier()
            #     logger.info(f"rank: {rank}, synchronized after Gemini Replicas recovery")
            
            end_recovery_time = time()
            recovery_time = end_recovery_time - start_recovery_time
            logger.info(f"rank: {rank}, Gemini Replicas hardware failure recovery time: {recovery_time:.4f} seconds")
            
            if recovered_state_dict:
                logger.info(f"rank: {rank}, returning loaded state dict from Gemini Replicas recovery")
                return recovered_state_dict
        
        # Normal Gemini checkpoint load (mutual exchange between paired ranks, and all rank recovery from peer replication)
        # if input_args.use_gemini:
        #     logger.info(f"rank: {rank}, using Gemini checkpointing (normal mode)")
        #     # Load directly from backup data and return the state_dict
        #     return self._load_gemini_checkpoint(sharded_state_dict, checkpoint_dir)
        
        # Check if this is EC-CHECK checkpoint or recovery scenario
        is_eccheck_checkpoint = self._is_eccheck_checkpoint(checkpoint_dir)
        is_recovery_scenario = (rank == 2) or (input_args.use_eccheck_software_failure and rank == 1)
        
        if input_args.use_eccheck and (is_eccheck_checkpoint or is_recovery_scenario):
            logger.info(f"Detected EC-CHECK format checkpoint at {checkpoint_dir}")
            # Load P2P checkpoint data (for rank2 recovery, this prepares the buffer)
            start_recovery_time = time()
            mapped_file_own, mapped_file_partner = self._load_ecccheck_p2p_checkpoint(checkpoint_dir)
            
            # _load_eccheck_checkpoint will use recovered data if available (rank2)
            mcore_state_dict = self._load_eccheck_checkpoint(sharded_state_dict, checkpoint_dir)
            torch.distributed.barrier()
            end_recovery_time = time()
            recovery_time = end_recovery_time - start_recovery_time
            logger.info(f"rank: {rank}, EC-CHECK recovery time: {recovery_time:.2f} seconds")
            return mcore_state_dict
        
        if input_args.use_eclatin and (self._is_eclatin_checkpoint(checkpoint_dir) or rank == 2):
            logger.info(f"Detected ECLATIN format checkpoint at {checkpoint_dir}")
            
            # Check if this is a layerwise checkpoint
            if input_args.use_eclatin_layerwise:
                logger.info(f"Using ECLATIN layerwise load mode")
                # Layerwise loading with pipelined recovery and model initialization
                mcore_state_dict = self._load_eclatin_layerwise_checkpoint(sharded_state_dict, checkpoint_dir)
                return mcore_state_dict
            else:
                logger.info(f"Using ECLATIN standard load mode")
                # Prepare checkpoint data (for rank2 recovery, this prepares the buffer)
                self._load_eclatin_block_checkpoint(checkpoint_dir, sharded_state_dict)
                eclatin_recovery_start_time = time()
                # _load_eclatin_checkpoint will use recovered data if available (rank2)
                mcore_state_dict = self._load_eclatin_checkpoint(sharded_state_dict, checkpoint_dir)
                # torch.distributed.barrier()
                eclatin_recovery_end_time = time()
                eclatin_recovery_time = eclatin_recovery_end_time - eclatin_recovery_start_time
                logger.info(f"ECLATIN: [Rank {rank}] ECLATIN recovery time: {eclatin_recovery_time:.4f} seconds")
                return mcore_state_dict
        
        if input_args.use_ecnaive and (self._is_ecnaive_checkpoint(checkpoint_dir) or rank == 2):
            logger.info(f"Detected EC-NAIVE format checkpoint at {checkpoint_dir}")
            logger.info(f"Using EC-NAIVE load mode")
            # Load block checkpoint data (for rank2 recovery, this prepares the buffer)
            mapped_file_own, mapped_file_partner = self._load_ecnaive_block_checkpoint(checkpoint_dir, sharded_state_dict)
            ecnaive_recovery_start_time = time()
            # _load_ecnaive_checkpoint will use recovered data if available (rank2)
            mcore_state_dict = self._load_ecnaive_checkpoint(sharded_state_dict, checkpoint_dir)
            # torch.distributed.barrier()
            ecnaive_recovery_end_time = time()
            ecnaive_recovery_time = ecnaive_recovery_end_time - ecnaive_recovery_start_time
            logger.info(f"EC-NAIVE: [Rank {rank}] EC-NAIVE recovery time: {ecnaive_recovery_time:.2f} seconds")
            return mcore_state_dict
        
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
