#!/usr/bin/env python3
"""
Calculate checkpoint size for GPT-2 XL model based on test_ft_simulate_xl_multinode_rtime.sh configuration
"""

import math

# Model configuration from test_ft_simulate_xl_multinode_rtime.sh
HIDDEN_SIZE = 2560
NUM_ATTENTION_HEADS = 32
NUM_LAYERS = 48
SEQ_LENGTH = 1024
MAX_POSITION_EMBEDDINGS = 1024
VOCAB_SIZE = 50257  # GPT-2 default vocab size
PADDED_VOCAB_SIZE = 50257  # Usually same as vocab_size unless specified
FFN_HIDDEN_SIZE = 4 * HIDDEN_SIZE  # Standard for GPT models

# Training configuration
USE_FP16 = True
OPTIMIZER = "adam"
USE_DISTRIBUTED_OPTIMIZER = False  # Not specified in script, assume False

# Parallel configuration (from script logic)
# For single GPU per node: TENSOR_PARALLEL=1, PIPELINE_PARALLEL=TOTAL_NODES
# For multi GPU per node: TENSOR_PARALLEL=GPUS_PER_NODE, PIPELINE_PARALLEL=1
# We'll calculate for different scenarios

def calculate_model_parameters():
    """Calculate total model parameters"""
    
    # Embedding layers
    # Word embeddings: vocab_size * hidden_size
    word_embeddings = PADDED_VOCAB_SIZE * HIDDEN_SIZE
    
    # Position embeddings: max_position_embeddings * hidden_size
    position_embeddings = MAX_POSITION_EMBEDDINGS * HIDDEN_SIZE
    
    embedding_params = word_embeddings + position_embeddings
    
    # Transformer layers
    # Self-attention: QKV projection + output projection
    # QKV: 3 * hidden_size * hidden_size (for Q, K, V)
    # Output: hidden_size * hidden_size
    attention_params_per_layer = 4 * HIDDEN_SIZE * HIDDEN_SIZE
    
    # MLP: 2 layers (up projection + down projection)
    # Up: hidden_size * ffn_hidden_size
    # Down: ffn_hidden_size * hidden_size
    mlp_params_per_layer = 2 * HIDDEN_SIZE * FFN_HIDDEN_SIZE
    
    # Layer norms: 2 per layer (input norm + post-attention norm)
    # Each norm: 2 * hidden_size (weight + bias)
    norm_params_per_layer = 2 * 2 * HIDDEN_SIZE
    
    params_per_layer = attention_params_per_layer + mlp_params_per_layer + norm_params_per_layer
    
    transformer_params = NUM_LAYERS * params_per_layer
    
    # Final layer norm
    final_norm_params = 2 * HIDDEN_SIZE
    
    # Output layer (LM head) - shares weights with word embeddings if not untied
    # Assuming tied weights, so no additional params
    
    total_params = embedding_params + transformer_params + final_norm_params
    
    return total_params, embedding_params, transformer_params, final_norm_params

def calculate_checkpoint_size_per_rank(total_params, tensor_parallel_size, pipeline_parallel_size, 
                                       data_parallel_size=1, save_optimizer=True):
    """Calculate checkpoint size per rank"""
    
    # Model parameters are split across tensor and pipeline parallel
    # Each rank stores: (1/pp_size transformer layers + embeddings) / tp_size
    transformer_params_per_rank = (total_params[2] / pipeline_parallel_size) / tensor_parallel_size
    embedding_params_per_rank = total_params[1] / tensor_parallel_size
    final_norm_params_per_rank = total_params[3] / tensor_parallel_size
    
    model_params_per_rank = transformer_params_per_rank + embedding_params_per_rank + final_norm_params_per_rank
    
    # Data type size
    if USE_FP16:
        bytes_per_param = 2
    else:
        bytes_per_param = 4
    
    # Model weights
    model_size = model_params_per_rank * bytes_per_param
    
    # Optimizer state (Adam optimizer)
    if save_optimizer and OPTIMIZER == "adam":
        if USE_DISTRIBUTED_OPTIMIZER:
            # Distributed optimizer: params stored in fp32, but shared across data parallel
            optimizer_size = model_params_per_rank * 4 * (2 + 12 / data_parallel_size)
        else:
            # Standard optimizer: momentum + variance (both fp32) = 2x params in fp32
            optimizer_size = model_params_per_rank * 4 * 2
    else:
        optimizer_size = 0
    
    # Additional checkpoint metadata (iteration, args, etc.) - estimate ~10MB
    metadata_size = 10 * 1024 * 1024
    
    total_size_per_rank = model_size + optimizer_size + metadata_size
    
    return {
        'model_params_per_rank': model_params_per_rank,
        'model_size_mb': model_size / (1024 * 1024),
        'optimizer_size_mb': optimizer_size / (1024 * 1024),
        'metadata_size_mb': metadata_size / (1024 * 1024),
        'total_size_mb': total_size_per_rank / (1024 * 1024),
        'total_size_gb': total_size_per_rank / (1024 * 1024 * 1024)
    }

def main():
    print("=" * 80)
    print("GPT-2 XL Checkpoint Size Calculator")
    print("=" * 80)
    print(f"\nModel Configuration:")
    print(f"  Hidden Size: {HIDDEN_SIZE}")
    print(f"  Num Attention Heads: {NUM_ATTENTION_HEADS}")
    print(f"  Num Layers: {NUM_LAYERS}")
    print(f"  Vocab Size: {VOCAB_SIZE}")
    print(f"  FFN Hidden Size: {FFN_HIDDEN_SIZE}")
    print(f"  Precision: {'FP16' if USE_FP16 else 'FP32'}")
    print(f"  Optimizer: {OPTIMIZER.upper()}")
    print(f"  Save Optimizer: True")
    
    # Calculate total parameters
    total_params, embedding_params, transformer_params, final_norm_params = calculate_model_parameters()
    
    print(f"\nTotal Model Parameters:")
    print(f"  Embedding: {embedding_params:,} ({embedding_params/1e9:.2f}B)")
    print(f"  Transformer Layers: {transformer_params:,} ({transformer_params/1e9:.2f}B)")
    print(f"  Final Norm: {final_norm_params:,} ({final_norm_params/1e9:.2f}B)")
    print(f"  Total: {total_params:,} ({total_params/1e9:.2f}B parameters)")
    
    # Calculate for different parallel configurations
    print(f"\n" + "=" * 80)
    print("Checkpoint Size per Rank (with optimizer state):")
    print("=" * 80)
    
    scenarios = [
        ("Single GPU (TP=1, PP=1)", 1, 1, 1),
        ("Single GPU per node (TP=1, PP=2)", 1, 2, 1),
        ("Single GPU per node (TP=1, PP=4)", 1, 4, 1),
        ("Multi GPU per node (TP=2, PP=1)", 2, 1, 1),
        ("Multi GPU per node (TP=4, PP=1)", 4, 1, 1),
        ("Multi GPU per node (TP=8, PP=1)", 8, 1, 1),
    ]
    
    for scenario_name, tp, pp, dp in scenarios:
        result = calculate_checkpoint_size_per_rank(
            (total_params, embedding_params, transformer_params, final_norm_params),
            tp, pp, dp, save_optimizer=True
        )
        print(f"\n{scenario_name}:")
        print(f"  Tensor Parallel: {tp}, Pipeline Parallel: {pp}")
        print(f"  Model params per rank: {result['model_params_per_rank']/1e6:.2f}M")
        print(f"  Model size: {result['model_size_mb']:.2f} MB")
        print(f"  Optimizer size: {result['optimizer_size_mb']:.2f} MB")
        print(f"  Metadata size: {result['metadata_size_mb']:.2f} MB")
        print(f"  Total per rank: {result['total_size_mb']:.2f} MB ({result['total_size_gb']:.2f} GB)")
    
    # Calculate total checkpoint size (sum across all ranks)
    print(f"\n" + "=" * 80)
    print("Total Checkpoint Size (sum across all ranks):")
    print("=" * 80)
    
    for scenario_name, tp, pp, dp in scenarios:
        result = calculate_checkpoint_size_per_rank(
            (total_params, embedding_params, transformer_params, final_norm_params),
            tp, pp, dp, save_optimizer=True
        )
        total_ranks = tp * pp * dp
        total_checkpoint_size_gb = result['total_size_gb'] * total_ranks
        print(f"\n{scenario_name} ({total_ranks} ranks):")
        print(f"  Total checkpoint size: {total_checkpoint_size_gb:.2f} GB")
    
    # Calculate without optimizer (model only)
    print(f"\n" + "=" * 80)
    print("Checkpoint Size per Rank (model only, no optimizer):")
    print("=" * 80)
    
    for scenario_name, tp, pp, dp in scenarios:
        result = calculate_checkpoint_size_per_rank(
            (total_params, embedding_params, transformer_params, final_norm_params),
            tp, pp, dp, save_optimizer=False
        )
        print(f"\n{scenario_name}:")
        print(f"  Model size: {result['model_size_mb']:.2f} MB")
        print(f"  Metadata size: {result['metadata_size_mb']:.2f} MB")
        print(f"  Total per rank: {result['total_size_mb']:.2f} MB ({result['total_size_gb']:.2f} GB)")

if __name__ == "__main__":
    main()

