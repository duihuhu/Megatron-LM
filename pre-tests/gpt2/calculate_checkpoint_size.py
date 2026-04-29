#!/usr/bin/env python3
"""
Calculate checkpoint size for GPT model based on configuration parameters.
"""

# Model parameters from test_eccheck_1nodes_node.sh
HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 16
NUM_LAYERS = 12
VOCAB_SIZE = 50257  # GPT2 standard vocabulary size
USE_FP16 = True  # --fp16 flag
POSITION_EMBEDDING_TYPE = "rope"  # No position embedding parameters

# Parallelism (no sharding)
TENSOR_MODEL_PARALLEL_SIZE = 1
PIPELINE_MODEL_PARALLEL_SIZE = 1

# Bytes per parameter
BYTES_PER_PARAM_FP16 = 2
BYTES_PER_PARAM_FP32 = 4

def calculate_model_parameters():
    """Calculate total number of model parameters."""
    
    bytes_per_param = BYTES_PER_PARAM_FP16 if USE_FP16 else BYTES_PER_PARAM_FP32
    
    total_params = 0
    breakdown = {}
    
    # 1. Word Embedding
    word_embedding_params = VOCAB_SIZE * HIDDEN_SIZE
    total_params += word_embedding_params
    breakdown["word_embedding"] = word_embedding_params
    print(f"Word Embedding: {word_embedding_params:,} parameters")
    
    # 2. Position Embedding (if not RoPE)
    if POSITION_EMBEDDING_TYPE != "rope":
        # Usually max_position_embeddings * hidden_size
        position_embedding_params = 1024 * HIDDEN_SIZE  # SEQ_LENGTH = 1024
        total_params += position_embedding_params
        breakdown["position_embedding"] = position_embedding_params
        print(f"Position Embedding: {position_embedding_params:,} parameters")
    else:
        print("Position Embedding: 0 (using RoPE)")
    
    # 3. Transformer Layers (12 layers)
    per_layer_params = 0
    
    # Attention QKV projection (combined)
    # Q, K, V each: hidden_size * hidden_size
    attention_qkv_params = 3 * HIDDEN_SIZE * HIDDEN_SIZE
    per_layer_params += attention_qkv_params
    
    # Attention output projection
    attention_output_params = HIDDEN_SIZE * HIDDEN_SIZE
    per_layer_params += attention_output_params
    
    # MLP layers (typically FFN dimension is 4 * hidden_size)
    FFN_DIM = 4 * HIDDEN_SIZE
    mlp_fc1_params = HIDDEN_SIZE * FFN_DIM
    mlp_fc2_params = FFN_DIM * HIDDEN_SIZE
    per_layer_params += mlp_fc1_params + mlp_fc2_params
    
    # LayerNorm parameters (2 per layer: input_layernorm and post_layernorm)
    # Each LayerNorm has weight and bias: 2 * hidden_size
    layernorm_params_per_norm = 2 * HIDDEN_SIZE
    layernorm_params = 2 * layernorm_params_per_norm  # input + post
    per_layer_params += layernorm_params
    
    # Attention bias (if used, typically 3 * hidden_size for QKV)
    # Most implementations don't use bias, but checking
    # attention_bias_params = 3 * HIDDEN_SIZE  # Usually not used
    # per_layer_params += attention_bias_params
    
    # MLP bias (if used)
    # mlp_bias_params = FFN_DIM + HIDDEN_SIZE  # Usually not used in modern implementations
    # per_layer_params += mlp_bias_params
    
    total_layer_params = NUM_LAYERS * per_layer_params
    total_params += total_layer_params
    breakdown["transformer_layers"] = total_layer_params
    
    print(f"\nPer Transformer Layer:")
    print(f"  Attention QKV: {attention_qkv_params:,} parameters")
    print(f"  Attention Output: {attention_output_params:,} parameters")
    print(f"  MLP FC1: {mlp_fc1_params:,} parameters")
    print(f"  MLP FC2: {mlp_fc2_params:,} parameters")
    print(f"  LayerNorm (2x): {layernorm_params:,} parameters")
    print(f"  Total per layer: {per_layer_params:,} parameters")
    print(f"\nAll {NUM_LAYERS} layers: {total_layer_params:,} parameters")
    
    # 4. Final LayerNorm
    final_layernorm_params = 2 * HIDDEN_SIZE
    total_params += final_layernorm_params
    breakdown["final_layernorm"] = final_layernorm_params
    print(f"\nFinal LayerNorm: {final_layernorm_params:,} parameters")
    
    # 5. Output Layer (LM Head)
    output_layer_params = VOCAB_SIZE * HIDDEN_SIZE
    total_params += output_layer_params
    breakdown["output_layer"] = output_layer_params
    print(f"Output Layer (LM Head): {output_layer_params:,} parameters")
    
    # Calculate sizes
    model_size_bytes = total_params * bytes_per_param
    model_size_mb = model_size_bytes / (1024 ** 2)
    model_size_gb = model_size_bytes / (1024 ** 3)
    
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETERS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Precision: {'FP16' if USE_FP16 else 'FP32'}")
    print(f"Model Size: {model_size_bytes:,} bytes")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Model Size: {model_size_gb:.2f} GB")
    
    return total_params, model_size_bytes, breakdown


def calculate_checkpoint_size(model_size_bytes):
    """Calculate full checkpoint size including optimizer states."""
    
    bytes_per_param_fp16 = BYTES_PER_PARAM_FP16
    bytes_per_param_fp32 = BYTES_PER_PARAM_FP32
    
    # Model parameters
    model_params = model_size_bytes // bytes_per_param_fp16 if USE_FP16 else model_size_bytes // bytes_per_param_fp32
    
    # Optimizer states (Adam optimizer)
    # Adam stores: momentum (fp32) + variance (fp32) = 2 * fp32 per parameter
    optimizer_momentum_bytes = model_params * bytes_per_param_fp32
    optimizer_variance_bytes = model_params * bytes_per_param_fp32
    optimizer_total_bytes = optimizer_momentum_bytes + optimizer_variance_bytes
    
    # Total checkpoint size
    total_checkpoint_bytes = model_size_bytes + optimizer_total_bytes
    
    # Additional metadata (usually small, ~few MB)
    metadata_bytes = 10 * 1024 * 1024  # ~10 MB for metadata
    
    total_with_metadata = total_checkpoint_bytes + metadata_bytes
    
    print(f"\n{'='*60}")
    print(f"CHECKPOINT SIZE BREAKDOWN")
    print(f"{'='*60}")
    print(f"Model Parameters: {model_size_bytes / (1024**3):.2f} GB")
    print(f"Optimizer Momentum: {optimizer_momentum_bytes / (1024**3):.2f} GB")
    print(f"Optimizer Variance: {optimizer_variance_bytes / (1024**3):.2f} GB")
    print(f"Optimizer Total: {optimizer_total_bytes / (1024**3):.2f} GB")
    print(f"Metadata: {metadata_bytes / (1024**2):.2f} MB")
    print(f"{'-'*60}")
    print(f"TOTAL CHECKPOINT SIZE: {total_with_metadata / (1024**3):.2f} GB")
    print(f"TOTAL CHECKPOINT SIZE: {total_with_metadata / (1024**2):.2f} MB")
    print(f"{'='*60}")
    
    return total_with_metadata


if __name__ == "__main__":
    print("GPT-2 345M Model Checkpoint Size Calculator")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Hidden Size: {HIDDEN_SIZE}")
    print(f"  Number of Layers: {NUM_LAYERS}")
    print(f"  Number of Attention Heads: {NUM_ATTENTION_HEADS}")
    print(f"  Vocabulary Size: {VOCAB_SIZE}")
    print(f"  Precision: {'FP16' if USE_FP16 else 'FP32'}")
    print(f"  Position Embedding: {POSITION_EMBEDDING_TYPE}")
    print(f"  Tensor Parallel Size: {TENSOR_MODEL_PARALLEL_SIZE}")
    print(f"  Pipeline Parallel Size: {PIPELINE_MODEL_PARALLEL_SIZE}")
    print("=" * 60)
    print()
    
    total_params, model_size_bytes, breakdown = calculate_model_parameters()
    total_checkpoint_size = calculate_checkpoint_size(model_size_bytes)
    
    print(f"\nNote: This calculation assumes:")
    print(f"  - Standard GPT architecture with 4x FFN dimension")
    print(f"  - Adam optimizer with momentum and variance states")
    print(f"  - No gradient accumulation buffers")
    print(f"  - No additional training state (iteration, learning rate, etc.)")

