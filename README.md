# Attention Kernels From Scratch — C++ / ARM NEON

A standalone C++ project implementing every core component of an LLM inference engine from scratch to deeply understand how attention, caching, quantization, and SIMD optimization work together. Built entirely without external dependencies.

## Core Components

| Component | Description |
|-----------|-------------|
| NEON SIMD library | Vectorized dot products, FMA, and precision conversions |
| Thread pool | Custom task queuing with parallel_for and parallel_reduce |
| Quantization engine | fp16 to int8 conversion and group quantization |
| Naive attention | Multi-Head Attention, GQA, and causal masking (correctness reference) |
| FlashAttention | Online softmax, NEON SIMD acceleration, and multi-threading |
| FlashAttention 2D | Advanced 2D tiling for optimal memory access patterns |
| KV cache | Sliding window eviction, sink tokens, and int8 storage |
| Hybrid attention | Int8-cached and fp16-new tokens evaluated in a single kernel |

Every phase is tested against the previous phase to ensure correctness. Advanced phases explore memory optimization patterns found in production inference engines.

## Project Structure

- `01_neon_basics/`: ARM NEON hardware instructions
- `02_threading/`: Custom thread pool and parallelization
- `03_quantization/`: FP16/INT8 conversion and group quantization
- `04_naive_attention/`: Baseline scalar attention implementation
- `05_flash_attention/`: Optimized attention with online softmax and SIMD
- `05_flash_attention_2d/`: Advanced 2D tiling implementation (research phase)
- `06_kv_cache/`: KV cache implementation with sliding window
- `07_hybrid_attention/`: Production decode path kernel

## Memory Access Patterns

The project explores three increasingly sophisticated attention implementations:

1. **Naive Attention**: Standard O(n²) algorithm materializing full attention matrix
2. **FlashAttention 1D**: Row-wise blocking with online softmax to avoid quadratic memory
3. **FlashAttention 2D**: Advanced tiling where both query and key-value dimensions are blocked for optimal cache reuse

Each approach demonstrates different memory-compute trade-offs fundamental to modern inference engines.

## Key References

- Attention Is All You Need (Vaswani 2017)
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao 2022)
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie 2023)
- Efficient Streaming Language Models with Attention Sinks (Xiao 2023)
