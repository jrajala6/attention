# Attention Kernels From Scratch — C++ / ARM NEON

A standalone C++ project implementing every core component of an LLM inference engine from scratch to deeply understand how attention, caching, quantization, and SIMD optimization work together. Built entirely without external dependencies.

## Requirements

- Apple Silicon Mac (M1/M2/M3) or ARM64 Linux with NEON support
- `clang++` with C++11 support
- `make`

## Build & Run

```bash
# Build all tests
make all

# Build and run full benchmark suite
make bench

# Run individual benchmarks
make neon          # NEON SIMD vs scalar
make threading     # Thread pool vs serial
make quant         # Quantization throughput
make naive         # Flash vs naive attention
make flash         # Flash attention (large sequences)
make flash-compare # NEON vs auto-vectorized vs naive
make kv-cache      # KV cache INT8 vs FP16
make hybrid        # Hybrid attention smoke test
make hybrid-bench  # Hybrid vs naive benchmark

# Build with debug symbols
make DEBUG=1 all

# Clean build artifacts
make clean
```

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

## Benchmarks (Apple M1)

Run `make bench` to reproduce. All speedups are measured against the naive/baseline implementation.

### NEON SIMD vs Scalar

Baseline: scalar fp16 loop compiled with `-fno-vectorize`.

| Vector Size | Scalar (ms) | NEON (ms) | Speedup | Throughput |
|-------------|-------------|-----------|---------|------------|
| 1M | 0.83 | 0.11 | 7.5x | 36.3 GB/s |
| 10M | 8.19 | 1.12 | 7.3x | 35.6 GB/s |
| 50M | 41.66 | 5.81 | 7.2x | 34.4 GB/s |

### Thread Pool vs Serial

Baseline: single-threaded serial loop.

| Problem Size | Serial (ms) | Parallel (ms) | Speedup | Throughput |
|--------------|-------------|---------------|---------|------------|
| 1M ops | 11.6 | 2.1 | 5.6x | 0.48 GOp/s |
| 10M ops | 117.6 | 19.7 | 6.0x | 0.51 GOp/s |
| 100M ops | 1189.8 | 147.6 | 8.1x | 0.68 GOp/s |

### Quantization (FP16 -> INT8)

Baseline: uncompressed FP16 (2 bytes/element).

| Group Size | Throughput | Compression | MSE | Size Saved |
|------------|------------|-------------|-----|------------|
| 32 (Ultra-fine) | 95.2 GB/s | 1.78x | 2.9e-05 | 43.8% |
| 64 (Fine) | 113.2 GB/s | 1.88x | 3.5e-05 | 46.9% |
| 128 (Standard) | 113.1 GB/s | 1.94x | 4.2e-05 | 48.4% |
| 256 (Coarse) | 120.2 GB/s | 1.97x | 4.9e-05 | 49.2% |
| 512 (Very coarse) | 102.5 GB/s | 1.98x | 5.5e-05 | 49.6% |

### Flash Attention vs Naive

Baseline: naive O(n^2) attention materializing full attention matrix.

**Sequence length scaling** (batch=1, q_heads=32, kv_heads=8, head_dim=128):

| Seq Length | Naive (ms) | Flash (ms) | Speedup | Mem Saved |
|------------|------------|------------|---------|-----------|
| 512 | 77.1 | 11.9 | 6.5x | 0.03 GB |
| 1K | 307.1 | 42.3 | 7.3x | 0.13 GB |
| 2K | 1247.7 | 184.8 | 6.8x | 0.53 GB |
| 4K | 5113.8 | 685.5 | 7.5x | 2.15 GB |

**GQA head scaling** (batch=1, seq_len=1024, head_dim=128):

| Heads (Q/KV) | Naive (ms) | Flash (ms) | Speedup | Mem Saved |
|--------------|------------|------------|---------|-----------|
| 16/4 | 154.1 | 23.0 | 6.7x | 0.07 GB |
| 32/8 | 307.4 | 41.4 | 7.4x | 0.13 GB |
| 64/8 | 615.2 | 77.2 | 8.0x | 0.26 GB |

### Flash NEON vs Auto-Vectorized vs Naive

All speedups vs naive baseline (batch=1, q_heads=32, kv_heads=8):

| Configuration | Naive (ms) | NEON (ms) | AutoVec (ms) | NEON vs Naive | AutoVec vs Naive |
|---------------|------------|-----------|--------------|---------------|------------------|
| 512, h=64 | 50.2 | 7.6 | 7.8 | 6.6x | 6.4x |
| 1K, h=128 | 309.8 | 45.5 | 46.1 | 6.8x | 6.7x |
| 2K, h=128 | 1278.1 | 182.8 | 214.9 | 7.0x | 5.9x |
| 4K, h=128 | 4977.7 | 640.5 | 657.9 | 7.8x | 7.6x |

### KV Cache: FP16 vs INT8 (with auto-quantization)

Both receive FP16 input. INT8 cache quantizes on the fly.

| Configuration | FP16 (ms) | INT8 (ms) | Quant Cost | Mem Saved |
|---------------|-----------|-----------|------------|-----------|
| 4L, 64T, 8H, d=64 | 0.02 | 0.05 | 3x | 49.6% |
| 12L, 256T, 12H, d=64 | 0.40 | 0.83 | 2x | 49.7% |
| 32L, 512T, 32H, d=128 | 15.56 | 15.42 | 1x | 50.0% |

### Hybrid Attention vs Naive

INT8-cached KV + FP16-new tokens, all speedups vs naive baseline.

| Configuration | Naive (us) | Hybrid (us) | Speedup | GOPS |
|---------------|------------|-------------|---------|------|
| 128seq, 8/2h, d=128 | 1153 | 416 | 2.8x | 60.5 |
| 256seq, 8/2h, d=128 | 4457 | 1665 | 2.7x | 60.5 |
| 128seq, 32/8h, d=128 | 4536 | 1401 | 3.2x | 71.8 |

## Project Structure

- `01_neon_basics/`: ARM NEON hardware instructions
- `02_threading/`: Custom thread pool and parallelization
- `03_quantization/`: FP16/INT8 conversion and group quantization
- `04_naive_attention/`: Baseline scalar attention implementation
- `05_flash_attention/`: Optimized attention with online softmax and SIMD
- `05_flash_attention_2d/`: Advanced 2D tiling implementation (in progress)
- `06_kv_cache/`: KV cache implementation with sliding window
- `07_hybrid_attention/`: Production decode path kernel

## Memory Access Patterns

The project explores three increasingly sophisticated attention implementations:

1. **Naive Attention**: Standard O(n²) algorithm materializing full attention matrix
2. **FlashAttention 1D**: Row-wise blocking with online softmax to avoid quadratic memory
3. **FlashAttention 2D**: Advanced tiling where both query and key-value dimensions are blocked for optimal cache reuse (Coming soon)
4. **Hybrid Attention**: INT8 cached KV with FP16 new tokens, per-head grouped quantization scales, and online softmax — the production decode path

Each approach demonstrates different memory-compute trade-offs fundamental to modern inference engines.

## Key References

- Attention Is All You Need (Vaswani 2017)
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao 2022)
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie 2023)
- Efficient Streaming Language Models with Attention Sinks (Xiao 2023)

Inspired by [Cactus Compute](https://github.com/cactus-compute/cactus).
