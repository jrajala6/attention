# Build: O3 optimization + Apple M1 NEON
CXX = clang++
CXXFLAGS = -std=c++11 -O3 -march=native -mtune=native -Wall -Wextra -Wpedantic -I. -Icommon
LDFLAGS = -pthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -mcpu=apple-m1 -ffast-math
endif
ifeq ($(UNAME_S),Linux)
    CXXFLAGS += -mcpu=native -ffast-math
endif

DEBUG ?= 0
ifeq ($(DEBUG), 1)
    CXXFLAGS += -g -DDEBUG -O0
    CXXFLAGS := $(filter-out -O3,$(CXXFLAGS))
endif

# Directory layout
NEON_DIR = 01_neon_basics
THREAD_DIR = 02_threading
QUANT_DIR = 03_quantization
NAIVE_DIR = 04_naive_attention
FLASH_DIR = 05_flash_attention
KV_CACHE_DIR = 06_kv_cache
HYBRID_DIR = 07_hybrid_attention

# Object dependencies
NEON_OBJS = $(NEON_DIR)/neon_ops.o
SIMPLE_OBJS = $(NEON_DIR)/simple_ops.o
THREAD_OBJS = $(THREAD_DIR)/thread_pool.o
QUANT_OBJS = $(QUANT_DIR)/quant.o $(THREAD_OBJS)
NAIVE_OBJS = $(NAIVE_DIR)/naive_attention.o $(NEON_OBJS) $(THREAD_OBJS)
FLASH_OBJS = $(FLASH_DIR)/flash_attention.o $(NAIVE_OBJS)
FLASH_SIMPLE_OBJS = $(FLASH_DIR)/flash_attention_simple.o $(SIMPLE_OBJS) $(THREAD_OBJS)
KV_CACHE_OBJS = $(KV_CACHE_DIR)/kv_cache.o $(NEON_OBJS) $(QUANT_OBJS)
HYBRID_OBJS = $(HYBRID_DIR)/hybrid_attention.o $(NEON_OBJS) $(THREAD_OBJS) $(KV_CACHE_OBJS)

# All test binaries
TESTS = test_neon test_threading test_quantization test_naive_attention \
        test_flash_attention test_flash_simple test_flash_comparison \
        test_kv_cache test_hybrid benchmark_hybrid

# === Main targets ===

.PHONY: all clean bench

all: $(TESTS)
	@echo "\nBuild complete. Run 'make bench' for benchmarks.\n"

bench: $(TESTS)
	@echo ""
	@./test_neon
	@./test_threading
	@./test_quantization
	@./test_naive_attention
	@./test_flash_attention
	@./test_flash_simple
	@./test_flash_comparison
	@./test_kv_cache
	@./test_hybrid
	@./benchmark_hybrid

clean:
	@find . -name "*.o" -delete
	@rm -f $(TESTS)

# === Link rules ===

test_neon: $(NEON_DIR)/test_neon.o $(NEON_OBJS)
	$(CXX) $(CXXFLAGS) -fno-vectorize -o $@ $^ $(LDFLAGS)

test_threading: $(THREAD_DIR)/test_threading.o $(THREAD_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_quantization: $(QUANT_DIR)/test_quant.o $(QUANT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_naive_attention: $(NAIVE_DIR)/test_naive.o $(FLASH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_flash_attention: $(FLASH_DIR)/test_flash.o $(FLASH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_flash_simple: $(FLASH_DIR)/test_flash_simple.o $(FLASH_SIMPLE_OBJS) $(NEON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_flash_comparison: $(FLASH_DIR)/test_flash_comparison.o $(FLASH_OBJS) $(FLASH_SIMPLE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_kv_cache: $(KV_CACHE_DIR)/test_kv_cache.o $(KV_CACHE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_hybrid: $(HYBRID_DIR)/test_hybrid.o $(HYBRID_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

benchmark_hybrid: $(HYBRID_DIR)/benchmark_hybrid.o $(HYBRID_OBJS) $(FLASH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# === Compile rules ===

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(NEON_DIR)/test_neon.o: $(NEON_DIR)/test_neon.cpp
	$(CXX) $(CXXFLAGS) -fno-vectorize -c -o $@ $<

# === Individual targets ===

.PHONY: neon threading quant naive flash flash-simple flash-compare kv-cache hybrid hybrid-bench

neon: test_neon
	@./test_neon
threading: test_threading
	@./test_threading
quant: test_quantization
	@./test_quantization
naive: test_naive_attention
	@./test_naive_attention
flash: test_flash_attention
	@./test_flash_attention
flash-simple: test_flash_simple
	@./test_flash_simple
flash-compare: test_flash_comparison
	@./test_flash_comparison
kv-cache: test_kv_cache
	@./test_kv_cache
hybrid: test_hybrid
	@./test_hybrid
hybrid-bench: benchmark_hybrid
	@./benchmark_hybrid

.PRECIOUS: %.o
