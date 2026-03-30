#include "quant.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  size_t total_elements = 10000000; // 10 million elements
  size_t group_size = 128;
  size_t num_groups = (total_elements + group_size - 1) / group_size;

  std::vector<__fp16> src(total_elements);
  std::vector<int8_t> dst(total_elements);
  std::vector<float> scales(num_groups);

  for (size_t i = 0; i < total_elements; ++i) {
    src[i] = static_cast<__fp16>(sin(i));
  }

  auto start = std::chrono::high_resolution_clock::now();
  // This will benchmark your current grouped quantization implementation
  quantize_fp16_to_int8_grouped(src.data(), dst.data(), scales.data(),
                                total_elements, group_size);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> quant_ms = end - start;

  std::cout << "--- 03 Quantization Benchmark ---\n";
  std::cout << "Total elements: " << total_elements
            << ", Group size: " << group_size << "\n";
  std::cout << "Time taken: " << quant_ms.count() << " ms\n";
  std::cout << "Throughput: "
            << (total_elements * sizeof(__fp16)) / (quant_ms.count() / 1000.0) /
                   1e9
            << " GB/s\n\n";
  std::cout
      << "Notice the threading overhead of launching parallel_reduce for only "
      << group_size << " elements repeatedly!\n";

  return 0;
}
