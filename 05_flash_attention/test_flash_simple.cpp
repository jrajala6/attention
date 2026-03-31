#include "01_neon_basics/simple_ops.h"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    const size_t len = 128;
    std::vector<__fp16> a(len), b(len);
    std::vector<float> accum(len, 0.0f);

    for (size_t i = 0; i < len; ++i) {
        a[i] = static_cast<__fp16>(0.5f);
        b[i] = static_cast<__fp16>(0.5f);
    }

    std::cout << "Testing simple auto-vectorizable operations...\n";

    auto start = std::chrono::high_resolution_clock::now();
    float result = dot_product_f16_simple(a.data(), b.data(), len);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Simple dot product result: " << result << "\n";
    std::cout << "Time: " << duration.count() << " microseconds\n";

    weighted_accumulate_simple(accum.data(), a.data(), 2.0f, len);
    std::cout << "Weighted accumulate test completed\n";

    return 0;
}