#include "naive_attention.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <random>

struct AttentionResult {
    double latency_ms;
    double throughput_tokens_s;
    double memory_gb;
    double flops_gflops;
};

void fill_random_realistic(std::vector<__fp16>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f); // Realistic initialization

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<__fp16>(dist(gen));
    }
}

AttentionResult benchmark_attention_config(size_t batch_size, size_t seq_len, size_t num_q_heads,
                                          size_t num_kv_heads, size_t head_dim, bool is_causal,
                                          int warmup_runs = 3, int test_runs = 8) {
    size_t kv_seq_len = seq_len; // Self-attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    size_t window_size = 0;

    size_t q_size = batch_size * num_q_heads * seq_len * head_dim;
    size_t kv_size = batch_size * num_kv_heads * kv_seq_len * head_dim;
    size_t out_size = q_size;

    std::vector<__fp16> Q(q_size);
    std::vector<__fp16> K(kv_size);
    std::vector<__fp16> V(kv_size);
    std::vector<__fp16> O(out_size, static_cast<__fp16>(0.0f));

    fill_random_realistic(Q);
    fill_random_realistic(K);
    fill_random_realistic(V);

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        naive_attention(Q.data(), K.data(), V.data(), O.data(), batch_size, seq_len, kv_seq_len,
                       num_q_heads, num_kv_heads, head_dim, scale, is_causal, window_size);
    }

    // Benchmark runs
    std::vector<double> run_times;
    for (int run = 0; run < test_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        naive_attention(Q.data(), K.data(), V.data(), O.data(), batch_size, seq_len, kv_seq_len,
                       num_q_heads, num_kv_heads, head_dim, scale, is_causal, window_size);
        auto end = std::chrono::high_resolution_clock::now();
        run_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // Use median for robustness
    std::sort(run_times.begin(), run_times.end());
    double median_latency = run_times[test_runs / 2];

    // Calculate throughput
    double total_tokens = batch_size * seq_len;
    double throughput_tokens_s = total_tokens / (median_latency / 1000.0);

    // Calculate memory usage
    double memory_bytes = (q_size + 2 * kv_size + out_size) * sizeof(__fp16);
    // Add attention matrix memory (this is what naive attention materializes)
    memory_bytes += batch_size * num_q_heads * seq_len * kv_seq_len * sizeof(float);
    double memory_gb = memory_bytes / 1e9;

    // Calculate FLOPs (approximate)
    // Q·K^T: batch * heads * seq_len * kv_seq_len * head_dim operations
    // Softmax: roughly 3 * batch * heads * seq_len * kv_seq_len operations
    // Attention·V: batch * heads * seq_len * kv_seq_len * head_dim operations
    double qk_flops = static_cast<double>(batch_size) * num_q_heads * seq_len * kv_seq_len * head_dim;
    double softmax_flops = 3.0 * static_cast<double>(batch_size) * num_q_heads * seq_len * kv_seq_len;
    double av_flops = static_cast<double>(batch_size) * num_q_heads * seq_len * kv_seq_len * head_dim;
    double total_flops = qk_flops + softmax_flops + av_flops;
    double gflops = total_flops / (median_latency / 1000.0) / 1e9;

    return {median_latency, throughput_tokens_s, memory_gb, gflops};
}

int main() {
    std::cout << "=== NAIVE ATTENTION SCALING ANALYSIS ===\n\n";
    std::cout << "Baseline attention implementation performance across model scales\n";
    std::cout << "Demonstrates O(n²) scaling and memory requirements\n\n";

    // Test sequence length scaling
    std::vector<std::pair<size_t, std::string> > seq_configs;
    seq_configs.push_back(std::make_pair(256,  std::string("Short (256)")));
    seq_configs.push_back(std::make_pair(512,  std::string("Medium (512)")));
    seq_configs.push_back(std::make_pair(1024, std::string("Long (1K)")));
    seq_configs.push_back(std::make_pair(2048, std::string("Very Long (2K)")));
    seq_configs.push_back(std::make_pair(4096, std::string("Ultra Long (4K)")));

    std::cout << "=== SEQUENCE LENGTH SCALING (32 heads, 128 head_dim, causal) ===\n";
    std::cout << std::setw(18) << "Sequence Length"
              << std::setw(15) << "Latency (ms)"
              << std::setw(18) << "Throughput"
              << std::setw(15) << "Memory (GB)"
              << std::setw(15) << "Compute"
              << "\n";
    std::cout << std::string(85, '-') << "\n";

    double baseline_latency = 0.0;

    for (size_t i = 0; i < seq_configs.size(); ++i) {
        size_t seq_len = seq_configs[i].first;
        const std::string& name = seq_configs[i].second;

        AttentionResult result = benchmark_attention_config(1, seq_len, 32, 8, 128, true);

        if (i == 0) baseline_latency = result.latency_ms;
        double scaling_factor = result.latency_ms / baseline_latency;

        std::cout << std::setw(18) << name
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.latency_ms
                  << std::setw(18) << std::fixed << std::setprecision(0) << result.throughput_tokens_s << " tok/s"
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.memory_gb
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.flops_gflops << " GFLOPS"
                  << "\n";
    }

    // Test head scaling
    std::vector<std::pair<std::pair<size_t, size_t>, std::string> > head_configs;
    head_configs.push_back(std::make_pair(std::make_pair(16, 4), "Small (16→4)"));
    head_configs.push_back(std::make_pair(std::make_pair(32, 8), "Standard (32→8)"));
    head_configs.push_back(std::make_pair(std::make_pair(64, 8), "Large (64→8)"));
    head_configs.push_back(std::make_pair(std::make_pair(96, 8), "XLarge (96→8)"));

    std::cout << "\n=== HEAD COUNT SCALING (1K sequence, 128 head_dim, causal) ===\n";
    std::cout << std::setw(18) << "Head Config"
              << std::setw(15) << "Latency (ms)"
              << std::setw(18) << "Throughput"
              << std::setw(15) << "Memory (GB)"
              << std::setw(15) << "Compute"
              << "\n";
    std::cout << std::string(85, '-') << "\n";

    double max_throughput = 0.0;
    double max_compute = 0.0;

    for (size_t i = 0; i < head_configs.size(); ++i) {
        size_t q_heads = head_configs[i].first.first;
        size_t kv_heads = head_configs[i].first.second;
        const std::string& name = head_configs[i].second;

        AttentionResult result = benchmark_attention_config(1, 1024, q_heads, kv_heads, 128, true);

        std::cout << std::setw(18) << name
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.latency_ms
                  << std::setw(18) << std::fixed << std::setprecision(0) << result.throughput_tokens_s << " tok/s"
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.memory_gb
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.flops_gflops << " GFLOPS"
                  << "\n";

        max_throughput = std::max(max_throughput, result.throughput_tokens_s);
        max_compute = std::max(max_compute, result.flops_gflops);
    }

    std::cout << "\n=== SCALING ANALYSIS ===\n";
    std::cout << "• Quadratic scaling: 4K sequence uses " << std::fixed << std::setprecision(0)
              << (4096.0 * 4096.0) / (256.0 * 256.0) << "x more memory than 256 length\n";
    std::cout << "• Memory-bound at large sequences due to O(n²) attention matrix\n";
    std::cout << "• Peak computational throughput: " << std::fixed << std::setprecision(1) << max_compute << " GFLOPS\n";
    std::cout << "• GQA (Grouped Query Attention) reduces KV memory overhead\n\n";

    std::cout << "=== PERFORMANCE LIMITATIONS ===\n";
    std::cout << "• Materializes full attention matrix: memory scales as O(batch × heads × seq²)\n";
    std::cout << "• No memory reuse: each attention head computed independently\n";
    std::cout << "• Memory bandwidth becomes bottleneck for large sequences\n";
    std::cout << "• Sets baseline for Flash Attention comparison\n\n";

    std::cout << "• Naive implementation demonstrates fundamental attention scaling challenges\n";
    std::cout << "• Memory requirements prohibit long contexts in production\n";
    std::cout << "• Motivates need for memory-efficient attention algorithms\n";
    std::cout << "• Establishes performance baseline for optimization techniques\n";

    return 0;
}
