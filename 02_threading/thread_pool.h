#include <vector>
#include <thread>
#include <queue>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <functional>

class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();  // join all threads

    template<typename F>
    std::future<void> enqueue(F&& task) {
        auto task_ptr = std::make_shared<std::packaged_task<void()>>(std::forward<F>(task));
        std::future<void> result = task_ptr->get_future();
        
        {
            std::unique_lock<std::mutex> lock(mutex_);
            tasks_.push_back([task_ptr] () { (*task_ptr)(); });
        }

        cv_.notify_one();
        return result;
    }

    size_t num_workers() const;

private:
    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};

extern ThreadPool pool;

// Work decomposition — splits total_work across threads
template<typename F>
void parallel_for(size_t total_work, size_t min_work_per_thread, F work_func) {
    size_t max_chunks = std::max((size_t)1, total_work / min_work_per_thread);
    size_t num_chunks = std::min(max_chunks, pool.num_workers());

    std::vector<std::future<void>> futures;
    futures.reserve(num_chunks);

    for (size_t chunk = 0; chunk < num_chunks; ++chunk)
    {
        size_t range_per_chunk = total_work / num_chunks;
        size_t start = chunk * range_per_chunk;
        size_t end = (chunk == num_chunks - 1) ? total_work : start + range_per_chunk;
        futures.push_back(pool.enqueue([start, end, work_func] () {
            for (size_t curr = start; curr < end; ++curr) 
                work_func(curr);
        }));
    }

    for (auto& f: futures)
        f.get();
}

// Reduce pattern — each thread produces a partial result, then combine
template<typename F, typename T, typename Combine>
T parallel_reduce(size_t total_work, F work_func, T init, Combine combine) {
    size_t chunk_range = total_work / pool.num_workers();

    std::vector<T> partials(pool.num_workers(), init);
    std::vector<std::future<void>> futures;
    for (size_t chunk = 0; chunk < pool.num_workers(); ++chunk)
    {
        size_t start = chunk * chunk_range;
        size_t end = (chunk == pool.num_workers() - 1) ? total_work : start + chunk_range;
        futures.push_back(pool.enqueue([chunk, start, end, work_func, &partials, combine] () { 
            for (size_t curr = start; curr < end; ++curr)
                partials[chunk] = combine(partials[chunk], work_func(curr));
        }));
    }

    for (auto& f: futures)
        f.get();

    T output = init;
    for (size_t i = 0; i < partials.size(); ++i)
    {
        output = combine(output, partials[i]);
    }

    
    return output;
}