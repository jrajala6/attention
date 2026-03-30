#include "thread_pool.h"


ThreadPool pool(std::thread::hardware_concurrency());

ThreadPool::ThreadPool(size_t num_threads)
{
    stop_ = false;
    for (size_t i = 0; i < num_threads; ++i)
    {
        workers_.emplace_back(
            [this, i]()
            {
                while (true)
                {
                    std::function<void()> task;
                    {   
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty())
                            return;
                        task = std::move(tasks_.front());
                        tasks_.pop_front();
                    }
                    task();
                }
            }
        );
    }
}

ThreadPool::~ThreadPool()
{
    {   
        std::unique_lock<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();

    for (auto& worker: workers_)
        worker.join();
}

size_t ThreadPool::num_workers() const 
{
    return workers_.size();
}

