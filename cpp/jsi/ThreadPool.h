#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

class ThreadPool {
public:
    static ThreadPool &getInstance() {
        static ThreadPool instance;
        return instance;
    }

    void ensureRunning();
    void shutdown();

    template <class F>
    void enqueue(F &&f);

    ~ThreadPool();

private:
    ThreadPool(size_t threads = defaultThreadCount());
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    static size_t defaultThreadCount() {
        auto maxThreads = std::thread::hardware_concurrency();
        return std::max<size_t>(2, std::min<size_t>(4, maxThreads == 0 ? 2 : maxThreads));
    }

    void startWorkers(size_t threads);

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::mutex shutdown_mutex;
    std::condition_variable condition;
    bool stop;
};

inline ThreadPool::ThreadPool(size_t threads)
    : stop(false) {
    startWorkers(threads);
}

inline void ThreadPool::startWorkers(size_t threads) {
    if (threads == 0) {
        threads = 1;
    }

    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty()) {
                        return;
                    }

                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
    }
}

inline void ThreadPool::ensureRunning() {
    std::unique_lock<std::mutex> lock(queue_mutex);

    if (stop) {
        stop = false;
    }

    if (!workers.empty()) {
        return;
    }

    startWorkers(defaultThreadCount());
}

inline void ThreadPool::shutdown() {
    std::unique_lock<std::mutex> shutdownLock(shutdown_mutex);

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop && workers.empty()) {
            return;
        }
        stop = true;
    }

    condition.notify_all();

    const auto selfId = std::this_thread::get_id();
    for (std::thread &worker : workers) {
        if (!worker.joinable()) {
            continue;
        }

        if (worker.get_id() == selfId) {
            worker.detach();
            continue;
        }

        try {
            worker.join();
        } catch (const std::system_error &) {
            worker.detach();
        }
    }

    workers.clear();

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        std::queue<std::function<void()>> empty;
        std::swap(tasks, empty);
    }
}

template <class F>
void ThreadPool::enqueue(F &&f) {
    ensureRunning();

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }

        tasks.emplace(std::forward<F>(f));
    }

    condition.notify_one();
}

inline ThreadPool::~ThreadPool() {
    shutdown();
}

#endif
