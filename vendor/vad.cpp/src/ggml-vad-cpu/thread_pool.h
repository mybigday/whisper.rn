// Small persistent thread pool for the vad-cpu backend.
// Workers spin briefly after each job (VAD graphs dispatch many short steps)
// and then sleep on a condition variable; a dispatch only pays for a wake-up
// when some worker is actually asleep.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#define VADK_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define VADK_PAUSE() __asm__ __volatile__("yield")
#else
#define VADK_PAUSE() ((void) 0)
#endif

namespace vadk {

class thread_pool {
public:
    explicit thread_pool(int n_threads) {
        slots_ = std::vector<slot>(n_threads);
        for (int i = 1; i < n_threads; ++i) workers_.emplace_back([this, i] { loop(i); });
    }

    ~thread_pool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            gen_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        for (auto & t : workers_) t.join();
    }

    int size() const { return (int) workers_.size() + 1; }

    // Runs fn(i) for i in [0, n) on all threads; returns when every task is done.
    // Thread t first takes the tasks [n*t/T, n*(t+1)/T): consecutive steps split the same
    // way, so a thread mostly reads rows it wrote itself (still in its cache). Then it
    // steals from the others. Completion is counted per task, not per worker, so a
    // worker that wakes late (asleep or preempted) never holds the caller up.
    void run(int n, const std::function<void(int)> & fn) {
        if (n <= 0) return;
        if (workers_.empty() || n == 1) { for (int i = 0; i < n; ++i) fn(i); return; }
        job_ = &fn;
        done_.store(0, std::memory_order_relaxed);
        const uint64_t tag = ++tag_ & TAG_MASK;
        const int T = size();
        for (int t = 0; t < T; ++t) {
            const uint64_t b = (uint64_t) n * t / T, e = (uint64_t) n * (t + 1) / T;
            slots_[t].v.store(tag << 40 | e << 20 | b, std::memory_order_release);
        }
        gen_.fetch_add(1, std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_seq_cst) > 0) {
            { std::lock_guard<std::mutex> lk(mu_); }   // a sleeper is either before its predicate check or waiting
            cv_.notify_all();
        }
        work(0);
        while (done_.load(std::memory_order_acquire) != n) VADK_PAUSE();
        job_ = nullptr;
    }

private:
    static constexpr uint64_t IDX_MASK = (1u << 20) - 1, TAG_MASK = (1u << 24) - 1;

    // slot word: tag (24 bits) | end (20) | next (20); a claim is one CAS on it, so a
    // straggler holding a word of an earlier job can never claim a task of this one
    struct alignas(64) slot { std::atomic<uint64_t> v{0}; };

    bool claim(int t) {
        uint64_t v = slots_[t].v.load(std::memory_order_acquire);
        while ((v & IDX_MASK) < ((v >> 20) & IDX_MASK)) {
            if (slots_[t].v.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                // the task is unfinished, so the caller is still waiting and job_ is this job's
                (*job_)((int) (v & IDX_MASK));
                done_.fetch_add(1, std::memory_order_acq_rel);
                return true;
            }
        }
        return false;
    }

    void work(int self) {
        const int T = size();
        while (claim(self)) {}
        for (int k = 1; k < T; ++k) {
            const int t = (self + k) % T;
            while (claim(t)) {}
        }
    }

    void loop(int self) {
        unsigned seen = gen_.load(std::memory_order_acquire);
        while (true) {
            // spin for ~250 us, then sleep. Timed, not counted: a pause/yield costs
            // ~60 cycles on Zen but ~1 on Cortex-A cores, where a fixed count let the
            // workers fall asleep between the steps of one compute.
            const auto t0 = std::chrono::steady_clock::now();
            for (int spins = 0; gen_.load(std::memory_order_acquire) == seen; ++spins) {
                VADK_PAUSE();
                if ((spins & 255) == 255 && std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(250)) break;
            }
            if (gen_.load(std::memory_order_acquire) == seen) {
                std::unique_lock<std::mutex> lk(mu_);
                sleepers_.fetch_add(1, std::memory_order_seq_cst);
                cv_.wait(lk, [&] { return gen_.load(std::memory_order_seq_cst) != seen; });
                sleepers_.fetch_sub(1, std::memory_order_relaxed);
            }
            seen = gen_.load(std::memory_order_acquire);
            if (stop_) return;
            work(self);
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_;
    // one cache line each: spinning readers of gen_ must not be invalidated by the
    // task counter and completion count every dispatch hammers
    alignas(64) std::atomic<unsigned> gen_{0};
    alignas(64) std::atomic<int> done_{0};
    alignas(64) std::atomic<int> sleepers_{0};
    alignas(64) const std::function<void(int)> * job_ = nullptr;
    std::vector<slot> slots_;
    uint64_t tag_ = 0;   // caller only
    std::atomic<bool> stop_{false};
};

} // namespace vadk
