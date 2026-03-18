#pragma once

#include <mutex>
#include <vector>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::runtime {

class BlockManager {
public:
    BlockManager(i32 num_blocks, i32 block_size, bool thread_safe = false);

    Result<i32> allocate_block();

    Result<std::vector<i32>> allocate_blocks(i32 num_blocks_needed);

    Result<void> free_block(i32 block_id);

    Result<void> free_blocks(const std::vector<i32>& block_ids);

    void reset();

private:
    class LockGuard {
    public:
        LockGuard(std::mutex& mutex, bool enable) : mutex_(mutex), enable_(enable) {
            if (enable_) {
                mutex_.lock();
            }
        }

        ~LockGuard() {
            if (enable_) {
                mutex_.unlock();
            }
        }

    private:
        std::mutex& mutex_;
        bool enable_;
    };

    // Total number of blocks
    i32 num_blocks_;

    i32 block_size_;

    bool thread_safe_;

    std::vector<i32> free_blocks_;
    std::vector<bool> allocated_;

    mutable std::mutex mutex_;

    bool is_valid_block_id(i32 block_id) {
        return block_id >= 0 && block_id < num_blocks_;
    }
};

}  // namespace pulse::runtime
