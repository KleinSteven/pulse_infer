#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"
#include "pulse/core/types.hpp"

namespace pulse::runtime {

class BlockTable {
public:
    explicit BlockTable(bool thread_safe = false);

    Result<void> allocate_sequence(i32 seq_id, const std::vector<i32>& block_ids);

    Result<void> append_block(i32 seq_id, i32 block_id);

    Result<void> append_blocks(i32 seq_id, const std::vector<i32>& block_ids);

    Result<i32> get_num_blocks(i32 seq_id) const;

    Result<std::vector<i32>> free_sequence(i32 seq_id);

    bool has_sequence(i32 seq_id) const;

    std::vector<i32> get_sequence_ids() const;

    /**
     * @brief Prepare block table in GPU format (flat array with padding)
     *
     * This converts the block table to a format suitable for GPU kernels:
     * - Flat array: [seq_0_blocks..., seq_1_blocks..., ...]
     * - Each sequence's blocks are padded to max_blocks_per_seq with -1
     * - Shape: [num_seqs, max_blocks_per_seq]
     *
     * @param seq_ids Ordered list of sequence IDs to include
     * @param max_blocks_per_seq Maximum blocks per sequence (for padding)
     * @return Tensor on CPU with shape [num_seqs, max_blocks_per_seq]
     */
    Result<Tensor> to_gpu_format(const std::vector<i32>& seq_ids, i32 max_blocks_per_seq) const;

    void reset();

private:
    class LockGuard {
    public:
        LockGuard(std::mutex& mutex, bool enabled) : mutex_(mutex), enabled_(enabled) {
            if (enabled_)
                mutex_.lock();
        }
        ~LockGuard() {
            if (enabled_)
                mutex_.unlock();
        }

    private:
        std::mutex& mutex_;
        bool enabled_;
    };

    bool thread_safe_;

    /// Mapping: seq_id -> [physical_block_0, physical_block_1, ...]
    std::unordered_map<i32, std::vector<i32>> seq_to_blocks_;

    mutable std::mutex mutex_;
};

}  // namespace pulse::runtime
