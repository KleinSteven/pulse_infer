#pragma once

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"
#include "pulse/core/types.hpp"
#include "pulse/runtime/block_manager.hpp"
#include "pulse/runtime/block_table.hpp"

namespace pulse::runtime {

class KVCacheManager {
public:
    KVCacheManager(i32 num_blocks,
                   i32 block_size,
                   i32 num_layers,
                   i32 num_kv_heads,
                   i32 head_size,
                   DeviceType device);


    Result<void> init();


    Result<i32> allocate_sequence(i32 seq_id, i32 num_tokens);


    /// Extend sequence by allocating additional blocks
    Result<i32> extend_sequence(i32 seq_id, i32 additional_tokens);


    Result<i32> free_sequence(i32 seq_id);


    Tensor& get_key_cache(i32 layer_idx);


    Tensor& get_value_cache(i32 layer_idx);

    /**
     * @brief Get block table in GPU format for a batch of sequences
     *
     * Prepares the block table for GPU kernels:
     * - Flat array: [seq_0_blocks..., seq_1_blocks..., ...]
     * - Padded with -1 to max_blocks_per_seq
     * - Returns CPU tensor that can be copied to GPU
     *
     * @param seq_ids Ordered list of sequence IDs
     * @return CPU Tensor with shape [num_seqs, max_blocks_per_seq], dtype=Int32
     */
    Result<Tensor> get_block_table_tensor(const std::vector<i32>& seq_ids);


    /// Get sequence lengths (in tokens)
    Result<std::vector<i32>> get_sequence_lengths(const std::vector<i32>& seq_ids) const;


    Result<i32> get_sequence_capacity(i32 seq_id) const;


    bool is_sequence_allocated(i32 seq_id) const;


    Result<void> update_sequence_length(i32 seq_id, i32 new_token_count);

    void reset();

    [[nodiscard]] i32 num_blocks() const noexcept {
        return num_blocks_;
    }

    [[nodiscard]] i32 block_size() const noexcept {
        return block_size_;
    }

    [[nodiscard]] i32 num_layers() const noexcept {
        return num_layers_;
    }

    [[nodiscard]] i32 num_kv_heads() const noexcept {
        return num_kv_heads_;
    }

    [[nodiscard]] i32 head_size() const noexcept {
        return head_size_;
    }

    [[nodiscard]] i32 kv_dim() const noexcept {
        return num_kv_heads_ * head_size_;
    }


    i32 get_max_blocks_per_seq() const;

private:
    // Total number of physical blocks available at each layer, not the total number of
    // physical blocks all layers
    i32 num_blocks_;

    // Tokens per block
    i32 block_size_;

    // Number of transformer layers
    i32 num_layers_;

    // Number of KV heads
    i32 num_kv_heads_;

    // Size of each head
    i32 head_size_;

    // Device for allocation
    DeviceType device_;

    std::unique_ptr<runtime::BlockManager> block_manager_;
    std::unique_ptr<runtime::BlockTable> block_table_;

    // KV cache tensors: one Tensor per layer
    // Each Tensor's shape is [num_blocks, num_kv_heads, block_size, head_size]
    std::vector<Tensor> key_caches_;
    std::vector<Tensor> value_caches_;

    // Record sequence token counts
    std::unordered_map<i32, i32> seq_num_tokens_;

    bool initialized_ = false;

    i32 calculate_num_blocks_needed(i32 num_tokens) const {
        return (num_tokens + block_size_ - 1) / block_size_;
    }
};

}  // namespace pulse::runtime
