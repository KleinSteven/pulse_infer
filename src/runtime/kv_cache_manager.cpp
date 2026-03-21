#include "pulse/runtime/kv_cache_manager.hpp"

#include <algorithm>

#include "pulse/logging.hpp"

namespace pulse::runtime {

KVCacheManager::KVCacheManager(i32 num_blocks,
                               i32 block_size,
                               i32 num_layers,
                               i32 num_kv_heads,
                               i32 head_size,
                               DeviceType device,
                               DataType dtype)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      device_(device),
      dtype_(dtype) {
    block_manager_ = std::make_unique<BlockManager>(num_blocks, block_size);

    block_table_ = std::make_unique<BlockTable>();

    pulse::info("KVCacheManager created:");
    pulse::info("   Num blocks: {}", num_blocks);
    pulse::info("   Block size: {} tokens", block_size);
    pulse::info("   Total capacity: {} tokens", num_blocks * block_size);
    pulse::info("   Num layers: {}", num_layers);
    pulse::info("   Num KV heads: {}", num_kv_heads);
    pulse::info("   Head size: {}", head_size);
    pulse::info("   KV dim: {}", num_kv_heads * head_size);
}

Result<void> KVCacheManager::init() {
    if (initialized_) {
        return Err<void>(ErrorCode::InvalidOperator, "KVCacheManager already initialized");
    }

    i64 elements_per_block = static_cast<i64>(num_kv_heads_) * block_size_ * head_size_;
    i64 elements_per_layer = elements_per_block * num_blocks_;
    i64 bytes_per_layer = elements_per_layer * static_cast<i64>(data_type_size(dtype_));
    i64 total_bytes = bytes_per_layer * num_layers_ * 2;

    pulse::info("Initializing block-based KV cache:");
    pulse::info("   Elements per block: {}", elements_per_block);
    pulse::info("   Elements per layer: {}", elements_per_layer);
    pulse::info("   Bytes per layer: {} MB", static_cast<f64>(bytes_per_layer) / (1024.0 * 1024.0));
    pulse::info("   Total cache size: {} MB", static_cast<f64>(total_bytes) / (1024.0 * 1024.0));

    key_caches_.reserve(static_cast<u32>(num_layers_));
    value_caches_.reserve(static_cast<u32>(num_layers_));

    for (i32 layer = 0; layer < num_layers_; ++layer) {
        // key cache: [num_blocks, num_kv_heads, block_size, head_size]
        auto key_result =
            Tensor::create({num_blocks_, num_kv_heads_, block_size_, head_size_}, dtype_, device_);

        if (!key_result) {
            return Err<void>(std::move(key_result.error()));
        }

        key_caches_.push_back(std::move(key_result.value()));

        // value cache: [num_blocks, num_kv_heads, block_size, head_size]
        auto value_result =
            Tensor::create({num_blocks_, num_kv_heads_, block_size_, head_size_}, dtype_, device_);

        if (!value_result) {
            return Err<void>(std::move(value_result.error()));
        }

        value_caches_.push_back(std::move(value_result.value()));
    }

    initialized_ = true;

    return Ok();
}

Result<i32> KVCacheManager::allocate_sequence(i32 seq_id, i32 num_tokens) {
    if (!initialized_) {
        return Err<i32>(ErrorCode::InvalidOperator, "KVCacheManager not initialized");
    }

    if (block_table_->has_sequence(seq_id)) {
        return Err<i32>(ErrorCode::InvalidArgument, std::format("Sequence {} already allocated", seq_id));
    }

    i32 num_blocks_needed = calculate_num_blocks_needed(num_tokens);

    auto blocks_result = block_manager_->allocate_blocks(num_blocks_needed);
    if (!blocks_result) {
        return Err<i32>(std::move(blocks_result.error()));
    }

    std::vector<i32> block_ids = std::move(blocks_result.value());

    // Register blocks in block table
    auto table_result = block_table_->allocate_sequence(seq_id, block_ids);
    if (!table_result) {
        // Rollback: free allocated blocks
        auto free_result = block_manager_->free_blocks(block_ids);

        if (!free_result) {
            return Err<i32>(std::move(free_result.error()));
        }

        return Err<i32>(std::move(table_result.error()));
    }

    // Record sequence token count
    seq_num_tokens_[seq_id] = num_tokens;

    return Ok(num_blocks_needed);
}

Result<i32> KVCacheManager::extend_sequence(i32 seq_id, i32 additional_tokens) {
    if (!initialized_) {
        return Err<i32>(ErrorCode::InvalidOperator, "KVCacheManager not initialized");
    }

    if (!block_table_->has_sequence(seq_id)) {
        return Err<i32>(ErrorCode::InvalidArgument, std::format("Sequence {} not found", seq_id));
    }

    auto current_blocks_result = block_table_->get_num_blocks(seq_id);
    if (!current_blocks_result) {
        return Err<i32>(std::move(current_blocks_result.error()));
    }

    i32 current_num_blocks = current_blocks_result.value();

    i32 current_tokens = seq_num_tokens_[seq_id];
    i32 new_total_tokens = current_tokens + additional_tokens;

    i32 total_blocks_needed = calculate_num_blocks_needed(new_total_tokens);
    i32 additional_blocks_needed = total_blocks_needed - current_num_blocks;

    if (additional_blocks_needed <= 0) {
        // Current allocation is sufficient
        seq_num_tokens_[seq_id] = new_total_tokens;

        return Ok(0);
    }

    // Allocate additional blocks
    auto blocks_result = block_manager_->allocate_blocks(additional_blocks_needed);
    if (!blocks_result) {
        return Err<i32>(blocks_result.error());
    }

    std::vector<i32> new_block_ids = std::move(blocks_result.value());

    auto append_result = block_table_->append_blocks(seq_id, new_block_ids);
    if (!append_result) {
        // Rollback: free newly allocated blocks
        auto free_result = block_manager_->free_blocks(new_block_ids);

        if (!free_result) {
            return Err<i32>(std::move(free_result.error()));
        }

        return Err<i32>(std::move(append_result.error()));
    }

    // Update token count
    seq_num_tokens_[seq_id] = new_total_tokens;

    return Ok(additional_blocks_needed);
}

Result<i32> KVCacheManager::free_sequence(i32 seq_id) {
    if (!block_table_->has_sequence(seq_id)) {
        return Err<i32>(ErrorCode::InvalidArgument, std::format("Sequence {} not found", seq_id));
    }

    auto blocks_result = block_table_->free_sequence(seq_id);
    if (!blocks_result) {
        return Err<i32>(std::move(blocks_result.error()));
    }

    std::vector<i32> block_ids = std::move(blocks_result.value());
    i32 num_blocks_freed = static_cast<i32>(block_ids.size());

    auto free_result = block_manager_->free_blocks(block_ids);
    if (!free_result) {
        return Err<i32>(std::move(free_result.error()));
    }

    seq_num_tokens_.erase(seq_id);

    return Ok(num_blocks_freed);
}

Tensor& KVCacheManager::get_key_cache(i32 layer_idx) {
    return key_caches_[static_cast<u32>(layer_idx)];
}

Tensor& KVCacheManager::get_value_cache(i32 layer_idx) {
    return value_caches_[static_cast<u32>(layer_idx)];
}

Result<Tensor> KVCacheManager::get_block_table_tensor(const std::vector<i32>& seq_ids) {
    if (seq_ids.empty()) {
        return Err<Tensor>(ErrorCode::InvalidArgument, "seq_ids cannot be empty");
    }

    // Calculate max blocks per sequence
    i32 max_blocks_per_seq = get_max_blocks_per_seq();

    // Get block table in GPU format (CPU tensor)
    return block_table_->to_gpu_format(seq_ids, max_blocks_per_seq);
}

Result<std::vector<i32>> KVCacheManager::get_sequence_lengths(const std::vector<i32>& seq_ids) const {
    std::vector<i32> lengths;
    lengths.reserve(seq_ids.size());

    for (i32 seq_id : seq_ids) {
        auto it = seq_num_tokens_.find(seq_id);
        if (it == seq_num_tokens_.end()) {
            return Err<std::vector<i32>>(ErrorCode::InvalidArgument,
                                         std::format("Sequence {} not found", seq_id));
        }
        lengths.push_back(it->second);
    }

    return Ok(std::move(lengths));
}

Result<void> KVCacheManager::update_sequence_length(i32 seq_id, i32 new_token_count) {
    auto it = seq_num_tokens_.find(seq_id);
    if (it == seq_num_tokens_.end()) {
        return Err<void>(ErrorCode::InvalidArgument, std::format("Sequence {} not allocated", seq_id));
    }

    auto capacity_result = get_sequence_capacity(seq_id);
    if (!capacity_result) {
        return Err<void>(capacity_result.error());
    }

    if (new_token_count > capacity_result.value()) {
        return Err<void>(
            ErrorCode::InvalidArgument,
            std::format("Token count {} exceeds capacity {}", new_token_count, capacity_result.value()));
    }

    seq_num_tokens_[seq_id] = new_token_count;
    return Ok();
}

Result<i32> KVCacheManager::get_sequence_capacity(i32 seq_id) const {
    auto num_blocks_result = block_table_->get_num_blocks(seq_id);
    if (!num_blocks_result) {
        return Err<i32>(num_blocks_result.error());
    }
    return Ok(num_blocks_result.value() * block_size_);
}

bool KVCacheManager::is_sequence_allocated(i32 seq_id) const {
    return block_table_->has_sequence(seq_id);
}

void KVCacheManager::reset() {
    // Get all active sequences
    auto seq_ids = block_table_->get_sequence_ids();

    for (i32 seq_id : seq_ids) {
        auto result = free_sequence(seq_id);
        (void)result;
    }

    seq_num_tokens_.clear();

    block_manager_->reset();
    block_table_->reset();
}

i32 KVCacheManager::get_max_blocks_per_seq() const {
    return std::max(1, num_blocks_);
}

}  // namespace pulse::runtime
