#include "pulse/runtime/block_table.hpp"

#include "pulse/logging.hpp"

namespace pulse::runtime {

BlockTable::BlockTable(bool thread_safe) : thread_safe_(thread_safe) {
    pulse::info("BlockTable initialized, thread-safe: {}", thread_safe);
}

Result<void> BlockTable::allocate_sequence(i32 seq_id, const std::vector<i32>& block_ids) {
    LockGuard lock(mutex_, thread_safe_);

    if (seq_to_blocks_.find(seq_id) != seq_to_blocks_.end()) {
        return Err<void>(ErrorCode::InvalidArgument, std::format("Sequence {} already exists", seq_id));
    }

    seq_to_blocks_[seq_id] = block_ids;

    return Ok();
}

Result<void> BlockTable::append_block(i32 seq_id, i32 block_id) {
    LockGuard lock(mutex_, thread_safe_);

    auto it = seq_to_blocks_.find(seq_id);
    if (it == seq_to_blocks_.end()) {
        return Err<void>(ErrorCode::InvalidArgument, std::format("Sequence {} does not exist", seq_id));
    }

    it->second.push_back(block_id);

    return Ok();
}

Result<void> BlockTable::append_blocks(i32 seq_id, const std::vector<i32>& block_ids) {
    LockGuard lock(mutex_, thread_safe_);

    auto it = seq_to_blocks_.find(seq_id);
    if (it == seq_to_blocks_.end()) {
        return Err<void>(ErrorCode::InvalidArgument, std::format("Sequence {} does not exist", seq_id));
    }

    it->second.insert(it->second.end(), block_ids.begin(), block_ids.end());

    return Ok();
}

Result<i32> BlockTable::get_num_blocks(i32 seq_id) const {
    LockGuard lock(mutex_, thread_safe_);

    auto it = seq_to_blocks_.find(seq_id);
    if (it == seq_to_blocks_.end()) {
        return Err<i32>(ErrorCode::InvalidArgument, std::format("Sequence {} does not exist", seq_id));
    }

    return Ok(static_cast<i32>(it->second.size()));
}

Result<std::vector<i32>> BlockTable::free_sequence(i32 seq_id) {
    LockGuard lock(mutex_, thread_safe_);

    auto it = seq_to_blocks_.find(seq_id);
    if (it == seq_to_blocks_.end()) {
        return Err<std::vector<i32>>(ErrorCode::InvalidArgument,
                                     std::format("Sequence {} does not exist", seq_id));
    }

    std::vector<i32> freed_blocks(std::move(it->second));

    seq_to_blocks_.erase(it);

    return Ok(std::move(freed_blocks));
}

bool BlockTable::has_sequence(i32 seq_id) const {
    LockGuard lock(mutex_, thread_safe_);
    return seq_to_blocks_.find(seq_id) != seq_to_blocks_.end();
}

std::vector<i32> BlockTable::get_sequence_ids() const {
    LockGuard lock(mutex_, thread_safe_);

    std::vector<i32> seq_ids;
    seq_ids.reserve(seq_to_blocks_.size());

    for (const auto& [seq_id, _] : seq_to_blocks_) {
        seq_ids.push_back(seq_id);
    }

    return seq_ids;
}

Result<Tensor> BlockTable::to_gpu_format(const std::vector<i32>& seq_ids, i32 max_blocks_per_seq) const {
    LockGuard lock(mutex_, thread_safe_);

    if (seq_ids.empty()) {
        return Err<Tensor>(ErrorCode::InvalidArgument, "seq_ids cannot be empty");
    }

    if (max_blocks_per_seq <= 0) {
        return Err<Tensor>(ErrorCode::InvalidArgument, "max_blocks_per_seq must be positive");
    }

    i32 num_seqs = static_cast<i32>(seq_ids.size());

    // Create CPU tensor: [num_seqs, max_blocks_per_seq]
    auto tensor_result = Tensor::create({num_seqs, max_blocks_per_seq}, DataType::Int32, DeviceType::CPU);
    if (!tensor_result) {
        return Err<Tensor>(tensor_result.error());
    }

    Tensor block_table_cpu = std::move(tensor_result.value());
    i32* data = block_table_cpu.ptr<i32>();

    // Fill with -1 (padding value)
    std::fill(data, data + num_seqs * max_blocks_per_seq, -1);

    // Copy block IDs for each sequence
    for (i32 i = 0; i < num_seqs; ++i) {
        i32 seq_id = seq_ids[static_cast<u32>(i)];

        auto it = seq_to_blocks_.find(seq_id);
        if (it == seq_to_blocks_.end()) {
            return Err<Tensor>(ErrorCode::InvalidArgument,
                               std::format("Sequence {} not found in block table", seq_id));
        }

        const auto& blocks = it->second;
        i32 num_blocks = static_cast<i32>(blocks.size());

        if (num_blocks > max_blocks_per_seq) {
            return Err<Tensor>(ErrorCode::InvalidArgument,
                               std::format("Sequence {} has {} blocks, exceeds max {}",
                                           seq_id,
                                           num_blocks,
                                           max_blocks_per_seq));
        }

        // Copy block IDs to flat array
        i32* seq_data = data + i * max_blocks_per_seq;
        for (i32 j = 0; j < num_blocks; ++j) {
            seq_data[j] = blocks[static_cast<u32>(j)];
        }
    }

    return Ok(std::move(block_table_cpu));
}

void BlockTable::reset() {
    LockGuard lock(mutex_, thread_safe_);

    i32 num_seqs = static_cast<i32>(seq_to_blocks_.size());
    seq_to_blocks_.clear();

    pulse::info("BlockTable reset: {} sequences cleared", num_seqs);
}

}  // namespace pulse::runtime
