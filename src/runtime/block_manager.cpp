#include "pulse/runtime/block_manager.hpp"

#include "pulse/logging.hpp"

namespace pulse::runtime {
BlockManager::BlockManager(i32 num_blocks, i32 block_size, bool thread_safe)
    : num_blocks_(num_blocks), block_size_(block_size), thread_safe_(thread_safe) {
    free_blocks_.reserve(static_cast<u32>(num_blocks_));
    allocated_.reserve(static_cast<u32>(num_blocks_));

    for (i32 i = 0; i < num_blocks; ++i) {
        free_blocks_.push_back(i);
        allocated_[static_cast<u32>(i)] = false;
    }

    pulse::info("BlockManager initialized:");
    pulse::info("   Total blocks: {}", num_blocks);
    pulse::info("   Block size: {}", block_size);
    pulse::info("   Total capacity: {}", num_blocks * block_size);
    pulse::info("   Thread safe: {}", thread_safe);
}

Result<i32> BlockManager::allocate_block() {
    LockGuard lock(mutex_, thread_safe_);

    if (free_blocks_.empty()) {
        return Err<i32>(ErrorCode::OutOfMemory, "No free blocks available");
    }

    i32 id = free_blocks_.back();
    free_blocks_.pop_back();

    allocated_[static_cast<u32>(id)] = true;

    return Ok(id);
}

Result<std::vector<i32>> BlockManager::allocate_blocks(i32 num_blocks_needed) {
    LockGuard lock(mutex_, thread_safe_);

    if (num_blocks_needed <= 0) {
        return Err<std::vector<i32>>(ErrorCode::InvalidArgument, "Number of blocks must be positive");
    }

    if (free_blocks_.empty() || free_blocks_.size() < static_cast<u32>(num_blocks_needed)) {
        return Err<std::vector<i32>>(ErrorCode::OutOfMemory, "Insufficient free blocks");
    }

    std::vector<i32> allocated_block_ids(static_cast<u32>(num_blocks_needed));

    for (i32 i = 0; i < num_blocks_needed; i++) {
        i32 id = free_blocks_.back();
        free_blocks_.pop_back();
        allocated_[static_cast<u32>(id)] = true;
        allocated_block_ids.push_back(id);
    }

    return Ok(std::move(allocated_block_ids));
}

Result<void> BlockManager::free_block(i32 block_id) {
    LockGuard lock(mutex_, thread_safe_);

    if (!is_valid_block_id(block_id)) {
        return Err<void>(ErrorCode::InvalidArgument, std::format("Invalid block ID: {}", block_id));
    }

    if (!allocated_[static_cast<u32>(block_id)]) {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("Block {}, is not allocated (double free?)", block_id));
    }

    free_blocks_.push_back(block_id);
    allocated_[static_cast<u32>(block_id)] = false;

    return Ok();
}

Result<void> BlockManager::free_blocks(const std::vector<i32>& block_ids) {
    LockGuard lock(mutex_, thread_safe_);

    for (i32 block_id : block_ids) {
        if (!is_valid_block_id(block_id)) {
            return Err<void>(ErrorCode::InvalidArgument,
                             std::format("Invalid block ID in batch: {}", block_id));
        }

        if (!allocated_[static_cast<u32>(block_id)]) {
            return Err<void>(ErrorCode::InvalidArgument, std::format("Block {} is not allocated", block_id));
        }
    }

    for (i32 block_id : block_ids) {
        free_blocks_.push_back(block_id);
        allocated_[static_cast<u32>(block_id)] = false;
    }

    return Ok();
}

void BlockManager::reset() {
    LockGuard lock(mutex_, thread_safe_);

    free_blocks_.clear();
    free_blocks_.reserve(static_cast<u32>(num_blocks_));

    for (i32 i = 0; i < num_blocks_; ++i) {
        allocated_[static_cast<u32>(i)] = false;
        free_blocks_.push_back(i);
    }

    pulse::info("BlockManager reset: all {} blocks freed", num_blocks_);
}

}  // namespace pulse::runtime
