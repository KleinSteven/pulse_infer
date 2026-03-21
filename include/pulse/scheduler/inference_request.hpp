#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::scheduler {

enum class RequestState {
    WAITING,
    RUNNING,
    FINISHED,
};

class InferenceRequest {
public:
    InferenceRequest(i64 request_id,
                     std::string prompt,
                     std::vector<i32> prompt_tokens,
                     i32 max_new_tokens)
        : request_id_(request_id),
          prompt_(std::move(prompt)),
          prompt_tokens_(std::move(prompt_tokens)),
          max_new_tokens_(max_new_tokens),
          arrival_time_(std::chrono::steady_clock::now()) {}

    [[nodiscard]] i64 request_id() const noexcept {
        return request_id_;
    }

    [[nodiscard]] const std::string& prompt() const noexcept {
        return prompt_;
    }

    [[nodiscard]] const std::vector<i32>& prompt_tokens() const noexcept {
        return prompt_tokens_;
    }

    [[nodiscard]] const std::vector<i32>& generated_tokens() const noexcept {
        return generated_tokens_;
    }

    [[nodiscard]] i32 max_new_tokens() const noexcept {
        return max_new_tokens_;
    }

    [[nodiscard]] RequestState state() const noexcept {
        return state_;
    }

    [[nodiscard]] i32 prompt_len() const noexcept {
        return static_cast<i32>(prompt_tokens_.size());
    }

    [[nodiscard]] i32 num_generated() const noexcept {
        return static_cast<i32>(generated_tokens_.size());
    }

    [[nodiscard]] i32 num_computed_tokens() const noexcept {
        return num_computed_tokens_;
    }

    [[nodiscard]] i32 total_input_tokens() const noexcept {
        return prompt_len() + num_generated();
    }

    [[nodiscard]] bool is_finished() const noexcept {
        return state_ == RequestState::FINISHED;
    }

    [[nodiscard]] bool is_prefill() const noexcept {
        return num_computed_tokens_ < prompt_len();
    }

    [[nodiscard]] bool has_pending_input() const noexcept {
        return !is_finished() && num_computed_tokens_ < total_input_tokens();
    }

    [[nodiscard]] bool can_generate_more() const noexcept {
        return !is_finished() && num_generated() < max_new_tokens_;
    }

    [[nodiscard]] auto arrival_time() const noexcept {
        return arrival_time_;
    }

    [[nodiscard]] auto start_time() const noexcept {
        return start_time_;
    }

    [[nodiscard]] auto finish_time() const noexcept {
        return finish_time_;
    }

    void start_running() noexcept {
        state_ = RequestState::RUNNING;
        if (start_time_.time_since_epoch().count() == 0) {
            start_time_ = std::chrono::steady_clock::now();
        }
    }

    void finish() noexcept {
        state_ = RequestState::FINISHED;
        finish_time_ = std::chrono::steady_clock::now();
    }

    [[nodiscard]] Result<void> add_computed_tokens(i32 count) {
        if (count <= 0) {
            return Err<void>(ErrorCode::InvalidArgument, "Computed token count must be positive");
        }

        if (is_finished()) {
            return Err<void>(ErrorCode::InvalidOperator, "Request already finished");
        }

        if (num_computed_tokens_ + count > total_input_tokens()) {
            return Err<void>(ErrorCode::InvalidArgument, "Computed token count exceeds available inputs");
        }

        num_computed_tokens_ += count;
        return Ok();
    }

    [[nodiscard]] bool add_sampled_token(i32 token, const std::vector<i32>& eos_token_ids) {
        if (is_finished()) {
            return false;
        }

        generated_tokens_.push_back(token);
        if (token == -1 || std::find(eos_token_ids.begin(), eos_token_ids.end(), token) != eos_token_ids.end() ||
            num_generated() >= max_new_tokens_) {
            finish();
            return false;
        }

        return true;
    }

    [[nodiscard]] i32 get_next_chunk_size(i32 chunk_size = 256) const noexcept {
        if (is_finished()) {
            return 0;
        }

        if (is_prefill()) {
            return std::min(chunk_size, prompt_len() - num_computed_tokens_);
        }

        return has_pending_input() ? 1 : 0;
    }

    [[nodiscard]] std::vector<i32> get_next_chunk_tokens(i32 chunk_size = 256) const {
        const i32 next_chunk_size = get_next_chunk_size(chunk_size);
        if (next_chunk_size <= 0) {
            return {};
        }

        if (is_prefill()) {
            const auto begin = prompt_tokens_.begin() + num_computed_tokens_;
            return std::vector<i32>(begin, begin + next_chunk_size);
        }

        const i32 generated_index = num_computed_tokens_ - prompt_len();
        return {generated_tokens_[static_cast<usize>(generated_index)]};
    }

    [[nodiscard]] std::vector<i32> get_next_chunk_positions(i32 chunk_size = 256) const {
        const i32 next_chunk_size = get_next_chunk_size(chunk_size);
        if (next_chunk_size <= 0) {
            return {};
        }

        std::vector<i32> positions;
        positions.reserve(static_cast<usize>(next_chunk_size));
        for (i32 index = 0; index < next_chunk_size; ++index) {
            positions.push_back(num_computed_tokens_ + index);
        }
        return positions;
    }

private:
    i64 request_id_ = 0;
    std::string prompt_;
    std::vector<i32> prompt_tokens_;
    std::vector<i32> generated_tokens_;
    i32 max_new_tokens_ = 0;
    RequestState state_ = RequestState::WAITING;
    i32 num_computed_tokens_ = 0;
    std::chrono::steady_clock::time_point arrival_time_;
    std::chrono::steady_clock::time_point start_time_{};
    std::chrono::steady_clock::time_point finish_time_{};
};

using InferenceRequestPtr = std::shared_ptr<InferenceRequest>;

}  // namespace pulse::scheduler
