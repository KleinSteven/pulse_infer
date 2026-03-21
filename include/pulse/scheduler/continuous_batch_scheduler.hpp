#pragma once

#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pulse/core/types.hpp"
#include "pulse/scheduler/inference_request.hpp"

namespace pulse::scheduler {

struct ScheduledBatch {
    std::vector<InferenceRequestPtr> requests;

    [[nodiscard]] i32 batch_size() const noexcept {
        return static_cast<i32>(requests.size());
    }

    [[nodiscard]] bool empty() const noexcept {
        return requests.empty();
    }
};

class ContinuousBatchScheduler {
public:
    struct Stats {
        i32 num_running = 0;
        i32 num_waiting = 0;
        i32 num_finished = 0;
        i32 total_requests = 0;
    };

    ContinuousBatchScheduler(i32 max_batch_size, i32 max_sequences, i32 chunk_size = 256)
        : max_batch_size_(max_batch_size), max_sequences_(max_sequences), chunk_size_(chunk_size) {}

    [[nodiscard]] i64 add_request(const std::string& prompt,
                                  const std::vector<i32>& prompt_tokens,
                                  i32 max_new_tokens) {
        const i64 request_id = next_request_id_++;
        auto request = std::make_shared<InferenceRequest>(request_id, prompt, prompt_tokens, max_new_tokens);
        waiting_queue_.push_back(request);
        request_map_[request_id] = request;
        return request_id;
    }

    [[nodiscard]] ScheduledBatch schedule_next_batch() {
        ScheduledBatch batch;

        for (const auto& request : running_requests_) {
            batch.requests.push_back(request);
        }

        i32 remaining_capacity = max_batch_size_ - batch.batch_size();

        i32 available_slots = max_sequences_ - num_running();

        while (remaining_capacity > 0 && available_slots > 0 && !waiting_queue_.empty()) {
            auto req = waiting_queue_.front();
            waiting_queue_.pop_front();

            req->start_running();
            running_requests_.push_back(req);
            batch.requests.push_back(req);

            remaining_capacity--;
            available_slots--;
        }

        return batch;
    }

    void update_after_step(const std::vector<i64>& finished_ids) {
        for (i64 request_id : finished_ids) {
            auto it = std::find_if(running_requests_.begin(),
                                   running_requests_.end(),
                                   [request_id](const InferenceRequestPtr& request) {
                                       return request->request_id() == request_id;
                                   });
            if (it == running_requests_.end()) {
                continue;
            }

            (*it)->finish();
            running_requests_.erase(it);
        }
    }

    [[nodiscard]] InferenceRequestPtr get_request(i64 request_id) const {
        const auto it = request_map_.find(request_id);
        if (it == request_map_.end()) {
            return nullptr;
        }

        return it->second;
    }

    [[nodiscard]] bool has_work() const noexcept {
        return !running_requests_.empty() || !waiting_queue_.empty();
    }

    [[nodiscard]] i32 num_active_sequences() const noexcept {
        return num_running();
    }

    [[nodiscard]] i32 num_running() const noexcept {
        return static_cast<i32>(running_requests_.size());
    }

    [[nodiscard]] i32 num_waiting() const noexcept {
        return static_cast<i32>(waiting_queue_.size());
    }

    [[nodiscard]] i32 chunk_size() const noexcept {
        return chunk_size_;
    }

    void set_chunk_size(i32 chunk_size) noexcept {
        chunk_size_ = chunk_size;
    }

    [[nodiscard]] Stats get_stats() const noexcept {
        Stats stats;
        stats.num_running = num_running();
        stats.num_waiting = num_waiting();
        stats.total_requests = static_cast<i32>(request_map_.size());
        for (const auto& [request_id, request] : request_map_) {
            (void)request_id;
            if (request->is_finished()) {
                stats.num_finished++;
            }
        }
        return stats;
    }

private:
    i32 max_batch_size_ = 0;
    i32 max_sequences_ = 0;
    i32 chunk_size_ = 256;
    i64 next_request_id_ = 0;

    std::deque<InferenceRequestPtr> waiting_queue_;
    std::vector<InferenceRequestPtr> running_requests_;
    std::unordered_map<i64, InferenceRequestPtr> request_map_;
};

}  // namespace pulse::scheduler
