#pragma once

#include <string>

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"
#include "pulse/core/types.hpp"
#include "pulse/model/qwen3/model.hpp"
#include "pulse/model/qwen3/tokenizer.hpp"
#include "pulse/scheduler/continuous_batch_scheduler.hpp"

namespace pulse::scheduler {

class ContinuousBatchEngine {
public:
    ContinuousBatchEngine(model::Qwen3Model& model,
                          const model::Qwen3Tokenizer& tokenizer,
                          i32 max_batch_size,
                          i32 chunk_size = 256);

    [[nodiscard]] i64 add_request(const std::string& prompt, i32 max_new_tokens);
    [[nodiscard]] Result<void> run_until_complete(bool print_progress = false);
    [[nodiscard]] std::string get_result(i64 request_id) const;
    [[nodiscard]] InferenceRequestPtr get_request(i64 request_id) const;
    [[nodiscard]] ContinuousBatchScheduler::Stats get_stats() const noexcept;

private:
    [[nodiscard]] Result<void> execute_batch_step(const ScheduledBatch& batch);
    [[nodiscard]] Result<i32> sample_argmax_token(const Tensor& logits) const;

    model::Qwen3Model& model_;
    const model::Qwen3Tokenizer& tokenizer_;
    i32 chunk_size_ = 256;
    ContinuousBatchScheduler scheduler_;
};

}  // namespace pulse::scheduler
