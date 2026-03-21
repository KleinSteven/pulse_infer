#include "pulse/scheduler/continuous_batch_engine.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <utility>

#include "pulse/logging.hpp"
#include "pulse/runtime/kv_cache_manager.hpp"

#ifdef PULSE_USE_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace pulse::scheduler {

namespace {

template<typename T>
i32 argmax_token(const Tensor& logits) {
    const auto* logits_ptr = logits.ptr<T>();
    if (logits_ptr == nullptr || logits.size() == 0) {
        return -1;
    }

    i32 best_index = 0;
    f32 best_value = 0.0f;
    if constexpr (std::is_same_v<T, f16>) {
        best_value = __half2float(logits_ptr[0]);
    } else if constexpr (std::is_same_v<T, bf16>) {
        best_value = __bfloat162float(logits_ptr[0]);
    } else {
        best_value = static_cast<f32>(logits_ptr[0]);
    }

    for (usize index = 1; index < logits.size(); ++index) {
        f32 value = 0.0f;
        if constexpr (std::is_same_v<T, f16>) {
            value = __half2float(logits_ptr[index]);
        } else if constexpr (std::is_same_v<T, bf16>) {
            value = __bfloat162float(logits_ptr[index]);
        } else {
            value = static_cast<f32>(logits_ptr[index]);
        }

        if (value > best_value) {
            best_value = value;
            best_index = static_cast<i32>(index);
        }
    }

    return best_index;
}

}  // namespace

ContinuousBatchEngine::ContinuousBatchEngine(model::Qwen3Model& model,
                                             const model::Qwen3Tokenizer& tokenizer,
                                             i32 max_batch_size,
                                             i32 chunk_size)
    : model_(model),
      tokenizer_(tokenizer),
      chunk_size_(chunk_size),
      scheduler_(max_batch_size, 32, chunk_size) {}

i64 ContinuousBatchEngine::add_request(const std::string& prompt, i32 max_new_tokens) {
    const auto prompt_tokens = tokenizer_.encode(prompt);
    return scheduler_.add_request(prompt, prompt_tokens, max_new_tokens);
}

Result<void> ContinuousBatchEngine::run_until_complete(bool print_progress) {
    i32 step = 0;

    while (scheduler_.has_work()) {
        auto batch = scheduler_.schedule_next_batch();
        if (batch.empty()) {
            break;
        }

        if (print_progress && step % 10 == 0) {
            const auto stats = scheduler_.get_stats();
            pulse::info("Continuous batch step {} running={} waiting={} finished={}",
                        step,
                        stats.num_running,
                        stats.num_waiting,
                        stats.num_finished);
        }

        auto step_result = execute_batch_step(batch);
        if (!step_result) {
            return step_result;
        }

        ++step;
    }

    if (print_progress) {
        const auto stats = scheduler_.get_stats();
        pulse::info("Continuous batch complete steps={} total_requests={}", step, stats.total_requests);
    }

    return Ok();
}

std::string ContinuousBatchEngine::get_result(i64 request_id) const {
    const auto request = scheduler_.get_request(request_id);
    if (request == nullptr || !request->is_finished()) {
        return {};
    }

    std::string result;
    for (i32 token : request->generated_tokens()) {
        if (std::find(model_.config().eos_token_ids.begin(), model_.config().eos_token_ids.end(), token) !=
            model_.config().eos_token_ids.end()) {
            break;
        }
        result += tokenizer_.decode_token(token);
    }

    return result;
}

InferenceRequestPtr ContinuousBatchEngine::get_request(i64 request_id) const {
    return scheduler_.get_request(request_id);
}

ContinuousBatchScheduler::Stats ContinuousBatchEngine::get_stats() const noexcept {
    return scheduler_.get_stats();
}

Result<void> ContinuousBatchEngine::execute_batch_step(const ScheduledBatch& batch) {
    std::vector<i32> all_tokens;
    std::vector<i32> all_positions;
    std::vector<i32> seq_ids;
    std::vector<i32> seq_lens;

    all_tokens.reserve(batch.requests.size() * static_cast<usize>(chunk_size_));
    all_positions.reserve(all_tokens.capacity());
    seq_ids.reserve(all_tokens.capacity());
    seq_lens.reserve(batch.requests.size());

    for (const auto& request : batch.requests) {
        auto chunk_tokens = request->get_next_chunk_tokens(chunk_size_);
        auto chunk_positions = request->get_next_chunk_positions(chunk_size_);
        if (chunk_tokens.empty() || chunk_tokens.size() != chunk_positions.size()) {
            return Err<void>(ErrorCode::InvalidOperator, "Continuous batch request has no executable chunk");
        }

        seq_lens.push_back(static_cast<i32>(chunk_tokens.size()));
        all_tokens.insert(all_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
        all_positions.insert(all_positions.end(), chunk_positions.begin(), chunk_positions.end());
        for (usize index = 0; index < chunk_tokens.size(); ++index) {
            seq_ids.push_back(static_cast<i32>(request->request_id()));
        }
    }

    auto logits_result = Tensor::create({static_cast<i32>(all_tokens.size()), model_.config().vocab_size},
                                        model_.dtype(),
                                        model_.device());
    if (!logits_result) {
        return Err<void>(std::move(logits_result.error()));
    }

    auto forward_result = model_.forward_batched(all_tokens, all_positions, seq_ids, logits_result.value());
    if (!forward_result) {
        return forward_result;
    }

    std::vector<i64> finished_ids;
    finished_ids.reserve(batch.requests.size());

    i32 token_offset = 0;

    for (usize request_index = 0; request_index < batch.requests.size(); ++request_index) {
        const auto& request = batch.requests[request_index];

        auto computed_result = request->add_computed_tokens(seq_lens[request_index]);

        if (!computed_result) {
            return computed_result;
        }

        if (request->is_prefill()) {
            token_offset += seq_lens[request_index];
            continue;
        }

        if (!request->can_generate_more()) {
            request->finish();
            finished_ids.push_back(request->request_id());
            if (auto* cache_manager = model_.paged_cache_manager(); cache_manager != nullptr) {
                auto free_result = cache_manager->free_sequence(static_cast<i32>(request->request_id()));
                (void)free_result;
            }
            token_offset += seq_lens[request_index];
            continue;
        }

        auto row_result = Tensor::create({1, model_.config().vocab_size},
                                         logits_result.value().dtype(),
                                         logits_result.value().device());
        if (!row_result) {
            return Err<void>(std::move(row_result.error()));
        }

        Tensor row_logits(std::move(row_result.value()));
        const auto* src = static_cast<const std::byte*>(logits_result.value().data()) +
                          static_cast<usize>(token_offset + seq_lens[request_index] - 1) *
                              static_cast<usize>(model_.config().vocab_size) *
                              data_type_size(logits_result.value().dtype());
        if (logits_result.value().device() == DeviceType::CPU) {
            std::memcpy(row_logits.data(), src, row_logits.byte_size());
        } else {
#ifdef PULSE_USE_CUDA
            auto err = cudaMemcpy(row_logits.data(),
                                  src,
                                  row_logits.byte_size(),
                                  cudaMemcpyKind::cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                return Err<void>(ErrorCode::CudaError,
                                 std::format("cudaMemcpy failed: {}", cudaGetErrorString(err)));
            }
#else
            return Err<void>(ErrorCode::DeviceMismatch, "CUDA logits copy requires CUDA support");
#endif
        }

        auto sample_result = sample_argmax_token(row_logits);
        if (!sample_result) {
            return Err<void>(std::move(sample_result.error()));
        }

        const bool should_continue =
            request->add_sampled_token(sample_result.value(), model_.config().eos_token_ids);
        if (!should_continue) {
            finished_ids.push_back(request->request_id());
            if (auto* cache_manager = model_.paged_cache_manager(); cache_manager != nullptr) {
                auto free_result = cache_manager->free_sequence(static_cast<i32>(request->request_id()));
                (void)free_result;
            }
        }

        token_offset += seq_lens[request_index];
    }

    scheduler_.update_after_step(finished_ids);
    return Ok();
}

Result<i32> ContinuousBatchEngine::sample_argmax_token(const Tensor& logits) const {
    Tensor cpu_logits;
    if (logits.device() == DeviceType::CPU) {
        auto clone_result = logits.clone();
        if (!clone_result) {
            return Err<i32>(std::move(clone_result.error()));
        }
        cpu_logits = std::move(clone_result.value());
    } else {
        auto to_cpu_result = logits.to(DeviceType::CPU);
        if (!to_cpu_result) {
            return Err<i32>(std::move(to_cpu_result.error()));
        }
        cpu_logits = std::move(to_cpu_result.value());
    }

    if (cpu_logits.size() == 0) {
        return Err<i32>(ErrorCode::InvalidArgument, "Logits tensor is empty");
    }

    i32 token = -1;
    switch (cpu_logits.dtype()) {
        case DataType::Float32:
            token = argmax_token<f32>(cpu_logits);
            break;
        case DataType::Float64:
            token = argmax_token<f64>(cpu_logits);
            break;
#ifdef PULSE_USE_CUDA
        case DataType::Float16:
            token = argmax_token<f16>(cpu_logits);
            break;
        case DataType::BFloat16:
            token = argmax_token<bf16>(cpu_logits);
            break;
#endif
        default:
            return Err<i32>(ErrorCode::NotImplemented,
                            "Continuous batch argmax does not support logits dtype");
    }

    if (token < 0 || static_cast<usize>(token) >= cpu_logits.size()) {
        return Err<i32>(ErrorCode::InvalidOperator, "Continuous batch argmax produced invalid token");
    }

    return Ok(token);
}

}  // namespace pulse::scheduler
