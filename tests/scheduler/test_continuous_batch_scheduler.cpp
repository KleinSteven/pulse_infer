#include <vector>

#include <gtest/gtest.h>

#include "pulse/scheduler/continuous_batch_scheduler.hpp"

using namespace pulse;
using namespace pulse::scheduler;

TEST(InferenceRequestTest, ChunkStateMovesFromPrefillToDecode) {
    InferenceRequest request(7, "prompt", {11, 12, 13}, 2);

    EXPECT_TRUE(request.is_prefill());
    EXPECT_EQ(request.get_next_chunk_tokens(2), std::vector<i32>({11, 12}));
    EXPECT_EQ(request.get_next_chunk_positions(2), std::vector<i32>({0, 1}));

    auto first_compute_result = request.add_computed_tokens(2);
    ASSERT_TRUE(first_compute_result.is_ok()) << first_compute_result.error().message();
    EXPECT_TRUE(request.is_prefill());

    EXPECT_EQ(request.get_next_chunk_tokens(2), std::vector<i32>({13}));
    EXPECT_EQ(request.get_next_chunk_positions(2), std::vector<i32>({2}));

    auto second_compute_result = request.add_computed_tokens(1);
    ASSERT_TRUE(second_compute_result.is_ok()) << second_compute_result.error().message();
    EXPECT_FALSE(request.is_prefill());
    EXPECT_FALSE(request.is_finished());

    EXPECT_TRUE(request.add_sampled_token(21, {99}));
    EXPECT_TRUE(request.has_pending_input());
    EXPECT_EQ(request.get_next_chunk_tokens(), std::vector<i32>({21}));
    EXPECT_EQ(request.get_next_chunk_positions(), std::vector<i32>({3}));

    auto decode_compute_result = request.add_computed_tokens(1);
    ASSERT_TRUE(decode_compute_result.is_ok()) << decode_compute_result.error().message();
    EXPECT_FALSE(request.add_sampled_token(99, {99}));
    EXPECT_TRUE(request.is_finished());
}

TEST(ContinuousBatchSchedulerTest, FinishedRequestsFreeSlotsForWaitingRequests) {
    ContinuousBatchScheduler scheduler(2, 4, 8);

    const i64 request0 = scheduler.add_request("a", {1, 2}, 8);
    const i64 request1 = scheduler.add_request("b", {3, 4}, 8);
    const i64 request2 = scheduler.add_request("c", {5, 6}, 8);

    auto first_batch = scheduler.schedule_next_batch();
    ASSERT_EQ(first_batch.batch_size(), 2);
    EXPECT_EQ(first_batch.requests[0]->request_id(), request0);
    EXPECT_EQ(first_batch.requests[1]->request_id(), request1);
    EXPECT_EQ(first_batch.requests[0]->state(), RequestState::RUNNING);
    EXPECT_EQ(first_batch.requests[1]->state(), RequestState::RUNNING);

    scheduler.update_after_step({request0});

    auto second_batch = scheduler.schedule_next_batch();
    ASSERT_EQ(second_batch.batch_size(), 2);
    EXPECT_EQ(second_batch.requests[0]->request_id(), request1);
    EXPECT_EQ(second_batch.requests[1]->request_id(), request2);

    const auto stats = scheduler.get_stats();
    EXPECT_EQ(stats.num_running, 2);
    EXPECT_EQ(stats.num_waiting, 0);
    EXPECT_EQ(stats.num_finished, 1);
    EXPECT_EQ(stats.total_requests, 3);
}
