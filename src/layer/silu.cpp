#include "pulse/layer/silu.hpp"

#include "pulse/ops/silu.hpp"

namespace pulse::layer {

Result<Tensor> SiLU::forward(const Tensor& input) const {
    auto output_result = Tensor::create(input.dims(), input.dtype(), input.device());
    if (!output_result) {
        return Err<Tensor>(std::move(output_result.error()));
    }

    Tensor output(std::move(output_result.value()));
    auto forward_result = forward(input, output);
    if (!forward_result) {
        return Err<Tensor>(std::move(forward_result.error()));
    }

    return Ok(std::move(output));
}

Result<void> SiLU::forward(const Tensor& input, Tensor& output) const {
    auto silu_result = ops::silu(input, output);
    if (!silu_result) {
        return Err<void>(std::move(silu_result.error()));
    }

    return Ok();
}

}  // namespace pulse::layer
