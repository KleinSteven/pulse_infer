# ============================================================================
# PulseInfer Unit Tests
# ============================================================================

# Test executable
add_executable(load_safetensor_demo
  load_safetensor.cpp
)

add_executable(qwen3_tokenizer_demo
  qwen3_tokenizer_demo.cpp
)

add_executable(qwen3_config_demo
  qwen3_config_demo.cpp
)

# Link against pulse_core and GTest
target_link_libraries(load_safetensor_demo
  PRIVATE
    pulse_core
)

target_link_libraries(qwen3_tokenizer_demo
  PRIVATE
    pulse_core
)

target_link_libraries(qwen3_config_demo
  PRIVATE
    pulse_core
)

# Target-scoped compile and link options
target_compile_options(load_safetensor_demo
  PRIVATE
    ${PULSE_COMPILE_OPTIONS}
)

target_compile_options(qwen3_tokenizer_demo
  PRIVATE
    ${PULSE_COMPILE_OPTIONS}
)

target_compile_options(qwen3_config_demo
  PRIVATE
    ${PULSE_COMPILE_OPTIONS}
)

target_link_options(load_safetensor_demo
  PRIVATE
    ${PULSE_LINK_OPTIONS}
)

target_link_options(qwen3_tokenizer_demo
  PRIVATE
    ${PULSE_LINK_OPTIONS}
)

target_link_options(qwen3_config_demo
  PRIVATE
    ${PULSE_LINK_OPTIONS}
)

if(PULSE_BUILD_CUDA)
  add_executable(qwen3_infer
    qwen3_infer.cpp
  )

  add_executable(qwen3_continuous_batch_demo
    qwen3_continuous_batch_demo.cpp
  )

  target_link_libraries(qwen3_infer
    PRIVATE
      pulse_core
  )

  target_link_libraries(qwen3_continuous_batch_demo
    PRIVATE
      pulse_core
  )

  target_compile_options(qwen3_infer
    PRIVATE
      ${PULSE_COMPILE_OPTIONS}
  )

  target_compile_options(qwen3_continuous_batch_demo
    PRIVATE
      ${PULSE_COMPILE_OPTIONS}
  )

  target_link_options(qwen3_infer
    PRIVATE
      ${PULSE_LINK_OPTIONS}
  )

  target_link_options(qwen3_continuous_batch_demo
    PRIVATE
      ${PULSE_LINK_OPTIONS}
  )
else()
  message(STATUS "Skipping qwen3_infer, qwen3_continuous_batch_demo and qwen3_paged_attention_demo because PULSE_BUILD_CUDA=OFF")
endif()
