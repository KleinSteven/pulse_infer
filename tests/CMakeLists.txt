# ============================================================================
# PulseInfer Unit Tests
# ============================================================================

# Test executable
add_executable(pulse_tests
  core/test_types.cpp
  core/test_error.cpp
  core/test_allocator.cpp
  core/test_buffer.cpp
  core/test_tensor.cpp
  core/test_safetensor.cpp
  core/test_var_builder.cpp
  ops/test_add.cpp
  test_main.cpp
)

# Link against pulse_core and GTest
target_link_libraries(pulse_tests
  PRIVATE
    pulse_core
    GTest::gtest
    GTest::gtest_main
)

# Target-scoped compile and link options
target_compile_options(pulse_tests
  PRIVATE
    ${PULSE_COMPILE_OPTIONS}
)

target_link_options(pulse_tests
  PRIVATE
    ${PULSE_LINK_OPTIONS}
)

if(PULSE_ENABLE_ASAN)
  # Apply ASan runtime options to both gtest discovery and test execution.
  set_property(TARGET pulse_tests PROPERTY
    TEST_LAUNCHER "${CMAKE_COMMAND};-E;env;ASAN_OPTIONS=detect_leaks=0"
  )
endif()

# Register tests
gtest_discover_tests(
  pulse_tests
  DISCOVERY_MODE PRE_TEST
)
