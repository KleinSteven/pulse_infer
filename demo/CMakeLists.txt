# ============================================================================
# PulseInfer Unit Tests
# ============================================================================

# Test executable
add_executable(load_safetemsor_demo
  load_safetemsor.cpp
)

# Link against pulse_core and GTest
target_link_libraries(load_safetemsor_demo
  PRIVATE
    pulse_core
)

# Target-scoped compile and link options
target_compile_options(load_safetemsor_demo
  PRIVATE
    ${PULSE_COMPILE_OPTIONS}
)

target_link_options(load_safetemsor_demo
  PRIVATE
    ${PULSE_LINK_OPTIONS}
)
