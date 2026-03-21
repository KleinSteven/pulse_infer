# ============================================================================
# PulseInfer Core Library
# ============================================================================

file(GLOB_RECURSE PULSE_CORE_SOURCES
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layer/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/*.cpp
)

if(PULSE_BUILD_CUDA)
  file(GLOB_RECURSE PULSE_CUDA_SOURCES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/kernels/cuda/*.cu
  )
endif()

if(PULSE_BUILD_SHARED)
  add_library(pulse_core SHARED)
else()
  add_library(pulse_core STATIC)
endif()

target_sources(pulse_core
  PRIVATE
    ${PULSE_CORE_SOURCES}
    ${PULSE_CUDA_SOURCES}
)

target_compile_options(pulse_core
  PRIVATE
    ${PULSE_COMPILE_OPTIONS}
)

if(PULSE_BUILD_SHARED)
  target_link_options(pulse_core
    PRIVATE
      ${PULSE_LINK_OPTIONS}
  )
endif()

# Link dependencies
target_link_libraries(pulse_core
  PUBLIC
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

if(PULSE_BUILD_CUDA)
  target_link_libraries(pulse_core
    PUBLIC
      CUDA::cudart
      CUDA::cublas
  )
endif()

# Set target properties
set_target_properties(pulse_core PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Include directories
target_include_directories(pulse_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compile definitions
if(PULSE_BUILD_CUDA)
  target_compile_definitions(pulse_core PUBLIC PULSE_USE_CUDA)
endif()

# Install library
install(TARGETS pulse_core
  EXPORT PulseInferTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
