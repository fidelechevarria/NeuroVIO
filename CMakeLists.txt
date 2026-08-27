cmake_minimum_required(VERSION 3.22)
project(NeuroVIO VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(NEUROVIO_BUILD_TESTS "Build unit tests" ON)
option(NEUROVIO_BUILD_APPS "Build CLI applications" ON)
option(NEUROVIO_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(NEUROVIO_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(NEUROVIO_USE_RERUN "Enable Rerun.io C++ telemetry SDK" OFF)

# -----------------------------------------------------------------------------
# Compiler Flags & Sanitizers
# -----------------------------------------------------------------------------
add_library(neurovio_compile_options INTERFACE)
target_compile_features(neurovio_compile_options INTERFACE cxx_std_20)

target_compile_options(neurovio_compile_options INTERFACE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
    -Wunused -Woverloaded-virtual -Wconversion -Wsign-conversion -Wnull-dereference
  >
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Use LLD linker if available
  include(CheckLinkerFlag OPTIONAL)
  if(COMMAND check_linker_flag)
    check_linker_flag(CXX "-fuse-ld=lld" LINKER_SUPPORTS_LLD)
    if(LINKER_SUPPORTS_LLD)
      target_link_options(neurovio_compile_options INTERFACE "-fuse-ld=lld")
    endif()
  endif()
endif()

if(NEUROVIO_ENABLE_ASAN)
  message(STATUS "NeuroVIO: AddressSanitizer & UndefinedBehaviorSanitizer enabled")
  target_compile_options(neurovio_compile_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(neurovio_compile_options INTERFACE -fsanitize=address,undefined)
elseif(NEUROVIO_ENABLE_TSAN)
  message(STATUS "NeuroVIO: ThreadSanitizer enabled")
  target_compile_options(neurovio_compile_options INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(neurovio_compile_options INTERFACE -fsanitize=thread)
endif()

# -----------------------------------------------------------------------------
# External Dependencies (FetchContent & find_package)
# -----------------------------------------------------------------------------
include(FetchContent)

# 1. Eigen3
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  message(STATUS "Eigen3 not found in system, fetching via FetchContent...")
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(eigen)
endif()

# 2. fmt
find_package(fmt 9.0 QUIET)
if(NOT TARGET fmt::fmt AND NOT TARGET fmt::fmt-header-only)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.2.1
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(fmt)
endif()

# 3. spdlog
find_package(spdlog 1.10 QUIET)
if(NOT TARGET spdlog::spdlog AND NOT TARGET spdlog::spdlog_header_only)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(spdlog)
endif()

# 4. Sophus (Lie Groups SO3, SE3)
find_package(Sophus QUIET)
if(NOT TARGET Sophus::Sophus)
  message(STATUS "Sophus not found in system, fetching header-only Sophus...")
  FetchContent_Declare(
    sophus
    GIT_REPOSITORY https://github.com/strasdat/Sophus.git
    GIT_TAG main
    GIT_SHALLOW TRUE
  )
  set(BUILD_SOPHUS_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_SOPHUS_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(sophus)
endif()

# 5. OpenCV
find_package(OpenCV QUIET COMPONENTS core imgproc highgui calib3d features2d video)
if(OpenCV_FOUND)
  message(STATUS "Found OpenCV ${OpenCV_VERSION} at ${OpenCV_INCLUDE_DIRS}")
endif()

# 6. ONNX Runtime C++
option(NEUROVIO_USE_ONNX "Enable ONNX Runtime neural feature extraction" ON)
if(NEUROVIO_USE_ONNX)
  message(STATUS "NeuroVIO: Enabling ONNX Runtime C++ neural frontend...")
  FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
  )
  FetchContent_MakeAvailable(onnxruntime)

  if(NOT TARGET OnnxRuntime::OnnxRuntime)
    add_library(OnnxRuntime::OnnxRuntime SHARED IMPORTED GLOBAL)
    set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
      IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
    )
  endif()
endif()

# 7. GTSAM (Optional for backend optimization)
find_package(GTSAM QUIET)

# -----------------------------------------------------------------------------
# NeuroTrack Frontend Subproject Integration
# -----------------------------------------------------------------------------
set(NEUROTRACK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../NeuroTrack")
if(EXISTS "${NEUROTRACK_DIR}/CMakeLists.txt")
  message(STATUS "NeuroVIO: Integrating NeuroTrack from ${NEUROTRACK_DIR}")
  set(NEUROTRACK_BUILD_APPS OFF CACHE BOOL "" FORCE)
  set(NEUROTRACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(NEUROTRACK_ENABLE_CUDA ON CACHE BOOL "" FORCE)
  add_subdirectory("${NEUROTRACK_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/neurotrack_build" EXCLUDE_FROM_ALL)
endif()

# -----------------------------------------------------------------------------
# NeuroVIO Library Targets
# -----------------------------------------------------------------------------
set(NEUROVIO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 1. neurovio_common (Types, Lie Algebra, Timestamps, RingBuffer)
add_library(neurovio_common INTERFACE)
target_include_directories(neurovio_common INTERFACE
  $<BUILD_INTERFACE:${NEUROVIO_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(neurovio_common SYSTEM INTERFACE
  $<TARGET_PROPERTY:Eigen3::Eigen,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:Sophus::Sophus,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:spdlog::spdlog,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(neurovio_common INTERFACE
  neurovio_compile_options
  Eigen3::Eigen
  Sophus::Sophus
  spdlog::spdlog
)

# 2. neurovio_sensors (Sensor structs, EuRoC dataset parser)
add_library(neurovio_sensors
  src/sensors/euroc_loader.cpp
)
target_include_directories(neurovio_sensors PUBLIC
  $<BUILD_INTERFACE:${NEUROVIO_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(neurovio_sensors PUBLIC
  neurovio_common
)
if(OpenCV_FOUND)
  target_link_libraries(neurovio_sensors PUBLIC ${OpenCV_LIBS})
  target_compile_definitions(neurovio_sensors PUBLIC NEUROVIO_HAS_OPENCV=1)
endif()

# 3. neurovio_frontend (NeuroTrack Adapter, KLT Tracker, ONNX SuperPoint, ONNX LightGlue)
add_library(neurovio_frontend
  src/frontend/neurotrack_frontend.cpp
  src/frontend/klt_tracker.cpp
  src/frontend/onnx_superpoint.cpp
  src/frontend/onnx_lightglue.cpp
)
target_include_directories(neurovio_frontend PUBLIC
  $<BUILD_INTERFACE:${NEUROVIO_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(neurovio_frontend PUBLIC
  neurovio_common
  neurovio_sensors
  neurotrack
)
if(OpenCV_FOUND)
  target_link_libraries(neurovio_frontend PUBLIC ${OpenCV_LIBS})
endif()
if(NEUROVIO_USE_ONNX)
  target_include_directories(neurovio_frontend SYSTEM PUBLIC
    $<TARGET_PROPERTY:OnnxRuntime::OnnxRuntime,INTERFACE_INCLUDE_DIRECTORIES>
  )
  target_link_libraries(neurovio_frontend PUBLIC OnnxRuntime::OnnxRuntime)
  target_compile_definitions(neurovio_frontend PUBLIC NEUROVIO_HAS_ONNX=1)
endif()

# 4. neurovio_backend (IMU Preintegration, Factor Graph & iSAM2)
add_library(neurovio_backend
  src/backend/imu_preintegrator.cpp
)
target_include_directories(neurovio_backend PUBLIC
  $<BUILD_INTERFACE:${NEUROVIO_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(neurovio_backend PUBLIC
  neurovio_common
  neurovio_sensors
)
if(GTSAM_FOUND OR TARGET gtsam)
  target_link_libraries(neurovio_backend PUBLIC gtsam)
  target_compile_definitions(neurovio_backend PUBLIC NEUROVIO_HAS_GTSAM=1)
endif()

# Top-level unified library
add_library(neurovio
  src/vio_pipeline.cpp
)
target_include_directories(neurovio PUBLIC
  $<BUILD_INTERFACE:${NEUROVIO_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(neurovio PUBLIC
  neurovio_common
  neurovio_sensors
  neurovio_frontend
  neurovio_backend
)

# -----------------------------------------------------------------------------
# Unit Testing (GoogleTest)
# -----------------------------------------------------------------------------
if(NEUROVIO_BUILD_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  if(NOT TARGET GTest::gtest_main)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# CLI Applications
# -----------------------------------------------------------------------------
if(NEUROVIO_BUILD_APPS)
  add_subdirectory(apps)
endif()
