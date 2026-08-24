add_executable(test_lie_algebra test_lie_algebra.cpp)
target_link_libraries(test_lie_algebra PRIVATE
  neurovio_common
  GTest::gtest_main
)
gtest_discover_tests(test_lie_algebra)

add_executable(test_ring_buffer test_ring_buffer.cpp)
target_link_libraries(test_ring_buffer PRIVATE
  neurovio_common
  GTest::gtest_main
)
gtest_discover_tests(test_ring_buffer)

add_executable(test_imu_preintegrator test_imu_preintegrator.cpp)
target_link_libraries(test_imu_preintegrator PRIVATE
  neurovio_backend
  GTest::gtest_main
)
gtest_discover_tests(test_imu_preintegrator)

add_executable(test_euroc_loader test_euroc_loader.cpp)
target_link_libraries(test_euroc_loader PRIVATE
  neurovio_sensors
  GTest::gtest_main
)
gtest_discover_tests(test_euroc_loader)

if(NEUROVIO_USE_ONNX)
  add_executable(test_onnx_features test_onnx_features.cpp)
  target_link_libraries(test_onnx_features PRIVATE
    neurovio_frontend
    GTest::gtest_main
  )
  gtest_discover_tests(test_onnx_features)
endif()
