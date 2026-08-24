add_executable(run_euroc
  run_euroc.cpp
)

target_link_libraries(run_euroc PRIVATE
  neurovio
  neurovio_compile_options
  spdlog::spdlog
)
