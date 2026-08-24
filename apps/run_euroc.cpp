#include <chrono>
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>
#include "neurovio/sensors/euroc_loader.hpp"
#include "neurovio/vio_pipeline.hpp"

int main(int argc, char** argv) {
  spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
  spdlog::info("=================================================");
  spdlog::info("  NeuroVIO: Spatial Visual-Inertial Odometry CLI ");
  spdlog::info("=================================================");

  std::string dataset_path = "/media/disk1/euroc_mav_dataset/machine_hall/MH_01_easy";
  std::string output_path = "trajectory_estimate.tum";
  size_t max_frames = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--dataset-path" && i + 1 < argc) {
      dataset_path = argv[++i];
    } else if (arg == "--output-trajectory" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--max-frames" && i + 1 < argc) {
      max_frames = std::stoul(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "  --dataset-path <path>       Path to EuRoC dataset directory\n"
                << "  --output-trajectory <path>  Path to save TUM format trajectory\n"
                << "  --max-frames <num>          Max frames to process (0 = all)\n";
      return 0;
    }
  }

  spdlog::info("Dataset Path: {}", dataset_path);
  spdlog::info("Output Trajectory: {}", output_path);

  // 1. Load EuRoC dataset
  neurovio::EuRoCLoader::Config loader_config;
  loader_config.dataset_path = dataset_path;
  loader_config.load_stereo = false;
  loader_config.load_ground_truth = true;

  neurovio::EuRoCLoader loader(loader_config);
  if (!loader.load()) {
    spdlog::error("Failed to load dataset from {}", dataset_path);
    return 1;
  }

  // 2. Initialize VIO Pipeline
  neurovio::VioPipeline::Config vio_config;
  vio_config.imu_params = loader.getImuParams();
  vio_config.camera_calib = loader.getLeftCameraCalib();
  vio_config.export_tum_trajectory = true;
  vio_config.tum_output_path = output_path;

  neurovio::VioPipeline pipeline(vio_config);

  // 3. Play dataset synchronously
  size_t imu_count = 0;
  size_t frame_count = 0;
  const auto start_wall_time = std::chrono::high_resolution_clock::now();

  loader.play(
      [&](const neurovio::ImuMeasurement& imu) {
        pipeline.processImu(imu);
        ++imu_count;
      },
      [&](const neurovio::StereoFrame& frame) {
        if (max_frames > 0 && frame_count >= max_frames) return;
        pipeline.processFrame(frame);
        ++frame_count;

        if (frame_count % 100 == 0) {
          const auto& state = pipeline.getCurrentNavState();
          spdlog::info("Processed {} frames | Current Pos: [{:.3f}, {:.3f}, {:.3f}] m",
                       frame_count, state.p.x(), state.p.y(), state.p.z());
        }
      });

  const auto end_wall_time = std::chrono::high_resolution_clock::now();
  const double duration_sec =
      std::chrono::duration<double>(end_wall_time - start_wall_time).count();

  pipeline.saveTrajectoryTum(output_path);

  spdlog::info("Finished processing {} frames and {} IMU measurements in {:.2f}s ({:.1f} FPS)",
               frame_count, imu_count, duration_sec,
               (duration_sec > 0 ? static_cast<double>(frame_count) / duration_sec : 0.0));
  return 0;
}
