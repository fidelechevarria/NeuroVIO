#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "neurovio/sensors/camera_data.hpp"
#include "neurovio/sensors/imu_data.hpp"

namespace neurovio {

class EuRoCLoader {
 public:
  struct Config {
    std::filesystem::path dataset_path;
    bool load_stereo{true};
    bool load_ground_truth{true};
  };

  explicit EuRoCLoader(Config config);
  ~EuRoCLoader() = default;

  bool load();

  [[nodiscard]] const std::vector<ImuMeasurement>& getImuData() const noexcept {
    return imu_measurements_;
  }

  [[nodiscard]] const std::vector<CameraFrame>& getLeftFrames() const noexcept {
    return left_frames_;
  }

  [[nodiscard]] const std::vector<CameraFrame>& getRightFrames() const noexcept {
    return right_frames_;
  }

  [[nodiscard]] const std::vector<NavState>& getGroundTruth() const noexcept {
    return ground_truth_states_;
  }

  [[nodiscard]] const CameraCalibration& getLeftCameraCalib() const noexcept {
    return left_camera_calib_;
  }

  [[nodiscard]] const CameraCalibration& getRightCameraCalib() const noexcept {
    return right_camera_calib_;
  }

  [[nodiscard]] const ImuParameters& getImuParams() const noexcept {
    return imu_params_;
  }

  // Iterate over synchronized events (e.g. streaming playback)
  using ImuCallback = std::function<void(const ImuMeasurement&)>;
  using FrameCallback = std::function<void(const StereoFrame&)>;

  void play(const ImuCallback& on_imu, const FrameCallback& on_frame) const;

 private:
  bool parseImuCsv(const std::filesystem::path& csv_path);
  bool parseCameraCsv(const std::filesystem::path& csv_path,
                      const std::filesystem::path& img_dir,
                      uint32_t cam_id,
                      std::vector<CameraFrame>& out_frames);
  bool parseGroundTruthCsv(const std::filesystem::path& csv_path);

  Config config_;
  std::vector<ImuMeasurement> imu_measurements_;
  std::vector<CameraFrame> left_frames_;
  std::vector<CameraFrame> right_frames_;
  std::vector<NavState> ground_truth_states_;

  CameraCalibration left_camera_calib_{};
  CameraCalibration right_camera_calib_{};
  ImuParameters imu_params_{};
};

}  // namespace neurovio
