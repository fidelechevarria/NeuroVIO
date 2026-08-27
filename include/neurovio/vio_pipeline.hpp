#pragma once

#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "neurovio/backend/imu_preintegrator.hpp"
#include "neurovio/common/ring_buffer.hpp"
#include "neurovio/common/types.hpp"
#include "neurovio/frontend/feature_tracker.hpp"
#include "neurovio/sensors/camera_data.hpp"
#include "neurovio/sensors/imu_data.hpp"

namespace neurovio {

enum class VioState {
  UNINITIALIZED,
  INITIALIZING_GRAVITY,
  TRACKING_OK,
  LOST
};

enum class TrackerType {
  NeuroTrack,
  KLT
};

class VioPipeline {
 public:
  struct Config {
    ImuParameters imu_params{};
    CameraCalibration camera_calib{};
    TrackerType tracker_type{TrackerType::NeuroTrack};
    double init_duration_sec{0.5};     // Stationary duration to compute initial gravity alignment
    bool export_tum_trajectory{true};
    std::string tum_output_path{"trajectory_estimate.tum"};
  };

  explicit VioPipeline(Config config);
  ~VioPipeline();

  void processImu(const ImuMeasurement& imu);
  void processFrame(const StereoFrame& frame);

  [[nodiscard]] VioState getState() const noexcept { return state_; }
  [[nodiscard]] const NavState& getCurrentNavState() const noexcept { return current_state_; }
  [[nodiscard]] const std::vector<NavState>& getEstimatedTrajectory() const noexcept {
    return estimated_trajectory_;
  }

  void saveTrajectoryTum(const std::string& filepath) const;

 private:
  bool tryInitializeGravity();
  void propagateState(const ImuMeasurement& imu);

  Config config_;
  VioState state_{VioState::UNINITIALIZED};
  NavState current_state_{};
  std::vector<NavState> estimated_trajectory_;

  std::unique_ptr<ImuPreintegrator> preintegrator_;
  std::unique_ptr<FeatureTracker> tracker_;

  std::vector<ImuMeasurement> init_imu_buffer_;
  TimestampNs last_imu_time_ns_{0};
  TimestampNs last_frame_time_ns_{0};
};

}  // namespace neurovio
