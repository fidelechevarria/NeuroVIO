#include "neurovio/vio_pipeline.hpp"
#include <iomanip>
#include <spdlog/spdlog.h>
#include "neurovio/frontend/neurotrack_frontend.hpp"
#include "neurovio/frontend/klt_tracker.hpp"

namespace neurovio {

VioPipeline::VioPipeline(Config config)
    : config_(std::move(config)),
      preintegrator_(std::make_unique<ImuPreintegrator>(config_.imu_params)) {
  if (config_.tracker_type == TrackerType::NeuroTrack) {
    tracker_ = std::make_unique<NeuroTrackFrontend>(config_.camera_calib);
    spdlog::info("VioPipeline: Initialized with NeuroTrack Neural Frontend (GPU/CUDA)");
  } else {
    tracker_ = std::make_unique<KltTracker>(config_.camera_calib);
    spdlog::info("VioPipeline: Initialized with KLT Classical Frontend");
  }
  state_ = VioState::INITIALIZING_GRAVITY;
}

VioPipeline::~VioPipeline() {
  if (config_.export_tum_trajectory && !config_.tum_output_path.empty()) {
    saveTrajectoryTum(config_.tum_output_path);
  }
}

bool VioPipeline::tryInitializeGravity() {
  if (init_imu_buffer_.size() < 50) return false;

  Vec3d sum_acc = Vec3d::Zero();
  Vec3d sum_gyro = Vec3d::Zero();
  for (const auto& imu : init_imu_buffer_) {
    sum_acc += imu.accel;
    sum_gyro += imu.gyro;
  }

  const double N = static_cast<double>(init_imu_buffer_.size());
  const Vec3d mean_acc = sum_acc / N;
  const Vec3d mean_gyro = sum_gyro / N;

  // Align body measured gravity direction to world +Z
  // In body frame, measured acceleration during rest is -g_body = a_measured
  const Vec3d gravity_body = mean_acc.normalized();
  const Vec3d gravity_world(0.0, 0.0, 1.0);

  // Compute rotation R_w_b that rotates gravity_body to gravity_world
  const Quatd q_w_b = Quatd::FromTwoVectors(gravity_body, gravity_world);

  current_state_.timestamp_ns = init_imu_buffer_.back().timestamp_ns;
  current_state_.R = q_w_b.toRotationMatrix();
  current_state_.p = Vec3d::Zero();
  current_state_.v = Vec3d::Zero();
  current_state_.bg = mean_gyro;
  current_state_.ba = Vec3d::Zero();

  preintegrator_->reset(current_state_.ba, current_state_.bg);
  state_ = VioState::TRACKING_OK;

  spdlog::info("VioPipeline: Gravity initialized successfully! Estimated Gyro Bias: [{:.4f}, {:.4f}, {:.4f}]",
               mean_gyro.x(), mean_gyro.y(), mean_gyro.z());
  return true;
}

void VioPipeline::propagateState(const ImuMeasurement& imu) {
  if (last_imu_time_ns_ == 0) {
    last_imu_time_ns_ = imu.timestamp_ns;
    return;
  }

  const double dt = static_cast<double>(imu.timestamp_ns - last_imu_time_ns_) * 1e-9;
  if (dt <= 0.0 || dt > 0.1) {
    last_imu_time_ns_ = imu.timestamp_ns;
    return;
  }

  preintegrator_->integrate(imu, dt);

  // Continuous strapdown propagation for high-rate state estimate
  const Vec3d w_unbiased = imu.gyro - current_state_.bg;
  const Vec3d a_unbiased = imu.accel - current_state_.ba;

  const Mat3d dR = LieAlgebra::expSO3(w_unbiased * dt);
  const Vec3d a_world = current_state_.R * a_unbiased + config_.imu_params.gravity;

  current_state_.p += current_state_.v * dt + 0.5 * a_world * dt * dt;
  current_state_.v += a_world * dt;
  current_state_.R = current_state_.R * dR;
  current_state_.timestamp_ns = imu.timestamp_ns;

  last_imu_time_ns_ = imu.timestamp_ns;
}

void VioPipeline::processImu(const ImuMeasurement& imu) {
  if (state_ == VioState::INITIALIZING_GRAVITY) {
    init_imu_buffer_.push_back(imu);
    const double elapsed_sec =
        static_cast<double>(imu.timestamp_ns - init_imu_buffer_.front().timestamp_ns) * 1e-9;
    if (elapsed_sec >= config_.init_duration_sec) {
      tryInitializeGravity();
    }
    return;
  }

  if (state_ == VioState::TRACKING_OK) {
    propagateState(imu);
  }
}

void VioPipeline::processFrame(const StereoFrame& frame) {
  if (state_ == VioState::INITIALIZING_GRAVITY) {
    return;
  }

  // 1. Visual Feature Tracking with NeuroTrack Neural Frontend
  const auto tracks = tracker_->trackFrame(frame.left);

  // 2. Register current state to trajectory
  estimated_trajectory_.push_back(current_state_);

  // Reset preintegrator between keyframes
  preintegrator_->reset(current_state_.ba, current_state_.bg);
  last_frame_time_ns_ = frame.timestamp_ns;
}

void VioPipeline::saveTrajectoryTum(const std::string& filepath) const {
  std::ofstream out(filepath);
  if (!out.is_open()) {
    spdlog::error("VioPipeline: Failed to open output trajectory file at {}", filepath);
    return;
  }

  out << std::fixed << std::setprecision(6);
  for (const auto& state : estimated_trajectory_) {
    const double timestamp_sec = static_cast<double>(state.timestamp_ns) * 1e-9;
    const Quatd q = state.quaternion();

    out << timestamp_sec << " "
        << state.p.x() << " " << state.p.y() << " " << state.p.z() << " "
        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
  }

  spdlog::info("VioPipeline: Successfully saved {} poses to {}",
               estimated_trajectory_.size(), filepath);
}

}  // namespace neurovio
