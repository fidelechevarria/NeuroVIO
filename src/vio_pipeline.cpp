#include "neurovio/vio_pipeline.hpp"
#include <iomanip>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <Eigen/Dense>
#include "neurovio/frontend/neurotrack_frontend.hpp"
#include "neurovio/frontend/klt_tracker.hpp"

namespace neurovio {

struct VioPipelineImpl {
  std::unordered_map<uint64_t, Vec2d> prev_features;
  Vec3d prev_position{Vec3d::Zero()};
  Mat3d prev_rotation{Mat3d::Identity()};
};

static std::unique_ptr<VioPipelineImpl> g_impl;

VioPipeline::VioPipeline(Config config)
    : config_(std::move(config)),
      preintegrator_(std::make_unique<ImuPreintegrator>(config_.imu_params)) {
  g_impl = std::make_unique<VioPipelineImpl>();

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

  // In resting body frame, measured specific force points opposite to world gravity
  const Vec3d z_body = mean_acc.normalized();
  const Vec3d z_world(0.0, 0.0, 1.0);
  const Quatd q_w_b = Quatd::FromTwoVectors(z_body, z_world);

  current_state_.timestamp_ns = init_imu_buffer_.back().timestamp_ns;
  current_state_.R = q_w_b.toRotationMatrix();
  current_state_.p = Vec3d::Zero();
  current_state_.v = Vec3d::Zero();
  current_state_.bg = mean_gyro;
  current_state_.ba = mean_acc + (current_state_.R.transpose() * config_.imu_params.gravity);

  preintegrator_->reset(current_state_.ba, current_state_.bg);
  state_ = VioState::TRACKING_OK;

  g_impl->prev_position = current_state_.p;
  g_impl->prev_rotation = current_state_.R;

  spdlog::info("VioPipeline: Gravity aligned! Estimated Gyro Bias: [{:.4f}, {:.4f}, {:.4f}] | Accel Bias: [{:.4f}, {:.4f}, {:.4f}]",
               mean_gyro.x(), mean_gyro.y(), mean_gyro.z(),
               current_state_.ba.x(), current_state_.ba.y(), current_state_.ba.z());
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

  const Vec3d w_unbiased = imu.gyro - current_state_.bg;
  const Vec3d a_unbiased = imu.accel - current_state_.ba;

  const Mat3d dR = LieAlgebra::expSO3(w_unbiased * dt);
  const Vec3d a_world = current_state_.R * a_unbiased + config_.imu_params.gravity;

  current_state_.R = current_state_.R * dR;
  current_state_.v += a_world * dt;
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

  const double dt_frame = (last_frame_time_ns_ > 0)
      ? static_cast<double>(frame.timestamp_ns - last_frame_time_ns_) * 1e-9
      : 0.033;

  // 1. Visual Feature Tracking with NeuroTrack Neural Frontend
  const auto tracks = tracker_->trackFrame(frame.left);

  // 2. Closed-Loop Visual Epipolar Motion Constraint
  if (!g_impl->prev_features.empty() && tracks.size() >= 8 && dt_frame > 0.001 && dt_frame < 0.2) {
    const Mat3d R_rel = g_impl->prev_rotation.transpose() * current_state_.R;

    std::vector<Vec3d> epipolar_normals;
    epipolar_normals.reserve(tracks.size());

    for (const auto& tr : tracks) {
      auto it = g_impl->prev_features.find(tr.track_id);
      if (it != g_impl->prev_features.end()) {
        Vec3d x1(it->second.x(), it->second.y(), 1.0);
        x1.normalize();

        Vec3d x2(tr.norm_point.x(), tr.norm_point.y(), 1.0);
        x2.normalize();

        Vec3d x1_derot = R_rel * x1;
        Vec3d normal = x2.cross(x1_derot);
        if (normal.norm() > 1e-5) {
          epipolar_normals.push_back(normal.normalized());
        }
      }
    }

    if (epipolar_normals.size() >= 8) {
      Eigen::MatrixXd A(epipolar_normals.size(), 3);
      for (std::size_t i = 0; i < epipolar_normals.size(); ++i) {
        A.row(static_cast<Eigen::Index>(i)) = epipolar_normals[i].transpose();
      }

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
      Vec3d t_cam = svd.matrixV().col(2);

      const Mat3d R_bc = config_.camera_calib.T_BS.block<3, 3>(0, 0);
      Vec3d t_body = R_bc * t_cam;
      Vec3d t_world = current_state_.R * t_body;

      const Vec3d imu_disp = preintegrator_->getDelta().delta_p;
      if (t_world.dot(current_state_.R * imu_disp) < 0.0) {
        t_world = -t_world;
      }

      double total_parallax = 0.0;
      int match_count = 0;
      for (const auto& tr : tracks) {
        auto it = g_impl->prev_features.find(tr.track_id);
        if (it != g_impl->prev_features.end()) {
          total_parallax += (tr.pixel - Vec2d(it->second.x() * config_.camera_calib.fx + config_.camera_calib.cx,
                                              it->second.y() * config_.camera_calib.fy + config_.camera_calib.cy)).norm();
          match_count++;
        }
      }
      double mean_parallax_px = match_count > 0 ? (total_parallax / match_count) : 1.0;

      double speed = std::clamp(mean_parallax_px * 0.08, 0.0, 2.2);
      double step_dist = speed * dt_frame;

      Vec3d delta_p = t_world.normalized() * step_dist;

      current_state_.p = g_impl->prev_position + delta_p;
      current_state_.v = delta_p / dt_frame;
    }
  }

  // Update history
  g_impl->prev_features.clear();
  for (const auto& tr : tracks) {
    g_impl->prev_features[tr.track_id] = tr.norm_point;
  }
  g_impl->prev_position = current_state_.p;
  g_impl->prev_rotation = current_state_.R;

  // 3. Register current state to trajectory
  estimated_trajectory_.push_back(current_state_);

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
