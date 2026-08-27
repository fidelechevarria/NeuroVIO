#include "neurovio/vio_pipeline.hpp"
#include <iomanip>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include "neurovio/frontend/neurotrack_frontend.hpp"
#include "neurovio/frontend/klt_tracker.hpp"

namespace neurovio {

struct Keyframe {
  TimestampNs timestamp_ns{0};
  Mat3d R_world{Mat3d::Identity()};
  Vec3d p_world{Vec3d::Zero()};
  std::unordered_map<uint64_t, Vec2d> features;
};

struct VioPipelineImpl {
  Keyframe last_keyframe;
  std::unordered_map<uint64_t, Vec2d> prev_features;
  Mat3d prev_R{Mat3d::Identity()};
  Vec3d prev_p{Vec3d::Zero()};
  double accumulated_distance{0.0};
};

static std::unique_ptr<VioPipelineImpl> g_impl;

static bool triangulatePoint(const Mat3d& R, const Vec3d& t, const Vec3d& x1, const Vec3d& x2, double& z1, double& z2) {
  Vec3d Rx1 = R * x1;
  Vec3d normal = x2.cross(Rx1);
  double denom = normal.squaredNorm();
  if (denom < 1e-7) return false;

  Vec3d tx2 = x2.cross(t);
  z1 = -tx2.dot(normal) / denom;
  Vec3d p2 = Rx1 * z1 + t;
  z2 = p2.dot(x2);
  return z1 > 0.02 && z2 > 0.02;
}

static bool recoverRelativePose(
    const std::vector<Vec3d>& rays1,
    const std::vector<Vec3d>& rays2,
    Mat3d& R_out,
    Vec3d& t_out) {
  const size_t N = rays1.size();
  if (N < 8) return false;

  Eigen::MatrixXd A(N, 9);
  for (size_t i = 0; i < N; ++i) {
    const Vec3d& x1 = rays1[i];
    const Vec3d& x2 = rays2[i];
    A.row(static_cast<Eigen::Index>(i)) <<
        x2.x() * x1.x(), x2.x() * x1.y(), x2.x() * x1.z(),
        x2.y() * x1.x(), x2.y() * x1.y(), x2.y() * x1.z(),
        x2.z() * x1.x(), x2.z() * x1.y(), x2.z() * x1.z();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  Eigen::VectorXd e_vec = svd.matrixV().col(8);
  Mat3d E;
  E << e_vec(0), e_vec(1), e_vec(2),
       e_vec(3), e_vec(4), e_vec(5),
       e_vec(6), e_vec(7), e_vec(8);

  Eigen::JacobiSVD<Mat3d> svd_E(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3d U = svd_E.matrixU();
  Mat3d V = svd_E.matrixV();
  if (U.determinant() < 0) U.col(2) *= -1.0;
  if (V.determinant() < 0) V.col(2) *= -1.0;

  Mat3d W;
  W << 0, -1, 0,
       1,  0, 0,
       0,  0, 1;

  Mat3d R1 = U * W * V.transpose();
  Mat3d R2 = U * W.transpose() * V.transpose();
  Vec3d t1 = U.col(2);
  Vec3d t2 = -U.col(2);

  if (R1.determinant() < 0) R1 = -R1;
  if (R2.determinant() < 0) R2 = -R2;

  Mat3d cand_R[4] = {R1, R1, R2, R2};
  Vec3d cand_t[4] = {t1, t2, t1, t2};

  int best_cand = 0;
  int max_positive_depths = -1;

  for (int c = 0; c < 4; ++c) {
    int valid_count = 0;
    for (size_t i = 0; i < std::min<size_t>(N, 35); ++i) {
      double z1 = 0, z2 = 0;
      if (triangulatePoint(cand_R[c], cand_t[c], rays1[i], rays2[i], z1, z2)) {
        valid_count++;
      }
    }
    if (valid_count > max_positive_depths) {
      max_positive_depths = valid_count;
      best_cand = c;
    }
  }

  R_out = cand_R[best_cand];
  t_out = cand_t[best_cand].normalized();
  return max_positive_depths >= 3;
}

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

  g_impl->prev_p = current_state_.p;
  g_impl->prev_R = current_state_.R;

  g_impl->last_keyframe.timestamp_ns = current_state_.timestamp_ns;
  g_impl->last_keyframe.R_world = current_state_.R;
  g_impl->last_keyframe.p_world = current_state_.p;

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

  // Initialize keyframe if empty
  if (g_impl->last_keyframe.features.empty()) {
    for (const auto& tr : tracks) {
      g_impl->last_keyframe.features[tr.track_id] = tr.norm_point;
    }
    g_impl->last_keyframe.R_world = current_state_.R;
    g_impl->last_keyframe.p_world = current_state_.p;
    g_impl->last_keyframe.timestamp_ns = frame.timestamp_ns;
    estimated_trajectory_.push_back(current_state_);
    last_frame_time_ns_ = frame.timestamp_ns;
    return;
  }

  // 2. Relative 6-DoF Pose against Reference Keyframe
  std::vector<Vec3d> kf_rays;
  std::vector<Vec3d> curr_rays;
  kf_rays.reserve(tracks.size());
  curr_rays.reserve(tracks.size());

  double total_parallax = 0.0;
  int match_count = 0;

  for (const auto& tr : tracks) {
    auto it = g_impl->last_keyframe.features.find(tr.track_id);
    if (it != g_impl->last_keyframe.features.end()) {
      Vec3d x1(it->second.x(), it->second.y(), 1.0);
      Vec3d x2(tr.norm_point.x(), tr.norm_point.y(), 1.0);
      kf_rays.push_back(x1.normalized());
      curr_rays.push_back(x2.normalized());

      double px_diff = (tr.pixel - Vec2d(it->second.x() * config_.camera_calib.fx + config_.camera_calib.cx,
                                         it->second.y() * config_.camera_calib.fy + config_.camera_calib.cy)).norm();
      total_parallax += px_diff;
      match_count++;
    }
  }

  double mean_parallax_px = match_count > 0 ? (total_parallax / match_count) : 0.0;

  Mat3d R_cam_rel = Mat3d::Identity();
  Vec3d t_cam_rel = Vec3d::Zero();

  if (kf_rays.size() >= 10 && recoverRelativePose(kf_rays, curr_rays, R_cam_rel, t_cam_rel)) {
    const Mat3d R_bc = config_.camera_calib.T_BS.block<3, 3>(0, 0);
    const Mat3d R_cb = R_bc.transpose();

    Mat3d R_body_rel = R_bc * R_cam_rel * R_cb;
    Vec3d t_body_rel = R_bc * t_cam_rel;

    // Update orientation from Keyframe
    current_state_.R = g_impl->last_keyframe.R_world * R_body_rel;

    // Metric scale from parallax and IMU displacement
    double time_since_kf = static_cast<double>(frame.timestamp_ns - g_impl->last_keyframe.timestamp_ns) * 1e-9;
    double scale = std::clamp(mean_parallax_px * 0.06, 0.05, 1.8);

    Vec3d delta_p = g_impl->last_keyframe.R_world * (t_body_rel * scale);
    current_state_.p = g_impl->last_keyframe.p_world + delta_p;
    current_state_.v = delta_p / std::max(time_since_kf, 0.01);
  }

  // Check for Keyframe Transition (parallax > 18 px or time > 0.4s or lost tracks)
  double time_since_kf = static_cast<double>(frame.timestamp_ns - g_impl->last_keyframe.timestamp_ns) * 1e-9;
  if (mean_parallax_px > 18.0 || time_since_kf > 0.45 || match_count < 25) {
    g_impl->last_keyframe.features.clear();
    for (const auto& tr : tracks) {
      g_impl->last_keyframe.features[tr.track_id] = tr.norm_point;
    }
    g_impl->last_keyframe.R_world = current_state_.R;
    g_impl->last_keyframe.p_world = current_state_.p;
    g_impl->last_keyframe.timestamp_ns = frame.timestamp_ns;
  }

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
