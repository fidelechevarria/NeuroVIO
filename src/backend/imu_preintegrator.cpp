#include "neurovio/backend/imu_preintegrator.hpp"

namespace neurovio {

ImuPreintegrator::ImuPreintegrator(ImuParameters params, const Vec3d& initial_ba,
                                   const Vec3d& initial_bg)
    : params_(std::move(params)) {
  reset(initial_ba, initial_bg);
}

void ImuPreintegrator::reset(const Vec3d& current_ba, const Vec3d& current_bg) {
  delta_.delta_t = 0.0;
  delta_.delta_R = Mat3d::Identity();
  delta_.delta_v = Vec3d::Zero();
  delta_.delta_p = Vec3d::Zero();

  delta_.dR_dbg = Mat3d::Zero();
  delta_.dv_dbg = Mat3d::Zero();
  delta_.dv_dba = Mat3d::Zero();
  delta_.dp_dbg = Mat3d::Zero();
  delta_.dp_dba = Mat3d::Zero();

  delta_.covariance = Mat9d::Zero();
  delta_.linearized_ba = current_ba;
  delta_.linearized_bg = current_bg;
}

void ImuPreintegrator::integrate(const ImuMeasurement& imu, double dt) {
  if (dt <= 0.0) return;

  const Vec3d w_unbiased = imu.gyro - delta_.linearized_bg;
  const Vec3d a_unbiased = imu.accel - delta_.linearized_ba;

  const Vec3d phi = w_unbiased * dt;
  const Mat3d dR = LieAlgebra::expSO3(phi);
  const Mat3d Jr = LieAlgebra::rightJacobianSO3(phi);

  const Mat3d R_k = delta_.delta_R;
  const Vec3d a_world = R_k * a_unbiased;
  const Mat3d a_skew = LieAlgebra::hat(a_unbiased);

  // Position and velocity propagation
  delta_.delta_p += delta_.delta_v * dt + 0.5 * a_world * dt * dt;
  delta_.delta_v += a_world * dt;
  delta_.delta_R = delta_.delta_R * dR;
  delta_.delta_t += dt;

  // Jacobian matrix propagation: F matrix (9x9)
  Mat9d F = Mat9d::Identity();
  F.block<3, 3>(0, 0) = dR.transpose();
  F.block<3, 3>(3, 0) = -R_k * a_skew * dt;
  F.block<3, 3>(6, 0) = -0.5 * R_k * a_skew * dt * dt;
  F.block<3, 3>(6, 3) = Mat3d::Identity() * dt;

  // Noise covariance matrix Q (6x6: gyro noise, accel noise)
  Eigen::Matrix<double, 9, 6> G = Eigen::Matrix<double, 9, 6>::Zero();
  G.block<3, 3>(0, 0) = Jr * dt;
  G.block<3, 3>(3, 3) = R_k * dt;
  G.block<3, 3>(6, 3) = 0.5 * R_k * dt * dt;

  Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
  const double gyro_var = params_.gyro_noise_density * params_.gyro_noise_density / dt;
  const double acc_var = params_.accel_noise_density * params_.accel_noise_density / dt;
  Q.block<3, 3>(0, 0) = Mat3d::Identity() * gyro_var;
  Q.block<3, 3>(3, 3) = Mat3d::Identity() * acc_var;

  // Propagate covariance: Sigma = F * Sigma * F^T + G * Q * G^T
  delta_.covariance = F * delta_.covariance * F.transpose() + G * Q * G.transpose();

  // Propagate Jacobians with respect to bias updates
  delta_.dp_dbg += delta_.dv_dbg * dt - 0.5 * R_k * a_skew * delta_.dR_dbg * dt * dt;
  delta_.dp_dba += delta_.dv_dba * dt - 0.5 * R_k * dt * dt;

  delta_.dv_dbg += -R_k * a_skew * delta_.dR_dbg * dt;
  delta_.dv_dba += -R_k * dt;

  delta_.dR_dbg = dR.transpose() * delta_.dR_dbg - Jr * dt;
}

Mat3d ImuPreintegrator::correctedDeltaR(const Vec3d& bg) const {
  const Vec3d dbg = bg - delta_.linearized_bg;
  return delta_.delta_R * LieAlgebra::expSO3(delta_.dR_dbg * dbg);
}

Vec3d ImuPreintegrator::correctedDeltaV(const Vec3d& ba, const Vec3d& bg) const {
  const Vec3d dba = ba - delta_.linearized_ba;
  const Vec3d dbg = bg - delta_.linearized_bg;
  return delta_.delta_v + delta_.dv_dba * dba + delta_.dv_dbg * dbg;
}

Vec3d ImuPreintegrator::correctedDeltaP(const Vec3d& ba, const Vec3d& bg) const {
  const Vec3d dba = ba - delta_.linearized_ba;
  const Vec3d dbg = bg - delta_.linearized_bg;
  return delta_.delta_p + delta_.dp_dba * dba + delta_.dp_dbg * dbg;
}

NavState ImuPreintegrator::predict(const NavState& start_state) const {
  const double dt = delta_.delta_t;
  const Mat3d R_corr = correctedDeltaR(start_state.bg);
  const Vec3d v_corr = correctedDeltaV(start_state.ba, start_state.bg);
  const Vec3d p_corr = correctedDeltaP(start_state.ba, start_state.bg);

  NavState predicted;
  predicted.timestamp_ns = start_state.timestamp_ns + static_cast<int64_t>(dt * 1e9);
  predicted.R = start_state.R * R_corr;
  predicted.p = start_state.p + start_state.v * dt + 0.5 * params_.gravity * dt * dt +
                start_state.R * p_corr;
  predicted.v = start_state.v + params_.gravity * dt + start_state.R * v_corr;
  predicted.ba = start_state.ba;
  predicted.bg = start_state.bg;

  return predicted;
}

}  // namespace neurovio
