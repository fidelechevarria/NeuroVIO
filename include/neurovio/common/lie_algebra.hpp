#pragma once

#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include "neurovio/common/types.hpp"

namespace neurovio {

class LieAlgebra {
 public:
  static constexpr double kEpsilon = 1e-10;

  // ---------------------------------------------------------------------------
  // SO(3) Lie Algebra: Hat (skew-symmetric) & Vee (vector extraction)
  // ---------------------------------------------------------------------------
  [[nodiscard]] static inline Mat3d hat(const Vec3d& omega) noexcept {
    Mat3d Omega;
    Omega << 0.0, -omega.z(), omega.y(),
             omega.z(), 0.0, -omega.x(),
             -omega.y(), omega.x(), 0.0;
    return Omega;
  }

  [[nodiscard]] static inline Vec3d vee(const Mat3d& Omega) noexcept {
    return Vec3d(Omega(2, 1), Omega(0, 2), Omega(1, 0));
  }

  // ---------------------------------------------------------------------------
  // SO(3) Exponential Map (Rodrigues Formula)
  // ---------------------------------------------------------------------------
  [[nodiscard]] static inline Mat3d expSO3(const Vec3d& phi) noexcept {
    const double theta2 = phi.squaredNorm();
    const double theta = std::sqrt(theta2);
    const Mat3d Phi = hat(phi);

    if (theta < kEpsilon) {
      return Mat3d::Identity() + Phi + 0.5 * Phi * Phi;
    }

    const double sin_theta = std::sin(theta);
    const double cos_theta = std::cos(theta);
    return Mat3d::Identity() + (sin_theta / theta) * Phi +
           ((1.0 - cos_theta) / theta2) * (Phi * Phi);
  }

  // ---------------------------------------------------------------------------
  // SO(3) Logarithmic Map
  // ---------------------------------------------------------------------------
  [[nodiscard]] static inline Vec3d logSO3(const Mat3d& R) noexcept {
    const double cos_theta = (R.trace() - 1.0) * 0.5;
    const double clamped_cos = std::clamp(cos_theta, -1.0, 1.0);
    const double theta = std::acos(clamped_cos);

    if (theta < kEpsilon) {
      return vee(0.5 * (R - R.transpose()));
    }

    const double sin_theta = std::sin(theta);
    if (std::abs(sin_theta) < kEpsilon) {
      // 180 degrees singularity handling
      const Mat3d sym = 0.5 * (R + Mat3d::Identity());
      Vec3d axis = sym.diagonal().cwiseSqrt();
      if (R(0, 1) < 0) axis.y() = -axis.y();
      if (R(0, 2) < 0) axis.z() = -axis.z();
      return theta * axis.normalized();
    }

    const Mat3d skew = (theta / (2.0 * sin_theta)) * (R - R.transpose());
    return vee(skew);
  }

  // ---------------------------------------------------------------------------
  // SO(3) Right Jacobian Jr(phi) such that Exp(phi + delta_phi) ~ Exp(phi) * Exp(Jr * delta_phi)
  // ---------------------------------------------------------------------------
  [[nodiscard]] static inline Mat3d rightJacobianSO3(const Vec3d& phi) noexcept {
    const double theta2 = phi.squaredNorm();
    const double theta = std::sqrt(theta2);
    const Mat3d Phi = hat(phi);

    if (theta < kEpsilon) {
      return Mat3d::Identity() - 0.5 * Phi;
    }

    const double sin_theta = std::sin(theta);
    const double cos_theta = std::cos(theta);

    return Mat3d::Identity() - ((1.0 - cos_theta) / theta2) * Phi +
           ((theta - sin_theta) / (theta2 * theta)) * (Phi * Phi);
  }

  // ---------------------------------------------------------------------------
  // SO(3) Inverse Right Jacobian Jr_inv(phi)
  // ---------------------------------------------------------------------------
  [[nodiscard]] static inline Mat3d rightJacobianSO3Inv(const Vec3d& phi) noexcept {
    const double theta2 = phi.squaredNorm();
    const double theta = std::sqrt(theta2);
    const Mat3d Phi = hat(phi);

    if (theta < kEpsilon) {
      return Mat3d::Identity() + 0.5 * Phi;
    }

    return Mat3d::Identity() + 0.5 * Phi +
           (1.0 / theta2 - (1.0 + cos(theta)) / (2.0 * theta * sin(theta))) * (Phi * Phi);
  }

  // ---------------------------------------------------------------------------
  // SE(3) Exponential and Logarithmic Maps
  // ---------------------------------------------------------------------------
  [[nodiscard]] static inline Mat4d expSE3(const Vec6d& xi) noexcept {
    const Vec3d rho = xi.head<3>();
    const Vec3d phi = xi.tail<3>();
    const Mat3d R = expSO3(phi);
    const Mat3d Jr = rightJacobianSO3(phi);
    const Vec3d t = Jr * rho;

    Mat4d T = Mat4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
  }

  [[nodiscard]] static inline Vec6d logSE3(const Mat4d& T) noexcept {
    const Mat3d R = T.block<3, 3>(0, 0);
    const Vec3d t = T.block<3, 1>(0, 3);
    const Vec3d phi = logSO3(R);
    const Mat3d Jr_inv = rightJacobianSO3Inv(phi);
    const Vec3d rho = Jr_inv * t;

    Vec6d xi;
    xi.head<3>() = rho;
    xi.tail<3>() = phi;
    return xi;
  }

  // ---------------------------------------------------------------------------
  // SE(3) Adjoint Matrix (6x6)
  // ---------------------------------------------------------------------------
  [[nodiscard]] static inline Mat6d adjointSE3(const Mat4d& T) noexcept {
    const Mat3d R = T.block<3, 3>(0, 0);
    const Vec3d t = T.block<3, 1>(0, 3);
    Mat6d Adj = Mat6d::Zero();
    Adj.block<3, 3>(0, 0) = R;
    Adj.block<3, 3>(0, 3) = hat(t) * R;
    Adj.block<3, 3>(3, 3) = R;
    return Adj;
  }
};

}  // namespace neurovio
