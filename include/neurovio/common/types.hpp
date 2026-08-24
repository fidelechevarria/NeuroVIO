#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace neurovio {

// -----------------------------------------------------------------------------
// Floating Point & Integer Type Aliases
// -----------------------------------------------------------------------------
using real_t = double;
using TimestampNs = int64_t;  // Nanoseconds since epoch

// -----------------------------------------------------------------------------
// Eigen Vector & Matrix Typedefs
// -----------------------------------------------------------------------------
using Vec2d = Eigen::Vector2d;
using Vec3d = Eigen::Vector3d;
using Vec4d = Eigen::Vector4d;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Vec9d = Eigen::Matrix<double, 9, 1>;
using Vec15d = Eigen::Matrix<double, 15, 1>;

using Mat2d = Eigen::Matrix2d;
using Mat3d = Eigen::Matrix3d;
using Mat4d = Eigen::Matrix4d;
using Mat6d = Eigen::Matrix<double, 6, 6>;
using Mat9d = Eigen::Matrix<double, 9, 9>;
using Mat15d = Eigen::Matrix<double, 15, 15>;

using Quatd = Eigen::Quaterniond;

// -----------------------------------------------------------------------------
// Navigation State on Manifold SO(3) x R^3 x R^3 x R^3 x R^3
// -----------------------------------------------------------------------------
struct NavState {
  TimestampNs timestamp_ns{0};
  Mat3d R{Mat3d::Identity()};     // Rotation from Body to World frame (R_wb)
  Vec3d p{Vec3d::Zero()};          // Position in World frame (p_wb)
  Vec3d v{Vec3d::Zero()};          // Velocity in World frame (v_wb)
  Vec3d bg{Vec3d::Zero()};         // Gyroscope bias (rad/s)
  Vec3d ba{Vec3d::Zero()};         // Accelerometer bias (m/s^2)

  NavState() = default;

  NavState(TimestampNs t, const Mat3d& rot, const Vec3d& pos, const Vec3d& vel,
           const Vec3d& gyro_bias = Vec3d::Zero(), const Vec3d& acc_bias = Vec3d::Zero())
      : timestamp_ns(t), R(rot), p(pos), v(vel), bg(gyro_bias), ba(acc_bias) {}

  [[nodiscard]] Quatd quaternion() const { return Quatd(R); }

  [[nodiscard]] Mat4d transformationMatrix() const {
    Mat4d T = Mat4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = p;
    return T;
  }
};

// -----------------------------------------------------------------------------
// Camera Intrinsics (Pinhole-RadTan Model)
// -----------------------------------------------------------------------------
struct CameraCalibration {
  int width{752};
  int height{480};
  double fx{458.654};
  double fy{457.296};
  double cx{367.215};
  double cy{248.375};

  // Radial-Tangential distortion [k1, k2, p1, p2]
  std::vector<double> distortion{-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};

  // Camera to IMU Extrinsics (T_BS: Transform from sensor to body/IMU frame)
  Mat4d T_BS{Mat4d::Identity()};

  [[nodiscard]] Mat3d K() const {
    Mat3d K_mat = Mat3d::Identity();
    K_mat(0, 0) = fx;
    K_mat(1, 1) = fy;
    K_mat(0, 2) = cx;
    K_mat(1, 2) = cy;
    return K_mat;
  }
};

}  // namespace neurovio
