#pragma once

#include "neurovio/common/timestamps.hpp"
#include "neurovio/common/types.hpp"

namespace neurovio {

// -----------------------------------------------------------------------------
// Raw IMU Measurement
// -----------------------------------------------------------------------------
struct ImuMeasurement {
  TimestampNs timestamp_ns{0};
  Vec3d accel{Vec3d::Zero()};  // Linear acceleration in body frame (m/s^2)
  Vec3d gyro{Vec3d::Zero()};   // Angular velocity in body frame (rad/s)

  ImuMeasurement() = default;
  ImuMeasurement(TimestampNs t, const Vec3d& a, const Vec3d& w)
      : timestamp_ns(t), accel(a), gyro(w) {}
};

// -----------------------------------------------------------------------------
// IMU Noise & Random Walk Parameters (Continuous-time)
// -----------------------------------------------------------------------------
struct ImuParameters {
  double gyro_noise_density{1.6968e-04};        // rad / (s * sqrt(Hz))
  double accel_noise_density{2.0000e-3};        // m / (s^2 * sqrt(Hz))
  double gyro_random_walk{1.9393e-05};          // rad / (s^2 * sqrt(Hz))
  double accel_random_walk{3.0000e-3};          // m / (s^3 * sqrt(Hz))
  double rate_hz{200.0};                         // IMU sampling frequency
  Vec3d gravity{0.0, 0.0, -9.81};                // World gravity vector (m/s^2)

  // Body to IMU Extrinsics (often identity if body frame == IMU frame)
  Mat4d T_BS{Mat4d::Identity()};
};

}  // namespace neurovio
