#include <gtest/gtest.h>
#include "neurovio/backend/imu_preintegrator.hpp"

using namespace neurovio;

TEST(ImuPreintegratorTest, PureRotationIntegration) {
  ImuParameters params;
  ImuPreintegrator preint(params);

  const Vec3d omega(0.0, 0.0, 1.0);  // 1 rad/s around Z-axis
  const double dt = 0.01;
  const int steps = 100;  // 1.0 second total -> 1.0 radian rotation

  for (int i = 0; i < steps; ++i) {
    ImuMeasurement imu(static_cast<int64_t>(i * dt * 1e9), Vec3d::Zero(), omega);
    preint.integrate(imu, dt);
  }

  const Mat3d expected_R = LieAlgebra::expSO3(omega * 1.0);
  const Mat3d actual_R = preint.getDelta().delta_R;

  EXPECT_NEAR((expected_R - actual_R).norm(), 0.0, 1e-6);
}

TEST(ImuPreintegratorTest, StationaryPrediction) {
  ImuParameters params;
  params.gravity = Vec3d(0.0, 0.0, -9.81);
  ImuPreintegrator preint(params);

  // When stationary, body accelerometer measures upward specific force (reaction force): -gravity
  const Vec3d a_measured(0.0, 0.0, 9.81);
  const Vec3d w_measured(0.0, 0.0, 0.0);
  const double dt = 0.01;

  for (int i = 0; i < 100; ++i) {
    ImuMeasurement imu(static_cast<int64_t>(i * dt * 1e9), a_measured, w_measured);
    preint.integrate(imu, dt);
  }

  NavState start_state;
  start_state.p = Vec3d(1.0, 2.0, 3.0);
  start_state.v = Vec3d::Zero();
  start_state.R = Mat3d::Identity();

  const NavState predicted = preint.predict(start_state);

  // Position and velocity should remain constant because reaction force cancels gravity
  EXPECT_NEAR((predicted.p - start_state.p).norm(), 0.0, 1e-4);
  EXPECT_NEAR((predicted.v - start_state.v).norm(), 0.0, 1e-4);
}
