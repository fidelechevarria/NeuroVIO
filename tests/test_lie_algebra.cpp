#include <gtest/gtest.h>
#include "neurovio/common/lie_algebra.hpp"

using namespace neurovio;

TEST(LieAlgebraTest, HatAndVeeAreInverses) {
  const Vec3d v(1.23, -4.56, 7.89);
  const Mat3d V_hat = LieAlgebra::hat(v);
  const Vec3d v_recovered = LieAlgebra::vee(V_hat);

  EXPECT_NEAR((v - v_recovered).norm(), 0.0, 1e-12);
  EXPECT_NEAR(V_hat(0, 0), 0.0, 1e-12);
  EXPECT_NEAR(V_hat(1, 1), 0.0, 1e-12);
  EXPECT_NEAR(V_hat(2, 2), 0.0, 1e-12);
  EXPECT_NEAR((V_hat + V_hat.transpose()).norm(), 0.0, 1e-12);
}

TEST(LieAlgebraTest, ExpAndLogSO3Roundtrip) {
  std::vector<Vec3d> test_vectors = {
      Vec3d(0.0, 0.0, 0.0),
      Vec3d(1e-8, 0.0, 0.0),
      Vec3d(0.1, -0.2, 0.3),
      Vec3d(M_PI * 0.5, 0.0, 0.0),
      Vec3d(0.0, M_PI * 0.99, 0.0),
      Vec3d(-1.0, 1.0, 0.5)};

  for (const auto& phi : test_vectors) {
    const Mat3d R = LieAlgebra::expSO3(phi);

    // Verify orthogonality R * R^T = I and det(R) = +1
    EXPECT_NEAR((R * R.transpose() - Mat3d::Identity()).norm(), 0.0, 1e-10);
    EXPECT_NEAR(R.determinant(), 1.0, 1e-10);

    const Vec3d phi_recovered = LieAlgebra::logSO3(R);
    EXPECT_NEAR((phi - phi_recovered).norm(), 0.0, 1e-10);
  }
}

TEST(LieAlgebraTest, RightJacobianInverseIdentity) {
  std::vector<Vec3d> test_vectors = {
      Vec3d(0.0, 0.0, 0.0),
      Vec3d(0.05, -0.1, 0.2),
      Vec3d(0.5, 0.8, -0.3)};

  for (const auto& phi : test_vectors) {
    const Mat3d Jr = LieAlgebra::rightJacobianSO3(phi);
    const Mat3d Jr_inv = LieAlgebra::rightJacobianSO3Inv(phi);

    const Mat3d identity_check = Jr * Jr_inv;
    EXPECT_NEAR((identity_check - Mat3d::Identity()).norm(), 0.0, 1e-10);
  }
}

TEST(LieAlgebraTest, ExpAndLogSE3Roundtrip) {
  Vec6d xi;
  xi << 1.0, -2.0, 3.0, 0.1, -0.2, 0.15;  // translation + rotation

  const Mat4d T = LieAlgebra::expSE3(xi);
  EXPECT_NEAR((T.block<3, 3>(0, 0) * T.block<3, 3>(0, 0).transpose() - Mat3d::Identity()).norm(),
              0.0, 1e-10);

  const Vec6d xi_recovered = LieAlgebra::logSE3(T);
  EXPECT_NEAR((xi - xi_recovered).norm(), 0.0, 1e-9);
}
