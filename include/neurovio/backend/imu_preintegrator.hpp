#pragma once

#include <memory>
#include <vector>
#include "neurovio/common/lie_algebra.hpp"
#include "neurovio/sensors/imu_data.hpp"

#if defined(NEUROVIO_HAS_GTSAM)
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#endif

namespace neurovio {

// -----------------------------------------------------------------------------
// Preintegrated IMU Delta & Jacobians Structure (Analytical & Self-Contained)
// -----------------------------------------------------------------------------
struct PreintegratedImuDelta {
  double delta_t{0.0};
  Mat3d delta_R{Mat3d::Identity()};
  Vec3d delta_v{Vec3d::Zero()};
  Vec3d delta_p{Vec3d::Zero()};

  // First-order Taylor series Jacobians w.r.t initial bias estimate
  Mat3d dR_dbg{Mat3d::Zero()};
  Mat3d dv_dbg{Mat3d::Zero()};
  Mat3d dv_dba{Mat3d::Zero()};
  Mat3d dp_dbg{Mat3d::Zero()};
  Mat3d dp_dba{Mat3d::Zero()};

  // 9x9 covariance matrix for [delta_theta, delta_v, delta_p]
  Mat9d covariance{Mat9d::Zero()};
  Vec3d linearized_ba{Vec3d::Zero()};
  Vec3d linearized_bg{Vec3d::Zero()};
};

class ImuPreintegrator {
 public:
  explicit ImuPreintegrator(ImuParameters params,
                            const Vec3d& initial_ba = Vec3d::Zero(),
                            const Vec3d& initial_bg = Vec3d::Zero());

  void reset(const Vec3d& current_ba, const Vec3d& current_bg);

  void integrate(const ImuMeasurement& imu, double dt);

  [[nodiscard]] NavState predict(const NavState& start_state) const;

  [[nodiscard]] const PreintegratedImuDelta& getDelta() const noexcept { return delta_; }
  [[nodiscard]] const ImuParameters& getParams() const noexcept { return params_; }

  // Update preintegration with bias updates (Taylor expansion on Lie manifold)
  [[nodiscard]] Mat3d correctedDeltaR(const Vec3d& bg) const;
  [[nodiscard]] Vec3d correctedDeltaV(const Vec3d& ba, const Vec3d& bg) const;
  [[nodiscard]] Vec3d correctedDeltaP(const Vec3d& ba, const Vec3d& bg) const;

 private:
  ImuParameters params_;
  PreintegratedImuDelta delta_;
  Mat9d noise_cov_{Mat9d::Zero()};

#if defined(NEUROVIO_HAS_GTSAM)
  boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> gtsam_params_;
  std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> gtsam_preintegrated_;
#endif
};

}  // namespace neurovio
