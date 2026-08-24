#pragma once

#include <memory>
#include <vector>
#include "neurovio/common/types.hpp"
#include "neurovio/sensors/camera_data.hpp"

namespace neurovio {

struct FeaturePoint {
  uint64_t track_id{0};
  Vec2d pixel{Vec2d::Zero()};          // Raw pixel coordinate (u, v)
  Vec2d norm_point{Vec2d::Zero()};     // Normalized spherical/plane coordinate (x, y)
  Vec2d velocity{Vec2d::Zero()};       // Pixel velocity (pixels/s)
  uint32_t track_length{1};
  bool is_inlier{true};
};

using FeatureTrackMap = std::vector<FeaturePoint>;

class FeatureTracker {
 public:
  virtual ~FeatureTracker() = default;

  virtual FeatureTrackMap trackFrame(const CameraFrame& frame) = 0;
  virtual void reset() = 0;
};

}  // namespace neurovio
