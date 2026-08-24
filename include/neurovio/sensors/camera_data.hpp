#pragma once

#include <string>
#include "neurovio/common/timestamps.hpp"
#include "neurovio/common/types.hpp"

#if defined(NEUROVIO_HAS_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace neurovio {

// -----------------------------------------------------------------------------
// Image Frame Representation
// -----------------------------------------------------------------------------
struct CameraFrame {
  TimestampNs timestamp_ns{0};
  uint32_t camera_id{0};         // 0: left (cam0), 1: right (cam1)
  uint64_t frame_id{0};
  std::string file_path{};

#if defined(NEUROVIO_HAS_OPENCV)
  cv::Mat image{};
#endif

  CameraFrame() = default;
  CameraFrame(TimestampNs t, uint32_t cam_id, std::string path)
      : timestamp_ns(t), camera_id(cam_id), file_path(std::move(path)) {}
};

// -----------------------------------------------------------------------------
// Stereo Frame Pair (Time-Synchronized)
// -----------------------------------------------------------------------------
struct StereoFrame {
  TimestampNs timestamp_ns{0};
  CameraFrame left{};
  CameraFrame right{};
  bool has_right{false};
};

}  // namespace neurovio
