#pragma once

#include <memory>
#include <string>
#include <vector>
#include "neurovio/common/types.hpp"
#include "neurovio/frontend/feature_tracker.hpp"

#if defined(NEUROVIO_HAS_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace neurovio {

struct NeuralFeatures {
  std::vector<Vec2d> keypoints;             // (x, y) coordinates
  std::vector<Vec2d> norm_keypoints;        // Normalized camera frame coordinates
  std::vector<std::vector<float>> descriptors; // 256-dimensional descriptor vectors
  std::vector<float> scores;                // Keypoint confidence scores
};

class OnnxSuperPoint {
 public:
  struct Config {
    std::string model_path{"models/superpoint.onnx"};
    bool use_cuda{false};
    int max_keypoints{512};
    float keypoint_threshold{0.005f};
    int input_width{752};
    int input_height{480};
  };

  explicit OnnxSuperPoint(Config config);
  ~OnnxSuperPoint();

  [[nodiscard]] bool isLoaded() const noexcept;

#if defined(NEUROVIO_HAS_OPENCV)
  [[nodiscard]] NeuralFeatures extract(const cv::Mat& image,
                                       const CameraCalibration* calib = nullptr);
#endif

 private:
  Config config_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace neurovio
