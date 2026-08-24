#pragma once

#include <vector>
#include "neurovio/frontend/feature_tracker.hpp"

#if defined(NEUROVIO_HAS_OPENCV)
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#endif

namespace neurovio {

struct KltTrackerConfig {
  int max_features{200};
  int min_distance{25};                // Minimum pixel distance between detected corners
  double quality_level{0.01};          // Shi-Tomasi corner detection quality
  int grid_rows{4};                    // Spatial bucketing grid rows
  int grid_cols{5};                    // Spatial bucketing grid cols
  double max_bidirectional_error{1.0}; // Forward-backward pixel error threshold
  bool enable_ransac_outlier_filter{true};
  double ransac_threshold_px{1.5};
};

class KltTracker : public FeatureTracker {
 public:
  using Config = KltTrackerConfig;

  explicit KltTracker(CameraCalibration calib, Config config = Config());
  ~KltTracker() override = default;

  FeatureTrackMap trackFrame(const CameraFrame& frame) override;
  void reset() override;

  [[nodiscard]] const CameraCalibration& getCalibration() const noexcept { return calib_; }

 private:
  CameraCalibration calib_;
  Config config_;
  uint64_t next_track_id_{0};

#if defined(NEUROVIO_HAS_OPENCV)
  cv::Mat prev_image_;
  std::vector<cv::Point2f> prev_points_;
  std::vector<uint64_t> prev_track_ids_;
  std::vector<uint32_t> prev_track_lengths_;
  TimestampNs prev_timestamp_ns_{0};

  void detectNewFeatures(const cv::Mat& image,
                         std::vector<cv::Point2f>& points,
                         std::vector<uint64_t>& ids,
                         std::vector<uint32_t>& lengths);
  [[nodiscard]] Vec2d unprojectPixel(const cv::Point2f& pt) const;
#endif
};

}  // namespace neurovio
