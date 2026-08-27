#pragma once

#include <memory>
#include <vector>
#include "neurovio/frontend/feature_tracker.hpp"
#include "neurotrack/feature_tracker.hpp"

namespace neurovio {

struct NeuroTrackConfig {
  int max_features{250};
  int cell_size_pixels{32};
  float max_search_radius_px{16.0f};
  bool use_cuda{true};
};

/**
 * @brief High-throughput Deep Neural Tracking Frontend adapter for NeuroVIO wrapping NeuroTrack.
 */
class NeuroTrackFrontend : public FeatureTracker {
 public:
  using Config = NeuroTrackConfig;

  explicit NeuroTrackFrontend(
      const CameraCalibration& calib,
      Config config = Config{});

  ~NeuroTrackFrontend() override;

  /**
   * @brief Track incoming camera frame using NeuroTrack C++/CUDA engine.
   * @param frame CameraFrame containing file_path or raw image buffer.
   * @return Active feature tracks with normalized 3D bearing rays and IDs.
   */
  FeatureTrackMap trackFrame(const CameraFrame& frame) override;

  /**
   * @brief Resets all active feature tracks and re-initializes memory.
   */
  void reset() override;

  [[nodiscard]] neurotrack::TrackingStats stats() const noexcept;

 private:
  Config config_;
  CameraCalibration calib_;
  std::unique_ptr<neurotrack::FeatureTracker> tracker_;
};

}  // namespace neurovio
