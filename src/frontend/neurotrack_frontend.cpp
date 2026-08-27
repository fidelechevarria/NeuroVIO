#include "neurovio/frontend/neurotrack_frontend.hpp"
#include <spdlog/spdlog.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "neurotrack/stb_image.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace neurovio {

NeuroTrackFrontend::NeuroTrackFrontend(
    const CameraCalibration& calib,
    Config config)
    : config_(config), calib_(calib) {
  neurotrack::TrackerConfig nt_config;
  nt_config.intrinsics.width = calib_.width;
  nt_config.intrinsics.height = calib_.height;
  nt_config.intrinsics.fx = calib_.fx;
  nt_config.intrinsics.fy = calib_.fy;
  nt_config.intrinsics.cx = calib_.cx;
  nt_config.intrinsics.cy = calib_.cy;

  if (calib_.distortion.size() >= 4) {
    nt_config.intrinsics.k1 = calib_.distortion[0];
    nt_config.intrinsics.k2 = calib_.distortion[1];
    nt_config.intrinsics.p1 = calib_.distortion[2];
    nt_config.intrinsics.p2 = calib_.distortion[3];
  }

  nt_config.spatial_config.max_total_features = config_.max_features;
  nt_config.spatial_config.cell_size_pixels = config_.cell_size_pixels;
  nt_config.max_local_search_radius_px = config_.max_search_radius_px;
  nt_config.inference_config.use_cuda = config_.use_cuda;

  tracker_ = std::make_unique<neurotrack::FeatureTracker>(nt_config);
  spdlog::info("NeuroTrackFrontend: Initialized Neural Visual Tracker with max {} features (cell: {} px)",
               config_.max_features, config_.cell_size_pixels);
}

NeuroTrackFrontend::~NeuroTrackFrontend() = default;

void NeuroTrackFrontend::reset() {
  if (tracker_) {
    tracker_->reset();
  }
}

neurotrack::TrackingStats NeuroTrackFrontend::stats() const noexcept {
  if (tracker_) {
    return tracker_->stats();
  }
  return {};
}

FeatureTrackMap NeuroTrackFrontend::trackFrame(const CameraFrame& frame) {
  if (!tracker_) return {};

  int width = calib_.width;
  int height = calib_.height;
  std::uint8_t* raw_img = nullptr;
  bool need_free = false;

#if defined(NEUROVIO_HAS_OPENCV)
  if (!frame.image.empty()) {
    raw_img = frame.image.data;
    width = frame.image.cols;
    height = frame.image.rows;
  }
#endif

  if (!raw_img && !frame.file_path.empty()) {
    int w = 0, h = 0, ch = 0;
    raw_img = stbi_load(frame.file_path.c_str(), &w, &h, &ch, 1);
    if (raw_img) {
      width = w;
      height = h;
      need_free = true;
    }
  }

  if (!raw_img) {
    spdlog::warn("NeuroTrackFrontend: No valid image data provided for frame timestamp {}", frame.timestamp_ns);
    return {};
  }

  auto nt_tracks = tracker_->processFrame(raw_img, width, height, static_cast<neurotrack::TimestampNs>(frame.timestamp_ns));

  if (need_free && raw_img) {
    stbi_image_free(raw_img);
  }

  FeatureTrackMap vio_tracks;
  vio_tracks.reserve(nt_tracks.size());

  for (const auto& track : nt_tracks) {
    FeaturePoint pt;
    pt.track_id = track.id;
    pt.pixel = Vec2d(static_cast<double>(track.current_point.pixel.x()),
                     static_cast<double>(track.current_point.pixel.y()));
    pt.norm_point = Vec2d(track.current_point.norm_ray.x(),
                          track.current_point.norm_ray.y());
    pt.track_length = track.age;
    pt.is_inlier = !track.is_outlier;
    pt.velocity = Vec2d::Zero();

    vio_tracks.push_back(std::move(pt));
  }

  return vio_tracks;
}

}  // namespace neurovio
