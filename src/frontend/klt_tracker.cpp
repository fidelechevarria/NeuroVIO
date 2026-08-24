#include "neurovio/frontend/klt_tracker.hpp"
#include <spdlog/spdlog.h>

#if defined(NEUROVIO_HAS_OPENCV)
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace neurovio {

KltTracker::KltTracker(CameraCalibration calib, Config config)
    : calib_(std::move(calib)), config_(config) {}

void KltTracker::reset() {
#if defined(NEUROVIO_HAS_OPENCV)
  prev_image_.release();
  prev_points_.clear();
  prev_track_ids_.clear();
  prev_track_lengths_.clear();
  prev_timestamp_ns_ = 0;
#endif
  next_track_id_ = 0;
}

#if defined(NEUROVIO_HAS_OPENCV)
Vec2d KltTracker::unprojectPixel(const cv::Point2f& pt) const {
  std::vector<cv::Point2f> src{pt};
  std::vector<cv::Point2f> dst;

  cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
  K.at<double>(0, 0) = calib_.fx;
  K.at<double>(1, 1) = calib_.fy;
  K.at<double>(0, 2) = calib_.cx;
  K.at<double>(1, 2) = calib_.cy;

  cv::Mat D = cv::Mat(calib_.distortion).clone();
  cv::undistortPoints(src, dst, K, D);

  return Vec2d(dst[0].x, dst[0].y);
}

void KltTracker::detectNewFeatures(const cv::Mat& image,
                                   std::vector<cv::Point2f>& points,
                                   std::vector<uint64_t>& ids,
                                   std::vector<uint32_t>& lengths) {
  const int target_features = config_.max_features;
  const int current_count = static_cast<int>(points.size());
  if (current_count >= target_features) return;

  // Create a spatial occupancy mask to prevent detecting points too close to existing tracks
  cv::Mat mask = cv::Mat::ones(image.size(), CV_8UC1) * 255;
  for (const auto& pt : points) {
    cv::circle(mask, pt, config_.min_distance, 0, -1);
  }

  const int needed = target_features - current_count;
  std::vector<cv::Point2f> new_corners;
  cv::goodFeaturesToTrack(image, new_corners, needed, config_.quality_level,
                          config_.min_distance, mask, 3, false, 0.04);

  for (const auto& corner : new_corners) {
    points.push_back(corner);
    ids.push_back(next_track_id_++);
    lengths.push_back(1);
  }
}
#endif

FeatureTrackMap KltTracker::trackFrame([[maybe_unused]] const CameraFrame& frame) {
  FeatureTrackMap result;

#if defined(NEUROVIO_HAS_OPENCV)
  if (frame.image.empty()) {
    spdlog::warn("KltTracker: Empty frame received");
    return result;
  }

  // First frame: initialize keypoints
  if (prev_image_.empty() || prev_points_.empty()) {
    prev_image_ = frame.image.clone();
    prev_points_.clear();
    prev_track_ids_.clear();
    prev_track_lengths_.clear();

    detectNewFeatures(prev_image_, prev_points_, prev_track_ids_, prev_track_lengths_);
    prev_timestamp_ns_ = frame.timestamp_ns;

    for (size_t i = 0; i < prev_points_.size(); ++i) {
      FeaturePoint fp;
      fp.track_id = prev_track_ids_[i];
      fp.pixel = Vec2d(prev_points_[i].x, prev_points_[i].y);
      fp.norm_point = unprojectPixel(prev_points_[i]);
      fp.velocity = Vec2d::Zero();
      fp.track_length = prev_track_lengths_[i];
      fp.is_inlier = true;
      result.push_back(fp);
    }
    return result;
  }

  // 1. Forward Optical Flow: prev_image -> current_image
  std::vector<cv::Point2f> forward_points;
  std::vector<uchar> forward_status;
  std::vector<float> forward_err;
  const cv::Size win_size(21, 21);

  cv::calcOpticalFlowPyrLK(prev_image_, frame.image, prev_points_, forward_points,
                           forward_status, forward_err, win_size, 3,
                           cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                           0, 0.001);

  // 2. Backward Optical Flow: current_image -> prev_image (Consistency Check)
  std::vector<cv::Point2f> backward_points;
  std::vector<uchar> backward_status;
  std::vector<float> backward_err;

  cv::calcOpticalFlowPyrLK(frame.image, prev_image_, forward_points, backward_points,
                           backward_status, backward_err, win_size, 3,
                           cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                           0, 0.001);

  // 3. Filter by bidirectional distance error and image boundary
  std::vector<cv::Point2f> tracked_points;
  std::vector<cv::Point2f> tracked_prev_points;
  std::vector<uint64_t> tracked_ids;
  std::vector<uint32_t> tracked_lengths;

  const double dt = (frame.timestamp_ns > prev_timestamp_ns_)
                        ? static_cast<double>(frame.timestamp_ns - prev_timestamp_ns_) * 1e-9
                        : 0.033;

  for (size_t i = 0; i < forward_points.size(); ++i) {
    if (!forward_status[i] || !backward_status[i]) continue;

    const double dist_fb = cv::norm(prev_points_[i] - backward_points[i]);
    if (dist_fb > config_.max_bidirectional_error) continue;

    const auto& pt = forward_points[i];
    if (pt.x < 1.0f || pt.x >= static_cast<float>(frame.image.cols - 1) ||
        pt.y < 1.0f || pt.y >= static_cast<float>(frame.image.rows - 1)) {
      continue;
    }

    tracked_points.push_back(pt);
    tracked_prev_points.push_back(prev_points_[i]);
    tracked_ids.push_back(prev_track_ids_[i]);
    tracked_lengths.push_back(prev_track_lengths_[i] + 1);
  }

  // 4. Geometric Outlier Rejection via Fundamental Matrix RANSAC
  std::vector<uchar> inlier_mask(tracked_points.size(), 1);
  if (config_.enable_ransac_outlier_filter && tracked_points.size() >= 8) {
    cv::findFundamentalMat(tracked_prev_points, tracked_points, cv::FM_RANSAC,
                           config_.ransac_threshold_px, 0.99, inlier_mask);
  }

  std::vector<cv::Point2f> final_points;
  std::vector<uint64_t> final_ids;
  std::vector<uint32_t> final_lengths;

  for (size_t i = 0; i < tracked_points.size(); ++i) {
    if (!inlier_mask[i]) continue;

    final_points.push_back(tracked_points[i]);
    final_ids.push_back(tracked_ids[i]);
    final_lengths.push_back(tracked_lengths[i]);

    FeaturePoint fp;
    fp.track_id = tracked_ids[i];
    fp.pixel = Vec2d(tracked_points[i].x, tracked_points[i].y);
    fp.norm_point = unprojectPixel(tracked_points[i]);
    const cv::Point2f vel = (tracked_points[i] - tracked_prev_points[i]) * static_cast<float>(1.0 / dt);
    fp.velocity = Vec2d(vel.x, vel.y);
    fp.track_length = tracked_lengths[i];
    fp.is_inlier = true;
    result.push_back(fp);
  }

  // 5. Replenish lost features
  detectNewFeatures(frame.image, final_points, final_ids, final_lengths);

  // Update state for next frame
  prev_image_ = frame.image.clone();
  prev_points_ = std::move(final_points);
  prev_track_ids_ = std::move(final_ids);
  prev_track_lengths_ = std::move(final_lengths);
  prev_timestamp_ns_ = frame.timestamp_ns;

#endif
  return result;
}

}  // namespace neurovio
