#include <gtest/gtest.h>
#include "neurovio/frontend/neurotrack_frontend.hpp"

TEST(NeuroVIOFrontend, NeuroTrackInitializationAndTracking) {
  neurovio::CameraCalibration calib;
  calib.width = 752;
  calib.height = 480;
  calib.fx = 458.654;
  calib.fy = 457.296;
  calib.cx = 367.215;
  calib.cy = 248.375;

  neurovio::NeuroTrackConfig config;
  config.max_features = 200;
  config.cell_size_pixels = 32;

  neurovio::NeuroTrackFrontend frontend(calib, config);

  frontend.reset();

  const auto stats = frontend.stats();
  EXPECT_GE(stats.active_tracks, 0);
}
