#include <gtest/gtest.h>
#include <filesystem>
#include "neurovio/frontend/onnx_lightglue.hpp"
#include "neurovio/frontend/onnx_superpoint.hpp"

#if defined(NEUROVIO_HAS_OPENCV)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

using namespace neurovio;

namespace {
std::string resolveModelPath(const std::string& filename) {
  if (std::filesystem::exists("models/" + filename)) {
    return "models/" + filename;
  }
  if (std::filesystem::exists("../models/" + filename)) {
    return "../models/" + filename;
  }
  if (std::filesystem::exists("../../models/" + filename)) {
    return "../../models/" + filename;
  }
  return filename;
}
}  // namespace

TEST(OnnxFeaturesTest, SuperPointExtraction) {
  OnnxSuperPoint::Config sp_config;
  sp_config.model_path = resolveModelPath("superpoint.onnx");
  sp_config.use_cuda = false;

  if (!std::filesystem::exists(sp_config.model_path)) {
    GTEST_SKIP() << "Model file " << sp_config.model_path << " not found, skipping test.";
  }

  OnnxSuperPoint sp(sp_config);
  EXPECT_TRUE(sp.isLoaded());

#if defined(NEUROVIO_HAS_OPENCV)
  // Create a synthetic image with high contrast checkerboard pattern
  cv::Mat image = cv::Mat::zeros(480, 752, CV_8UC1);
  for (int r = 50; r < 400; r += 50) {
    for (int c = 50; c < 700; c += 50) {
      cv::rectangle(image, cv::Rect(c, r, 25, 25), cv::Scalar(255), -1);
    }
  }

  const auto feats = sp.extract(image);
  EXPECT_GT(feats.keypoints.size(), 0);
  EXPECT_EQ(feats.keypoints.size(), feats.descriptors.size());
  if (!feats.descriptors.empty()) {
    EXPECT_EQ(feats.descriptors.front().size(), 256);
  }
#endif
}

TEST(OnnxFeaturesTest, LightGlueMatching) {
  OnnxLightGlue::Config lg_config;
  lg_config.model_path = resolveModelPath("lightglue_superpoint.onnx");
  lg_config.use_cuda = false;

  if (!std::filesystem::exists(lg_config.model_path)) {
    GTEST_SKIP() << "Model file " << lg_config.model_path << " not found, skipping test.";
  }

  OnnxLightGlue lg(lg_config);
  EXPECT_TRUE(lg.isLoaded());

  // Create synthetic keypoints and descriptors
  NeuralFeatures feats0;
  NeuralFeatures feats1;

  constexpr size_t N = 100;
  for (size_t i = 0; i < N; ++i) {
    const double x = static_cast<double>(i * 5 + 50);
    const double y = static_cast<double>(i * 3 + 50);
    feats0.keypoints.emplace_back(x, y);
    feats1.keypoints.emplace_back(x + 1.0, y + 1.0);  // slight shift

    std::vector<float> desc(256, 0.0f);
    desc[i % 256] = 1.0f;  // distinct one-hot-like pattern
    feats0.descriptors.push_back(desc);
    feats1.descriptors.push_back(desc);
  }

  const auto matches = lg.match(feats0, feats1);
  EXPECT_GT(matches.size(), 0);
}
