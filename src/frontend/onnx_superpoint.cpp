#include "neurovio/frontend/onnx_superpoint.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>

#if defined(NEUROVIO_HAS_ONNX)
#include <onnxruntime_cxx_api.h>
#endif

#if defined(NEUROVIO_HAS_OPENCV)
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace neurovio {

struct OnnxSuperPoint::Impl {
#if defined(NEUROVIO_HAS_ONNX)
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "NeuroVIO_SuperPoint"};
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;
  bool loaded{false};

  void init(const std::string& model_path, bool use_cuda) {
    if (!std::filesystem::exists(model_path)) {
      spdlog::warn("OnnxSuperPoint: Model file not found at {}", model_path);
      return;
    }

    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_cuda) {
#if defined(ORT_CUDA_AVAILABLE)
      OrtCUDAProviderOptions cuda_options;
      session_options.AppendExecutionProvider_CUDA(cuda_options);
      spdlog::info("OnnxSuperPoint: CUDA Execution Provider enabled");
#else
      spdlog::info("OnnxSuperPoint: Running on CPU Provider (CUDA not compiled)");
#endif
    }

    try {
      session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
      loaded = true;
      spdlog::info("OnnxSuperPoint: Model loaded successfully from {}", model_path);
    } catch (const std::exception& e) {
      spdlog::error("OnnxSuperPoint: Failed to initialize session: {}", e.what());
    }
  }
#else
  bool loaded{false};
  void init([[maybe_unused]] const std::string& path, [[maybe_unused]] bool cuda) {
    spdlog::warn("OnnxSuperPoint: Built without NEUROVIO_HAS_ONNX support");
  }
#endif
};

OnnxSuperPoint::OnnxSuperPoint(Config config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>()) {
  impl_->init(config_.model_path, config_.use_cuda);
}

OnnxSuperPoint::~OnnxSuperPoint() = default;

bool OnnxSuperPoint::isLoaded() const noexcept {
  return impl_ && impl_->loaded;
}

#if defined(NEUROVIO_HAS_OPENCV)
NeuralFeatures OnnxSuperPoint::extract(const cv::Mat& image, const CameraCalibration* calib) {
  NeuralFeatures features;
  if (image.empty()) return features;

#if defined(NEUROVIO_HAS_ONNX)
  if (!isLoaded()) {
    spdlog::warn("OnnxSuperPoint: Model not loaded, returning empty features");
    return features;
  }

  // Preprocessing: Convert image to grayscale float [0, 1] with shape [1, 1, H, W]
  cv::Mat gray_float;
  if (image.channels() == 3) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray_float, CV_32FC1, 1.0 / 255.0);
  } else {
    image.convertTo(gray_float, CV_32FC1, 1.0 / 255.0);
  }

  const int64_t H = gray_float.rows;
  const int64_t W = gray_float.cols;
  std::vector<int64_t> input_shape = {1, 1, H, W};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, reinterpret_cast<float*>(gray_float.data), H * W,
      input_shape.data(), input_shape.size());

  const char* input_names[] = {"image"};
  const char* output_names[] = {"keypoints", "descriptors", "scores"};

  try {
    auto output_tensors = impl_->session->Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 3);

    // 1. Extract Keypoints [1, N, 2]
    float* kpts_ptr = output_tensors[0].GetTensorMutableData<float>();
    auto kpts_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    const size_t num_kpts = kpts_info.GetShape()[1];

    // 2. Extract Descriptors [1, N, 256]
    float* desc_ptr = output_tensors[1].GetTensorMutableData<float>();
    auto desc_info = output_tensors[1].GetTensorTypeAndShapeInfo();
    const size_t desc_dim = desc_info.GetShape()[2];

    // 3. Extract Scores [1, N]
    float* scores_ptr = output_tensors[2].GetTensorMutableData<float>();

    features.keypoints.reserve(num_kpts);
    features.descriptors.reserve(num_kpts);
    features.scores.reserve(num_kpts);

    std::vector<cv::Point2f> cv_pts;
    cv_pts.reserve(num_kpts);

    for (size_t i = 0; i < num_kpts; ++i) {
      const float score = scores_ptr[i];
      if (score < config_.keypoint_threshold) continue;

      const float x = kpts_ptr[i * 2 + 0];
      const float y = kpts_ptr[i * 2 + 1];

      features.keypoints.emplace_back(x, y);
      features.scores.push_back(score);
      cv_pts.emplace_back(x, y);

      std::vector<float> desc(desc_dim);
      std::copy(desc_ptr + i * desc_dim, desc_ptr + (i + 1) * desc_dim, desc.begin());
      features.descriptors.push_back(std::move(desc));
    }

    // Unproject to normalized camera coordinates if calibration provided
    if (calib && !cv_pts.empty()) {
      std::vector<cv::Point2f> norm_pts;
      cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
      K.at<double>(0, 0) = calib->fx;
      K.at<double>(1, 1) = calib->fy;
      K.at<double>(0, 2) = calib->cx;
      K.at<double>(1, 2) = calib->cy;
      cv::Mat D = cv::Mat(calib->distortion).clone();
      cv::undistortPoints(cv_pts, norm_pts, K, D);

      features.norm_keypoints.reserve(norm_pts.size());
      for (const auto& pt : norm_pts) {
        features.norm_keypoints.emplace_back(pt.x, pt.y);
      }
    }
  } catch (const std::exception& e) {
    spdlog::error("OnnxSuperPoint: Inference exception: {}", e.what());
  }
#endif

  return features;
}
#endif

}  // namespace neurovio
