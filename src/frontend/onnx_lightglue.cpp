#include "neurovio/frontend/onnx_lightglue.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>

#if defined(NEUROVIO_HAS_ONNX)
#include <onnxruntime_cxx_api.h>
#endif

namespace neurovio {

struct OnnxLightGlue::Impl {
#if defined(NEUROVIO_HAS_ONNX)
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "NeuroVIO_LightGlue"};
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;
  bool loaded{false};

  void init(const std::string& model_path, bool use_cuda) {
    if (!std::filesystem::exists(model_path)) {
      spdlog::warn("OnnxLightGlue: Model file not found at {}", model_path);
      return;
    }

    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_cuda) {
#if defined(ORT_CUDA_AVAILABLE)
      OrtCUDAProviderOptions cuda_options;
      session_options.AppendExecutionProvider_CUDA(cuda_options);
      spdlog::info("OnnxLightGlue: CUDA Execution Provider enabled");
#else
      spdlog::info("OnnxLightGlue: Running on CPU Provider (CUDA not compiled)");
#endif
    }

    try {
      session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
      loaded = true;
      spdlog::info("OnnxLightGlue: Model loaded successfully from {}", model_path);
    } catch (const std::exception& e) {
      spdlog::error("OnnxLightGlue: Failed to initialize session: {}", e.what());
    }
  }
#else
  bool loaded{false};
  void init([[maybe_unused]] const std::string& path, [[maybe_unused]] bool cuda) {
    spdlog::warn("OnnxLightGlue: Built without NEUROVIO_HAS_ONNX support");
  }
#endif
};

OnnxLightGlue::OnnxLightGlue(Config config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>()) {
  impl_->init(config_.model_path, config_.use_cuda);
}

OnnxLightGlue::~OnnxLightGlue() = default;

bool OnnxLightGlue::isLoaded() const noexcept {
  return impl_ && impl_->loaded;
}

std::vector<NeuralMatch> OnnxLightGlue::match(const NeuralFeatures& feats0,
                                             const NeuralFeatures& feats1) {
  std::vector<NeuralMatch> matches;

  const size_t N0 = feats0.keypoints.size();
  const size_t N1 = feats1.keypoints.size();
  if (N0 == 0 || N1 == 0) return matches;

#if defined(NEUROVIO_HAS_ONNX)
  if (!isLoaded()) {
    spdlog::warn("OnnxLightGlue: Model not loaded, returning empty matches");
    return matches;
  }

  const size_t D0 = feats0.descriptors.front().size();
  const size_t D1 = feats1.descriptors.front().size();
  if (D0 != D1) {
    spdlog::error("OnnxLightGlue: Descriptor dimension mismatch ({} vs {})", D0, D1);
    return matches;
  }

  // 1. Flatten kpts0, desc0
  std::vector<float> kpts0_buf(N0 * 2);
  std::vector<float> desc0_buf(N0 * D0);
  for (size_t i = 0; i < N0; ++i) {
    kpts0_buf[i * 2 + 0] = static_cast<float>(feats0.keypoints[i].x());
    kpts0_buf[i * 2 + 1] = static_cast<float>(feats0.keypoints[i].y());
    std::copy(feats0.descriptors[i].begin(), feats0.descriptors[i].end(), desc0_buf.begin() + i * D0);
  }

  // 2. Flatten kpts1, desc1
  std::vector<float> kpts1_buf(N1 * 2);
  std::vector<float> desc1_buf(N1 * D1);
  for (size_t i = 0; i < N1; ++i) {
    kpts1_buf[i * 2 + 0] = static_cast<float>(feats1.keypoints[i].x());
    kpts1_buf[i * 2 + 1] = static_cast<float>(feats1.keypoints[i].y());
    std::copy(feats1.descriptors[i].begin(), feats1.descriptors[i].end(), desc1_buf.begin() + i * D1);
  }

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int64_t> kpts0_shape = {1, static_cast<int64_t>(N0), 2};
  std::vector<int64_t> desc0_shape = {1, static_cast<int64_t>(N0), static_cast<int64_t>(D0)};
  std::vector<int64_t> kpts1_shape = {1, static_cast<int64_t>(N1), 2};
  std::vector<int64_t> desc1_shape = {1, static_cast<int64_t>(N1), static_cast<int64_t>(D1)};

  Ort::Value tensors[4] = {
      Ort::Value::CreateTensor<float>(memory_info, kpts0_buf.data(), kpts0_buf.size(),
                                      kpts0_shape.data(), kpts0_shape.size()),
      Ort::Value::CreateTensor<float>(memory_info, desc0_buf.data(), desc0_buf.size(),
                                      desc0_shape.data(), desc0_shape.size()),
      Ort::Value::CreateTensor<float>(memory_info, kpts1_buf.data(), kpts1_buf.size(),
                                      kpts1_shape.data(), kpts1_shape.size()),
      Ort::Value::CreateTensor<float>(memory_info, desc1_buf.data(), desc1_buf.size(),
                                      desc1_shape.data(), desc1_shape.size()),
  };

  const char* input_names[] = {"kpts0", "desc0", "kpts1", "desc1"};
  const char* output_names[] = {"matches0", "mscores0"};

  try {
    auto output_tensors = impl_->session->Run(
        Ort::RunOptions{nullptr}, input_names, tensors, 4, output_names, 2);

    int64_t* matches_ptr = output_tensors[0].GetTensorMutableData<int64_t>();
    float* scores_ptr = output_tensors[1].GetTensorMutableData<float>();

    matches.reserve(N0);
    for (size_t i = 0; i < N0; ++i) {
      const int64_t match_idx1 = matches_ptr[i];
      const float conf = scores_ptr[i];
      if (match_idx1 >= 0 && match_idx1 < static_cast<int64_t>(N1) && conf >= config_.match_threshold) {
        NeuralMatch m;
        m.index0 = static_cast<int>(i);
        m.index1 = static_cast<int>(match_idx1);
        m.confidence = conf;
        matches.push_back(m);
      }
    }
  } catch (const std::exception& e) {
    spdlog::error("OnnxLightGlue: Inference exception: {}", e.what());
  }
#endif

  return matches;
}

}  // namespace neurovio
