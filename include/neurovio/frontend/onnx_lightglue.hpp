#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "neurovio/frontend/onnx_superpoint.hpp"

namespace neurovio {

struct NeuralMatch {
  int index0{-1};
  int index1{-1};
  float confidence{0.0f};
};

class OnnxLightGlue {
 public:
  struct Config {
    std::string model_path{"models/lightglue_superpoint.onnx"};
    bool use_cuda{false};
    float match_threshold{0.2f};
    int num_layers{9};
  };

  explicit OnnxLightGlue(Config config);
  ~OnnxLightGlue();

  [[nodiscard]] bool isLoaded() const noexcept;

  [[nodiscard]] std::vector<NeuralMatch> match(const NeuralFeatures& feats0,
                                               const NeuralFeatures& feats1);

 private:
  Config config_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace neurovio
