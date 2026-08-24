#include "neurovio/sensors/euroc_loader.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <spdlog/spdlog.h>

#if defined(NEUROVIO_HAS_OPENCV)
#include <opencv2/imgcodecs.hpp>
#endif

namespace neurovio {

namespace {

std::filesystem::path resolveMav0Path(const std::filesystem::path& base_path) {
  if (std::filesystem::exists(base_path / "mav0")) {
    return base_path / "mav0";
  }
  if (std::filesystem::exists(base_path / "imu0")) {
    return base_path;
  }
  return base_path;
}

}  // namespace

EuRoCLoader::EuRoCLoader(Config config) : config_(std::move(config)) {}

bool EuRoCLoader::load() {
  const auto root = resolveMav0Path(config_.dataset_path);
  spdlog::info("EuRoCLoader: Loading dataset from {}", root.string());

  if (!std::filesystem::exists(root)) {
    spdlog::error("EuRoCLoader: Path {} does not exist!", root.string());
    return false;
  }

  // 1. Parse IMU data
  const auto imu_csv = root / "imu0" / "data.csv";
  if (!parseImuCsv(imu_csv)) {
    spdlog::error("EuRoCLoader: Failed to parse IMU CSV at {}", imu_csv.string());
    return false;
  }

  // 2. Parse Left Camera (cam0)
  const auto cam0_csv = root / "cam0" / "data.csv";
  const auto cam0_dir = root / "cam0" / "data";
  if (!parseCameraCsv(cam0_csv, cam0_dir, 0, left_frames_)) {
    spdlog::error("EuRoCLoader: Failed to parse Cam0 CSV at {}", cam0_csv.string());
    return false;
  }

  // 3. Parse Right Camera (cam1) if requested
  if (config_.load_stereo) {
    const auto cam1_csv = root / "cam1" / "data.csv";
    const auto cam1_dir = root / "cam1" / "data";
    parseCameraCsv(cam1_csv, cam1_dir, 1, right_frames_);
  }

  // 4. Parse Ground Truth if present
  if (config_.load_ground_truth) {
    const auto gt_csv = root / "state_groundtruth_estimate0" / "data.csv";
    if (std::filesystem::exists(gt_csv)) {
      parseGroundTruthCsv(gt_csv);
    }
  }

  spdlog::info("EuRoCLoader: Loaded {} IMU samples, {} left frames, {} right frames, {} GT states",
               imu_measurements_.size(), left_frames_.size(), right_frames_.size(),
               ground_truth_states_.size());
  return true;
}

bool EuRoCLoader::parseImuCsv(const std::filesystem::path& csv_path) {
  std::ifstream file(csv_path);
  if (!file.is_open()) return false;

  std::string line;
  imu_measurements_.clear();

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::stringstream ss(line);
    std::string item;
    std::vector<std::string> tokens;

    while (std::getline(ss, item, ',')) {
      tokens.push_back(item);
    }

    if (tokens.size() < 7) continue;

    try {
      const TimestampNs t = std::stoll(tokens[0]);
      const double gx = std::stod(tokens[1]);
      const double gy = std::stod(tokens[2]);
      const double gz = std::stod(tokens[3]);
      const double ax = std::stod(tokens[4]);
      const double ay = std::stod(tokens[5]);
      const double az = std::stod(tokens[6]);

      imu_measurements_.emplace_back(t, Vec3d(ax, ay, az), Vec3d(gx, gy, gz));
    } catch (const std::exception& e) {
      spdlog::warn("EuRoCLoader: Failed to parse IMU line: {}", line);
    }
  }

  std::sort(imu_measurements_.begin(), imu_measurements_.end(),
            [](const ImuMeasurement& a, const ImuMeasurement& b) {
              return a.timestamp_ns < b.timestamp_ns;
            });
  return !imu_measurements_.empty();
}

bool EuRoCLoader::parseCameraCsv(const std::filesystem::path& csv_path,
                                 const std::filesystem::path& img_dir,
                                 uint32_t cam_id,
                                 std::vector<CameraFrame>& out_frames) {
  std::ifstream file(csv_path);
  if (!file.is_open()) return false;

  std::string line;
  out_frames.clear();
  uint64_t frame_idx = 0;

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::stringstream ss(line);
    std::string token_t;
    std::string filename;

    if (std::getline(ss, token_t, ',') && std::getline(ss, filename)) {
      // Remove trailing CR/LF or whitespace
      filename.erase(std::remove_if(filename.begin(), filename.end(), [](unsigned char c) {
        return std::isspace(c);
      }), filename.end());

      try {
        const TimestampNs t = std::stoll(token_t);
        CameraFrame frame(t, cam_id, (img_dir / filename).string());
        frame.frame_id = frame_idx++;
        out_frames.push_back(std::move(frame));
      } catch (const std::exception& e) {
        spdlog::warn("EuRoCLoader: Failed to parse camera line: {}", line);
      }
    }
  }

  std::sort(out_frames.begin(), out_frames.end(),
            [](const CameraFrame& a, const CameraFrame& b) {
              return a.timestamp_ns < b.timestamp_ns;
            });
  return !out_frames.empty();
}

bool EuRoCLoader::parseGroundTruthCsv(const std::filesystem::path& csv_path) {
  std::ifstream file(csv_path);
  if (!file.is_open()) return false;

  std::string line;
  ground_truth_states_.clear();

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::stringstream ss(line);
    std::string item;
    std::vector<std::string> tokens;

    while (std::getline(ss, item, ',')) {
      tokens.push_back(item);
    }

    if (tokens.size() < 17) continue;

    try {
      const TimestampNs t = std::stoll(tokens[0]);
      const Vec3d p(std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3]));
      const Quatd q(std::stod(tokens[4]), std::stod(tokens[5]), std::stod(tokens[6]),
                    std::stod(tokens[7]));  // qw, qx, qy, qz
      const Vec3d v(std::stod(tokens[8]), std::stod(tokens[9]), std::stod(tokens[10]));
      const Vec3d bg(std::stod(tokens[11]), std::stod(tokens[12]), std::stod(tokens[13]));
      const Vec3d ba(std::stod(tokens[14]), std::stod(tokens[15]), std::stod(tokens[16]));

      ground_truth_states_.emplace_back(t, q.toRotationMatrix(), p, v, bg, ba);
    } catch (const std::exception& e) {
      spdlog::warn("EuRoCLoader: Failed to parse GT line: {}", line);
    }
  }
  return !ground_truth_states_.empty();
}

void EuRoCLoader::play(const ImuCallback& on_imu, const FrameCallback& on_frame) const {
  size_t imu_idx = 0;
  size_t left_idx = 0;
  size_t right_idx = 0;

  while (imu_idx < imu_measurements_.size() || left_idx < left_frames_.size()) {
    const TimestampNs t_imu = (imu_idx < imu_measurements_.size())
                                  ? imu_measurements_[imu_idx].timestamp_ns
                                  : std::numeric_limits<TimestampNs>::max();
    const TimestampNs t_left = (left_idx < left_frames_.size())
                                   ? left_frames_[left_idx].timestamp_ns
                                   : std::numeric_limits<TimestampNs>::max();

    if (t_imu <= t_left) {
      if (on_imu) {
        on_imu(imu_measurements_[imu_idx]);
      }
      ++imu_idx;
    } else {
      if (on_frame) {
        StereoFrame stereo;
        stereo.timestamp_ns = t_left;
        stereo.left = left_frames_[left_idx];

#if defined(NEUROVIO_HAS_OPENCV)
        stereo.left.image = cv::imread(stereo.left.file_path, cv::IMREAD_GRAYSCALE);
#endif

        if (config_.load_stereo && right_idx < right_frames_.size()) {
          stereo.right = right_frames_[right_idx];
#if defined(NEUROVIO_HAS_OPENCV)
          stereo.right.image = cv::imread(stereo.right.file_path, cv::IMREAD_GRAYSCALE);
#endif
          stereo.has_right = true;
          ++right_idx;
        }
        on_frame(stereo);
      }
      ++left_idx;
    }
  }
}

}  // namespace neurovio
