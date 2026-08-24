#pragma once

#include <chrono>
#include <cstdint>
#include "neurovio/common/types.hpp"

namespace neurovio::time {

constexpr double kNsToSec = 1e-9;
constexpr double kSecToNs = 1e9;

[[nodiscard]] inline constexpr double nsToSeconds(TimestampNs ns) noexcept {
  return static_cast<double>(ns) * kNsToSec;
}

[[nodiscard]] inline constexpr TimestampNs secondsToNs(double sec) noexcept {
  return static_cast<TimestampNs>(sec * kSecToNs);
}

[[nodiscard]] inline TimestampNs nowNs() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

}  // namespace neurovio::time
