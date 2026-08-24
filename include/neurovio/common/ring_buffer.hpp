#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <new>
#include <type_traits>
#include <vector>

namespace neurovio {

#if defined(__cpp_lib_hardware_interference_size)
using std::hardware_destructive_interference_size;
#else
// Default 64 bytes cache line size for x86_64 and ARM64
constexpr size_t hardware_destructive_interference_size = 64;
#endif

template <typename T, size_t Capacity = 4096>
class SPSCQueue {
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
  static_assert(std::is_nothrow_destructible_v<T>, "T must be nothrow destructible");

 public:
  SPSCQueue() {
    // Zero-initialize storage buffer
    for (size_t i = 0; i < Capacity; ++i) {
      slots_[i].initialized = false;
    }
  }

  ~SPSCQueue() {
    T discarded;
    while (pop(discarded)) {
    }
  }

  SPSCQueue(const SPSCQueue&) = delete;
  SPSCQueue& operator=(const SPSCQueue&) = delete;
  SPSCQueue(SPSCQueue&&) = delete;
  SPSCQueue& operator=(SPSCQueue&&) = delete;

  template <typename... Args>
  bool emplace(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) {
    const size_t current_tail = tail_.load(std::memory_order_relaxed);
    const size_t current_head = head_cache_;

    if (current_tail - current_head >= Capacity) {
      head_cache_ = head_.load(std::memory_order_acquire);
      if (current_tail - head_cache_ >= Capacity) {
        return false;  // Queue is full
      }
    }

    const size_t index = current_tail & kIndexMask;
    auto* slot_ptr = reinterpret_cast<T*>(&slots_[index].storage);
    ::new (static_cast<void*>(slot_ptr)) T(std::forward<Args>(args)...);
    slots_[index].initialized = true;

    tail_.store(current_tail + 1, std::memory_order_release);
    return true;
  }

  bool push(const T& item) noexcept(std::is_nothrow_copy_constructible_v<T>) {
    return emplace(item);
  }

  bool push(T&& item) noexcept(std::is_nothrow_move_constructible_v<T>) {
    return emplace(std::move(item));
  }

  bool pop(T& value) noexcept(std::is_nothrow_move_assignable_v<T>) {
    const size_t current_head = head_.load(std::memory_order_relaxed);
    const size_t current_tail = tail_cache_;

    if (current_head == current_tail) {
      tail_cache_ = tail_.load(std::memory_order_acquire);
      if (current_head == tail_cache_) {
        return false;  // Queue is empty
      }
    }

    const size_t index = current_head & kIndexMask;
    auto* slot_ptr = reinterpret_cast<T*>(&slots_[index].storage);
    value = std::move(*slot_ptr);
    slot_ptr->~T();
    slots_[index].initialized = false;

    head_.store(current_head + 1, std::memory_order_release);
    return true;
  }

  [[nodiscard]] size_t size() const noexcept {
    const size_t head = head_.load(std::memory_order_relaxed);
    const size_t tail = tail_.load(std::memory_order_relaxed);
    return (tail >= head) ? (tail - head) : 0;
  }

  [[nodiscard]] bool empty() const noexcept {
    return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] constexpr size_t capacity() const noexcept { return Capacity; }

 private:
  static constexpr size_t kIndexMask = Capacity - 1;

  struct Node {
    alignas(alignof(T)) std::byte storage[sizeof(T)];
    bool initialized{false};
  };

  // Prevent false sharing by putting head and tail on separate cache lines
  alignas(hardware_destructive_interference_size) std::atomic<size_t> tail_{0};
  alignas(hardware_destructive_interference_size) size_t head_cache_{0};

  alignas(hardware_destructive_interference_size) std::atomic<size_t> head_{0};
  alignas(hardware_destructive_interference_size) size_t tail_cache_{0};

  alignas(hardware_destructive_interference_size) Node slots_[Capacity];
};

}  // namespace neurovio
