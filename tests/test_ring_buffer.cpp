#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>
#include "neurovio/common/ring_buffer.hpp"

using namespace neurovio;

TEST(RingBufferTest, SingleThreadPushPop) {
  SPSCQueue<int, 16> queue;
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.size(), 0);

  for (int i = 0; i < 15; ++i) {
    EXPECT_TRUE(queue.push(i));
  }

  EXPECT_EQ(queue.size(), 15);
  EXPECT_FALSE(queue.empty());

  for (int i = 0; i < 15; ++i) {
    int val = -1;
    EXPECT_TRUE(queue.pop(val));
    EXPECT_EQ(val, i);
  }

  EXPECT_TRUE(queue.empty());
}

TEST(RingBufferTest, MultiThreadedProducerConsumer) {
  constexpr size_t kNumItems = 100000;
  SPSCQueue<size_t, 1024> queue;
  std::atomic<bool> producer_done{false};

  std::thread producer([&]() {
    for (size_t i = 0; i < kNumItems; ++i) {
      while (!queue.push(i)) {
        std::this_thread::yield();
      }
    }
    producer_done.store(true, std::memory_order_release);
  });

  std::vector<size_t> received;
  received.reserve(kNumItems);

  std::thread consumer([&]() {
    while (!producer_done.load(std::memory_order_acquire) || !queue.empty()) {
      size_t val = 0;
      if (queue.pop(val)) {
        received.push_back(val);
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(received.size(), kNumItems);
  for (size_t i = 0; i < kNumItems; ++i) {
    EXPECT_EQ(received[i], i);
  }
}
