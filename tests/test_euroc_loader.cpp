#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "neurovio/sensors/euroc_loader.hpp"

using namespace neurovio;

TEST(EuRoCLoaderTest, ParseSyntheticData) {
  const auto temp_dir = std::filesystem::temp_directory_path() / "neurovio_euroc_test";
  std::filesystem::create_directories(temp_dir / "mav0" / "imu0");
  std::filesystem::create_directories(temp_dir / "mav0" / "cam0" / "data");

  // Create fake IMU CSV
  std::ofstream imu_file(temp_dir / "mav0" / "imu0" / "data.csv");
  imu_file << "#timestamp [ns],w_RS_x [rad s^-1],w_RS_y [rad s^-1],w_RS_z [rad s^-1],a_RS_x [m s^-2],a_RS_y [m s^-2],a_RS_z [m s^-2]\n";
  imu_file << "1403636579758555392,0.01,-0.02,0.03,0.1,0.2,9.81\n";
  imu_file << "1403636579763555328,0.01,-0.02,0.03,0.1,0.2,9.81\n";
  imu_file.close();

  // Create fake Cam0 CSV
  std::ofstream cam_file(temp_dir / "mav0" / "cam0" / "data.csv");
  cam_file << "#timestamp [ns],filename\n";
  cam_file << "1403636579763555328,1403636579763555328.png\n";
  cam_file.close();

  EuRoCLoader::Config config;
  config.dataset_path = temp_dir;
  config.load_stereo = false;
  config.load_ground_truth = false;

  EuRoCLoader loader(config);
  EXPECT_TRUE(loader.load());
  EXPECT_EQ(loader.getImuData().size(), 2);
  EXPECT_EQ(loader.getLeftFrames().size(), 1);

  std::filesystem::remove_all(temp_dir);
}
