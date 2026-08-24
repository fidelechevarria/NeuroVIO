#!/usr/bin/env bash
# NeuroVIO: EuRoC MAV Dataset Downloader Script
set -euo pipefail

TARGET_DIR="${1:-./datasets/euroc}"
mkdir -p "${TARGET_DIR}"

BASE_URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"

SEQUENCES=(
  "machine_hall/MH_01_easy/MH_01_easy.zip"
  "machine_hall/MH_02_easy/MH_02_easy.zip"
  "machine_hall/MH_03_medium/MH_03_medium.zip"
  "machine_hall/MH_04_difficult/MH_04_difficult.zip"
  "machine_hall/MH_05_difficult/MH_05_difficult.zip"
  "vicon_room1/V1_01_easy/V1_01_easy.zip"
  "vicon_room1/V1_02_medium/V1_02_medium.zip"
  "vicon_room1/V1_03_difficult/V1_03_difficult.zip"
  "vicon_room2/V2_01_easy/V2_01_easy.zip"
  "vicon_room2/V2_02_medium/V2_02_medium.zip"
  "vicon_room2/V2_03_difficult/V2_03_difficult.zip"
)

echo "======================================================="
echo " NeuroVIO: Downloading EuRoC MAV Datasets to ${TARGET_DIR}"
echo "======================================================="

for SEQ in "${SEQUENCES[@]}"; do
  DIR_NAME=$(dirname "${SEQ}")
  ZIP_NAME=$(basename "${SEQ}")
  FULL_DIR="${TARGET_DIR}/${DIR_NAME}"
  mkdir -p "${FULL_DIR}"

  if [ ! -f "${FULL_DIR}/${ZIP_NAME}" ]; then
    echo "Downloading ${SEQ}..."
    curl -L "${BASE_URL}/${SEQ}" -o "${FULL_DIR}/${ZIP_NAME}"
  else
    echo "Found cached ${ZIP_NAME} in ${FULL_DIR}"
  fi

  if [ ! -d "${FULL_DIR}/mav0" ]; then
    echo "Extracting ${ZIP_NAME}..."
    unzip -q -o "${FULL_DIR}/${ZIP_NAME}" -d "${FULL_DIR}"
  fi
done

echo "EuRoC dataset download and extraction complete!"
