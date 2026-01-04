#include "mityaeva_d_contrast_enhancement_histogram_stretching/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

namespace {

std::pair<uint8_t, uint8_t> FindGlobalMinMax(const std::vector<uint8_t> &local_pixels) {
  unsigned char local_min = 255;
  unsigned char local_max = 0;

  for (uint8_t p : local_pixels) {
    auto v = static_cast<unsigned char>(p);
    local_min = std::min(local_min, v);
    local_max = std::max(local_max, v);
  }

  unsigned char global_min = 0;
  unsigned char global_max = 0;

  MPI_Allreduce(&local_min, &global_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);

  return {static_cast<uint8_t>(global_min), static_cast<uint8_t>(global_max)};
}

std::vector<uint8_t> ProcessLocalPixels(const std::vector<uint8_t> &local_pixels, uint8_t global_min,
                                        uint8_t global_max) {
  if (global_min == global_max) {
    return local_pixels;
  }

  const double scale = 255.0 / static_cast<double>(global_max - global_min);
  std::vector<uint8_t> result;
  result.reserve(local_pixels.size());

  for (uint8_t p : local_pixels) {
    double v = static_cast<double>(p - global_min) * scale;
    int r = static_cast<int>(std::round(v));
    r = std::max(0, std::min(255, r));
    result.push_back(static_cast<uint8_t>(r));
  }

  return result;
}

void BuildCountsDispls(int total_pixels, int size, std::vector<int> &counts, std::vector<int> &displs) {
  counts.assign(size, 0);
  displs.assign(size, 0);

  int base = total_pixels / size;
  int rem = total_pixels % size;
  int offset = 0;

  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }
}

int LocalCount(int total_pixels, int rank, int size) {
  int base = total_pixels / size;
  int rem = total_pixels % size;
  return base + (rank < rem ? 1 : 0);
}

}  // namespace

ContrastEnhancementMPI::ContrastEnhancementMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<uint8_t>{};
}

bool ContrastEnhancementMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (input.size() < 3) {
    return false;
  }

  width_ = static_cast<int>(input[0]);
  height_ = static_cast<int>(input[1]);

  if (width_ <= 0 || height_ <= 0) {
    return false;
  }

  total_pixels_ = width_ * height_;
  return input.size() == static_cast<size_t>(total_pixels_) + 2;
}

bool ContrastEnhancementMPI::PreProcessingImpl() {
  return true;
}

bool ContrastEnhancementMPI::RunImpl() {
  try {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
      const auto &input = GetInput();
      width_ = static_cast<int>(input[0]);
      height_ = static_cast<int>(input[1]);
      total_pixels_ = width_ * height_;
    }

    MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_pixels_, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int my_pixels = LocalCount(total_pixels_, rank, size);

    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
      BuildCountsDispls(total_pixels_, size, counts, displs);
    }

    std::vector<uint8_t> local_pixels(static_cast<size_t>(my_pixels));

    const unsigned char *sendbuf = nullptr;
    if (rank == 0) {
      sendbuf = reinterpret_cast<const unsigned char *>(GetInput().data() + 2);
    }

    MPI_Scatterv(sendbuf, (rank == 0 ? counts.data() : nullptr), (rank == 0 ? displs.data() : nullptr),
                 MPI_UNSIGNED_CHAR, reinterpret_cast<unsigned char *>(local_pixels.data()), my_pixels,
                 MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    auto [global_min, global_max] = FindGlobalMinMax(local_pixels);
    std::vector<uint8_t> local_result = ProcessLocalPixels(local_pixels, global_min, global_max);

    std::vector<uint8_t> final_output(static_cast<size_t>(total_pixels_) + 2);
    if (rank == 0) {
      final_output[0] = static_cast<uint8_t>(width_);
      final_output[1] = static_cast<uint8_t>(height_);
    }

    MPI_Gatherv(reinterpret_cast<const unsigned char *>(local_result.data()), my_pixels, MPI_UNSIGNED_CHAR,
                (rank == 0 ? reinterpret_cast<unsigned char *>(final_output.data() + 2) : nullptr),
                (rank == 0 ? counts.data() : nullptr), (rank == 0 ? displs.data() : nullptr), MPI_UNSIGNED_CHAR, 0,
                MPI_COMM_WORLD);

    int final_size = 0;
    if (rank == 0) {
      final_size = static_cast<int>(final_output.size());
    }

    MPI_Bcast(&final_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
      final_output.resize(final_size);
    }

    MPI_Bcast(reinterpret_cast<unsigned char *>(final_output.data()), final_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    GetOutput() = std::move(final_output);

    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  } catch (...) {
    return false;
  }
}

bool ContrastEnhancementMPI::PostProcessingImpl() {
  const auto &output = GetOutput();
  if (output.size() < 2) {
    return false;
  }

  int out_w = static_cast<int>(output[0]);
  int out_h = static_cast<int>(output[1]);

  if (out_w != width_ || out_h != height_) {
    return false;
  }
  if (output.size() != static_cast<size_t>(total_pixels_) + 2) {
    return false;
  }

  return true;
}

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
