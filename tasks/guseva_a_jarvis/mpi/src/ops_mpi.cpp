#include "guseva_a_jarvis/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "guseva_a_jarvis/common/include/common.hpp"

namespace guseva_a_jarvis {

GusevaAJarvisMPI::GusevaAJarvisMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

bool GusevaAJarvisMPI::ValidationImpl() {
  const auto &[width, height, image] = GetInput();

  if (rank_ == 0) {
    const bool is_size_match = static_cast<int>(image.size()) == width * height;
    bool is_image_binary = true;
    for (const int pixel_value : image) {
      if (pixel_value != 0 && pixel_value != 1) {
        is_image_binary = false;
        break;
      }
    }
    const bool is_size_possible = width > 0 && height > 0;
    return is_image_binary && is_size_possible && is_size_match;
  }

  return true;
}

bool GusevaAJarvisMPI::PreProcessingImpl() {
  const auto &input_tuple = GetInput();
  const int width = std::get<0>(input_tuple);
  const int height = std::get<1>(input_tuple);
  const std::vector<int> &image = std::get<2>(input_tuple);

  points_.clear();
  int rows_per_process = height / size_;
  int remainder = height % size_;

  int start_row = (rank_ * rows_per_process) + std::min(rank_, remainder);
  int end_row = start_row + rows_per_process + (rank_ < remainder ? 1 : 0);

  for (int yy = start_row; yy < end_row; ++yy) {
    for (int xx = 0; xx < width; ++xx) {
      if (image[(yy * width) + xx] == 1) {
        points_.emplace_back(xx, yy);
      }
    }
  }

  return true;
}

bool GusevaAJarvisMPI::RunImpl() {
  std::vector<std::pair<int, int>> local_hull = BuildConvexHull(points_);

  int local_hull_size = static_cast<int>(local_hull.size());
  std::vector<int> all_hull_sizes(size_, 0);

  MPI_Gather(&local_hull_size, 1, MPI_INT, all_hull_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displacements(size_, 0);
  std::vector<int> recv_counts(size_, 0);
  int64_t total_points = 0;

  if (rank_ == 0) {
    for (int i = 0; i < size_; ++i) {
      recv_counts[i] = all_hull_sizes[i] * 2;
      displacements[i] = static_cast<int>(total_points) * 2;
      total_points += all_hull_sizes[i];
    }
  }

  std::vector<int> local_data(static_cast<int64_t>(local_hull_size) * 2);
  for (int64_t i = 0; i < local_hull_size; ++i) {
    local_data[i * 2] = local_hull[i].first;
    local_data[(i * 2) + 1] = local_hull[i].second;
  }

  std::vector<int> all_data;
  if (rank_ == 0) {
    all_data.resize(total_points * 2, 0);
  }

  MPI_Gatherv(local_data.data(), local_hull_size * 2, MPI_INT, all_data.data(), recv_counts.data(),
              displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    std::vector<std::pair<int, int>> all_points;
    all_points.reserve(total_points);
    for (int64_t i = 0; i < total_points; ++i) {
      all_points.emplace_back(all_data[i * 2], all_data[(i * 2) + 1]);
    }

    hull_ = BuildConvexHull(all_points);
  }

  return true;
}

bool GusevaAJarvisMPI::PostProcessingImpl() {
  const auto &input_tuple = GetInput();
  const int width = std::get<0>(input_tuple);
  const int height = std::get<1>(input_tuple);

  int hull_size = 0;
  if (rank_ == 0) {
    hull_size = static_cast<int>(hull_.size());
  }

  MPI_Bcast(&hull_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> hull_data;
  if (rank_ == 0) {
    hull_data.resize(static_cast<int64_t>(hull_size) * 2, 0);
    for (int64_t i = 0; i < hull_size; ++i) {
      hull_data[i * 2] = hull_[i].first;
      hull_data[(i * 2) + 1] = hull_[i].second;
    }
  } else {
    hull_data.resize(static_cast<int64_t>(hull_size) * 2, 0);
  }

  MPI_Bcast(hull_data.data(), hull_size * 2, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    hull_.clear();
    hull_.reserve(hull_size);
    for (int64_t i = 0; i < hull_size; ++i) {
      hull_.emplace_back(hull_data[i * 2], hull_data[(i * 2) + 1]);
    }
  }

  std::vector<int> output_vector(static_cast<int64_t>(width) * height, 0);

  for (const auto &point : hull_) {
    const int x = point.first;
    const int y = point.second;
    if (x >= 0 && x < width && y >= 0 && y < height) {
      output_vector[(y * width) + x] = 1;
    }
  }

  GetOutput() = output_vector;

  return true;
}

}  // namespace guseva_a_jarvis
