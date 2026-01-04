#include "pylaeva_s_convex_hull_bin/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "pylaeva_s_convex_hull_bin/common/include/common.hpp"

namespace pylaeva_s_convex_hull_bin {

namespace {

int Cross(const Point &o, const Point &a, const Point &b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

}  // namespace

PylaevaSConvexHullBinMPI::PylaevaSConvexHullBinMPI(const InType &in) : local_data_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  if (rank_ == 0) {
    full_data_ = in;
  }
}

bool PylaevaSConvexHullBinMPI::ValidationImpl() {
  return GetInput().width > 0 && GetInput().height > 0 && !GetInput().pixels.empty() &&
         GetInput().pixels.size() == static_cast<size_t>(GetInput().width) * static_cast<size_t>(GetInput().height);
}

bool PylaevaSConvexHullBinMPI::PreProcessingImpl() {
  const uint8_t threshold = 128;

  if (rank_ == 0) {
    int total_pixels = full_data_.width * full_data_.height;
    for (int i = 0; i < total_pixels; ++i) {
      if (full_data_.pixels[i] > threshold) {
        full_data_.pixels[i] = 255;
      } else {
        full_data_.pixels[i] = 0;
      }
    }
  }

  ScatterDataAndDistributeWork();

  return true;
}

void PylaevaSConvexHullBinMPI::ScatterDataAndDistributeWork() {
  int width = 0;
  int height = 0;

  if (rank_ == 0) {
    width = full_data_.width;
    height = full_data_.height;
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  local_data_.width = width;
  local_data_.height = height;

  rows_per_proc_ = height / size_;
  int remainder = height % size_;

  start_row_ = (rank_ * rows_per_proc_) + std::min(rank_, remainder);
  end_row_ = start_row_ + rows_per_proc_ + (rank_ < remainder ? 1 : 0);

  std::vector<int> send_counts(size_, 0);
  std::vector<int> displs(size_, 0);

  if (rank_ == 0) {
    for (int i = 0; i < size_; ++i) {
      int proc_start = (i * rows_per_proc_) + std::min(i, remainder);
      int proc_end = proc_start + rows_per_proc_ + (i < remainder ? 1 : 0);
      send_counts[i] = (proc_end - proc_start) * width;
      displs[i] = proc_start * width;
    }
  }

  int local_rows = end_row_ - start_row_;
  size_t local_pixel_count = static_cast<size_t>(local_rows) * static_cast<size_t>(width);
  local_data_.pixels.resize(local_pixel_count);

  MPI_Scatterv(rank_ == 0 ? full_data_.pixels.data() : nullptr, send_counts.data(), displs.data(), MPI_UINT8_T,
               local_data_.pixels.data(), static_cast<int>(local_pixel_count), MPI_UINT8_T, 0, MPI_COMM_WORLD);
}

bool PylaevaSConvexHullBinMPI::RunImpl() {
  FindConnectedComponentsMpi();
  ProcessComponentsAndComputeHulls();
  GatherConvexHullsToRank0();
  GetOutput() = local_data_;
  return true;
}

void PylaevaSConvexHullBinMPI::GatherConvexHullsToRank0() {
  std::vector<int> hull_counts(size_, 0);
  int local_hull_count = static_cast<int>(local_data_.convex_hulls.size());
  MPI_Gather(&local_hull_count, 1, MPI_INT, hull_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    std::vector<std::vector<Point>> rank0_hulls = local_data_.convex_hulls;
    local_data_.convex_hulls.clear();

    for (int i = 1; i < size_; ++i) {
      ReceiveHullsFromProcess(i, hull_counts[i]);
    }

    for (const auto &hull : rank0_hulls) {
      local_data_.convex_hulls.push_back(hull);
    }
  } else {
    SendHullsToRank0();
  }
}

void PylaevaSConvexHullBinMPI::ReceiveHullsFromProcess(int source_rank, int hull_count) {
  for (int j = 0; j < hull_count; ++j) {
    int hull_size = 0;
    MPI_Recv(&hull_size, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (hull_size <= 0) {
      continue;
    }

    std::vector<Point> hull(hull_size);
    std::vector<int> point_data(static_cast<size_t>(hull_size) * 2);
    MPI_Recv(point_data.data(), hull_size * 2, MPI_INT, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int k = 0; k < hull_size; ++k) {
      size_t base_idx = static_cast<size_t>(k) * 2;
      hull[k].x = point_data[base_idx];
      hull[k].y = point_data[base_idx + 1];
    }

    local_data_.convex_hulls.push_back(hull);
  }
}

void PylaevaSConvexHullBinMPI::SendHullsToRank0() {
  for (const auto &hull : local_data_.convex_hulls) {
    int hull_size = static_cast<int>(hull.size());
    MPI_Send(&hull_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    if (hull_size <= 0) {
      continue;
    }

    std::vector<int> point_data;
    point_data.reserve(static_cast<size_t>(hull_size) * 2);
    for (const auto &point : hull) {
      point_data.push_back(point.x);
      point_data.push_back(point.y);
    }

    MPI_Send(point_data.data(), hull_size * 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
}

void PylaevaSConvexHullBinMPI::ProcessComponentsAndComputeHulls() {
  local_data_.convex_hulls.clear();

  for (const auto &component : local_data_.components) {
    if (component.size() >= 3) {
      local_data_.convex_hulls.push_back(GrahamScan(component));
    } else if (!component.empty()) {
      local_data_.convex_hulls.push_back(component);
    }
  }
}

void PylaevaSConvexHullBinMPI::FindConnectedComponentsMpi() {
  int width = local_data_.width;
  int height = local_data_.height;
  int local_rows = end_row_ - start_row_;

  int extended_start_row = std::max(0, start_row_ - 1);
  int extended_end_row = std::min(height, end_row_ + 1);
  int extended_local_rows = extended_end_row - extended_start_row;

  std::vector<uint8_t> extended_pixels(static_cast<size_t>(extended_local_rows) * width);

  for (int row = 0; row < local_rows; ++row) {
    int global_row = start_row_ + row;
    int ext_row = global_row - extended_start_row;
    for (int col = 0; col < width; ++col) {
      size_t local_idx = (static_cast<size_t>(row) * static_cast<size_t>(width)) + static_cast<size_t>(col);
      size_t ext_idx = (static_cast<size_t>(ext_row) * static_cast<size_t>(width)) + static_cast<size_t>(col);
      extended_pixels[ext_idx] = local_data_.pixels[local_idx];
    }
  }

  ExchangeBoundaryRows(width, local_rows, extended_start_row, extended_local_rows, extended_pixels);

  std::vector<bool> visited_extended(static_cast<size_t>(extended_local_rows) * width, false);
  std::vector<std::vector<Point>> all_components;

  ProcessExtendedRegion(width, extended_start_row, extended_local_rows, extended_pixels, visited_extended,
                        all_components);

  FilterLocalComponents(all_components);
}

void PylaevaSConvexHullBinMPI::ExchangeBoundaryRows(int width, int local_rows, int extended_start_row,
                                                    int extended_local_rows,
                                                    std::vector<uint8_t> &extended_pixels) const {
  if (start_row_ > 0) {
    int prev_rank = rank_ - 1;
    int top_row_in_extended = 0;
    MPI_Send(&extended_pixels[static_cast<size_t>(top_row_in_extended) * static_cast<size_t>(width)], width,
             MPI_UINT8_T, prev_rank, 0, MPI_COMM_WORLD);
  }

  if (end_row_ < local_data_.height) {
    int next_rank = rank_ + 1;
    if (next_rank < size_) {
      int bottom_row_in_extended = extended_local_rows - 1;
      MPI_Recv(&extended_pixels[static_cast<size_t>(bottom_row_in_extended) * static_cast<size_t>(width)], width,
               MPI_UINT8_T, next_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  if (end_row_ < local_data_.height) {
    int next_rank = rank_ + 1;
    if (next_rank < size_) {
      int bottom_local_row = local_rows - 1;
      int bottom_row_in_extended = (start_row_ + bottom_local_row) - extended_start_row;
      MPI_Send(&extended_pixels[static_cast<size_t>(bottom_row_in_extended) * static_cast<size_t>(width)], width,
               MPI_UINT8_T, next_rank, 1, MPI_COMM_WORLD);
    }
  }

  if (start_row_ > 0) {
    int prev_rank = rank_ - 1;
    int top_row_in_extended = 0;
    MPI_Recv(&extended_pixels[static_cast<size_t>(top_row_in_extended) * static_cast<size_t>(width)], width,
             MPI_UINT8_T, prev_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void PylaevaSConvexHullBinMPI::ProcessExtendedRegion(int width, int extended_start_row, int extended_local_rows,
                                                     const std::vector<uint8_t> &extended_pixels,
                                                     std::vector<bool> &visited_extended,
                                                     std::vector<std::vector<Point>> &all_components) {
  for (int ext_row = 0; ext_row < extended_local_rows; ++ext_row) {
    int global_row = extended_start_row + ext_row;
    for (int col = 0; col < width; ++col) {
      size_t ext_idx = (static_cast<size_t>(ext_row) * static_cast<size_t>(width)) + static_cast<size_t>(col);

      if (extended_pixels[ext_idx] == 255 && !visited_extended[ext_idx]) {
        std::vector<Point> component;
        std::queue<Point> q;
        q.emplace(col, global_row);
        visited_extended[ext_idx] = true;

        while (!q.empty()) {
          Point p = q.front();
          q.pop();
          component.push_back(p);

          ProcessExtendedNeighbors(p, width, extended_start_row, extended_local_rows, extended_pixels, visited_extended,
                                   q);
        }

        if (!component.empty()) {
          all_components.push_back(component);
        }
      }
    }
  }
}

void PylaevaSConvexHullBinMPI::ProcessExtendedNeighbors(const Point &p, int width, int extended_start_row,
                                                        int extended_local_rows,
                                                        const std::vector<uint8_t> &extended_pixels,
                                                        std::vector<bool> &visited_extended, std::queue<Point> &q) {
  const std::vector<std::pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  for (const auto &dir : directions) {
    int nx = p.x + dir.first;
    int ny = p.y + dir.second;

    int ext_row = ny - extended_start_row;
    if (nx >= 0 && nx < width && ext_row >= 0 && ext_row < extended_local_rows) {
      size_t ext_idx = (static_cast<size_t>(ext_row) * static_cast<size_t>(width)) + static_cast<size_t>(nx);

      if (extended_pixels[ext_idx] == 255 && !visited_extended[ext_idx]) {
        visited_extended[ext_idx] = true;
        q.emplace(nx, ny);
      }
    }
  }
}

void PylaevaSConvexHullBinMPI::FilterLocalComponents(const std::vector<std::vector<Point>> &all_components) {
  local_data_.components.clear();

  for (const auto &component : all_components) {
    bool belongs_to_local = false;
    for (const auto &point : component) {
      if (point.y >= start_row_ && point.y < end_row_) {
        belongs_to_local = true;
        break;
      }
    }

    if (belongs_to_local) {
      local_data_.components.push_back(component);
    }
  }
}

bool PylaevaSConvexHullBinMPI::PostProcessingImpl() {
  return true;
}

std::vector<Point> PylaevaSConvexHullBinMPI::GrahamScan(const std::vector<Point> &points) {
  if (points.size() <= 3) {
    return points;
  }

  std::vector<Point> pts = points;
  int n = static_cast<int>(pts.size());

  int min_idx = 0;
  for (int i = 1; i < n; ++i) {
    if (pts[i].y < pts[min_idx].y || (pts[i].y == pts[min_idx].y && pts[i].x < pts[min_idx].x)) {
      min_idx = i;
    }
  }
  std::swap(pts[0], pts[min_idx]);

  Point pivot = pts[0];
  std::sort(pts.begin() + 1, pts.end(), [&pivot](const Point &a, const Point &b) {
    int orient = Cross(pivot, a, b);
    if (orient == 0) {
      return ((a.x - pivot.x) * (a.x - pivot.x)) + ((a.y - pivot.y) * (a.y - pivot.y)) <
             ((b.x - pivot.x) * (b.x - pivot.x)) + ((b.y - pivot.y) * (b.y - pivot.y));
    }
    return orient > 0;
  });

  std::vector<Point> hull;
  hull.reserve(pts.size());
  for (int i = 0; i < n; ++i) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), pts[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(pts[i]);
  }

  return hull;
}

}  // namespace pylaeva_s_convex_hull_bin
