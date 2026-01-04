#include "gasenin_l_image_smooth/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gasenin_l_image_smooth/common/include/common.hpp"

namespace gasenin_l_image_smooth {

GaseninLImageSmoothMPI::GaseninLImageSmoothMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GaseninLImageSmoothMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::array<int, 4> params = {0, 0, 0, 0};

  if (rank == 0) {
    params[1] = GetInput().width;
    params[2] = GetInput().height;
    params[3] = GetInput().kernel_size;

    if (params[1] > 0 && params[2] > 0 && params[3] > 0) {
      params[0] = 1;
    } else {
      params[0] = 0;
    }
  }

  MPI_Bcast(params.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);

  if (params[0] == 0) {
    return false;
  }

  GetInput().width = params[1];
  GetInput().height = params[2];
  GetInput().kernel_size = params[3];

  return true;
}

bool GaseninLImageSmoothMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = GetInput();
    std::fill(GetOutput().data.begin(), GetOutput().data.end(), 0);
  } else {
    GetOutput().width = GetInput().width;
    GetOutput().height = GetInput().height;
    GetOutput().kernel_size = GetInput().kernel_size;
  }
  return true;
}

namespace {

void CalculateRowDistribution(int rank, int size, int height, int &start_row, int &end_row, int &local_rows) {
  const int base_rows = height / size;
  const int extra_rows = height % size;

  start_row = (rank * base_rows) + std::min(rank, extra_rows);
  end_row = start_row + base_rows + (rank < extra_rows ? 1 : 0);
  local_rows = end_row - start_row;
}

void PrepareScatterParameters(int rank, int size, int width, int height, int kernel_radius, int base_rows,
                              int extra_rows, std::vector<int> &sendcounts, std::vector<int> &displs) {
  if (rank == 0) {
    sendcounts.resize(size);
    displs.resize(size);
    for (int process_idx = 0; process_idx < size; ++process_idx) {
      const int row_start = (process_idx * base_rows) + std::min(process_idx, extra_rows);
      const int row_end = row_start + base_rows + (process_idx < extra_rows ? 1 : 0);

      if (row_end > row_start) {
        const int actual_start = std::max(0, row_start - kernel_radius);
        const int actual_end = std::min(height, row_end + kernel_radius);
        sendcounts[process_idx] = (actual_end - actual_start) * width;
        displs[process_idx] = actual_start * width;
      } else {
        sendcounts[process_idx] = 0;
        displs[process_idx] = 0;
      }
    }
  }
}

void ProcessInteriorPixel(int buffer_y, int col, int width, int kernel_size, const std::vector<uint8_t> &local_data,
                          std::vector<uint8_t> &local_result, int local_row_idx) {
  const int kernel_sq = kernel_size * kernel_size;
  const int radius = kernel_size / 2;

  const auto *row_ptr = local_data.data() + (static_cast<ptrdiff_t>(buffer_y - radius) * width) + (col - radius);
  int sum = 0;

  for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y) {
    for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
      sum += row_ptr[kernel_x];
    }
    row_ptr += width;
  }
  local_result[(local_row_idx * width) + col] = static_cast<uint8_t>(sum / kernel_sq);
}

void ProcessBorderPixel(int global_y, int col, int width, int height, int kernel_radius, int actual_start,
                        const std::vector<uint8_t> &local_data, std::vector<uint8_t> &local_result, int local_row_idx) {
  int sum = 0;
  int count = 0;

  for (int kernel_y = -kernel_radius; kernel_y <= kernel_radius; ++kernel_y) {
    const int global_y_offset = global_y + kernel_y;
    const int clamped_y = Clamp(global_y_offset, 0, height - 1);
    const int buffer_y = clamped_y - actual_start;

    for (int kernel_x = -kernel_radius; kernel_x <= kernel_radius; ++kernel_x) {
      const int global_x_offset = Clamp(col + kernel_x, 0, width - 1);
      sum += local_data[(buffer_y * width) + global_x_offset];
      ++count;
    }
  }

  const int index = (local_row_idx * width) + col;
  local_result[index] = (count > 0) ? static_cast<uint8_t>(sum / count) : 0;
}

void ProcessLocalRows(int start_row, int local_rows, int width, int height, int kernel_size, int kernel_radius,
                      int actual_start, const std::vector<uint8_t> &local_data, std::vector<uint8_t> &local_result) {
  const int row_offset_in_buffer = start_row - actual_start;

  for (int local_row_idx = 0; local_row_idx < local_rows; ++local_row_idx) {
    const int global_y = start_row + local_row_idx;
    const int buffer_y = row_offset_in_buffer + local_row_idx;
    const bool is_global_border_y = (global_y < kernel_radius) || (global_y >= height - kernel_radius);

    for (int col = 0; col < width; ++col) {
      const bool is_border_col = (col < kernel_radius) || (col >= width - kernel_radius);

      if (!is_global_border_y && !is_border_col) {
        ProcessInteriorPixel(buffer_y, col, width, kernel_size, local_data, local_result, local_row_idx);
      } else {
        ProcessBorderPixel(global_y, col, width, height, kernel_radius, actual_start, local_data, local_result,
                           local_row_idx);
      }
    }
  }
}

void PrepareGatherParameters(int rank, int size, int width, int height, std::vector<int> &sendcounts,
                             std::vector<int> &displs) {
  if (rank == 0) {
    sendcounts.resize(size);
    displs.resize(size);

    const int base_rows = height / size;
    const int extra_rows = height % size;

    for (int process_idx = 0; process_idx < size; ++process_idx) {
      const int row_start = (process_idx * base_rows) + std::min(process_idx, extra_rows);
      const int row_end = row_start + base_rows + (process_idx < extra_rows ? 1 : 0);
      sendcounts[process_idx] = (row_end - row_start) * width;
      displs[process_idx] = row_start * width;
    }
  }
}

}  // namespace

bool GaseninLImageSmoothMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int width = GetInput().width;
  const int height = GetInput().height;
  const int kernel_size = GetInput().kernel_size;
  const int kernel_radius = kernel_size / 2;

  int start_row = 0;
  int end_row = 0;
  int local_rows = 0;
  CalculateRowDistribution(rank, size, height, start_row, end_row, local_rows);

  if (local_rows <= 0) {
    return true;
  }

  const int overlap_top = kernel_radius;
  const int overlap_bottom = kernel_radius;
  const int actual_start = std::max(0, start_row - overlap_top);
  const int actual_end = std::min(height, end_row + overlap_bottom);
  const int extended_rows = actual_end - actual_start;
  const int recv_count = extended_rows * width;

  std::vector<uint8_t> local_data(recv_count);

  const int base_rows = height / size;
  const int extra_rows = height % size;
  std::vector<int> sendcounts;
  std::vector<int> displs;

  PrepareScatterParameters(rank, size, width, height, kernel_radius, base_rows, extra_rows, sendcounts, displs);

  MPI_Scatterv((rank == 0 ? GetInput().data.data() : nullptr), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
               local_data.data(), recv_count, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  std::vector<uint8_t> local_result(static_cast<size_t>(local_rows) * width);

  ProcessLocalRows(start_row, local_rows, width, height, kernel_size, kernel_radius, actual_start, local_data,
                   local_result);

  PrepareGatherParameters(rank, size, width, height, sendcounts, displs);

  MPI_Gatherv(local_result.data(), local_rows * width, MPI_UNSIGNED_CHAR,
              (rank == 0 ? GetOutput().data.data() : nullptr), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0,
              MPI_COMM_WORLD);

  return true;
}

bool GaseninLImageSmoothMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_image_smooth
