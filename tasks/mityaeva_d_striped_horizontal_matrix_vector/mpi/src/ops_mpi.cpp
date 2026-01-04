#include "mityaeva_d_striped_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "mityaeva_d_striped_horizontal_matrix_vector/common/include/common.hpp"

namespace mityaeva_d_striped_horizontal_matrix_vector {

namespace {

void CalcRows(int rows, int size, int rank, int &my_rows, int &start_row) {
  const int base = rows / size;
  const int rem = rows % size;

  my_rows = base + static_cast<int>(rank < rem);

  start_row = 0;
  for (int rr = 0; rr < rank; ++rr) {
    start_row += base + static_cast<int>(rr < rem);
  }
}

void BuildScatter(int rows, int cols, int size, std::vector<int> &counts, std::vector<int> &displs) {
  counts.resize(size);
  displs.resize(size);

  const int base = rows / size;
  const int rem = rows % size;

  int disp = 0;
  for (int rr = 0; rr < size; ++rr) {
    const int rr_rows = base + static_cast<int>(rr < rem);
    counts[rr] = rr_rows * cols;
    displs[rr] = disp;
    disp += counts[rr];
  }
}

void BuildGather(int rows, int size, std::vector<int> &counts, std::vector<int> &displs) {
  counts.resize(size);
  displs.resize(size);

  const int base = rows / size;
  const int rem = rows % size;

  int disp = 0;
  for (int rr = 0; rr < size; ++rr) {
    const int rr_rows = base + static_cast<int>(rr < rem);
    counts[rr] = rr_rows;
    displs[rr] = disp;
    disp += counts[rr];
  }
}

}  // namespace

StripedHorizontalMatrixVectorMPI::StripedHorizontalMatrixVectorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>{0.0};
}

bool StripedHorizontalMatrixVectorMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (input.size() < 3) {
    return false;
  }

  const int rows = static_cast<int>(input[0]);
  const int cols = static_cast<int>(input[1]);

  if (rows <= 0 || cols <= 0) {
    return false;
  }
  if (static_cast<int>(input[2]) != cols) {
    return false;
  }

  const size_t expected = 3 + (static_cast<size_t>(rows) * static_cast<size_t>(cols)) + static_cast<size_t>(cols);
  return input.size() == expected;
}

bool StripedHorizontalMatrixVectorMPI::PreProcessingImpl() {
  return true;
}

bool StripedHorizontalMatrixVectorMPI::RunImpl() {
  const auto &input = GetInput();
  try {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rows = static_cast<int>(input[0]);
    const int cols = static_cast<int>(input[1]);

    int my_rows = 0;
    int start_row = 0;
    CalcRows(rows, size, rank, my_rows, start_row);

    std::vector<int> sendcounts;
    std::vector<int> sdispls;
    std::vector<int> recvcounts;
    std::vector<int> rdispls;

    BuildScatter(rows, cols, size, sendcounts, sdispls);
    BuildGather(rows, size, recvcounts, rdispls);

    const size_t matrix_start = 3;
    const size_t vector_start = matrix_start + (static_cast<size_t>(rows) * static_cast<size_t>(cols));

    const double *a_ptr = input.data() + matrix_start;
    const double *x_all_ptr = input.data() + vector_start;

    std::vector<double> local_a(static_cast<size_t>(my_rows) * static_cast<size_t>(cols));

    std::vector<double> x(static_cast<size_t>(cols), 0.0);
    if (rank == 0) {
      std::copy(x_all_ptr, x_all_ptr + cols, x.begin());
    }

    MPI_Scatterv((rank == 0) ? a_ptr : nullptr, sendcounts.data(), sdispls.data(), MPI_DOUBLE, local_a.data(),
                 my_rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(x.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> y_local(static_cast<size_t>(my_rows), 0.0);
    for (int i = 0; i < my_rows; ++i) {
      double sum = 0.0;
      const size_t row_base = static_cast<size_t>(i) * static_cast<size_t>(cols);
      for (int j = 0; j < cols; ++j) {
        sum += local_a[row_base + static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
      }
      y_local[static_cast<size_t>(i)] = sum;
    }

    std::vector<double> y(static_cast<size_t>(rows), 0.0);
    MPI_Allgatherv(y_local.data(), my_rows, MPI_DOUBLE, y.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    auto &out = GetOutput();
    out.clear();
    out.reserve(static_cast<size_t>(rows) + 1);
    out.push_back(static_cast<double>(rows));
    out.insert(out.end(), y.begin(), y.end());

    return true;
  } catch (...) {
    return false;
  }
}

bool StripedHorizontalMatrixVectorMPI::PostProcessingImpl() {
  const auto &output = GetOutput();
  if (output.empty()) {
    return false;
  }

  const int rows = static_cast<int>(GetInput()[0]);
  return static_cast<int>(output[0]) == rows && output.size() == static_cast<size_t>(rows) + 1;
}

}  // namespace mityaeva_d_striped_horizontal_matrix_vector
