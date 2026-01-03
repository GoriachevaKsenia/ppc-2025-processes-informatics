#include "matrix_band_multiplication/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "matrix_band_multiplication/common/include/common.hpp"

namespace matrix_band_multiplication {

namespace {
std::vector<int> BuildCounts(int total, int parts) {
  std::vector<int> counts(parts, 0);
  if (parts <= 0) {
    return counts;
  }
  const int base = total / parts;
  int remainder = total % parts;
  for (int i = 0; i < parts; ++i) {
    counts[i] = base + (remainder > 0 ? 1 : 0);
    if (remainder > 0) {
      --remainder;
    }
  }
  return counts;
}

std::vector<int> BuildDisplacements(const std::vector<int> &counts) {
  std::vector<int> displs(counts.size(), 0);
  for (std::size_t i = 1; i < counts.size(); ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
  return displs;
}

bool MatrixIsValid(const Matrix &matrix) {
  return matrix.rows > 0 && matrix.cols > 0 && matrix.values.size() == matrix.rows * matrix.cols;
}

}  // namespace

MatrixBandMultiplicationMpi::MatrixBandMultiplicationMpi(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Matrix{};
}

bool MatrixBandMultiplicationMpi::ValidationImpl() {
  const auto &matrix_a = GetInput().a;
  const auto &matrix_b = GetInput().b;
  if (!MatrixIsValid(matrix_a) || !MatrixIsValid(matrix_b)) {
    return false;
  }
  return matrix_a.cols == matrix_b.rows;
}

bool MatrixBandMultiplicationMpi::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  const auto &matrix_a = GetInput().a;
  const auto &matrix_b = GetInput().b;
  if (!BroadcastDimensions(matrix_a, matrix_b)) {
    return false;
  }

  PrepareRowDistribution(matrix_a);
  PrepareColumnDistribution(matrix_b);
  PrepareResultGatherInfo();
  return true;
}

bool MatrixBandMultiplicationMpi::BroadcastDimensions(const Matrix &matrix_a, const Matrix &matrix_b) {
  if (rank_ == 0) {
    rows_a_ = matrix_a.rows;
    cols_a_ = matrix_a.cols;
    rows_b_ = matrix_b.rows;
    cols_b_ = matrix_b.cols;
  }

  std::array<std::size_t, 4> dims = {rows_a_, cols_a_, rows_b_, cols_b_};
  MPI_Bcast(dims.data(), static_cast<int>(dims.size()), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  rows_a_ = dims[0];
  cols_a_ = dims[1];
  rows_b_ = dims[2];
  cols_b_ = dims[3];

  return rows_a_ > 0 && cols_a_ > 0 && rows_b_ > 0 && cols_b_ > 0;
}

void MatrixBandMultiplicationMpi::PrepareRowDistribution(const Matrix &matrix_a) {
  row_counts_ = BuildCounts(static_cast<int>(rows_a_), world_size_);
  row_displs_ = BuildDisplacements(row_counts_);

  std::vector<int> send_counts(row_counts_.size());
  std::ranges::transform(row_counts_, send_counts.begin(),
                         [this](int rows) { return rows * static_cast<int>(cols_a_); });
  std::vector<int> send_displs = BuildDisplacements(send_counts);

  const double *a_ptr = rank_ == 0 ? matrix_a.values.data() : nullptr;
  const int local_elems = send_counts[rank_];
  local_a_.assign(static_cast<std::size_t>(local_elems), 0.0);

  MPI_Scatterv(a_ptr, send_counts.data(), send_displs.data(), MPI_DOUBLE, local_a_.data(), local_elems, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
}

void MatrixBandMultiplicationMpi::PrepareColumnDistribution(const Matrix &matrix_b) {
  col_counts_ = BuildCounts(static_cast<int>(cols_b_), world_size_);
  col_displs_ = BuildDisplacements(col_counts_);
  max_cols_per_proc_ = ComputeMaxColumns();

  const std::size_t stripe_capacity = rows_b_ * static_cast<std::size_t>(max_cols_per_proc_);
  current_b_.assign(stripe_capacity, 0.0);
  rotation_buffer_.assign(stripe_capacity, 0.0);

  std::vector<double> packed;
  std::vector<int> send_counts(world_size_, 0);
  std::vector<int> send_displs(world_size_, 0);
  PreparePackedColumns(matrix_b, packed, send_counts, send_displs);

  const int recv_elements = col_counts_[rank_] * static_cast<int>(rows_b_);
  ScatterInitialStripe(packed, send_counts, send_displs, recv_elements);

  stripe_owner_ = rank_;
  current_cols_ = col_counts_[rank_];
}

int MatrixBandMultiplicationMpi::ComputeMaxColumns() const {
  int max_cols = 0;
  for (int count : col_counts_) {
    max_cols = std::max(max_cols, count);
  }
  return max_cols;
}

void MatrixBandMultiplicationMpi::PreparePackedColumns(const Matrix &matrix_b, std::vector<double> &packed,
                                                       std::vector<int> &send_counts,
                                                       std::vector<int> &send_displs) const {
  if (rank_ != 0) {
    return;
  }

  packed.reserve(matrix_b.values.size());
  int offset = 0;
  for (int owner = 0; owner < world_size_; ++owner) {
    const int cols = col_counts_[owner];
    const int elems = cols * static_cast<int>(rows_b_);
    send_counts[owner] = elems;
    send_displs[owner] = offset;
    const int col_start = col_displs_[owner];
    for (std::size_t row = 0; row < rows_b_; ++row) {
      const std::size_t base = row * cols_b_;
      for (int col = 0; col < cols; ++col) {
        const std::size_t src_index = base + static_cast<std::size_t>(col_start + col);
        packed.push_back(matrix_b.values[src_index]);
      }
    }
    offset += elems;
  }
}

void MatrixBandMultiplicationMpi::ScatterInitialStripe(const std::vector<double> &packed,
                                                       const std::vector<int> &send_counts,
                                                       const std::vector<int> &send_displs, int recv_elements) {
  const double *send_buffer = rank_ == 0 ? packed.data() : nullptr;
  const int *counts_ptr = rank_ == 0 ? send_counts.data() : nullptr;
  const int *displs_ptr = rank_ == 0 ? send_displs.data() : nullptr;

  MPI_Scatterv(send_buffer, counts_ptr, displs_ptr, MPI_DOUBLE, current_b_.data(), recv_elements, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
}

void MatrixBandMultiplicationMpi::PrepareResultGatherInfo() {
  const auto local_elements = static_cast<std::size_t>(row_counts_[rank_]) * cols_b_;
  local_result_.assign(local_elements, 0.0);

  result_counts_ = BuildCounts(static_cast<int>(rows_a_), world_size_);
  result_displs_ = BuildDisplacements(result_counts_);

  std::ranges::transform(result_counts_, result_counts_.begin(),
                         [this](int rows) { return rows * static_cast<int>(cols_b_); });
  std::ranges::transform(result_displs_, result_displs_.begin(),
                         [this](int rows_prefix) { return rows_prefix * static_cast<int>(cols_b_); });
}

void MatrixBandMultiplicationMpi::MultiplyStripe(const double *stripe_data, int stripe_cols, int stripe_offset,
                                                 int local_rows) {
  if (stripe_cols == 0 || local_rows == 0) {
    return;
  }

  for (int row = 0; row < local_rows; ++row) {
    for (int col = 0; col < stripe_cols; ++col) {
      double sum = 0.0;
      for (std::size_t k = 0; k < cols_a_; ++k) {
        const std::size_t a_idx = (static_cast<std::size_t>(row) * cols_a_) + k;
        const std::size_t b_idx = (k * static_cast<std::size_t>(stripe_cols)) + static_cast<std::size_t>(col);
        sum += local_a_[a_idx] * stripe_data[b_idx];
      }
      const std::size_t result_idx =
          (static_cast<std::size_t>(row) * cols_b_) + static_cast<std::size_t>(stripe_offset + col);
      local_result_[result_idx] = sum;
    }
  }
}

bool MatrixBandMultiplicationMpi::RunImpl() {
  const int local_rows = row_counts_[rank_];
  const int total_steps = world_size_;

  for (int step = 0; step < total_steps; ++step) {
    const int stripe_offset = col_displs_.empty() ? 0 : col_displs_[stripe_owner_];
    MultiplyStripe(current_b_.data(), current_cols_, stripe_offset, local_rows);

    if (world_size_ == 1 || step == total_steps - 1) {
      continue;
    }

    // Rotate column stripes so every rank multiplies against each subset of B.
    const int send_to = (rank_ - 1 + world_size_) % world_size_;
    const int recv_from = (rank_ + 1) % world_size_;
    const int send_cols = current_cols_;
    int recv_cols = 0;
    MPI_Sendrecv(&send_cols, 1, MPI_INT, send_to, 0, &recv_cols, 1, MPI_INT, recv_from, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    const int send_elements = send_cols * static_cast<int>(rows_b_);
    const int recv_elements = recv_cols * static_cast<int>(rows_b_);
    MPI_Sendrecv(current_b_.data(), send_elements, MPI_DOUBLE, send_to, 1, rotation_buffer_.data(), recv_elements,
                 MPI_DOUBLE, recv_from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::swap(current_b_, rotation_buffer_);
    current_cols_ = recv_cols;
    stripe_owner_ = (stripe_owner_ + 1) % world_size_;
  }

  return true;
}

bool MatrixBandMultiplicationMpi::PostProcessingImpl() {
  std::vector<double> gathered;
  if (rank_ == 0) {
    gathered.resize(rows_a_ * cols_b_);
  }

  const int local_result_elements = row_counts_[rank_] * static_cast<int>(cols_b_);
  MPI_Gatherv(local_result_.data(), local_result_elements, MPI_DOUBLE, gathered.data(), result_counts_.data(),
              result_displs_.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  auto &output = GetOutput();
  if (rank_ == 0) {
    output.rows = rows_a_;
    output.cols = cols_b_;
    output.values = std::move(gathered);
  }

  std::array<std::size_t, 2> dims = {output.rows, output.cols};
  MPI_Bcast(dims.data(), static_cast<int>(dims.size()), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  output.rows = dims[0];
  output.cols = dims[1];

  const std::size_t total_size = output.rows * output.cols;
  output.values.resize(total_size);
  MPI_Bcast(output.values.data(), static_cast<int>(total_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

}  // namespace matrix_band_multiplication
