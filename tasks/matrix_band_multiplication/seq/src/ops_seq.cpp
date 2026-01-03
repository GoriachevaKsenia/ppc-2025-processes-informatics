#include "matrix_band_multiplication/seq/include/ops_seq.hpp"

#include <cstddef>

#include "matrix_band_multiplication/common/include/common.hpp"

namespace matrix_band_multiplication {

namespace {
bool MatrixIsValid(const Matrix &m) {
  return m.rows > 0 && m.cols > 0 && m.values.size() == m.rows * m.cols;
}

}  // namespace

MatrixBandMultiplicationSeq::MatrixBandMultiplicationSeq(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Matrix{};
}

bool MatrixBandMultiplicationSeq::ValidationImpl() {
  const auto &matrix_a = GetInput().a;
  const auto &matrix_b = GetInput().b;
  if (!MatrixIsValid(matrix_a) || !MatrixIsValid(matrix_b)) {
    return false;
  }
  return matrix_a.cols == matrix_b.rows;
}

bool MatrixBandMultiplicationSeq::PreProcessingImpl() {
  const auto &matrix_a = GetInput().a;
  const auto &matrix_b = GetInput().b;
  result_.assign(matrix_a.rows * matrix_b.cols, 0.0);
  transposed_b_.assign(matrix_b.cols * matrix_b.rows, 0.0);
  const std::size_t cols_b = matrix_b.cols;
  const std::size_t rows_b = matrix_b.rows;
  for (std::size_t i = 0; i < rows_b; ++i) {
    const std::size_t row_offset = i * cols_b;
    for (std::size_t j = 0; j < cols_b; ++j) {
      transposed_b_[(j * rows_b) + i] = matrix_b.values[row_offset + j];
    }
  }
  return true;
}

bool MatrixBandMultiplicationSeq::RunImpl() {
  const auto &matrix_a = GetInput().a;
  const auto &matrix_b = GetInput().b;
  const std::size_t rows_a = matrix_a.rows;
  const std::size_t cols_a = matrix_a.cols;
  const std::size_t cols_b = matrix_b.cols;
  const double *a_data = matrix_a.values.data();
  const double *b_data = transposed_b_.data();
  for (std::size_t i = 0; i < rows_a; ++i) {
    const double *a_row = a_data + (i * cols_a);
    for (std::size_t j = 0; j < cols_b; ++j) {
      const double *b_col = b_data + (j * cols_a);
      double sum = 0.0;
      for (std::size_t k = 0; k < cols_a; ++k) {
        sum += a_row[k] * b_col[k];
      }
      result_[(i * cols_b) + j] = sum;
    }
  }
  return true;
}

bool MatrixBandMultiplicationSeq::PostProcessingImpl() {
  OutType output;
  output.rows = GetInput().a.rows;
  output.cols = GetInput().b.cols;
  output.values = result_;
  GetOutput() = output;
  return true;
}

}  // namespace matrix_band_multiplication
