#include "leonova_a_star/seq/include/ops_seq.hpp"

#include <cstddef>
#include <new>
#include <tuple>
#include <vector>

#include "leonova_a_star/common/include/common.hpp"

namespace leonova_a_star {

namespace {

constexpr size_t kMaxMatrixSize = 10000;

bool CheckMatricesNotEmpty(const std::vector<std::vector<int>> &matrix_a,
                           const std::vector<std::vector<int>> &matrix_b) {
  return !matrix_a.empty() && !matrix_b.empty() && !matrix_a[0].empty() && !matrix_b[0].empty();
}

int ComputeDotProduct(const std::vector<std::vector<int>> &matrix_a, const std::vector<std::vector<int>> &matrix_b,
                      size_t row, size_t col, size_t length) {
  int sum = 0;
  for (size_t k = 0; k < length; ++k) {
    if (row < matrix_a.size() && k < matrix_a[row].size() && k < matrix_b.size() && col < matrix_b[k].size()) {
      sum += matrix_a[row][k] * matrix_b[k][col];
    }
  }
  return sum;
}

}  // namespace

LeonovaAStarSEQ::LeonovaAStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  if (CheckMatricesNotEmpty(matrix_a, matrix_b)) {
    size_t rows = matrix_a.size();
    size_t cols = matrix_b[0].size();

    if (rows <= kMaxMatrixSize && cols <= kMaxMatrixSize) {
      GetOutput().resize(rows);
      for (auto &row : GetOutput()) {
        row.resize(cols, 0);
      }
    }
  }
}

bool LeonovaAStarSEQ::PreProcessingImpl() {
  return true;
}

bool LeonovaAStarSEQ::ValidationImpl() {
  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  if (!CheckMatricesNotEmpty(matrix_a, matrix_b)) {
    return false;
  }

  if (matrix_a.size() > kMaxMatrixSize || matrix_b.size() > kMaxMatrixSize) {
    return false;
  }

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();

  for (size_t index = 1; index < rows_a; ++index) {
    if (index >= matrix_a.size() || matrix_a[index].size() != cols_a) {
      return false;
    }
  }

  size_t rows_b = matrix_b.size();
  size_t cols_b = matrix_b[0].size();

  for (size_t index = 1; index < rows_b; ++index) {
    if (index >= matrix_b.size() || matrix_b[index].size() != cols_b) {
      return false;
    }
  }

  return cols_a == rows_b;
}

bool LeonovaAStarSEQ::ResizeOutputMatrix(size_t rows, size_t cols) {
  try {
    if (GetOutput().size() != rows || (rows > 0 && !GetOutput().empty() && GetOutput()[0].size() != cols)) {
      GetOutput().resize(rows);
      for (auto &row : GetOutput()) {
        row.resize(cols, 0);
      }
    }
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

void LeonovaAStarSEQ::MultiplyMatrices(const std::vector<std::vector<int>> &matrix_a,
                                       const std::vector<std::vector<int>> &matrix_b, size_t rows_a, size_t cols_a,
                                       size_t cols_b) {
  for (size_t index = 0; index < rows_a; ++index) {
    for (size_t jndex = 0; jndex < cols_b; ++jndex) {
      size_t length = cols_a;
      GetOutput()[index][jndex] = ComputeDotProduct(matrix_a, matrix_b, index, jndex, length);
    }
  }
}

bool LeonovaAStarSEQ::RunImpl() {
  if (!ValidationImpl()) {
    return false;
  }
  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  if (!CheckMatricesNotEmpty(matrix_a, matrix_b)) {
    return false;
  }

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();
  size_t cols_b = matrix_b[0].size();

  if (rows_a > kMaxMatrixSize || cols_b > kMaxMatrixSize) {
    return false;
  }

  if (!ResizeOutputMatrix(rows_a, cols_b)) {
    return false;
  }

  MultiplyMatrices(matrix_a, matrix_b, rows_a, cols_a, cols_b);
  return true;
}

bool LeonovaAStarSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace leonova_a_star
