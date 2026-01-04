#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace romanov_a_crs_product {

constexpr double kEps = 1e-6;

struct Dense {
  uint64_t n;
  uint64_t m;

  std::vector<double> data;

  Dense(uint64_t n, uint64_t m) : n(n), m(m), data(n * m, 0.0) {}
  explicit Dense(uint64_t n) : n(n), m(n), data(n * n, 0.0) {}

  [[nodiscard]] uint64_t GetRows() const {
    return n;
  }

  [[nodiscard]] uint64_t GetCols() const {
    return m;
  }

  double &operator()(uint64_t i, uint64_t j) {
    return data[(i * m) + j];
  }

  double operator()(uint64_t i, uint64_t j) const {
    return data[(i * m) + j];
  }
};

struct CRS {
  uint64_t n;
  uint64_t m;

  std::vector<double> value;
  std::vector<uint64_t> column;
  std::vector<uint64_t> row_index;

  CRS() : n(0), m(0), row_index(1, 0) {}
  CRS(uint64_t n, uint64_t m) : n(n), m(m), row_index(n + 1, 0) {}
  explicit CRS(uint64_t n) : n(n), m(n), row_index(n + 1, 0) {}

  [[nodiscard]] uint64_t GetRows() const {
    return n;
  }

  [[nodiscard]] uint64_t GetCols() const {
    return m;
  }

  [[nodiscard]] uint64_t Nnz() const {
    return value.size();
  }

  bool operator==(const CRS &other) const {
    if (n != other.n || m != other.m) {
      return false;
    }

    if (row_index != other.row_index || column != other.column) {
      return false;
    }

    for (uint64_t i = 0; i < value.size(); ++i) {
      if (std::abs(value[i] - other.value[i]) > kEps) {
        return false;
      }
    }

    return true;
  }

  void Transpose() {
    uint64_t nnz_val = this->Nnz();

    std::vector<double> new_value(nnz_val);
    std::vector<uint64_t> new_column(nnz_val);
    std::vector<uint64_t> new_row_index(m + 1, 0);

    for (uint64_t i = 0; i < nnz_val; ++i) {
      ++new_row_index[column[i] + 1];
    }

    for (uint64_t i = 1; i <= m; ++i) {
      new_row_index[i] += new_row_index[i - 1];
    }

    std::vector<uint64_t> offset = new_row_index;
    for (uint64_t row = 0; row < n; ++row) {
      for (uint64_t idx = row_index[row]; idx < row_index[row + 1]; ++idx) {
        uint64_t col = column[idx];
        uint64_t pos = offset[col]++;

        new_value[pos] = value[idx];
        new_column[pos] = row;
      }
    }

    value.swap(new_value);
    column.swap(new_column);
    row_index.swap(new_row_index);

    std::swap(n, m);
  }

  [[nodiscard]] CRS GetTransposed() const {
    CRS result = *this;
    result.Transpose();
    return result;
  }

  void FillRandom(double density, unsigned int seed = 0) {
    if (density < 0.0 || density > 1.0) {
      throw std::invalid_argument("Density must be within [0, 1]!");
    }

    value.clear();
    column.clear();
    row_index.assign(n + 1, 0);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> val(-1.0, 1.0);

    for (uint64_t row = 0; row < n; ++row) {
      row_index[row + 1] = row_index[row];
      for (uint64_t col = 0; col < m; ++col) {
        if (prob(gen) < density) {
          value.push_back(val(gen));
          column.push_back(col);
          ++row_index[row + 1];
        }
      }
    }
  }

  [[nodiscard]] CRS ExtractRows(uint64_t start_r, uint64_t end_r) const {
    if (start_r >= end_r || start_r >= n) {
      return CRS{0, m};
    }

    end_r = std::min(end_r, n);
    uint64_t new_n = end_r - start_r;

    uint64_t nnz_start = row_index[start_r];
    uint64_t nnz_end = row_index[end_r];
    uint64_t nnz_count = nnz_end - nnz_start;

    CRS result(new_n, m);
    result.value.resize(nnz_count);
    result.column.resize(nnz_count);
    result.row_index.resize(new_n + 1);

    std::copy(value.begin() + static_cast<std::ptrdiff_t>(nnz_start),
              value.begin() + static_cast<std::ptrdiff_t>(nnz_end), result.value.begin());
    std::copy(column.begin() + static_cast<std::ptrdiff_t>(nnz_start),
              column.begin() + static_cast<std::ptrdiff_t>(nnz_end), result.column.begin());

    for (uint64_t i = 0; i <= new_n; ++i) {
      result.row_index[i] = row_index[start_r + i] - nnz_start;
    }

    return result;
  }

  static CRS ConcatRows(const std::vector<CRS> &parts) {
    if (parts.empty()) {
      return CRS{};
    }

    uint64_t total_n = 0;
    uint64_t total_nnz = 0;
    uint64_t m = parts[0].m;

    for (const auto &part : parts) {
      total_n += part.n;
      total_nnz += part.Nnz();
    }

    CRS result(total_n, m);
    result.value.resize(total_nnz);
    result.column.resize(total_nnz);
    result.row_index.resize(total_n + 1);

    uint64_t row_offset = 0;
    uint64_t nnz_offset = 0;

    for (const auto &part : parts) {
      std::ranges::copy(part.value, result.value.begin() + static_cast<std::ptrdiff_t>(nnz_offset));
      std::ranges::copy(part.column, result.column.begin() + static_cast<std::ptrdiff_t>(nnz_offset));

      for (uint64_t i = 0; i <= part.n; ++i) {
        result.row_index[row_offset + i] = part.row_index[i] + nnz_offset;
      }

      row_offset += part.n;
      nnz_offset += part.Nnz();
    }

    return result;
  }
};

inline double ComputeDotProduct(uint64_t a_begin, uint64_t a_end, const std::vector<uint64_t> &a_col,
                                const std::vector<double> &a_val, uint64_t b_begin, uint64_t b_end,
                                const std::vector<uint64_t> &b_col, const std::vector<double> &b_val) {
  double sum = 0.0;
  uint64_t a_pos = a_begin;
  uint64_t b_pos = b_begin;

  while (a_pos < a_end && b_pos < b_end) {
    uint64_t a_col_idx = a_col[a_pos];
    uint64_t b_col_idx = b_col[b_pos];

    if (a_col_idx == b_col_idx) {
      sum += a_val[a_pos] * b_val[b_pos];
      ++a_pos;
      ++b_pos;
    } else if (a_col_idx < b_col_idx) {
      ++a_pos;
    } else {
      ++b_pos;
    }
  }

  return sum;
}

inline std::vector<std::pair<uint64_t, double>> ComputeResultRow(const CRS &a, uint64_t row_idx,
                                                                 const CRS &b_transposed, uint64_t num_cols) {
  std::vector<std::pair<uint64_t, double>> result_row;
  result_row.reserve(num_cols);

  uint64_t a_begin = a.row_index[row_idx];
  uint64_t a_end = a.row_index[row_idx + 1];

  if (a_begin == a_end) {
    return result_row;
  }

  for (uint64_t j = 0; j < num_cols; ++j) {
    uint64_t b_begin = b_transposed.row_index[j];
    uint64_t b_end = b_transposed.row_index[j + 1];

    if (b_begin == b_end) {
      continue;
    }

    double sum =
        ComputeDotProduct(a_begin, a_end, a.column, a.value, b_begin, b_end, b_transposed.column, b_transposed.value);

    if (std::abs(sum) > kEps) {
      result_row.emplace_back(j, sum);
    }
  }

  return result_row;
}

inline void AppendRowToCRS(CRS &result, const std::vector<std::pair<uint64_t, double>> &row_data, uint64_t row_index) {
  result.row_index[row_index] = result.column.size();

  for (const auto &[col, val] : row_data) {
    result.column.push_back(col);
    result.value.push_back(val);
  }
}

inline CRS operator*(const CRS &a, const CRS &b) {
  if (a.GetCols() != b.GetRows()) {
    throw std::runtime_error("Matrix dimensions do not match for multiplication!");
  }

  uint64_t n_rows = a.GetRows();
  uint64_t n_cols = b.GetCols();

  CRS result{n_rows, n_cols};
  CRS b_transposed = b.GetTransposed();

  result.row_index.resize(n_rows + 1);
  result.row_index[0] = 0;

  for (uint64_t i = 0; i < n_rows; ++i) {
    auto row_result = ComputeResultRow(a, i, b_transposed, n_cols);
    AppendRowToCRS(result, row_result, i);
  }

  result.row_index[n_rows] = result.column.size();

  return result;
}

inline CRS ToCRS(const Dense &a) {
  uint64_t rows = a.GetRows();
  uint64_t cols = a.GetCols();
  CRS crs(rows, cols);

  for (uint64_t i = 0; i < rows; ++i) {
    crs.row_index[i + 1] = crs.row_index[i];
    for (uint64_t j = 0; j < cols; ++j) {
      if (std::abs(a(i, j)) > kEps) {
        crs.value.push_back(a(i, j));
        crs.column.push_back(j);
        ++crs.row_index[i + 1];
      }
    }
  }
  return crs;
}

using InType = std::tuple<CRS, CRS>;
using OutType = CRS;
using TestType = std::tuple<CRS, CRS, CRS>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace romanov_a_crs_product
