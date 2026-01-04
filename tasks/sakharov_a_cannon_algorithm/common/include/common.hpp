#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace sakharov_a_cannon_algorithm {

struct MatrixInput {
  int rows_a = 0;
  int cols_a = 0;
  int rows_b = 0;
  int cols_b = 0;
  std::vector<double> a;
  std::vector<double> b;
};

using InType = MatrixInput;
using OutType = std::vector<double>;
using TestType = std::tuple<MatrixInput, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::size_t Idx(int cols, int row, int col) {
  return (static_cast<std::size_t>(row) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(col);
}

inline bool IsValidInput(const MatrixInput &input) {
  if (input.rows_a <= 0 || input.cols_a <= 0 || input.rows_b <= 0 || input.cols_b <= 0 ||
      input.cols_a != input.rows_b) {
    return false;
  }
  auto expected_a = static_cast<std::size_t>(input.rows_a) * static_cast<std::size_t>(input.cols_a);
  auto expected_b = static_cast<std::size_t>(input.rows_b) * static_cast<std::size_t>(input.cols_b);
  return input.a.size() == expected_a && input.b.size() == expected_b;
}

}  // namespace sakharov_a_cannon_algorithm
