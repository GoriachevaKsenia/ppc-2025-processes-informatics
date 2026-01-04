#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace maslova_u_row_matr_vec_mult {
struct Matrix {
  std::vector<double> data;
  size_t rows = 0;
  size_t cols = 0;
};

using InType = std::pair<Matrix, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace maslova_u_row_matr_vec_mult
