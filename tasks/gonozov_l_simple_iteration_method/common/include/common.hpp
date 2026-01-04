#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace gonozov_l_simple_iteration_method {

using InType = std::tuple<int, std::vector<double>,
                          std::vector<double>>;  // int - число неизвестных в СЛАУ, std::vector<double> - для правой
                                                 // колонки СЛАУ (2) и для остального СЛАУ (1)
using OutType = std::vector<double>;             // результат приближений для каждой неизвестной
using TestType = std::tuple<int, std::vector<double>, std::vector<double>, std::string, std::vector<double>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gonozov_l_simple_iteration_method
