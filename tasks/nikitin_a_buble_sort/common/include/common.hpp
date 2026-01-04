#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace nikitin_a_buble_sort {

using InType = std::vector<double>;   // Входной тип: вектор для сортировки
using OutType = std::vector<double>;  // Выходной тип: отсортированный вектор
using TestType = std::tuple<double, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nikitin_a_buble_sort
