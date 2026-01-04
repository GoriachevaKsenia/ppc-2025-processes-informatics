#pragma once

#include <functional>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace gonozov_l_global_search {

using InType = std::tuple<std::function<double(double)>, double, double, double, double>;
// 0 - минимизируемая функция,
// 1 - значение коэффициента r
// 2 - левая граница интервала
// 3 - правая граница интервала
// 4 - точность вычислений
using OutType = double;  // значение координаты в точке глобального минимума
using TestType = std::tuple<std::function<double(double)>, double, double, double, double, std::string, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gonozov_l_global_search
