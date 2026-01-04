#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace spichek_d_simpson_integral {

using InType = int;   // количество разбиений (должно быть чётным)
using OutType = int;  // результат (приводим к int для тестов)
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace spichek_d_simpson_integral
