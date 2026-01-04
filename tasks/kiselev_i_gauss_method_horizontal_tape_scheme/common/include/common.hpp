#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kiselev_i_gauss_method_horizontal_tape_scheme {

using InType = std::tuple<std::vector<std::vector<double>>, std::vector<double>, std::size_t>;
using OutType = std::vector<double>;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
