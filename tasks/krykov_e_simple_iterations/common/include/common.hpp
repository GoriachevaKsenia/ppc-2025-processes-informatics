#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace krykov_e_simple_iterations {

using InType = std::tuple<size_t,               // n
                          std::vector<double>,  // A
                          std::vector<double>   // b
                          >;

using OutType = std::vector<double>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace krykov_e_simple_iterations
