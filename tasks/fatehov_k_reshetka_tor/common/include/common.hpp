#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace fatehov_k_reshetka_tor {

using InType = std::tuple<size_t, size_t, std::vector<double>>;
using OutType = double;
using TestType = std::tuple<int, size_t, size_t, std::vector<double>, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace fatehov_k_reshetka_tor
