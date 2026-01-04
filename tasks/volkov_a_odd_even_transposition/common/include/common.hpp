#pragma once

#include <vector>

#include "task/include/task.hpp"

namespace volkov_a_odd_even_transposition {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace volkov_a_odd_even_transposition
