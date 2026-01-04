#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"

namespace ovsyannikov_n_shell_batcher {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::size_t;
using BaseTask = ppc::task::Task<InType, OutType>;
}  // namespace ovsyannikov_n_shell_batcher
