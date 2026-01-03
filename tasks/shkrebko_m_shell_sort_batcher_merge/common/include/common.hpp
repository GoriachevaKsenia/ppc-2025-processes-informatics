#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"

namespace shkrebko_m_shell_sort_batcher_merge {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::size_t;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shkrebko_m_shell_sort_batcher_merge
