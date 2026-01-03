#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace rozenberg_a_radix_simple_merge {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace rozenberg_a_radix_simple_merge
