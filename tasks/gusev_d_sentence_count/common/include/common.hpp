#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace gusev_d_sentence_count {

using InType = std::string;
using OutType = size_t;
using TestType = std::tuple<std::string, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gusev_d_sentence_count
