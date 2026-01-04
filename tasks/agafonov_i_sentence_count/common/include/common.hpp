#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace agafonov_i_sentence_count {

using InType = std::string;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace agafonov_i_sentence_count
