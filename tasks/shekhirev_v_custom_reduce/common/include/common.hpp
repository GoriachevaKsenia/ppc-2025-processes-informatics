#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace shekhirev_v_custom_reduce {

using InType = int;
using OutType = int;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shekhirev_v_custom_reduce
