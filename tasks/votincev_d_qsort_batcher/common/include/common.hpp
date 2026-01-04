#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace votincev_d_qsort_batcher {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace votincev_d_qsort_batcher
