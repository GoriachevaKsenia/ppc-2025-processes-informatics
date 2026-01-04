#pragma once

#include <string>

#include "gutyansky_a_monte_carlo_multi_dimension/common/include/integration_task.hpp"
#include "task/include/task.hpp"

namespace gutyansky_a_monte_carlo_multi_dimension {

using InType = IntegrationTask;
using OutType = double;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gutyansky_a_monte_carlo_multi_dimension
