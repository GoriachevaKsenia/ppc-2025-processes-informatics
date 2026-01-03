#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace dolov_v_torus_topology {

struct InputData {
  int sender_rank = 0;
  int receiver_rank = 0;
  int total_procs = 0;
  std::vector<int> message;
};

struct OutputData {
  std::vector<int> route;
  std::vector<int> received_message;
};

using InType = InputData;
using OutType = OutputData;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace dolov_v_torus_topology
