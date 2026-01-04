#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace potashnik_m_star_topol {

using InType = std::vector<int>;
using OutType = std::tuple<int, int>;  // Second int = 0 if number of processes = 1 (or sequential version)
using TestType = int;
using BaseTask = ppc::task::Task<InType, OutType>;

// Function to generate next (source -> destination) call. Cyclicly goes through number of non-center processes
inline std::pair<int, int> GetCyclicSrcDst(int world_size, int iter) {
  int num_other_processes = world_size - 1;
  int src = (iter % num_other_processes) + 1;
  int dst = ((iter + 1) % num_other_processes) + 1;
  return {src, dst};
}

}  // namespace potashnik_m_star_topol
