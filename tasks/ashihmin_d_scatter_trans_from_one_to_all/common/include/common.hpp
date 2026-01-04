#pragma once

#include <mpi.h>

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace ashihmin_d_scatter_trans_from_one_to_all {

struct ScatterParams {
  std::vector<int> data;
  int elements_per_process = 0;
  int root = -1;
  MPI_Datatype datatype = MPI_DATATYPE_NULL;
};

using InType = ScatterParams;
using OutType = std::vector<int>;
using TestType = std::tuple<int, int, std::string>;

template <typename T>
using BaseTask = ppc::task::Task<ScatterParams, std::vector<T>>;

}  // namespace ashihmin_d_scatter_trans_from_one_to_all
