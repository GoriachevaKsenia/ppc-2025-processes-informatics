#pragma once

#include <mpi.h>

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace nikolaev_d_gather {

struct GatherInput {
  std::vector<char> data;
  int count{0};
  MPI_Datatype datatype{MPI_DATATYPE_NULL};
  int root{0};
};

using InType = GatherInput;
using OutType = std::vector<char>;
using TestType = std::tuple<int, int, MPI_Datatype, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nikolaev_d_gather
