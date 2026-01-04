#pragma once

#include <mpi.h>

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace pikhotskiy_r_scatter {

using InType = std::tuple<const void *, int, MPI_Datatype, void *, int, MPI_Datatype, int, MPI_Comm>;
using OutType = void *;
using TestType = std::tuple<InType, std::string, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace pikhotskiy_r_scatter
