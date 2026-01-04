#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace luzan_e_matrix_horis_rib_mult_sheme {
//    0      1       2       3        4
// matrix, height, width, vector, vector_len
using InType = std::tuple<std::vector<int>, int, int, std::vector<int>, int>;
// output matrix
using OutType = std::vector<int>;

using TestType = std::tuple<int, int>;  // height, width (also len for vec)
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace luzan_e_matrix_horis_rib_mult_sheme
