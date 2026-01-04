#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace savva_d_min_elem_vec {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<std::vector<int>, std::string>;  // нужен как вспомогательный тип, чтобы было удобно
                                                             // проводить тесты (например хранит путь к файлу)
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace savva_d_min_elem_vec
