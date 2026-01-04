#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace chaschin_v_broadcast {

using InTypeI = std::vector<int>;
using OutTypeI = std::vector<int>;

using InTypeF = std::vector<float>;
using OutTypeF = std::vector<float>;

using InTypeD = std::vector<double>;
using OutTypeD = std::vector<double>;

using InType = std::vector<int>;
using OutType = std::vector<int>;

using TestType = std::tuple<int, std::string>;

template <typename T>
using BaseTask = ppc::task::Task<std::vector<T>, std::vector<T>>;

}  // namespace chaschin_v_broadcast
