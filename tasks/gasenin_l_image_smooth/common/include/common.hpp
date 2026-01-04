#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace gasenin_l_image_smooth {

struct TaskData {
  std::vector<uint8_t> data;
  int width = 0;
  int height = 0;
  int kernel_size = 0;

  bool operator==(const TaskData &other) const {
    return data == other.data && width == other.width && height == other.height && kernel_size == other.kernel_size;
  }

  bool operator!=(const TaskData &other) const {
    return !(*this == other);
  }
};

using InType = TaskData;
using OutType = TaskData;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline int Clamp(int val, int min_val, int max_val) {
  if (val < min_val) {
    return min_val;
  }
  if (val > max_val) {
    return max_val;
  }
  return val;
}

}  // namespace gasenin_l_image_smooth
