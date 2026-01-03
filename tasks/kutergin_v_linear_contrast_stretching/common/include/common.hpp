#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kutergin_v_linear_contrast_stretching {

struct Image {
  std::vector<unsigned char> data;  // вектор значений интенсивностей
  int width = 0;                    // ширина изображения
  int height = 0;                   // высота изображения
};

using InType = Image;
using OutType = std::vector<unsigned char>;
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<InType, OutType, std::string>;

}  // namespace kutergin_v_linear_contrast_stretching
