#include "../include/linear_contrast_stretching_seq.hpp"

#include <algorithm>  // для std::minmax_element
#include <cmath>      // для std::round
#include <cstddef>
#include <vector>

#include "../../common/include/common.hpp"

namespace kutergin_v_linear_contrast_stretching {

LinearContrastStretchingSequential::LinearContrastStretchingSequential(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // установка типа задачи
  GetInput() = in;                       // сохранение входных данных
  if (!in.data.empty()) {
    GetOutput().resize(in.data.size());  // инициализация выходных данных
  }
}

bool LinearContrastStretchingSequential::ValidationImpl() {
  return !GetInput().data.empty();  // проверка, что входное изображение не пустое
}

bool LinearContrastStretchingSequential::PreProcessingImpl() {
  return true;
}

bool LinearContrastStretchingSequential::RunImpl() {
  const auto &in = GetInput();
  const auto &image_in = in.data;
  auto &image_out = GetOutput();

  const auto minmax_it = std::ranges::minmax_element(image_in);  // нахождение минимума и максимума за один проход
  unsigned char min_in = *minmax_it.min;
  unsigned char max_in = *minmax_it.max;

  if (min_in == max_in)  // обработка крайнего случая (все пиксели одного цвета
  {
    std::ranges::copy(image_in, image_out.begin());  // копирование исходного изображения
    return true;
  }

  const double scale = 255.0 / (max_in - min_in);
  for (size_t i = 0; i < image_in.size(); i++) {
    double p_out = (image_in[i] - min_in) * scale;  // линейная растяжка интенсивности
    image_out[i] = static_cast<unsigned char>(
        std::round(p_out));  // округление до ближайшего целого с явным преобразованием к unsigned char
  }

  return true;
}

bool LinearContrastStretchingSequential::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_v_linear_contrast_stretching
