#pragma once
#include <stb/stb_image.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace papulina_y_gauss_filter_block {

struct Picture {
  int width = 0;            // ширина картинки в пикселях
  int height = 0;           // высота картинки в пикселях
  int channels = STBI_rgb;  // цветная(3 канала) или черно-белая(1 канал)
  std::vector<unsigned char> pixels;
  bool operator==(const Picture &other) const {
    return (other.width == width && other.height == height && other.channels == channels && other.pixels == pixels);
  }
  Picture() = default;

  Picture(int w, int h, int c, std::vector<unsigned char> p) : width(w), height(h), channels(c), pixels(std::move(p)) {}
};

using InType = Picture;
using OutType = Picture;
using TestType = std::tuple<std::string, int, int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace papulina_y_gauss_filter_block
