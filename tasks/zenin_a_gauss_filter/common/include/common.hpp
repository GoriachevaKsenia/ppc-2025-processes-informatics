#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace zenin_a_gauss_filter {

struct Image {
  int height = 0;
  int width = 0;
  int channels = 0;
  std::vector<std::uint8_t> pixels;
  Image() = default;
  Image(int h, int w, int c, std::vector<std::uint8_t> p) : height(h), width(w), channels(c), pixels(std::move(p)) {}
  bool operator==(const Image &another) const {
    return width == another.width && height == another.height && channels == another.channels &&
           pixels == another.pixels;
  }
};

using InType = Image;
using OutType = Image;
using TestType = std::tuple<std::string, int, int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zenin_a_gauss_filter
