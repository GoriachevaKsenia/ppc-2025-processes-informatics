#include "guseva_a_jarvis/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "guseva_a_jarvis/common/include/common.hpp"

namespace guseva_a_jarvis {

GusevaAJarvisSEQ::GusevaAJarvisSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool GusevaAJarvisSEQ::ValidationImpl() {
  const auto &[width, height, image] = GetInput();
  bool is_size_match = static_cast<int>(image.size()) == width * height;
  bool is_image_binary = std::ranges::all_of(image.begin(), image.end(), [](int x) { return x == 0 || x == 1; });
  bool is_size_possible = width > 0 && height > 0;
  return is_image_binary && is_size_possible && is_size_match;
}

bool GusevaAJarvisSEQ::PreProcessingImpl() {
  const auto &[width, height, image] = GetInput();
  for (int yy = 0; yy < height; ++yy) {
    for (int xx = 0; xx < width; ++xx) {
      if (image[(yy * width) + xx] == 1) {
        points_.emplace_back(xx, yy);
      }
    }
  }
  return true;
}

bool GusevaAJarvisSEQ::RunImpl() {
  hull_ = BuildConvexHull(points_);
  return true;
}

bool GusevaAJarvisSEQ::PostProcessingImpl() {
  GetOutput().resize(static_cast<int64_t>(std::get<0>(GetInput())) * std::get<1>(GetInput()), 0);
  for (const auto &[x, y] : hull_) {
    GetOutput()[(y * std::get<0>(GetInput())) + x] = 1;
  }
  return true;
}

}  // namespace guseva_a_jarvis
