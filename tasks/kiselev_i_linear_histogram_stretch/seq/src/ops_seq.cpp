#include "kiselev_i_linear_histogram_stretch/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "kiselev_i_linear_histogram_stretch/common/include/common.hpp"

namespace kiselev_i_linear_histogram_stretch {

KiselevITestTaskSEQ::KiselevITestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

  if (!in.pixels.empty()) {
    GetOutput().resize(in.pixels.size());
  }
}

bool KiselevITestTaskSEQ::ValidationImpl() {
  const auto &img = GetInput();
  return img.width > 0 && img.height > 0 && img.pixels.size() == img.width * img.height;
}

bool KiselevITestTaskSEQ::PreProcessingImpl() {
  return true;
}

bool KiselevITestTaskSEQ::RunImpl() {
  const auto &input = GetInput().pixels;
  auto &output = GetOutput();

  std::uint8_t min_val = std::numeric_limits<std::uint8_t>::max();
  std::uint8_t max_val = std::numeric_limits<std::uint8_t>::min();

  for (std::uint8_t value : input) {
    min_val = std::min(min_val, value);
    max_val = std::max(max_val, value);
  }

  if (min_val == max_val) {
    output = input;
    return true;
  }

  const double scale = 255.0 / static_cast<double>(max_val - min_val);

  for (std::size_t i = 0; i < input.size(); ++i) {
    const double stretched = static_cast<double>(input[i] - min_val) * scale;
    output[i] = static_cast<std::uint8_t>(std::lround(stretched));
  }

  return true;
}

bool KiselevITestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_linear_histogram_stretch
