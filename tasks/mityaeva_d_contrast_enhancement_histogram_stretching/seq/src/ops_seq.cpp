#include "mityaeva_d_contrast_enhancement_histogram_stretching/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

ContrastEnhancementSEQ::ContrastEnhancementSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<uint8_t>{};
}

bool ContrastEnhancementSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.size() < 3) {
    return false;
  }

  const auto w_u8 = static_cast<std::uint8_t>(input[0]);
  const auto h_u8 = static_cast<std::uint8_t>(input[1]);

  width_ = static_cast<int>(w_u8);
  height_ = static_cast<int>(h_u8);

  if (width_ <= 0 || height_ <= 0) {
    return false;
  }

  total_pixels_ = width_ * height_;
  if (total_pixels_ <= 0) {
    return false;
  }

  return input.size() == static_cast<std::size_t>(total_pixels_) + 2U;
}

bool ContrastEnhancementSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  min_pixel_ = 255;
  max_pixel_ = 0;

  for (std::size_t i = 2; i < input.size(); ++i) {
    const auto pixel = static_cast<std::uint8_t>(input[i]);
    min_pixel_ = std::min(min_pixel_, pixel);
    max_pixel_ = std::max(max_pixel_, pixel);
  }

  return true;
}

bool ContrastEnhancementSEQ::RunImpl() {
  const auto &input = GetInput();

  if (width_ <= 0 || height_ <= 0 || total_pixels_ <= 0) {
    return false;
  }
  const std::size_t expected_size = static_cast<std::size_t>(total_pixels_) + 2U;
  if (input.size() != expected_size) {
    return false;
  }

  OutType result;
  result.resize(expected_size);

  result[0] = static_cast<std::uint8_t>(width_);
  result[1] = static_cast<std::uint8_t>(height_);

  if (min_pixel_ == max_pixel_) {
    std::copy(input.begin() + 2, input.end(), result.begin() + 2);
    GetOutput() = std::move(result);
    return true;
  }

  const double scale = 255.0 / static_cast<double>(max_pixel_ - min_pixel_);

  for (std::size_t i = 2; i < expected_size; ++i) {
    const auto pixel = static_cast<std::uint8_t>(input[i]);
    const double stretched = static_cast<double>(pixel - min_pixel_) * scale;

    int rounded_value = static_cast<int>(std::lround(stretched));
    rounded_value = std::clamp(rounded_value, 0, 255);

    result[i] = static_cast<std::uint8_t>(rounded_value);
  }

  GetOutput() = std::move(result);
  return true;
}

bool ContrastEnhancementSEQ::PostProcessingImpl() {
  const auto &output = GetOutput();

  if (output.size() < 2) {
    return false;
  }

  const int out_width = static_cast<int>(static_cast<std::uint8_t>(output[0]));
  const int out_height = static_cast<int>(static_cast<std::uint8_t>(output[1]));

  if (out_width != width_ || out_height != height_) {
    return false;
  }

  return output.size() == static_cast<std::size_t>(total_pixels_) + 2U;
}

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
