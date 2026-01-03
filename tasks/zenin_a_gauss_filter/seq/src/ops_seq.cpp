#include "zenin_a_gauss_filter/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "zenin_a_gauss_filter/common/include/common.hpp"

namespace zenin_a_gauss_filter {

namespace {
int Clamp(const Image &img, int x, int y, int ch) {
  const int h = img.height;
  const int w = img.width;
  const int c = img.channels;
  x = std::clamp(x, 0, w - 1);
  y = std::clamp(y, 0, h - 1);
  return static_cast<int>(img.pixels[(((y * w) + x) * c) + ch]);
}
}  // namespace

ZeninAGaussFilterSEQ::ZeninAGaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninAGaussFilterSEQ::ValidationImpl() {
  const auto &in = GetInput();
  const std::size_t need = static_cast<std::size_t>(in.width) * in.height * in.channels;
  return in.width > 0 && in.height > 0 && (in.channels == 1 || in.channels == 3) && in.pixels.size() == need;
}

bool ZeninAGaussFilterSEQ::PreProcessingImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  out.width = in.width;
  out.height = in.height;
  out.channels = in.channels;
  out.pixels.assign(in.pixels.size(), 0);

  return true;
}

bool ZeninAGaussFilterSEQ::RunImpl() {
  const auto &input_image = GetInput();
  auto &output_image = GetOutput();

  const int width = input_image.width;
  const int height = input_image.height;
  const int channels = input_image.channels;

  static constexpr int kKernelSum = 16;

  for (int iy = 0; iy < height; ++iy) {
    for (int ix = 0; ix < width; ++ix) {
      for (int channel = 0; channel < channels; ++channel) {
        const int v00 = Clamp(input_image, ix - 1, iy - 1, channel);
        const int v01 = Clamp(input_image, ix, iy - 1, channel);
        const int v02 = Clamp(input_image, ix + 1, iy - 1, channel);

        const int v10 = Clamp(input_image, ix - 1, iy, channel);
        const int v11 = Clamp(input_image, ix, iy, channel);
        const int v12 = Clamp(input_image, ix + 1, iy, channel);

        const int v20 = Clamp(input_image, ix - 1, iy + 1, channel);
        const int v21 = Clamp(input_image, ix, iy + 1, channel);
        const int v22 = Clamp(input_image, ix + 1, iy + 1, channel);

        int sum = 0;
        sum += v00 * 1;
        sum += v01 * 2;
        sum += v02 * 1;
        sum += v10 * 2;
        sum += v11 * 4;
        sum += v12 * 2;
        sum += v20 * 1;
        sum += v21 * 2;
        sum += v22 * 1;

        const int res = (sum + (kKernelSum / 2)) / kKernelSum;
        output_image.pixels[((iy * width + ix) * channels) + channel] =
            static_cast<std::uint8_t>(std::clamp(res, 0, 255));
      }
    }
  }

  return true;
}

bool ZeninAGaussFilterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_gauss_filter
