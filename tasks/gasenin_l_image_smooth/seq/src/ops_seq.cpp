#include "gasenin_l_image_smooth/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gasenin_l_image_smooth/common/include/common.hpp"

namespace gasenin_l_image_smooth {

GaseninLImageSmoothSEQ::GaseninLImageSmoothSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GaseninLImageSmoothSEQ::ValidationImpl() {
  return GetInput().width > 0 && GetInput().height > 0 && GetInput().kernel_size > 0 &&
         GetInput().data.size() == static_cast<size_t>(GetInput().width) * static_cast<size_t>(GetInput().height);
}

bool GaseninLImageSmoothSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

namespace {

void ProcessInteriorPixelSeq(int row, int col, int width, int kernel_size, const uint8_t *src, uint8_t *dst) {
  const int kernel_sq = kernel_size * kernel_size;
  const int radius = kernel_size / 2;

  const auto *row_ptr = src + (static_cast<ptrdiff_t>(row - radius) * width) + (col - radius);
  int sum = 0;

  for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y) {
    for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
      sum += row_ptr[kernel_x];
    }
    row_ptr += width;
  }
  dst[(row * width) + col] = static_cast<uint8_t>(sum / kernel_sq);
}

void ProcessBorderPixelSeq(int row, int col, int width, int height, int kernel_radius, const uint8_t *src,
                           uint8_t *dst) {
  int sum = 0;
  int count = 0;

  for (int kernel_y = -kernel_radius; kernel_y <= kernel_radius; ++kernel_y) {
    const int neighbor_row = Clamp(row + kernel_y, 0, height - 1);
    const int row_offset = neighbor_row * width;

    for (int kernel_x = -kernel_radius; kernel_x <= kernel_radius; ++kernel_x) {
      const int neighbor_col = Clamp(col + kernel_x, 0, width - 1);
      sum += src[row_offset + neighbor_col];
      ++count;
    }
  }

  const int index = (row * width) + col;
  if (count > 0) {
    dst[index] = static_cast<uint8_t>(sum / count);
  } else {
    dst[index] = src[index];
  }
}

}  // namespace

bool GaseninLImageSmoothSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  const int width = in.width;
  const int height = in.height;
  const int kernel_size = in.kernel_size;
  const int kernel_radius = kernel_size / 2;
  const uint8_t *src = in.data.data();
  uint8_t *dst = out.data.data();

  for (int row = 0; row < height; ++row) {
    const bool is_border_row = (row < kernel_radius) || (row >= height - kernel_radius);

    for (int col = 0; col < width; ++col) {
      const bool is_border_col = (col < kernel_radius) || (col >= width - kernel_radius);

      if (!is_border_row && !is_border_col) {
        ProcessInteriorPixelSeq(row, col, width, kernel_size, src, dst);
      } else {
        ProcessBorderPixelSeq(row, col, width, height, kernel_radius, src, dst);
      }
    }
  }
  return true;
}

bool GaseninLImageSmoothSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_image_smooth
