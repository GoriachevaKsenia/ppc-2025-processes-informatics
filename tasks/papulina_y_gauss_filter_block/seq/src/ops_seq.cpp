#include "papulina_y_gauss_filter_block/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "papulina_y_gauss_filter_block/common/include/common.hpp"

namespace papulina_y_gauss_filter_block {

PapulinaYGaussFilterSEQ::PapulinaYGaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Picture();
}

bool PapulinaYGaussFilterSEQ::ValidationImpl() {
  return (GetInput().width >= 1 && GetInput().height >= 1);
}

bool PapulinaYGaussFilterSEQ::PreProcessingImpl() {
  Pic_ = GetInput();
  return true;
}

bool PapulinaYGaussFilterSEQ::RunImpl() {
  int k = Pic_.channels;
  Picture result(Pic_.width, Pic_.height, Pic_.channels,
                 std::vector<unsigned char>(static_cast<size_t>(Pic_.width * Pic_.height * k), 0));
  for (int col = 0; col < Pic_.width; col++) {
    for (int row = 0; row < Pic_.height; row++) {
      int index = (row * Pic_.width + col) * k;
      std::vector<unsigned char> pixel_new = NewPixel(row, col);
      for (int ch = 0; ch < k; ch++) {
        result.pixels[index + ch] = pixel_new[ch];
      }
    }
  }
  GetOutput() = result;
  return true;
}
bool PapulinaYGaussFilterSEQ::PostProcessingImpl() {
  return true;
}
void PapulinaYGaussFilterSEQ::ClampCoordinates(int &n_x, int &n_y) const {
  n_x = std::max(0, std::min(Pic_.width - 1, n_x));
  n_y = std::max(0, std::min(Pic_.height - 1, n_y));
}
std::vector<unsigned char> PapulinaYGaussFilterSEQ::NewPixel(const int &row, const int &col) {
  static constexpr std::array<float, 9> kErnel = {1.0F / 16, 2.0F / 16, 1.0F / 16, 2.0F / 16, 4.0F / 16,
                                                  2.0F / 16, 1.0F / 16, 2.0F / 16, 1.0F / 16};
  const float *kernel_ptr = kErnel.data();
  int k = Pic_.channels;
  std::vector<float> sums(k, 0);
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      int n_x = col + dx;
      int n_y = row + dy;
      ClampCoordinates(n_x, n_y);
      int index = (n_y * Pic_.width + n_x) * k;
      for (int ch = 0; ch < k; ch++) {
        sums[ch] += static_cast<float>(Pic_.pixels[index + ch]) * (*kernel_ptr);
      }
      ++kernel_ptr;
    }
  }
  std::vector<unsigned char> result(k);
  for (int ch = 0; ch < k; ch++) {
    float value = sums[ch];
    value = std::max(0.0F, std::min(255.0F, value));
    result[ch] = static_cast<unsigned char>(std::lround(value));
  }
  return result;
}
}  // namespace papulina_y_gauss_filter_block
