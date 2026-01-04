#include "chaschin_v_sobel_operator/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "chaschin_v_sobel_operator/common/include/common.hpp"

namespace chaschin_v_sobel_operator {

ChaschinVSobelOperatorSEQ::ChaschinVSobelOperatorSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto in_copy = in;
  GetInput() = std::move(in_copy);
  this->GetOutput().clear();
}

bool ChaschinVSobelOperatorSEQ::ValidationImpl() {
  const auto &in = GetInput();

  const auto &image = std::get<0>(in);

  return !image.empty();
}

bool ChaschinVSobelOperatorSEQ::PreProcessingImpl() {
  return true;
}

bool ChaschinVSobelOperatorSEQ::RunImpl() {
  const auto &in = GetInput();

  const auto &mat = std::get<0>(in);
  auto &out = GetOutput();

  int n = std::get<1>(in);
  int m = std::get<2>(in);
  std::vector<std::vector<float>> gray(n, std::vector<float>(m, 0.0F));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      Pixel p = mat[i][j];
      gray[i][j] =
          (0.299F * static_cast<float>(p.r)) + (0.587F * static_cast<float>(p.g)) + (0.114F * static_cast<float>(p.b));
    }
  }

  std::vector<float> post_process_gray = SobelSeq(gray);

  out.resize(n);

  for (int i = 0; i < n; ++i) {
    out[i].resize(m);
    for (int j = 0; j < m; ++j) {
      float v = post_process_gray[(i * m) + j];
      unsigned char c = static_cast<unsigned char>(std::clamp(v, 0.0F, 255.0F));
      out[i][j] = Pixel{.r = c, .g = c, .b = c};
    }
  }

  return true;
}

bool ChaschinVSobelOperatorSEQ::PostProcessingImpl() {
  return true;
}

std::vector<float> SobelSeq(const std::vector<std::vector<float>> &image) {
  const int n = static_cast<int>(image.size());
  assert(n > 0);
  const int m = static_cast<int>(image[0].size());
  assert(m > 0);

  std::vector<std::vector<int>> k_kx = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

  std::vector<std::vector<int>> k_ky = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  std::vector<float> out(static_cast<size_t>(n) * static_cast<size_t>(m), 0.0F);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      float gx = 0.0F;
      float gy = 0.0F;

      const int i0 = std::max(0, i - 1);
      const int i1 = std::min(n - 1, i + 1);
      const int j0 = std::max(0, j - 1);
      const int j1 = std::min(m - 1, j + 1);

      for (int ni = i0; ni <= i1; ++ni) {
        for (int nj = j0; nj <= j1; ++nj) {
          const int di = ni - i;
          const int dj = nj - j;

          float v = image[ni][nj];

          volatile int vi = i;
          volatile int vj = j;
          if ((vi + vj) > -1) {
            gx += v * static_cast<float>(k_kx[di + 1][dj + 1]);
            gy += v * static_cast<float>(k_ky[di + 1][dj + 1]);
          }
        }
      }
      out[(i * m) + j] = std::sqrt((gx * gx) + (gy * gy));
    }
  }

  return out;
}

}  // namespace chaschin_v_sobel_operator
