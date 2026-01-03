#include "fatehov_k_reshetka_tor/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "fatehov_k_reshetka_tor/common/include/common.hpp"

namespace fatehov_k_reshetka_tor {

FatehovKReshetkaTorSEQ::FatehovKReshetkaTorSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FatehovKReshetkaTorSEQ::ValidationImpl() {
  auto &data = GetInput();
  return (std::get<0>(data) > 0 && std::get<0>(data) <= kMaxRows) &&
         (std::get<1>(data) > 0 && std::get<1>(data) <= kMaxCols) &&
         (std::get<0>(data) * std::get<1>(data) <= kMaxMatrixSize) &&
         (std::get<2>(data).size() <= kMaxMatrixSize &&
          std::get<2>(data).size() == std::get<0>(data) * std::get<1>(data)) &&
         (!std::get<2>(data).empty());
}

bool FatehovKReshetkaTorSEQ::PreProcessingImpl() {
  return true;
}

bool FatehovKReshetkaTorSEQ::RunImpl() {
  auto &data = GetInput();
  std::vector<double> &matrix = std::get<2>(data);
  double global_max = -1e18;
  for (double val : matrix) {
    double heavy_val = val;
    for (int k = 0; k < 100; ++k) {
      heavy_val = (std::sin(heavy_val) * std::cos(heavy_val)) + std::exp(std::complex<double>(0, heavy_val).real()) +
                  std::sqrt(std::abs(heavy_val) + 1.0);
      if (std::isinf(heavy_val)) {
        heavy_val = val;
      }
    }

    global_max = std::max(heavy_val, global_max);
  }
  GetOutput() = global_max;
  return true;
}

bool FatehovKReshetkaTorSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace fatehov_k_reshetka_tor
