#include "luzan_e_matrix_horis_rib_mult_sheme/seq/include/ops_seq.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

#include "luzan_e_matrix_horis_rib_mult_sheme/common/include/common.hpp"

namespace luzan_e_matrix_horis_rib_mult_sheme {

LuzanEMatrixHorisRibMultShemeSEQ::LuzanEMatrixHorisRibMultShemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LuzanEMatrixHorisRibMultShemeSEQ::ValidationImpl() {
  bool res = true;
  int height = std::get<1>(GetInput());
  int width = std::get<2>(GetInput());
  int vec_height = std::get<4>(GetInput());

  // matrix check
  res = std::get<0>(GetInput()).size() == static_cast<size_t>(height) * static_cast<size_t>(width) && height > 0 &&
        width > 0;

  // vec check
  res = res && std::get<3>(GetInput()).size() == static_cast<size_t>(vec_height);

  // matrix & vec sizes cmp.
  res = res && (width == vec_height);

  return res;
}

bool LuzanEMatrixHorisRibMultShemeSEQ::PreProcessingImpl() {
  int height = std::get<1>(GetInput());
  // int width = std::get<2>(GetInput());
  // int vec_height = std::get<4>(GetInput());

  GetOutput().resize(height);
  for (int cell = 0; cell < height; cell++) {
    GetOutput()[cell] = 0;
  }
  return true;
}

bool LuzanEMatrixHorisRibMultShemeSEQ::RunImpl() {
  int height = std::get<1>(GetInput());
  int width = std::get<2>(GetInput());
  // int vec_height = std::get<4>(GetInput());

  const std::tuple_element_t<0, InType> &mat = std::get<0>(GetInput());
  const std::tuple_element_t<3, InType> &vec = std::get<3>(GetInput());
  int tmp_sum = 0;
  for (int row = 0; row < height; row++) {
    tmp_sum = 0;
    for (int col = 0; col < width; col++) {
      tmp_sum += mat[(width * row) + col] * vec[col];
    }
    GetOutput()[row] += tmp_sum;
  }
  return true;
}

bool LuzanEMatrixHorisRibMultShemeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace luzan_e_matrix_horis_rib_mult_sheme
