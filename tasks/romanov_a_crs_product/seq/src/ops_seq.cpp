#include "romanov_a_crs_product/seq/include/ops_seq.hpp"

#include <cstdint>
#include <utility>

#include "romanov_a_crs_product/common/include/common.hpp"

namespace romanov_a_crs_product {

RomanovACRSProductSEQ::RomanovACRSProductSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CRS(static_cast<uint64_t>(0));
}

bool RomanovACRSProductSEQ::ValidationImpl() {
  return (std::get<0>(GetInput()).GetCols() == std::get<1>(GetInput()).GetRows());
}

bool RomanovACRSProductSEQ::PreProcessingImpl() {
  return true;
}

bool RomanovACRSProductSEQ::RunImpl() {
  const CRS &a = std::get<0>(GetInput());
  const CRS &b = std::get<1>(GetInput());

  CRS c = a * b;

  GetOutput() = std::move(c);

  return true;
}

bool RomanovACRSProductSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_crs_product
