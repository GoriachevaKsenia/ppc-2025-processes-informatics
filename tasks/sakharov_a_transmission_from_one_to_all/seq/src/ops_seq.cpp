#include "sakharov_a_transmission_from_one_to_all/seq/include/ops_seq.hpp"

#include <vector>

#include "sakharov_a_transmission_from_one_to_all/common/include/common.hpp"

namespace sakharov_a_transmission_from_one_to_all {

SakharovATransmissionFromOneToAllSEQ::SakharovATransmissionFromOneToAllSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SakharovATransmissionFromOneToAllSEQ::ValidationImpl() {
  return std::get<0>(GetInput()) == 0;
}

bool SakharovATransmissionFromOneToAllSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool SakharovATransmissionFromOneToAllSEQ::RunImpl() {
  GetOutput() = std::get<1>(GetInput());
  return true;
}

bool SakharovATransmissionFromOneToAllSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_transmission_from_one_to_all
