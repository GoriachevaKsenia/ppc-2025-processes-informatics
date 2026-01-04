#include "egorova_l_a_broadcast/seq/include/ops_seq.hpp"

#include <cstring>

#include "egorova_l_a_broadcast/common/include/common.hpp"

namespace egorova_l_a_broadcast {

EgorovaLBroadcastSEQ::EgorovaLBroadcastSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool EgorovaLBroadcastSEQ::ValidationImpl() {
  return true;
}

bool EgorovaLBroadcastSEQ::PreProcessingImpl() {
  return true;
}

bool EgorovaLBroadcastSEQ::RunImpl() {
  auto &in = GetInput();
  auto &out = GetOutput();

  if (in.type_indicator == 0) {
    out.resize(in.data_int.size() * sizeof(int));
    std::memcpy(out.data(), in.data_int.data(), out.size());
  } else if (in.type_indicator == 1) {
    out.resize(in.data_float.size() * sizeof(float));
    std::memcpy(out.data(), in.data_float.data(), out.size());
  } else {
    out.resize(in.data_double.size() * sizeof(double));
    std::memcpy(out.data(), in.data_double.data(), out.size());
  }
  return true;
}

bool EgorovaLBroadcastSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_a_broadcast
