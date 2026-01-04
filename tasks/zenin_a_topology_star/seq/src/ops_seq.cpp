#include "zenin_a_topology_star/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

#include "zenin_a_topology_star/common/include/common.hpp"

namespace zenin_a_topology_star {

ZeninATopologyStarSEQ::ZeninATopologyStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninATopologyStarSEQ::ValidationImpl() {
  const auto &in = GetInput();

  return !std::get<2>(in).empty();
}

bool ZeninATopologyStarSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ZeninATopologyStarSEQ::RunImpl() {
  const auto &in = GetInput();
  const auto &data = std::get<2>(in);

  GetOutput() = data;
  return true;
}

bool ZeninATopologyStarSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_topology_star
