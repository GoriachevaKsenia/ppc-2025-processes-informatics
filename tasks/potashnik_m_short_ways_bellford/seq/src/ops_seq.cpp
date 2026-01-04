#include "potashnik_m_short_ways_bellford/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

#include "potashnik_m_short_ways_bellford/common/include/common.hpp"

namespace potashnik_m_short_ways_bellford {

PotashnikMShortWaysBellfordSEQ::PotashnikMShortWaysBellfordSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool PotashnikMShortWaysBellfordSEQ::ValidationImpl() {
  return true;
}

bool PotashnikMShortWaysBellfordSEQ::PreProcessingImpl() {
  return true;
}

bool PotashnikMShortWaysBellfordSEQ::RunImpl() {
  std::vector<int> dist;
  potashnik_m_short_ways_bellford::BellmanFordAlgoSeq(GetInput(), 0, dist);
  GetOutput() = dist;
  return true;
}

bool PotashnikMShortWaysBellfordSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace potashnik_m_short_ways_bellford
