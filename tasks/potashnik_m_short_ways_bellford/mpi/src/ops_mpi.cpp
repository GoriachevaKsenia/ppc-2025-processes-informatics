#include "potashnik_m_short_ways_bellford/mpi/include/ops_mpi.hpp"

#include <cmath>
#include <vector>

#include "potashnik_m_short_ways_bellford/common/include/common.hpp"

namespace potashnik_m_short_ways_bellford {

PotashnikMShortWaysBellfordMPI::PotashnikMShortWaysBellfordMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool PotashnikMShortWaysBellfordMPI::ValidationImpl() {
  return true;
}

bool PotashnikMShortWaysBellfordMPI::PreProcessingImpl() {
  return true;
}

bool PotashnikMShortWaysBellfordMPI::RunImpl() {
  std::vector<int> dist;
  potashnik_m_short_ways_bellford::BellmanFordAlgoMpi(GetInput(), 0, dist);
  GetOutput() = dist;
  return true;
}

bool PotashnikMShortWaysBellfordMPI::PostProcessingImpl() {
  return true;
}

}  // namespace potashnik_m_short_ways_bellford
