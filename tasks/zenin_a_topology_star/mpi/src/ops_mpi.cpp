#include "zenin_a_topology_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "zenin_a_topology_star/common/include/common.hpp"

namespace zenin_a_topology_star {

ZeninATopologyStarMPI::ZeninATopologyStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninATopologyStarMPI::ValidationImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &in = GetInput();
  const int src = static_cast<int>(std::get<0>(in));
  const int dst = static_cast<int>(std::get<1>(in));
  return src >= 0 && dst >= 0 && src < world_size && dst < world_size;
}

bool ZeninATopologyStarMPI::PreProcessingImpl() {
  return true;
}

bool ZeninATopologyStarMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &in = GetInput();
  const size_t src = std::get<0>(in);
  const size_t dst = std::get<1>(in);
  const auto &data = std::get<2>(in);

  auto &out = GetOutput();
  out.clear();

  const int center = 0;
  const int tag = 0;

  const int src_rank = static_cast<int>(src);
  const int dst_rank = static_cast<int>(dst);
  const int center_rank = center;

  if (src_rank == dst_rank) {
    if (world_rank == src_rank) {
      out = data;
    }
    return true;
  }

  if (src_rank == center_rank || dst_rank == center_rank) {
    if (world_rank == src_rank) {
      MPI_Send(data.data(), static_cast<int>(data.size()), MPI_DOUBLE, dst_rank, tag, MPI_COMM_WORLD);
    } else if (world_rank == dst_rank) {
      out.resize(data.size());
      MPI_Recv(out.data(), static_cast<int>(out.size()), MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return true;
  }

  if (world_rank == src_rank) {
    MPI_Send(data.data(), static_cast<int>(data.size()), MPI_DOUBLE, center_rank, tag, MPI_COMM_WORLD);
  } else if (world_rank == center_rank) {
    std::vector<double> buf(data.size());
    MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, dst_rank, tag, MPI_COMM_WORLD);
  } else if (world_rank == dst_rank) {
    out.resize(data.size());
    MPI_Recv(out.data(), static_cast<int>(out.size()), MPI_DOUBLE, center_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return true;
}

bool ZeninATopologyStarMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_topology_star
