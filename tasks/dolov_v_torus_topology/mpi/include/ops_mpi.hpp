#pragma once

#include <mpi.h>

#include <cstdint>

#include "dolov_v_torus_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dolov_v_torus_topology {

class DolovVTorusTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DolovVTorusTopologyMPI(InType in);
  ~DolovVTorusTopologyMPI() override;

 private:
  enum class MoveSide : uint8_t { kNorth, kSouth, kWest, kEast, kStay };

  struct ProtocolTags {
    static constexpr int kDataTransfer = 0;
    static constexpr int kRouteSync = 1;
  };

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void DefineGridDimensions(int total_procs, int &r, int &c);
  static MoveSide FindShortestPathStep(int current, int target, int r, int c);
  static int GetTargetNeighbor(int current, MoveSide side, int r, int c);

  InputData input_;
  OutputData output_;

  MPI_Comm torus_comm_ = MPI_COMM_NULL;
};

}  // namespace dolov_v_torus_topology
