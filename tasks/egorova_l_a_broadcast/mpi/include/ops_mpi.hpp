#pragma once

#include <mpi.h>

#include "egorova_l_a_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egorova_l_a_broadcast {

class EgorovaLBroadcastMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit EgorovaLBroadcastMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void TreeBroadcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
};

}  // namespace egorova_l_a_broadcast
